"""LWA station metadata: antenna identity, positions, cable delays, and stand status.

These lookups come from LSL's LWA-SV station description (SSMIF). Channels are
always identified by physical (stand, polarization); list order in LSL or in a
file is never used as an identity.
"""

import numpy as np

# Stands known to misbehave in the 2023-12-09 LWA-SV data. The SSMIF status
# flags (``--bad-stands status``) are a complementary, date-independent source.
BAD_LWA_STANDS = [6, 10, 16, 20, 21, 27, 60, 76, 93, 108, 110, 116, 133, 155, 163, 164, 178, 228, 229, 232]


def parse_stands(spec="all", available=None):
    """Parse "all", "first", "5", "1-10,20" into a sorted list.

    "all" means every available stand (or 1-256); "first" the lowest available stand.
    """
    if str(spec).strip().lower() == "first":
        if available is None or not len(available):
            raise ValueError('"--stands first" needs a list of available stands')
        return [int(min(available))]
    if spec is None or str(spec).strip().lower() == "all":
        return sorted(int(s) for s in (available if available is not None else range(1, 257)))
    stands = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            stands.extend(range(int(lo), int(hi) + 1))
        else:
            stands.append(int(part))
    return sorted(set(stands))


def pol_indices(pol):
    p = str(pol).strip().lower()
    if p == "both":
        return [0, 1]
    if p in ("x", "0"):
        return [0]
    if p in ("y", "1"):
        return [1]
    raise ValueError("--pol must be both, x/y, or 0/1")


def lwasv_antennas():
    """Map (stand, pol) -> LSL Antenna for LWA-SV."""
    try:
        from lsl.common import stations
    except ImportError as exc:
        raise ImportError("LWA-SV station metadata requires LSL; run in your LSL environment") from exc
    result = {}
    for antenna in stations.lwasv.antennas:
        key = (int(antenna.stand.id), int(antenna.pol))
        if key in result:
            raise ValueError(f"Duplicate LWA-SV antenna mapping: {key}")
        result[key] = antenna
    return result


def station_stands():
    return sorted({stand for stand, _ in lwasv_antennas()})


def selected_antennas(stands, pols):
    lookup = lwasv_antennas()
    keys = [(int(s), int(p)) for s, p in zip(stands, pols)]
    missing = [k for k in keys if k not in lookup]
    if missing:
        raise ValueError(f"Missing LWA-SV antenna mappings: {missing}")
    return [lookup[k] for k in keys]


def tbn_stream_indices(stands, pol):
    """Rows of LSL's TBN ``read()`` output for the requested stands/pol.

    TBN data rows are ordered by digitizer (row = digitizer - 1), which is not
    the same as physical stand order.
    """
    channel_stands = [int(s) for s in stands for _ in pol_indices(pol)]
    channel_pols = [p for _ in stands for p in pol_indices(pol)]
    rows = [int(a.digitizer) - 1 for a in selected_antennas(channel_stands, channel_pols)]
    if any(r < 0 for r in rows) or len(set(rows)) != len(rows):
        raise ValueError("Selected LWA-SV antennas have invalid or duplicate digitizer mappings")
    return np.asarray(rows, int), np.asarray(channel_stands, int), np.asarray(channel_pols, int)


def resolve_legacy_channel_ids(stands, pols, metadata):
    """Return physical (stands, pols). Old real-data caches labeled digitizer pairs as stands."""
    kind = metadata.get("channel_id_kind", "physical_stand" if metadata.get("synthetic") else "legacy_input_pair")
    if kind in ("physical_stand", "receiver"):
        return np.asarray(stands, int), np.asarray(pols, int)
    if kind != "legacy_input_pair":
        raise ValueError(f"Unknown channel ID convention: {kind}")
    by_digitizer = {int(a.digitizer): a for a in lwasv_antennas().values() if a.digitizer > 0}
    digitizers = [2 * (int(s) - 1) + int(p) + 1 for s, p in zip(stands, pols)]
    missing = sorted(set(digitizers) - set(by_digitizer))
    if missing:
        raise ValueError(f"Legacy cache digitizers absent from LWA-SV metadata: {missing}")
    antennas = [by_digitizer[d] for d in digitizers]
    metadata["channel_id_kind"] = "physical_stand"
    metadata["channel_digitizers"] = digitizers
    return np.array([a.stand.id for a in antennas], int), np.array([a.pol for a in antennas], int)


def stand_positions(stands):
    """(N, 3) east/north/up positions in metres and a mask of stands that were found."""
    by_stand = {}
    for antenna in lwasv_antennas().values():
        by_stand[int(antenna.stand.id)] = (float(antenna.stand.x), float(antenna.stand.y), float(antenna.stand.z))
    found = np.array([int(s) in by_stand for s in stands], bool)
    positions = np.array([by_stand[int(s)] for s, ok in zip(stands, found) if ok], float).reshape(-1, 3)
    return positions, found


def cable_delays_s(stands, pols, frequency_hz):
    return np.array([a.cable.delay(float(frequency_hz)) for a in selected_antennas(stands, pols)], float)


def cable_phase_factors(stands, pols, frequency_hz):
    """Multiply channels by these to remove relative cable phase at ``frequency_hz``.

    A cable delay d contributes exp(-j 2 pi f d); the factor exp(+j 2 pi f (d - d_min))
    removes it (checked against a physical time-delay model in the tests).
    """
    if frequency_hz is None or not np.isfinite(frequency_hz) or frequency_hz <= 0:
        raise ValueError("Cable correction requires a positive tuning frequency")
    d = cable_delays_s(stands, pols, frequency_hz)
    if not d.size or not np.all(np.isfinite(d)):
        raise ValueError("LSL returned empty or nonfinite cable delays")
    return np.exp(2j * np.pi * float(frequency_hz) * (d - d.min()))


def bad_stand_mask(stands, pols, mode="list"):
    """True for channels to drop. mode: none, list (BAD_LWA_STANDS), status (SSMIF), both."""
    stands = np.asarray(stands, int)
    bad = np.zeros(stands.shape, bool)
    if mode in ("list", "both"):
        bad |= np.isin(stands, BAD_LWA_STANDS)
    if mode in ("status", "both"):
        antennas = selected_antennas(stands, pols)
        bad |= np.array([a.combined_status != 33 for a in antennas])
    if mode not in ("none", "list", "status", "both"):
        raise ValueError("--bad-stands must be none, list, status or both")
    return bad
