"""Everything specific to the LWA: station metadata (LWA-SV via LSL) and TBN recordings.

Station
    ``parse_stands``/``pol_indices``, antenna lookup by physical (stand, pol),
    positions, cable delays and phase factors, digitizer rows for TBN,
    bad stands (known list or SSMIF status), legacy channel labels.
TBN
    ``open_tbn``/``tbn_info``/``probe_timestamp``, the gap-aware sample reader
    ``read_tbn_blocks``, and capture alignment/copying for ``preprocess_tbn.py``.

Channels are always identified by physical (stand, polarization); list order
in LSL or in a file is never used as an identity.
"""

import functools
import os
from datetime import datetime, timedelta

import numpy as np


# ===========================================================================
# Station metadata
# ===========================================================================


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


SSMIF_PATH = os.environ.get("LWA_SSMIF") or None


def use_ssmif(path):
    """Use this station description instead of LSL's bundled LWA-SV one.

    ``path`` may be an SSMIF file or an observation metadata tarball (.tgz).
    The SSMIF must match the observation date: digitizer wiring, cable lengths
    and stand status change over time.
    """
    global SSMIF_PATH
    SSMIF_PATH = None if path is None else os.path.expanduser(path)
    _station.cache_clear()


@functools.lru_cache(maxsize=1)
def _station():
    try:
        from lsl.common import stations
    except ImportError as exc:
        raise ImportError("LWA-SV station metadata requires LSL; run in your LSL environment") from exc
    if SSMIF_PATH is None:
        return stations.lwasv
    if SSMIF_PATH.endswith((".tgz", ".tar.gz", ".tar")):
        from lsl.common import metabundle
        return metabundle.get_station(SSMIF_PATH)
    return stations.parse_ssmif(SSMIF_PATH)


def lwasv_antennas():
    """Map (stand, pol) -> LSL Antenna for LWA-SV (bundled SSMIF unless ``use_ssmif`` was called)."""
    result = {}
    for antenna in _station().antennas:
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


def ids_from_digitizers(digitizers):
    """Physical (stands, pols) for TBN digitizer numbers under the current station description."""
    by_digitizer = {int(a.digitizer): a for a in lwasv_antennas().values() if a.digitizer > 0}
    missing = sorted(set(int(d) for d in digitizers) - set(by_digitizer))
    if missing:
        raise ValueError(f"Digitizers absent from the station description: {missing}")
    antennas = [by_digitizer[int(d)] for d in digitizers]
    return np.array([a.stand.id for a in antennas], int), np.array([a.pol for a in antennas], int)


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


# ===========================================================================
# TBN recordings
# ===========================================================================

TBN_SAMPLES_PER_FRAME = 512


# ---------------------------------------------------------------------------
# Timestamps
# ---------------------------------------------------------------------------

def to_datetime(timestamp):
    """LSL timestamps expose ``utc_datetime`` or ``datetime``; plain datetimes pass through."""
    if timestamp is None:
        return None
    for attr in ("utc_datetime", "datetime"):
        if hasattr(timestamp, attr):
            return getattr(timestamp, attr)
    return timestamp if isinstance(timestamp, datetime) else None


def fractional_second(timestamp):
    dt = to_datetime(timestamp)
    return None if dt is None else dt.microsecond / 1e6


# ---------------------------------------------------------------------------
# LSL access
# ---------------------------------------------------------------------------

def open_tbn(path):
    from lsl.reader.ldp import LWADataFile, TBNFile
    idf = LWADataFile(path)
    if not isinstance(idf, TBNFile):
        raise RuntimeError(f"'{path}' does not appear to be a TBN file (LSL type {type(idf).__name__})")
    return idf


def tbn_info(idf):
    fs = float(idf.get_info("sample_rate"))
    try:
        n_antpol = int(idf.get_info("nantenna")) or 520
    except Exception:
        n_antpol = 520
    nframe = float(idf.get_info("nframe"))
    return dict(
        sample_rate_hz=fs,
        tuning_frequency_hz=float(idf.get_info("freq1")),
        start_time=to_datetime(idf.get_info("start_time")),
        n_frames=int(nframe),
        n_antpol=n_antpol,
        usable_duration_s=np.floor(nframe / n_antpol) * TBN_SAMPLES_PER_FRAME / fs,
    )


def print_tbn_info(info):
    print("=== TBN metadata ===")
    print(f"Start time:  {info['start_time']}")
    print(f"Sample rate: {info['sample_rate_hz']:g} Hz")
    print(f"Tuning:      {info['tuning_frequency_hz']:.3f} Hz")
    print(f"Frames:      {info['n_frames']} ({info['n_antpol']} ant/pol streams)")
    print(f"Duration:    {info['usable_duration_s']:.3f} s")
    print("====================", flush=True)


def probe_timestamp(path, tstart):
    """UTC timestamp of the first sample at ``tstart`` seconds into the file."""
    idf = open_tbn(path)
    try:
        idf.offset(float(tstart))
        _, timestamp, _ = idf.read(0.001)
        return timestamp
    finally:
        idf.close()


def read_tbn_blocks(idf, tstart, duration, rows, chunk_seconds=0.25):
    """Yield ``(block, gap_samples)`` for digitizer ``rows`` over [tstart, tstart+duration).

    ``block`` is complex64 (len(rows), n). Timetag jumps and invalid-timetag
    skips become NaN-filled blocks of the right length (``gap_samples`` > 0), so
    downstream sweep timing stays correct. The first yielded item is preceded by
    ``idf`` being positioned at ``tstart``. Also sets ``read_tbn_blocks.first_timestamp``.
    """
    from lsl.reader import errors
    rows = np.asarray(rows, int)
    fs = float(idf.get_info("sample_rate"))
    idf.offset(float(tstart))
    got = timeline = 0.0
    first = None
    tolerance = max(2.0 / fs, 0.001)
    read_tbn_blocks.first_timestamp = None

    def gap(seconds):
        n = int(round(seconds * fs))
        return np.full((len(rows), n), np.nan + 1j * np.nan, np.complex64), n

    while got < duration:
        try:
            _, stamp, data = idf.read(min(chunk_seconds, duration - got))
        except errors.EOFError:
            print("Reached end of TBN file", flush=True)
            return
        except RuntimeError as exc:
            if "Invalid timetag skip" not in str(exc):
                raise
            block, n = gap(min(0.002, duration - got))
            print(f"Invalid timetag skip at {tstart + got:.3f} s; inserting {n} NaN samples", flush=True)
            idf.offset(n / fs)
            got += n / fs
            timeline += n / fs
            if n:
                yield block, n
            continue
        if first is None:
            first = to_datetime(stamp)
            if first is not None and got > 0:
                first = first - timedelta(seconds=got)
            read_tbn_blocks.first_timestamp = first
        else:
            now = to_datetime(stamp)
            if now is not None and first is not None:
                jump = (now - first).total_seconds() - timeline
                if jump > tolerance:
                    block, n = gap(min(jump, duration - got))
                    print(f"Timestamp gap at {tstart + got:.3f} s; inserting {n} NaN samples", flush=True)
                    got += n / fs
                    timeline += n / fs
                    if n:
                        yield block, n
        if data.ndim != 2 or rows.max() >= data.shape[0]:
            raise ValueError(f"TBN block has shape {data.shape}; cannot select rows up to {rows.max()}")
        keep = min(data.shape[1], int(round((duration - got) * fs)))
        if keep <= 0:
            return
        yield data[rows, :keep].astype(np.complex64), 0
        got += keep / fs
        timeline += keep / fs


# ---------------------------------------------------------------------------
# Frame-level scan and trimming (preprocess_tbn.py)
# ---------------------------------------------------------------------------
# A TBN file is a sequence of 1048-byte frames: 24-byte header (sync word
# 0x5CDEC0DE little-endian, frame count, tuning word, TBN id, gain), 8-byte
# big-endian timetag, 512 complex int8 samples. All ant/pol frames of one
# 512-sample capture share a timetag, but their order inside the capture is
# not fixed, so captures are delimited by timetag changes, not by stand 1.

TBN_FRAME_SIZE = 1048
FPGA_CLOCK_HZ = 196e6


def _frame_block(mm, first, count):
    return np.asarray(mm[first * TBN_FRAME_SIZE:(first + count) * TBN_FRAME_SIZE]).reshape(count, TBN_FRAME_SIZE)


def scan_tbn(path, block_frames=1 << 16):
    """Read only frame headers. Returns a dict describing captures (timetag groups).

    Keys: n_frames, bad_sync (indices), group_first (first frame of each capture),
    group_size, group_timetag, sample_rate_hz, frames_per_capture.
    """
    mm = np.memmap(path, np.uint8, mode="r")
    n = mm.size // TBN_FRAME_SIZE
    firsts, tags, bad = [], [], []
    prev = None
    for f0 in range(0, n, block_frames):
        blk = _frame_block(mm, f0, min(block_frames, n - f0))
        ok = (blk[:, 0] == 0xDE) & (blk[:, 1] == 0xC0) & (blk[:, 2] == 0xDE) & (blk[:, 3] == 0x5C)
        bad.extend((f0 + np.flatnonzero(~ok)).tolist())
        tag = np.ascontiguousarray(blk[:, 16:24]).view(">u8").ravel().astype(np.int64)
        idx = np.flatnonzero(ok)
        if not idx.size:
            continue
        t = tag[idx]
        change = np.ones(t.size, bool)
        change[1:] = t[1:] != t[:-1]
        if prev is not None and t[0] == prev:
            change[0] = False
        firsts.extend((f0 + idx[change]).tolist())
        tags.extend(t[change].tolist())
        prev = t[-1]
    firsts, tags = np.asarray(firsts, np.int64), np.asarray(tags, np.int64)
    if firsts.size < 3:
        raise RuntimeError("Fewer than three captures with valid sync words")
    sizes = np.diff(np.append(firsts, n))
    step = np.median(np.diff(tags))
    return dict(n_frames=n, bad_sync=np.asarray(bad, np.int64), group_first=firsts, group_size=sizes,
                group_timetag=tags, frames_per_capture=int(np.median(sizes)),
                sample_rate_hz=TBN_SAMPLES_PER_FRAME * FPGA_CLOCK_HZ / step)


def summarize_tbn_scan(scan):
    fs, per = scan["sample_rate_hz"], scan["frames_per_capture"]
    step = TBN_SAMPLES_PER_FRAME * FPGA_CLOCK_HZ / fs
    jumps = np.diff(scan["group_timetag"]) / step
    return dict(captures=int(scan["group_first"].size), sample_rate_hz=fs, frames_per_capture=per,
                incomplete_captures=int(np.count_nonzero(scan["group_size"] != per)),
                missing_captures=int(np.sum(np.round(jumps[jumps > 1.5]) - 1)),
                backwards_timetags=int(np.count_nonzero(jumps <= 0)),
                bad_sync_frames=int(scan["bad_sync"].size),
                duration_s=float((scan["group_timetag"][-1] - scan["group_timetag"][0]) / FPGA_CLOCK_HZ
                                 + TBN_SAMPLES_PER_FRAME / fs))


def write_tbn_range(path, output, scan, tstart=0.0, tend=None, drop_incomplete=False, block_frames=1 << 16):
    """Copy whole captures whose time (from the first capture) lies in [tstart, tend).

    Frames with bad sync words are always dropped; with ``drop_incomplete`` so are
    captures holding the wrong number of frames. Returns (captures, frames) written.
    """
    t = (scan["group_timetag"] - scan["group_timetag"][0]) / FPGA_CLOCK_HZ
    complete = scan["group_size"] == scan["frames_per_capture"]
    keep = t >= tstart
    if tend is not None:
        keep &= t < tend
    if drop_incomplete:
        keep &= complete
    else:                                   # always drop partial captures at the file edges
        keep[0] &= complete[0]
        keep[-1] &= complete[-1]
    if not keep.any():
        raise ValueError("No captures in the requested time range")
    mm = np.memmap(path, np.uint8, mode="r")
    firsts, sizes = scan["group_first"], scan["group_size"]
    bad = scan["bad_sync"]
    bad_groups = set((np.searchsorted(firsts, bad, side="right") - 1).tolist()) if bad.size else set()
    frames = 0
    with open(output, "wb") as fo:
        for g in np.flatnonzero(keep):
            first, size = int(firsts[g]), int(sizes[g])
            blk = _frame_block(mm, first, size)
            if g in bad_groups:
                blk = blk[~np.isin(np.arange(first, first + size), bad)]
            fo.write(blk.tobytes())
            frames += len(blk)
    return int(keep.sum()), frames
