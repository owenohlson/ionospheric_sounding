"""Everything specific to the LWA: station metadata (LWA-SV via LSL) and TBN recordings.

Station
    ``parse_stands``/``pol_indices``, antenna lookup by physical (stand, pol),
    positions, cable delays and phase factors, digitizer rows for TBN,
    bad stands (SSMIF status flags), legacy channel labels.
TBN
    Read directly from the frame headers (no LSL): ``scan_tbn`` (captures by
    timetag, gaps, late frames), ``TBNReader`` (samples placed by timetag, so
    out-of-order frames land in the right place and missing frames become NaN),
    and ``write_tbn_range`` for ``preprocess_tbn.py``.

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


def bad_stand_mask(stands, pols, mode="status"):
    """True for channels to drop. mode: none, or status (SSMIF status flags, 33 = OK).

    The hand-made stand list was removed: it was built under a different station
    description and named the wrong stands under the dated SSMIF. Use the SSMIF for
    the observation (``--ssmif``) plus the flags from a self-calibration file.
    """
    stands = np.asarray(stands, int)
    if mode in ("list", "both"):
        raise ValueError("--bad-stands list/both were removed; use 'status' (SSMIF flags) "
                         "and the flags in a calibration file (--calibration)")
    if mode not in ("none", "status"):
        raise ValueError("--bad-stands must be none or status")
    if mode == "none":
        return np.zeros(stands.shape, bool)
    antennas = selected_antennas(stands, pols)
    return np.array([a.combined_status != 33 for a in antennas], bool)


# ===========================================================================
# TBN recordings
# ===========================================================================

TBN_SAMPLES_PER_FRAME = 512


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

    Frames are grouped by consecutive (timetag, tuning word). The recording's own
    captures are the groups with the dominant tuning word whose timetags sit on the
    dominant timetag grid; anything else (for example a few frames left over from
    a previous TBN setup at another frequency, ahead of a gap) is marked invalid
    and ignored by the reader, the summary and trimming.

    Keys: n_frames, bad_sync (indices), group_first, group_size, group_timetag,
    group_tuning, valid (per group), tuning_word, timetag_step, sample_rate_hz,
    frames_per_capture.
    """
    mm = np.memmap(path, np.uint8, mode="r")
    n = mm.size // TBN_FRAME_SIZE
    firsts, tags, tunes, bad = [], [], [], []
    prev = None
    for f0 in range(0, n, block_frames):
        blk = _frame_block(mm, f0, min(block_frames, n - f0))
        ok = (blk[:, 0] == 0xDE) & (blk[:, 1] == 0xC0) & (blk[:, 2] == 0xDE) & (blk[:, 3] == 0x5C)
        bad.extend((f0 + np.flatnonzero(~ok)).tolist())
        idx = np.flatnonzero(ok)
        if not idx.size:
            continue
        t = np.ascontiguousarray(blk[idx, 16:24]).view(">u8").ravel().astype(np.int64)
        w = np.ascontiguousarray(blk[idx, 8:12]).view(">u4").ravel().astype(np.int64)
        change = np.ones(t.size, bool)
        change[1:] = (t[1:] != t[:-1]) | (w[1:] != w[:-1])
        if prev is not None and (t[0], w[0]) == prev:
            change[0] = False
        firsts.extend((f0 + idx[change]).tolist())
        tags.extend(t[change].tolist())
        tunes.extend(w[change].tolist())
        prev = (t[-1], w[-1])
    firsts, tags, tunes = (np.asarray(x, np.int64) for x in (firsts, tags, tunes))
    if firsts.size < 3:
        raise RuntimeError("Fewer than three captures with valid sync words")
    sizes = np.diff(np.append(firsts, n))
    words = np.unique(tunes)
    weight = {w_: int(sizes[tunes == w_].sum()) for w_ in words}
    main = max(weight, key=weight.get)                      # tuning word holding most frames
    m = tunes == main
    diffs = np.diff(np.sort(np.unique(tags[m])))
    step = int(np.median(diffs)) if diffs.size else 0
    if step <= 0:
        raise RuntimeError("Could not determine the timetag step")
    phases, pc = np.unique(tags[m] % step, return_counts=True)
    phase = int(phases[np.argmax(pc)])                      # grid the recording's captures sit on
    valid = m & (tags % step == phase)
    return dict(n_frames=n, bad_sync=np.asarray(bad, np.int64), group_first=firsts, group_size=sizes,
                group_timetag=tags, group_tuning=tunes, valid=valid, tuning_word=int(main), timetag_step=step,
                frames_per_capture=int(np.median(sizes[valid])),
                sample_rate_hz=TBN_SAMPLES_PER_FRAME * FPGA_CLOCK_HZ / step)


def capture_slots(scan):
    """Capture index of every frame group, (timetag - first valid timetag) / step; -1 for invalid groups."""
    tags, valid = scan["group_timetag"], scan["valid"]
    slots = np.full(tags.size, -1, np.int64)
    slots[valid] = (tags[valid] - tags[valid].min()) // scan["timetag_step"]
    return slots


def summarize_tbn_scan(scan):
    """Data-quality numbers for a scanned TBN file (see ``preprocess_tbn.py``)."""
    fs, per = scan["sample_rate_hz"], scan["frames_per_capture"]
    valid = scan["valid"]
    slots = capture_slots(scan)[valid]
    n_slots = int(slots.max()) + 1
    frames = np.zeros(n_slots, np.int64)
    np.add.at(frames, slots, scan["group_size"][valid])
    present = frames > 0
    missing = ~present
    # runs of missing captures
    edges = np.diff(np.concatenate([[0], missing.astype(np.int8), [0]]))
    runs = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
    capture_s = TBN_SAMPLES_PER_FRAME / fs
    late = np.diff(scan["group_timetag"][valid]) <= 0
    other = ~valid
    other_tunings = sorted({round(w * FPGA_CLOCK_HZ / 2 ** 32 / 1e6, 4) for w in scan["group_tuning"][other]})
    return dict(captures=int(present.sum()), sample_rate_hz=fs,
                tuning_mhz=round(scan["tuning_word"] * FPGA_CLOCK_HZ / 2 ** 32 / 1e6, 6),
                start_utc=_utc_from_ticks(int(scan["group_timetag"][valid].min())).isoformat(),
                frames_per_capture=per,
                complete_captures=int(np.count_nonzero(frames == per)),
                partial_captures=int(np.count_nonzero(present & (frames < per))),
                overfull_captures=int(np.count_nonzero(frames > per)),
                missing_captures=int(missing.sum()),
                missing_s=float(missing.sum() * capture_s),
                gaps=int(runs.size), longest_gap_s=float(runs.max() * capture_s) if runs.size else 0.0,
                out_of_order_groups=int(late.sum()),
                bad_sync_frames=int(scan["bad_sync"].size),
                ignored_frames=int(scan["group_size"][other].sum()),
                ignored_tunings_mhz=", ".join(f"{f:g}" for f in other_tunings) or "-",
                duration_s=float(n_slots * capture_s))


def _utc_from_ticks(ticks):
    from datetime import timezone
    whole, frac = divmod(int(ticks), int(FPGA_CLOCK_HZ))
    return datetime.fromtimestamp(whole, timezone.utc) + timedelta(seconds=frac / FPGA_CLOCK_HZ)


def write_tbn_range(path, output, scan, tstart=0.0, tend=None, drop_incomplete=False, block_frames=1 << 16):
    """Copy captures whose time (from the first capture) lies in [tstart, tend), in timetag order.

    Late (out-of-order) frame groups are written next to the rest of their
    capture. Frames with bad sync words are always dropped; with
    ``drop_incomplete`` so are captures holding the wrong number of frames.
    Missing captures stay missing. Returns (captures, frames) written.
    """
    slots = capture_slots(scan)
    valid = scan["valid"]
    per_slot = np.zeros(int(slots.max()) + 1, np.int64)
    np.add.at(per_slot, slots[valid], scan["group_size"][valid])
    t = slots * TBN_SAMPLES_PER_FRAME / scan["sample_rate_hz"]
    complete = valid & (per_slot[np.maximum(slots, 0)] == scan["frames_per_capture"])
    keep = valid & (t >= tstart)
    if tend is not None:
        keep &= t < tend
    if drop_incomplete:
        keep &= complete
    else:                                   # always drop partial captures at the file edges
        keep &= complete | ((slots > 0) & (slots < slots.max()))
    if not keep.any():
        raise ValueError("No captures in the requested time range")
    mm = np.memmap(path, np.uint8, mode="r")
    firsts, sizes = scan["group_first"], scan["group_size"]
    bad = scan["bad_sync"]
    bad_groups = set((np.searchsorted(firsts, bad, side="right") - 1).tolist()) if bad.size else set()
    frames = 0
    groups = np.flatnonzero(keep)
    groups = groups[np.argsort(slots[groups], kind="stable")]
    with open(output, "wb") as fo:
        for g in groups:
            first, size = int(firsts[g]), int(sizes[g])
            blk = _frame_block(mm, first, size)
            if g in bad_groups:
                blk = blk[~np.isin(np.arange(first, first + size), bad)]
            fo.write(blk.tobytes())
            frames += len(blk)
    return int(np.unique(slots[groups]).size), frames


# ---------------------------------------------------------------------------
# Sample reader
# ---------------------------------------------------------------------------

class TBNReader:
    """Read TBN samples straight from the file, placing every frame by its timetag.

    Frame layout (LSL ``tbn.cpp``): sync word, frame count, tuning word
    (big-endian u32 at byte 8), TBN id (big-endian u16 at byte 12; digitizer =
    id & 1023, LSL row = digitizer - 1), gain, timetag (big-endian u64 at byte
    16, 196 MHz ticks since the Unix epoch), then 512 complex samples as int8
    I/Q pairs. Unlike LSL's reader this tolerates late frames, duplicate frames
    and missing captures: samples are placed at
    (timetag - first timetag) / step, and anything absent is NaN.
    """

    def __init__(self, path, scan=None):
        self.path = str(path)
        self.scan = scan if scan is not None else scan_tbn(self.path)
        self.sample_rate_hz = float(self.scan["sample_rate_hz"])
        self.step = int(self.scan["timetag_step"])
        valid = self.scan["valid"]
        self.tag0 = int(self.scan["group_timetag"][valid].min())
        self.group_slot = capture_slots(self.scan)
        self.n_captures = int(self.group_slot.max()) + 1
        order = np.argsort(self.group_slot, kind="stable")
        self._order = order[valid[order]]                        # invalid groups (slot -1) never read
        self._frames = np.memmap(self.path, np.uint8, mode="r",
                                 shape=(int(self.scan["n_frames"]), TBN_FRAME_SIZE))
        self.tuning_word = int(self.scan["tuning_word"])
        self.tuning_frequency_hz = self.tuning_word * FPGA_CLOCK_HZ / 2 ** 32
        self.n_rows = int(self.scan["frames_per_capture"])

    @property
    def duration_s(self):
        return self.n_captures * TBN_SAMPLES_PER_FRAME / self.sample_rate_hz

    def utc_at(self, t):
        """UTC datetime of time ``t`` [s] after the first capture (exact to the 196 MHz clock)."""
        return _utc_from_ticks(self.tag0 + int(round(t * FPGA_CLOCK_HZ)))

    def fractional_second_at(self, t):
        ticks = self.tag0 + int(round(t * FPGA_CLOCK_HZ))
        return (ticks % int(FPGA_CLOCK_HZ)) / FPGA_CLOCK_HZ

    def info(self):
        return dict(sample_rate_hz=self.sample_rate_hz, tuning_frequency_hz=self.tuning_frequency_hz,
                    start_time=self.utc_at(0.0), n_frames=int(self.scan["n_frames"]), n_antpol=self.n_rows,
                    usable_duration_s=self.duration_s)

    def read_blocks(self, rows, tstart=0.0, duration=None, chunk_seconds=0.25):
        """Yield ``(block, missing)`` covering [tstart, tstart + duration).

        ``block`` is complex64 (len(rows), n) for digitizer ``rows`` (digitizer - 1);
        samples whose frame is absent are NaN; ``missing`` counts them. tstart is
        rounded down to a capture boundary (5.12 ms at 100 kHz).
        """
        rows = np.asarray(rows, int)
        where = np.full(1024, -1, np.int64)
        where[rows] = np.arange(rows.size)
        per = TBN_SAMPLES_PER_FRAME
        first = int(np.floor(tstart * self.sample_rate_hz / per + 1e-9))
        stop = self.n_captures if duration is None else min(
            self.n_captures, first + int(np.ceil(duration * self.sample_rate_hz / per - 1e-9)))
        chunk = max(1, int(round(chunk_seconds * self.sample_rate_hz / per)))
        sorted_slots = self.group_slot[self._order]
        step = self.step
        bad = self.scan["bad_sync"]
        for a in range(first, stop, chunk):
            b = min(a + chunk, stop)
            lo, hi = np.searchsorted(sorted_slots, [a, b])
            groups = self._order[lo:hi]
            out = np.full((rows.size, b - a, per), np.nan + 1j * np.nan, np.complex64)
            if groups.size:
                idx = np.concatenate([np.arange(f, f + n) for f, n in
                                      zip(self.scan["group_first"][groups], self.scan["group_size"][groups])])
                if bad.size:
                    idx = idx[~np.isin(idx, bad)]
                raw = np.asarray(self._frames[idx])
                ident = np.ascontiguousarray(raw[:, 12:14]).view(">u2").ravel().astype(np.int64)
                tune = np.ascontiguousarray(raw[:, 8:12]).view(">u4").ravel().astype(np.int64)
                tbn = ((ident >> 15) & 1 == 0) & (tune == self.tuning_word)   # TBW bit set / other tuning
                digit = ident & 1023
                tag = np.ascontiguousarray(raw[:, 16:24]).view(">u8").ravel().astype(np.int64)
                on_grid = (tag - self.tag0) % step == 0
                slot = (tag - self.tag0) // step - a
                ch = np.where((digit >= 1) & (digit <= 1024), where[np.clip(digit - 1, 0, 1023)], -1)
                use = tbn & on_grid & (ch >= 0) & (slot >= 0) & (slot < b - a)
                iq = raw[use, 24:].view(np.int8).astype(np.float32).reshape(-1, per, 2)
                out[ch[use], slot[use], :] = iq[..., 0] + 1j * iq[..., 1]
            block = out.reshape(rows.size, -1)
            yield block, int(np.count_nonzero(np.isnan(block.real)))


def print_tbn_info(info, quality=None):
    print("=== TBN metadata ===")
    print(f"Start time:  {info['start_time']}")
    print(f"Sample rate: {info['sample_rate_hz']:g} Hz")
    print(f"Tuning:      {info['tuning_frequency_hz']:.3f} Hz")
    print(f"Frames:      {info['n_frames']} ({info['n_antpol']} ant/pol streams)")
    print(f"Duration:    {info['usable_duration_s']:.3f} s")
    if quality and quality.get("ignored_frames"):
        print(f"Ignored:     {quality['ignored_frames']} frames at other tunings ({quality['ignored_tunings_mhz']} MHz) "
              f"or off the timetag grid")
    if quality:
        print(f"Missing:     {quality['missing_captures']} captures ({quality['missing_s']:.2f} s in "
              f"{quality['gaps']} gaps, longest {quality['longest_gap_s'] * 1e3:.1f} ms); "
              f"{quality['partial_captures']} partial; {quality['out_of_order_groups']} late frame groups placed by timetag")
    print("====================", flush=True)
