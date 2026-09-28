"""Reading and repairing LWA TBN recordings (the only TBN-specific code).

* ``open_tbn`` / ``tbn_info`` / ``probe_timestamp``: LSL access and metadata.
* ``read_tbn_blocks``: a generator of time-ordered sample blocks for selected
  digitizer rows, with timing gaps filled by NaN so sweep timing is preserved.
* ``find_capture_start`` / ``copy_captures``: frame-level realignment used by
  ``preprocess_tbn.py``.
"""

from datetime import datetime, timedelta

import numpy as np

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
# Frame-level repair (preprocess_tbn.py)
# ---------------------------------------------------------------------------

def _frame_row(frame):
    stand, pol = frame.id
    return 2 * (stand - 1) + pol


def find_capture_start(fh, max_frames=10_000):
    """Advance ``fh`` to the next frame for digitizer row 0 (stand 1, pol 0). Returns frames skipped."""
    from lsl.reader import tbn
    from lsl.reader.errors import SyncError
    skipped = 0
    while skipped < max_frames:
        pos = fh.tell()
        try:
            frame = tbn.read_frame(fh)
        except SyncError:
            fh.seek(pos + 1)
            skipped += 1
            continue
        if _frame_row(frame) == 0:
            fh.seek(pos)
            return skipped
        skipped += 1
    raise RuntimeError(f"No capture boundary found within {max_frames} frames")


def probe_stream_layout(fh, attempts=100, expected_rate=(95e3, 105e3)):
    """From the current position, find the first capture whose metadata is readable and sane.

    Returns (position, sample_rate, frames_per_capture) with ``fh`` left at that
    capture boundary. Failed probes move on by whole captures.
    """
    from lsl.reader import tbn
    for attempt in range(attempts):
        find_capture_start(fh)
        pos = fh.tell()
        try:
            rate = tbn.get_sample_rate(fh)
            fh.seek(pos)
            nx, ny = tbn.get_frames_per_obs(fh)
            fh.seek(pos)
        except Exception as exc:
            print(f"Metadata probe {attempt}: {exc}; trying the next capture", flush=True)
        else:
            if expected_rate[0] < rate < expected_rate[1]:
                return pos, float(rate), int(nx + ny)
            print(f"Metadata probe {attempt}: implausible sample rate {rate}; trying the next capture", flush=True)
        fh.seek(pos + tbn.FRAME_SIZE)
    raise RuntimeError(f"No readable TBN metadata after {attempts} captures")


def copy_captures(fh_in, fh_out, n_captures, frames_per_capture, verify=True, progress_every=1000):
    """Copy whole captures; with ``verify``, resynchronise when a capture does not start at row 0.

    Returns a dict with counts of captures written and resynchronisations.
    """
    from lsl.reader import tbn
    size = tbn.FRAME_SIZE
    written = resyncs = 0
    for c in range(int(n_captures)):
        pos = fh_in.tell()
        if verify:
            try:
                frame = tbn.read_frame(fh_in)
                ok = _frame_row(frame) == 0
            except Exception:
                ok = False
            fh_in.seek(pos)
            if not ok:
                skipped = find_capture_start(fh_in)
                resyncs += 1
                print(f"Capture {c}: resynchronised after skipping {skipped} frame(s)", flush=True)
        chunk = fh_in.read(size * frames_per_capture)
        if len(chunk) < size * frames_per_capture:
            break
        fh_out.write(chunk)
        written += 1
        if progress_every and written % progress_every == 0:
            print(f"  {written}/{int(n_captures)} captures", flush=True)
    return dict(captures=written, resyncs=resyncs)
