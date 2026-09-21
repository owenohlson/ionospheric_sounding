# tbn_utils.py
from __future__ import annotations

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for plotting
import matplotlib.pyplot as plt
import numpy as np
import os
import psutil
import textwrap
process = psutil.Process(os.getpid())
from datetime import datetime, timedelta

# LFM utils
from lfm_utils import window_from_arg


# -----------------------------
# Helpers
# -----------------------------

def mem():
    print(f"Memory usage: {process.memory_info().rss / 1e9:.2f} GB", flush=True)

def _timestamp_datetime(timestamp):
    if timestamp is None:
        return None
    if hasattr(timestamp, "utc_datetime"):
        return timestamp.utc_datetime
    if hasattr(timestamp, "datetime"):
        return timestamp.datetime
    return timestamp

def _format_timestamp(timestamp):
    dt = _timestamp_datetime(timestamp)
    if dt is None:
        return "unknown UTC"
    if isinstance(dt, datetime):
        return f"{dt.strftime('%H:%M:%S.%f')[:-3]} UTC"
    return f"{dt} UTC"

def _timestamp_at(start_timestamp, seconds_after_start):
    start_dt = _timestamp_datetime(start_timestamp)
    if isinstance(start_dt, datetime):
        return start_dt + timedelta(seconds=float(seconds_after_start))
    return None

def _timestamp_delta_seconds(later_timestamp, earlier_timestamp):
    later_dt = _timestamp_datetime(later_timestamp)
    earlier_dt = _timestamp_datetime(earlier_timestamp)
    if isinstance(later_dt, datetime) and isinstance(earlier_dt, datetime):
        return (later_dt - earlier_dt).total_seconds()
    return None

def _gap_warning(reason, plot_time, timestamp):
    print(
        f"{reason} at local plot time {plot_time:.3f}s ({_format_timestamp(timestamp)})",
        flush=True,
    )

def timestamp_range_note(start_timestamp, duration):
    start_dt = _timestamp_datetime(start_timestamp)
    if start_dt is None:
        return None
    end_dt = start_dt + timedelta(seconds=float(duration))
    return (
        f"UTC start: {start_dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}\n"
        f"UTC end:   {end_dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}"
    )

def file_timestamp_range_note(idf, tstart, tend):
    file_start = _timestamp_datetime(idf.get_info("start_time"))
    if file_start is None:
        return None
    start_dt = file_start + timedelta(seconds=float(tstart))
    end_dt = file_start + timedelta(seconds=float(tend))
    return (
        f"UTC start: {start_dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}\n"
        f"UTC end:   {end_dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}"
    )

def _add_corner_note(corner_note: str = None):
    if not corner_note:
        return
    wrapped_note = "\n".join(
        wrapped_line
        for line in str(corner_note).splitlines()
        for wrapped_line in (textwrap.wrap(line, width=52) or [""])
    )
    plt.gcf().text(
        0.99,
        0.99,
        wrapped_note,
        ha="right",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none", "pad": 3},
    )

def _savefig(out_png: str, dpi=150):
    full_path = os.path.expanduser(out_png)
    output_dir = os.path.dirname(full_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(full_path, dpi=dpi)
    return full_path

def best_freq_units(freq_hz: np.ndarray):
    """Return (freq_scaled, units) where units is Hz/kHz/MHz/GHz."""
    mx = float(np.max(np.abs(freq_hz))) if freq_hz.size else 1.0
    if mx >= 1e9:
        return freq_hz / 1e9, "GHz"
    if mx >= 1e6:
        return freq_hz / 1e6, "MHz"
    if mx >= 1e3:
        return freq_hz / 1e3, "kHz"
    return freq_hz, "Hz"

def pol_to_index(pol) -> int:
    p = str(pol).strip().lower()
    if p in ("x", "0"):
        return 0
    if p in ("y", "1"):
        return 1
    raise ValueError("--pol must be both, x/y, or 0/1")

def pol_to_indices(pol):
    p = str(pol).strip().lower()
    if p == "both":
        return [0, 1]
    return [pol_to_index(pol)]

def lsl_open_tbn(filename: str):
    from lsl.reader.ldp import LWADataFile, TBNFile
    idf = LWADataFile(filename)
    if TBNFile is not None and not isinstance(idf, TBNFile):
        raise RuntimeError(f"'{filename}' does not appear to be a valid TBN file (LSL type={type(idf)})")
    if TBNFile is None:
        try:
            idf.get_info("sample_rate")
            idf.get_info("freq1")
        except Exception as exc:
            raise RuntimeError(f"'{filename}' does not appear to be a valid TBN file (LSL type={type(idf)})") from exc
    return idf

def probe_start_timestamp(filename, tstart):
    """Read the timestamp at a requested TBN offset without keeping a reader open."""
    idf = lsl_open_tbn(filename)
    try:
        idf.offset(float(tstart))
        _, timestamp, _ = idf.read(0.001)
        return timestamp
    finally:
        idf.close()


def lsl_print_metadata(idf: TBNFile):
    nFramesFile = idf.get_info("nframe")
    fs = idf.get_info("sample_rate")
    beginDate = idf.get_info("start_time").datetime
    central_freq = idf.get_info("freq1")
    usable_duration = tbn_usable_duration(idf)
    print("=== LSL metadata ===")
    print(f"Opened as: {type(idf)}")
    print(f"Start time: {beginDate}")
    print(f"Sample rate: {fs} Hz")
    print(f"Tuning frequency (freq1): {central_freq:.3f} Hz")
    print(f"Frames in file: {nFramesFile}")
    print(f"Usable duration: {usable_duration:.3f} seconds")
    print("====================")

def tbn_antpol_count(idf: TBNFile, default: int = 520) -> int:
    """
    Return the number of ant/pol frame streams interleaved in the TBN file.

    LWA TBN captures typically have 260 stands x 2 pols = 520 antpol streams.
    LSL's nframe metadata counts raw TBN frames across all antpol streams, while
    each frame contributes 512 samples to one stream.  Therefore one stream's
    usable duration is nframe * 512 / (sample_rate * n_antpol).
    """
    try:
        n_antpol = idf.get_info("nantenna")
    except Exception:
        n_antpol = None
    if n_antpol is not None:
        n_antpol = int(n_antpol)
        if n_antpol > 0:
            return n_antpol
    return default

def tbn_usable_duration(idf: TBNFile) -> float:
    n_frames_file = float(idf.get_info("nframe"))
    fs = float(idf.get_info("sample_rate"))
    n_antpol = float(tbn_antpol_count(idf))
    frames_per_stream = np.floor(n_frames_file / n_antpol)
    return frames_per_stream * 512.0 / fs

def set_default_tbn_time_bounds(args, idf: TBNFile):
    if args.tstart is None:
        args.tstart = 0.0
        print("No --tstart provided, starting from beginning of file")
    if args.tend is None:
        args.tend = tbn_usable_duration(idf)
        print(f"No --tend provided, using end time of file: {args.tend:.2f} seconds")
    return args.tstart, args.tend

def _gap_fill_array(n_samples: int, gap_fill: str):
    if gap_fill == "zero":
        return np.zeros(n_samples, dtype=np.complex64)
    if gap_fill == "nan":
        return np.full(n_samples, np.nan + 1j * np.nan, dtype=np.complex64)
    raise ValueError("--gap-fill must be 'nan' or 'zero'")


def gap_fill_note(gap_info):
    if not gap_info or int(gap_info.get("events", 0)) == 0:
        return None
    mode = str(gap_info.get("mode", "fill"))
    events = int(gap_info.get("events", 0))
    samples = int(gap_info.get("samples", 0))
    fs = float(gap_info.get("sample_rate", 1.0))
    seconds = samples / fs if fs > 0 else 0.0
    fill_name = "Zeros" if mode == "zero" else "NaNs"
    return f"* {fill_name} used to fill corrupted data/timetag gaps ({events} event(s), {seconds:.3f} s)."


def _gap_fill_block(n_streams: int, n_samples: int, gap_fill: str):
    block = _gap_fill_array(n_samples, gap_fill)
    if n_streams == 1:
        return block
    return np.tile(block[None, :], (n_streams, 1))


def lsl_read_block_for_one_stream(
    idf: TBNFile,
    start_time,
    duration,
    stand_id=None,
    pol=None,
    gap_fill="nan",
    return_gap_info=False,
):
    """
    Robustly read enough samples from one stand and one or both polarization streams.

    Returns:
        x   : complex samples, shape (n_samples,) for one pol or (2, n_samples) for pol="both"
        t   : starting timestamp
    """
    from lsl.reader import errors
    # Decide stream_index
    pol_indices = pol_to_indices(pol)
    stream_indices = [2 * (int(stand_id) - 1) + pol_idx for pol_idx in pol_indices]
    if min(stream_indices) < 0:
        raise ValueError("Invalid --stand/--pol combination (negative index)")
    n_requested_streams = len(stream_indices)
    if gap_fill not in ("nan", "zero"):
        raise ValueError("--gap-fill must be 'nan' or 'zero'")

    # Position reader
    idf.offset(start_time)
    fs = float(idf.get_info("sample_rate"))
    target_seconds = duration

    chunk_seconds = min(0.25, target_seconds)
    timetag_skip_seconds = 0.002
    timestamp_gap_tolerance = max(2.0 / fs, 0.001)

    n_chunks = int(np.ceil(target_seconds / chunk_seconds))
    print("n_chunks = ", n_chunks)

    xs = []
    got = 0.0
    timeline_elapsed = 0.0
    antpols_seen = None
    chunk = 1
    start_timestamp = None
    first_timestamp = None
    first_requested_timestamp = None
    gap_info = {
        "mode": gap_fill,
        "events": 0,
        "samples": 0,
        "sample_rate": fs,
    }

    # print(f"Memory usage before reading: ", flush=True)
    # mem()

    while got < target_seconds:
        try:
            readT, start_timestamp, data = idf.read(min(chunk_seconds, target_seconds - got))
            if first_timestamp is None:
                if got > 0:
                    first_timestamp = _timestamp_at(start_timestamp, -got)
                    first_requested_timestamp = first_timestamp
                else:
                    first_timestamp = start_timestamp
                    first_requested_timestamp = start_timestamp
            else:
                observed_elapsed = _timestamp_delta_seconds(start_timestamp, first_timestamp)
                if observed_elapsed is not None:
                    timestamp_gap = observed_elapsed - timeline_elapsed
                    if timestamp_gap > timestamp_gap_tolerance:
                        gap_seconds = min(timestamp_gap, target_seconds - got)
                        gap_samples = int(round(gap_seconds * fs))
                        if gap_samples > 0:
                            gap_timestamp = _timestamp_at(first_requested_timestamp, got)
                            _gap_warning(
                                f"Timestamp gap; inserting {gap_fill} fill",
                                start_time + got,
                                gap_timestamp,
                            )
                            xs.append(_gap_fill_block(n_requested_streams, gap_samples, gap_fill))
                            gap_info["events"] += 1
                            gap_info["samples"] += gap_samples
                            gap_seconds = gap_samples / fs
                            got += gap_seconds
                            timeline_elapsed += gap_seconds
                    elif timestamp_gap < -timestamp_gap_tolerance:
                        _gap_warning(
                            "Timestamp discontinuity; keeping returned data",
                            start_time + got,
                            start_timestamp,
                        )

            # if start_timestamp is not None and chunk % 10 == 0:
            #     print(f"Start timestamp: {start_timestamp:.3f} s", flush=True)
            # else:
            #     print(f"Start timestamp: None", flush=True)

            # if chunk % 10 == 0:
            #     print(f"Chunk {chunk}/{n_chunks}: readT={readT:.3f} s, got={got:.3f} s total", flush=True)
            #     mem()

        except errors.EOFError:
            print("Reached EOF", flush=True)
            break
        except RuntimeError as e:
            if "Invalid timetag skip" in str(e):
                gap_seconds = min(timetag_skip_seconds, target_seconds - got)
                gap_samples = int(round(gap_seconds * fs))
                gap_timestamp = _timestamp_at(first_requested_timestamp, got)
                _gap_warning(
                    f"Invalid timetag skip; inserting {gap_fill} fill",
                    start_time + got,
                    gap_timestamp,
                )
                if gap_samples > 0:
                    xs.append(_gap_fill_block(n_requested_streams, gap_samples, gap_fill))
                    gap_info["events"] += 1
                    gap_info["samples"] += gap_samples
                    gap_seconds = gap_samples / fs
                    got += gap_seconds
                    timeline_elapsed += gap_seconds
                idf.offset(gap_seconds)
                continue
            raise

        if data.ndim != 2:
            raise RuntimeError(f"Unexpected data shape from idf.read(): {data.shape}")

        antpols_seen = data.shape[0]
        if max(stream_indices) >= antpols_seen:
            raise ValueError(f"Requested stream_indices={stream_indices} but data has only antpols={antpols_seen}. "
                             f"(Check --stand/--pol)")

        keep_samples = min(data.shape[1], int(round((target_seconds - got) * fs)))
        if keep_samples <= 0:
            break

        if n_requested_streams == 1:
            xs.append(data[stream_indices[0], :keep_samples].copy())
        else:
            xs.append(data[stream_indices, :keep_samples].copy())

        del data # free memory immediately

        chunk += 1
        read_seconds = keep_samples / fs
        got += read_seconds
        timeline_elapsed += read_seconds

        if readT <= 0:
            break

    if len(xs) == 0:
        if return_gap_info:
            return None, first_timestamp, gap_info
        return None, first_timestamp

    x = np.concatenate(xs, axis=-1)

    if return_gap_info:
        return x, first_timestamp, gap_info
    return x, first_timestamp


def lsl_average_spectrum_all_antpols(idf: TBNFile, args):
    from lsl.correlator import fx as fxc
    """
    Compute PSD for ALL antpols using LSL SpecMaster,
    then average across antpols to produce ONE global spectrum.
    """
    duration = args.tend - args.tstart
    idf.offset(args.tstart)
    fs = idf.get_info("sample_rate")
    window = window_from_arg(args.window_size, args.window)

    readT, t, data = idf.read(duration)
    freq, tempSpec = fxc.SpecMaster(
        data,
        LFFT=args.window_size,
        window=window,
        pfb=args.pfb,
        verbose=args.verbose,
        sample_rate=fs,
    )

    # tempSpec: (antpols, LFFT)
    spec_avg = np.mean(tempSpec, axis=0)
    return fs, freq, spec_avg

# -----------------------------
# Averaged Spectrum
# -----------------------------
def plot_averaged_spectrum(freq_hz: np.ndarray, psd_linear: np.ndarray, center_freq_hz: float | None, out_png: str | None, corner_note: str = None):
    # Convert to dB
    psd_db = 10.0 * np.log10(np.maximum(psd_linear, 1e-30))

    if center_freq_hz is not None:
        freq_plot = freq_hz + center_freq_hz
    else:
        freq_plot = freq_hz

    freq_scaled, units = best_freq_units(freq_plot)

    plt.figure()
    plt.plot(freq_scaled, psd_db)
    plt.xlabel(f"Frequency [{units}]")
    plt.ylabel("PSD [dB/RBW]")
    plt.title("Averaged Spectrum (single output)")
    _add_corner_note(corner_note)
    plt.tight_layout()
    if out_png:
        _savefig(out_png, dpi=150)
    plt.show()
