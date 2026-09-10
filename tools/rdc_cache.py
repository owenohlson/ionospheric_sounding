"""Views of stored RDC2 windows or new windows computed from retained RDC1.

The public iteration shape remains (windows, delay, Doppler, channels), even
when the on-disk RDC2 is three-dimensional. No inverse FFT reconstruction of
windowed, cropped, or lossy legacy RDC2 is attempted.
"""
import os

import h5py
import numpy as np

from rdc_utils import doppler_fft_rdc1, load_rdc2_hdf5_metadata, iter_rdc2_hdf5_channel_chunks


def prepare_rdc2_view(args):
    path = os.path.expanduser(args.rdc2_input)
    view = load_rdc2_hdf5_metadata(path)
    view["path"] = path
    view["recompute"] = False
    timing = {key: getattr(args, key, None) for key in
              ("integration_time", "hop_time", "tstart", "tend", "duration")}
    if not any(value is not None for value in timing.values()):
        return view
    for key, value in timing.items():
        if value is not None and not np.isfinite(value):
            raise ValueError(f"{key} must be finite")
    if not view["has_rdc1"]:
        # Existing caches remain useful with their saved timing. Explicitly
        # reject different integration/hop rather than silently ignoring them.
        durations = view["frame_tend_s"] - view["frame_tstart_s"]
        for key, actual in (("integration_time", durations),
                            ("hop_time", np.diff(view["frame_tstart_s"]))):
            requested = timing[key]
            if key == "hop_time" and not actual.size:
                actual = np.array([view["metadata"].get("hop_time_s", durations[0])])
            if requested is not None and (requested <= 0 or not np.allclose(actual, requested, rtol=0, atol=1e-8)):
                raise ValueError("This RDC2 cache has no retained RDC1; integration/hop are fixed. "
                                 "Re-export from TBN/RDC0 to change them.")
        if any(timing[key] is not None for key in ("tstart", "tend", "duration")):
            raise ValueError("Time cropping requires retained RDC1; select saved legacy frames instead")
        return view

    with h5py.File(path, "r") as h5:
        group = h5["rdc1"]
        real, imag = group["real"], group["imag"]
        if real.shape != imag.shape or real.ndim != 3 or real.shape[2] != view["shape"][-1]:
            raise ValueError("Invalid retained RDC1 shape or channel count")
        if not np.array_equal(group["delay_ms"][...], view["delay_ms"]):
            raise ValueError("Retained RDC1 delay axis differs from RDC2")
        for key in ("channel_stands", "channel_pols"):
            if not np.array_equal(group[key][...], h5["rdc2"][key][...]):
                raise ValueError("Retained RDC1 channel identities differ from RDC2")
        shape = real.shape
        origin = float(group.attrs["tstart_s"])
        prf = float(group.attrs["sweep_frequency_hz"])
    if not np.isfinite(prf) or prf <= 0:
        raise ValueError("Retained RDC1 sweep frequency must be positive")
    start = origin if timing["tstart"] is None else timing["tstart"]
    end = origin + shape[1] / prf if timing["tend"] is None else timing["tend"]
    if timing["duration"] is not None:
        if timing["duration"] <= 0:
            raise ValueError("duration must be positive")
        end = start + timing["duration"]
    first, stop = int(round((start - origin) * prf)), int(round((end - origin) * prf))
    if first < 0 or stop > shape[1] or stop <= first:
        raise ValueError("Requested time range is outside retained RDC1")
    integration = timing["integration_time"]
    integration = (stop - first) / prf if integration is None else integration
    hop = integration if timing["hop_time"] is None else timing["hop_time"]
    count, stride = int(round(integration * prf)), int(round(hop * prf))
    if integration <= 0 or hop <= 0 or count < 1 or stride < 1 or count > stop - first:
        raise ValueError("Integration/hop must contain complete sweeps within retained RDC1")
    nfft = getattr(args, "nfft_doppler", None)
    nfft = 1 << (count - 1).bit_length() if nfft is None else int(nfft)
    if nfft < count:
        raise ValueError("--nfft-doppler must be >= the integration sweep count")
    starts = np.arange(first, stop - count + 1, stride, dtype=int)
    view.update(
        recompute=True, pulse_starts=starts, pulse_count=count, prf=prf, nfft=nfft,
        slow_window=getattr(args, "slow_window", "hann"),
        gap_fill=getattr(args, "gap_fill", "nan"),
        shape=(len(starts), shape[0], nfft, shape[2]),
        doppler_hz=np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / prf)),
        frame_tstart_s=origin + starts / prf,
        frame_tend_s=origin + (starts + count) / prf,
    )
    view["metadata"] = dict(view["metadata"], integration_time_s=count / prf,
                            hop_time_s=stride / prf, nfft_doppler=nfft,
                            slow_window=view["slow_window"], sweep_frequency_hz=prf,
                            tstart=origin + first / prf, tend=origin + stop / prf)
    return view


def iter_rdc2_view_chunks(view, channel_chunk_size=8, channel_indices=None,
                          frame_start=0, frame_stop=None,
                          delay_selection=None, doppler_selection=None):
    if not view["recompute"]:
        yield from iter_rdc2_hdf5_channel_chunks(
            view["path"], channel_chunk_size, channel_indices,
            frame_start, frame_stop, delay_selection, doppler_selection,
        )
        return
    channel_chunk_size = int(channel_chunk_size)
    if channel_chunk_size < 1:
        raise ValueError("channel_chunk_size must be >= 1")
    frame_stop = view["shape"][0] if frame_stop is None else frame_stop
    if frame_start < 0 or frame_stop > view["shape"][0] or frame_stop <= frame_start:
        raise ValueError("Invalid RDC2 frame range")
    selected = (np.arange(view["shape"][-1]) if channel_indices is None
                else np.asarray(channel_indices, dtype=int))
    if selected.ndim != 1 or np.any(selected < 0) or np.any(selected >= view["shape"][-1]):
        raise ValueError("Invalid channel indices")
    ds = slice(None) if delay_selection is None else delay_selection
    fs = slice(None) if doppler_selection is None else doppler_selection
    with h5py.File(view["path"], "r") as h5:
        group = h5["rdc1"]
        for start in range(0, len(selected), channel_chunk_size):
            stop = min(start + channel_chunk_size, len(selected))
            idx = selected[start:stop]
            frames = []
            for first in view["pulse_starts"][frame_start:frame_stop]:
                selection = (ds, slice(first, first + view["pulse_count"]), idx)
                rdc1 = group["real"][selection] + 1j * group["imag"][selection]
                if view["gap_fill"] == "zero":
                    rdc1 = np.nan_to_num(rdc1, nan=0, posinf=0, neginf=0)
                cube, _ = doppler_fft_rdc1(rdc1, view["prf"], window=view["slow_window"], nfft=view["nfft"])
                frames.append(cube[:, fs, :])
            yield start, stop, np.stack(frames)
