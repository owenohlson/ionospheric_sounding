# rdc_utils.py

import csv
import json
import os
from datetime import datetime, timedelta

import numpy as np

from lfm_utils import dechirp_delay_axis_ms, window_from_arg


BAD_LWA_STANDS = [
    10, 16, 27, 76, 108, 133, 228, 229, 232, 164,
    93, 60, 20, 163, 155, 116, 6, 110, 178, 21,
]

def get_lwasv_antennas():
    """Index LSL antennas by physical (stand, polarization), never list order."""
    from lsl.common import stations

    result = {}
    for antenna in stations.lwasv.antennas:
        key = (int(antenna.stand.id), int(antenna.pol))
        if key in result:
            raise ValueError(f"Duplicate LWA-SV antenna mapping: {key}")
        result[key] = antenna
    return result


def lwasv_selected_antennas(stands, pols):
    if len(stands) != len(pols):
        raise ValueError("Stand and polarization arrays must have matching lengths")
    lookup = get_lwasv_antennas()
    keys = [(int(s), int(p)) for s, p in zip(stands, pols)]
    missing = [key for key in keys if key not in lookup]
    if missing:
        raise ValueError(f"Missing LWA-SV antenna mappings: {missing}")
    return [lookup[key] for key in keys]


def cable_phase_factors(stands, pols, frequency_hz):
    """Undo relative cable phase at RF center; channel axis is last in RDCs."""
    if frequency_hz is None or not np.isfinite(frequency_hz) or frequency_hz <= 0:
        raise ValueError("Cable correction requires a positive carrier/tuning frequency")
    antennas = lwasv_selected_antennas(stands, pols)
    delays = np.array([a.cable.delay(float(frequency_hz)) for a in antennas], dtype=float)
    if not delays.size or not np.all(np.isfinite(delays)):
        raise ValueError("LSL returned empty or nonfinite cable delays")
    delays -= delays.min()
    return np.exp(2j * np.pi * float(frequency_hz) * delays)


def cached_physical_channel_ids(stands, pols, metadata):
    """Legacy real-data exporters labeled digitizer pairs as stands."""
    kind = metadata.get("channel_id_kind", "physical_stand" if metadata.get("synthetic") else "legacy_input_pair")
    if kind in ("physical_stand", "receiver"):
        return np.asarray(stands, dtype=int), np.asarray(pols, dtype=int)
    if kind != "legacy_input_pair":
        raise ValueError(f"Unknown cached channel ID convention: {kind}")
    by_digitizer = {int(a.digitizer): a for a in get_lwasv_antennas().values() if a.digitizer > 0}
    digitizers = [2 * (int(s) - 1) + int(p) + 1 for s, p in zip(stands, pols)]
    missing = sorted(set(digitizers) - set(by_digitizer))
    if missing:
        raise ValueError(f"Legacy cache digitizers absent from LWA-SV metadata: {missing}")
    antennas = [by_digitizer[d] for d in digitizers]
    metadata["channel_id_kind"] = "physical_stand"
    metadata["channel_digitizers"] = digitizers
    return np.array([a.stand.id for a in antennas]), np.array([a.pol for a in antennas])


def get_lwasv_stand_positions():
    """Return physical stand coordinates in meters from LSL's LWA-SV metadata."""
    try:
        from lsl.common import stations
    except ImportError as exc:
        raise ImportError(
            "LWA-SV antenna positions require LSL; run in your LSL environment."
        ) from exc

    positions = {}
    for antenna in stations.lwasv.antennas:
        stand = antenna.stand
        xyz = (float(stand.x), float(stand.y), float(stand.z))
        if stand.id in positions and positions[stand.id] != xyz:
            raise ValueError(f"Inconsistent LSL coordinates for LWA-SV stand {stand.id}")
        positions[int(stand.id)] = xyz
    return positions


def pol_to_indices(pol):
    p = str(pol).strip().lower()
    if p == "both":
        return [0, 1]
    if p in ("x", "0"):
        return [0]
    if p in ("y", "1"):
        return [1]
    raise ValueError("--pol must be both, x/y, or 0/1")


def _timestamp_datetime(timestamp):
    if timestamp is None:
        return None
    if hasattr(timestamp, "utc_datetime"):
        return timestamp.utc_datetime
    if hasattr(timestamp, "datetime"):
        return timestamp.datetime
    return timestamp


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


def _format_timestamp(timestamp):
    dt = _timestamp_datetime(timestamp)
    if dt is None:
        return "unknown UTC"
    if isinstance(dt, datetime):
        return f"{dt.strftime('%H:%M:%S.%f')[:-3]} UTC"
    return f"{dt} UTC"


def _gap_warning(reason, plot_time, timestamp):
    print(f"{reason} at local plot time {plot_time:.3f}s ({_format_timestamp(timestamp)})", flush=True)


def _gap_fill_array(n_samples, gap_fill):
    if gap_fill == "zero":
        return np.zeros(n_samples, dtype=np.complex64)
    if gap_fill == "nan":
        return np.full(n_samples, np.nan + 1j * np.nan, dtype=np.complex64)
    raise ValueError("--gap-fill must be 'nan' or 'zero'")


def _gap_fill_block(n_streams, n_samples, gap_fill):
    block = _gap_fill_array(n_samples, gap_fill)
    if n_streams == 1:
        return block[None, :]
    return np.tile(block[None, :], (n_streams, 1))


def parse_stands(stand_spec="all", max_stand=256, drop_bad=False):
    if stand_spec is None or str(stand_spec).strip().lower() == "all":
        stands = list(range(1, int(max_stand) + 1))
    else:
        stands = []
        for part in str(stand_spec).split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                lo, hi = part.split("-", 1)
                stands.extend(range(int(lo), int(hi) + 1))
            else:
                stands.append(int(part))
    stands = sorted(dict.fromkeys(stands))
    if drop_bad:
        bad = set(BAD_LWA_STANDS)
        stands = [stand for stand in stands if stand not in bad]
    return stands


def channel_selection(stands, pol):
    pol_indices = pol_to_indices(pol)
    stream_indices = []
    channel_stands = []
    channel_pols = []
    for stand in stands:
        for pol_idx in pol_indices:
            channel_stands.append(int(stand))
            channel_pols.append(int(pol_idx))
    antennas = lwasv_selected_antennas(channel_stands, channel_pols)
    stream_indices = [int(a.digitizer) - 1 for a in antennas]
    if any(i < 0 for i in stream_indices) or len(set(stream_indices)) != len(stream_indices):
        raise ValueError("Selected LWA-SV antennas have invalid or duplicate digitizer mappings")
    return np.asarray(stream_indices, dtype=int), np.asarray(channel_stands, dtype=int), np.asarray(channel_pols, dtype=int)


def polarization_metadata(channel_pols):
    channel_pols = np.asarray(channel_pols, dtype=int)
    unique_pols = sorted(int(pol) for pol in np.unique(channel_pols))
    if unique_pols == [0]:
        mode = "x"
    elif unique_pols == [1]:
        mode = "y"
    elif unique_pols == [0, 1]:
        mode = "both"
    else:
        mode = "unknown"
    return {
        "polarization_mode": mode,
        "polarization_streams": unique_pols,
        "n_polarizations": len(unique_pols),
    }


def estimate_rdc0_bytes(adc_count, pulses, channels, dtype=np.float32):
    return int(adc_count) * int(pulses) * int(channels) * np.dtype(dtype).itemsize * 2


def estimate_rdc2_bytes(frames, delay_bins, doppler_bins, channels, dtype=np.float32):
    return int(frames) * int(delay_bins) * int(doppler_bins) * int(channels) * np.dtype(dtype).itemsize * 2


def format_bytes(n_bytes):
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(n_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0


def reshape_samples_to_rdc0(samples, sweep_length, start_offset_samples=0, adc_start=0, adc_count=None):
    samples = np.asarray(samples)
    if samples.ndim == 1:
        samples = samples[None, :]
    if samples.ndim != 2:
        raise ValueError("samples must be 1D or 2D (channels, samples)")
    if start_offset_samples < 0:
        raise ValueError("start_offset_samples must be >= 0")
    if adc_start < 0:
        raise ValueError("adc_start must be >= 0")
    if adc_count is None:
        adc_count = sweep_length - adc_start
    if adc_count <= 0 or adc_start + adc_count > sweep_length:
        raise ValueError("adc_start/adc_count must select samples inside one sweep")

    samples = samples[:, start_offset_samples:]
    pulses = samples.shape[-1] // sweep_length
    trimmed = samples[:, :pulses * sweep_length]
    cube = trimmed.reshape(samples.shape[0], pulses, sweep_length)
    cube = cube[:, :, adc_start:adc_start + adc_count]
    return np.transpose(cube, (2, 1, 0))


def dechirp_range_fft_rdc0(rdc0, lfm_config, window="hann", adc_start=0):
    """RDC0 (adc, pulses, channels) -> RDC1 (ranges, pulses, channels)."""
    rdc0 = np.asarray(rdc0)
    if rdc0.ndim != 3:
        raise ValueError("rdc0 must have shape (adc, pulses, channels)")
    adc_len = rdc0.shape[0]
    reference = lfm_config.waveform[int(adc_start):int(adc_start) + adc_len].astype(rdc0.dtype, copy=False)
    if reference.size != adc_len:
        raise ValueError("adc_start plus RDC0 ADC length exceeds the reference sweep length")
    w = window_from_arg(adc_len, window).astype(np.float32, copy=False)
    coherent_gain = np.mean(w)
    if coherent_gain == 0:
        raise ValueError(f"Window '{window}' has zero coherent gain")

    beat = rdc0 * np.conj(reference)[:, None, None]
    return np.fft.fftshift(np.fft.fft(beat * w[:, None, None], axis=0) / coherent_gain, axes=0).astype(np.complex64)


def dechirp_range_fft_delay_band_rdc0(rdc0, lfm_config, d_min, d_max, window="hann", adc_start=0):
    """RDC0 -> selected RDC1 delay band without keeping the full RDC1 after selection."""
    rdc0 = np.asarray(rdc0)
    if rdc0.ndim != 3:
        raise ValueError("rdc0 must have shape (adc, pulses, channels)")
    if d_min is None or d_max is None:
        raise ValueError("d_min and d_max are required for delay-band RDC1 processing")

    adc_len = rdc0.shape[0]
    delay_ms, delay_order = dechirp_delay_axis_ms(lfm_config, n_bins=adc_len)
    mask = (delay_ms >= float(d_min)) & (delay_ms <= float(d_max))
    if not np.any(mask):
        raise ValueError(
            "No delay bins selected. "
            f"Available delay range is {delay_ms.min():.3f} to {delay_ms.max():.3f} ms."
        )

    reference = lfm_config.waveform[int(adc_start):int(adc_start) + adc_len].astype(rdc0.dtype, copy=False)
    if reference.size != adc_len:
        raise ValueError("adc_start plus RDC0 ADC length exceeds the reference sweep length")
    w = window_from_arg(adc_len, window).astype(np.float32, copy=False)
    coherent_gain = np.mean(w)
    if coherent_gain == 0:
        raise ValueError(f"Window '{window}' has zero coherent gain")

    beat = rdc0 * np.conj(reference)[:, None, None]
    spectra = np.fft.fftshift(np.fft.fft(beat * w[:, None, None], axis=0) / coherent_gain, axes=0)
    return spectra[delay_order[mask], :, :].astype(np.complex64), delay_ms[mask]


def select_rdc1_delay_band(rdc1, lfm_config, d_min=None, d_max=None):
    delay_ms, delay_order = dechirp_delay_axis_ms(lfm_config, n_bins=rdc1.shape[0])
    ordered = rdc1[delay_order, :, :]
    mask = np.ones(delay_ms.shape, dtype=bool)
    if d_min is not None:
        mask &= delay_ms >= d_min
    if d_max is not None:
        mask &= delay_ms <= d_max
    if not np.any(mask):
        raise ValueError(
            "No delay bins selected. "
            f"Available delay range is {delay_ms.min():.3f} to {delay_ms.max():.3f} ms."
        )
    return ordered[mask, :, :], delay_ms[mask]


def doppler_fft_rdc1(rdc1, prf, window="hann", nfft=None):
    """RDC1 (ranges, pulses, channels) -> RDC2 (ranges, doppler, channels)."""
    rdc1 = np.asarray(rdc1)
    if rdc1.ndim != 3:
        raise ValueError("rdc1 must have shape (ranges, pulses, channels)")
    pulses = rdc1.shape[1]
    if nfft is None:
        nfft = 1 << int(np.ceil(np.log2(max(pulses, 1))))
    if nfft < pulses:
        raise ValueError(
            f"Doppler FFT length ({nfft}) must be >= the number of sweeps ({pulses}); "
            "a shorter FFT truncates the slow-time window. Increase --nfft-doppler or omit it."
        )
    w = window_from_arg(pulses, window).astype(np.float32, copy=False)
    coherent_gain = np.mean(w)
    if coherent_gain == 0:
        raise ValueError(f"Window '{window}' has zero coherent gain")
    x = rdc1 * (w[None, :, None] / coherent_gain)
    rdc2 = np.fft.fftshift(np.fft.fft(x, n=nfft, axis=1), axes=1).astype(np.complex64)
    fd_hz = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / float(prf)))
    return rdc2, fd_hz


def delay_doppler_power_db(rdc2, combine="sum"):
    power = np.abs(rdc2) ** 2
    if combine == "sum":
        out = np.sum(power, axis=2)
    elif combine == "mean":
        out = np.mean(power, axis=2)
    elif combine == "max":
        out = np.max(power, axis=2)
    else:
        raise ValueError("--channel-combine must be one of: sum, mean, max")
    return 10.0 * np.log10(out + 1e-12)


def load_lwa_antenna_positions(channel_stands, strict=True, return_mask=False):
    stand_positions = get_lwasv_stand_positions()
    positions = []
    missing = []
    mask = []
    for stand in channel_stands:
        stand = int(stand)
        if stand not in stand_positions:
            missing.append(stand)
            mask.append(False)
        else:
            positions.append(stand_positions[stand])
            mask.append(True)
    if missing and strict:
        raise ValueError(
            "LSL LWA-SV metadata is missing physical stand(s): "
            f"{sorted(set(missing))}."
        )
    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    if return_mask:
        return positions, np.asarray(mask, dtype=bool), sorted(set(missing))
    return positions


def load_antenna_positions_csv(path, channel_stands):
    positions_by_stand = {}
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            stand = int(row["stand"])
            positions_by_stand[stand] = [float(row["x_m"]), float(row["y_m"]), float(row.get("z_m", 0.0))]
    positions = []
    missing = []
    for stand in channel_stands:
        if int(stand) not in positions_by_stand:
            missing.append(int(stand))
        else:
            positions.append(positions_by_stand[int(stand)])
    if missing:
        raise ValueError(f"Antenna position CSV is missing stand(s): {sorted(set(missing))}")
    return np.asarray(positions, dtype=np.float64)


def steering_vector(positions_m, wavelength_m, az_deg, el_deg):
    az = np.deg2rad(az_deg)
    el = np.deg2rad(el_deg)
    direction = np.array([
        np.cos(el) * np.sin(az),
        np.cos(el) * np.cos(az),
        np.sin(el),
    ])
    phase = 2.0 * np.pi / wavelength_m * (positions_m @ direction)
    return np.exp(1j * phase)


def capon_mvdr_spectrum(
    snapshot,
    positions_m,
    wavelength_m,
    az_grid_deg,
    el_grid_deg,
    diagonal_loading=1e-2,
    method="mvdr",
):
    """Spatial spectrum from one or more snapshots for a range-Doppler cell."""
    x = np.asarray(snapshot, dtype=np.complex128)
    if x.ndim == 1:
        x = x[None, :]
    elif x.ndim != 2:
        x = x.reshape(-1, x.shape[-1])

    good_snapshots = np.all(np.isfinite(x.real) & np.isfinite(x.imag), axis=1)
    if np.count_nonzero(good_snapshots) < 1:
        return np.full((len(el_grid_deg), len(az_grid_deg)), np.nan, dtype=np.float32)
    x = x[good_snapshots]

    good_channels = np.all(np.isfinite(x.real) & np.isfinite(x.imag), axis=0)
    if np.count_nonzero(good_channels) < 2:
        return np.full((len(el_grid_deg), len(az_grid_deg)), np.nan, dtype=np.float32)
    x = x[:, good_channels]
    pos = np.asarray(positions_m)[good_channels]

    R = (x.T @ np.conj(x)) / max(x.shape[0], 1)
    loading = float(diagonal_loading) * (np.trace(R).real / max(R.shape[0], 1) + 1e-12)
    R = R + loading * np.eye(R.shape[0])
    method = str(method).lower()
    R_inv = np.linalg.pinv(R) if method == "mvdr" else None

    spec = np.empty((len(el_grid_deg), len(az_grid_deg)), dtype=np.float32)
    for ei, el in enumerate(el_grid_deg):
        for ai, az in enumerate(az_grid_deg):
            a = steering_vector(pos, wavelength_m, az, el)
            if method == "mvdr":
                denom = np.conj(a) @ R_inv @ a
                spec[ei, ai] = float(1.0 / np.maximum(np.real(denom), 1e-30))
            elif method == "conventional":
                spec[ei, ai] = float(np.real(np.conj(a) @ R @ a) / max(len(a) ** 2, 1))
            else:
                raise ValueError("angle spectrum method must be mvdr or conventional")
    return spec


def write_complex_dataset(group, name, array, compression="gzip", compression_opts=4):
    compression_arg = None if compression == "none" else compression
    opts = None if compression in ("none", "lzf") else compression_opts
    group.create_dataset(f"{name}_real", data=np.asarray(array.real), compression=compression_arg, compression_opts=opts)
    group.create_dataset(f"{name}_imag", data=np.asarray(array.imag), compression=compression_arg, compression_opts=opts)


def read_complex_dataset(group, name):
    return group[f"{name}_real"][...] + 1j * group[f"{name}_imag"][...]


def _open_hdf5_read(path, h5py):
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"RDC0 HDF5 file not found: {path}")
    try:
        return h5py.File(path, "r")
    except OSError as exc:
        raise OSError(
            f"Could not open RDC HDF5 file '{path}'. This usually means the file is "
            "not a valid HDF5/RDC cache, is truncated/corrupted, or was written by a job "
            "that did not finish cleanly. Re-run the RDC exporter or point the input "
            "flag at a valid .h5 file."
        ) from exc


def _rdc0_group(h5, path):
    if "rdc0" not in h5:
        raise ValueError(f"HDF5 file '{path}' does not contain the required '/rdc0' group")
    grp = h5["rdc0"]
    required = ("real", "imag", "channel_stands", "channel_pols")
    missing = [name for name in required if name not in grp]
    if missing:
        raise ValueError(f"HDF5 file '{path}' is missing required RDC0 dataset(s): {', '.join(missing)}")
    if grp["real"].shape != grp["imag"].shape:
        raise ValueError(f"HDF5 file '{path}' has mismatched RDC0 real/imag dataset shapes")
    return grp


def _rdc2_group(h5, path):
    if "rdc2" not in h5:
        raise ValueError(f"HDF5 file '{path}' does not contain the required '/rdc2' group")
    grp = h5["rdc2"]
    required = (
        "real",
        "imag",
        "delay_ms",
        "doppler_hz",
        "frame_tstart_s",
        "frame_tend_s",
        "channel_stands",
        "channel_pols",
    )
    missing = [name for name in required if name not in grp]
    if missing:
        raise ValueError(f"HDF5 file '{path}' is missing required RDC2 dataset(s): {', '.join(missing)}")
    if grp["real"].shape != grp["imag"].shape:
        raise ValueError(f"HDF5 file '{path}' has mismatched RDC2 real/imag dataset shapes")
    if len(grp["real"].shape) not in (3, 4):
        raise ValueError(f"HDF5 file '{path}' RDC2 datasets must have shape (delay, doppler, channels) or (frames, delay, doppler, channels)")
    return grp


def _rdc1_group(h5, path):
    if "rdc1" not in h5:
        raise ValueError(f"HDF5 file '{path}' does not contain '/rdc1'")
    grp = h5["rdc1"]
    for name in ("real", "imag", "delay_ms", "channel_stands", "channel_pols"):
        if name not in grp:
            raise ValueError(f"Missing RDC1 dataset: {name}")
    shape = grp["real"].shape
    if len(shape) != 3 or shape != grp["imag"].shape or any(n == 0 for n in shape):
        raise ValueError("RDC1 real/imag must have matching nonempty (delay, pulses, channels) shapes")
    for name, count in (("delay_ms", shape[0]), ("channel_stands", shape[2]), ("channel_pols", shape[2])):
        if grp[name].shape != (count,):
            raise ValueError(f"RDC1 {name} does not match cube dimensions")
    return grp


def load_rdc1_hdf5_metadata(path):
    """Read standalone RDC1 or the retained RDC1 inside an RDC2 cache."""
    import h5py

    path = os.path.expanduser(path)
    with _open_hdf5_read(path, h5py) as h5:
        grp = _rdc1_group(h5, path)
        metadata = json.loads(h5.attrs.get("metadata_json", "{}"))
        for attr, key in (("tstart_s", "tstart"), ("sweep_frequency_hz", "sweep_frequency_hz")):
            if attr in grp.attrs:
                value = float(grp.attrs[attr])
                if key in metadata and not np.isclose(value, metadata[key], rtol=1e-10, atol=1e-10):
                    raise ValueError(f"Conflicting RDC1 metadata for {key}")
                metadata[key] = value
        for key in ("sample_rate_hz", "sweep_frequency_hz", "bandwidth_hz"):
            value = metadata.get(key)
            if value is None or not np.isfinite(value) or value == 0 or (key != "bandwidth_hz" and value < 0):
                raise ValueError(f"Missing or invalid RDC1 metadata: {key}")
        delay_ms = grp["delay_ms"][...]
        if not np.all(np.isfinite(delay_ms)) or np.any(np.diff(delay_ms) <= 0):
            raise ValueError("RDC1 delay_ms must be finite and strictly increasing")
        stands, pols = cached_physical_channel_ids(grp["channel_stands"][...], grp["channel_pols"][...], metadata)
        return dict(shape=grp["real"].shape, delay_ms=delay_ms,
                    channel_stands=stands, channel_pols=pols, metadata=metadata)


def iter_rdc1_hdf5_channel_chunks(path, channel_chunk_size=8, channel_indices=None,
                                  pulse_start=None, pulse_stop=None, delay_selection=None):
    yield from iter_rdc0_hdf5_channel_chunks(
        path, channel_chunk_size, channel_indices, pulse_start, pulse_stop,
        _stage=1, _delay_selection=delay_selection,
    )


def load_rdc0_hdf5(path):
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required to load RDC HDF5 files") from exc
    path = os.path.expanduser(path)
    with _open_hdf5_read(path, h5py) as h5:
        grp = _rdc0_group(h5, path)
        rdc0 = grp["real"][...] + 1j * grp["imag"][...]
        channel_stands = grp["channel_stands"][...]
        channel_pols = grp["channel_pols"][...]
        adc_sample_index = grp["adc_sample_index"][...] if "adc_sample_index" in grp else np.arange(rdc0.shape[0])
        metadata = {}
        if "metadata_json" in h5.attrs:
            metadata = json.loads(h5.attrs["metadata_json"])
    channel_stands, channel_pols = cached_physical_channel_ids(channel_stands, channel_pols, metadata)
    return {
        "rdc0": rdc0.astype(np.complex64, copy=False),
        "channel_stands": channel_stands,
        "channel_pols": channel_pols,
        "adc_sample_index": adc_sample_index,
        "metadata": metadata,
    }


def load_rdc0_hdf5_metadata(path):
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required to load RDC HDF5 files") from exc
    path = os.path.expanduser(path)
    with _open_hdf5_read(path, h5py) as h5:
        grp = _rdc0_group(h5, path)
        shape = tuple(grp["real"].shape)
        channel_stands = grp["channel_stands"][...]
        channel_pols = grp["channel_pols"][...]
        adc_sample_index = grp["adc_sample_index"][...] if "adc_sample_index" in grp else np.arange(shape[0])
        metadata = {}
        if "metadata_json" in h5.attrs:
            metadata = json.loads(h5.attrs["metadata_json"])
    channel_stands, channel_pols = cached_physical_channel_ids(channel_stands, channel_pols, metadata)
    return {
        "shape": shape,
        "channel_stands": channel_stands,
        "channel_pols": channel_pols,
        "adc_sample_index": adc_sample_index,
        "metadata": metadata,
    }


def iter_rdc0_hdf5_channel_chunks(
    path,
    channel_chunk_size=8,
    channel_indices=None,
    pulse_start=None,
    pulse_stop=None,
    *,
    _stage=0,
    _delay_selection=None,
):
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required to load RDC HDF5 files") from exc
    path = os.path.expanduser(path)
    channel_chunk_size = int(channel_chunk_size)
    if channel_chunk_size < 1:
        raise ValueError("channel_chunk_size must be >= 1")

    with _open_hdf5_read(path, h5py) as h5:
        grp = _rdc0_group(h5, path) if _stage == 0 else _rdc1_group(h5, path)
        real = grp["real"]
        imag = grp["imag"]
        n_channels = real.shape[2]
        pulse_start = 0 if pulse_start is None else int(pulse_start)
        pulse_stop = real.shape[1] if pulse_stop is None else int(pulse_stop)
        if pulse_start < 0 or pulse_stop > real.shape[1] or pulse_stop <= pulse_start:
            raise ValueError("pulse_start/pulse_stop select an invalid RDC0 pulse range")
        if channel_indices is None:
            selected = np.arange(n_channels, dtype=int)
        else:
            selected = np.asarray(channel_indices, dtype=int)
            if selected.ndim != 1:
                raise ValueError("channel_indices must be a 1D sequence")
            if selected.size and (np.min(selected) < 0 or np.max(selected) >= n_channels):
                raise ValueError("channel_indices contains an out-of-range channel")
        for start in range(0, selected.size, channel_chunk_size):
            stop = min(start + channel_chunk_size, selected.size)
            idx = selected[start:stop]
            band = slice(None) if _delay_selection is None else _delay_selection
            chunk = real[band, pulse_start:pulse_stop, idx] + 1j * imag[band, pulse_start:pulse_stop, idx]
            yield start, stop, chunk.astype(np.complex64, copy=False)


def load_rdc2_hdf5_metadata(path):
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required to load RDC HDF5 files") from exc
    path = os.path.expanduser(path)
    with _open_hdf5_read(path, h5py) as h5:
        grp = _rdc2_group(h5, path)
        metadata = {}
        if "metadata_json" in h5.attrs:
            metadata = json.loads(h5.attrs["metadata_json"])
        channel_stands, channel_pols = cached_physical_channel_ids(
            grp["channel_stands"][...], grp["channel_pols"][...], metadata,
        )
        return {
            "shape": ((1,) + grp["real"].shape if grp["real"].ndim == 3 else grp["real"].shape),
            "storage_shape": grp["real"].shape,
            "has_rdc1": "rdc1" in h5,
            "delay_ms": grp["delay_ms"][...],
            "doppler_hz": grp["doppler_hz"][...],
            "frame_tstart_s": grp["frame_tstart_s"][...],
            "frame_tend_s": grp["frame_tend_s"][...],
            "channel_stands": channel_stands,
            "channel_pols": channel_pols,
            "metadata": metadata,
        }


def iter_rdc2_hdf5_channel_chunks(
    path,
    channel_chunk_size=8,
    channel_indices=None,
    frame_start=None,
    frame_stop=None,
    delay_selection=None,
    doppler_selection=None,
):
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required to load RDC HDF5 files") from exc
    path = os.path.expanduser(path)
    channel_chunk_size = int(channel_chunk_size)
    if channel_chunk_size < 1:
        raise ValueError("channel_chunk_size must be >= 1")

    with _open_hdf5_read(path, h5py) as h5:
        grp = _rdc2_group(h5, path)
        real = grp["real"]
        imag = grp["imag"]
        n_frames = 1 if real.ndim == 3 else real.shape[0]
        n_channels = real.shape[-1]
        frame_start = 0 if frame_start is None else int(frame_start)
        frame_stop = n_frames if frame_stop is None else int(frame_stop)
        if frame_start < 0 or frame_stop > n_frames or frame_stop <= frame_start:
            raise ValueError("frame_start/frame_stop select an invalid RDC2 frame range")
        delay_selection = slice(None) if delay_selection is None else delay_selection
        doppler_selection = slice(None) if doppler_selection is None else doppler_selection
        if channel_indices is None:
            selected = np.arange(n_channels, dtype=int)
        else:
            selected = np.asarray(channel_indices, dtype=int)
            if selected.ndim != 1:
                raise ValueError("channel_indices must be a 1D sequence")
            if selected.size and (np.min(selected) < 0 or np.max(selected) >= n_channels):
                raise ValueError("channel_indices contains an out-of-range channel")
        for start in range(0, selected.size, channel_chunk_size):
            stop = min(start + channel_chunk_size, selected.size)
            idx = selected[start:stop]
            if real.ndim == 3:
                chunk = (real[delay_selection, doppler_selection, idx]
                         + 1j * imag[delay_selection, doppler_selection, idx])[None, ...]
            else:
                chunk = (
                    real[frame_start:frame_stop, delay_selection, doppler_selection, idx]
                    + 1j * imag[frame_start:frame_stop, delay_selection, doppler_selection, idx]
                )
            yield start, stop, chunk.astype(np.complex64, copy=False)


def metadata_json(**kwargs):
    return json.dumps(kwargs, indent=2, sort_keys=True)


def export_tbn_rdc0_hdf5(
    idf,
    output_file,
    tstart,
    duration,
    sweep_frequency,
    stands,
    pol,
    start_offset_samples=0,
    adc_start=0,
    adc_count=None,
    gap_fill="nan",
    dtype="float32",
    compression="gzip",
    compression_level=4,
    chunk_pulses=1,
):
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required for RDC HDF5 export") from exc

    fs = float(idf.get_info("sample_rate"))
    try:
        tuning_frequency_hz = float(idf.get_info("freq1"))
    except Exception:
        tuning_frequency_hz = None
    sweep_length = int(round(fs / float(sweep_frequency)))
    if adc_count is None:
        adc_count = sweep_length - int(adc_start)
    total_pulses = int(np.floor((duration * fs - start_offset_samples) / sweep_length))
    if total_pulses < 1:
        raise ValueError("Requested duration does not contain one complete sweep after alignment")

    stream_indices, channel_stands, channel_pols = channel_selection(stands, pol)
    n_channels = len(stream_indices)
    out_dtype = np.dtype(dtype)
    chunks = (int(adc_count), int(chunk_pulses), min(n_channels, 32))
    compression_arg = None if compression == "none" else compression
    opts = None if compression in ("none", "lzf") else int(compression_level)

    output_file = os.path.expanduser(output_file)
    os.makedirs(os.path.dirname(os.path.abspath(output_file)) or ".", exist_ok=True)
    idf.offset(float(tstart))
    chunk_seconds = min(0.25, float(duration))
    target_seconds = float(duration)
    timestamp_gap_tolerance = max(2.0 / fs, 0.001)
    timetag_skip_seconds = 0.002

    buffer = np.empty((n_channels, 0), dtype=np.complex64)
    got = 0.0
    timeline_elapsed = 0.0
    first_timestamp = None
    first_requested_timestamp = None
    pulse_idx = 0
    offset_pending = int(start_offset_samples)
    gap_info = {"mode": gap_fill, "events": 0, "samples": 0, "sample_rate": fs}

    with h5py.File(output_file, "w") as h5:
        grp = h5.create_group("rdc0")
        dset_real = grp.create_dataset(
            "real",
            shape=(int(adc_count), total_pulses, n_channels),
            maxshape=(int(adc_count), total_pulses, n_channels),
            dtype=out_dtype,
            chunks=chunks,
            compression=compression_arg,
            compression_opts=opts,
        )
        dset_imag = grp.create_dataset(
            "imag",
            shape=(int(adc_count), total_pulses, n_channels),
            maxshape=(int(adc_count), total_pulses, n_channels),
            dtype=out_dtype,
            chunks=chunks,
            compression=compression_arg,
            compression_opts=opts,
        )
        invalid_counts = grp.create_dataset("invalid_counts", (total_pulses, n_channels), dtype="int64",
                                            maxshape=(total_pulses, n_channels))

        while got < target_seconds and pulse_idx < total_pulses:
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
                            gap_samples = int(round(min(timestamp_gap, target_seconds - got) * fs))
                            if gap_samples > 0:
                                _gap_warning("Timestamp gap; inserting RDC0 fill", tstart + got, _timestamp_at(first_requested_timestamp, got))
                                buffer = np.concatenate([buffer, _gap_fill_block(n_channels, gap_samples, "nan")], axis=-1)
                                gap_info["events"] += 1
                                gap_info["samples"] += gap_samples
                                gap_seconds = gap_samples / fs
                                got += gap_seconds
                                timeline_elapsed += gap_seconds
            except RuntimeError as exc:
                if "Invalid timetag skip" not in str(exc):
                    raise
                gap_seconds = min(timetag_skip_seconds, target_seconds - got)
                gap_samples = int(round(gap_seconds * fs))
                _gap_warning("Invalid timetag skip; inserting RDC0 fill", tstart + got, _timestamp_at(first_requested_timestamp, got))
                buffer = np.concatenate([buffer, _gap_fill_block(n_channels, gap_samples, "nan")], axis=-1)
                gap_info["events"] += 1
                gap_info["samples"] += gap_samples
                got += gap_seconds
                timeline_elapsed += gap_seconds
                idf.offset(gap_seconds)
                continue

            if data.ndim != 2:
                raise RuntimeError(f"Unexpected data shape from idf.read(): {data.shape}")
            if np.max(stream_indices) >= data.shape[0]:
                raise ValueError(f"Requested stream index {np.max(stream_indices)} but read data has {data.shape[0]} antpols")
            keep_samples = min(data.shape[1], int(round((target_seconds - got) * fs)))
            if keep_samples <= 0:
                break
            buffer = np.concatenate([buffer, data[stream_indices, :keep_samples].astype(np.complex64, copy=False)], axis=-1)
            del data

            read_seconds = keep_samples / fs
            got += read_seconds
            timeline_elapsed += read_seconds

            if offset_pending > 0 and buffer.shape[1] > 0:
                drop = min(offset_pending, buffer.shape[1])
                buffer = buffer[:, drop:]
                offset_pending -= drop

            while buffer.shape[1] >= sweep_length and pulse_idx < total_pulses:
                sweep = buffer[:, :sweep_length]
                adc = sweep[:, adc_start:adc_start + adc_count].T
                invalid_counts[pulse_idx] = np.count_nonzero(~np.isfinite(adc), axis=0)
                if gap_fill == "zero":
                    adc = np.where(np.isfinite(adc), adc, 0)
                dset_real[:, pulse_idx, :] = adc.real.astype(out_dtype, copy=False)
                dset_imag[:, pulse_idx, :] = adc.imag.astype(out_dtype, copy=False)
                buffer = buffer[:, sweep_length:]
                pulse_idx += 1
                if pulse_idx % 10 == 0:
                    print(f"Wrote RDC0 pulse {pulse_idx}/{total_pulses}", flush=True)

        if pulse_idx < total_pulses:
            invalid_counts.resize((pulse_idx, n_channels))
            dset_real.resize((int(adc_count), pulse_idx, n_channels))
            dset_imag.resize((int(adc_count), pulse_idx, n_channels))

        grp.create_dataset("channel_stands", data=channel_stands)
        grp.create_dataset("channel_pols", data=channel_pols)
        grp.create_dataset("adc_sample_index", data=np.arange(adc_start, adc_start + adc_count, dtype=np.int32))
        grp.create_dataset("pulse_index", data=np.arange(pulse_idx, dtype=np.int32))
        h5.attrs["metadata_json"] = metadata_json(
            rdc_stage=0,
            dimension_order=["adc", "pulses", "channels"],
            gap_fill=gap_fill,
            dtype=str(out_dtype),
            compression=compression,
            compression_level=compression_level,
            shape_adc_pulses_channels=[int(adc_count), int(pulse_idx), int(n_channels)],
            station="lwasv",
            source_format="tbn",
            pulse0_utc=(_timestamp_at(first_requested_timestamp, start_offset_samples / fs).isoformat()
                        if _timestamp_at(first_requested_timestamp, start_offset_samples / fs) is not None else None),
            channel_id_kind="physical_stand",
            channel_digitizers=(stream_indices + 1).tolist(),
            cable_phase_corrected=False,
            tstart=float(tstart),
            duration_requested_s=float(duration),
            sample_rate_hz=fs,
            tuning_frequency_hz=tuning_frequency_hz,
            sweep_frequency_hz=float(sweep_frequency),
            sweep_length_samples=int(sweep_length),
            start_offset_samples=int(start_offset_samples),
            adc_start=int(adc_start),
            adc_count=int(adc_count),
            stands=[int(s) for s in stands],
            pol=str(pol),
            n_channels=int(n_channels),
            n_stands=int(len(np.unique(channel_stands))),
            **polarization_metadata(channel_pols),
            gap_info=gap_info,
        )

    return {
        "output_file": output_file,
        "shape": (int(adc_count), int(pulse_idx), int(n_channels)),
        "bytes_uncompressed": estimate_rdc0_bytes(adc_count, pulse_idx, n_channels, out_dtype),
        "gap_info": gap_info,
    }
