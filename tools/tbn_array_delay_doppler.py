# tbn_array_delay_doppler.py

import argparse
import json
import os
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rdc_cache import prepare_rdc2_view, iter_rdc2_view_chunks

from lfm_utils import LFMWaveform, reference_gate_frequency_from_args
from plotting_utils import (
    INFO_PANEL_RECT,
    _add_colorbar,
    _add_delay_range_axis,
    _add_info_panel,
    _format_delay_peaks_note,
    _lfm_info_rows,
    _new_figure_with_info,
    _savefig,
    _tight_layout_with_info,
    _timestamp_sweep_offset_samples,
)
from rdc_utils import (
    cable_phase_factors,
    lwasv_selected_antennas,
    BAD_LWA_STANDS,
    capon_mvdr_spectrum,
    dechirp_range_fft_delay_band_rdc0,
    doppler_fft_rdc1,
    export_tbn_rdc0_hdf5,
    format_bytes,
    iter_rdc0_hdf5_channel_chunks,
    load_lwa_antenna_positions,
    load_rdc0_hdf5_metadata,
    parse_stands,
    polarization_metadata,
    steering_vector,
)
from tbn_utils import file_timestamp_range_note, gap_fill_note, lsl_open_tbn, lsl_print_metadata, set_default_tbn_time_bounds


def probe_start_timestamp(filename, tstart):
    idf = lsl_open_tbn(filename)
    idf.offset(float(tstart))
    readT, timestamp, data = idf.read(0.001)
    del data
    return timestamp


def maybe_make_rdc0_cache(args, fs, start_offset_samples):
    if args.rdc0_input:
        return os.path.expanduser(args.rdc0_input), False
    if args.rdc0_cache:
        cache_path = os.path.expanduser(args.rdc0_cache)
        delete_cache = False
    else:
        fd, cache_path = tempfile.mkstemp(prefix="rdc0_array_", suffix=".h5")
        os.close(fd)
        delete_cache = True

    stands = parse_stands(args.stands, max_stand=args.max_stand, drop_bad=args.drop_bad_stands)
    idf = lsl_open_tbn(args.input_file)
    result = export_tbn_rdc0_hdf5(
        idf=idf,
        output_file=cache_path,
        tstart=args.tstart,
        duration=args.tend - args.tstart,
        sweep_frequency=args.sweep_frequency,
        stands=stands,
        pol=args.pol,
        start_offset_samples=start_offset_samples,
        adc_start=args.adc_start,
        adc_count=args.adc_count,
        gap_fill=args.gap_fill,
        dtype=args.rdc0_dtype,
        compression=args.rdc0_compression,
        compression_level=args.rdc0_compression_level,
    )
    print(f"RDC0 cache shape: {result['shape']} ({format_bytes(result['bytes_uncompressed'])} uncompressed)", flush=True)
    return cache_path, delete_cache


def write_angle_output(path, spectra, az_grid, el_grid, delay_cells, doppler_cells, metadata=None):
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required for angle output") from exc
    path = os.path.expanduser(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with h5py.File(path, "w") as h5:
        h5.create_dataset("mvdr_power", data=spectra.astype(np.float32), compression="gzip", compression_opts=4)
        h5.create_dataset("azimuth_deg", data=az_grid)
        h5.create_dataset("elevation_deg", data=el_grid)
        h5.create_dataset("cell_delay_ms", data=delay_cells)
        h5.create_dataset("cell_doppler_hz", data=doppler_cells)
        if metadata:
            h5.attrs["metadata_json"] = json.dumps(metadata, indent=2, sort_keys=True)


def plot_angle_spectra(
    path,
    spectra,
    az_grid,
    el_grid,
    delay_cells,
    doppler_cells,
    vmin=-40.0,
    vmax=0.0,
    title="Capon/MVDR Angle Spectrum",
    info_rows=None,
    corner_note=None,
    lower_note=None,
    footer_note=None,
    cell_snapshot_info=None,
    cell_labels=None,
    normalize_peak=False,
):
    spectra = np.asarray(spectra, dtype=float)
    if spectra.ndim != 3:
        raise ValueError("angle spectra must have shape (cell, elevation, azimuth)")
    path = os.path.expanduser(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)

    az_plot, spectra_plot = _angle_plot_azimuth_centers(az_grid, spectra)
    theta_edges = np.deg2rad(_axis_edges_from_centers(az_plot))
    radius_centers = 90.0 - np.asarray(el_grid, dtype=float)
    radius_order = np.argsort(radius_centers)
    radius_edges = _axis_edges_from_centers(radius_centers[radius_order])
    radial_tick_r = np.array([0.0, 30.0, 60.0, 90.0])
    radial_tick_labels = [f"{90.0 - r:.0f}" for r in radial_tick_r]

    n_cells = spectra.shape[0]
    n_cols = min(3, n_cells)
    n_rows = int(np.ceil(n_cells / n_cols))
    has_info = bool(info_rows or corner_note or lower_note or footer_note)
    if has_info:
        fig = plt.figure(figsize=(14, max(5.2, 4.3 * n_rows)))
        left, bottom, width, height = INFO_PANEL_RECT
        col_gap = 0.035
        row_gap = 0.09 if n_rows > 1 else 0.0
        ax_w = (width - col_gap * (n_cols - 1)) / max(n_cols, 1)
        ax_h = (height - row_gap * (n_rows - 1)) / max(n_rows, 1)
        axes = np.empty((n_rows, n_cols), dtype=object)
        for row in range(n_rows):
            for col in range(n_cols):
                x0 = left + col * (ax_w + col_gap)
                y0 = bottom + (n_rows - 1 - row) * (ax_h + row_gap)
                axes[row, col] = fig.add_axes([x0, y0, ax_w, ax_h], projection="polar")
    else:
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(5.2 * n_cols, 4.3 * n_rows),
            squeeze=False,
            subplot_kw={"projection": "polar"},
        )
    last_pcm = None
    lower_lines = []
    for idx, ax in enumerate(axes.flat):
        if idx >= n_cells:
            ax.axis("off")
            continue
        spec = spectra_plot[idx][radius_order, :]
        peak = np.nanmax(spec)
        reference = peak if normalize_peak else 1.0
        spec_db = np.full_like(spec, np.nan)
        valid = np.isfinite(spec) & (spec >= 0)
        if np.isfinite(reference) and reference > 0:
            spec_db[valid] = 10.0 * np.log10(np.maximum(spec[valid] / reference, 1e-30))
        label_text = str(cell_labels[idx]).lower() if cell_labels is not None and idx < len(cell_labels) else ""
        suppress_cell_title = "window" in label_text or "beam pattern" in label_text
        if np.any(np.isfinite(spec_db)):
            spread = float(np.nanmax(spec_db) - np.nanmin(spec_db))
            pi = np.unravel_index(np.nanargmax(spec), spec.shape)
            if cell_labels is not None and idx < len(cell_labels):
                cell_line = str(cell_labels[idx]) if suppress_cell_title else f"Cell {idx + 1}: {cell_labels[idx]}"
            else:
                cell_line = f"Cell {idx + 1}: delay {delay_cells[idx]:.3f} ms, Doppler {doppler_cells[idx]:.3f} Hz"
            if cell_snapshot_info is not None and idx < len(cell_snapshot_info):
                snap = cell_snapshot_info[idx]
                cell_line += f", snapshots {snap.get('snapshots', '?')}"
                if "delay_bins" in snap and "doppler_bins" in snap:
                    cell_line += f" ({snap['delay_bins']} x {snap['doppler_bins']} bins)"
            lower_lines.append(cell_line)
            lower_lines.append(
                f"Angle peak {idx + 1}: az {az_plot[pi[1]]:.2f} deg, "
                f"el {el_grid[radius_order[pi[0]]]:.2f} deg, span {spread:.2f} dB"
            )
        last_pcm = ax.pcolormesh(theta_edges, radius_edges, spec_db, shading="auto", cmap="inferno", vmin=vmin, vmax=vmax)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_rlim(0.0, 90.0)
        ax.set_rticks(radial_tick_r)
        ax.set_yticklabels(radial_tick_labels)
        ax.set_rlabel_position(135)
        ax.grid(color="white", alpha=0.28, linewidth=0.6)
        ax.set_title("" if suppress_cell_title else f"Cell {idx + 1}")
    if last_pcm is not None:
        cbar = fig.colorbar(last_pcm, ax=axes.ravel().tolist(), pad=0.08)
        cbar.set_label("Normalized Beam Power [dB]" if normalize_peak else "Angle Power [dB re 1 processing unit]")
    combined_lower_note = "\n".join([note for note in [lower_note, "\n".join(lower_lines)] if note])
    _add_info_panel(
        fig,
        info_rows=info_rows,
        corner_note=corner_note,
        lower_note=combined_lower_note,
        footer_note=footer_note,
    )
    if title:
        fig.suptitle(title)
    _tight_layout_with_info(fig, has_info=has_info)
    _savefig(path, dpi=300)
    plt.close(fig)


def _axis_edges_from_centers(centers):
    centers = np.asarray(centers, dtype=float)
    if centers.ndim != 1 or centers.size == 0:
        raise ValueError("Axis centers must be a non-empty 1D array")
    if centers.size == 1:
        step = 1.0
        return np.array([centers[0] - 0.5 * step, centers[0] + 0.5 * step], dtype=float)
    diffs = np.diff(centers)
    first = centers[0] - 0.5 * diffs[0]
    last = centers[-1] + 0.5 * diffs[-1]
    mids = 0.5 * (centers[:-1] + centers[1:])
    return np.concatenate([[first], mids, [last]]).astype(float)


def _angle_plot_azimuth_centers(az_grid, spectra):
    az_grid = np.asarray(az_grid, dtype=float)
    spectra = np.asarray(spectra, dtype=float)
    if az_grid.size > 1 and np.isclose(abs(az_grid[-1] - az_grid[0]), 360.0):
        return az_grid[:-1], spectra[:, :, :-1]
    return az_grid, spectra


def pol_to_index(pol):
    p = str(pol).strip().lower()
    if p in ("x", "0"):
        return 0
    if p in ("y", "1"):
        return 1
    raise ValueError("--pol must be x/y or 0/1 for array beamforming")


def select_array_channels(channel_stands, channel_pols, pol, drop_bad=True, require_positions=True,
                          operational_only=False):
    channel_stands = np.asarray(channel_stands, dtype=int)
    channel_pols = np.asarray(channel_pols, dtype=int)
    wanted_pol = pol_to_index(pol)
    mask = channel_pols == wanted_pol
    if drop_bad:
        mask &= ~np.isin(channel_stands, np.asarray(BAD_LWA_STANDS, dtype=int))
    if operational_only:
        candidates = np.flatnonzero(mask)
        antennas = lwasv_selected_antennas(channel_stands[candidates], channel_pols[candidates])
        mask[candidates] &= np.array([a.combined_status == 33 for a in antennas])

    indices = np.flatnonzero(mask)
    stands = channel_stands[indices]
    pols = channel_pols[indices]
    if indices.size == 0:
        raise ValueError("No RDC0 channels remain after applying --pol/--drop-bad-stands selection")

    positions = None
    missing_position_stands = []
    if require_positions:
        positions, position_mask, missing_position_stands = load_lwa_antenna_positions(
            stands,
            strict=False,
            return_mask=True,
        )
        if missing_position_stands:
            indices = indices[position_mask]
            stands = stands[position_mask]
            pols = pols[position_mask]
        if indices.size == 0:
            raise ValueError("No selected channels have LSL LWA-SV antenna positions")
        if len(np.unique(stands)) != len(stands):
            raise ValueError("Array beamforming requires one selected channel per stand; choose one polarization with --pol")
    return indices, stands, pols, positions, missing_position_stands


def array_cable_factors(args, metadata, stands, pols, frequency_hz):
    """Resolve correction once for all selected channels, before chunking."""
    mode = getattr(args, "cable_correction", "auto")
    already_corrected = metadata.get("cable_phase_corrected", False)
    if mode == "off":
        note = "already corrected in input" if already_corrected else "off"
    elif already_corrected:
        note = "already corrected in input"
    elif mode == "auto" and metadata.get("synthetic", False) and not metadata.get("cable_delays_injected", False):
        note = "ideal simulation (not applied)"
    else:
        factors = cable_phase_factors(stands, pols, frequency_hz)
        note = f"LSL LWA-SV, {frequency_hz / 1e6:.6f} MHz"
        args._cable_note = note
        print(f"Cable phase correction: {note}", flush=True)
        return factors
    args._cable_note = note
    print(f"Cable phase correction: {note}", flush=True)
    return np.ones(len(stands), dtype=np.complex128)


def estimate_spatial_covariance(
    rdc2,
    diagonal_loading=1e-2,
    max_power_percentile=None,
):
    rdc2 = np.asarray(rdc2)
    if rdc2.ndim != 3:
        raise ValueError("rdc2 must have shape (delay, doppler, channels)")
    n_channels = rdc2.shape[2]
    snapshots = rdc2.reshape(-1, n_channels).astype(np.complex128, copy=False)
    finite = np.isfinite(snapshots.real) & np.isfinite(snapshots.imag)
    usable = np.all(finite, axis=1)
    if max_power_percentile is not None and np.any(usable):
        snapshot_power = np.mean(np.abs(snapshots) ** 2, axis=1)
        threshold = np.nanpercentile(snapshot_power[usable], float(max_power_percentile))
        percentile_usable = usable & (snapshot_power <= threshold)
        if np.count_nonzero(percentile_usable) >= max(2, n_channels):
            usable = percentile_usable
    if np.count_nonzero(usable) < 2:
        raise ValueError("Not enough finite delay-Doppler snapshots to estimate spatial covariance")
    snapshots = snapshots[usable]
    covariance = (snapshots.T @ snapshots.conj()) / snapshots.shape[0]
    loading = float(diagonal_loading) * (np.trace(covariance).real / max(n_channels, 1) + 1e-12)
    return covariance + loading * np.eye(n_channels)


def beamformer_weights(positions_m, carrier_frequency_hz, look_az_deg, look_el_deg, method, covariance=None):
    if carrier_frequency_hz is None:
        raise ValueError("--carrier-frequency is required for beamforming when RDC0 metadata does not provide TBN freq1")
    wavelength = 299_792_458.0 / float(carrier_frequency_hz)
    a = steering_vector(positions_m, wavelength, look_az_deg, look_el_deg).astype(np.complex128)
    method = str(method).lower()
    if method == "conventional":
        weights = a / max(len(a), 1)
    elif method == "mvdr":
        if covariance is None:
            raise ValueError("MVDR beamforming requires a spatial covariance matrix")
        inv_covariance = np.linalg.pinv(covariance)
        denom = np.conj(a) @ inv_covariance @ a
        if np.abs(denom) < 1e-30:
            raise ValueError("MVDR denominator is too close to zero; try larger --diagonal-loading")
        weights = (inv_covariance @ a) / denom
    else:
        raise ValueError("--beamformer must be mvdr, conventional, or incoherent")

    return weights


def scan_look_direction(positions_m, carrier_frequency_hz, covariance, az_grid, el_grid, method="mvdr"):
    wavelength = 299_792_458.0 / float(carrier_frequency_hz)
    method = str(method).lower()
    inv_covariance = np.linalg.pinv(covariance) if method == "mvdr" else None
    best = (-np.inf, None, None)
    for el in el_grid:
        for az in az_grid:
            a = steering_vector(positions_m, wavelength, az, el).astype(np.complex128)
            if method == "mvdr":
                denom = np.conj(a) @ inv_covariance @ a
                score = 1.0 / max(float(np.real(denom)), 1e-30)
            elif method == "conventional":
                score = float(np.real(np.conj(a) @ covariance @ a) / max(len(a) ** 2, 1))
            else:
                raise ValueError("Look-direction scan requires mvdr or conventional beamforming")
            if np.isfinite(score) and score > best[0]:
                best = (score, float(az), float(el))
    if best[1] is None:
        raise ValueError("Could not auto-select a finite look direction")
    return best


def spatial_spectrum_from_covariance(covariance, positions_m, carrier_frequency_hz, az_grid, el_grid, method="mvdr"):
    wavelength = 299_792_458.0 / float(carrier_frequency_hz)
    method = str(method).lower()
    covariance = np.asarray(covariance, dtype=np.complex128)
    inv_covariance = np.linalg.pinv(covariance) if method == "mvdr" else None
    spec = np.empty((len(el_grid), len(az_grid)), dtype=np.float32)
    for ei, el in enumerate(el_grid):
        for ai, az in enumerate(az_grid):
            a = steering_vector(positions_m, wavelength, az, el).astype(np.complex128)
            if method == "mvdr":
                denom = np.conj(a) @ inv_covariance @ a
                spec[ei, ai] = float(1.0 / max(float(np.real(denom)), 1e-30))
            elif method == "conventional":
                spec[ei, ai] = float(np.real(np.conj(a) @ covariance @ a) / max(len(a) ** 2, 1))
            else:
                raise ValueError("angle spectrum method must be mvdr or conventional")
    return spec


def array_beam_pattern_spectrum(positions_m, carrier_frequency_hz, look_az_deg, look_el_deg, az_grid, el_grid):
    wavelength = 299_792_458.0 / float(carrier_frequency_hz)
    a0 = steering_vector(positions_m, wavelength, look_az_deg, look_el_deg).astype(np.complex128)
    weights = a0 / max(len(a0), 1)
    spec = np.empty((len(el_grid), len(az_grid)), dtype=np.float32)
    for ei, el in enumerate(el_grid):
        for ai, az in enumerate(az_grid):
            a = steering_vector(positions_m, wavelength, az, el).astype(np.complex128)
            spec[ei, ai] = float(np.abs(np.conj(weights) @ a) ** 2)
    return spec


def write_beam_pattern_plot(
    path,
    positions_m,
    carrier_frequency_hz,
    look_az_deg,
    look_el_deg,
    az_grid,
    el_grid,
    info_rows=None,
    corner_note=None,
    footer_note=None,
    vmin=-40.0,
    vmax=0.0,
):
    if look_az_deg is None or look_el_deg is None:
        raise ValueError("Beam-pattern plotting requires a resolved look direction")
    spectrum = array_beam_pattern_spectrum(
        positions_m,
        carrier_frequency_hz,
        look_az_deg,
        look_el_deg,
        az_grid,
        el_grid,
    )
    rows = list(info_rows or [])
    rows.extend([
        ("Pattern", "conventional array factor"),
        ("Look az", f"{look_az_deg:.2f} deg"),
        ("Look el", f"{look_el_deg:.2f} deg"),
    ])
    plot_angle_spectra(
        path,
        spectrum[None, :, :],
        az_grid,
        el_grid,
        np.array([np.nan]),
        np.array([np.nan]),
        vmin=vmin,
        vmax=vmax,
        title="Array Beam Pattern",
        info_rows=rows,
        corner_note=corner_note,
        footer_note=footer_note,
        cell_labels=["beam pattern reference"],
        normalize_peak=True,
    )


def normalized_spectrum_span_db(spectrum):
    spectrum = np.asarray(spectrum, dtype=float)
    if not np.any(np.isfinite(spectrum)):
        return np.nan
    peak = np.nanmax(spectrum)
    if not np.isfinite(peak) or peak <= 0:
        return np.nan
    spectrum_db = 10.0 * np.log10(spectrum / peak + 1e-30)
    return float(np.nanmax(spectrum_db) - np.nanmin(spectrum_db))


def fallback_if_angle_spectrum_flat(
    spectrum,
    fallback_spectrum,
    requested_method,
    fallback_mode="conventional",
    threshold_db=0.25,
):
    if str(requested_method).lower() != "mvdr" or str(fallback_mode).lower() == "none":
        return spectrum, None
    span = normalized_spectrum_span_db(spectrum)
    if np.isfinite(span) and span >= float(threshold_db):
        return spectrum, None
    fallback_span = normalized_spectrum_span_db(fallback_spectrum)
    note = (
        f"MVDR angle spectrum was flat ({span:.3g} dB span); "
        f"plotted conventional fallback ({fallback_span:.3g} dB span)."
    )
    return fallback_spectrum, note


def beamform_rdc2(rdc2, positions_m, carrier_frequency_hz, look_az_deg, look_el_deg, method="mvdr", covariance=None):
    rdc2 = np.asarray(rdc2)
    if rdc2.ndim != 3:
        raise ValueError("rdc2 must have shape (delay, doppler, channels)")
    if rdc2.shape[2] != len(positions_m):
        raise ValueError("rdc2 channel count does not match antenna position count")
    weights = beamformer_weights(
        positions_m,
        carrier_frequency_hz,
        look_az_deg,
        look_el_deg,
        method,
        covariance=covariance,
    )
    return np.tensordot(rdc2, np.conj(weights), axes=([2], [0]))


def array_delay_doppler_power_from_rdc2(rdc2_plot, args, positions, fc, blank_on_insufficient=False):
    if getattr(args, "gap_fill", None) == "zero":
        rdc2_plot, _ = zero_fill_nonfinite_complex(rdc2_plot)
    look_auto = args.beamformer in ("mvdr", "conventional") and args.look_az is None and args.look_el is None
    look_az = args.look_az
    look_el = args.look_el
    covariance_note = None
    frame_warning = None

    if args.beamformer == "incoherent":
        power = np.abs(rdc2_plot) ** 2
        if args.channel_combine == "sum":
            power_linear = np.nansum(power, axis=2)
        elif args.channel_combine == "mean":
            power_linear = np.nanmean(power, axis=2)
        elif args.channel_combine == "max":
            power_linear = np.nanmax(power, axis=2)
        else:
            raise ValueError("--channel-combine must be one of: sum, mean, max")
        return 10.0 * np.log10(power_linear + 1e-12), look_az, look_el, look_auto, covariance_note, frame_warning

    try:
        signal_covariance = estimate_spatial_covariance(rdc2_plot, diagonal_loading=args.diagonal_loading)
        covariance = signal_covariance
        covariance_note = "all cells"
        if args.beamformer == "mvdr" and args.mvdr_covariance == "background":
            covariance = estimate_spatial_covariance(
                rdc2_plot,
                diagonal_loading=args.diagonal_loading,
                max_power_percentile=args.mvdr_background_percentile,
            )
            covariance_note = f"background <= {args.mvdr_background_percentile:g} pct"
        if look_auto:
            az_grid = np.arange(args.az_min, args.az_max + 0.5 * args.az_step, args.az_step)
            el_grid = np.arange(args.el_min, args.el_max + 0.5 * args.el_step, args.el_step)
            scan_method = "conventional" if args.beamformer == "mvdr" else args.beamformer
            _, look_az, look_el = scan_look_direction(
                positions,
                fc,
                signal_covariance,
                az_grid,
                el_grid,
                method=scan_method,
            )
        beamformed = beamform_rdc2(
            rdc2_plot,
            positions,
            fc,
            look_az,
            look_el,
            method=args.beamformer,
            covariance=covariance,
        )
        return 10.0 * np.log10(np.abs(beamformed) ** 2 + 1e-12), look_az, look_el, look_auto, covariance_note, frame_warning
    except ValueError as exc:
        if not blank_on_insufficient or "Not enough finite delay-Doppler snapshots" not in str(exc):
            raise
        frame_warning = str(exc)
        return None, look_az, look_el, look_auto, "unavailable", frame_warning


def format_time_span(metadata, pulses, sweep_frequency):
    tstart = metadata.get("tstart")
    duration = pulses / float(sweep_frequency)
    if tstart is None:
        return f"0.000-{duration:.3f} s"
    return f"{float(tstart):.3f}-{float(tstart) + duration:.3f} s"


def apply_rdc2_metadata_defaults(args, metadata):
    metadata = metadata or {}
    if args.sweep_frequency is None and metadata.get("sweep_frequency_hz") is not None:
        args.sweep_frequency = float(metadata["sweep_frequency_hz"])
    if args.bandwidth is None and metadata.get("bandwidth_hz") is not None:
        args.bandwidth = float(metadata["bandwidth_hz"])
    if args.d_min is None and metadata.get("d_min_ms") is not None:
        args.d_min = float(metadata["d_min_ms"])
    if args.d_max is None and metadata.get("d_max_ms") is not None:
        args.d_max = float(metadata["d_max_ms"])
    if metadata.get("dechirp_window") is not None:
        args.dechirp_window = metadata["dechirp_window"]
    if metadata.get("slow_window") is not None:
        args.slow_window = metadata["slow_window"]
    if metadata.get("nfft_doppler") is not None:
        args.nfft_doppler = int(metadata["nfft_doppler"])
    if args.fd_min is None and metadata.get("fd_min_hz") is not None:
        args.fd_min = float(metadata["fd_min_hz"])
    if args.fd_max is None and metadata.get("fd_max_hz") is not None:
        args.fd_max = float(metadata["fd_max_hz"])


def zero_fill_nonfinite_complex(x):
    finite = np.isfinite(x.real) & np.isfinite(x.imag)
    count = int(x.size - np.count_nonzero(finite))
    if count:
        x = np.nan_to_num(x, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return x, count


def contiguous_axis_selection(values, vmin=None, vmax=None, axis_name="axis"):
    values = np.asarray(values)
    mask = np.ones(values.shape, dtype=bool)
    if vmin is not None:
        mask &= values >= float(vmin)
    if vmax is not None:
        mask &= values <= float(vmax)
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        raise ValueError(f"No cached RDC2 {axis_name} bins selected; check requested bounds")
    if np.any(np.diff(indices) != 1):
        raise ValueError(f"Requested cached RDC2 {axis_name} bins are not contiguous")
    selection = slice(int(indices[0]), int(indices[-1]) + 1)
    return selection, values[selection]


def load_rdc2_frame_for_array(args, frame_idx=0, require_positions=True):
    loaded = getattr(args, "_rdc2_view", None)
    if loaded is None:
        loaded = prepare_rdc2_view(args)
        args._rdc2_view = loaded
    metadata = loaded["metadata"]
    apply_rdc2_metadata_defaults(args, metadata)
    shape = loaded["shape"]
    frame_idx = int(frame_idx)
    if frame_idx < 0 or frame_idx >= shape[0]:
        raise ValueError(f"--rdc2-frame-index {frame_idx} is out of range for {shape[0]} RDC2 frame(s)")
    delay_selection, delay_ms = contiguous_axis_selection(
        loaded["delay_ms"],
        args.d_min,
        args.d_max,
        axis_name="delay",
    )
    doppler_selection, fd_hz = contiguous_axis_selection(
        loaded["doppler_hz"],
        args.fd_min,
        args.fd_max,
        axis_name="Doppler",
    )

    selected_channel_indices, channel_stands, channel_pols, positions, missing_position_stands = select_array_channels(
        loaded["channel_stands"],
        loaded["channel_pols"],
        args.pol,
        drop_bad=args.drop_bad_stands,
        require_positions=require_positions,
        operational_only=getattr(args, "operational_only", False),
    )
    rdc2_chunks = []
    for _, _, rdc2_chunk in iter_rdc2_view_chunks(
        loaded,
        args.channel_chunk_size,
        channel_indices=selected_channel_indices,
        frame_start=frame_idx,
        frame_stop=frame_idx + 1,
        delay_selection=delay_selection,
        doppler_selection=doppler_selection,
    ):
        rdc2_chunks.append(rdc2_chunk[0])
    if not rdc2_chunks:
        raise ValueError("No RDC2 channels were loaded")
    rdc2 = np.concatenate(rdc2_chunks, axis=2)
    nonfinite_values_filled = 0
    if args.gap_fill == "zero":
        rdc2, nonfinite_values_filled = zero_fill_nonfinite_complex(rdc2)
    frequency = args.carrier_frequency or metadata.get("tuning_frequency_hz") or metadata.get("carrier_frequency_hz") or metadata.get("freq1_hz")
    factors = array_cable_factors(args, metadata, channel_stands, channel_pols, frequency)
    rdc2 *= factors.astype(rdc2.dtype)
    return {
        "rdc2": rdc2,
        "nonfinite_values_filled": nonfinite_values_filled,
        "delay_ms": delay_ms,
        "fd_hz": fd_hz,
        "metadata": metadata,
        "shape": shape,
        "frame_tstart_s": loaded["frame_tstart_s"],
        "frame_tend_s": loaded["frame_tend_s"],
        "selected_channel_indices": selected_channel_indices,
        "channel_stands": channel_stands,
        "channel_pols": channel_pols,
        "positions": positions,
        "missing_position_stands": missing_position_stands,
    }


def angle_snapshot_block(
    rdc2,
    delay_idx,
    doppler_idx,
    delay_radius_bins=1,
    doppler_radius_bins=1,
    min_snapshots=None,
):
    delay_radius_bins = max(0, int(delay_radius_bins))
    doppler_radius_bins = max(0, int(doppler_radius_bins))
    min_snapshots = 0 if min_snapshots is None else max(0, int(min_snapshots))
    delay_idx = int(delay_idx)
    doppler_idx = int(doppler_idx)
    max_delay_radius = max(delay_idx, rdc2.shape[0] - delay_idx - 1)
    max_doppler_radius = max(doppler_idx, rdc2.shape[1] - doppler_idx - 1)

    while True:
        r0 = max(0, delay_idx - delay_radius_bins)
        r1 = min(rdc2.shape[0], delay_idx + delay_radius_bins + 1)
        d0 = max(0, doppler_idx - doppler_radius_bins)
        d1 = min(rdc2.shape[1], doppler_idx + doppler_radius_bins + 1)
        n_snapshots = (r1 - r0) * (d1 - d0)
        if n_snapshots >= min_snapshots:
            break
        if doppler_radius_bins < max_doppler_radius:
            doppler_radius_bins += 1
        elif delay_radius_bins < max_delay_radius:
            delay_radius_bins += 1
        else:
            break

    block = rdc2[r0:r1, d0:d1, :].reshape(-1, rdc2.shape[2])
    return block, {
        "snapshots": int(block.shape[0]),
        "delay_bins": int(r1 - r0),
        "doppler_bins": int(d1 - d0),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Array delay-Doppler processing from TBN/RDC0 through RDC1/RDC2 and optional MVDR angle spectra.",
        formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, max_help_position=35),
    )
    parser.add_argument("input_file", nargs="?", help="Input TBN file; omit when using --rdc0-input")
    parser.add_argument("--rdc0-input", default=None, help="Existing RDC0 HDF5 file to process")
    parser.add_argument("--rdc2-input", default=None, help="RDC2 cache; timing overrides recompute Doppler from retained RDC1")
    parser.add_argument("--rdc2-frame-index", type=int, default=0,
                        help="RDC2 frame index to plot when --rdc2-input contains multiple frames")
    parser.add_argument("--rdc0-cache", default=None, help="RDC0 HDF5 cache path to create/reuse for TBN input")
    parser.add_argument("--keep-temp-rdc0", action="store_true", help="Keep temporary RDC0 cache if --rdc0-cache is omitted")

    parser.add_argument("--output", required=True, help="Output delay-Doppler PNG")
    parser.add_argument("--title", default="Array Delay-Doppler Map")
    parser.add_argument("--tstart", type=float, default=None)
    parser.add_argument("--tend", type=float, default=None)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--integration-time", type=float, default=None, help="RDC2 cache integration; requires retained RDC1 to change")
    parser.add_argument("--hop-time", type=float, default=None, help="RDC2 cache hop; choose a window with --rdc2-frame-index")
    parser.add_argument("--sweep-frequency", type=float, default=None)
    parser.add_argument("--bandwidth", type=float, default=None)
    parser.add_argument("--offset", type=float, default=0.0)
    parser.add_argument("--no-timestamp-align", action="store_true")

    parser.add_argument("--stands", default="all")
    parser.add_argument("--max-stand", type=int, default=256)
    parser.add_argument("-p", "--pol", choices=["x", "y", "0", "1"], default="x",
                        help="Array processing should use one receiver polarization at a time")
    parser.add_argument("--drop-bad-stands", action="store_true", default=True,
                        help=f"Drop known bad stands before array processing: {BAD_LWA_STANDS}")
    parser.add_argument("--keep-bad-stands", action="store_false", dest="drop_bad_stands",
                        help="Do not drop known bad stands")

    parser.add_argument("--adc-start", type=int, default=0)
    parser.add_argument("--adc-count", type=int, default=None)
    parser.add_argument("--gap-fill", choices=["nan", "zero"], default="nan")
    parser.add_argument("--rdc0-dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--rdc0-compression", choices=["gzip", "lzf", "none"], default="gzip")
    parser.add_argument("--rdc0-compression-level", type=int, default=4)
    parser.add_argument("--channel-chunk-size", type=int, default=8,
                        help="Number of RDC0 channels to process at once")

    parser.add_argument("--dechirp-window", choices=["hamming", "hann", "blackman", "cheb60", "cheb80", "cheb100", "cheb120", "none"], default="hann")
    parser.add_argument("--slow-window", choices=["hann", "hamming", "blackman", "cheb60", "cheb80", "cheb100", "cheb120", "none"], default="hann")
    parser.add_argument("--nfft-doppler", type=int, default=None)
    parser.add_argument("--beamformer", choices=["mvdr", "conventional", "incoherent"], default="conventional",
                        help="Spatial processing before plotting; incoherent reproduces power-only channel combining")
    parser.add_argument("--channel-combine", choices=["sum", "mean", "max"], default="sum",
                        help="Only used with --beamformer incoherent")
    parser.add_argument("--look-az", type=float, default=None,
                        help="Beam look azimuth in degrees; omit with --look-el to auto-scan")
    parser.add_argument("--look-el", type=float, default=None,
                        help="Beam look elevation in degrees; omit with --look-az to auto-scan")
    parser.add_argument("--d-min", type=float, default=None)
    parser.add_argument("--d-max", type=float, default=None)
    parser.add_argument("--fd-min", type=float, default=None)
    parser.add_argument("--fd-max", type=float, default=None)
    parser.add_argument("--vmin", type=float, default=None)
    parser.add_argument("--vmax", type=float, default=None)

    parser.add_argument("--angle-output", default=None, help="Output HDF5 for selected-cell MVDR spectra")
    parser.add_argument("--angle-plot-output", default=None, help="Output PNG for selected-cell MVDR azimuth/elevation spectra")
    parser.add_argument("--beam-pattern-output", default=None,
                        help="Output PNG showing the conventional array point-spread/beam pattern for the resolved look direction")
    parser.add_argument("--angle-plot-vmin", type=float, default=-40.0)
    parser.add_argument("--angle-plot-vmax", type=float, default=0.0)
    parser.add_argument("--beam-pattern-vmin", type=float, default=-40.0,
                        help="Beam-pattern color minimum in dB relative to its peak")
    parser.add_argument("--beam-pattern-vmax", type=float, default=0.0,
                        help="Beam-pattern color maximum in dB relative to its peak")
    parser.add_argument("--cable-correction", choices=["auto", "on", "off"], default="auto",
                        help="Correct LSL LWA-SV cable phase at RF center; auto skips ideal simulations and already-corrected input")
    parser.add_argument("--operational-only", action="store_true",
                        help="Additionally require LSL combined_status=33 (installed station status, not observation-date status)")
    parser.add_argument("--angle-source", choices=["window", "cells"], default="window",
                        help="Use the full plotted delay-Doppler window or selected cells for angle spectra")
    parser.add_argument("--angle-top-n", type=int, default=0,
                        help="Compute MVDR spectra for the top N delay-Doppler cells")
    parser.add_argument("--angle-delay-ms", type=float, default=None,
                        help="Specific delay cell for MVDR")
    parser.add_argument("--angle-doppler-hz", type=float, default=None,
                        help="Specific Doppler cell for MVDR")
    parser.add_argument("--angle-delay-radius-bins", type=int, default=1,
                        help="Delay-bin half-width used as snapshots for each angle spectrum")
    parser.add_argument("--angle-doppler-radius-bins", type=int, default=1,
                        help="Doppler-bin half-width used as snapshots for each angle spectrum")
    parser.add_argument("--angle-min-snapshots", type=int, default=256,
                        help="Automatically expand the angle-spectrum snapshot window until at least this many snapshots are used")
    parser.add_argument("--angle-method", choices=["mvdr", "conventional"], default="conventional",
                        help="Spatial spectrum used for angle output/plots")
    parser.add_argument("--angle-flat-fallback", choices=["conventional", "none"], default="conventional",
                        help="When --angle-method=mvdr gives a nearly flat map, plot this fallback instead")
    parser.add_argument("--angle-flat-threshold-db", type=float, default=0.25,
                        help="MVDR angle-map span below this threshold is treated as flat")
    parser.add_argument("--az-min", type=float, default=-180.0)
    parser.add_argument("--az-max", type=float, default=180.0)
    parser.add_argument("--az-step", type=float, default=2.0)
    parser.add_argument("--el-min", type=float, default=0.0)
    parser.add_argument("--el-max", type=float, default=90.0)
    parser.add_argument("--el-step", type=float, default=2.0)
    parser.add_argument("--diagonal-loading", type=float, default=0.1)
    parser.add_argument("--mvdr-covariance", choices=["background", "all"], default="background",
                        help="MVDR covariance source; background excludes the strongest delay-Doppler cells")
    parser.add_argument("--mvdr-background-percentile", type=float, default=80.0,
                        help="Use cells at or below this power percentile for background MVDR covariance")
    parser.add_argument("--carrier-frequency", type=float, default=None,
                        help="Carrier/tuning frequency for beamforming; defaults to TBN freq1 when reading TBN")

    parser.add_argument("--reference-gate-frequency", type=float, default=None)
    parser.add_argument("--reference-gate-period", type=float, default=None)
    parser.add_argument("--reference-gate-duty", type=float, default=0.5)
    parser.add_argument("--reference-gate-phase", type=float, default=0.0)

    args = parser.parse_args()
    if args.rdc2_input is None and args.rdc0_input is None and args.input_file is None:
        raise ValueError("Provide a TBN input_file, --rdc0-input, or --rdc2-input")
    if args.rdc2_input is not None and (args.rdc0_input is not None or args.input_file is not None):
        raise ValueError("Use --rdc2-input by itself, not together with TBN input_file or --rdc0-input")
    if (args.d_min is None) != (args.d_max is None):
        raise ValueError("Provide both --d-min and --d-max, or omit both for auto delay selection")
    if (args.angle_delay_ms is None) != (args.angle_doppler_hz is None):
        raise ValueError("Provide both --angle-delay-ms and --angle-doppler-hz")
    angle_source = args.angle_source
    if args.angle_delay_ms is not None or args.angle_top_n > 0:
        angle_source = "cells"
    if (args.look_az is None) != (args.look_el is None):
        raise ValueError("Provide both --look-az and --look-el, or omit both for auto look-direction scanning")
    if args.beam_pattern_output and args.beamformer == "incoherent" and args.look_az is None:
        raise ValueError("--beam-pattern-output with --beamformer=incoherent requires --look-az and --look-el")

    fs = None
    fc = args.carrier_frequency
    start_offset_samples = 0
    corner_note = None
    if args.rdc2_input:
        wants_angle_product = args.angle_output or args.angle_plot_output
        wants_pattern_product = args.beam_pattern_output is not None
        loaded_rdc2 = load_rdc2_frame_for_array(
            args,
            frame_idx=args.rdc2_frame_index,
            require_positions=args.beamformer in ("mvdr", "conventional") or wants_angle_product or wants_pattern_product,
        )
        metadata = loaded_rdc2["metadata"]
        fs = float(metadata.get("sample_rate_hz"))
        if args.sweep_frequency is None or args.bandwidth is None:
            raise ValueError("RDC2 metadata is missing sweep_frequency_hz or bandwidth_hz")
        if fc is None:
            fc = metadata.get("tuning_frequency_hz", metadata.get("carrier_frequency_hz", metadata.get("freq1_hz")))
        if fc is not None:
            fc = float(fc)
        if (wants_angle_product or wants_pattern_product or args.beamformer in ("mvdr", "conventional")) and fc is None:
            raise ValueError("--carrier-frequency is required for beamforming/angle output when RDC2 metadata does not provide frequency")

        rdc2_plot = loaded_rdc2["rdc2"]
        delay_ms = loaded_rdc2["delay_ms"]
        fd_plot = loaded_rdc2["fd_hz"]
        selected_channel_indices = loaded_rdc2["selected_channel_indices"]
        channel_stands = loaded_rdc2["channel_stands"]
        channel_pols = loaded_rdc2["channel_pols"]
        positions = loaded_rdc2["positions"]
        missing_position_stands = loaded_rdc2["missing_position_stands"]
        nonfinite_rdc2_values_filled = loaded_rdc2["nonfinite_values_filled"]
        frame_tstart = float(loaded_rdc2["frame_tstart_s"][args.rdc2_frame_index])
        frame_tend = float(loaded_rdc2["frame_tend_s"][args.rdc2_frame_index])
        if missing_position_stands:
            print(
                "Dropping selected channel(s) without LSL LWA-SV positions before array processing: "
                f"{missing_position_stands}",
                flush=True,
            )
        power_plot, look_az, look_el, look_auto, covariance_note, frame_warning = array_delay_doppler_power_from_rdc2(
            rdc2_plot,
            args,
            positions,
            fc,
            blank_on_insufficient=True,
        )
        if power_plot is None:
            power_plot = np.full((len(delay_ms), len(fd_plot)), np.nan, dtype=float)
        if args.beamformer in ("mvdr", "conventional") and look_az is not None and look_el is not None:
            print(f"Selected/using look direction: az={look_az:.2f} deg, el={look_el:.2f} deg", flush=True)

        lfm_config = LFMWaveform(
            sample_rate=fs,
            sweep_frequency=args.sweep_frequency,
            bandwidth=args.bandwidth,
            reference_gate_frequency=metadata.get("reference_gate_frequency"),
            reference_gate_duty=metadata.get("reference_gate_duty", args.reference_gate_duty),
            reference_gate_phase=metadata.get("reference_gate_phase", args.reference_gate_phase),
        )
        info_rows = [
            ("File", os.path.basename(args.rdc2_input)),
            ("RDC2 frame", f"{args.rdc2_frame_index}/{loaded_rdc2['shape'][0] - 1}"),
        ]
        if fc is not None:
            info_rows.append(("Tuning freq", f"{fc / 1e6:.6f} MHz"))
        info_rows.extend([
            ("Array channels", f"{len(selected_channel_indices)} {args.pol}-pol"),
            ("Stands used", str(len(np.unique(channel_stands)))),
            ("Beamformer", args.beamformer.upper() if args.beamformer != "incoherent" else f"Incoherent {args.channel_combine}"),
        ])
        if args.drop_bad_stands:
            info_rows.append(("Bad stands", "Dropped"))
        if missing_position_stands:
            info_rows.append(("Missing pos", f"Dropped {len(missing_position_stands)}"))
        if args.beamformer in ("mvdr", "conventional"):
            look_label = "auto" if look_auto else "fixed"
            info_rows.append(("Look dir", f"{look_label}: az {look_az:.2f} deg, el {look_el:.2f} deg" if look_az is not None else "unavailable"))
        if args.beamformer == "mvdr":
            info_rows.append(("MVDR cov", covariance_note))
            info_rows.append(("Diag loading", f"{args.diagonal_loading:g}"))
        info_rows.extend([
            ("Plot span", f"{frame_tstart:.3f}-{frame_tend:.3f} s"),
            ("Integration", f"{frame_tend - frame_tstart:.3f} s"),
            ("Doppler FFT", str(len(fd_plot))),
            ("Delay limits", f"{args.d_min} to {args.d_max} ms"),
            ("RDC2 cache", "yes"),
        ])
        info_rows.extend(_lfm_info_rows(lfm_config))
        info_rows.append(("Cable phase", getattr(args, "_cable_note", "unknown")))
        footer_note = "\n".join(note for note in [
            gap_fill_note(metadata.get("gap_info")),
            f"RDC2 nonfinite values replaced with zeros for this plot: {nonfinite_rdc2_values_filled}"
            if nonfinite_rdc2_values_filled else None,
            frame_warning,
        ] if note)
        lower_note = _format_delay_peaks_note(power_plot.T, delay_ms, n=5)
        has_info = bool(info_rows or corner_note or lower_note or footer_note)
        fig, ax = _new_figure_with_info(figsize=(14, 6)) if has_info else plt.subplots(figsize=(10, 6))
        pcm = ax.pcolormesh(fd_plot, delay_ms, power_plot, shading="nearest", cmap="inferno", vmin=args.vmin, vmax=args.vmax)
        ax.set_xlabel("Doppler Frequency [Hz]")
        ax.set_ylabel("Delay [ms]")
        ax.set_title(args.title)
        _add_delay_range_axis(ax)
        _add_colorbar(fig, pcm, ax, "Power [dB]")
        _add_info_panel(fig, info_rows=info_rows, corner_note=corner_note, lower_note=lower_note, footer_note=footer_note)
        _tight_layout_with_info(fig, has_info=has_info)
        _savefig(args.output, dpi=300)
        plt.close(fig)
        print(f"Wrote {args.output}", flush=True)
        if args.beam_pattern_output:
            az_grid = np.arange(args.az_min, args.az_max + 0.5 * args.az_step, args.az_step)
            el_grid = np.arange(args.el_min, args.el_max + 0.5 * args.el_step, args.el_step)
            write_beam_pattern_plot(
                args.beam_pattern_output,
                positions,
                fc,
                look_az,
                look_el,
                az_grid,
                el_grid,
                info_rows=[
                    ("File", os.path.basename(args.rdc2_input)),
                    ("Tuning freq", f"{fc / 1e6:.6f} MHz"),
                    ("Array channels", f"{len(selected_channel_indices)} {args.pol}-pol"),
                    ("Stands used", str(len(np.unique(channel_stands)))),
                ],
                corner_note=corner_note,
                footer_note=footer_note,
                vmin=args.beam_pattern_vmin,
                vmax=args.beam_pattern_vmax,
            )
            print(f"Wrote {args.beam_pattern_output}", flush=True)
        return

    if args.sweep_frequency is None or args.bandwidth is None:
        raise ValueError("--sweep-frequency and --bandwidth are required unless --rdc2-input metadata provides them")

    if args.rdc0_input is None:
        idf = lsl_open_tbn(args.input_file)
        lsl_print_metadata(idf)
        fs = float(idf.get_info("sample_rate"))
        if fc is None:
            fc = float(idf.get_info("freq1"))
        set_default_tbn_time_bounds(args, idf)
        if args.duration is not None:
            args.tend = args.tstart + args.duration
        corner_note = file_timestamp_range_note(idf, args.tstart, args.tend)
        if not args.no_timestamp_align:
            probe_ts = probe_start_timestamp(args.input_file, args.tstart)
            lfm_for_offset = LFMWaveform(sample_rate=fs, sweep_frequency=args.sweep_frequency, bandwidth=args.bandwidth)
            start_offset_samples = _timestamp_sweep_offset_samples(probe_ts, lfm_for_offset, sweep_offset=args.offset)
            print(f"Timestamp alignment start_offset_samples={start_offset_samples}", flush=True)
    elif args.duration is not None or args.tstart is not None or args.tend is not None:
        print("Ignoring --tstart/--tend/--duration because --rdc0-input is already extracted.", flush=True)

    cache_path, delete_cache = maybe_make_rdc0_cache(args, fs, start_offset_samples)
    loaded = load_rdc0_hdf5_metadata(cache_path)
    rdc0_shape = loaded["shape"]
    channel_stands = loaded["channel_stands"]
    channel_pols = loaded["channel_pols"]
    metadata = loaded["metadata"]
    fs = float(metadata.get("sample_rate_hz", fs))
    adc_start = int(metadata.get("adc_start", args.adc_start))
    if fc is None:
        fc = metadata.get("tuning_frequency_hz", metadata.get("carrier_frequency_hz", metadata.get("freq1_hz")))
    if fc is not None:
        fc = float(fc)
    wants_angle_product = args.angle_output or args.angle_plot_output
    wants_pattern_product = args.beam_pattern_output is not None
    if (wants_angle_product or wants_pattern_product or args.beamformer in ("mvdr", "conventional")) and fc is None:
        raise ValueError("--carrier-frequency is required for beamforming/MVDR angle output when RDC0 metadata does not provide TBN freq1")

    if args.d_min is None and args.d_max is None:
        raise ValueError(
            "Automatic delay selection is not enabled for array RDC0 streaming yet. "
            "Please provide --d-min and --d-max for this script."
        )

    selected_channel_indices, channel_stands, channel_pols, positions, missing_position_stands = select_array_channels(
        channel_stands,
        channel_pols,
        args.pol,
        drop_bad=args.drop_bad_stands,
        require_positions=args.beamformer in ("mvdr", "conventional") or wants_angle_product or wants_pattern_product,
        operational_only=args.operational_only,
    )
    if missing_position_stands:
        print(
            "Dropping selected channel(s) without LSL LWA-SV positions before array processing: "
            f"{missing_position_stands}",
            flush=True,
        )

    print(f"Loaded RDC0 shape (ADC, pulses, channels): {rdc0_shape}", flush=True)
    print(f"Selected {len(selected_channel_indices)} channel(s) for array processing.", flush=True)
    print(f"Streaming RDC1/RDC2 in channel chunks of {args.channel_chunk_size}...", flush=True)
    lfm_config = LFMWaveform(
        sample_rate=fs,
        sweep_frequency=args.sweep_frequency,
        bandwidth=args.bandwidth,
        reference_gate_frequency=reference_gate_frequency_from_args(args),
        reference_gate_duty=args.reference_gate_duty,
        reference_gate_phase=args.reference_gate_phase,
    )
    delay_ms = None
    fd_hz = None
    fd_mask = None
    fd_plot = None
    power_linear = None
    total_channels = 0
    nonfinite_rdc0_values_filled = 0
    nonfinite_rdc2_values_filled = 0
    need_channel_cube = args.beamformer in ("mvdr", "conventional") or wants_angle_product
    rdc2_chunks = [] if need_channel_cube else None
    look_auto = args.beamformer in ("mvdr", "conventional") and args.look_az is None and args.look_el is None

    cable_factors = array_cable_factors(args, metadata, channel_stands, channel_pols, fc)
    for chunk_start, chunk_stop, rdc0_chunk in iter_rdc0_hdf5_channel_chunks(
        cache_path,
        args.channel_chunk_size,
        channel_indices=selected_channel_indices,
    ):
        print(f"Processing selected channels {chunk_start + 1}-{chunk_stop}/{len(selected_channel_indices)}", flush=True)
        if args.gap_fill == "zero":
            finite = np.isfinite(rdc0_chunk.real) & np.isfinite(rdc0_chunk.imag)
            bad_count = int(rdc0_chunk.size - np.count_nonzero(finite))
            if bad_count:
                nonfinite_rdc0_values_filled += bad_count
                rdc0_chunk = np.nan_to_num(rdc0_chunk, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        rdc1_band, chunk_delay_ms = dechirp_range_fft_delay_band_rdc0(
            rdc0_chunk,
            lfm_config,
            args.d_min,
            args.d_max,
            window=None if args.dechirp_window == "none" else args.dechirp_window,
            adc_start=adc_start,
        )
        rdc2_chunk, chunk_fd_hz = doppler_fft_rdc1(
            rdc1_band,
            prf=args.sweep_frequency,
            window=None if args.slow_window == "none" else args.slow_window,
            nfft=args.nfft_doppler,
        )
        if delay_ms is None:
            delay_ms = chunk_delay_ms
            fd_hz = chunk_fd_hz
            fd_mask = np.ones(fd_hz.shape, dtype=bool)
            if args.fd_min is not None:
                fd_mask &= fd_hz >= args.fd_min
            if args.fd_max is not None:
                fd_mask &= fd_hz <= args.fd_max
            fd_plot = fd_hz[fd_mask]
            if fd_plot.size == 0:
                raise ValueError("No Doppler bins selected; check --fd-min/--fd-max")
        else:
            if not np.array_equal(delay_ms, chunk_delay_ms):
                raise RuntimeError("Delay axis changed unexpectedly between RDC0 chunks")
            if not np.array_equal(fd_hz, chunk_fd_hz):
                raise RuntimeError("Doppler axis changed unexpectedly between RDC0 chunks")

        rdc2_chunk = rdc2_chunk[:, fd_mask, :]
        rdc2_chunk *= cable_factors[chunk_start:chunk_stop].astype(rdc2_chunk.dtype)
        if args.gap_fill == "zero":
            rdc2_chunk, bad_count = zero_fill_nonfinite_complex(rdc2_chunk)
            nonfinite_rdc2_values_filled += bad_count
        if args.beamformer == "incoherent":
            chunk_power = np.abs(rdc2_chunk) ** 2
            if args.channel_combine in ("sum", "mean"):
                chunk_combined = np.nansum(chunk_power, axis=2)
                if power_linear is None:
                    power_linear = chunk_combined.astype(np.float64, copy=False)
                else:
                    power_linear += chunk_combined
            elif args.channel_combine == "max":
                chunk_combined = np.nanmax(chunk_power, axis=2)
                if power_linear is None:
                    power_linear = chunk_combined.astype(np.float64, copy=False)
                else:
                    power_linear = np.fmax(power_linear, chunk_combined)
            else:
                raise ValueError("--channel-combine must be one of: sum, mean, max")
            del chunk_power, chunk_combined
        total_channels += rdc2_chunk.shape[2]

        if rdc2_chunks is not None:
            rdc2_chunks.append(rdc2_chunk)

        del rdc0_chunk, rdc1_band, rdc2_chunk

    if args.beamformer == "incoherent":
        if power_linear is None:
            raise ValueError("No RDC0 channels were processed")
        if args.channel_combine == "mean":
            power_linear = power_linear / max(total_channels, 1)
        power_plot = 10.0 * np.log10(power_linear + 1e-12)
        rdc2_plot = np.concatenate(rdc2_chunks, axis=2) if rdc2_chunks is not None else None
    else:
        if not rdc2_chunks:
            raise ValueError("No RDC2 chunks were created for beamforming")
        rdc2_plot = np.concatenate(rdc2_chunks, axis=2)
        signal_covariance = estimate_spatial_covariance(rdc2_plot, diagonal_loading=args.diagonal_loading)
        covariance = signal_covariance
        covariance_note = "all cells"
        if args.beamformer == "mvdr" and args.mvdr_covariance == "background":
            covariance = estimate_spatial_covariance(
                rdc2_plot,
                diagonal_loading=args.diagonal_loading,
                max_power_percentile=args.mvdr_background_percentile,
            )
            covariance_note = f"background <= {args.mvdr_background_percentile:g} pct"
        if look_auto:
            az_grid = np.arange(args.az_min, args.az_max + 0.5 * args.az_step, args.az_step)
            el_grid = np.arange(args.el_min, args.el_max + 0.5 * args.el_step, args.el_step)
            scan_method = "conventional" if args.beamformer == "mvdr" else args.beamformer
            print(
                f"Auto-scanning look direction with {scan_method.upper()} power over "
                f"{len(az_grid)} azimuth x {len(el_grid)} elevation points...",
                flush=True,
            )
            _, args.look_az, args.look_el = scan_look_direction(
                positions,
                fc,
                signal_covariance,
                az_grid,
                el_grid,
                method=scan_method,
            )
            print(f"Selected look direction: az={args.look_az:.2f} deg, el={args.look_el:.2f} deg", flush=True)
        print(
            f"Combining {len(selected_channel_indices)} channels with {args.beamformer.upper()} "
            f"beamforming at az={args.look_az:.2f} deg, el={args.look_el:.2f} deg...",
            flush=True,
        )
        beamformed = beamform_rdc2(
            rdc2_plot,
            positions,
            fc,
            args.look_az,
            args.look_el,
            method=args.beamformer,
            covariance=covariance,
        )
        power_plot = 10.0 * np.log10(np.abs(beamformed) ** 2 + 1e-12)
        del beamformed, signal_covariance

    info_rows = [
        ("File", os.path.basename(args.rdc0_input or args.input_file or cache_path)),
    ]
    if fc is not None:
        info_rows.append(("Tuning freq", f"{fc / 1e6:.6f} MHz"))
    info_rows.extend([
        ("Array channels", f"{len(selected_channel_indices)} {args.pol}-pol"),
        ("Stands used", str(len(np.unique(channel_stands)))),
        ("Beamformer", args.beamformer.upper() if args.beamformer != "incoherent" else f"Incoherent {args.channel_combine}"),
    ])
    if args.drop_bad_stands:
        info_rows.append(("Bad stands", "Dropped"))
    if missing_position_stands:
        info_rows.append(("Missing pos", f"Dropped {len(missing_position_stands)}"))
    if args.beamformer in ("mvdr", "conventional"):
        look_label = "auto" if look_auto else "fixed"
        info_rows.append(("Look dir", f"{look_label}: az {args.look_az:.2f} deg, el {args.look_el:.2f} deg"))
    if args.beamformer == "mvdr":
        info_rows.append(("MVDR cov", covariance_note))
        info_rows.append(("Diag loading", f"{args.diagonal_loading:g}"))
    info_rows.extend([
        ("Plot span", format_time_span(metadata, rdc0_shape[1], args.sweep_frequency)),
        ("Integration", f"{rdc0_shape[1] / float(args.sweep_frequency):.3f} s"),
        ("Doppler FFT", str(len(fd_hz))),
        ("Delay limits", f"{args.d_min} to {args.d_max} ms"),
    ])
    if args.fd_min is not None or args.fd_max is not None:
        info_rows.append(("Doppler limits", f"{args.fd_min} to {args.fd_max} Hz"))
    info_rows.extend(_lfm_info_rows(lfm_config))
    info_rows.append(("Cable phase", getattr(args, "_cable_note", "unknown")))
    footer_note = "\n".join(note for note in [
        gap_fill_note(metadata.get("gap_info")),
        f"RDC0 nonfinite values replaced with zeros for this plot: {nonfinite_rdc0_values_filled}"
        if nonfinite_rdc0_values_filled else None,
        f"RDC2 nonfinite values replaced with zeros for this plot: {nonfinite_rdc2_values_filled}"
        if nonfinite_rdc2_values_filled else None,
    ] if note)
    lower_note = _format_delay_peaks_note(power_plot.T, delay_ms, n=5)
    has_info = bool(info_rows or corner_note or lower_note or footer_note)

    fig, ax = _new_figure_with_info(figsize=(14, 6)) if has_info else plt.subplots(figsize=(10, 6))
    pcm = ax.pcolormesh(fd_plot, delay_ms, power_plot, shading="nearest", cmap="inferno", vmin=args.vmin, vmax=args.vmax)
    ax.set_xlabel("Doppler Frequency [Hz]")
    ax.set_ylabel("Delay [ms]")
    ax.set_title(args.title)
    _add_delay_range_axis(ax)
    _add_colorbar(fig, pcm, ax, "Power [dB]")
    _add_info_panel(
        fig,
        info_rows=info_rows,
        corner_note=corner_note,
        lower_note=lower_note,
        footer_note=footer_note,
    )
    _tight_layout_with_info(fig, has_info=has_info)
    _savefig(args.output, dpi=300)
    plt.close(fig)
    print(f"Wrote {args.output}", flush=True)

    if args.angle_output or args.angle_plot_output:
        print("Using LWA-SV stand positions from lsl.common.stations.lwasv.", flush=True)
        wavelength = 299_792_458.0 / float(fc)
        az_grid = np.arange(args.az_min, args.az_max + 0.5 * args.az_step, args.az_step)
        el_grid = np.arange(args.el_min, args.el_max + 0.5 * args.el_step, args.el_step)

        spectra = []
        delay_cells = []
        doppler_cells = []
        cell_snapshot_info = []
        cell_labels = []
        angle_warnings = []
        if angle_source == "window":
            print(
                f"Computing {args.angle_method.upper()} angle spectrum over the full plotted "
                "delay-Doppler window",
                flush=True,
            )
            window_covariance = estimate_spatial_covariance(
                rdc2_plot,
                diagonal_loading=args.diagonal_loading,
            )
            spectrum = spatial_spectrum_from_covariance(window_covariance, positions, fc, az_grid, el_grid, method=args.angle_method)
            if args.angle_method == "mvdr":
                fallback_spectrum = spatial_spectrum_from_covariance(window_covariance, positions, fc, az_grid, el_grid, method="conventional")
                spectrum, fallback_note = fallback_if_angle_spectrum_flat(
                    spectrum,
                    fallback_spectrum,
                    args.angle_method,
                    fallback_mode=args.angle_flat_fallback,
                    threshold_db=args.angle_flat_threshold_db,
                )
                if fallback_note:
                    print(fallback_note, flush=True)
                    angle_warnings.append(fallback_note)
            spectra.append(spectrum)
            delay_cells.append(float(np.nanmean(delay_ms)))
            doppler_cells.append(float(np.nanmean(fd_plot)))
            cell_snapshot_info.append({
                "snapshots": int(np.prod(rdc2_plot.shape[:2])),
                "delay_bins": int(rdc2_plot.shape[0]),
                "doppler_bins": int(rdc2_plot.shape[1]),
            })
            cell_labels.append(
                f"full DD window, delay {np.nanmin(delay_ms):.3f}-{np.nanmax(delay_ms):.3f} ms, "
                f"Doppler {np.nanmin(fd_plot):.3f}-{np.nanmax(fd_plot):.3f} Hz"
            )
        else:
            cell_indices = []
            if args.angle_delay_ms is not None:
                ri = int(np.nanargmin(np.abs(delay_ms - args.angle_delay_ms)))
                di = int(np.nanargmin(np.abs(fd_plot - args.angle_doppler_hz)))
                cell_indices.append((ri, di))
            if args.angle_top_n > 0:
                work = power_plot.copy()
                for _ in range(args.angle_top_n):
                    if not np.any(np.isfinite(work)):
                        break
                    ri, di = np.unravel_index(np.nanargmax(work), work.shape)
                    cell_indices.append((int(ri), int(di)))
                    work[ri, di] = np.nan
            if not cell_indices:
                raise ValueError("Use --angle-top-n or --angle-delay-ms/--angle-doppler-hz with --angle-source=cells")

            for ri, di in cell_indices:
                snapshot_block, snapshot_info = angle_snapshot_block(
                    rdc2_plot,
                    ri,
                    di,
                    delay_radius_bins=args.angle_delay_radius_bins,
                    doppler_radius_bins=args.angle_doppler_radius_bins,
                    min_snapshots=args.angle_min_snapshots,
                )
                print(
                    f"Computing {args.angle_method.upper()} angle spectrum at delay={delay_ms[ri]:.3f} ms, "
                    f"fd={fd_plot[di]:.3f} Hz using {snapshot_info['snapshots']} snapshots",
                    flush=True,
                )
                spectrum = capon_mvdr_spectrum(
                    snapshot_block,
                    positions,
                    wavelength,
                    az_grid,
                    el_grid,
                    diagonal_loading=args.diagonal_loading,
                    method=args.angle_method,
                )
                if args.angle_method == "mvdr":
                    fallback_spectrum = capon_mvdr_spectrum(
                        snapshot_block,
                        positions,
                        wavelength,
                        az_grid,
                        el_grid,
                        diagonal_loading=args.diagonal_loading,
                        method="conventional",
                    )
                    spectrum, fallback_note = fallback_if_angle_spectrum_flat(
                        spectrum,
                        fallback_spectrum,
                        args.angle_method,
                        fallback_mode=args.angle_flat_fallback,
                        threshold_db=args.angle_flat_threshold_db,
                    )
                    if fallback_note:
                        print(fallback_note, flush=True)
                        angle_warnings.append(fallback_note)
                spectra.append(spectrum)
                delay_cells.append(delay_ms[ri])
                doppler_cells.append(fd_plot[di])
                cell_snapshot_info.append(snapshot_info)
        spectra = np.asarray(spectra)
        delay_cells = np.asarray(delay_cells)
        doppler_cells = np.asarray(doppler_cells)
        angle_metadata = {
            "source_file": os.path.basename(args.rdc0_input or args.input_file or cache_path),
            "beamformer": f"{args.angle_method}_angle_spectrum",
            "angle_method": args.angle_method,
            "cable_phase_correction": getattr(args, "_cable_note", "unknown"),
            "operational_only": args.operational_only,
            "angle_source": angle_source,
            "carrier_frequency_hz": fc,
            "selected_stands": [int(s) for s in channel_stands],
            "n_channels": int(len(channel_stands)),
            "n_stands": int(len(np.unique(channel_stands))),
            "diagonal_loading": float(args.diagonal_loading),
            "angle_delay_radius_bins": int(args.angle_delay_radius_bins),
            "angle_doppler_radius_bins": int(args.angle_doppler_radius_bins),
            "angle_min_snapshots": int(args.angle_min_snapshots),
            "angle_snapshot_info": cell_snapshot_info,
            "delay_cells_ms": delay_cells.tolist(),
            "doppler_cells_hz": doppler_cells.tolist(),
            **polarization_metadata(channel_pols),
        }
        if args.angle_output:
            write_angle_output(args.angle_output, spectra, az_grid, el_grid, delay_cells, doppler_cells, metadata=angle_metadata)
            print(f"Wrote {args.angle_output}", flush=True)
        if args.angle_plot_output:
            plot_angle_spectra(
                args.angle_plot_output,
                spectra,
                az_grid,
                el_grid,
                delay_cells,
                doppler_cells,
                vmin=args.angle_plot_vmin,
                vmax=args.angle_plot_vmax,
                title="Capon/MVDR Angle Spectrum",
                info_rows=[
                    ("File", os.path.basename(args.rdc0_input or args.input_file or cache_path)),
                    ("Tuning freq", f"{fc / 1e6:.6f} MHz"),
                    ("Array channels", f"{len(selected_channel_indices)} {args.pol}-pol"),
                    ("Stands used", str(len(np.unique(channel_stands)))),
                    ("Angle method", args.angle_method.upper()),
                    ("Cable phase", getattr(args, "_cable_note", "unknown")),
                    ("Angle source", angle_source),
                    ("Min snapshots", str(args.angle_min_snapshots)),
                    ("Diag loading", f"{args.diagonal_loading:g}"),
                    ("Plot span", format_time_span(metadata, rdc0_shape[1], args.sweep_frequency)),
                ],
                corner_note=corner_note,
                footer_note="\n".join(note for note in [footer_note, *angle_warnings] if note),
                cell_snapshot_info=cell_snapshot_info,
                cell_labels=cell_labels,
            )
            print(f"Wrote {args.angle_plot_output}", flush=True)

    if args.beam_pattern_output:
        az_grid = np.arange(args.az_min, args.az_max + 0.5 * args.az_step, args.az_step)
        el_grid = np.arange(args.el_min, args.el_max + 0.5 * args.el_step, args.el_step)
        write_beam_pattern_plot(
            args.beam_pattern_output,
            positions,
            fc,
            args.look_az,
            args.look_el,
            az_grid,
            el_grid,
            info_rows=[
                ("File", os.path.basename(args.rdc0_input or args.input_file or cache_path)),
                ("Tuning freq", f"{fc / 1e6:.6f} MHz"),
                ("Array channels", f"{len(selected_channel_indices)} {args.pol}-pol"),
                ("Stands used", str(len(np.unique(channel_stands)))),
            ],
            corner_note=corner_note,
            footer_note=footer_note,
            vmin=args.beam_pattern_vmin,
            vmax=args.beam_pattern_vmax,
        )
        print(f"Wrote {args.beam_pattern_output}", flush=True)

    if delete_cache and not args.keep_temp_rdc0:
        try:
            os.remove(cache_path)
        except OSError:
            pass


if __name__ == "__main__":
    main()
