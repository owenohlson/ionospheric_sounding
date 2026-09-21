# sliding_array_delay_doppler.py

import argparse
import gc
import glob
import os
import subprocess
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import psutil

from rdc_cache import prepare_rdc2_view

from lfm_utils import LFMWaveform, reference_gate_frequency_from_args
from plotting_utils import (
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
    BAD_LWA_STANDS,
    capon_mvdr_spectrum,
    dechirp_range_fft_delay_band_rdc0,
    doppler_fft_rdc1,
    export_tbn_rdc0_hdf5,
    format_bytes,
    iter_rdc0_hdf5_channel_chunks,
    load_rdc0_hdf5_metadata,
    parse_stands,
)
from array_delay_doppler import (
    prepare_array_input,
    apply_rdc2_metadata_defaults,
    array_cable_factors,
    angle_snapshot_block,
    array_delay_doppler_power_from_rdc2,
    beamform_rdc2,
    estimate_spatial_covariance,
    fallback_if_angle_spectrum_flat,
    load_rdc2_frame_for_array,
    plot_angle_spectra,
    scan_look_direction,
    select_array_channels,
    spatial_spectrum_from_covariance,
    write_beam_pattern_plot,
    zero_fill_nonfinite_complex,
)
from tbn_utils import file_timestamp_range_note, gap_fill_note, lsl_open_tbn, lsl_print_metadata, probe_start_timestamp, set_default_tbn_time_bounds


process = psutil.Process(os.getpid())


def mem():
    print(f"Memory usage: {process.memory_info().rss / 1e9:.2f} GB", flush=True)


def cleanup_frame_pngs(output_dir):
    frame_paths = sorted(glob.glob(os.path.join(os.path.expanduser(output_dir), "frame_*.png")))
    removed = 0
    for frame_path in frame_paths:
        try:
            os.remove(frame_path)
            removed += 1
        except OSError as exc:
            print(f"Could not remove frame PNG {frame_path}: {exc}", flush=True)
    print(f"Removed {removed} frame PNG(s). Use --keep-frames to preserve them.", flush=True)


def make_video(output_dir, framerate, video_name="array_delay_doppler_video.mp4"):
    full_output_path = os.path.expanduser(output_dir)
    video_path = os.path.join(full_output_path, video_name)
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-r", str(framerate),
        "-start_number", "0",
        "-i", os.path.join(full_output_path, "frame_%04d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        video_path,
    ]
    try:
        subprocess.run(ffmpeg_cmd, check=True)
        print(f"Video created successfully: {video_path}", flush=True)
        return True
    except subprocess.CalledProcessError as exc:
        print(f"Error creating video: {exc}", flush=True)
    except FileNotFoundError:
        print("ffmpeg not found. Please install ffmpeg to create the video.", flush=True)
    return False


def angle_frame_path(angle_output_dir, frame_idx):
    return os.path.join(os.path.expanduser(angle_output_dir), f"angle_frame_{frame_idx:04d}.png")


def make_angle_video(angle_output_dir, framerate):
    full_output_path = os.path.expanduser(angle_output_dir)
    video_path = os.path.join(full_output_path, "array_angle_spectrum_video.mp4")
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-r", str(framerate),
        "-start_number", "0",
        "-i", os.path.join(full_output_path, "angle_frame_%04d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        video_path,
    ]
    try:
        subprocess.run(ffmpeg_cmd, check=True)
        print(f"Angle video created successfully: {video_path}", flush=True)
        return True
    except subprocess.CalledProcessError as exc:
        print(f"Error creating angle video: {exc}", flush=True)
    except FileNotFoundError:
        print("ffmpeg not found. Please install ffmpeg to create the angle video.", flush=True)
    return False


def cleanup_angle_frame_pngs(angle_output_dir):
    frame_paths = sorted(glob.glob(os.path.join(os.path.expanduser(angle_output_dir), "angle_frame_*.png")))
    removed = 0
    for frame_path in frame_paths:
        try:
            os.remove(frame_path)
            removed += 1
        except OSError as exc:
            print(f"Could not remove angle frame PNG {frame_path}: {exc}", flush=True)
    print(f"Removed {removed} angle frame PNG(s). Use --keep-frames to preserve them.", flush=True)


def maybe_write_beam_pattern(args, positions, fc, look_az, look_el, selected_channel_indices, channel_stands, source_name, corner_note, footer_note):
    if args.beam_pattern_output is None or getattr(args, "_beam_pattern_written", False):
        return
    if look_az is None or look_el is None:
        return
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
            ("File", os.path.basename(source_name)),
            ("Tuning freq", f"{fc / 1e6:.6f} MHz"),
            ("Array channels", f"{len(selected_channel_indices)} {args.pol}-pol"),
            ("Stands used", str(len(np.unique(channel_stands)))),
        ],
        corner_note=corner_note,
        footer_note=footer_note,
        vmin=args.beam_pattern_vmin,
        vmax=args.beam_pattern_vmax,
    )
    args._beam_pattern_written = True
    print(f"Wrote {args.beam_pattern_output}", flush=True)


def export_frame_rdc0(args, frame_tstart, start_offset_samples):
    fd, cache_path = tempfile.mkstemp(prefix="rdc0_array_frame_", suffix=".h5")
    os.close(fd)
    stands = parse_stands(args.stands, max_stand=args.max_stand, drop_bad=args.drop_bad_stands)
    idf = lsl_open_tbn(args.input_file)
    result = export_tbn_rdc0_hdf5(
        idf=idf,
        output_file=cache_path,
        tstart=frame_tstart,
        duration=args.integration_time,
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
    print(f"Frame RDC0 shape: {result['shape']} ({format_bytes(result['bytes_uncompressed'])} uncompressed)", flush=True)
    return cache_path


def process_array_delay_doppler_frame(
    cache_path,
    args,
    lfm_config,
    fc,
    frame_idx,
    nframes,
    frame_tstart,
    frame_tend,
    source_name,
    corner_note=None,
    pulse_start=None,
    pulse_stop=None,
):
    loaded = load_rdc0_hdf5_metadata(cache_path)
    rdc0_shape = loaded["shape"]
    channel_stands_all = loaded["channel_stands"]
    channel_pols_all = loaded["channel_pols"]
    metadata = loaded["metadata"]
    fs = float(metadata.get("sample_rate_hz", lfm_config.sample_rate))
    adc_start = int(metadata.get("adc_start", args.adc_start))
    if fc is None:
        fc = metadata.get("tuning_frequency_hz", metadata.get("carrier_frequency_hz", metadata.get("freq1_hz")))
    if fc is not None:
        fc = float(fc)

    wants_angle_product = args.angle_plot_output is not None
    wants_pattern_product = args.beam_pattern_output is not None
    wants_positions = args.beamformer in ("mvdr", "conventional") or wants_angle_product or wants_pattern_product
    if wants_positions and fc is None:
        raise ValueError("--carrier-frequency is required for array beamforming when RDC0 metadata does not provide TBN freq1")

    selected_channel_indices, channel_stands, channel_pols, positions, missing_position_stands = select_array_channels(
        channel_stands_all,
        channel_pols_all,
        args.pol,
        drop_bad=args.drop_bad_stands,
        require_positions=wants_positions,
        operational_only=args.operational_only,
    )
    if missing_position_stands:
        print(
            "Dropping selected channel(s) without LSL LWA-SV positions before array processing: "
            f"{missing_position_stands}",
            flush=True,
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
    look_az = args.look_az
    look_el = args.look_el
    covariance_note = None
    frame_warning = None

    cable_factors = array_cable_factors(args, metadata, channel_stands, channel_pols, fc)
    for chunk_start, chunk_stop, rdc0_chunk in iter_rdc0_hdf5_channel_chunks(
        cache_path,
        args.channel_chunk_size,
        channel_indices=selected_channel_indices,
        pulse_start=pulse_start,
        pulse_stop=pulse_stop,
    ):
        print(
            f"Frame {frame_idx + 1}/{nframes}: processing selected channels "
            f"{chunk_start + 1}-{chunk_stop}/{len(selected_channel_indices)}",
            flush=True,
        )
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
                print(
                    f"Frame {frame_idx + 1}/{nframes}: auto-scanning look direction with "
                    f"{scan_method.upper()} power...",
                    flush=True,
                )
                _, look_az, look_el = scan_look_direction(
                    positions,
                    fc,
                    signal_covariance,
                    az_grid,
                    el_grid,
                    method=scan_method,
                )
            print(
                f"Frame {frame_idx + 1}/{nframes}: {args.beamformer.upper()} beamforming "
                f"at az={look_az:.2f} deg, el={look_el:.2f} deg",
                flush=True,
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
            power_plot = 10.0 * np.log10(np.abs(beamformed) ** 2 + 1e-12)
            del beamformed, signal_covariance
        except ValueError as exc:
            if "Not enough finite delay-Doppler snapshots" not in str(exc):
                raise
            frame_warning = str(exc)
            covariance_note = "unavailable"
            power_plot = np.full((len(delay_ms), len(fd_plot)), np.nan, dtype=float)
            print(
                f"Frame {frame_idx + 1}/{nframes}: {frame_warning}; writing blank frame",
                flush=True,
            )

    if wants_angle_product:
        if rdc2_plot is None:
            raise ValueError("Angle plotting requires the selected per-channel RDC2 cube")
        wavelength = 299_792_458.0 / float(fc)
        az_grid = np.arange(args.az_min, args.az_max + 0.5 * args.az_step, args.az_step)
        el_grid = np.arange(args.el_min, args.el_max + 0.5 * args.el_step, args.el_step)
        spectra = []
        delay_cells = []
        doppler_cells = []
        cell_snapshot_info = []
        cell_labels = []
        angle_warnings = []
        angle_source = args.angle_source
        if args.angle_delay_ms is not None or args.angle_top_n > 0:
            angle_source = "cells"

        if angle_source == "window":
            try:
                print(
                    f"Frame {frame_idx + 1}/{nframes}: computing {args.angle_method.upper()} angle spectrum "
                    "over the full plotted delay-Doppler window",
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
                        print(f"Frame {frame_idx + 1}/{nframes}: {fallback_note}", flush=True)
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
            except ValueError as exc:
                if "Not enough finite delay-Doppler snapshots" not in str(exc):
                    raise
                spectra.append(np.full((len(el_grid), len(az_grid)), np.nan, dtype=np.float32))
                delay_cells.append(float(np.nanmean(delay_ms)))
                doppler_cells.append(float(np.nanmean(fd_plot)))
                cell_snapshot_info.append({
                    "snapshots": 0,
                    "delay_bins": int(rdc2_plot.shape[0]),
                    "doppler_bins": int(rdc2_plot.shape[1]),
                })
                cell_labels.append("full DD window, no finite snapshots")
                frame_warning = str(exc)
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
                spectra.append(np.full((len(el_grid), len(az_grid)), np.nan, dtype=np.float32))
                delay_cells.append(float(np.nanmean(delay_ms)))
                doppler_cells.append(float(np.nanmean(fd_plot)))
                cell_snapshot_info.append({
                    "snapshots": 0,
                    "delay_bins": 0,
                    "doppler_bins": 0,
                })
                cell_labels.append("selected cells unavailable")

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
                    f"Frame {frame_idx + 1}/{nframes}: computing {args.angle_method.upper()} angle spectrum "
                    f"at delay={delay_ms[ri]:.3f} ms, fd={fd_plot[di]:.3f} Hz "
                    f"using {snapshot_info['snapshots']} snapshots",
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
                        print(f"Frame {frame_idx + 1}/{nframes}: {fallback_note}", flush=True)
                        angle_warnings.append(fallback_note)
                spectra.append(spectrum)
                delay_cells.append(delay_ms[ri])
                doppler_cells.append(fd_plot[di])
                cell_snapshot_info.append(snapshot_info)

        angle_output_dir = os.path.expanduser(args.angle_plot_output)
        os.makedirs(angle_output_dir, exist_ok=True)
        plot_angle_spectra(
            angle_frame_path(angle_output_dir, frame_idx),
            np.asarray(spectra),
            az_grid,
            el_grid,
            np.asarray(delay_cells),
            np.asarray(doppler_cells),
            vmin=args.angle_plot_vmin,
            vmax=args.angle_plot_vmax,
            title="Array Angle Spectrum",
            info_rows=[
                ("File", os.path.basename(source_name)),
                ("Frame", f"{frame_idx + 1}/{nframes}"),
                ("Tuning freq", f"{fc / 1e6:.6f} MHz"),
                ("Array channels", f"{len(selected_channel_indices)} {args.pol}-pol"),
                ("Stands used", str(len(np.unique(channel_stands)))),
                ("Angle method", args.angle_method.upper()),
                ("Cable phase", getattr(args, "_cable_note", "unknown")),
                ("Angle source", angle_source),
                ("Min snapshots", str(args.angle_min_snapshots)),
                ("Diag loading", f"{args.diagonal_loading:g}"),
                ("Plot span", f"{frame_tstart:.3f}-{frame_tend:.3f} s"),
                ("Integration", f"{args.integration_time:.3f} s"),
                ("Hop", f"{args.hop_time:.3f} s"),
            ],
            corner_note=corner_note,
            footer_note="\n".join(note for note in [
                gap_fill_note(metadata.get("gap_info")),
                f"RDC0 nonfinite values replaced with zeros in this frame: {nonfinite_rdc0_values_filled}"
                if nonfinite_rdc0_values_filled else None,
                f"RDC2 nonfinite values replaced with zeros in this frame: {nonfinite_rdc2_values_filled}"
                if nonfinite_rdc2_values_filled else None,
                *angle_warnings,
                frame_warning,
            ] if note),
            cell_snapshot_info=cell_snapshot_info,
            cell_labels=cell_labels,
        )

    maybe_write_beam_pattern(
        args,
        positions,
        fc,
        look_az,
        look_el,
        selected_channel_indices,
        channel_stands,
        source_name,
        corner_note,
        "\n".join(note for note in [
            gap_fill_note(metadata.get("gap_info")),
            f"RDC0 nonfinite values replaced with zeros in this frame: {nonfinite_rdc0_values_filled}"
            if nonfinite_rdc0_values_filled else None,
            f"RDC2 nonfinite values replaced with zeros in this frame: {nonfinite_rdc2_values_filled}"
            if nonfinite_rdc2_values_filled else None,
            frame_warning,
        ] if note),
    )

    if rdc2_plot is not None:
        del rdc2_plot

    info_rows = [
        ("File", os.path.basename(source_name)),
    ]
    if fc is not None:
        info_rows.append(("Tuning freq", f"{fc / 1e6:.6f} MHz"))
    info_rows.extend([
        ("Frame", f"{frame_idx + 1}/{nframes}"),
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
        if look_az is None or look_el is None:
            info_rows.append(("Look dir", "unavailable"))
        else:
            info_rows.append(("Look dir", f"{look_label}: az {look_az:.2f} deg, el {look_el:.2f} deg"))
    if args.beamformer == "mvdr":
        info_rows.append(("MVDR cov", covariance_note))
        info_rows.append(("Diag loading", f"{args.diagonal_loading:g}"))
    info_rows.extend([
        ("Plot span", f"{frame_tstart:.3f}-{frame_tend:.3f} s"),
        ("Integration", f"{args.integration_time:.3f} s"),
        ("Hop", f"{args.hop_time:.3f} s"),
        ("Doppler FFT", str(len(fd_hz))),
        ("Delay limits", f"{args.d_min} to {args.d_max} ms"),
    ])
    if args.fd_min is not None or args.fd_max is not None:
        info_rows.append(("Doppler limits", f"{args.fd_min} to {args.fd_max} Hz"))
    info_rows.extend(_lfm_info_rows(lfm_config))
    info_rows.append(("Cable phase", getattr(args, "_cable_note", "unknown")))

    footer_note = "\n".join(note for note in [
        gap_fill_note(metadata.get("gap_info")),
        f"RDC0 nonfinite values replaced with zeros in this frame: {nonfinite_rdc0_values_filled}"
        if nonfinite_rdc0_values_filled else None,
        f"RDC2 nonfinite values replaced with zeros in this frame: {nonfinite_rdc2_values_filled}"
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
    _add_info_panel(
        fig,
        info_rows=info_rows,
        corner_note=corner_note,
        lower_note=lower_note,
        footer_note=footer_note,
    )
    _tight_layout_with_info(fig, has_info=has_info)

    output_dir = os.path.expanduser(args.output)
    os.makedirs(output_dir, exist_ok=True)
    frame_path = os.path.join(output_dir, f"frame_{frame_idx:04d}.png")
    _savefig(frame_path, dpi=300)
    plt.close(fig)
    print(f"Wrote {frame_path}", flush=True)


def process_array_delay_doppler_rdc2_frame(args, fc, frame_idx, nframes, source_name, corner_note=None):
    wants_angle_product = args.angle_plot_output is not None
    wants_pattern_product = args.beam_pattern_output is not None
    loaded = load_rdc2_frame_for_array(
        args,
        frame_idx=frame_idx,
        require_positions=args.beamformer in ("mvdr", "conventional") or wants_angle_product or wants_pattern_product,
    )
    metadata = loaded["metadata"]
    fs = float(metadata.get("sample_rate_hz"))
    if args.sweep_frequency is None or args.bandwidth is None:
        raise ValueError("RDC2 metadata is missing sweep_frequency_hz or bandwidth_hz")
    if fc is None:
        fc = metadata.get("tuning_frequency_hz", metadata.get("carrier_frequency_hz", metadata.get("freq1_hz")))
    if fc is not None:
        fc = float(fc)
    if (wants_angle_product or wants_pattern_product or args.beamformer in ("mvdr", "conventional")) and fc is None:
        raise ValueError("--carrier-frequency is required for array beamforming when RDC2 metadata does not provide a frequency")

    rdc2_plot = loaded["rdc2"]
    delay_ms = loaded["delay_ms"]
    fd_plot = loaded["fd_hz"]
    selected_channel_indices = loaded["selected_channel_indices"]
    channel_stands = loaded["channel_stands"]
    channel_pols = loaded["channel_pols"]
    positions = loaded["positions"]
    missing_position_stands = loaded["missing_position_stands"]
    nonfinite_rdc2_values_filled = loaded["nonfinite_values_filled"]
    frame_tstart = float(loaded["frame_tstart_s"][frame_idx])
    frame_tend = float(loaded["frame_tend_s"][frame_idx])
    integration = frame_tend - frame_tstart
    hop = float(metadata.get("hop_time_s", args.hop_time if args.hop_time is not None else integration))

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
        print(f"Frame {frame_idx + 1}/{nframes}: {frame_warning}; writing blank frame", flush=True)
    elif args.beamformer in ("mvdr", "conventional") and look_az is not None and look_el is not None:
        print(
            f"Frame {frame_idx + 1}/{nframes}: {args.beamformer.upper()} beamforming "
            f"at az={look_az:.2f} deg, el={look_el:.2f} deg",
            flush=True,
        )

    lfm_config = LFMWaveform(
        sample_rate=fs,
        sweep_frequency=args.sweep_frequency,
        bandwidth=args.bandwidth,
        reference_gate_frequency=metadata.get("reference_gate_frequency"),
        reference_gate_duty=metadata.get("reference_gate_duty", args.reference_gate_duty),
        reference_gate_phase=metadata.get("reference_gate_phase", args.reference_gate_phase),
    )

    if wants_angle_product:
        az_grid = np.arange(args.az_min, args.az_max + 0.5 * args.az_step, args.az_step)
        el_grid = np.arange(args.el_min, args.el_max + 0.5 * args.el_step, args.el_step)
        spectra = []
        delay_cells = []
        doppler_cells = []
        cell_snapshot_info = []
        cell_labels = []
        angle_warnings = []
        angle_source = args.angle_source
        if args.angle_delay_ms is not None or args.angle_top_n > 0:
            angle_source = "cells"

        if angle_source == "window":
            try:
                print(
                    f"Frame {frame_idx + 1}/{nframes}: computing {args.angle_method.upper()} angle spectrum "
                    "over the cached delay-Doppler window",
                    flush=True,
                )
                window_covariance = estimate_spatial_covariance(rdc2_plot, diagonal_loading=args.diagonal_loading)
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
                        print(f"Frame {frame_idx + 1}/{nframes}: {fallback_note}", flush=True)
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
                    f"full cached DD window, delay {np.nanmin(delay_ms):.3f}-{np.nanmax(delay_ms):.3f} ms, "
                    f"Doppler {np.nanmin(fd_plot):.3f}-{np.nanmax(fd_plot):.3f} Hz"
                )
            except ValueError as exc:
                if "Not enough finite delay-Doppler snapshots" not in str(exc):
                    raise
                spectra.append(np.full((len(el_grid), len(az_grid)), np.nan, dtype=np.float32))
                delay_cells.append(float(np.nanmean(delay_ms)))
                doppler_cells.append(float(np.nanmean(fd_plot)))
                cell_snapshot_info.append({"snapshots": 0, "delay_bins": int(rdc2_plot.shape[0]), "doppler_bins": int(rdc2_plot.shape[1])})
                cell_labels.append("full cached DD window, no finite snapshots")
                frame_warning = str(exc)
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
                spectra.append(np.full((len(el_grid), len(az_grid)), np.nan, dtype=np.float32))
                delay_cells.append(float(np.nanmean(delay_ms)))
                doppler_cells.append(float(np.nanmean(fd_plot)))
                cell_snapshot_info.append({"snapshots": 0, "delay_bins": 0, "doppler_bins": 0})
                cell_labels.append("selected cells unavailable")
            for ri, di in cell_indices:
                snapshot_block, snapshot_info = angle_snapshot_block(
                    rdc2_plot,
                    ri,
                    di,
                    delay_radius_bins=args.angle_delay_radius_bins,
                    doppler_radius_bins=args.angle_doppler_radius_bins,
                    min_snapshots=args.angle_min_snapshots,
                )
                spectrum = capon_mvdr_spectrum(
                    snapshot_block,
                    positions,
                    299_792_458.0 / float(fc),
                    az_grid,
                    el_grid,
                    diagonal_loading=args.diagonal_loading,
                    method=args.angle_method,
                )
                spectra.append(spectrum)
                delay_cells.append(delay_ms[ri])
                doppler_cells.append(fd_plot[di])
                cell_snapshot_info.append(snapshot_info)

        angle_output_dir = os.path.expanduser(args.angle_plot_output)
        os.makedirs(angle_output_dir, exist_ok=True)
        plot_angle_spectra(
            angle_frame_path(angle_output_dir, frame_idx),
            np.asarray(spectra),
            az_grid,
            el_grid,
            np.asarray(delay_cells),
            np.asarray(doppler_cells),
            vmin=args.angle_plot_vmin,
            vmax=args.angle_plot_vmax,
            title="Array Angle Spectrum",
            info_rows=[
                ("File", os.path.basename(source_name)),
                ("Frame", f"{frame_idx + 1}/{nframes}"),
                ("Tuning freq", f"{fc / 1e6:.6f} MHz"),
                ("Array channels", f"{len(selected_channel_indices)} {args.pol}-pol"),
                ("Stands used", str(len(np.unique(channel_stands)))),
                ("Angle method", args.angle_method.upper()),
                ("Cable phase", getattr(args, "_cable_note", "unknown")),
                ("Angle source", angle_source),
                ("RDC2 cache", "yes"),
                ("Plot span", f"{frame_tstart:.3f}-{frame_tend:.3f} s"),
                ("Integration", f"{integration:.3f} s"),
                ("Hop", f"{hop:.3f} s"),
            ],
            corner_note=corner_note,
            footer_note="\n".join(note for note in [
                gap_fill_note(metadata.get("gap_info")),
                f"RDC2 nonfinite values replaced with zeros in this frame: {nonfinite_rdc2_values_filled}"
                if nonfinite_rdc2_values_filled else None,
                *angle_warnings,
                frame_warning,
            ] if note),
            cell_snapshot_info=cell_snapshot_info,
            cell_labels=cell_labels,
        )

    maybe_write_beam_pattern(
        args,
        positions,
        fc,
        look_az,
        look_el,
        selected_channel_indices,
        channel_stands,
        source_name,
        corner_note,
        "\n".join(note for note in [
            gap_fill_note(metadata.get("gap_info")),
            f"RDC2 nonfinite values replaced with zeros in this frame: {nonfinite_rdc2_values_filled}"
            if nonfinite_rdc2_values_filled else None,
            frame_warning,
        ] if note),
    )

    info_rows = [
        ("File", os.path.basename(source_name)),
        ("Frame", f"{frame_idx + 1}/{nframes}"),
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
        ("Integration", f"{integration:.3f} s"),
        ("Hop", f"{hop:.3f} s"),
        ("Doppler FFT", str(len(fd_plot))),
        ("Delay limits", f"{args.d_min} to {args.d_max} ms"),
        ("RDC2 cache", "yes"),
    ])
    info_rows.extend(_lfm_info_rows(lfm_config))
    info_rows.append(("Cable phase", getattr(args, "_cable_note", "unknown")))
    footer_note = "\n".join(note for note in [
        gap_fill_note(metadata.get("gap_info")),
        f"RDC2 nonfinite values replaced with zeros in this frame: {nonfinite_rdc2_values_filled}"
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

    output_dir = os.path.expanduser(args.output)
    os.makedirs(output_dir, exist_ok=True)
    frame_path = os.path.join(output_dir, f"frame_{frame_idx:04d}.png")
    _savefig(frame_path, dpi=300)
    plt.close(fig)
    print(f"Wrote {frame_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Create a sliding array/beamformed delay-Doppler video from TBN or RDC0 data.",
        formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, max_help_position=35),
    )
    parser.add_argument("input_file", nargs="?", help="Input TBN file; omit when using --rdc0-input")
    parser.add_argument("--rdc0-input", default=None, help="Existing RDC0 HDF5 file to process")
    parser.add_argument("--rdc2-input", default=None, help="RDC2 cache; timing overrides recompute Doppler from retained RDC1")
    parser.add_argument("--output", required=True, help="Output directory for frame PNGs and video")

    parser.add_argument("--integration-time", type=float, default=None)
    parser.add_argument("--hop-time", type=float, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--framerate", type=int, default=2)
    parser.add_argument("--keep-frames", action="store_true")
    parser.add_argument("--angle-plot-output", default=None,
                        help="Output directory for per-frame MVDR azimuth/elevation spectra and video")
    parser.add_argument("--beam-pattern-output", default=None,
                        help="Output PNG showing the conventional array point-spread/beam pattern for the first resolved look direction")
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
                        help="Compute angle spectra for the top N delay-Doppler cells in each frame")
    parser.add_argument("--angle-delay-ms", type=float, default=None,
                        help="Specific delay cell for angle-spectrum video")
    parser.add_argument("--angle-doppler-hz", type=float, default=None,
                        help="Specific Doppler cell for angle-spectrum video")
    parser.add_argument("--angle-delay-radius-bins", type=int, default=1,
                        help="Delay-bin half-width used as snapshots for each angle spectrum")
    parser.add_argument("--angle-doppler-radius-bins", type=int, default=1,
                        help="Doppler-bin half-width used as snapshots for each angle spectrum")
    parser.add_argument("--angle-min-snapshots", type=int, default=256,
                        help="Automatically expand the angle-spectrum snapshot window until at least this many snapshots are used")
    parser.add_argument("--angle-method", choices=["mvdr", "conventional"], default="conventional",
                        help="Spatial spectrum used for angle plots")
    parser.add_argument("--angle-flat-fallback", choices=["conventional", "none"], default="conventional",
                        help="When --angle-method=mvdr gives a nearly flat map, plot this fallback instead")
    parser.add_argument("--angle-flat-threshold-db", type=float, default=0.25,
                        help="MVDR angle-map span below this threshold is treated as flat")

    parser.add_argument("--tstart", type=float, default=None)
    parser.add_argument("--tend", type=float, default=None)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--sweep-frequency", type=float, default=None)
    parser.add_argument("--bandwidth", type=float, default=None)
    parser.add_argument("--offset", type=float, default=0.0)
    parser.add_argument("--no-timestamp-align", action="store_true")

    parser.add_argument("--stands", default="all")
    parser.add_argument("--max-stand", type=int, default=256)
    parser.add_argument("-p", "--pol", choices=["x", "y", "0", "1"], default="x",
                        help="Array processing uses one receiver polarization at a time")
    parser.add_argument("--drop-bad-stands", action="store_true", default=True,
                        help=f"Drop known bad stands before array processing: {BAD_LWA_STANDS}")
    parser.add_argument("--keep-bad-stands", action="store_false", dest="drop_bad_stands")

    parser.add_argument("--adc-start", type=int, default=0)
    parser.add_argument("--adc-count", type=int, default=None)
    parser.add_argument("--gap-fill", choices=["nan", "zero"], default="nan")
    parser.add_argument("--rdc0-dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--rdc0-compression", choices=["gzip", "lzf", "none"], default="gzip")
    parser.add_argument("--rdc0-compression-level", type=int, default=4)
    parser.add_argument("--channel-chunk-size", type=int, default=8)

    parser.add_argument("--dechirp-window", choices=["hamming", "hann", "blackman", "cheb60", "cheb80", "cheb100", "cheb120", "none"], default="hann")
    parser.add_argument("--slow-window", choices=["hann", "hamming", "blackman", "cheb60", "cheb80", "cheb100", "cheb120", "none"], default="hann")
    parser.add_argument("--nfft-doppler", type=int, default=None)
    parser.add_argument("--beamformer", choices=["mvdr", "conventional", "incoherent"], default="conventional")
    parser.add_argument("--channel-combine", choices=["sum", "mean", "max"], default="sum",
                        help="Only used with --beamformer incoherent")
    parser.add_argument("--look-az", type=float, default=None,
                        help="Beam look azimuth in degrees; omit with --look-el to auto-scan each frame")
    parser.add_argument("--look-el", type=float, default=None,
                        help="Beam look elevation in degrees; omit with --look-az to auto-scan each frame")
    parser.add_argument("--az-min", type=float, default=-180.0)
    parser.add_argument("--az-max", type=float, default=180.0)
    parser.add_argument("--az-step", type=float, default=2.0)
    parser.add_argument("--el-min", type=float, default=0.0)
    parser.add_argument("--el-max", type=float, default=90.0)
    parser.add_argument("--el-step", type=float, default=2.0)
    parser.add_argument("--diagonal-loading", type=float, default=0.1)
    parser.add_argument("--mvdr-covariance", choices=["background", "all"], default="background")
    parser.add_argument("--mvdr-background-percentile", type=float, default=80.0)
    parser.add_argument("--carrier-frequency", type=float, default=None)

    parser.add_argument("--d-min", type=float, default=None)
    parser.add_argument("--d-max", type=float, default=None)
    parser.add_argument("--fd-min", type=float, default=None)
    parser.add_argument("--fd-max", type=float, default=None)
    parser.add_argument("--vmin", type=float, default=None)
    parser.add_argument("--vmax", type=float, default=None)
    parser.add_argument("--title", default="Array Delay-Doppler Video")

    parser.add_argument("--reference-gate-frequency", type=float, default=None)
    parser.add_argument("--reference-gate-period", type=float, default=None)
    parser.add_argument("--reference-gate-duty", type=float, default=0.5)
    parser.add_argument("--reference-gate-phase", type=float, default=0.0)

    args = parser.parse_args()
    prepare_array_input(args)
    if args.rdc2_input is None and args.rdc0_input is None and args.input_file is None:
        raise ValueError("Provide a TBN input_file, --rdc0-input, or --rdc2-input")
    if args.rdc2_input is not None and (args.rdc0_input is not None or args.input_file is not None):
        raise ValueError("Use --rdc2-input by itself, not together with TBN input_file or --rdc0-input")
    if (args.look_az is None) != (args.look_el is None):
        raise ValueError("Provide both --look-az and --look-el, or omit both for auto look-direction scanning")
    if args.beam_pattern_output and args.beamformer == "incoherent" and args.look_az is None:
        raise ValueError("--beam-pattern-output with --beamformer=incoherent requires --look-az and --look-el")
    if (args.angle_delay_ms is None) != (args.angle_doppler_hz is None):
        raise ValueError("Provide both --angle-delay-ms and --angle-doppler-hz")

    output_dir = os.path.expanduser(args.output)
    os.makedirs(output_dir, exist_ok=True)
    if args.angle_plot_output:
        os.makedirs(os.path.expanduser(args.angle_plot_output), exist_ok=True)

    fc = args.carrier_frequency
    fs = None
    source_name = args.rdc0_input or args.input_file
    start_timestamp_for_alignment = None
    if args.rdc2_input:
        source_name = args.rdc2_input
        rdc2_metadata = prepare_rdc2_view(args)
        args._rdc2_view = rdc2_metadata
        apply_rdc2_metadata_defaults(args, rdc2_metadata["metadata"])
        if args.sweep_frequency is None or args.bandwidth is None:
            raise ValueError("RDC2 metadata is missing sweep_frequency_hz or bandwidth_hz")
        fc = fc if fc is not None else rdc2_metadata["metadata"].get("tuning_frequency_hz")
        total_frames = int(rdc2_metadata["shape"][0])
        nframes = total_frames if args.max_frames is None else min(total_frames, int(args.max_frames))
        for frame_idx in range(nframes):
            print(f"Processing RDC2 frame {frame_idx + 1}/{nframes}", flush=True)
            mem()
            process_array_delay_doppler_rdc2_frame(
                args,
                fc,
                frame_idx,
                nframes,
                source_name,
                corner_note=None,
            )
            gc.collect()
            print(f"Finished frame {frame_idx + 1}/{nframes}", flush=True)
            mem()
        print("Plotting complete, creating video with ffmpeg...", flush=True)
        video_created = make_video(output_dir, args.framerate)
        if video_created and not args.keep_frames:
            cleanup_frame_pngs(output_dir)
        if args.angle_plot_output:
            print("Creating angle-spectrum video with ffmpeg...", flush=True)
            angle_video_created = make_angle_video(args.angle_plot_output, args.framerate)
            if angle_video_created and not args.keep_frames:
                cleanup_angle_frame_pngs(args.angle_plot_output)
        return

    if args.sweep_frequency is None or args.bandwidth is None:
        raise ValueError("--sweep-frequency and --bandwidth are required unless --rdc2-input metadata provides them")
    if args.d_min is None or args.d_max is None:
        raise ValueError("--d-min and --d-max are required unless --rdc2-input metadata provides them")
    if args.integration_time is None:
        args.integration_time = 120.0
    if args.hop_time is None:
        args.hop_time = 30.0
    if args.integration_time <= 0 or args.hop_time <= 0:
        raise ValueError("--integration-time and --hop-time must be > 0")

    if args.rdc0_input:
        cache_path = os.path.expanduser(args.rdc0_input)
        loaded = load_rdc0_hdf5_metadata(cache_path)
        metadata = loaded["metadata"]
        rdc0_shape = loaded["shape"]
        if metadata.get("sample_rate_hz") is None:
            raise ValueError("RDC0 metadata is missing sample_rate_hz")
        fs = float(metadata.get("sample_rate_hz"))
        fc = fc if fc is not None else metadata.get("tuning_frequency_hz", metadata.get("carrier_frequency_hz", metadata.get("freq1_hz")))
        cache_duration = rdc0_shape[1] / float(args.sweep_frequency)
        if args.tstart is None:
            args.tstart = 0.0
        if args.tend is None:
            args.tend = cache_duration
        if args.duration is not None:
            args.tend = args.tstart + args.duration
        print(f"Using RDC0 cache duration {cache_duration:.3f} s; frame times are relative to cache start.", flush=True)
    else:
        idf = lsl_open_tbn(args.input_file)
        lsl_print_metadata(idf)
        fs = float(idf.get_info("sample_rate"))
        fc = fc if fc is not None else float(idf.get_info("freq1"))
        set_default_tbn_time_bounds(args, idf)
        if args.duration is not None:
            args.tend = args.tstart + args.duration

    if args.tend <= args.tstart:
        raise ValueError("--tend must be greater than --tstart")
    if args.tend - args.tstart < args.integration_time:
        raise ValueError("Selected time range is shorter than --integration-time")

    lfm_config = LFMWaveform(
        sample_rate=fs,
        sweep_frequency=args.sweep_frequency,
        bandwidth=args.bandwidth,
        reference_gate_frequency=reference_gate_frequency_from_args(args),
        reference_gate_duty=args.reference_gate_duty,
        reference_gate_phase=args.reference_gate_phase,
    )

    available = args.tend - args.tstart - args.integration_time
    nframes = int(np.floor(available / args.hop_time)) + 1
    if args.max_frames is not None:
        nframes = min(nframes, int(args.max_frames))

    pulse_count = int(round(args.integration_time * args.sweep_frequency))
    if pulse_count < 1:
        raise ValueError("--integration-time is shorter than one sweep")

    for frame_idx in range(nframes):
        frame_tstart = args.tstart + frame_idx * args.hop_time
        frame_tend = frame_tstart + args.integration_time
        print(f"Processing frame {frame_idx + 1}/{nframes} (t={frame_tstart:.2f} to {frame_tend:.2f} s)", flush=True)
        mem()

        frame_cache_path = None
        delete_frame_cache = False
        corner_note = None
        pulse_start = None
        pulse_stop = None
        try:
            if args.rdc0_input:
                frame_cache_path = os.path.expanduser(args.rdc0_input)
                pulse_start = int(round(frame_tstart * args.sweep_frequency))
                pulse_stop = pulse_start + pulse_count
            else:
                if not args.no_timestamp_align:
                    start_timestamp_for_alignment = probe_start_timestamp(args.input_file, frame_tstart)
                    start_offset_samples = _timestamp_sweep_offset_samples(
                        start_timestamp_for_alignment,
                        lfm_config,
                        sweep_offset=args.offset,
                    )
                else:
                    start_offset_samples = 0
                idf_for_note = lsl_open_tbn(args.input_file)
                corner_note = file_timestamp_range_note(idf_for_note, frame_tstart, frame_tend)
                frame_cache_path = export_frame_rdc0(args, frame_tstart, start_offset_samples)
                delete_frame_cache = True

            process_array_delay_doppler_frame(
                frame_cache_path,
                args,
                lfm_config,
                fc,
                frame_idx,
                nframes,
                frame_tstart,
                frame_tend,
                source_name,
                corner_note=corner_note,
                pulse_start=pulse_start,
                pulse_stop=pulse_stop,
            )
        finally:
            if delete_frame_cache and frame_cache_path:
                try:
                    os.remove(frame_cache_path)
                except OSError:
                    pass
        gc.collect()
        print(f"Finished frame {frame_idx + 1}/{nframes}", flush=True)
        mem()

    print("Plotting complete, creating video with ffmpeg...", flush=True)
    video_created = make_video(output_dir, args.framerate)
    if video_created and not args.keep_frames:
        cleanup_frame_pngs(output_dir)
    if args.angle_plot_output:
        print("Creating angle-spectrum video with ffmpeg...", flush=True)
        angle_video_created = make_angle_video(args.angle_plot_output, args.framerate)
        if angle_video_created and not args.keep_frames:
            cleanup_angle_frame_pngs(args.angle_plot_output)


if __name__ == "__main__":
    main()
