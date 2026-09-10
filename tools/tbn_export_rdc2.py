# tbn_export_rdc2.py

import argparse
import os
import tempfile

import numpy as np

from lfm_utils import LFMWaveform, dechirp_delay_axis_ms, reference_gate_frequency_from_args
from plotting_utils import _timestamp_sweep_offset_samples
from rdc_utils import (
    BAD_LWA_STANDS,
    dechirp_range_fft_delay_band_rdc0,
    doppler_fft_rdc1,
    estimate_rdc2_bytes,
    export_tbn_rdc0_hdf5,
    format_bytes,
    iter_rdc0_hdf5_channel_chunks,
    load_rdc0_hdf5_metadata,
    metadata_json,
    parse_stands,
    polarization_metadata,
)
from tbn_utils import lsl_open_tbn, lsl_print_metadata, set_default_tbn_time_bounds


def probe_start_timestamp(filename, tstart):
    idf = lsl_open_tbn(filename)
    idf.offset(float(tstart))
    readT, timestamp, data = idf.read(0.001)
    del data
    return timestamp


def selected_rdc0_channels(channel_stands, channel_pols, stands, pol, drop_bad=False):
    from rdc_utils import pol_to_indices

    channel_stands = np.asarray(channel_stands, dtype=int)
    channel_pols = np.asarray(channel_pols, dtype=int)
    stand_set = set(int(stand) for stand in stands)
    pol_set = set(int(pol_idx) for pol_idx in pol_to_indices(pol))
    mask = np.isin(channel_stands, list(stand_set)) & np.isin(channel_pols, list(pol_set))
    if drop_bad:
        mask &= ~np.isin(channel_stands, np.asarray(BAD_LWA_STANDS, dtype=int))
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        raise ValueError("No RDC0 channels remain after applying --stands/--pol/--drop-bad-stands")
    return indices, channel_stands[indices], channel_pols[indices]


def infer_delay_bin_count(lfm_config, adc_count, d_min, d_max):
    delay_ms, delay_order = dechirp_delay_axis_ms(lfm_config, n_bins=int(adc_count))
    mask = (delay_ms >= float(d_min)) & (delay_ms <= float(d_max))
    if not np.any(mask):
        raise ValueError(
            "No delay bins selected. "
            f"Available delay range is {delay_ms.min():.3f} to {delay_ms.max():.3f} ms."
        )
    return int(np.count_nonzero(mask))


def compression_args(compression, compression_level):
    compression_arg = None if compression == "none" else compression
    opts = None if compression in ("none", "lzf") else int(compression_level)
    return compression_arg, opts


def make_temp_rdc0(args, fs):
    duration = args.tend - args.tstart
    sweep_length = int(round(fs / args.sweep_frequency))
    if args.adc_count is None:
        args.adc_count = sweep_length - args.adc_start
    if args.adc_count <= 0 or args.adc_start + args.adc_count > sweep_length:
        raise ValueError("--adc-start/--adc-count must select samples inside one sweep")

    lfm_for_offset = LFMWaveform(sample_rate=fs, sweep_frequency=args.sweep_frequency, bandwidth=args.bandwidth)
    start_offset_samples = 0
    if not args.no_timestamp_align:
        start_timestamp = probe_start_timestamp(args.input_file, args.tstart)
        start_offset_samples = _timestamp_sweep_offset_samples(
            start_timestamp,
            lfm_for_offset,
            sweep_offset=args.offset,
        )
        print(f"Timestamp alignment start_offset_samples={start_offset_samples}", flush=True)

    fd, cache_path = tempfile.mkstemp(prefix="rdc0_for_rdc2_", suffix=".h5")
    os.close(fd)
    stands = parse_stands(args.stands, max_stand=args.max_stand, drop_bad=args.drop_bad_stands)
    idf = lsl_open_tbn(args.input_file)
    result = export_tbn_rdc0_hdf5(
        idf=idf,
        output_file=cache_path,
        tstart=args.tstart,
        duration=duration,
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
    print(f"Temporary RDC0 shape: {result['shape']} ({format_bytes(result['bytes_uncompressed'])} uncompressed)", flush=True)
    return cache_path


def main():
    parser = argparse.ArgumentParser(
        description="Export a 3D RDC2 with retained RDC1 for later integration/hop selection; explicit timing exports windowed RDC2.",
        formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, max_help_position=35),
    )
    parser.add_argument("input_file", nargs="?", help="Input TBN file; omit when using --rdc0-input")
    parser.add_argument("--rdc0-input", default=None, help="Existing RDC0 HDF5 file to process")
    parser.add_argument("--output", required=True, help="Output RDC2 HDF5 file")

    parser.add_argument("--no-store-rdc1", action="store_true",
                        help="Omit per-sweep data; integration/hop cannot then be changed after export")
    parser.add_argument("--tstart", type=float, default=None)
    parser.add_argument("--tend", type=float, default=None)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--integration-time", type=float, default=None,
                        help="Seconds per RDC2 frame; default uses the full selected span")
    parser.add_argument("--hop-time", type=float, default=None,
                        help="Seconds between RDC2 frames; default equals integration time")

    parser.add_argument("--sweep-frequency", type=float, default=None)
    parser.add_argument("--bandwidth", type=float, required=True)
    parser.add_argument("--offset", type=float, default=0.0)
    parser.add_argument("--no-timestamp-align", action="store_true")

    parser.add_argument("--stands", default="all")
    parser.add_argument("--max-stand", type=int, default=256)
    parser.add_argument("-p", "--pol", choices=["both", "x", "y", "0", "1"], default="both")
    parser.add_argument("--drop-bad-stands", action="store_true",
                        help=f"Drop known bad stands before export: {BAD_LWA_STANDS}")

    parser.add_argument("--adc-start", type=int, default=0)
    parser.add_argument("--adc-count", type=int, default=None)
    parser.add_argument("--gap-fill", choices=["nan", "zero"], default="nan")
    parser.add_argument("--rdc0-dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--rdc0-compression", choices=["gzip", "lzf", "none"], default="gzip")
    parser.add_argument("--rdc0-compression-level", type=int, default=4)

    parser.add_argument("--d-min", type=float, required=True)
    parser.add_argument("--d-max", type=float, required=True)
    parser.add_argument("--fd-min", type=float, default=None)
    parser.add_argument("--fd-max", type=float, default=None)
    parser.add_argument("--dechirp-window", choices=["hamming", "hann", "blackman", "cheb60", "cheb80", "cheb100", "cheb120", "none"], default="hann")
    parser.add_argument("--slow-window", choices=["hann", "hamming", "blackman", "cheb60", "cheb80", "cheb100", "cheb120", "none"], default="hann")
    parser.add_argument("--nfft-doppler", type=int, default=None)

    parser.add_argument("--dtype", choices=["float32", "float16"], default="float32",
                        help="Real/imag RDC2 storage dtype")
    parser.add_argument("--compression", choices=["gzip", "lzf", "none"], default="gzip")
    parser.add_argument("--compression-level", type=int, default=4)
    parser.add_argument("--channel-chunk-size", type=int, default=8)
    parser.add_argument("--dry-run-size", action="store_true")

    parser.add_argument("--reference-gate-frequency", type=float, default=None)
    parser.add_argument("--reference-gate-period", type=float, default=None)
    parser.add_argument("--reference-gate-duty", type=float, default=0.5)
    parser.add_argument("--reference-gate-phase", type=float, default=0.0)

    args = parser.parse_args()
    if args.rdc0_input is None and args.input_file is None:
        raise ValueError("Provide a TBN input_file or --rdc0-input")

    windowed_export = args.integration_time is not None or args.hop_time is not None
    delete_rdc0 = False
    if args.rdc0_input:
        rdc0_path = os.path.expanduser(args.rdc0_input)
        loaded = load_rdc0_hdf5_metadata(rdc0_path)
        rdc0_shape = loaded["shape"]
        rdc0_metadata = loaded["metadata"]
        fs = float(rdc0_metadata["sample_rate_hz"])
        if args.sweep_frequency is None:
            args.sweep_frequency = float(rdc0_metadata["sweep_frequency_hz"])
        cache_duration = rdc0_shape[1] / float(args.sweep_frequency)
        if args.tstart is None:
            args.tstart = 0.0
        if args.tend is None:
            args.tend = cache_duration
        if args.duration is not None:
            args.tend = args.tstart + args.duration
        print(f"Using RDC0 cache duration {cache_duration:.3f} s; times are relative to cache start.", flush=True)
    else:
        if args.sweep_frequency is None:
            raise ValueError("--sweep-frequency is required when reading raw TBN input")
        idf = lsl_open_tbn(args.input_file)
        lsl_print_metadata(idf)
        fs = float(idf.get_info("sample_rate"))
        set_default_tbn_time_bounds(args, idf)
        if args.duration is not None:
            args.tend = args.tstart + args.duration
        rdc0_path = make_temp_rdc0(args, fs)
        loaded = load_rdc0_hdf5_metadata(rdc0_path)
        rdc0_shape = loaded["shape"]
        rdc0_metadata = loaded["metadata"]
        delete_rdc0 = True

    if args.tend <= args.tstart:
        raise ValueError("--tend must be greater than --tstart")

    # TBN extraction has already applied the requested start time. Cache reads
    # instead index directly into the existing RDC0; never apply TBN tstart twice.
    selection_start = int(round(args.tstart * args.sweep_frequency)) if args.rdc0_input else 0
    selection_stop = (int(round(args.tend * args.sweep_frequency)) if args.rdc0_input
                      else rdc0_shape[1])
    if selection_start < 0 or selection_stop > rdc0_shape[1] or selection_stop <= selection_start:
        raise ValueError("Selected time range is outside the RDC0 cache")
    selected_pulses = selection_stop - selection_start
    selected_span = selected_pulses / args.sweep_frequency
    time_origin = (selection_start / args.sweep_frequency if args.rdc0_input else
                   args.tstart + float(rdc0_metadata.get("start_offset_samples", 0)) / fs)
    if args.integration_time is None:
        args.integration_time = selected_span
    if args.hop_time is None:
        args.hop_time = args.integration_time
    if not np.isfinite(args.integration_time) or not np.isfinite(args.hop_time) or args.integration_time <= 0 or args.hop_time <= 0:
        raise ValueError("--integration-time and --hop-time must be finite and > 0")
    pulse_count = int(round(args.integration_time * args.sweep_frequency))
    hop_pulses = int(round(args.hop_time * args.sweep_frequency))
    if pulse_count < 1 or hop_pulses < 1 or pulse_count > selected_pulses:
        raise ValueError("Integration/hop must contain complete sweeps within the selected span")
    args.integration_time = pulse_count / args.sweep_frequency
    args.hop_time = hop_pulses / args.sweep_frequency

    lfm_config = LFMWaveform(
        sample_rate=fs,
        sweep_frequency=args.sweep_frequency,
        bandwidth=args.bandwidth,
        reference_gate_frequency=reference_gate_frequency_from_args(args),
        reference_gate_duty=args.reference_gate_duty,
        reference_gate_phase=args.reference_gate_phase,
    )

    adc_count = int(rdc0_shape[0])
    adc_start = int(rdc0_metadata.get("adc_start", args.adc_start))
    delay_count = infer_delay_bin_count(lfm_config, adc_count, args.d_min, args.d_max)
    pulse_count = int(round(args.integration_time * args.sweep_frequency))
    if pulse_count < 1:
        raise ValueError("--integration-time is shorter than one sweep")
    nfft_doppler = args.nfft_doppler
    if nfft_doppler is None:
        nfft_doppler = 1 << int(np.ceil(np.log2(max(pulse_count, 1))))
    fd_hz_full = np.fft.fftshift(np.fft.fftfreq(nfft_doppler, d=1.0 / float(args.sweep_frequency)))
    fd_mask = np.ones(fd_hz_full.shape, dtype=bool)
    if args.fd_min is not None:
        fd_mask &= fd_hz_full >= args.fd_min
    if args.fd_max is not None:
        fd_mask &= fd_hz_full <= args.fd_max
    if not np.any(fd_mask):
        raise ValueError("No Doppler bins selected; check --fd-min/--fd-max")
    doppler_count = int(np.count_nonzero(fd_mask))

    frame_count = (selected_pulses - pulse_count) // hop_pulses + 1
    frame_tstart = time_origin + np.arange(frame_count, dtype=float) * args.hop_time
    frame_tend = frame_tstart + args.integration_time

    stands = parse_stands(args.stands, max_stand=args.max_stand, drop_bad=False)
    channel_indices, channel_stands, channel_pols = selected_rdc0_channels(
        loaded["channel_stands"],
        loaded["channel_pols"],
        stands,
        args.pol,
        drop_bad=args.drop_bad_stands,
    )
    n_channels = int(len(channel_indices))
    expected_bytes = estimate_rdc2_bytes(frame_count, delay_count, doppler_count, n_channels, args.dtype)
    storage_shape = ((frame_count, delay_count, doppler_count, n_channels) if windowed_export
                     else (delay_count, doppler_count, n_channels))
    rdc1_bytes = 0 if args.no_store_rdc1 else 2 * delay_count * selected_pulses * n_channels * np.dtype(args.dtype).itemsize
    print(f"RDC2 shape will be {storage_shape}", flush=True)
    print(f"Retained RDC1 uncompressed size: {format_bytes(rdc1_bytes)}", flush=True)
    print(f"Uncompressed real/imag size: {format_bytes(expected_bytes)}", flush=True)
    print(f"Channels include {len(set(channel_stands.tolist()))} stand(s), pol stream(s): {sorted(set(channel_pols.tolist()))}", flush=True)
    if args.dry_run_size:
        if delete_rdc0:
            try:
                os.remove(rdc0_path)
            except OSError:
                pass
        return

    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required for RDC2 export") from exc

    output_file = os.path.expanduser(args.output)
    os.makedirs(os.path.dirname(os.path.abspath(output_file)) or ".", exist_ok=True)
    tmp_output = output_file + ".tmp"
    out_dtype = np.dtype(args.dtype)
    compression_arg, opts = compression_args(args.compression, args.compression_level)
    chunks = (min(delay_count, 32), min(doppler_count, 256), min(n_channels, max(1, int(args.channel_chunk_size))))
    if windowed_export:
        chunks = (1,) + chunks
    nonfinite_rdc0_values_filled = 0

    try:
        with h5py.File(tmp_output, "w") as h5:
            grp = h5.create_group("rdc2")
            dset_real = grp.create_dataset(
                "real",
                shape=storage_shape,
                dtype=out_dtype,
                chunks=chunks,
                compression=compression_arg,
                compression_opts=opts,
            )
            dset_imag = grp.create_dataset(
                "imag",
                shape=storage_shape,
                dtype=out_dtype,
                chunks=chunks,
                compression=compression_arg,
                compression_opts=opts,
            )

            rdc1_group = None
            if not args.no_store_rdc1:
                rdc1_group = h5.create_group("rdc1")
                for component in ("real", "imag"):
                    rdc1_group.create_dataset(
                        component, shape=(delay_count, selected_pulses, n_channels),
                        dtype=out_dtype, chunks=(min(delay_count, 32), min(selected_pulses, 256), chunks[-1]),
                        compression=compression_arg, compression_opts=opts,
                    )
                rdc1_group.attrs["tstart_s"] = time_origin
                rdc1_group.attrs["sweep_frequency_hz"] = args.sweep_frequency

            delay_ms = None
            doppler_hz = fd_hz_full[fd_mask]
            for chunk_start, chunk_stop, rdc0_chunk in iter_rdc0_hdf5_channel_chunks(
                rdc0_path, args.channel_chunk_size, channel_indices=channel_indices,
                pulse_start=selection_start, pulse_stop=selection_stop,
            ):
                if args.gap_fill == "zero":
                    finite = np.isfinite(rdc0_chunk.real) & np.isfinite(rdc0_chunk.imag)
                    nonfinite_rdc0_values_filled += int(finite.size - np.count_nonzero(finite))
                    rdc0_chunk = np.nan_to_num(rdc0_chunk, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                rdc1_band, delay_ms = dechirp_range_fft_delay_band_rdc0(
                    rdc0_chunk, lfm_config, args.d_min, args.d_max,
                    window=None if args.dechirp_window == "none" else args.dechirp_window,
                    adc_start=adc_start,
                )
                del rdc0_chunk
                if rdc1_group is not None:
                    rdc1_group["real"][:, :, chunk_start:chunk_stop] = rdc1_band.real.astype(out_dtype)
                    rdc1_group["imag"][:, :, chunk_start:chunk_stop] = rdc1_band.imag.astype(out_dtype)
                for frame_idx in range(frame_count):
                    start = frame_idx * hop_pulses
                    cube, _ = doppler_fft_rdc1(
                        rdc1_band[:, start:start + pulse_count, :], args.sweep_frequency,
                        window=None if args.slow_window == "none" else args.slow_window,
                        nfft=nfft_doppler,
                    )
                    cube = cube[:, fd_mask, :]
                    selection = (slice(None), slice(None), slice(chunk_start, chunk_stop))
                    if windowed_export:
                        selection = (frame_idx,) + selection
                    dset_real[selection] = cube.real.astype(out_dtype)
                    dset_imag[selection] = cube.imag.astype(out_dtype)
                print(f"Exported channels {chunk_start + 1}-{chunk_stop}/{n_channels}", flush=True)
                del rdc1_band, cube
            if rdc1_group is not None:
                for name, values in (("delay_ms", delay_ms), ("channel_stands", channel_stands), ("channel_pols", channel_pols)):
                    rdc1_group.create_dataset(name, data=values)

            grp.create_dataset("delay_ms", data=delay_ms)
            grp.create_dataset("doppler_hz", data=doppler_hz)
            grp.create_dataset("frame_tstart_s", data=frame_tstart)
            grp.create_dataset("frame_tend_s", data=frame_tend)
            grp.create_dataset("channel_stands", data=channel_stands)
            grp.create_dataset("channel_pols", data=channel_pols)
            h5.attrs["metadata_json"] = metadata_json(
                rdc_stage=2,
                cache_version=2,
                retains_rdc1=not args.no_store_rdc1,
                storage_shape=list(storage_shape),
                station=rdc0_metadata.get("station", "lwasv"),
                channel_id_kind=rdc0_metadata.get("channel_id_kind", "legacy_input_pair"),
                channel_digitizers=([int(rdc0_metadata["channel_digitizers"][i]) for i in channel_indices]
                                    if rdc0_metadata.get("channel_digitizers") is not None else None),
                cable_phase_corrected=rdc0_metadata.get("cable_phase_corrected", False),
                synthetic=rdc0_metadata.get("synthetic", False),
                cable_delays_injected=rdc0_metadata.get("cable_delays_injected", False),
                source_file=os.path.basename(args.input_file) if args.input_file else None,
                source_rdc0=os.path.basename(args.rdc0_input) if args.rdc0_input else None,
                shape_frames_delay_doppler_channels=[int(frame_count), int(delay_count), int(doppler_count), int(n_channels)],
                sample_rate_hz=float(fs),
                tuning_frequency_hz=rdc0_metadata.get("tuning_frequency_hz"),
                sweep_frequency_hz=float(args.sweep_frequency),
                bandwidth_hz=float(args.bandwidth),
                tstart=float(time_origin),
                tend=float(time_origin + selected_span),
                integration_time_s=float(args.integration_time),
                hop_time_s=float(args.hop_time),
                d_min_ms=float(args.d_min),
                d_max_ms=float(args.d_max),
                fd_min_hz=args.fd_min,
                fd_max_hz=args.fd_max,
                dechirp_window=str(args.dechirp_window),
                slow_window=str(args.slow_window),
                nfft_doppler=int(nfft_doppler),
                adc_start=int(adc_start),
                adc_count=int(adc_count),
                gap_fill=str(args.gap_fill),
                nonfinite_rdc0_values_filled=int(nonfinite_rdc0_values_filled),
                selected_stands=[int(s) for s in channel_stands],
                stands=[int(s) for s in stands],
                pol=str(args.pol),
                drop_bad_stands=bool(args.drop_bad_stands),
                n_channels=int(n_channels),
                n_stands=int(len(np.unique(channel_stands))),
                reference_gate_frequency=reference_gate_frequency_from_args(args),
                reference_gate_duty=float(args.reference_gate_duty),
                reference_gate_phase=float(args.reference_gate_phase),
                **polarization_metadata(channel_pols),
            )
        os.replace(tmp_output, output_file)
    finally:
        if os.path.exists(tmp_output):
            try:
                os.remove(tmp_output)
            except OSError:
                pass
        if delete_rdc0:
            try:
                os.remove(rdc0_path)
            except OSError:
                pass

    print(f"Wrote {output_file}", flush=True)
    print(f"Actual RDC2 shape: {storage_shape}", flush=True)
    print(f"Actual uncompressed size: {format_bytes(expected_bytes + rdc1_bytes)}", flush=True)
    if nonfinite_rdc0_values_filled:
        print(f"Replaced {nonfinite_rdc0_values_filled} nonfinite RDC0 values with zeros before processing", flush=True)


if __name__ == "__main__":
    main()
