"""Shared RDC0/RDC1/RDC2 export orchestration; numerical methods live in rdc_utils."""

import argparse
import os
import tempfile
from contextlib import ExitStack, closing, contextmanager

import numpy as np

from lfm_utils import LFMWaveform, dechirp_delay_axis_ms, reference_gate_frequency_from_args
from plotting_utils import _timestamp_sweep_offset_samples
from rdc_utils import (
    BAD_LWA_STANDS,
    channel_selection,
    dechirp_range_fft_delay_band_rdc0,
    doppler_fft_rdc1,
    estimate_rdc2_bytes,
    estimate_rdc0_bytes,
    export_tbn_rdc0_hdf5,
    format_bytes,
    iter_rdc0_hdf5_channel_chunks,
    iter_rdc1_hdf5_channel_chunks,
    load_rdc0_hdf5_metadata,
    load_rdc1_hdf5_metadata,
    metadata_json,
    parse_stands,
    polarization_metadata,
)


@contextmanager
def atomic_output(output_file):
    """Replace the destination only after the HDF5 writer has closed successfully."""
    output_file = os.path.expanduser(output_file)
    directory = os.path.dirname(os.path.abspath(output_file))
    os.makedirs(directory, exist_ok=True)
    fd, tmp_output = tempfile.mkstemp(prefix=os.path.basename(output_file) + ".", suffix=".tmp", dir=directory)
    os.close(fd)
    try:
        yield tmp_output
        os.replace(tmp_output, output_file)
    finally:
        if os.path.exists(tmp_output):
            os.remove(tmp_output)


def select_cached_channels(channel_stands, channel_pols, stands, pol, drop_bad=False):
    from rdc_utils import pol_to_indices

    channel_stands = np.asarray(channel_stands, dtype=int)
    channel_pols = np.asarray(channel_pols, dtype=int)
    stand_set = set(int(stand) for stand in stands)
    pol_set = set(int(pol_idx) for pol_idx in pol_to_indices(pol))
    if str(pol) == "both":
        pol_set.add(-1)  # Unspecified polarization (e.g. one WAV IQ receiver).
    mask = np.isin(channel_stands, list(stand_set)) & np.isin(channel_pols, list(pol_set))
    if drop_bad:
        mask &= ~np.isin(channel_stands, np.asarray(BAD_LWA_STANDS, dtype=int))
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        raise ValueError("No cached channels remain after applying --stands/--pol/--drop-bad-stands")
    return indices, channel_stands[indices], channel_pols[indices]


def compression_args(compression, compression_level):
    compression_arg = None if compression == "none" else compression
    opts = None if compression in ("none", "lzf") else int(compression_level)
    return compression_arg, opts


def inherit_settings(args, metadata, processed=False):
    """Keep fixed sampling and already-applied processing truthful in new caches."""
    settings = [("sweep_frequency", "sweep_frequency_hz", None)]
    if processed:
        settings += [("bandwidth", "bandwidth_hz", None),
                     ("dechirp_window", "dechirp_window", None),
                     ("adc_start", "adc_start", None), ("adc_count", "adc_count", None),
                     ("reference_gate_frequency", "reference_gate_frequency", None),
                     ("reference_gate_duty", "reference_gate_duty", None),
                     ("reference_gate_phase", "reference_gate_phase", None)]
        if args.reference_gate_period is not None:
            args.reference_gate_frequency = reference_gate_frequency_from_args(args)
            args.reference_gate_period = None
    for attr, key, default in settings:
        requested = getattr(args, attr)
        stored = metadata.get(key, default)
        if requested is not None:
            same = requested == stored
            if isinstance(requested, (float, int)) and isinstance(stored, (float, int)):
                same = np.isclose(requested, stored, rtol=1e-10, atol=1e-12)
            if not same:
                raise ValueError(f"--{attr.replace('_', '-')} conflicts with cached {key}={stored}; re-export from an earlier stage")
        setattr(args, attr, stored)
    if args.gap_fill is None:
        args.gap_fill = metadata.get("gap_fill", metadata.get("gap_info", {}).get("mode", "nan"))


def processing_defaults(args, stage=2):
    for attr, default in (("adc_start", 0), ("offset", 0.), ("dechirp_window", "hann"),
                          ("reference_gate_duty", .5), ("reference_gate_phase", 0.), ("gap_fill", "nan")):
        if getattr(args, attr) is None:
            setattr(args, attr, default)
    for attr in (("sweep_frequency",) if stage == 0 else ("sweep_frequency", "bandwidth")):
        value = getattr(args, attr)
        if value is None or not np.isfinite(value) or value == 0 or (attr == "sweep_frequency" and value < 0):
            raise ValueError(f"A valid --{attr.replace('_', '-')} is required")


def export_raw_rdc0(args, fs, output_file, *, temporary=False):
    """One TBN extraction path for standalone RDC0 and intermediate RDC0 caches."""
    if args.input_file.lower().endswith(".wav"):
        from wav_rdc import export_wav_rdc0
        if args.dry_run_size and not temporary:
            export_wav_rdc0(args, output_file, temporary=temporary)
            return
        with atomic_output(output_file) as temporary_path:
            export_wav_rdc0(args, temporary_path, temporary=temporary)
        return
    from tbn_utils import lsl_open_tbn, probe_start_timestamp

    duration = args.tend - args.tstart
    sweep_length = int(round(fs / args.sweep_frequency))
    if args.adc_count is None:
        args.adc_count = sweep_length - args.adc_start
    if args.adc_start < 0 or args.adc_count <= 0 or args.adc_start + args.adc_count > sweep_length:
        raise ValueError("--adc-start/--adc-count must select samples inside one sweep")

    lfm_for_offset = LFMWaveform(sample_rate=fs, sweep_frequency=args.sweep_frequency, bandwidth=1.0)
    start_offset_samples = 0
    if not args.no_timestamp_align:
        start_timestamp = probe_start_timestamp(args.input_file, args.tstart)
        start_offset_samples = _timestamp_sweep_offset_samples(
            start_timestamp,
            lfm_for_offset,
            sweep_offset=args.offset,
        )
        print(f"Timestamp alignment start_offset_samples={start_offset_samples}", flush=True)

    stands = parse_stands(args.stands, max_stand=args.max_stand, drop_bad=args.drop_bad_stands)
    indices, channel_stands, channel_pols = channel_selection(stands, args.pol)
    pulses = int((duration * fs - start_offset_samples) // sweep_length)
    if pulses < 1:
        raise ValueError("Requested duration does not contain one complete sweep after alignment")
    dtype = args.rdc0_dtype if temporary else args.dtype
    compression = args.rdc0_compression if temporary else args.compression
    level = args.rdc0_compression_level if temporary else args.compression_level
    print(f"RDC0 shape will be (ADC={args.adc_count}, pulses={pulses}, channels={len(indices)})", flush=True)
    print(f"Uncompressed real/imag size: {format_bytes(estimate_rdc0_bytes(args.adc_count, pulses, len(indices), dtype))}", flush=True)
    print(f"Channels include {len(set(channel_stands))} stand(s), pol stream(s): {sorted(set(channel_pols.tolist()))}", flush=True)
    if args.dry_run_size and not temporary:
        return
    with closing(lsl_open_tbn(args.input_file)) as idf, atomic_output(output_file) as tmp_output:
        result = export_tbn_rdc0_hdf5(
            idf=idf, output_file=tmp_output, tstart=args.tstart, duration=duration,
            sweep_frequency=args.sweep_frequency, stands=stands, pol=args.pol,
            start_offset_samples=start_offset_samples, adc_start=args.adc_start,
            adc_count=args.adc_count, gap_fill=args.gap_fill, dtype=dtype,
            compression=compression, compression_level=level,
        )
        import h5py
        import json
        with h5py.File(tmp_output, "r+") as h5:
            metadata = json.loads(h5.attrs["metadata_json"])
            metadata.update(bandwidth_hz=args.bandwidth, source_file=os.path.basename(args.input_file),
                            timestamp_aligned=not args.no_timestamp_align, sweep_offset_s=args.offset)
            h5.attrs["metadata_json"] = metadata_json(**metadata)
    print(f"Wrote {'temporary ' if temporary else ''}RDC0: {output_file}", flush=True)
    print(f"Actual RDC0 shape: {result['shape']}", flush=True)
    print(f"Actual uncompressed size: {format_bytes(result['bytes_uncompressed'])}", flush=True)


def build_parser(stage):
    descriptions = {
        0: "Export TBN as raw RDC0 (ADC, pulses, channels).",
        1: "Export RDC1 (delay, pulses, channels), before the Doppler FFT.",
        2: "Export RDC2 with optional retained RDC1 for later integration/hop selection.",
    }
    parser = argparse.ArgumentParser(
        description=descriptions[stage],
        formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, max_help_position=35),
    )
    parser.add_argument("input_file", nargs="?", help="Input TBN (.tbn/.dat) or WAV (mono real or stereo I/Q)")
    parser.add_argument("--start-timestamp", help="WAV sample-zero UTC time, ISO 8601; ordinary WAV has no reliable UTC clock")
    parser.add_argument("--carrier-frequency", type=float, help="WAV RF carrier in Hz; unspecified means unknown")
    parser.add_argument("--stage", type=int, choices=[0, 1, 2], default=stage, help="Output stage (fixed for tbn_export_rdcN entry points)")
    if stage > 0:
        parser.add_argument("--rdc0-input", default=None, help="Existing RDC0 HDF5 file to process")
    else:
        parser.set_defaults(rdc0_input=None)
    if stage == 2:
        parser.add_argument("--rdc1-input", "--input-rdc1", default=None,
                            help="Existing RDC1 HDF5 file; inherits its dechirp settings")
    else:
        parser.set_defaults(rdc1_input=None)
    parser.add_argument("--output", required=True, help=f"Output RDC{stage} HDF5 file")

    if stage == 2:
        parser.add_argument("--no-store-rdc1", action="store_true",
                            help="Omit per-sweep data; integration/hop cannot then be changed after export")
    parser.add_argument("--tstart", type=float, default=None, help="Start seconds relative to the input file/cache")
    parser.add_argument("--tend", type=float, default=None, help="End seconds relative to the input file/cache")
    parser.add_argument("--duration", type=float, default=None)
    if stage == 2:
        parser.add_argument("--integration-time", type=float, default=None,
                            help="Seconds per RDC2 frame; default uses the full selected span")
        parser.add_argument("--hop-time", type=float, default=None,
                            help="Seconds between RDC2 frames; default equals integration time")
    else:
        parser.set_defaults(no_store_rdc1=False, integration_time=None, hop_time=None)

    parser.add_argument("--sweep-frequency", type=float, default=None)
    parser.add_argument("--bandwidth", type=float, default=None,
                        help="Chirp bandwidth in Hz; stored only at stage 0, required for range processing")
    parser.add_argument("--offset", type=float, default=None)
    parser.add_argument("--no-timestamp-align", action="store_true")

    parser.add_argument("--stands", default="all")
    parser.add_argument("--max-stand", type=int, default=256)
    parser.add_argument("-p", "--pol", choices=["both", "x", "y", "0", "1"], default="both")
    parser.add_argument("--drop-bad-stands", action="store_true",
                        help=f"Drop known bad stands before export: {BAD_LWA_STANDS}")

    parser.add_argument("--adc-start", type=int, default=None)
    parser.add_argument("--adc-count", type=int, default=None)
    parser.add_argument("--gap-fill", choices=["nan", "zero"], default=None,
                        help="Inherit cache policy, or nan for raw input; zero fills before FFT")
    if stage > 0:
        parser.add_argument("--rdc0-dtype", choices=["float32", "float16"], default="float32")
        parser.add_argument("--rdc0-compression", choices=["gzip", "lzf", "none"], default="gzip")
        parser.add_argument("--rdc0-compression-level", type=int, default=4)
        parser.add_argument("--d-min", type=float, default=None, help="Minimum delay (ms); default keeps all available bins")
        parser.add_argument("--d-max", type=float, default=None, help="Maximum delay (ms); default keeps all available bins")
        parser.add_argument("--dechirp-window", choices=["hamming", "hann", "blackman", "cheb60", "cheb80", "cheb100", "cheb120", "none"], default=None)
    else:
        parser.set_defaults(dechirp_window=None)
    if stage == 2:
        parser.add_argument("--fd-min", type=float, default=None)
        parser.add_argument("--fd-max", type=float, default=None)
        parser.add_argument("--slow-window", choices=["hann", "hamming", "blackman", "cheb60", "cheb80", "cheb100", "cheb120", "none"], default="hann")
        parser.add_argument("--nfft-doppler", type=int, default=None)
    else:
        parser.set_defaults(fd_min=None, fd_max=None, slow_window="hann", nfft_doppler=None)

    parser.add_argument("--dtype", choices=["float32", "float16"], default="float32",
                        help="Real/imag storage dtype")
    parser.add_argument("--compression", choices=["gzip", "lzf", "none"], default="gzip")
    parser.add_argument("--compression-level", type=int, default=4)
    if stage > 0:
        parser.add_argument("--channel-chunk-size", type=int, default=8)
    else:
        parser.set_defaults(channel_chunk_size=8)
    parser.add_argument("--dry-run-size", action="store_true")

    if stage > 0:
        parser.add_argument("--reference-gate-frequency", type=float, default=None)
        parser.add_argument("--reference-gate-period", type=float, default=None)
        parser.add_argument("--reference-gate-duty", type=float, default=None)
        parser.add_argument("--reference-gate-phase", type=float, default=None)
    else:
        parser.set_defaults(reference_gate_duty=None, reference_gate_phase=None)
    return parser


def main(argv=None, stage=None):
    if stage is None:
        selector = argparse.ArgumentParser(add_help=False)
        selector.add_argument("--stage", type=int, choices=[0, 1, 2], default=2)
        stage = selector.parse_known_args(argv)[0].stage
    parser = build_parser(stage)

    args = parser.parse_args(argv)
    if args.stage != stage:
        parser.error(f"This entry point exports RDC{stage}; use rdc_export.py to select a different stage")
    if sum(value is not None for value in (args.input_file, args.rdc0_input, args.rdc1_input)) != 1:
        parser.error("Provide exactly one raw input_file, --rdc0-input, or --rdc1-input")
    if args.channel_chunk_size < 1:
        parser.error("--channel-chunk-size must be positive")
    if not 0 <= args.compression_level <= 9:
        parser.error("--compression-level must be in [0, 9]")
    if stage > 0 and not 0 <= args.rdc0_compression_level <= 9:
        parser.error("--rdc0-compression-level must be in [0, 9]")
    input_path = args.rdc0_input or args.rdc1_input or args.input_file
    if os.path.realpath(os.path.expanduser(input_path)) == os.path.realpath(os.path.expanduser(args.output)):
        parser.error("Output must not overwrite the input file")
    with ExitStack() as resources:
        run_export(args, stage, resources)


def run_export(args, stage, resources):
    windowed_export = args.integration_time is not None or args.hop_time is not None
    cached_input = args.rdc0_input or args.rdc1_input
    if cached_input:
        if args.start_timestamp is not None or args.carrier_frequency is not None:
            raise ValueError("WAV timestamp/carrier overrides belong at ingestion, not on an existing cache")
        if args.offset is not None or args.no_timestamp_align:
            raise ValueError("Cached sweeps are already aligned; --offset/--no-timestamp-align require raw TBN")
        source_path = os.path.expanduser(cached_input)
        loader = load_rdc1_hdf5_metadata if args.rdc1_input else load_rdc0_hdf5_metadata
        loaded = loader(source_path)
        source_shape = loaded["shape"]
        source_metadata = loaded["metadata"]
        fs = float(source_metadata["sample_rate_hz"])
        inherit_settings(args, source_metadata, processed=bool(args.rdc1_input))
        if not args.rdc1_input:
            for attr, stored in (("adc_start", source_metadata.get("adc_start", 0)), ("adc_count", source_shape[0])):
                if getattr(args, attr) is not None and getattr(args, attr) != stored:
                    raise ValueError(f"--{attr.replace('_', '-')} conflicts with RDC0 sample selection")
                setattr(args, attr, stored)
            if args.bandwidth is None:
                args.bandwidth = source_metadata.get("bandwidth_hz")
        processing_defaults(args)
        cache_duration = source_shape[1] / float(args.sweep_frequency)
        if args.tstart is None:
            args.tstart = 0.0
        if args.tend is None:
            args.tend = cache_duration
        if args.duration is not None:
            args.tend = args.tstart + args.duration
        print(f"Using cache duration {cache_duration:.3f} s; selection times are relative to cache start.", flush=True)
    else:
        processing_defaults(args, stage)
        args.input_file = os.path.expanduser(args.input_file)
        if args.input_file.lower().endswith(".wav"):
            import soundfile as sf
            info = sf.info(args.input_file)
            fs = float(info.samplerate)
            args.tstart = 0.0 if args.tstart is None else args.tstart
            args.tend = info.duration if args.tend is None else args.tend
        else:
            if args.start_timestamp is not None or args.carrier_frequency is not None:
                raise ValueError("TBN timing and tuning come from its headers; WAV overrides are not applicable")
            from tbn_utils import lsl_open_tbn, lsl_print_metadata, set_default_tbn_time_bounds
            with closing(lsl_open_tbn(args.input_file)) as idf:
                lsl_print_metadata(idf)
                fs = float(idf.get_info("sample_rate"))
                set_default_tbn_time_bounds(args, idf)
        if args.duration is not None:
            args.tend = args.tstart + args.duration
        if not np.isfinite(args.tstart) or not np.isfinite(args.tend) or args.tstart < 0 or args.tend <= args.tstart:
            raise ValueError("--tend must be greater than --tstart")
        if stage == 0:
            export_raw_rdc0(args, fs, os.path.expanduser(args.output))
            return
        directory = resources.enter_context(tempfile.TemporaryDirectory(prefix="rdc0_for_export_"))
        source_path = os.path.join(directory, "rdc0.h5")
        export_raw_rdc0(args, fs, source_path, temporary=True)
        loaded = load_rdc0_hdf5_metadata(source_path)
        source_shape = loaded["shape"]
        source_metadata = loaded["metadata"]

    if not np.isfinite(args.tstart) or not np.isfinite(args.tend) or args.tstart < 0 or args.tend <= args.tstart:
        raise ValueError("--tend must be greater than --tstart")

    # TBN extraction has already applied the requested start time. Cache reads
    # instead index directly into the existing RDC0; never apply TBN tstart twice.
    selection_start = int(round(args.tstart * args.sweep_frequency)) if cached_input else 0
    selection_stop = (int(round(args.tend * args.sweep_frequency)) if cached_input
                      else source_shape[1])
    if selection_start < 0 or selection_stop > source_shape[1] or selection_stop <= selection_start:
        raise ValueError("Selected time range is outside the input cache")
    selected_pulses = selection_stop - selection_start
    selected_span = selected_pulses / args.sweep_frequency
    time_origin = (selection_start / args.sweep_frequency if cached_input else
                   args.tstart + float(source_metadata.get("start_offset_samples", 0)) / fs)
    if args.rdc1_input:
        time_origin += float(source_metadata.get("tstart", 0.))
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

    adc_count = source_metadata.get("adc_count") if args.rdc1_input else int(source_shape[0])
    adc_start = int(source_metadata.get("adc_start", args.adc_start))
    available_delay = (loaded["delay_ms"] if args.rdc1_input else
                       dechirp_delay_axis_ms(lfm_config, n_bins=adc_count)[0])
    if args.d_min is None:
        args.d_min = float(available_delay[0])
    if args.d_max is None:
        args.d_max = float(available_delay[-1])
    delay_indices = np.flatnonzero((available_delay >= args.d_min) & (available_delay <= args.d_max))
    if not delay_indices.size:
        raise ValueError("No delay bins selected; check --d-min/--d-max")
    delay_selection = slice(int(delay_indices[0]), int(delay_indices[-1]) + 1)
    delay_count = len(delay_indices)
    pulse_count = int(round(args.integration_time * args.sweep_frequency))
    if pulse_count < 1:
        raise ValueError("--integration-time is shorter than one sweep")
    nfft_doppler = args.nfft_doppler
    if nfft_doppler is None:
        nfft_doppler = 1 << int(np.ceil(np.log2(max(pulse_count, 1))))
    if nfft_doppler < pulse_count:
        raise ValueError("--nfft-doppler must be >= the integration sweep count")
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
    channel_indices, channel_stands, channel_pols = select_cached_channels(
        loaded["channel_stands"],
        loaded["channel_pols"],
        stands,
        args.pol,
        drop_bad=args.drop_bad_stands,
    )
    n_channels = int(len(channel_indices))
    expected_bytes = (estimate_rdc2_bytes(frame_count, delay_count, doppler_count, n_channels, args.dtype)
                      if stage == 2 else 0)
    storage_shape = ((frame_count, delay_count, doppler_count, n_channels) if windowed_export
                     else (delay_count, doppler_count, n_channels))
    rdc1_bytes = 0 if args.no_store_rdc1 else 2 * delay_count * selected_pulses * n_channels * np.dtype(args.dtype).itemsize
    if stage == 1:
        storage_shape = (delay_count, selected_pulses, n_channels)
    print(f"RDC{stage} shape will be {storage_shape}", flush=True)
    if stage == 2:
        print(f"Retained RDC1 uncompressed size: {format_bytes(rdc1_bytes)}", flush=True)
    print(f"Total uncompressed real/imag size: {format_bytes(expected_bytes + rdc1_bytes)}", flush=True)
    print(f"Channels include {len(set(channel_stands.tolist()))} stand(s), pol stream(s): {sorted(set(channel_pols.tolist()))}", flush=True)
    if args.dry_run_size:
        return

    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required for RDC export") from exc

    output_file = os.path.expanduser(args.output)
    out_dtype = np.dtype(args.dtype)
    compression_arg, opts = compression_args(args.compression, args.compression_level)
    chunks = (min(delay_count, 32), min(doppler_count if stage == 2 else selected_pulses, 256), min(n_channels, args.channel_chunk_size))
    if stage == 2 and windowed_export:
        chunks = (1,) + chunks
    nonfinite_values_filled = 0

    with atomic_output(output_file) as tmp_output:
        with h5py.File(tmp_output, "w") as h5:
            grp = h5.create_group(f"rdc{stage}")
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
            if stage == 2:
                grp.create_dataset("invalid_counts", (frame_count, n_channels), dtype="int64")
            if stage == 1:
                rdc1_group = grp
            elif not args.no_store_rdc1:
                rdc1_group = h5.create_group("rdc1")
                for component in ("real", "imag"):
                    rdc1_group.create_dataset(
                        component, shape=(delay_count, selected_pulses, n_channels),
                        dtype=out_dtype, chunks=(min(delay_count, 32), min(selected_pulses, 256), chunks[-1]),
                        compression=compression_arg, compression_opts=opts,
                    )
            if rdc1_group is not None:
                rdc1_group.attrs["dimension_order"] = "delay,pulses,channels"
                rdc1_group.attrs["tstart_s"] = time_origin
                rdc1_group.attrs["sweep_frequency_hz"] = args.sweep_frequency
                rdc1_group.create_dataset("invalid_counts", (selected_pulses, n_channels), dtype="int64")

            delay_ms = available_delay[delay_selection]
            doppler_hz = fd_hz_full[fd_mask]
            iterator = iter_rdc1_hdf5_channel_chunks if args.rdc1_input else iter_rdc0_hdf5_channel_chunks
            iterator_options = {"delay_selection": delay_selection} if args.rdc1_input else {}
            for chunk_start, chunk_stop, input_chunk in iterator(
                source_path, args.channel_chunk_size, channel_indices=channel_indices,
                pulse_start=selection_start, pulse_stop=selection_stop,
                **iterator_options,
            ):
                invalid_counts = np.count_nonzero(~np.isfinite(input_chunk), axis=0)
                with h5py.File(source_path, "r") as source_h5:
                    source_group = source_h5["rdc1" if args.rdc1_input else "rdc0"]
                    if "invalid_counts" in source_group:
                        invalid_counts = np.maximum(invalid_counts, source_group["invalid_counts"][
                            selection_start:selection_stop, channel_indices[chunk_start:chunk_stop]])
                if args.gap_fill == "zero":
                    finite = np.isfinite(input_chunk.real) & np.isfinite(input_chunk.imag)
                    nonfinite_values_filled += int(finite.size - np.count_nonzero(finite))
                    input_chunk[~finite] = 0.0
                if args.rdc1_input:
                    rdc1_band = input_chunk
                else:
                    rdc1_band, delay_ms = dechirp_range_fft_delay_band_rdc0(
                        input_chunk, lfm_config, args.d_min, args.d_max,
                        window=None if args.dechirp_window == "none" else args.dechirp_window,
                        adc_start=adc_start,
                    )
                del input_chunk
                if rdc1_group is not None:
                    rdc1_group["invalid_counts"][:, chunk_start:chunk_stop] = invalid_counts
                    rdc1_group["real"][:, :, chunk_start:chunk_stop] = rdc1_band.real.astype(out_dtype)
                    rdc1_group["imag"][:, :, chunk_start:chunk_stop] = rdc1_band.imag.astype(out_dtype)
                for frame_idx in range(frame_count if stage == 2 else 0):
                    start = frame_idx * hop_pulses
                    grp["invalid_counts"][frame_idx, chunk_start:chunk_stop] = invalid_counts[start:start + pulse_count].sum(axis=0)
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
                del rdc1_band
            if rdc1_group is not None:
                for name, values in (("delay_ms", delay_ms), ("channel_stands", channel_stands), ("channel_pols", channel_pols)):
                    rdc1_group.create_dataset(name, data=values)

            if stage == 2:
                grp.create_dataset("delay_ms", data=delay_ms)
                grp.create_dataset("doppler_hz", data=doppler_hz)
                grp.create_dataset("frame_tstart_s", data=frame_tstart)
                grp.create_dataset("frame_tend_s", data=frame_tend)
                grp.create_dataset("channel_stands", data=channel_stands)
                grp.create_dataset("channel_pols", data=channel_pols)
            output_metadata = dict(source_metadata)
            output_metadata.update(
                rdc_stage=stage,
                cache_version=2,
                dimension_order=(["delay", "pulses", "channels"] if stage == 1 else
                                 (["frames"] if windowed_export else []) + ["delay", "doppler", "channels"]),
                retains_rdc1=not args.no_store_rdc1,
                storage_shape=list(storage_shape),
                station=source_metadata.get("station", "lwasv"),
                channel_id_kind=source_metadata.get("channel_id_kind", "legacy_input_pair"),
                channel_digitizers=([int(source_metadata["channel_digitizers"][i]) for i in channel_indices]
                                    if source_metadata.get("channel_digitizers") is not None else None),
                cable_phase_corrected=source_metadata.get("cable_phase_corrected", False),
                synthetic=source_metadata.get("synthetic", False),
                cable_delays_injected=source_metadata.get("cable_delays_injected", False),
                source_file=os.path.basename(args.input_file) if args.input_file else source_metadata.get("source_file"),
                source_rdc0=os.path.basename(args.rdc0_input) if args.rdc0_input else source_metadata.get("source_rdc0"),
                source_rdc1=os.path.basename(args.rdc1_input) if args.rdc1_input else None,
                source_metadata=source_metadata,
                shape_frames_delay_doppler_channels=[int(frame_count), int(delay_count), int(doppler_count), int(n_channels)],
                sample_rate_hz=float(fs),
                tuning_frequency_hz=source_metadata.get("tuning_frequency_hz"),
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
                adc_count=int(adc_count) if adc_count is not None else None,
                gap_fill=str(args.gap_fill),
                nonfinite_rdc0_values_filled=(int(nonfinite_values_filled) if not args.rdc1_input else
                                            source_metadata.get("nonfinite_rdc0_values_filled", 0)),
                nonfinite_rdc1_values_filled=int(nonfinite_values_filled) if args.rdc1_input else 0,
                selected_stands=[int(s) for s in channel_stands],
                stands=[int(s) for s in np.unique(channel_stands)],
                pol=polarization_metadata(channel_pols)["polarization_mode"],
                drop_bad_stands=bool(args.drop_bad_stands or source_metadata.get("drop_bad_stands", False)),
                n_channels=int(n_channels),
                n_stands=(0 if source_metadata.get("channel_id_kind") == "receiver" else int(len(np.unique(channel_stands)))),
                reference_gate_frequency=reference_gate_frequency_from_args(args),
                reference_gate_duty=float(args.reference_gate_duty),
                reference_gate_phase=float(args.reference_gate_phase),
                **polarization_metadata(channel_pols),
            )
            if stage == 1:
                output_metadata["shape_delay_pulses_channels"] = list(storage_shape)
                for key in ("shape_frames_delay_doppler_channels", "integration_time_s", "hop_time_s",
                            "fd_min_hz", "fd_max_hz", "slow_window", "nfft_doppler"):
                    output_metadata.pop(key, None)
            elif rdc1_group is not None:
                output_metadata["shape_delay_pulses_channels"] = [delay_count, selected_pulses, n_channels]
            else:
                output_metadata.pop("shape_delay_pulses_channels", None)
            output_metadata.pop("shape_adc_pulses_channels", None)
            if source_metadata.get("pulse0_utc"):
                from datetime import datetime, timedelta
                output_metadata["pulse0_utc"] = (datetime.fromisoformat(source_metadata["pulse0_utc"]) +
                                                timedelta(seconds=selection_start / args.sweep_frequency)).isoformat()
            output_metadata.update(dtype=args.dtype, compression=args.compression,
                                   compression_level=args.compression_level)
            if "gap_info" in source_metadata and args.gap_fill == "zero":
                output_metadata["gap_info"] = dict(source_metadata["gap_info"], mode="zero")
            h5.attrs["metadata_json"] = metadata_json(**output_metadata)

    print(f"Wrote {output_file}", flush=True)
    print(f"Actual RDC{stage} shape: {storage_shape}", flush=True)
    print(f"Actual uncompressed size: {format_bytes(expected_bytes + rdc1_bytes)}", flush=True)
    if nonfinite_values_filled:
        print(f"Replaced {nonfinite_values_filled} nonfinite input values with zeros before processing", flush=True)
        if args.rdc1_input:
            print("NaN-contaminated range FFTs cannot be recovered here; re-export RDC1 from RDC0 with zero fill to preserve those sweeps.", flush=True)


if __name__ == "__main__":
    main()
