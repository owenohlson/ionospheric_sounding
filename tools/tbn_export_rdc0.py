# tbn_export_rdc0.py

import argparse

from lfm_utils import LFMWaveform
from plotting_utils import _timestamp_sweep_offset_samples
from rdc_utils import (
    BAD_LWA_STANDS,
    channel_selection,
    estimate_rdc0_bytes,
    export_tbn_rdc0_hdf5,
    format_bytes,
    parse_stands,
)
from tbn_utils import lsl_open_tbn, lsl_print_metadata, set_default_tbn_time_bounds


def probe_start_timestamp(filename, tstart):
    idf = lsl_open_tbn(filename)
    idf.offset(float(tstart))
    readT, timestamp, data = idf.read(0.001)
    del data
    return timestamp


def main():
    parser = argparse.ArgumentParser(
        description="Export TBN data as RDC0 raw radar data cube: (ADC, pulses, channels).",
        formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, max_help_position=35),
    )
    parser.add_argument("input_file", help="Input TBN file")
    parser.add_argument("--output", required=True, help="Output HDF5 file readable from MATLAB/Python")

    parser.add_argument("--tstart", type=float, default=None)
    parser.add_argument("--tend", type=float, default=None)
    parser.add_argument("--duration", type=float, default=None,
                        help="Duration to export in seconds; overrides --tend if provided")
    parser.add_argument("--sweep-frequency", type=float, required=True,
                        help="Sweep repetition rate / PRF used to reshape samples into pulses")
    parser.add_argument("--offset", type=float, default=0.0,
                        help="Timestamp mode: sweep start offset in seconds after each integer-second boundary")
    parser.add_argument("--no-timestamp-align", action="store_true",
                        help="Do not align pulse 0 to timestamp second boundaries")

    parser.add_argument("--stands", default="all",
                        help="Stand list such as all, 1-256, or 1,2,3")
    parser.add_argument("--max-stand", type=int, default=256,
                        help="Largest stand number to include when --stands=all")
    parser.add_argument("-p", "--pol", choices=["both", "x", "y", "0", "1"], default="both",
                        help="Receiver polarization streams to export")
    parser.add_argument("--drop-bad-stands", action="store_true",
                        help=f"Drop known bad stands before export: {BAD_LWA_STANDS}")

    parser.add_argument("--adc-start", type=int, default=0,
                        help="First ADC sample within each sweep to store")
    parser.add_argument("--adc-count", type=int, default=None,
                        help="Number of ADC samples per sweep to store; default stores to end of sweep")
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float32",
                        help="Real/imag storage dtype")
    parser.add_argument("--compression", choices=["gzip", "lzf", "none"], default="gzip")
    parser.add_argument("--compression-level", type=int, default=4,
                        help="gzip compression level")
    parser.add_argument("--gap-fill", choices=["nan", "zero"], default="nan")
    parser.add_argument("--dry-run-size", action="store_true",
                        help="Print expected shape/size and exit without writing")

    args = parser.parse_args()

    idf = lsl_open_tbn(args.input_file)
    lsl_print_metadata(idf)
    fs = float(idf.get_info("sample_rate"))
    set_default_tbn_time_bounds(args, idf)
    if args.duration is not None:
        args.tend = args.tstart + args.duration
    duration = args.tend - args.tstart
    if duration <= 0:
        raise ValueError("Export duration must be > 0")

    sweep_length = int(round(fs / args.sweep_frequency))
    if args.adc_count is None:
        args.adc_count = sweep_length - args.adc_start
    if args.adc_count <= 0 or args.adc_start + args.adc_count > sweep_length:
        raise ValueError("--adc-start/--adc-count must select samples inside one sweep")

    lfm_config = LFMWaveform(
        sample_rate=fs,
        sweep_frequency=args.sweep_frequency,
        bandwidth=1.0,
    )
    start_offset_samples = 0
    if not args.no_timestamp_align:
        start_timestamp = probe_start_timestamp(args.input_file, args.tstart)
        start_offset_samples = _timestamp_sweep_offset_samples(
            start_timestamp,
            lfm_config,
            sweep_offset=args.offset,
        )
        print(f"Timestamp alignment start_offset_samples={start_offset_samples}", flush=True)

    stands = parse_stands(args.stands, max_stand=args.max_stand, drop_bad=args.drop_bad_stands)
    stream_indices, channel_stands, channel_pols = channel_selection(stands, args.pol)
    pulses = int((duration * fs - start_offset_samples) // sweep_length)
    expected_bytes = estimate_rdc0_bytes(args.adc_count, pulses, len(stream_indices), args.dtype)
    print(
        f"RDC0 shape will be (ADC={args.adc_count}, pulses={pulses}, channels={len(stream_indices)})",
        flush=True,
    )
    print(f"Uncompressed real/imag size: {format_bytes(expected_bytes)}", flush=True)
    print(f"Channels include {len(set(channel_stands))} stand(s), pol stream(s): {sorted(set(channel_pols.tolist()))}", flush=True)
    if args.dry_run_size:
        return

    idf = lsl_open_tbn(args.input_file)
    result = export_tbn_rdc0_hdf5(
        idf=idf,
        output_file=args.output,
        tstart=args.tstart,
        duration=duration,
        sweep_frequency=args.sweep_frequency,
        stands=stands,
        pol=args.pol,
        start_offset_samples=start_offset_samples,
        adc_start=args.adc_start,
        adc_count=args.adc_count,
        gap_fill=args.gap_fill,
        dtype=args.dtype,
        compression=args.compression,
        compression_level=args.compression_level,
    )
    print(f"Wrote {result['output_file']}", flush=True)
    print(f"Actual RDC0 shape: {result['shape']}", flush=True)
    print(f"Actual uncompressed size: {format_bytes(result['bytes_uncompressed'])}", flush=True)


if __name__ == "__main__":
    main()
