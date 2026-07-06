# tbn_dechirp_plot.py

import argparse

from lfm_utils import (
    LFMWaveform,
    dechirp_fft_complex,
    reference_gate_frequency_from_args,
)
from plotting_utils import (
    _timestamp_sweep_offset_samples,
    plot_dechirp,
    plot_dechirp_streaming,
)
from tbn_utils import (
    lsl_open_tbn,
    lsl_print_metadata,
    lsl_read_block_for_one_stream,
)


def main():
    parser = argparse.ArgumentParser(
        description="Dechirp range-time plot for TBN data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("filename", help="TBN file to process")

    # Antenna selection
    parser.add_argument("-s", "--stand", type=int, default=1, help="stand ID to select")
    parser.add_argument("-p", "--pol", type=str, default="x", help="pol to select (x/y)")

    # LFM waveform parameters
    parser.add_argument("--sweep-frequency", type=float, required=True, help="Sweep repetition rate (Hz)")
    parser.add_argument("--bandwidth", type=float, default=100e3, help="LFM sweep bandwidth in Hz")

    # Plot options
    parser.add_argument("--title", type=str, default=None, help="Plot title")
    parser.add_argument("--output", default=None, help="output PNG filename (optional)")
    parser.add_argument("--vmin", type=float, default=None, help="Minimum dB scale")
    parser.add_argument("--vmax", type=float, default=None, help="Maximum dB scale")
    parser.add_argument("--tstart", type=float, default=None)
    parser.add_argument("--tend", type=float, default=None)
    parser.add_argument("--navg", type=int, default=4, help="Number of sweeps to average before plotting")
    parser.add_argument("--d-min", type=float, default=None, help="Min delay to display (ms)")
    parser.add_argument("--d-max", type=float, default=None, help="Max delay to display (ms)")
    parser.add_argument("--streaming", action="store_true",
                        help="Process one chirp at a time and store only displayed delay bins")

    # Dechirp/reference options
    parser.add_argument("--dechirp-window", type=str, default="hann",
                        choices=["hamming", "hann", "blackman", "cheb60", "cheb80", "cheb100", "cheb120", "none"],
                        help="Fast-time dechirp FFT window")
    parser.add_argument("--offset", type=float, default=0.0,
                        help="Timestamp mode: sweep start offset in seconds after each integer-second boundary")
    parser.add_argument("--no-timestamp-align", action="store_true",
                        help="Do not align dechirp chunks to timestamp second boundaries")
    parser.add_argument("--reference-gate-frequency", type=float, default=None,
                        help="Gate the reference chirp at this frequency in Hz")
    parser.add_argument("--reference-gate-period", type=float, default=None,
                        help="Gate the reference chirp at this period in seconds; e.g. 0.005 for 5 ms")
    parser.add_argument("--reference-gate-duty", type=float, default=0.5,
                        help="Reference gate duty cycle in (0, 1]")
    parser.add_argument("--reference-gate-phase", type=float, default=0.0,
                        help="Reference gate phase/time offset in seconds")

    args = parser.parse_args()

    idf = lsl_open_tbn(args.filename)
    lsl_print_metadata(idf)

    fs = float(idf.get_info("sample_rate"))

    if args.tstart is None:
        args.tstart = 0.0
        print("No --tstart provided, starting from beginning of file")
    if args.tend is None:
        nFramesFile = idf.get_info("nframe")
        args.tend = nFramesFile / fs
        print(f"No --tend provided, using end time of file: {args.tend:.2f} seconds")

    duration = args.tend - args.tstart

    lfm_config = LFMWaveform(
        sample_rate=fs,
        sweep_frequency=args.sweep_frequency,
        bandwidth=args.bandwidth,
        reference_gate_frequency=reference_gate_frequency_from_args(args),
        reference_gate_duty=args.reference_gate_duty,
        reference_gate_phase=args.reference_gate_phase,
    )

    iq, start_timestamp = lsl_read_block_for_one_stream(
        idf,
        args.tstart,
        duration,
        stand_id=args.stand,
        pol=args.pol,
    )

    start_offset_samples = 0
    if not args.no_timestamp_align:
        start_offset_samples = _timestamp_sweep_offset_samples(
            start_timestamp,
            lfm_config,
            sweep_offset=args.offset,
        )

    if args.title is None:
        args.title = (
            f"Dechirp Plot [fs={round(fs/1000, 3)} kHz, "
            f"BW={round(args.bandwidth/1e3, 3)} kHz, "
            f"sweep_freq={round(args.sweep_frequency, 3)} Hz, "
            f"stand={args.stand}, pol={args.pol}]"
        )

    dechirp_window = None if args.dechirp_window == "none" else args.dechirp_window

    if args.streaming or args.d_min is not None or args.d_max is not None:
        plot_dechirp_streaming(
            iq=iq,
            lfm_config=lfm_config,
            vmin=args.vmin,
            vmax=args.vmax,
            title=args.title,
            output_file=args.output,
            navg=args.navg,
            d_min=args.d_min,
            d_max=args.d_max,
            dechirp_window=dechirp_window,
            tstart=args.tstart,
            start_offset_samples=start_offset_samples,
        )
    else:
        dechirp_mag_out, _ = dechirp_fft_complex(
            iq,
            lfm_config,
            window=dechirp_window,
            start_offset_samples=start_offset_samples,
        )

        plot_dechirp(
            stretch_result=dechirp_mag_out,
            lfm_config=lfm_config,
            vmin=args.vmin,
            vmax=args.vmax,
            title=args.title,
            save_path=args.output,
            navg=args.navg,
            tstart=args.tstart,
        )


if __name__ == "__main__":
    main()
