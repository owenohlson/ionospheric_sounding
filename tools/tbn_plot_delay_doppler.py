# tbn_plot_delay_doppler.py

import argparse

from lfm_utils import LFMWaveform, lfm_matched_filtering, dechirp_fft_complex, reference_gate_frequency_from_args
from tbn_utils import lsl_open_tbn, lsl_print_metadata, lsl_read_block_for_one_stream
from plotting_utils import plot_delay_doppler_mf, plot_delay_doppler_dechirp


def main():
    parser = argparse.ArgumentParser(
        description="Plot a delay-Doppler (delay in milliseconds) map using MF or dechirp.",
        formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, max_help_position=35),
    )

    parser.add_argument("input_file", type=str, help="Path to the TBN file")

    # Antenna selection
    parser.add_argument("-s", "--stand", type=int, default=1, help="stand ID to select for spectrogram")
    parser.add_argument("-p", "--pol", type=str, default="x", help="pol to select (x/y) for spectrogram")

    # LFM waveform parameters
    parser.add_argument("--sweep-frequency", type=float, required=True, help="Sweep repetition rate / PRF (Hz)")
    parser.add_argument("--bandwidth", type=float, default=100e3, help="Chirp bandwidth (Hz)")

    # Delay processing method
    parser.add_argument("--method", type=str, default="dechirp", choices=["mf", "dechirp"],
                        help="Delay processing method")

    # Plotting parameters
    parser.add_argument("--title", type=str, default="Delay-Doppler Map", help="Plot title")
    parser.add_argument("--output", type=str, default=None, help="Save path (omit to just display)")
    parser.add_argument("--vmin", type=float, default=None)
    parser.add_argument("--vmax", type=float, default=None)
    parser.add_argument("--tstart", type=float, default=None)
    parser.add_argument("--tend", type=float, default=None)

    parser.add_argument("--slow-window", type=str, default="hann",
                        choices=["hann", "hamming", "blackman", "cheb60", "cheb80", "cheb100", "cheb120", "none"])
    parser.add_argument("--nfft-doppler", type=int, default=None)
    parser.add_argument("--fd-max", type=float, default=None)
    parser.add_argument("--fd-min", type=float, default=None)
    parser.add_argument("--d-max", type=float, default=None,
                        help="Max delay to display (ms)")
    parser.add_argument("--d-min", type=float, default=None,
                        help="Min delay to display (ms)")
    parser.add_argument("--offset", type=float, default=0.0,
                        help="Timestamp mode: sweep start offset in seconds after each integer-second boundary")
    parser.add_argument("--interactive", type=bool, default=False, 
                        help="Whether to display each frame interactively")

    # MF-only
    parser.add_argument("--window-width", type=float, default=None, help="MF: fast-time window width (s)")
    parser.add_argument("--window-center", type=float, default=None, help="MF: center time (s) for window")

    # Dechirp-only
    parser.add_argument("--dechirp-window", type=str, default="hann",
                        choices=["hamming", "hann", "blackman", "cheb60", "cheb80", "cheb100", "cheb120", "none"])
    parser.add_argument("--reference-gate-frequency", type=float, default=None,
                        help="Gate the reference chirp at this frequency in Hz")
    parser.add_argument("--reference-gate-period", type=float, default=None,
                        help="Gate the reference chirp at this period in seconds; e.g. 0.005 for 5 ms")
    parser.add_argument("--reference-gate-duty", type=float, default=0.5,
                        help="Reference gate duty cycle in (0, 1]")
    parser.add_argument("--reference-gate-phase", type=float, default=0.0,
                        help="Reference gate phase/time offset in seconds")

    args = parser.parse_args()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Open TBN file and read metadata
    idf = lsl_open_tbn(args.input_file)
    lsl_print_metadata(idf)

    fs = float(idf.get_info("sample_rate"))
    # fc = float(idf.get_info("freq1"))

    if args.tstart is None:
        args.tstart = 0.0
        print(f"No --tstart provided, starting from beginning of file")
    if args.tend is None:
        nFramesFile = idf.get_info("nframe")
        args.tend = nFramesFile / fs
        print(f"No --tend provided, using end time of file: {args.tend:.2f} seconds")

    duration = args.tend - args.tstart

    # Construct LFM waveform for pulse compression
    lfm_config = LFMWaveform(
        sample_rate=fs,
        sweep_frequency=args.sweep_frequency,
        bandwidth=args.bandwidth,
        reference_gate_frequency=reference_gate_frequency_from_args(args),
        reference_gate_duty=args.reference_gate_duty,
        reference_gate_phase=args.reference_gate_phase,
    )

    # Read IQ data for the specified time range and antenna stream
    iq, start_timestamp = lsl_read_block_for_one_stream(idf, args.tstart, duration, stand_id=args.stand, pol=args.pol)

    # Process and plot delay-Doppler map using the selected method
    if args.method == "mf":
        _, _, complex_response = lfm_matched_filtering(iq, lfm_config)

        plot_delay_doppler_mf(
            complex_response=complex_response,
            lfm_config=lfm_config,
            window_width=args.window_width,
            title=args.title + " (MF)",
            output_file=args.output,
            vmin=args.vmin,
            vmax=args.vmax,
            tstart=None,
            tend=None,
            tcenter=args.window_center,
            window_slow=args.slow_window,
            nfft_doppler=args.nfft_doppler,
            fd_max=args.fd_max,
            fd_min=args.fd_min,
            d_max=args.d_max,
            d_min=args.d_min,
            interactive=args.interactive,
        )

    else:
        start_offset_samples = 0
        if start_timestamp is not None:
            frac = start_timestamp.utc_datetime.microsecond / 1e6
            sweep_period = 1.0 / lfm_config.sweep_frequency
            offset_s = (args.offset - frac) % sweep_period
            start_offset_samples = int(round(offset_s * lfm_config.sample_rate))
            if start_offset_samples >= lfm_config.sweep_length:
                start_offset_samples = 0

        _, complex_spectra = dechirp_fft_complex(
            received_signal=iq,
            lfm_config=lfm_config,
            window=args.dechirp_window,
            start_offset_samples=start_offset_samples,
        )

        plot_delay_doppler_dechirp(
            dechirp_spectra=complex_spectra,
            lfm_config=lfm_config,
            title=args.title + " (dechirp)",
            output_file=args.output,
            vmin=args.vmin,
            vmax=args.vmax,
            window_slow=args.slow_window,
            nfft_doppler=args.nfft_doppler,
            fd_max=args.fd_max,
            fd_min=args.fd_min,
            d_max=args.d_max,
            d_min=args.d_min,
            interactive=args.interactive,
            positive_delay_axis=start_timestamp is not None,
        )


if __name__ == "__main__":
    main()
