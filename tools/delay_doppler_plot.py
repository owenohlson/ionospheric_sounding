# delay_doppler_plot.py

import argparse
import numpy as np

from plotting_utils import plot_delay_doppler_mf, plot_delay_doppler_dechirp
from lfm_utils import LFMWaveform, load_iq_audio, lfm_matched_filtering, dechirp_fft_complex 

import matplotlib
matplotlib.use('TkAgg')


def main():
    parser = argparse.ArgumentParser(
        description="Plot a delay-Doppler (delay in milliseconds) map using MF or dechirp.",
        formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, max_help_position=35),
    )

    parser.add_argument("input_file", type=str, help="Path to the stereo IQ audio file (.wav, .flac)")
    parser.add_argument("--sweep-frequency", type=float, required=True, help="Sweep repetition rate / PRF (Hz)")
    parser.add_argument("--bandwidth", type=float, default=100e3, help="Chirp bandwidth (Hz)")

    parser.add_argument("--method", type=str, default="mf", choices=["mf", "dechirp"],
                        help="Delay processing method")

    parser.add_argument("--title", type=str, default="Delay-Doppler Map", help="Plot title")
    parser.add_argument("--output", type=str, default=None, help="Save path (omit to just display)")
    parser.add_argument("--vmin", type=float, default=None)
    parser.add_argument("--vmax", type=float, default=None)
    parser.add_argument("--tstart", type=float, default=None)
    parser.add_argument("--tend", type=float, default=None)

    parser.add_argument("--slow-window", type=str, default="hann",
                        choices=["hann", "hanning", "hamming", "blackman", "none"])
    parser.add_argument("--nfft-doppler", type=int, default=None)
    parser.add_argument("--fd-max", type=float, default=None)
    parser.add_argument("--fd-min", type=float, default=None)
    parser.add_argument("--d-max", type=float, default=None,
                        help="Max delay to display (ms)")
    parser.add_argument("--d-min", type=float, default=None,
                        help="Min delay to display (ms)")
    # parser.add_argument("--tx-velocity", type=float, default=None,
    #                     help="Transmitter velocity (km/s)")
    # parser.add_argument("--tx-altitude", type=float, default=None,
    #                     help="Transmitter altitude (km)")
    # parser.add_argument("--fc", type=float, default=29e6,
    #                     help="Carrier frequency (Hz) for velocity compensation")
    # parser.add_argument("--time-offset", type=float, default=0,
    #                     help="Time offset (s) to apply to the delay model, to align with the closest approach at t=0")
    # parser.add_argument("--motion-compensation", action="store_true")

    # MF-only
    parser.add_argument("--window", type=float, default=None, help="MF: fast-time window width (s)")
    parser.add_argument("--window-center", type=float, default=None, help="MF: center time (s) for window")

    # Dechirp-only
    parser.add_argument("--dechirp-window", type=str, default="hamming",
                        choices=["hamming", "hann", "none"])

    args = parser.parse_args()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Load IQ data
    iq, fs = load_iq_audio(args.input_file)

    # Set default tstart/tend if not provided
    if args.tstart is None:
        args.tstart = 0.0
        print(f"No --tstart provided, starting from beginning of file")
    if args.tend is None:
        duration = iq.shape[0] / fs
        args.tend = duration
        print(f"No --tend provided, using end time of file: {args.tend:.2f} seconds")

    # Construct LFM waveform for pulse compression
    lfm_config = LFMWaveform(
        sample_rate=fs,
        sweep_frequency=args.sweep_frequency,
        bandwidth=args.bandwidth,
    )
            
    # Process and plot delay-Doppler map using the selected method
    if args.method == "mf":
        _, _, complex_response = lfm_matched_filtering(iq, lfm_config)

        plot_delay_doppler_mf(
            complex_response=complex_response,
            lfm_config=lfm_config,
            window_width=args.window,
            title=args.title + " (MF)",
            output_file=args.output,
            vmin=args.vmin,
            vmax=args.vmax,
            tstart=args.tstart,
            tend=args.tend,
            tcenter=args.window_center,
            window_slow=args.slow_window,
            nfft_doppler=args.nfft_doppler,
            fd_max=args.fd_max,
            fd_min=args.fd_min,
            d_max=args.d_max,
            d_min=args.d_min,
        )

    else:
        _, complex_spectra = dechirp_fft_complex(
            received_signal=iq,
            lfm_config=lfm_config,
            window=None if args.dechirp_window == "none" else args.dechirp_window,
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
        )


if __name__ == "__main__":
    main()