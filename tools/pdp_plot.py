# pdp_plot.py

import argparse
from plotting_utils import plot_pdp
from lfm_utils import LFMWaveform, load_iq_audio, lfm_matched_filtering


def main():
    parser = argparse.ArgumentParser(description="Plot a power delay profile from a matched filter magnitude response.",
                                     formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog,
                                                                                                         max_help_position=35))

    parser.add_argument("input_file", type=str, help="Path to the stereo IQ audio file (.wav, .flac)")
    # LFM waveform parameters
    parser.add_argument("--sweep-frequency", type=float, required=True, help="Sweep repetition rate (Hz)")
    parser.add_argument("--bandwidth", type=float, default=100e3, help="LFM sweep bandwidth in Hz")

    # Optional arguments
    parser.add_argument("--title", type=str, default="Power Delay Profile", help="Plot title")
    parser.add_argument("--output", type=str, default=None,
                        help="Output image file path, displays the plot without saving if omitted")
    parser.add_argument("--vmin", type=float, default=None, help="Minimum dB scale (Auto scales if not provided)")
    parser.add_argument("--vmax", type=float, default=None, help="Maximum dB scale (Auto scales if not provided)")
    parser.add_argument("--window", type=float, default=5e-3, help="Sliding window width in seconds")
    parser.add_argument("--tstart", type=float, default=None,
                        help="Trim start time in seconds, plots the full signal if omitted")
    parser.add_argument("--tend", type=float, default=None,
                        help="Trim end time in seconds, plots the full signal if omitted")
    parser.add_argument("--navg", type=int, default=4, help="Number of sweeps to average before plotting")
    parser.add_argument("--window-center", type=float, default=None,
                        help="The time to center the window around each sweep, automatically calculates if omitted")


    args = parser.parse_args()

    # Load IQ data
    iq, fs = load_iq_audio(args.input_file)

    # Construct LFM waveform config
    lfm_config = LFMWaveform(
        sample_rate=fs,
        sweep_frequency=args.sweep_frequency,
        bandwidth=args.bandwidth,
    )

    magnitude_response, _, _ = lfm_matched_filtering(iq, lfm_config)

    # Call plotting function
    plot_pdp(
        magnitude_response=magnitude_response,
        lfm_config=lfm_config,
        window_width=args.window,
        title=args.title,
        output_file=args.output,
        vmin=args.vmin,
        vmax=args.vmax,
        tstart=args.tstart,
        tend=args.tend,
        navg=args.navg,
        tcenter=args.window_center,
    )


if __name__ == "__main__":
    main()
