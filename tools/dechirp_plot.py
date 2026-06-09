# dechirp_plot.py

import argparse
from plotting_utils import plot_dechirp
from lfm_utils import LFMWaveform, load_iq_audio, dechirp_fft_complex


def main():
    parser = argparse.ArgumentParser(description="Plot a range-time profile from a dechirp output.",
                                     formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog,
                                                                                                         max_help_position=35))

    parser.add_argument("input_file", type=str, help="Path to the stereo IQ audio file (.wav, .flac)")
    # LFM waveform parameters
    parser.add_argument("--sweep-frequency", type=float, required=True, help="Sweep repetition rate (Hz)")
    parser.add_argument("--bandwidth", type=float, default=100e3, help="LFM sweep bandwidth in Hz")

    # Optional arguments
    parser.add_argument("--title", type=str, default="Stretch-Processed Range-Time Plot", help="Plot title")
    parser.add_argument("--output", type=str, default=None,
                        help="Output image file path, displays the plot without saving if omitted")
    parser.add_argument("--vmin", type=float, default=None, help="Minimum dB scale (Auto scales if not provided)")
    parser.add_argument("--vmax", type=float, default=None, help="Maximum dB scale (Auto scales if not provided)")
    parser.add_argument("--window", type=float, default=None, help="Sliding window width in seconds")
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

    # Set default tstart/tend if not provided
    if args.tstart is None:
        args.tstart = 0.0
        print(f"No --tstart provided, starting from beginning of file")
    if args.tend is None:
        duration = iq.shape[0] / fs
        args.tend = duration
        print(f"No --tend provided, using end time of file: {args.tend:.2f} seconds")

    start_idx = int(args.tstart * fs)
    end_idx = int(args.tend * fs)

    iq = iq[start_idx:end_idx]

    # Construct LFM waveform config
    lfm_config = LFMWaveform(
        sample_rate=fs,
        sweep_frequency=args.sweep_frequency,
        bandwidth=args.bandwidth,
    )

    dechirp_mag_out, _ = dechirp_fft_complex(iq, lfm_config)

    # Call plotting function
    plot_dechirp(
        stretch_result=dechirp_mag_out,
        lfm_config=lfm_config,
        vmin=args.vmin,
        vmax=args.vmax,
        title=args.title,
        save_path=args.output
    )


if __name__ == "__main__":
    main()
