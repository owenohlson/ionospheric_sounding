# plot_filter_output.py

import argparse
from plotting_utils import plot_matched_filter_output
from lfm_utils import load_iq_audio, lfm_matched_filtering, LFMWaveform


def main():
    parser = argparse.ArgumentParser(description="Compute and plot matched filter output from an IQ recording.")

    parser.add_argument("input_file", type=str, help="Path to the stereo IQ audio file (.wav, .flac)")
    parser.add_argument("--title", type=str, default="Matched Filter Output", help="Plot title")
    parser.add_argument("--output", type=str, default=None, help="Output image file path (e.g., output.png), displays the plot without saving if omitted")
    parser.add_argument("--units", choices=["s", "ms"], default="s", help="Time units on x-axis")

    # LFM waveform parameters
    parser.add_argument("--bandwidth", type=float, required=True, help="LFM sweep bandwidth in Hz")
    parser.add_argument("--sweep-frequency", type=float, required=True, help="Sweep repetition rate in Hz")
    parser.add_argument("--duration", type=float, default=1, help="The total duration of the signal that should be plotted in seconds")
    parser.add_argument("--offset", type=float, default=0, help="Offset time of the plot start time from the recording start time in seconds")

    args = parser.parse_args()

    # Load IQ audio file
    iq, fs = load_iq_audio(args.input_file)
    iq = iq[int(args.offset*fs):int((args.offset + args.duration + 1/args.sweep_frequency)*fs)]

    # Construct LFM waveform config
    lfm_config = LFMWaveform(
        sample_rate=fs,
        bandwidth=args.bandwidth,
        sweep_frequency=args.sweep_frequency,
    )

    # Compute matched filter
    magnitude_response, lags, _ = lfm_matched_filtering(iq, lfm_config)

    # Plot matched filter output
    plot_matched_filter_output(
        lags=lags,
        magnitude_response=magnitude_response,
        fs=fs,
        title=args.title,
        output_file=args.output,
        time_units=args.units
    )


if __name__ == "__main__":
    main()


