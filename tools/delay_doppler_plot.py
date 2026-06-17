# delay_doppler_plot.py

import argparse

from plotting_utils import delay_doppler_process_window
from lfm_utils import LFMWaveform, load_iq_audio

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
    parser.add_argument("--interactive", type=bool, default=False, 
                        help="Whether to display the plot interactively")

    # MF-only
    parser.add_argument("--window", type=float, default=None, help="MF: fast-time window width (s)")
    parser.add_argument("--window-center", type=float, default=None, help="MF: center time (s) for window")

    # Dechirp-only
    parser.add_argument("--dechirp-window", type=str, default="hann",
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

    iq_chunk = iq[int(args.tstart * fs):int(args.tend * fs)]

    # Construct LFM waveform for pulse compression
    lfm_config = LFMWaveform(
        sample_rate=fs,
        sweep_frequency=args.sweep_frequency,
        bandwidth=args.bandwidth,
    )
            
    # Process and plot delay-Doppler map using the selected method
    delay_doppler_process_window(
        iq_chunk=iq_chunk,
        frame_idx=None,
        args=args,
        lfm_config=lfm_config,
        timestamps=None,
    )

if __name__ == "__main__":
    main()