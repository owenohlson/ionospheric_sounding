# spectrogram.py

from plotting_utils import plot_iq_spectrogram
from lfm_utils import load_iq_audio
import argparse
# import matplotlib
# matplotlib.use('TkAgg')

def main():
    parser = argparse.ArgumentParser(description="Plot a spectrogram from a complex IQ wav file")
    parser.add_argument("input_file", type=str, help="Path to the stereo IQ file")

    # Optional args
    parser.add_argument("--title", type=str, default="IQ Spectrogram", help="Plot title")
    parser.add_argument("--vmin", type=float, default=None, help="Minimum dB scale (Auto scales if not provided)")
    parser.add_argument("--vmax", type=float, default=None, help="Maximum dB scale (Auto scales if not provided)")
    parser.add_argument("--tstart", type=float, default=None)
    parser.add_argument("--tend", type=float, default=None)
    parser.add_argument("--window-size", type=int, default=1024, help="STFT window size in samples")
    parser.add_argument("--hop-size", type=int, default=512, help="STFT hop size in samples")
    parser.add_argument("--output", type=str, default=None)
    
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

    plot_iq_spectrogram(
        iq=iq,
        fs=fs,
        plot_title=args.title,
        vmin=args.vmin,
        vmax=args.vmax,
        tstart=args.tstart,
        tend=args.tend,
        window_size=args.window_size,
        hop_size=args.hop_size,
        output_file=args.output
    )

if __name__ == "__main__":
    main()