# tbn_spectrogram.py

import argparse
from tbn_utils import lsl_open_tbn, lsl_print_metadata, lsl_read_block_for_one_stream, lsl_average_spectrum_all_antpols, plot_averaged_spectrum
from plotting_utils import plot_iq_spectrogram


def main():
    parser = argparse.ArgumentParser(
        description="Single-output TBN spectrogram",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Input file
    parser.add_argument("filename", help="TBN file to process")

    # What plot to generate
    parser.add_argument("--plot", choices=["single-stream", "average"], default="single-stream",
                    help="plot a single-stream spectrogram or one averaged spectrum")
    parser.add_argument("-s", "--stand", type=int, default=1, help="stand ID to select for spectrogram")
    parser.add_argument("-p", "--pol", type=str, default="x", help="pol to select (x/y) for spectrogram")

    # Spectrogram controls
    parser.add_argument("--window-size", type=int, default=1024, help="FFT length")
    parser.add_argument("--hop-size", type=int, default=512, help="STFT hop size (spectrogram)")
    parser.add_argument("--vmin", type=float, default=None, help="Minimum dB scale (Auto scales if not provided)")
    parser.add_argument("--vmax", type=float, default=None, help="Maximum dB scale (Auto scales if not provided)")
    parser.add_argument("--tstart", type=float, default=None)
    parser.add_argument("--tend", type=float, default=None)
    parser.add_argument("-w", "--window", choices=["none", "bartlett", "blackman", "hanning", "hann"], default="hann",
                    help="window function for LSL spectrum path")
    parser.add_argument("--title", type=str, default=None, help="Plot title (optional)")
    parser.add_argument("--pfb", action="store_true", help="enable PFB in LSL SpecMaster path")
    parser.add_argument("-q", "--quiet", dest="verbose", action="store_false", help="quiet mode for LSL SpecMaster path")
    parser.set_defaults(verbose=True)

    # Output file
    parser.add_argument("--output", default=None, help="output PNG filename (optional)")

    args = parser.parse_args()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    idf = lsl_open_tbn(args.filename)
    lsl_print_metadata(idf)

    fs = float(idf.get_info("sample_rate"))
    fc = float(idf.get_info("freq1"))

    if args.tstart is None:
        args.tstart = 0.0
        print(f"No --tstart provided, starting from beginning of file")
    if args.tend is None:
        nFramesFile = idf.get_info("nframe")
        args.tend = nFramesFile / fs
        print(f"No --tend provided, using end time of file: {args.tend:.2f} seconds")

    duration = args.tend - args.tstart

    if args.plot == "average":
        # One global averaged spectrum across antpols (single output)
        _, freq, spec_avg = lsl_average_spectrum_all_antpols(idf, args)
        plot_averaged_spectrum(freq, spec_avg, center_freq_hz=fc, out_png=args.output)
        return

    x = lsl_read_block_for_one_stream(idf, args.tstart, duration, stand_id=args.stand, pol=args.pol)

    if args.title is None:
        args.title = f"Spectrogram [fs={round(fs/1000, 3)} kHz, stand={args.stand}, pol={args.pol}, offset={round(args.tstart, 3)}s, duration={round(duration, 3)}s]"

    plot_iq_spectrogram(
        iq=x,
        fs=fs,
        plot_title=args.title,
        vmin=args.vmin,
        vmax=args.vmax,
        tstart=None,
        tend=None,
        window=args.window,
        window_size=args.window_size,
        hop_size=args.hop_size,
        output_file=args.output,
    )

if __name__ == "__main__":
    main()