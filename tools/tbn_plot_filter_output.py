# tbn_plot_filter_output.py

import argparse
from tbn_utils import lsl_open_tbn, lsl_print_metadata, lsl_read_block_for_one_stream, timestamp_range_note
from plotting_utils import plot_matched_filter_output
from lfm_utils import LFMWaveform, lfm_matched_filtering


def main():
    parser = argparse.ArgumentParser(
        description="MF output plot for TBN data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Input file
    parser.add_argument("filename", help="TBN file to process")

    # Antenna selection
    parser.add_argument("-s", "--stand", type=int, default=1, help="stand ID to select for spectrogram")
    parser.add_argument("-p", "--pol", type=str, default="x", help="pol to select (x/y) for spectrogram")
    
    # LFM waveform parameters
    parser.add_argument("--sweep-frequency", type=float, required=True, help="Sweep repetition rate (Hz)")
    parser.add_argument("--bandwidth", type=float, default=100e3, help="LFM sweep bandwidth in Hz")

    # Plotting optional arguments
    parser.add_argument("--title", type=str, default=None, help="Plot title")
    parser.add_argument("--tstart", type=float, default=None)
    parser.add_argument("--tend", type=float, default=None)
    parser.add_argument("--units", choices=["s", "ms"], default="s", help="Time units on x-axis")

    # Output file
    parser.add_argument("--output", default=None, help="output PNG filename (optional)")

    args = parser.parse_args()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    idf = lsl_open_tbn(args.filename)
    lsl_print_metadata(idf)

    fs = float(idf.get_info("sample_rate"))
    center_freq = float(idf.get_info("freq1"))

    if args.tstart is None:
        args.tstart = 0.0
        print(f"No --tstart provided, starting from beginning of file")
    if args.tend is None:
        nFramesFile = idf.get_info("nframe")
        args.tend = nFramesFile / fs
        print(f"No --tend provided, using end time of file: {args.tend:.2f} seconds")

    duration = args.tend - args.tstart

    # Construct LFM waveform config
    lfm_config = LFMWaveform(
        sample_rate=fs,
        sweep_frequency=args.sweep_frequency,
        bandwidth=args.bandwidth,
    )

    x, start_timestamp = lsl_read_block_for_one_stream(
        idf,
        args.tstart,
        duration,
        stand_id=args.stand,
        pol=args.pol,
    )
    corner_note = timestamp_range_note(start_timestamp, len(x) / fs)

    # Compute matched filter
    magnitude_response, lags, _ = lfm_matched_filtering(x, lfm_config)

    if args.title is None:
        args.title = f"MF Output Plot [fs={round(fs/1000, 3)} kHz, fc={round(center_freq/1e6, 3)} MHz, stand={args.stand}, pol={args.pol}, BW={round(args.bandwidth/1e3, 3)} kHz, sweep_freq={round(args.sweep_frequency, 3)} Hz]"

    # Plot matched filter output
    plot_matched_filter_output(
        lags=lags,
        magnitude_response=magnitude_response,
        fs=fs,
        title=args.title,
        output_file=args.output,
        time_units=args.units,
        corner_note=corner_note,
    )

if __name__ == "__main__":
    main()
