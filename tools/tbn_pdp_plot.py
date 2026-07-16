# tbn_pdp_plot.py

import argparse
from tbn_utils import lsl_open_tbn, lsl_print_metadata, lsl_read_block_for_one_stream, timestamp_range_note
from plotting_utils import _timestamp_fractional_second, plot_pdp
from lfm_utils import LFMWaveform, lfm_matched_filtering, reference_gate_frequency_from_args


def main():
    parser = argparse.ArgumentParser(
        description="PDP plot for TBN data",
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

    # PDP optional arguments
    parser.add_argument("--title", type=str, default=None, help="Plot title")
    parser.add_argument("--vmin", type=float, default=None, help="Minimum dB scale (Auto scales if not provided)")
    parser.add_argument("--vmax", type=float, default=None, help="Maximum dB scale (Auto scales if not provided)")
    parser.add_argument("--window", type=float, default=5e-3, help="Sliding window width in seconds")
    parser.add_argument("--tstart", type=float, default=None)
    parser.add_argument("--tend", type=float, default=None)
    parser.add_argument("--navg", type=int, default=4, help="Number of sweeps to average before plotting")
    parser.add_argument("--window-center", type=float, default=None, help="The time to center the window around each sweep, automatically calculates if omitted")
    parser.add_argument("--reference-gate-frequency", type=float, default=None,
                        help="Gate the reference chirp at this frequency in Hz")
    parser.add_argument("--reference-gate-period", type=float, default=None,
                        help="Gate the reference chirp at this period in seconds; e.g. 0.005 for 5 ms")
    parser.add_argument("--reference-gate-duty", type=float, default=0.5,
                        help="Reference gate duty cycle in (0, 1]")
    parser.add_argument("--reference-gate-phase", type=float, default=0.0,
                        help="Reference gate phase/time offset in seconds")

    # Output file
    parser.add_argument("--output", default=None, help="output PNG filename (optional)")

    args = parser.parse_args()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

    # Construct LFM waveform config
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
    corner_note = timestamp_range_note(start_timestamp, len(iq) / fs)

    # Compute matched filter on the requested time span.
    magnitude_response, _, _ = lfm_matched_filtering(iq, lfm_config)

    if args.title is None:
        args.title = f"PDP Plot [fs={round(fs/1000, 3)} kHz, BW={round(args.bandwidth/1e3, 3)} kHz, sweep_freq={round(args.sweep_frequency, 3)} Hz, stand={args.stand}, pol={args.pol}]"

    if args.window_center is None:
        delay_zero = "Relative delay 0 ms = auto matched-filter peak/window center"
    else:
        delay_zero = f"Relative delay 0 ms = --window-center {args.window_center * 1e3:.3f} ms within each matched-filter sweep"

    if start_timestamp is None:
        delay_reference_note = f"{delay_zero}; TBN read timestamp unavailable, so integer-second offset is unknown."
    else:
        frac_s = _timestamp_fractional_second(start_timestamp)
        delay_reference_note = (
            f"{delay_zero}; first read sample is {frac_s * 1e3:.3f} ms "
            "after a GPS integer second."
        )
    print(delay_reference_note)

    # Call plotting function
    plot_pdp(
        magnitude_response=magnitude_response,
        lfm_config=lfm_config,
        window_width=args.window,
        title=args.title,
        output_file=args.output,
        vmin=args.vmin,
        vmax=args.vmax,
        tstart=None,
        tend=None,
        navg=args.navg,
        tcenter=args.window_center,
        delay_reference_note=delay_reference_note,
        corner_note=corner_note,
    )

if __name__ == "__main__":
    main()
