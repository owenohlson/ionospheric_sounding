# tbn_spectrogram.py

import argparse
import os
from tbn_utils import (
    file_timestamp_range_note,
    gap_fill_note,
    lsl_average_spectrum_all_antpols,
    lsl_open_tbn,
    lsl_print_metadata,
    lsl_read_block_for_one_stream,
    plot_averaged_spectrum,
    set_default_tbn_time_bounds,
    timestamp_range_note,
)
from plotting_utils import plot_iq_spectrogram
from plotting_utils import _timestamp_fractional_second


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
    parser.add_argument("--gap-fill", choices=["nan", "zero"], default="nan",
                        help="Fill TBN timetag gaps with NaNs for blank/corrupt regions or zeros for continuity")
    parser.add_argument("-w", "--window", choices=["none", "bartlett", "blackman", "hanning", "hann"], default="hann",
                    help="window function for LSL spectrum path")
    parser.add_argument("--title", type=str, default=None, help="Plot title (optional)")
    parser.add_argument("--pfb", action="store_true", help="enable PFB in LSL SpecMaster path")
    parser.add_argument("-q", "--quiet", dest="verbose", action="store_false", help="quiet mode for LSL SpecMaster path")
    parser.set_defaults(verbose=True)
    parser.add_argument("--estimate-lfm", action="store_true",
                        help="Estimate LFM bandwidth, sweep frequency, gating, and UTC sweep offset from the spectrogram")
    parser.add_argument("--estimate-lfm-overlay", action="store_true",
                        help="Draw the estimated or manually selected LFM sweep as a thin cyan overlay")
    parser.add_argument("--no-estimate-lfm-gate", action="store_true",
                        help="Skip lightweight gate estimation")
    parser.add_argument("--lfm-overlay-sweep-frequency", type=float, default=None,
                        help="Override overlay sweep frequency in Hz")
    parser.add_argument("--lfm-overlay-bandwidth-khz", type=float, default=None,
                        help="Override overlay bandwidth in kHz; sign controls sweep direction")
    parser.add_argument("--lfm-overlay-offset", type=float, default=None,
                        help="Override overlay sweep start offset in seconds")
    parser.add_argument("--lfm-overlay-gate-frequency", type=float, default=None,
                        help="Override displayed gate frequency in Hz")
    parser.add_argument("--lfm-overlay-gate-duty", type=float, default=None,
                        help="Override displayed gate duty as a 0-1 fraction")
    parser.add_argument("--lfm-overlay-gate-phase", type=float, default=None,
                        help="Override displayed gate phase in seconds")

    # Output file
    parser.add_argument("--output", default=None, help="output PNG filename (optional)")

    args = parser.parse_args()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    idf = lsl_open_tbn(args.filename)
    lsl_print_metadata(idf)

    fs = float(idf.get_info("sample_rate"))
    fc = float(idf.get_info("freq1"))

    set_default_tbn_time_bounds(args, idf)

    duration = args.tend - args.tstart

    if args.plot == "average":
        # One global averaged spectrum across antpols (single output)
        _, freq, spec_avg = lsl_average_spectrum_all_antpols(idf, args)
        corner_note = file_timestamp_range_note(idf, args.tstart, args.tend)
        plot_averaged_spectrum(freq, spec_avg, center_freq_hz=fc, out_png=args.output, corner_note=corner_note)
        return

    x, start_timestamp, gap_info = lsl_read_block_for_one_stream(
        idf,
        args.tstart,
        duration,
        stand_id=args.stand,
        pol=args.pol,
        gap_fill=args.gap_fill,
        return_gap_info=True,
    )
    footer_note = gap_fill_note(gap_info)
    corner_note = timestamp_range_note(start_timestamp, len(x) / fs)

    if args.title is None:
        args.title = os.path.basename(args.output) if args.output else "TBN Spectrogram"

    file_start = idf.get_info("start_time")
    info_rows = [
        ("File", os.path.basename(args.filename)),
        ("Sample rate", f"{fs / 1e3:.3f} kHz"),
        ("Tuning freq", f"{fc / 1e6:.6f} MHz"),
        ("Stand/pol", f"{args.stand}/{args.pol}"),
        ("Plot span", f"{args.tstart:.3f}-{args.tend:.3f} s"),
        ("Duration", f"{duration:.3f} s"),
        ("Window", f"{args.window_size} samples"),
        ("Hop", f"{args.hop_size} samples"),
        ("Gap fill", args.gap_fill),
    ]
    if file_start is not None:
        info_rows.insert(1, ("File start", str(getattr(file_start, "datetime", file_start))))

    plot_iq_spectrogram(
        iq=x,
        fs=fs,
        plot_title=args.title,
        vmin=args.vmin,
        vmax=args.vmax,
        tstart=0.0,
        tend=None,
        window=args.window,
        window_size=args.window_size,
        hop_size=args.hop_size,
        output_file=args.output,
        corner_note=corner_note,
        estimate_lfm=args.estimate_lfm or args.estimate_lfm_overlay,
        estimate_lfm_gate=not args.no_estimate_lfm_gate,
        overlay_estimated_lfm=args.estimate_lfm_overlay,
        utc_fractional_second=_timestamp_fractional_second(start_timestamp),
        lfm_overlay_sweep_frequency=args.lfm_overlay_sweep_frequency,
        lfm_overlay_bandwidth=None if args.lfm_overlay_bandwidth_khz is None else args.lfm_overlay_bandwidth_khz * 1e3,
        lfm_overlay_offset=args.lfm_overlay_offset,
        lfm_overlay_gate_frequency=args.lfm_overlay_gate_frequency,
        lfm_overlay_gate_duty=args.lfm_overlay_gate_duty,
        lfm_overlay_gate_phase=args.lfm_overlay_gate_phase,
        info_rows=info_rows,
        info_panel_footer_note=footer_note,
    )

if __name__ == "__main__":
    main()
