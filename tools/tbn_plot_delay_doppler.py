# tbn_plot_delay_doppler.py

import argparse
import os

from lfm_utils import LFMWaveform, dechirp_fft_complex, reference_gate_frequency_from_args
from tbn_utils import (
    gap_fill_note,
    lsl_open_tbn,
    lsl_print_metadata,
    lsl_read_block_for_one_stream,
    set_default_tbn_time_bounds,
    timestamp_range_note,
)
from plotting_utils import _timestamp_sweep_offset_samples, plot_delay_doppler_dechirp


def main():
    parser = argparse.ArgumentParser(
        description="Plot a dechirped delay-Doppler map with delay in milliseconds.",
        formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, max_help_position=35),
    )

    parser.add_argument("input_file", type=str, help="Path to the TBN file")

    # Antenna selection
    parser.add_argument("-s", "--stand", type=int, default=1, help="stand ID to select for spectrogram")
    parser.add_argument("-p", "--pol", type=str, default="x", help="pol to select (x/y) for spectrogram")

    # LFM waveform parameters
    parser.add_argument("--sweep-frequency", type=float, required=True, help="Sweep repetition rate / PRF (Hz)")
    parser.add_argument("--bandwidth", type=float, default=100e3, help="Chirp bandwidth (Hz)")

    # Plotting parameters
    parser.add_argument("--title", type=str, default="Delay-Doppler Map", help="Plot title")
    parser.add_argument("--output", type=str, default=None, help="Save path (omit to just display)")
    parser.add_argument("--vmin", type=float, default=None)
    parser.add_argument("--vmax", type=float, default=None)
    parser.add_argument("--tstart", type=float, default=None)
    parser.add_argument("--tend", type=float, default=None)

    parser.add_argument("--slow-window", type=str, default="hann",
                        choices=["hann", "hamming", "blackman", "cheb60", "cheb80", "cheb100", "cheb120", "none"])
    parser.add_argument("--nfft-doppler", type=int, default=None)
    parser.add_argument("--fd-max", type=float, default=None)
    parser.add_argument("--fd-min", type=float, default=None)
    parser.add_argument("--d-max", type=float, default=None,
                        help="Max delay to display (ms)")
    parser.add_argument("--d-min", type=float, default=None,
                        help="Min delay to display (ms)")
    parser.add_argument("--offset", type=float, default=0.0,
                        help="Timestamp mode: sweep start offset in seconds after each integer-second boundary")
    parser.add_argument("--gap-fill", choices=["nan", "zero"], default="nan",
                        help="Fill TBN timetag gaps with NaNs for blank/corrupt regions or zeros for continuity")
    parser.add_argument("--interactive", type=bool, default=False, 
                        help="Whether to display each frame interactively")

    parser.add_argument("--dechirp-window", type=str, default="hann",
                        choices=["hamming", "hann", "blackman", "cheb60", "cheb80", "cheb100", "cheb120", "none"])
    parser.add_argument("--reference-gate-frequency", type=float, default=None,
                        help="Gate the reference chirp at this frequency in Hz")
    parser.add_argument("--reference-gate-period", type=float, default=None,
                        help="Gate the reference chirp at this period in seconds; e.g. 0.005 for 5 ms")
    parser.add_argument("--reference-gate-duty", type=float, default=0.5,
                        help="Reference gate duty cycle in (0, 1]")
    parser.add_argument("--reference-gate-phase", type=float, default=0.0,
                        help="Reference gate phase/time offset in seconds")

    args = parser.parse_args()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Open TBN file and read metadata
    idf = lsl_open_tbn(args.input_file)
    lsl_print_metadata(idf)

    fs = float(idf.get_info("sample_rate"))
    fc = float(idf.get_info("freq1"))
    args.tuning_frequency = fc

    set_default_tbn_time_bounds(args, idf)

    duration = args.tend - args.tstart

    # Construct LFM waveform for pulse compression
    lfm_config = LFMWaveform(
        sample_rate=fs,
        sweep_frequency=args.sweep_frequency,
        bandwidth=args.bandwidth,
        reference_gate_frequency=reference_gate_frequency_from_args(args),
        reference_gate_duty=args.reference_gate_duty,
        reference_gate_phase=args.reference_gate_phase,
    )

    # Read IQ data for the specified time range and antenna stream
    iq, start_timestamp, gap_info = lsl_read_block_for_one_stream(
        idf,
        args.tstart,
        duration,
        stand_id=args.stand,
        pol=args.pol,
        gap_fill=args.gap_fill,
        return_gap_info=True,
    )
    footer_note = gap_fill_note(gap_info)
    corner_note = timestamp_range_note(start_timestamp, len(iq) / fs)
    info_rows = [
        ("File", os.path.basename(args.input_file)),
        ("Tuning freq", f"{fc / 1e6:.6f} MHz"),
        ("Stand/pol", f"{args.stand}/{args.pol}"),
        ("Plot span", f"{args.tstart:.3f}-{args.tend:.3f} s"),
        ("Integration", f"{duration:.3f} s"),
        ("Gap fill", args.gap_fill),
    ]
    if args.fd_min is not None or args.fd_max is not None:
        info_rows.append(("Doppler limits", f"{args.fd_min} to {args.fd_max} Hz"))
    if args.d_min is not None or args.d_max is not None:
        info_rows.append(("Delay limits", f"{args.d_min} to {args.d_max} ms"))

    start_offset_samples = _timestamp_sweep_offset_samples(
        start_timestamp,
        lfm_config,
        sweep_offset=args.offset,
    )

    _, complex_spectra = dechirp_fft_complex(
        received_signal=iq,
        lfm_config=lfm_config,
        window=args.dechirp_window,
        start_offset_samples=start_offset_samples,
    )

    plot_delay_doppler_dechirp(
        dechirp_spectra=complex_spectra,
        lfm_config=lfm_config,
        title=args.title,
        output_file=args.output,
        vmin=args.vmin,
        vmax=args.vmax,
        window_slow=args.slow_window,
        nfft_doppler=args.nfft_doppler,
        fd_max=args.fd_max,
        fd_min=args.fd_min,
        d_max=args.d_max,
        d_min=args.d_min,
        interactive=args.interactive,
        positive_delay_axis=True,
        corner_note=corner_note,
        info_rows=info_rows,
        info_panel_footer_note=footer_note,
    )


if __name__ == "__main__":
    main()
