# doppler_time_plot.py

import argparse
import soundfile as sf

from lfm_utils import (
    LFMWaveform,
    dechirp_fft_delay_band_complex,
    load_iq_audio,
    reference_gate_frequency_from_args,
)
from plotting_utils import plot_doppler_time_from_delay_band


def main():
    parser = argparse.ArgumentParser(
        description="Plot slow-time Doppler vs time for a selected dechirp delay band.",
        formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, max_help_position=35),
    )

    parser.add_argument("input_file", type=str, help="Path to the stereo IQ audio file (.wav, .flac)")
    parser.add_argument("--sweep-frequency", type=float, required=True, help="Sweep repetition rate / PRF (Hz)")
    parser.add_argument("--bandwidth", type=float, default=100e3, help="Chirp bandwidth (Hz)")

    parser.add_argument("--title", type=str, default="Doppler vs Time", help="Plot title")
    parser.add_argument("--output", type=str, default=None, help="Output PNG filename; displays if omitted")
    parser.add_argument("--vmin", type=float, default=None)
    parser.add_argument("--vmax", type=float, default=None)
    parser.add_argument("--tstart", type=float, default=None)
    parser.add_argument("--tend", type=float, default=None)

    parser.add_argument("--d-min", type=float, required=True, help="Minimum delay band edge (ms)")
    parser.add_argument("--d-max", type=float, required=True, help="Maximum delay band edge (ms)")
    parser.add_argument("--fd-min", type=float, default=None, help="Minimum Doppler to display (Hz)")
    parser.add_argument("--fd-max", type=float, default=None, help="Maximum Doppler to display (Hz)")
    parser.add_argument("--integration-time", type=float, default=60.0,
                        help="Slow-time Doppler integration time (s)")
    parser.add_argument("--hop-time", type=float, default=10.0,
                        help="Hop time between Doppler estimates (s)")
    parser.add_argument("--slow-window", type=str, default="hann",
                        choices=["hann", "hamming", "blackman", "cheb60", "cheb80", "cheb100", "cheb120", "none"])
    parser.add_argument("--nfft-doppler", type=int, default=None)
    parser.add_argument("--combine", type=str, default="incoherent",
                        choices=["incoherent", "coherent", "peak"],
                        help="How to combine selected delay bins before plotting Doppler power")

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

    info = sf.info(args.input_file)
    fs = info.samplerate

    if args.tstart is None:
        args.tstart = 0.0
        print("No --tstart provided, starting from beginning of file")
    if args.tend is None:
        args.tend = info.frames / fs
        print(f"No --tend provided, using end time of file: {args.tend:.2f} seconds")

    start_idx = int(args.tstart * fs)
    end_idx = int(args.tend * fs)
    iq, fs = load_iq_audio(args.input_file, start=start_idx, stop=end_idx)

    lfm_config = LFMWaveform(
        sample_rate=fs,
        sweep_frequency=args.sweep_frequency,
        bandwidth=args.bandwidth,
        reference_gate_frequency=reference_gate_frequency_from_args(args),
        reference_gate_duty=args.reference_gate_duty,
        reference_gate_phase=args.reference_gate_phase,
    )

    selected_delay_ms, complex_spectra = dechirp_fft_delay_band_complex(
        received_signal=iq,
        lfm_config=lfm_config,
        d_min=args.d_min,
        d_max=args.d_max,
        window=None if args.dechirp_window == "none" else args.dechirp_window,
    )

    plot_doppler_time_from_delay_band(
        dechirp_spectra=complex_spectra,
        lfm_config=lfm_config,
        title=args.title,
        output_file=args.output,
        vmin=args.vmin,
        vmax=args.vmax,
        d_min=args.d_min,
        d_max=args.d_max,
        fd_min=args.fd_min,
        fd_max=args.fd_max,
        integration_time=args.integration_time,
        hop_time=args.hop_time,
        slow_window=args.slow_window,
        nfft_doppler=args.nfft_doppler,
        combine=args.combine,
        tstart=args.tstart,
        selected_delay_ms=selected_delay_ms,
    )


if __name__ == "__main__":
    main()
