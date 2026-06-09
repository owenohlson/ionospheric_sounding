# delay_microdoppler.py

import argparse

from plotting_utils import plot_micro_doppler
from lfm_utils import LFMWaveform, load_iq_audio, dechirp_fft_complex


def main():
    parser = argparse.ArgumentParser(
        description="Plot a delay-Doppler (delay in milliseconds) map using MF or dechirp.",
        formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, max_help_position=35),
    )

    parser.add_argument("input_file", type=str, help="Path to the stereo IQ audio file (.wav, .flac)")
    parser.add_argument("--sweep-frequency", type=float, required=True, help="Sweep repetition rate / PRF (Hz)")
    parser.add_argument("--bandwidth", type=float, default=100e3, help="Chirp bandwidth (Hz)")

    parser.add_argument("--title", type=str, default="Delay-Doppler Map", help="Plot title")
    parser.add_argument("--output", type=str, default=None, help="Save path (omit to just display)")
    parser.add_argument("--vmin", type=float, default=None)
    parser.add_argument("--vmax", type=float, default=None)
    parser.add_argument("--tstart", type=float, default=None)
    parser.add_argument("--tend", type=float, default=None)

    parser.add_argument("--window-stft", type=str, default="hann",
                        choices=["hann", "hamming", "blackman", "none"])
    parser.add_argument("--nfft-stft", type=int, default=None)
    parser.add_argument("--fd-limit", type=float, default=None)
    # parser.add_argument("--d-limit", type=float, default=None,
    #                     help="Max delay to display (ms)")
    # parser.add_argument("--tx-velocity", type=float, default=None,
    #                     help="Transmitter velocity (km/s)")
    # parser.add_argument("--tx-altitude", type=float, default=None,
    #                     help="Transmitter altitude (km)")
    # parser.add_argument("--fc", type=float, default=29e6,
    #                     help="Carrier frequency (Hz) for velocity compensation")
    # parser.add_argument("--time-offset", type=float, default=0,
    #                     help="Time offset (s) to apply to the delay model, to align with the closest approach at t=0")
    parser.add_argument("--show-phase-track", action="store_true")

    # Dechirp-only
    parser.add_argument("--dechirp-window", type=str, default="hamming",
                        choices=["hamming", "hann", "none"])

    args = parser.parse_args()

    iq, fs = load_iq_audio(args.input_file)

    lfm_config = LFMWaveform(
        sample_rate=fs,
        sweep_frequency=args.sweep_frequency,
        bandwidth=args.bandwidth,
    )

    sweep_duration = 1 / args.sweep_frequency

    window_stft = None if args.window_stft == "none" else args.window_stft

    num_sweeps = int(len(iq) / (sweep_duration * fs))
    # start_time_offset = -30.0  # seconds before closest approach
    # time_offsets = start_time_offset + np.arange(num_sweeps) * sweep_duration
            
    _, complex_spectra = dechirp_fft_complex(
        received_signal=iq,
        lfm_config=lfm_config,
        window=None if args.dechirp_window == "none" else args.dechirp_window,
    )

    plot_micro_doppler(
        dechirp_spectra=complex_spectra,
        lfm_config=lfm_config,
        title=args.title,
        output_file=args.output,
        vmin=args.vmin,
        vmax=args.vmax,
        chirp_index=0,
        use_mean_chirp=False,
        nperseg=256,
        noverlap=None,
        nfft_stft=args.nfft_stft,
        window_stft=window_stft,
        beat_bin_window=10,
        fd_limit=args.fd_limit,
        show_phase_track=args.show_phase_track,
    )


if __name__ == "__main__":
    main()