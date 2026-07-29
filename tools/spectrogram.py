# spectrogram.py

from plotting_utils import plot_iq_spectrogram
from lfm_utils import load_iq_audio
import argparse
import os
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
    parser.add_argument("--estimate-lfm", action="store_true",
                        help="Estimate LFM bandwidth, sweep frequency, and optional gating from the spectrogram")
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
    if args.title == "IQ Spectrogram" and args.output:
        args.title = os.path.basename(args.output)

    info_rows = [
        ("File", os.path.basename(args.input_file)),
        ("Sample rate", f"{fs / 1e3:.3f} kHz"),
        ("Time span", f"{args.tstart:.3f}-{args.tend:.3f} s"),
        ("Window", f"{args.window_size} samples"),
        ("Hop", f"{args.hop_size} samples"),
    ]

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
        output_file=args.output,
        estimate_lfm=args.estimate_lfm or args.estimate_lfm_overlay,
        estimate_lfm_gate=not args.no_estimate_lfm_gate,
        overlay_estimated_lfm=args.estimate_lfm_overlay,
        lfm_overlay_sweep_frequency=args.lfm_overlay_sweep_frequency,
        lfm_overlay_bandwidth=None if args.lfm_overlay_bandwidth_khz is None else args.lfm_overlay_bandwidth_khz * 1e3,
        lfm_overlay_offset=args.lfm_overlay_offset,
        lfm_overlay_gate_frequency=args.lfm_overlay_gate_frequency,
        lfm_overlay_gate_duty=args.lfm_overlay_gate_duty,
        lfm_overlay_gate_phase=args.lfm_overlay_gate_phase,
        info_rows=info_rows,
    )

if __name__ == "__main__":
    main()
