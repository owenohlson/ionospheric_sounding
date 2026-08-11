# sliding_phase_doppler.py

import argparse
import os
import subprocess

import soundfile as sf

from lfm_utils import (
    LFMWaveform,
    dechirp_fft_delay_band_complex,
    load_iq_audio,
    reference_gate_frequency_from_args,
)
from plotting_utils import (
    _phase_doppler_from_dechirp_spectra,
    plot_phase_doppler_delay_frame,
)


def make_video(output_dir, framerate, video_name="phase_doppler_video.mp4"):
    video_path = os.path.join(output_dir, video_name)
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-r", str(framerate),
        "-start_number", "0",
        "-i", os.path.join(output_dir, "frame_%04d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        video_path,
    ]
    try:
        subprocess.run(ffmpeg_cmd, check=True)
        print(f"Video created successfully: {video_path}", flush=True)
    except subprocess.CalledProcessError as e:
        print(f"Error creating video: {e}", flush=True)
    except FileNotFoundError:
        print("ffmpeg not found. Please install ffmpeg to create the video.", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Create a sweep-to-sweep phase-Doppler video from a WAV/FLAC IQ file.",
        formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, max_help_position=35),
    )
    parser.add_argument("input_file", type=str, help="Path to the stereo IQ audio file (.wav, .flac)")
    parser.add_argument("--sweep-frequency", type=float, required=True, help="Sweep repetition rate / PRF (Hz)")
    parser.add_argument("--bandwidth", type=float, default=100e3, help="Chirp bandwidth (Hz)")

    parser.add_argument("--title", type=str, default="Sliding Phase Doppler", help="Frame title")
    parser.add_argument("--output", type=str, required=True, help="Output directory for frames and video")
    parser.add_argument("--tstart", type=float, default=None)
    parser.add_argument("--tend", type=float, default=None)

    parser.add_argument("--d-min", type=float, required=True, help="Minimum delay band edge (ms)")
    parser.add_argument("--d-max", type=float, required=True, help="Maximum delay band edge (ms)")
    parser.add_argument("--fd-min", type=float, default=None,
                        help="Minimum Doppler axis (Hz), clipped to +/- PRF/2")
    parser.add_argument("--fd-max", type=float, default=None,
                        help="Maximum Doppler axis (Hz), clipped to +/- PRF/2")
    parser.add_argument("--vmin", type=float, default=None, help="Minimum power color scale (dB)")
    parser.add_argument("--vmax", type=float, default=None, help="Maximum power color scale (dB)")
    parser.add_argument("--power-floor-db", type=float, default=None,
                        help="Mask delay bins below this dechirped power level")
    parser.add_argument("--estimator-sweeps", type=int, default=1,
                        help="Number of adjacent sweep-pair phase products to average per frame")
    parser.add_argument("--frame-hop-sweeps", type=int, default=1,
                        help="Number of phase estimates to skip between rendered frames")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Optional cap on rendered frames")
    parser.add_argument("--framerate", type=int, default=10, help="Output video frame rate")

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
    if args.frame_hop_sweeps < 1:
        raise ValueError("--frame-hop-sweeps must be >= 1")

    info = sf.info(args.input_file)
    fs = info.samplerate
    if args.tstart is None:
        args.tstart = 0.0
        print("No --tstart provided, starting from beginning of file")
    if args.tend is None:
        args.tend = info.frames / fs
        print(f"No --tend provided, using end time of file: {args.tend:.2f} seconds")

    os.makedirs(args.output, exist_ok=True)
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
    time_centers, fd_hz, power_db = _phase_doppler_from_dechirp_spectra(
        complex_spectra,
        lfm_config.sweep_frequency,
        estimator_sweeps=args.estimator_sweeps,
    )

    frame_indices = range(0, len(time_centers), args.frame_hop_sweeps)
    for frame_num, estimate_idx in enumerate(frame_indices):
        if args.max_frames is not None and frame_num >= args.max_frames:
            break
        t_abs = args.tstart + time_centers[estimate_idx]
        info_rows = [
            ("File", os.path.basename(args.input_file)),
            ("Frame time", f"{t_abs:.3f} s"),
            ("Delay band", f"{selected_delay_ms.min():.3f}-{selected_delay_ms.max():.3f} ms"),
            ("Estimator sweeps", str(args.estimator_sweeps)),
            ("Frame hop", f"{args.frame_hop_sweeps} estimates"),
            ("Dechirp window", args.dechirp_window),
        ]
        plot_phase_doppler_delay_frame(
            fd_hz=fd_hz[estimate_idx],
            power_db=power_db[estimate_idx],
            selected_delay_ms=selected_delay_ms,
            lfm_config=lfm_config,
            title=f"{args.title} | t={t_abs:.3f} s",
            output_file=os.path.join(args.output, f"frame_{frame_num:04d}.png"),
            fd_min=args.fd_min,
            fd_max=args.fd_max,
            power_floor_db=args.power_floor_db,
            vmin=args.vmin,
            vmax=args.vmax,
            info_rows=info_rows,
        )
        print(f"Rendered frame {frame_num + 1} at t={t_abs:.3f}s", flush=True)

    make_video(args.output, args.framerate)


if __name__ == "__main__":
    main()
