# tbn_sliding_phase_doppler.py

import argparse
import glob
import os
import subprocess

from lfm_utils import (
    LFMWaveform,
    auto_delay_bounds_from_iq,
    dechirp_fft_delay_band_complex,
    reference_gate_frequency_from_args,
)
from plotting_utils import (
    _phase_doppler_from_dechirp_spectra,
    _timestamp_sweep_offset_samples,
    plot_phase_doppler_delay_frame,
)
from tbn_utils import (
    gap_fill_note,
    lsl_open_tbn,
    lsl_print_metadata,
    lsl_read_block_for_one_stream,
    set_default_tbn_time_bounds,
    timestamp_range_note,
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
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error creating video: {e}", flush=True)
    except FileNotFoundError:
        print("ffmpeg not found. Please install ffmpeg to create the video.", flush=True)
    return False


def cleanup_frame_pngs(output_dir):
    frame_paths = sorted(glob.glob(os.path.join(os.path.expanduser(output_dir), "frame_*.png")))
    removed = 0
    for frame_path in frame_paths:
        try:
            os.remove(frame_path)
            removed += 1
        except OSError as exc:
            print(f"Could not remove frame PNG {frame_path}: {exc}", flush=True)
    print(f"Removed {removed} frame PNG(s). Use --keep-frames to preserve them.", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Create a sweep-to-sweep phase-Doppler video from a TBN file.",
        formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, max_help_position=35),
    )
    parser.add_argument("filename", help="TBN file to process")
    parser.add_argument("-s", "--stand", type=int, default=1, help="stand ID to select")
    parser.add_argument("-p", "--pol", choices=["both", "x", "y", "0", "1"], default="both",
                        help="receiver polarization to use; both sums x and y power")

    parser.add_argument("--sweep-frequency", type=float, required=True, help="Sweep repetition rate / PRF (Hz)")
    parser.add_argument("--bandwidth", type=float, default=100e3, help="Chirp bandwidth (Hz)")

    parser.add_argument("--title", type=str, default="Sliding Phase Doppler", help="Frame title")
    parser.add_argument("--output", type=str, required=True, help="Output directory for frames and video")
    parser.add_argument("--tstart", type=float, default=None)
    parser.add_argument("--tend", type=float, default=None)

    parser.add_argument("--d-min", type=float, default=None, help="Minimum delay band edge (ms)")
    parser.add_argument("--d-max", type=float, default=None, help="Maximum delay band edge (ms)")
    parser.add_argument("--auto-delay-sweeps", type=int, default=32,
                        help="Number of initial sweeps used when auto-selecting delay bounds")
    parser.add_argument("--auto-delay-half-width", type=float, default=1.5,
                        help="Half-width in ms around the strongest preview delay bin when auto-selecting")
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
    parser.add_argument("--keep-frames", action="store_true",
                        help="Keep individual frame_*.png files after successfully creating the video")

    parser.add_argument("--dechirp-window", type=str, default="hann",
                        choices=["hamming", "hann", "blackman", "cheb60", "cheb80", "cheb100", "cheb120", "none"])
    parser.add_argument("--offset", type=float, default=0.0,
                        help="Timestamp mode: sweep start offset in seconds after each integer-second boundary")
    parser.add_argument("--gap-fill", choices=["nan", "zero"], default="nan",
                        help="Fill TBN timetag gaps with NaNs for blank/corrupt regions or zeros for continuity")
    parser.add_argument("--no-timestamp-align", action="store_true",
                        help="Do not align dechirp chunks to timestamp second boundaries")
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

    idf = lsl_open_tbn(args.filename)
    lsl_print_metadata(idf)
    fs = float(idf.get_info("sample_rate"))
    center_freq = float(idf.get_info("freq1"))

    set_default_tbn_time_bounds(args, idf)
    duration = args.tend - args.tstart
    os.makedirs(args.output, exist_ok=True)

    lfm_config = LFMWaveform(
        sample_rate=fs,
        sweep_frequency=args.sweep_frequency,
        bandwidth=args.bandwidth,
        reference_gate_frequency=reference_gate_frequency_from_args(args),
        reference_gate_duty=args.reference_gate_duty,
        reference_gate_phase=args.reference_gate_phase,
    )

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
    corner_note = timestamp_range_note(start_timestamp, iq.shape[-1] / fs)

    start_offset_samples = 0
    if not args.no_timestamp_align:
        start_offset_samples = _timestamp_sweep_offset_samples(
            start_timestamp,
            lfm_config,
            sweep_offset=args.offset,
        )
    dechirp_window = None if args.dechirp_window == "none" else args.dechirp_window
    auto_delay_used = args.d_min is None and args.d_max is None
    if (args.d_min is None) != (args.d_max is None):
        raise ValueError("Provide both --d-min and --d-max, or omit both for auto delay selection")
    if auto_delay_used:
        args.d_min, args.d_max, _ = auto_delay_bounds_from_iq(
            iq,
            lfm_config,
            window=dechirp_window,
            start_offset_samples=start_offset_samples,
            preview_sweeps=args.auto_delay_sweeps,
            half_width_ms=args.auto_delay_half_width,
        )

    selected_delay_ms, complex_spectra = dechirp_fft_delay_band_complex(
        received_signal=iq,
        lfm_config=lfm_config,
        d_min=args.d_min,
        d_max=args.d_max,
        window=dechirp_window,
        start_offset_samples=start_offset_samples,
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
        t_abs = args.tstart + start_offset_samples / fs + time_centers[estimate_idx]
        info_rows = [
            ("File", os.path.basename(args.filename)),
            ("Tuning freq", f"{center_freq / 1e6:.6f} MHz"),
            ("Stand/pol", f"{args.stand}/{args.pol}"),
            ("Frame time", f"{t_abs:.3f} s"),
            ("Delay band", f"{selected_delay_ms.min():.3f}-{selected_delay_ms.max():.3f} ms"),
            ("Estimator sweeps", str(args.estimator_sweeps)),
            ("Frame hop", f"{args.frame_hop_sweeps} estimates"),
            ("Dechirp window", args.dechirp_window),
            ("Gap fill", args.gap_fill),
        ]
        if auto_delay_used:
            info_rows.append(("Auto delay", f"first {args.auto_delay_sweeps} sweeps"))
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
            corner_note=corner_note,
            info_rows=info_rows,
            info_panel_footer_note=footer_note,
        )
        print(f"Rendered frame {frame_num + 1} at t={t_abs:.3f}s", flush=True)

    video_created = make_video(args.output, args.framerate)
    if video_created and not args.keep_frames:
        cleanup_frame_pngs(args.output)


if __name__ == "__main__":
    main()
