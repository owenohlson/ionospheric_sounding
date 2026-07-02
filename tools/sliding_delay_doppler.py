# sliding_delay_doppler.py

import argparse
import gc
import numpy as np
import os
import psutil
import subprocess

from lfm_utils import LFMWaveform, load_iq_audio, reference_gate_frequency_from_args
from plotting_utils import delay_doppler_process_window


process = psutil.Process(os.getpid())

def mem():
    print(f"Memory usage: {process.memory_info().rss / 1e9:.2f} GB", flush=True)

def main():
    parser = argparse.ArgumentParser(
        description="Create a video made of several delay-Doppler maps.",
        formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, max_help_position=35),
    )

    parser.add_argument("input_file", type=str, help="Path to the TBN file")

    # Video parameters
    parser.add_argument("--integration-time", type=float, default=10.0, help="Integration time for each delay-Doppler map (s)")
    parser.add_argument("--hop-time", type=float, default=5.0, help="Hop time between consecutive maps (s)")

    # LFM waveform parameters
    parser.add_argument("--sweep-frequency", type=float, required=True, help="Sweep repetition rate / PRF (Hz)")
    parser.add_argument("--bandwidth", type=float, default=100e3, help="Chirp bandwidth (Hz)")

    # Delay processing method
    parser.add_argument("--method", type=str, default="dechirp", choices=["mf", "dechirp"],
                        help="Delay processing method")

    # Plotting parameters
    parser.add_argument("--title", type=str, default="Delay-Doppler Video", help="Video title")
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
    parser.add_argument("--interactive", type=bool, default=False, 
                        help="Whether to display each frame interactively")

    # MF-only
    parser.add_argument("--window-width", type=float, default=None, help="MF: fast-time window width (s)")
    parser.add_argument("--window-center", type=float, default=None, help="MF: center time (s) for window")

    # Dechirp-only
    parser.add_argument("--dechirp-window", type=str, default="hann",
                        choices=["hamming", "hann", "cheb60", "cheb80", "cheb100", "cheb120", "none"])
    parser.add_argument("--reference-gate-frequency", type=float, default=None,
                        help="Gate the reference chirp at this frequency in Hz")
    parser.add_argument("--reference-gate-period", type=float, default=None,
                        help="Gate the reference chirp at this period in seconds; e.g. 0.005 for 5 ms")
    parser.add_argument("--reference-gate-duty", type=float, default=0.5,
                        help="Reference gate duty cycle in (0, 1]")
    parser.add_argument("--reference-gate-phase", type=float, default=0.0,
                        help="Reference gate phase/time offset in seconds")
    
    # FFmpeg parameters
    parser.add_argument("--framerate", type=int, default=2, help="Frame rate for the output video (frames per second)")

    args = parser.parse_args()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Open file and read metadata
    iq, fs = load_iq_audio(args.input_file)

    if args.tstart is None:
        args.tstart = 0.0
        print(f"No --tstart provided, starting from beginning of file")
    if args.tend is None:
        nFramesFile = iq.shape[0]
        args.tend = (nFramesFile - 1) / fs
        print(f"No --tend provided, using end time of file: {args.tend:.2f} seconds")

    # Construct LFM waveform for pulse compression
    lfm_config = LFMWaveform(
        sample_rate=fs,
        sweep_frequency=args.sweep_frequency,
        bandwidth=args.bandwidth,
        reference_gate_frequency=reference_gate_frequency_from_args(args),
        reference_gate_duty=args.reference_gate_duty,
        reference_gate_phase=args.reference_gate_phase,
    )
    
    # Delay-Doppler processing and plotting
    available = args.tend - args.tstart - args.integration_time
    nframes = int(np.floor(available / args.hop_time)) + 1
    frame_idx = 0
    anchor_start = args.tstart

    for frame_idx in range(nframes):
        current_tstart = anchor_start + (frame_idx * args.hop_time)
        time_range_str = None

        print(f"Processing frame {frame_idx+1}/{nframes} (t={current_tstart:.2f} to {current_tstart + args.integration_time:.2f} s)", flush=True)
        mem()

        # Slice a frame of IQ data for the current frame
        iq_frame = iq[int(current_tstart * fs) : int((current_tstart + args.integration_time) * fs)]
        # print("iq_frame shape:", iq_frame.shape, flush=True)
        # print("iq_frame GB:", iq_frame.nbytes / 1e9, flush=True)

        if iq_frame is None:
            break

        time_range_str = f"{str(current_tstart)} to {str(current_tstart + args.integration_time)} s"

        print(f"Frame {frame_idx+1} timestamps: {time_range_str}", flush=True)
        
        # Process frame and generate delay-Doppler plot
        delay_doppler_process_window(
            iq_frame,
            frame_idx,
            args,
            lfm_config,
            time_range_str
        )

        # print(f"Finished processing frame {frame_idx+1}/{nframes}", flush=True)
        # mem()

        del iq_frame
        gc.collect()

        print(f"Finished processing and cleaning up frame {frame_idx+1}/{nframes}", flush=True)
        mem()

        frame_idx += 1

    # Use ffmpeg to create video from frames
    if args.output:
        print("Plotting complete, creating video with ffmpeg...", flush=True)

        full_output_path = os.path.expanduser(args.output)

        # Output video name
        video_name = os.path.join(full_output_path, "delay_doppler_video.mp4")

        # ffmpeg command to create video
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",  # overwrite output file if it exists
            "-r", str(args.framerate),  # frame rate (adjust as needed)
            "-start_number", "0",
            "-i", f"{full_output_path}/frame_%04d.png",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            video_name
        ]

        # Run ffmpeg command
        try:
            subprocess.run(ffmpeg_cmd, check=True)
            print(f"Video created successfully: {video_name}", flush=True)
        except subprocess.CalledProcessError as e:
            print(f"Error creating video: {e}", flush=True)
        except FileNotFoundError:
            print("ffmpeg not found. Please install ffmpeg to create the video.", flush=True)

if __name__ == "__main__":
    main()
