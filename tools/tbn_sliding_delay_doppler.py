# tbn_sliding_delay_doppler.py

import argparse
import gc
import numpy as np
import os
import psutil
import subprocess

from datetime import timedelta

from lfm_utils import LFMWaveform
from tbn_utils import lsl_open_tbn, lsl_print_metadata, lsl_read_block_for_one_stream
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

    # Antenna selection
    parser.add_argument("-s", "--stand", type=int, default=1, help="stand ID to select for spectrogram")
    parser.add_argument("-p", "--pol", type=str, default="x", help="pol to select (x/y) for spectrogram")

    # LFM waveform parameters
    parser.add_argument("--sweep-frequency", type=float, required=True, help="Sweep repetition rate / PRF (Hz)")
    parser.add_argument("--bandwidth", type=float, default=100e3, help="Chirp bandwidth (Hz)")

    # Delay processing method
    parser.add_argument("--method", type=str, default="mf", choices=["mf", "dechirp"],
                        help="Delay processing method")

    # Plotting parameters
    parser.add_argument("--title", type=str, default="Delay-Doppler Video", help="Video title")
    parser.add_argument("--output", type=str, default=None, help="Save path (omit to just display)")
    parser.add_argument("--tstart", type=float, default=None)
    parser.add_argument("--tend", type=float, default=None)
    parser.add_argument("--slow-window", type=str, default="hann",
                        choices=["hann", "hamming", "blackman", "none"])
    parser.add_argument("--nfft-doppler", type=int, default=None)
    parser.add_argument("--vmin", type=float, default=None)
    parser.add_argument("--vmax", type=float, default=None)
    parser.add_argument("--fd-max", type=float, default=None)
    parser.add_argument("--fd-min", type=float, default=None)
    parser.add_argument("--d-max", type=float, default=None,
                        help="Max delay to display (ms)")
    parser.add_argument("--d-min", type=float, default=None,
                        help="Min delay to display (ms)")

    # MF-only
    parser.add_argument("--window-width", type=float, default=None, help="MF: fast-time window width (s)")
    parser.add_argument("--window-center", type=float, default=None, help="MF: center time (s) for window")

    # Dechirp-only
    parser.add_argument("--dechirp-window", type=str, default="hamming",
                        choices=["hamming", "hann", "none"])

    args = parser.parse_args()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Open TBN file and read metadata
    idf = lsl_open_tbn(args.input_file)
    lsl_print_metadata(idf)

    fs = float(idf.get_info("sample_rate"))
    # fc = float(idf.get_info("freq1"))

    if args.tstart is None:
        args.tstart = 0.0
        print(f"No --tstart provided, starting from beginning of file")
    if args.tend is None:
        nFramesFile = idf.get_info("nframe")
        args.tend = nFramesFile * 512 / (fs * 520) - args.tstart
        print(f"No --tend provided, using end time of file: {args.tend:.2f} seconds")

    # Construct LFM waveform for pulse compression
    lfm_config = LFMWaveform(
        sample_rate=fs,
        sweep_frequency=args.sweep_frequency,
        bandwidth=args.bandwidth,
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

        idf = lsl_open_tbn(args.input_file)  # reopen file for each frame to avoid memory leaks

        # Read a frame of IQ data for the current frame
        iq_frame, start_timestamp = lsl_read_block_for_one_stream(idf, current_tstart, args.integration_time, args.stand, args.pol)
        # print("iq_frame shape:", iq_frame.shape, flush=True)
        # print("iq_frame GB:", iq_frame.nbytes / 1e9, flush=True)

        if iq_frame is None:
            break

        # Convert timestamp to datetime string for plotting
        if start_timestamp is not None:
            start_time = start_timestamp.utc_datetime
            end_time = start_time + timedelta(seconds=args.integration_time)

            time_range_str = f"{start_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} to {end_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}"

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
            "-r", "2",  # frame rate (adjust as needed)
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