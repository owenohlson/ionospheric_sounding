"""Shared CLI and dispatch for HDF5 single-receiver plots.

Numerical processing and figure layout remain in rdc_utils/plotting_utils.
"""

import argparse
import os
import subprocess
import tempfile
import shutil

import h5py
import numpy as np
import matplotlib.pyplot as plt

import plotting_utils as plots
from rdc_plot_input import PlotInput


def parser_for(kind):
    parser = argparse.ArgumentParser(description=f"{kind.replace('_', ' ').title()} from HDF5 RDC data")
    parser.set_defaults(kind=kind)
    parser.add_argument("input_file", nargs="?", help="RDC HDF5 file (use rdc_export.py for raw TBN/WAV)")
    parser.add_argument("--rdc0-input", "--rdc1-input", "--rdc2-input", "--input-rdc2", dest="cache_input")
    parser.add_argument("--output", required=kind.startswith("sliding"))
    parser.add_argument("--title", default=kind.replace("_", " ").title())
    parser.add_argument("-s", "--stand", type=int, help="Physical stand; default first available receiver/stand")
    parser.add_argument("-p", "--pol", choices=["both", "x", "y", "0", "1"], default="both")
    for flag in ("tstart", "tend", "vmin", "vmax", "sweep-frequency", "bandwidth"):
        parser.add_argument("--" + flag, type=float)
    parser.add_argument("--gap-fill", choices=["nan", "zero"], help="Default inherits cache; apply before FFT for RDC0")
    parser.add_argument("--dechirp-window", choices=["hann", "hamming", "blackman", "cheb60", "cheb80", "cheb100", "cheb120", "none"])
    for flag in ("reference-gate-frequency", "reference-gate-duty", "reference-gate-phase"):
        parser.add_argument("--" + flag, type=float, help="RDC0 reference setting; must match stored settings for later stages")
    parser.add_argument("--reference-gate-period", type=float, help="Alternative to reference gate frequency, in seconds")
    parser.add_argument("--d-min", type=float)
    parser.add_argument("--d-max", type=float)
    parser.add_argument("--full-delay", action="store_true", help="Display every available delay bin instead of auto peak +/-1.5 ms")
    parser.add_argument("--auto-delay-sweeps", type=int, default=32)
    parser.add_argument("--auto-delay-half-width", type=float, default=1.5)
    if "doppler" in kind:
        parser.add_argument("--fd-min", type=float)
        parser.add_argument("--fd-max", type=float)
    if "phase" in kind:
        parser.add_argument("--estimator-sweeps", type=int, default=1)
        parser.add_argument("--power-floor-db", type=float)
        if kind == "phase_doppler_time":
            parser.add_argument("--fd-bins", type=int, default=128)
            parser.add_argument("--bin-combine", choices=["max", "max2", "max3", "max4", "sum", "mean"], default="max")
    elif "doppler" in kind:
        parser.add_argument("--integration-time", type=float)
        parser.add_argument("--hop-time", type=float)
        parser.add_argument("--nfft-doppler", type=int)
        parser.add_argument("--slow-window", choices=["hann", "hamming", "blackman", "cheb60", "cheb80", "cheb100", "cheb120", "none"])
        if kind == "doppler_time":
            parser.add_argument("--combine", choices=["incoherent", "coherent", "peak"], default="incoherent")
    if kind == "dechirp":
        parser.add_argument("--navg", type=int, default=4)
        parser.add_argument("--streaming", action="store_true", help="Compatibility option; RDC0 is always processed in sweep blocks")
    if kind == "filter_output":
        parser.add_argument("--units", choices=["s", "ms"], default="ms")
    if kind == "delay_doppler":
        parser.add_argument("--interactive", action="store_true", help="Also display the saved plot")
    if kind.startswith("sliding"):
        parser.add_argument("--framerate", type=float, default=10)
        parser.add_argument("--keep-frames", action="store_true")
        parser.add_argument("--max-frames", type=int)
        if "phase" in kind:
            parser.add_argument("--frame-hop-sweeps", type=int, default=1)
    if kind == "spectrogram":
        parser.add_argument("--plot", choices=["single-stream", "average"], default="single-stream")
        parser.add_argument("--pfb", action="store_true", help="Use LSL SpecMaster PFB for averaged spectrum (requires LSL)")
        parser.add_argument("-q", "--quiet", action="store_true")
        parser.add_argument("--window-size", type=int, default=1024)
        parser.add_argument("--hop-size", type=int, default=512)
        parser.add_argument("-w", "--window", default="hann", choices=["none", "bartlett", "blackman", "hann", "hanning"])
        parser.add_argument("--estimate-lfm", action="store_true")
        parser.add_argument("--estimate-lfm-overlay", action="store_true")
        parser.add_argument("--no-estimate-lfm-gate", action="store_true")
        for flag in ("sweep-frequency", "bandwidth-khz", "offset", "gate-frequency", "gate-duty", "gate-phase"):
            parser.add_argument("--lfm-overlay-" + flag, type=float)
    return parser


def video_frames(args, render):
    output = os.path.expanduser(args.output)
    os.makedirs(output, exist_ok=True)
    # Isolate this run so stale PNGs from a previous run cannot enter the video.
    directory = tempfile.mkdtemp(prefix="frames_", dir=output)
    count = render(directory)
    if count < 1:
        raise ValueError("No complete frames within the selected time span")
    path = os.path.join(output, args.kind + ".mp4")
    try:
        subprocess.run(["ffmpeg", "-y", "-framerate", str(args.framerate), "-i",
                        os.path.join(directory, "frame_%04d.png"), "-frames:v", str(count),
                        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p", path], check=True)
    except (OSError, subprocess.CalledProcessError):
        print(f"Video encoding failed; individual frames preserved in {directory}")
        raise
    if args.keep_frames:
        print(f"Individual frames: {directory}")
    else:
        shutil.rmtree(directory)
    print(f"Video: {path}")


def draw_dd(args, source, spectra, start, stop, gaps, output, cube=None):
    annotation = source.annotation(start, stop, gaps)
    annotation["info_rows"] += [("Integration", f"{stop - start:g} s"),
                                ("Hop", f"{args.hop_time:g} s"), ("Slow window", args.slow_window)]
    plots.plot_delay_doppler_dechirp(
        spectra, source.config, args.title, output_file=output, vmin=args.vmin, vmax=args.vmax,
        window_slow=args.slow_window if cube is None else "none", nfft_doppler=args.nfft_doppler if cube is None else None,
        fd_min=args.fd_min, fd_max=args.fd_max, d_min=float(source.delay[0]), d_max=float(source.delay[-1]),
        selected_delay_ms=source.delay, stored_rdc2=cube,
        stored_doppler_hz=source.loaded.get("doppler_hz"), auto_delay_window=False,
        interactive=getattr(args, "interactive", False), **annotation)


def plot_dd(args, source):
    if source.stage == 2:
        for flag, key in (("integration_time", "integration_time_s"), ("hop_time", "hop_time_s"),
                          ("nfft_doppler", "nfft_doppler"), ("slow_window", "slow_window")):
            setattr(args, flag, source.setting(flag, key))
        args.hop_time = args.hop_time or float(source.ends[0] - source.starts[0])
        args.slow_window = args.slow_window or "unknown"
        frames = np.flatnonzero((source.starts >= source.start - 1e-8) & (source.ends <= source.stop + 1e-8))
        def render(directory=None):
            chosen = frames[:args.max_frames] if getattr(args, "max_frames", None) else frames
            if directory is None:
                chosen = chosen[:1]
            for number, frame in enumerate(chosen):
                cube, gaps = source.doppler(frame)
                output = os.path.join(directory, f"frame_{number:04d}.png") if directory else args.output
                draw_dd(args, source, np.empty((1, 1, len(source.delay))), source.starts[frame], source.ends[frame], gaps, output, cube)
            return len(chosen)
    else:
        args.integration_time = args.integration_time or (10. if args.kind.startswith("sliding") else (source.last - source.first) / source.prf)
        args.hop_time = args.hop_time or (5. if args.kind.startswith("sliding") else args.integration_time)
        args.slow_window = args.slow_window or "hann"
        n = int(round(args.integration_time * source.prf))
        hop = int(round(args.hop_time * source.prf))
        if n < 1 or hop < 1:
            raise ValueError("Integration and hop must contain at least one sweep")
        args.hop_time = hop / source.prf
        def render(directory=None):
            count = 0
            for first in range(source.first, source.last - n + 1, hop):
                if count >= (getattr(args, "max_frames", None) or float("inf")):
                    break
                spectra, gaps = source.spectra(first, first + n)
                output = os.path.join(directory, f"frame_{count:04d}.png") if directory else args.output
                draw_dd(args, source, spectra, first / source.prf, (first + n) / source.prf, gaps, output)
                count += 1
                if directory is None:
                    break
            return count
    if args.kind.startswith("sliding"):
        video_frames(args, render)
    elif render() < 1:
        raise ValueError("No complete integration frame within the selected time span")


def plot_spectrogram(args, source):
    if source.stage != 0:
        raise ValueError("Spectrogram requires raw RDC0")
    if source.meta.get("adc_start", 0) != 0 or source.shape[0] != int(round(source.fs / source.prf)):
        raise ValueError("Spectrogram requires full ADC sweeps; cropped ADC would create time discontinuities")
    if args.plot == "average":
        plot_average_spectrum(args, source)
        return
    if args.pfb:
        raise ValueError("--pfb applies only to --plot=average")
    with h5py.File(source.path, "r") as h5:
        grp = h5["rdc0"]
        select = (slice(None), slice(source.first, source.last), source.channels)
        cube, gaps = source.clean(grp["real"][select] + 1j * grp["imag"][select])
        if "invalid_counts" in grp:
            gaps |= np.any(grp["invalid_counts"][source.first:source.last, source.channels] > 0, axis=1)
    iq = cube.transpose(2, 1, 0).reshape(len(source.channels), -1)
    plots.plot_iq_spectrogram(
        iq, source.fs, args.title, args.vmin, args.vmax, sample_time_origin=source.first / source.prf,
        window="boxcar" if args.window == "none" else ("hann" if args.window == "hanning" else args.window),
        window_size=args.window_size, hop_size=args.hop_size, output_file=args.output,
        estimate_lfm=args.estimate_lfm or args.estimate_lfm_overlay, estimate_lfm_gate=not args.no_estimate_lfm_gate,
        overlay_estimated_lfm=args.estimate_lfm_overlay,
        utc_fractional_second=(source.epoch.microsecond / 1e6 if source.epoch else None),
        lfm_overlay_sweep_frequency=args.lfm_overlay_sweep_frequency,
        lfm_overlay_bandwidth=(args.lfm_overlay_bandwidth_khz * 1000 if args.lfm_overlay_bandwidth_khz is not None else None),
        lfm_overlay_offset=args.lfm_overlay_offset, lfm_overlay_gate_frequency=args.lfm_overlay_gate_frequency,
        lfm_overlay_gate_duty=args.lfm_overlay_gate_duty, lfm_overlay_gate_phase=args.lfm_overlay_gate_phase,
        **source.annotation(source.first / source.prf, source.last / source.prf, gaps))


def plot_average_spectrum(args, source):
    from scipy.signal import periodogram
    from lfm_utils import window_from_arg
    window = window_from_arg(args.window_size, args.window)
    total, n = None, 0
    had_gaps = False
    with h5py.File(source.path, "r") as h5:
        group = h5["rdc0"]
        for channel in source.channels:
            carry = np.empty(0, dtype=np.complex64)
            for first in range(source.first, source.last, 32):
                last = min(first + 32, source.last)
                x = (group["real"][:, first:last, channel] + 1j * group["imag"][:, first:last, channel]).T.ravel()
                bad = ~np.isfinite(x)
                had_gaps |= np.any(bad)
                if "invalid_counts" in group:
                    had_gaps |= np.any(group["invalid_counts"][first:last, channel] > 0)
                if source.gap_fill == "zero":
                    x[bad] = 0
                x = np.r_[carry, x]
                block_size = args.window_size * (4 if args.pfb else 1)
                count = len(x) // block_size
                carry = x[count * block_size:]
                if count == 0:
                    continue
                segments = x[:count * block_size].reshape(count, block_size)
                segments = segments[np.all(np.isfinite(segments), axis=1)]
                if not len(segments):
                    continue
                if args.pfb:
                    from lsl.correlator import fx
                    frequency, power = fx.SpecMaster(segments, LFFT=args.window_size, window=window,
                                                    pfb=True, verbose=not args.quiet, sample_rate=source.fs)
                else:
                    frequency, power = periodogram(segments, fs=source.fs, window=window, detrend=False,
                                                  return_onesided=False, scaling="spectrum", axis=-1)
                total = power.sum(axis=0) if total is None else total + power.sum(axis=0)
                n += len(segments)
    if not n:
        raise ValueError("No finite complete FFT blocks for the averaged spectrum")
    annotation = source.annotation(source.first / source.prf, source.last / source.prf, [had_gaps])
    annotation["info_rows"].append(("Channels averaged", str(len(source.channels))))
    fig, ax = plots._new_figure_with_info()
    order = np.argsort(frequency)
    fc = source.meta.get("tuning_frequency_hz") or 0.
    ax.plot((frequency[order] + fc) / 1000, 10 * np.log10(total[order] / n + 1e-30))
    ax.set(xlabel="Frequency [kHz]", ylabel="Power [dB/RBW]", title=args.title, ylim=(args.vmin, args.vmax))
    plots._add_info_panel(fig, info_rows=annotation["info_rows"], corner_note=annotation["corner_note"],
                          footer_note=annotation["info_panel_footer_note"])
    plots._tight_layout_with_info(fig, has_info=True)
    if args.output:
        plots._savefig(args.output)
    else:
        plt.show()
    plt.close(fig)


def main(kind, argv=None):
    parser = parser_for(kind)
    args = parser.parse_args(argv)
    if bool(args.input_file) == bool(args.cache_input):
        parser.error("Provide one HDF5 input, positional or via --rdcN-input")
    args.input_file = args.input_file or args.cache_input
    if args.reference_gate_period is not None:
        from lfm_utils import reference_gate_frequency_from_args
        args.reference_gate_frequency = reference_gate_frequency_from_args(args)
    if args.output:
        args.output = os.path.expanduser(args.output)
    if args.auto_delay_sweeps < 1 or args.auto_delay_half_width <= 0:
        parser.error("Auto-delay preview count and width must be positive")
    for flag in ("integration_time", "hop_time", "navg", "estimator_sweeps", "frame_hop_sweeps", "framerate", "max_frames"):
        value = getattr(args, flag, None)
        if value is not None and (not np.isfinite(value) or value <= 0):
            parser.error(f"--{flag.replace('_', '-')} must be positive")
    source = PlotInput(args, prefer_doppler=kind in ("delay_doppler", "sliding_delay_doppler"))
    if kind == "spectrogram":
        plot_spectrogram(args, source)
        return
    source.select_delay()
    if kind in ("delay_doppler", "sliding_delay_doppler"):
        plot_dd(args, source)
        return
    spectra, gaps = source.spectra(source.first, source.last)
    start, stop = source.first / source.prf, source.last / source.prf
    annotation = source.annotation(start, stop, gaps)
    common = dict(lfm_config=source.config, title=args.title, output_file=args.output, **annotation)
    power_scale = dict(vmin=args.vmin, vmax=args.vmax)
    if kind == "dechirp":
        if args.navg > spectra.shape[1]:
            raise ValueError("--navg exceeds available sweeps")
        annotation["info_rows"].append(("Navg", str(args.navg)))
        plots.plot_dechirp(np.sum(np.abs(spectra) ** 2, axis=0), source.config,
            title=args.title, save_path=args.output, navg=args.navg, tstart=start,
            selected_delay_ms=source.delay, auto_delay_window=False, **annotation, **power_scale)
    elif kind == "filter_output":
        plots.plot_dechirp_profile(source.delay, spectra, time_units=args.units, **common, **power_scale)
    elif kind == "doppler_time":
        args.integration_time = args.integration_time or source.meta.get("integration_time_s", 60.)
        args.hop_time = args.hop_time or source.meta.get("hop_time_s", 10.)
        args.slow_window = args.slow_window or source.meta.get("slow_window", "hann")
        args.nfft_doppler = args.nfft_doppler or source.meta.get("nfft_doppler")
        annotation["info_rows"] += [("Integration", f"{args.integration_time:g} s"), ("Hop", f"{args.hop_time:g} s")]
        plots.plot_doppler_time_from_delay_band(spectra, selected_delay_ms=source.delay,
            d_min=float(source.delay[0]), d_max=float(source.delay[-1]),
            fd_min=args.fd_min, fd_max=args.fd_max, integration_time=args.integration_time,
            hop_time=args.hop_time, slow_window=args.slow_window or "hann", nfft_doppler=args.nfft_doppler,
            combine=args.combine, tstart=start, **common, **power_scale)
    elif kind == "sliding_phase_doppler":
        times, fd, power = plots._phase_doppler_from_dechirp_spectra(spectra, source.prf, args.estimator_sweeps)
        def render(directory):
            count = 0
            for index in range(0, len(times), args.frame_hop_sweeps):
                if args.max_frames is not None and count >= args.max_frames:
                    break
                first = source.first + index
                last = first + args.estimator_sweeps + 1
                note = source.annotation(first / source.prf, last / source.prf, gaps[index:index + args.estimator_sweeps + 1])
                note["info_rows"] += [("Estimator sweeps", str(args.estimator_sweeps)), ("Frame hop", f"{args.frame_hop_sweeps / source.prf:g} s")]
                plots.plot_phase_doppler_delay_frame(fd[index], power[index], source.delay, source.config,
                    args.title, output_file=os.path.join(directory, f"frame_{count:04d}.png"),
                    fd_min=args.fd_min, fd_max=args.fd_max, power_floor_db=args.power_floor_db, **note, **power_scale)
                count += 1
                plt.close("all")
            return count
        video_frames(args, render)
    else:
        function = plots.plot_phase_doppler_time_delay if kind == "phase_doppler" else plots.plot_phase_doppler_time_from_delay_band
        extra = {} if kind == "phase_doppler" else dict(fd_bins=args.fd_bins, bin_combine=args.bin_combine, **power_scale)
        function(spectra, source.delay, estimator_sweeps=args.estimator_sweeps,
                 power_floor_db=args.power_floor_db, fd_min=args.fd_min, fd_max=args.fd_max,
                 tstart=start, **common, **extra)
    plt.close("all")
