"""Wavefront tracking: follow one echo and plot how its phase across the array changes.

A TID is far larger than the array, so it shows up as a slow tilt of the
wavefront (the arrival direction wanders), not as fringes across the stands.
This command follows one echo through the delay-Doppler frames and measures
that tilt with the per-stand phases (method: ``wavefront.py``):

* reference plane: the session's median arrival direction (or --reference-az/el),
  divided out together with the common (Doppler) phase;
* each stand unwrapped in time only (never across stands);
* each stand's time mean removed, so the *changes* do not depend on calibration;
* a plane fitted every frame: its tilt gives az(t) and el(t), the rest is the
  residual (non-planar structure, multipath, noise).

Outputs
  --output         summary PNG: az/el from the phase planes (grey dots = the coarse
                   grid fit of each frame), echo Doppler/delay/SNR, residual, and the
                   arrival-direction track in direction cosines.
  --maps-output    small multiples of the per-stand phase change (or residual) on the
                   ground, plus the static residual map (what calibration left).
  --surface-output 3D view (east, north, phase change), coloured by the residual,
                   the change or the amplitude: a PNG for one frame, or a directory
                   for a video of every frame.
  --csv / --npz    per-frame table / all arrays for your own analysis.

Input: anything with RDC1 behind it (RDC0, RDC1, RDC2 with retained RDC1) or
stored RDC2 frames. One polarization. Doppler frames default to 2 s
integration every 0.5 s when computed from RDC1.

Example:
    python tools/plot_wavefront.py rdc1.h5 --calibration cal_13p5.h5 --d-min 3.5 --d-max 6 \\
        --echo-delay 4.37 --echo-doppler -0.4 --output wavefront.png \\
        --maps-output wavefront_maps.png --surface-output wavefront_3d/ --csv wavefront.csv
"""

import csv
import os

import numpy as np

import calibration_utils as cu
import cli
import plotting_utils as plots
import rdc_utils as ru
import wavefront as wf

DEFAULT_INTEGRATION_S, DEFAULT_HOP_S = 2.0, 0.5


def build_parser():
    p = cli.parser(__doc__.split("\n\n")[0])
    cli.add_input(p)
    g = p.add_argument_group("output")
    g.add_argument("--output", required=True, help="Summary PNG")
    g.add_argument("--maps-output", default=None, help="PNG of per-stand phase maps (small multiples)")
    g.add_argument("--maps", type=int, default=7, help="Number of times shown in --maps-output")
    g.add_argument("--map-quantity", choices=["change", "residual"], default="change",
                   help="change = phase change since the session mean (includes the tilt); "
                        "residual = what is left after that frame's plane")
    g.add_argument("--surface-output", default=None,
                   help="3D phase-change view: a PNG (one frame) or a directory (video of all frames)")
    g.add_argument("--surface-time", type=float, default=None,
                   help="Frame time [s] for a PNG --surface-output (default: the largest tilt)")
    g.add_argument("--surface-color", choices=["residual", "change", "amplitude"], default="residual")
    g.add_argument("--framerate", type=float, default=4.0, help="Video frames per second")
    g.add_argument("--max-frames", type=int, default=None, help="Video: at most this many frames")
    g.add_argument("--keep-frames", action="store_true", help="Video: keep the PNG frames")
    g.add_argument("--csv", default=None, help="Per-frame table")
    g.add_argument("--npz", default=None, help="All arrays (snapshots, phases, directions)")
    g.add_argument("--title", default=None)
    cli.add_selection(p, array=True)
    cli.add_range_processing(p)
    cli.add_delay(p)
    cli.add_doppler_processing(p)
    cli.add_doppler_limits(p)
    g = p.add_argument_group("array")
    g.add_argument("--carrier-frequency", type=float, default=None, help="Default: stored tuning frequency")
    g.add_argument("--cable-correction", choices=["auto", "on", "off"], default="auto")
    cli.add_calibration(g)
    g = p.add_argument_group("echo and reference")
    g.add_argument("--echo-delay", type=float, default=None, help="Start the track at this delay [ms]")
    g.add_argument("--echo-doppler", type=float, default=None, help="... and this Doppler [Hz]")
    g.add_argument("--track-delay-bins", type=int, default=3, help="Search box half-width in delay bins")
    g.add_argument("--track-doppler-bins", type=int, default=3, help="Search box half-width in Doppler bins")
    g.add_argument("--min-snr-db", type=float, default=10.0, help="Frames with a weaker echo are skipped")
    g.add_argument("--reference-az", type=float, default=None, help="Reference plane azimuth [deg]")
    g.add_argument("--reference-el", type=float, default=None, help="Reference plane elevation [deg]")
    g.add_argument("--min-consistency", type=float, default=0.3,
                   help="Drop stands whose residual phase does not stay put (0-1)")
    return p


def validate(p, args):
    cli.validate(p, args)
    if (args.echo_delay is None) != (args.echo_doppler is None):
        p.error("Give both --echo-delay and --echo-doppler, or neither")
    if (args.reference_az is None) != (args.reference_el is None):
        p.error("Give both --reference-az and --reference-el, or neither")
    if args.maps < 1:
        p.error("--maps must be at least 1")


def collect(view, processor, frames, tracker, min_snr_db):
    """One snapshot per frame from the tracked echo cell."""
    rec = dict(t0=[], t1=[], delay=[], doppler=[], snr=[], X=[])
    first = None
    for frame in frames:
        cube = processor.prepare(frame["cube"]).astype(np.complex128)
        power, sigma2 = cu.normalized_power(cube)
        hit = tracker.step(power, frame["doppler_hz"])
        rec["t0"].append(frame["t0"])
        rec["t1"].append(frame["t1"])
        if hit is None:
            rec["delay"].append(np.nan), rec["doppler"].append(np.nan), rec["snr"].append(np.nan)
            rec["X"].append(np.full(cube.shape[-1], np.nan, complex))
        else:
            r, c, snr = hit
            rec["delay"].append(float(view.delay_ms[r]))
            rec["doppler"].append(float(frame["doppler_hz"][c]))
            rec["snr"].append(snr)
            rec["X"].append(cube[r, c] / np.sqrt(np.maximum(sigma2, 1e-30)))
        first = first or frame
        if (frame["index"] + 1) % 25 == 0:
            print(f"Frame {frame['index'] + 1}: {frame['t0']:.1f}-{frame['t1']:.1f} s", flush=True)
    if first is None:
        raise ValueError("No Doppler frames in the selection")
    out = {k: np.array(v) for k, v in rec.items()}
    out["used"] = np.isfinite(out["snr"]) & (out["snr"] >= min_snr_db)
    out["first"] = first
    return out


def unwrap_deg(a):
    a = np.array(a, float)
    ok = np.isfinite(a)
    a[ok] = np.degrees(np.unwrap(np.radians(a[ok])))
    return a


def run(args):
    view = cli.view_from_args(args)
    view.select_delay(args.d_min, args.d_max, args.full_delay)
    integration, hop = args.integration_time, args.hop_time
    if integration is None and hop is None and view.pulse_source() is not None and not view.file.has("rdc2"):
        integration, hop = DEFAULT_INTEGRATION_S, DEFAULT_HOP_S
        print(f"Doppler frames: {integration:g} s integration every {hop:g} s (set --integration-time/--hop-time)",
              flush=True)
    frames = view.rdc2_frames(integration, hop, args.slow_window, args.nfft_doppler, args.fd_min, args.fd_max)
    fc = float(args.carrier_frequency or view.fc)
    processor = ru.ArrayProcessor(view.stands, view.pols, fc, cable_correction=args.cable_correction,
                                  metadata=view.meta, calibration=cli.load_calibration(args))
    start = None if args.echo_delay is None else (args.echo_delay, args.echo_doppler)
    tracker = wf.EchoTracker(view.delay_ms, start, args.track_delay_bins, args.track_doppler_bins, args.min_snr_db)
    rec = collect(view, processor, frames, tracker, args.min_snr_db)
    used = rec["used"]
    if used.sum() < 3:
        raise ValueError(f"Only {int(used.sum())} frame(s) have an echo {args.min_snr_db:g} dB above the median "
                         "cell; lower --min-snr-db, start the track with --echo-delay/--echo-doppler, "
                         "or widen the delay/Doppler window")
    X = rec["X"]
    valid = np.isfinite(X) & (np.abs(np.nan_to_num(X)) > 0) & used[:, None]
    pos = processor.positions
    ref_az, ref_el, grid_az, grid_el, grid_eta = wf.reference_direction(np.where(valid, X, 0), pos, fc, used)
    how = "median of frame fits"
    if args.reference_az is not None:
        ref_az, ref_el, how = args.reference_az, args.reference_el, "given"
    res = wf.track(X, pos, fc, ref_az, ref_el, valid, min_consistency=args.min_consistency)
    ok = res["frames"]
    grid_az, grid_el = np.where(ok, grid_az, np.nan), np.where(ok, grid_el, np.nan)
    t0, t1 = rec["t0"], rec["t1"]
    t = 0.5 * (t0 + t1)
    stands = processor.stands
    first = rec["first"]

    # ---- table rows -------------------------------------------------------------------------------
    rows = view.info_rows(float(t0[0]), float(t1[-1]))
    rows += [("Integration", f"{first['integration']:.3f} s"), ("Hop", f"{float(first['hop'] or 0):.3f} s"),
             ("Slow window", str(first.get("window"))),
             ("Array", f"{len(stands)} stands, pol {'xy'[int(processor.pols[0])]}"),
             ("Cable phase", processor.station_note)]
    if processor.calibration_note:
        rows.append(("Self-cal", processor.calibration_note))
    i0 = int(np.flatnonzero(used)[0])
    rows += [("Echo start", f"{rec['delay'][i0]:.3f} ms, {rec['doppler'][i0]:+.3f} Hz"
                            + (" (given)" if start else " (strongest)")),
             ("Track box", f"+/-{args.track_delay_bins} delay, +/-{args.track_doppler_bins} Doppler bins"),
             ("Frames used", f"{int(ok.sum())} of {len(t)} (SNR >= {args.min_snr_db:g} dB)"),
             ("Reference", f"az {ref_az:.2f}, el {ref_el:.2f} deg ({how})"),
             ("Unwrapping", f"per stand in time; {res['slips']} 2-pi slip(s) repaired")]
    if res["noisy"].any():
        rows.append(("Stands dropped", ", ".join(str(int(s)) for s in stands[res["noisy"]])))
    utc = view.utc_note(float(t0[0]), float(t1[-1]))
    if utc:
        rows += [(k.strip(), v.strip()) for k, v in (line.split(":", 1) for line in utc.splitlines())]
    az, el = unwrap_deg(res["az_deg"]), res["el_deg"]
    mean_az, mean_el = res["mean_az_el_deg"]
    scatter = lambda a: np.nanstd(np.diff(a[ok])) / np.sqrt(2) if ok.sum() > 2 else np.nan  # noqa: E731
    notes = "\n".join([
        f"Mean direction: az {mean_az:.2f}, el {mean_el:.2f} deg",
        f"Az range: {np.nanmin(az):.2f} to {np.nanmax(az):.2f} deg",
        f"El range: {np.nanmin(el):.2f} to {np.nanmax(el):.2f} deg",
        f"Frame scatter: az {scatter(az):.3f}, el {scatter(el):.3f} deg",
        f"Plane coherence: median {np.nanmedian(res['coherence']):.3f}",
        f"Residual rms: median {np.nanmedian(res['rms_residual_deg']):.1f} deg"])
    print(notes, flush=True)

    # ---- summary ----------------------------------------------------------------------------------
    series = [("Azimuth [deg]", az, ("grid fit of each frame", unwrap_deg(grid_az))),
              ("Elevation [deg]", el, ("grid fit of each frame", grid_el)),
              ("Echo Doppler [Hz]", np.where(ok, rec["doppler"], np.nan), None),
              ("Echo delay [ms]", np.where(ok, rec["delay"], np.nan), None),
              ("Echo SNR [dB]", rec["snr"], None),
              ("Residual rms [deg]", res["rms_residual_deg"], None)]
    u, v = wf.direction_cosines(res["az_deg"], res["el_deg"])
    u0, v0 = wf.direction_cosines(mean_az, mean_el)
    title = args.title or "Wavefront tracking: arrival direction from per-stand phase"
    footer = "Grey: the per-frame grid fit (0.25 deg steps). Blue: plane fitted to each stand's time-unwrapped phase."
    plots.plot_wavefront_summary(t, series, (u, v, u0, v0), title, args.output, rows, notes, footer)
    print(f"Wrote {args.output}", flush=True)

    xy, k = res["xy"], res["wavenumber"]
    idx = np.flatnonzero(ok)

    # ---- phase maps -------------------------------------------------------------------------------
    if args.maps_output:
        pick = idx[np.unique(np.round(np.linspace(0, len(idx) - 1, min(args.maps, len(idx)))).astype(int))]
        data = res["change"] if args.map_quantity == "change" else res["residual"]
        labels = [f"{t[i]:.1f} s: az {res['az_deg'][i]:.2f}, el {res['el_deg'][i]:.2f}" for i in pick]
        what = "phase change since the session mean" if args.map_quantity == "change" else \
            "residual after each frame's plane"
        plots.plot_phase_maps(xy, [data[i] for i in pick], labels, f"Per-stand {what}", args.maps_output,
                              static=res["static"], info_rows=rows, notes=notes)
        print(f"Wrote {args.maps_output}", flush=True)

    # ---- 3D surface -------------------------------------------------------------------------------
    if args.surface_output:
        change = res["change"][ok]
        lim = float(np.nanpercentile(np.abs(change), 99.5)) or 1.0
        if args.surface_color == "amplitude":
            amp = 20 * np.log10(np.maximum(np.abs(np.nan_to_num(X)), 1e-12))
            amp -= np.nanmedian(np.where(valid, amp, np.nan), axis=1, keepdims=True)
            color, clim, clabel = amp, (-6.0, 6.0), "Amplitude re frame median [dB]"
        elif args.surface_color == "change":
            color, clim, clabel = res["change"], (-lim, lim), "Phase change [rad]"
        else:
            rlim = float(np.nanpercentile(np.abs(res["residual"][ok]), 99)) or 1.0
            color, clim, clabel = res["residual"], (-rlim, rlim), "Residual from the plane [rad]"

        def draw(i, path):
            a, b = k * res["du"][i], k * res["dv"][i]
            offset = float(np.nanmean(res["change"][i] - a * xy[:, 0] - b * xy[:, 1]))
            label = f"{t[i]:.1f} s: az {res['az_deg'][i]:.2f}, el {res['el_deg'][i]:.2f} deg"
            plots.plot_phase_surface(xy, res["change"][i], np.where(np.isfinite(res["change"][i]), color[i], np.nan),
                                     (a, b, offset), f"Phase change since the session mean, {label}", path,
                                     (-lim, lim), clim, clabel, rows, notes)

        if args.surface_output.lower().endswith(".png"):
            if args.surface_time is not None:
                i = int(idx[np.argmin(np.abs(t[idx] - args.surface_time))])
            else:
                i = int(idx[np.nanargmax(np.hypot(res["du"][idx], res["dv"][idx]))])
            draw(i, args.surface_output)
            print(f"Wrote {args.surface_output}", flush=True)
        else:
            video = plots.FrameVideo(args.surface_output, "wavefront_surface", args.framerate, args.keep_frames)
            for i in idx[:args.max_frames]:
                draw(int(i), video.next_path())
            video.finish()

    # ---- tables -----------------------------------------------------------------------------------
    if args.csv:
        with open(os.path.expanduser(args.csv), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["t_start_s", "t_end_s", "t_mid_s", "utc_mid", "delay_ms", "doppler_hz", "snr_db", "used",
                        "az_deg", "el_deg", "grid_az_deg", "grid_el_deg", "du", "dv", "coherence",
                        "rms_residual_deg", "stands_used"])
            for i in range(len(t)):
                when = view.utc_at(float(t[i]))
                w.writerow([f"{t0[i]:.4f}", f"{t1[i]:.4f}", f"{t[i]:.4f}",
                            when.isoformat(timespec="milliseconds") if when else "",
                            f"{rec['delay'][i]:.5f}", f"{rec['doppler'][i]:.5f}", f"{rec['snr'][i]:.2f}", int(ok[i]),
                            f"{res['az_deg'][i]:.4f}", f"{res['el_deg'][i]:.4f}", f"{grid_az[i]:.3f}",
                            f"{grid_el[i]:.3f}", f"{res['du'][i]:.6g}", f"{res['dv'][i]:.6g}",
                            f"{res['coherence'][i]:.4f}", f"{res['rms_residual_deg'][i]:.2f}",
                            int(res["stands_used"][i])])
        print(f"Wrote {args.csv}", flush=True)
    if args.npz:
        np.savez_compressed(os.path.expanduser(args.npz), t_mid_s=t, t_start_s=t0, t_end_s=t1, stands=stands,
                            positions_m=pos, snapshots=X.astype(np.complex64), used=ok, valid=res["valid"],
                            delay_ms=rec["delay"], doppler_hz=rec["doppler"], snr_db=rec["snr"],
                            phase_unwrapped=res["cum"], phase_change=res["change"], residual=res["residual"],
                            static_residual=res["static"], az_deg=res["az_deg"], el_deg=res["el_deg"],
                            du=res["du"], dv=res["dv"], coherence=res["coherence"],
                            reference_az_el_deg=np.array([ref_az, ref_el]), frequency_hz=fc)
        print(f"Wrote {args.npz}", flush=True)
    return res


def main(argv=None):
    p = build_parser()
    args = p.parse_args(argv)
    validate(p, args)
    if args.gap_fill is None:
        args.gap_fill = "zero"
    return run(args)


if __name__ == "__main__":
    cli.run(main)
