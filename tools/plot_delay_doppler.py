"""Delay-Doppler map from any RDC file, for one receiver, several receivers, or the array.

Input stage decides the path: RDC2 frames are used directly (or recomputed from
retained RDC1 when Doppler settings change); RDC1/RDC0 are dechirped and
Doppler-processed on the fly; RDC3 plots a stored beam.

Channel handling: ``--beamformer incoherent`` (default) sums channel powers;
``conventional``/``mvdr`` steer the array (one polarization) and enable sky
maps (``--angle-output``) and the beam pattern (``--beam-pattern-output``).
Without --stands, a single-receiver plot uses the first stand; array
processing uses all stands.
"""

import csv
import os
from datetime import timedelta

import numpy as np

import cli
import plotting_utils as plots
import rdc_utils as ru
import summary_utils


def build_parser(video=False):
    p = cli.parser(__doc__)
    cli.add_input(p)
    cli.add_plot_output(p, video=video)
    cli.add_selection(p, array=None)
    cli.add_range_processing(p)
    cli.add_delay(p)
    cli.add_doppler_processing(p)
    cli.add_doppler_limits(p)
    cli.add_array(p)
    p.add_argument("--beam", type=int, default=0, help="RDC3 input: beam index to plot")
    cli.add_summary(p)
    return p


def frame_rows(frame):
    rows = [("Integration", f"{frame['integration']:.3f} s")]
    if frame.get("hop"):
        rows.append(("Hop", f"{float(frame['hop']):.3f} s"))
    rows += [("Slow window", str(frame.get("window"))), ("Doppler FFT", str(frame.get("nfft")))]
    return rows


def station_context(args, view):
    """Station tables and receiver description for --codar-stations, or None."""
    if getattr(args, "codar_stations", None) is None:
        return None
    import codar_stations as cs
    import lwa_utils
    if not view.bandwidth or not view.fs or not view.fc:
        raise ValueError("--codar-stations needs the sweep bandwidth, sample rate and tuning of the input")
    lat, lon, _ = lwa_utils.station_location()
    stations = cs.load_tables(args.codar_stations or None)      # bare flag: built-in table
    return dict(cs=cs, stations=stations, lat=lat, lon=lon, view=view)


def _per_map(markers):
    """Markers as a list of peak lists per map (also accepts one (az, el) per map)."""
    return [[m] if m is not None and len(m) and np.isscalar(m[0]) else list(m or []) for m in markers]


def identify_sources(ctx, frame_t0, markers, args, idents=None):
    """Per sky map, per peak: annotation dict (or None); and side-panel lines. ``markers`` holds a
    list of (az, el) peaks per map. ``idents`` collects the codar_stations.Identification of each
    peak (with the receiver), in map then peak order, for the summary."""
    cs, view = ctx["cs"], ctx["view"]
    zero = view.file.time_zero_utc
    when = (zero + timedelta(seconds=float(frame_t0))).date() if zero else None
    rx = cs.Receiver(ctx["lat"], ctx["lon"], float(view.fc), float(view.fs), float(view.prf),
                     float(view.bandwidth), when)
    cands = cs.candidates(ctx["stations"], rx)
    lines = cs.panel_header(rx, cands, len(ctx["stations"]), args.codar_stations or None)
    annotations = []
    markers = _per_map(markers)
    for i, peaks in enumerate(markers):
        per_map = []
        for j, (az, el) in enumerate(peaks):
            ident = cs.identify(az, el, cands, rx, az_tol=args.station_az_tol)
            if idents is not None:
                idents.append((ident, rx))
            name = f"Map {i + 1}" if len(markers) > 1 else "Sky map"
            lines += ["", f"{name}, peak {j + 1}: az {az:.1f}, el {el:.1f} deg"] + cs.describe(ident, max_rows=3)
            if ident.confident:
                pts = [(m.bearing_deg, el) for m in ident.matches if m.passed]
                per_map.append(dict(label=ident.label, points=pts))
            else:
                per_map.append(None)
        annotations.append(per_map)
    return annotations, lines


def sky_peaks(args, processor, spec, R, grid):
    """[(az, el, dB re strongest)] of one sky map: the strongest only, or up to --angle-peaks."""
    if args.angle_peaks > 1:
        return ru.peak_directions(spec, grid[0], grid[1], R, processor.positions, processor.fc, args.angle_method,
                                  args.angle_peaks, args.angle_peak_db)
    paz, pel, _ = ru.peak_direction(R, processor.positions, processor.fc, args.angle_method, args.angle_step)
    return [(paz, pel, 0.0)]


def angle_maps(args, processor, cube, power_db, grid, delay_ms, doppler_hz, whole=None):
    """Sky maps for the whole window or the strongest cells. Returns (maps, labels, markers, notes);
    ``markers`` holds the list of (az, el) peaks of each map. ``whole``: (spec, R, peaks) of the
    whole-window map when already computed (for --looks)."""
    az, el = grid
    maps, labels, markers, notes = [], [], [], []
    args._angle_cells_used = []         # (delay index, Doppler index) behind each map, for the summary
    if args.angle_cells <= 0:
        blocks = [(cube.reshape(-1, cube.shape[-1]), None)]      # every cell of the plotted window
    else:
        blocks = []
        for r, c in ru.pick_cells(power_db, args.angle_cells):
            block = cube[max(r - 1, 0):r + 2, max(c - 2, 0):c + 3]
            blocks.append((block.reshape(-1, cube.shape[-1]), (r, c)))
    for snaps, label in blocks:
        if label is None and whole is not None:
            spec, R, peaks = whole
            peaks = peaks[:args.angle_peaks] if args.angle_peaks > 1 else peaks[:1]
        else:
            spec, R = processor.angle_map(snaps, az, el, args.angle_method)
            peaks = sky_peaks(args, processor, spec, R, grid)
        maps.append(spec)
        markers.append([(p[0], p[1]) for p in peaks])
        if isinstance(label, tuple):
            args._angle_cells_used.append(label)
            r, c = label
            label = f"cell {len(maps)}: {delay_ms[r]:.3f} ms, {doppler_hz[c]:.3f} Hz\n({len(snaps)} snapshots)"
        labels.append(label)
        for j, (paz, pel, db) in enumerate(peaks):
            name = f"Map {len(maps)} peak {j + 1}" if len(blocks) > 1 else f"Peak {j + 1}"
            notes.append(f"{name}: az {paz:.1f} deg, el {pel:.1f} deg" + (f" ({db:+.1f} dB re peak 1)" if j else ""))
    if args.angle_method == "mvdr" and min(len(s) for s, _ in blocks) < 2 * len(processor.positions):
        notes.append(f"MVDR note: fewer than 2N snapshots; relying on diagonal loading {args.diagonal_loading:g}")
    return maps, labels, markers, "\n".join(notes)



def summary_frame(view, frame, delay, args):
    """Columns shared by every summary row of one frame."""
    utc = view.utc_at(frame["t0"])
    period = 1e3 / float(view.prf)
    offset = float(view.meta.get("sweep_offset_s") or 0.0) * 1e3
    return dict(utc_start=utc.isoformat() if utc else "", frame_t0_s=frame["t0"], frame_t1_s=frame["t1"],
                tuning_mhz=float(view.fc) / 1e6 if view.fc else None, bandwidth_hz=view.bandwidth,
                sweep_rate_hz=float(view.prf), sample_rate_hz=view.fs, range_method=view.range_method,
                _period=period, _offset=offset)


def _cell(frame_info, power_db, delay, fd, r, c):
    med = float(np.nanmedian(power_db))
    return dict(delay_ms=(float(delay[r]) + frame_info["_offset"]) % frame_info["_period"],
                doppler_hz=float(fd[c]), _db=float(power_db[r, c]) - med, _r=r, _c=c)


def _snr(power_db, delay, cell, busy_rows):
    """SNR of a map cell (lfm_utils.local_snr_db): noise from cells within 25 ms of its delay,
    away from the echoes and copies of the frame (``busy_rows``)."""
    import lfm_utils
    p = 10 ** (np.asarray(power_db, float) / 10)
    d = np.asarray(delay, float)
    return lfm_utils.local_snr_db(p[cell["_r"], cell["_c"]], p, d, float(d[cell["_r"]]),
                                  [float(d[r]) for r in busy_rows])[0]


SNR_KIND = "delay-Doppler cell, noise from +/-25 ms of the echo"


def _public(cell):
    return {k: v for k, v in cell.items() if not k.startswith("_")}


def echo_findings(power_db, delay, fd, frame_info, look=None, n=3, min_db=10.0):
    """'echo' rows: the strongest delay-Doppler cells of a frame (picked at >= min_db above the map
    median; recorded with their SNR, ``_snr``).

    Gating and sweep-wrap copies of a stronger cell (same Doppler, symmetric or at the wrap offset;
    lfm_utils.classify_copies) are not recorded as echoes; the echo's row lists them instead."""
    import lfm_utils
    base = {k: v for k, v in frame_info.items() if not k.startswith("_")}
    cells = [_cell(frame_info, power_db, delay, fd, r, c) for r, c in ru.pick_cells(power_db, 3 * n)]
    cells = [c for c in cells if c["_db"] >= min_db - 6.0]
    busy = [c["_r"] for c in cells]
    peaks = [dict(delay_ms=c["delay_ms"], db=c["_db"], doppler_hz=c["doppler_hz"], cell=c) for c in cells]
    echoes, _ = lfm_utils.classify_copies(peaks, frame_info["_period"], frame_info.get("sample_rate_hz"),
                                          frame_info.get("bandwidth_hz"), frame_info.get("sweep_rate_hz"),
                                          range_method=frame_info.get("range_method") or "dechirp")
    rows = []
    for e in echoes[:n]:
        if e["db"] < min_db:
            break
        copies = "; ".join(f"{x['kind']} {x['offset_ms']:+.2f} ms ({x['db'] - e['db']:+.1f} dB)" for x in e["copies"])
        rows.append(dict(base, kind="echo", snr_db=_snr(power_db, delay, e["cell"], busy), snr_kind=SNR_KIND,
                         look_az_deg=look[0] if look else None, look_el_deg=look[1] if look else None,
                         copies=copies, **_public(e["cell"])))
    return rows


def doa_findings(markers, cells, power_db, delay, fd, frame_info, idents, beam_power_db=None):
    """'doa' rows: one per sky-map peak (``markers``: list of peaks per map), with the cell it came
    from: the map's cell (--angle-cells), else the strongest cell of the beam steered at that peak
    (``beam_power_db``, one map per peak, with --looks), else the window's strongest cell; and, with
    --codar-stations, the station named for it."""
    base = {k: v for k, v in frame_info.items() if not k.startswith("_")}
    rows = []
    k = 0
    for i, peaks in enumerate(_per_map(markers)):
        for j, (az, el) in enumerate(peaks):
            pdb = power_db
            if i < len(cells):
                r, c = cells[i]
            else:
                if beam_power_db is not None and j < len(beam_power_db):
                    pdb = beam_power_db[j]
                r, c = ru.pick_cells(pdb, 1)[0]
            cell = _cell(frame_info, pdb, delay, fd, r, c)
            row = dict(base, kind="doa", az_deg=az, el_deg=el, snr_db=_snr(pdb, delay, cell, [r]),
                       snr_kind=SNR_KIND, **_public(cell))
            if k < len(idents):
                ident, rx = idents[k]
                passed = [m for m in ident.matches if m.passed]
                row["candidate_stations"] = " ".join(dict.fromkeys(m.config.code for m in passed))
                if ident.confident and passed:
                    best = passed[0]
                    row.update(station=ident.label, **summary_utils.station_columns(best, el))
                    row["reflection_points"] = summary_utils.reflection_points(rx.lat_deg, rx.lon_deg,
                                                                               best.config.lat_deg,
                                                                               best.config.lon_deg, best.hops)
                row["notes"] = ident.label
                if passed and "sign flipped vs table" in passed[0].config.sources:
                    row["notes"] += "; matched with the station table's bandwidth sign reversed"
            k += 1
            rows.append(row)
    return rows


def run(args, video=False):
    array = cli.resolve_selection(args)
    view = cli.view_from_args(args)
    only_beams = view.file.has("rdc3") and len(view.file.stages) == 1
    delay = view.select_delay(args.d_min, args.d_max, args.full_delay)
    if only_beams:
        frames = view.rdc3_frames(args.beam, args.fd_min, args.fd_max)
        array = False
    else:
        frames = view.rdc2_frames(args.integration_time, args.hop_time, args.slow_window, args.nfft_doppler,
                                  args.fd_min, args.fd_max)
    processor = None
    if array:
        processor = ru.ArrayProcessor(view.stands, view.pols, args.carrier_frequency or view.fc,
                                      "conventional" if args.beamformer == "incoherent" else args.beamformer,
                                      args.cable_correction, view.meta, args.diagonal_loading,
                                      calibration=cli.load_calibration(args))
    stations = station_context(args, view) if processor is not None and args.angle_output else None
    grid = (np.arange(-180, 180, args.angle_step), np.arange(0, 90 + 0.5 * args.angle_step, args.angle_step))
    title = args.title or ("Delay-Doppler" + (f" ({args.beamformer} beam)" if array and args.beamformer != "incoherent" else ""))
    dd_video = plots.FrameVideo(args.output, "delay_doppler", args.framerate, args.keep_frames) if video else None
    angle_video = (plots.FrameVideo(args.angle_output, "angle_maps", args.framerate, args.keep_frames)
                   if video and args.angle_output else None)
    looks = []
    findings = []
    n_frames = 0
    for frame in frames:
        if (not video and frame["index"] > 0) or (video and args.max_frames and frame["index"] >= args.max_frames):
            break
        cube, fd = frame["cube"], frame["doppler_hz"]
        n_frames += 1
        rows = view.info_rows(frame["t0"], frame["t1"]) + frame_rows(frame)
        whole = None                    # whole-window sky map (spec, R, peaks), when --looks needs it
        if processor is not None:
            cube = processor.prepare(cube)
            if args.look_az is not None:
                dirs = [(args.look_az, args.look_el, "fixed")]
            elif args.looks > 1:
                spec, R = processor.angle_map(cube.reshape(-1, cube.shape[-1]), grid[0], grid[1], args.angle_method)
                peaks = ru.peak_directions(spec, grid[0], grid[1], R, processor.positions, processor.fc,
                                           args.angle_method, max(args.looks, args.angle_peaks), args.angle_peak_db)
                whole = (spec, R, peaks)
                dirs = [(paz, pel, f"sky-map peak {j + 1}" + (f", {db:+.1f} dB re peak 1" if j else ""))
                        for j, (paz, pel, db) in enumerate(peaks[:args.looks])]
                if not dirs:
                    dirs = [(*processor.look_direction(cube, args.angle_step), "auto")]
            else:
                dirs = [(*processor.look_direction(cube, args.angle_step), "auto")]
            looks += [(frame["t0"], frame["t1"], j + 1, paz, pel) for j, (paz, pel, _) in enumerate(dirs)]
            rows += [("Array", f"{len(processor.stands)} stands, pol {'xy'[int(processor.pols[0])]}"),
                     ("Cable phase", processor.station_note)]
            if processor.calibration_note:
                rows.append(("Self-cal", processor.calibration_note))
            if args.beamformer == "incoherent":
                powers = [ru.incoherent_power(cube, args.combine)]
                rows.append(("Beamformer", f"incoherent {args.combine}"))
            else:
                Rb = processor.beam_covariance(cube)
                powers = [np.abs(processor.beamform(cube, paz, pel, Rb)) ** 2 for paz, pel, _ in dirs]
                rows.append(("Beamformer", args.beamformer))
                rows += [("Look dir" if args.looks == 1 else f"Beam {j + 1}", f"{how}: az {paz:.1f}, el {pel:.1f} deg")
                         for j, (paz, pel, how) in enumerate(dirs)]
                if args.beamformer == "mvdr":
                    rows.append(("MVDR cov", f"quietest 80% of cells, loading {args.diagonal_loading:g}"))
        elif only_beams:
            powers = [np.abs(cube[..., 0]) ** 2]
            az = view.file.axis("rdc3", "beam_az_deg")
            rows.append(("Beam", f"{args.beam}: az {az[frame['index'], args.beam]:.1f} deg" if az is not None else str(args.beam)))
        else:
            powers = [ru.incoherent_power(cube, args.combine)]
            rows.append(("Channel combine", f"incoherent {args.combine}"))
        power_dbs = [ru.to_db(p) for p in powers]
        power_db = power_dbs[0]
        frame_info = summary_frame(view, frame, delay, args)
        steered = processor is not None and args.beamformer != "incoherent"
        for j, pdb in enumerate(power_dbs):
            findings += echo_findings(pdb, delay, fd, frame_info, dirs[j][:2] if steered else None)
        notes = plots.delay_peaks_note(ru.strongest_delays(np.maximum.reduce(power_dbs).T, delay))
        footer = cli.gap_footer(frame["gaps"], view.gap_fill)
        utc = view.utc_note(frame["t0"], frame["t1"])
        if utc:
            rows += [(k.strip(), v.strip()) for k, v in (line.split(":", 1) for line in utc.splitlines())]
        idents = []
        if processor is not None and args.angle_output:
            maps, labels, markers, anotes = angle_maps(args, processor, cube, power_db, grid, delay, fd,
                                                       whole if args.angle_cells <= 0 else None)
            path = angle_video.next_path() if video else args.angle_output
            annotations, side = (identify_sources(stations, frame["t0"], markers, args, idents) if stations
                                 else (None, None))
            if stations and side is None:
                side = []                                  # keep the panel's room: every frame one layout
            same = whole is not None and args.angle_cells <= 0      # beams j = the sky map's peaks j
            findings += doa_findings(markers, getattr(args, "_angle_cells_used", []), power_db, delay, fd,
                                     frame_info, idents, power_dbs if same else None)
            used = (f"all cells of the plotted window ({delay[0]:.3f}-{delay[-1]:.3f} ms, "
                    f"{fd[0]:+.2f} to {fd[-1]:+.2f} Hz)" if args.angle_cells <= 0
                    else f"{len(maps)} strongest cells, each with its 3 delay x 5 Doppler neighbourhood")
            plots.plot_angle_maps(maps, grid[0], grid[1], labels, f"Sky map ({args.angle_method})", path,
                                  args.angle_range_db, rows + [("Sky-map data", used)], anotes, footer, markers,
                                  annotations, side)
        out = dd_video.next_path() if video else args.output
        if steered and args.looks > 1:
            same = whole is not None and args.angle_cells <= 0 and args.angle_output
            panels = []
            for j in range(args.looks):
                if j >= len(dirs):
                    panels.append((None, f"Beam {j + 1}: no further sky-map peak within {args.angle_peak_db:g} dB of peak 1"))
                    continue
                paz, pel, _ = dirs[j]
                name = ""
                if same and j < len(idents) and idents[j][0].confident:
                    name = "\n" + idents[j][0].label
                panels.append((power_dbs[j], f"Beam {j + 1}: az {paz:.1f}, el {pel:.1f} deg{name}"))
            plots.plot_delay_doppler_beams(fd, delay, panels, title, out, args.vmin, args.vmax, rows, notes, footer)
        else:
            plots.plot_delay_doppler(fd, delay, power_db, title, out, args.vmin, args.vmax, rows, notes, footer)
        if processor is not None and args.beam_pattern_output and frame["index"] == 0:
            az, el = looks[0][3:]
            pattern = ru.beam_pattern(processor.positions, processor.fc, az, el, *grid)
            plots.plot_angle_maps([pattern], grid[0], grid[1], [f"look az {az:.1f}, el {el:.1f} deg"],
                                  "Array beam pattern (conventional)", args.beam_pattern_output,
                                  args.angle_range_db, rows[:6])
        print(f"Frame {frame['index'] + 1}: {frame['t0']:.3f}-{frame['t1']:.3f} s", flush=True)
    summary = summary_utils.summary_path(args.output, args.summary)
    if summary:
        summary_utils.record(summary, "sliding_delay_doppler" if video else "plot_delay_doppler", args.input,
                             args.output, findings, total_frames=n_frames)
    if video:
        dd_video.finish()
        if angle_video:
            angle_video.finish()
        if looks:
            with open(os.path.join(os.path.expanduser(args.output), "look_directions.csv"), "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["t_start_s", "t_end_s", "beam", "look_az_deg", "look_el_deg"])
                w.writerows(looks)
    else:
        print(f"Wrote {args.output}", flush=True)


def main(video=False):
    p = build_parser(video)
    args = p.parse_args()
    cli.validate(p, args)
    run(args, video)


if __name__ == "__main__":
    cli.run(main)
