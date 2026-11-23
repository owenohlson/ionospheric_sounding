"""Map of the receiver, the HF radar stations, their great-circle paths and reflection points.

Stations come from the HFRnet station table (built-in, or --codar-stations CSV):

    --stations PCYC@11.78,FOCF@10.3,SCRZ:1,GCAP

* ``CODE@ELEV``: the hop count is estimated from the measured elevation [deg]
  (mirror model, codar_stations.choose_hops; the label shows the virtual height,
  and "or 2E" when another hop count also fits).
* ``CODE:HOPS``: draw that many hops.
* ``CODE``: one hop.

For an n-hop path the sky reflection points (circles for one hop = the path
midpoint, squares for more) and the ground bounces (x) are drawn at equal
spacing along the great circle.

    --band 25.6 --bandwidth -150727 --sweep-rate 2 [--date 2023-12-09]

adds every station whose waveform matches that dechirp and whose sweep overlaps
the receive band (as --codar-stations does for sky maps), in grey, so the
identified paths can be seen among the alternatives. ``--points NAME:LAT,LON``
adds other sites (e.g. ionosondes). The path table is printed and can be saved
with --csv.

Elevations from the processing summary

    --from-summary [CSV] [--files 4515,5465] [--min-snr 15]

takes the stations and their elevations from the signals in the processing
summary (default: the log next to --output) that array processing gave a
direction of arrival (sky-map peaks, or the beam look direction): for each, the
stations whose documented waveform and band match the signal's file (bandwidth,
sweep rate, tuning, date) and whose bearing and a 1-3 hop path fit its
azimuth/elevation (codar_stations.identify, as the sky maps do) are drawn with
the hop count that elevation gives. A station seen in several signals is drawn
once per hop count, labelled with the elevation range and number of signals.

Ionosondes: the GIRO stations (resources/giro_stations.txt) inside the map are
drawn as stars. With GIRO FastChar files in resources/giro (the default
--iono-data; see ionosonde.py) each path's hop count comes from the measured ionosphere: the
profile of the ionosonde nearest a reflection point (within --iono-max-km), at the
time of the recording, gives the elevation each hop count would arrive at, and the
one closest to the measured elevation (within 5 deg) is drawn. Without data near a
path, the mirror model with a fixed range of layer heights is used, as before.

Coastlines and borders come from resources/map_outlines.npz (Natural Earth, shipped with
the code; see map_utils.py).
"""

import argparse
import csv
import os
import re
import sys
from datetime import date, datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import codar_stations as cs
import ionosonde
import map_utils as mu
import summary_utils

OUTLINE_STYLE = {"coast": dict(color="#6f7a86", lw=0.6), "lakes": dict(color="#6f7a86", lw=0.6),
                 "borders": dict(color="#8a8a8a", lw=0.6),
                 "states": dict(color="#b9b9b9", lw=0.35)}


def parse_stations(text):
    """'CODE[@ELEV|:HOPS],...' -> [(code, elevation or None, hops or None)]."""
    out = []
    for item in (text or "").split(","):
        item = item.strip()
        if not item:
            continue
        if "@" in item:
            code, el = item.split("@", 1)
            out.append((code.strip().upper(), float(el), None))
        elif ":" in item:
            code, hops = item.split(":", 1)
            out.append((code.strip().upper(), None, int(hops)))
        else:
            out.append((item.upper(), None, None))
    return out


def parse_points(text):
    """'NAME:LAT,LON NAME:LAT,LON' (spaces or semicolons between sites) -> [(name, lat, lon)]."""
    out = []
    for item in re.split(r"[;\s]+", (text or "").strip()):
        if item:
            name, _, ll = item.partition(":")
            lat, lon = (float(v) for v in ll.split(","))
            out.append((name, lat, lon))
    return out


def path_rows(stations, picks, rx_lat, rx_lon, max_hops=3, iono=None, when=None, freq_mhz=None, iono_max_km=1500.0):
    """One dict per requested station: geometry, hop count and reflection points.

    With ``iono`` (ionosonde.load_soundings), ``when`` and a measured elevation, the hop count is the
    one whose ionosonde-derived elevation fits best (ionosonde.predict_hops); otherwise the mirror
    model with a fixed range of layer heights (codar_stations.choose_hops)."""
    by_code = {}
    for c in stations:
        if c.frequency_hz is not None:
            by_code.setdefault(c.code.upper(), c)
    rows = []
    for code, el, hops in picks:
        c = by_code.get(code)
        if c is None:
            raise ValueError(f"Station {code} is not in the station table")
        bearing, dist = cs.bearing_distance(rx_lat, rx_lon, c.lat_deg, c.lon_deg)
        row = dict(code=c.code, name=c.name, network=c.network, lat=c.lat_deg, lon=c.lon_deg, bearing=bearing, distance=dist,
                   elevation=el, hops=hops or 1, height=None, layer=None, alternatives="", how="given" if hops else
                   "default", ionosonde=None)
        options = None
        if el is not None and iono and when is not None and not hops:
            options = ionosonde.predict_hops(rx_lat, rx_lon, c.lat_deg, c.lon_deg, freq_mhz or c.frequency_hz / 1e6,
                                             when, iono, max_hops, iono_max_km)
        if options is not None:
            best, alts = ionosonde.choose(options, el)
            if best is not None:
                row.update(hops=best.hops, height=best.height_km, layer=best.layer + ("" if best.ray == "low" else " high"),
                           ionosonde=best.station, how=f"ionosonde {best.station} ({best.station_km:.0f} km from a "
                           f"reflection point), {when:%H:%M} UTC, residual {best.residual_deg:+.1f} deg",
                           alternatives="/".join(f"{a.hops}{a.layer}" for a in alts))
            elif options:
                near = sorted(options, key=lambda o: abs(o.residual_deg))[:2]
                row["how"] = (f"ionosonde {near[0].station}: no hop count within 5 deg of el {el:.1f} (closest " +
                              ", ".join(f"{o.hops}{o.layer} at {o.elevation_deg:.1f}" for o in near) + ")")
            else:
                row["how"] = f"ionosonde: {freq_mhz or c.frequency_hz / 1e6:.2f} MHz is above the MUF of every hop count"
        if el is not None and not row["ionosonde"] and not hops:
            best, alts = cs.choose_hops(cs.hop_options(dist, el, max_hops=max_hops))
            prefix = row["how"] + "; " if row["how"].startswith("ionosonde") else ""
            if best is None:
                row["how"] = prefix + "elevation fits no hop count; drawn as 1 hop"
            else:
                row.update(hops=best.hops, height=best.height_km, layer=best.layer, alternatives="/".join(
                    f"{a.hops}{a.layer}" for a in alts), how=prefix + "mirror model (fixed layer heights)")
        elif hops:
            row["height"] = None
        (sky_lat, sky_lon), (gnd_lat, gnd_lon) = mu.hop_points(rx_lat, rx_lon, c.lat_deg, c.lon_deg, row["hops"])
        row.update(sky=list(zip(sky_lat, sky_lon)), ground=list(zip(gnd_lat, gnd_lon)))
        rows.append(row)
    return rows


def summary_signals(merged, stations, rx_lat, rx_lon, files=None, min_snr=None, sample_rate=100e3):
    """Signals of the merged processing summary with a direction of arrival, and the stations that fit.

    Returns [dict(file, delay_ms, az, el, snr, waveform, codes, label)]; ``codes`` are the stations
    whose waveform, band and date match and whose bearing and hop geometry fit (az, el)."""
    f = summary_utils._f
    want = [x.strip() for x in (files or "").split(",") if x.strip()]
    out = []
    for m in merged:
        az, el, b, rate, tune = (f(m.get(k, "")) for k in ("az_deg", "el_deg", "bandwidth_hz", "sweep_rate_hz",
                                                             "tuning_mhz"))
        if not m.get("file") or None in (az, el, b, rate, tune):
            continue
        if want and not any(m["file"] == w or m["file"].endswith(w) for w in want):
            continue
        snr = max([x for x in (f(m.get("dd_snr_db", "")), f(m.get("scan_snr_db", ""))) if x is not None],
                  default=None)
        if min_snr is not None and (snr is None or snr < min_snr):
            continue
        when = None
        if m.get("utc_start"):
            try:
                when = date.fromisoformat(m["utc_start"][:10])
            except ValueError:
                pass
        rx = cs.Receiver(rx_lat, rx_lon, tune * 1e6, sample_rate, rate, b, when)
        ident = cs.identify(az, el, cs.candidates(stations, rx), rx)
        codes = list(dict.fromkeys(x.config.code for x in ident.matches if x.passed))
        utc = None
        if m.get("utc_start"):
            try:
                utc = datetime.fromisoformat(m["utc_start"].replace("Z", "+00:00"))
                utc = utc if utc.tzinfo else utc.replace(tzinfo=timezone.utc)
            except ValueError:
                pass
        out.append(dict(file=m["file"], delay_ms=f(m.get("delay_ms", "")), az=az, el=el, snr=snr, utc=utc, freq_mhz=tune,
                        waveform=f"{b / 1e3:+.3f} kHz @ {rate:g} Hz, {tune:g} MHz", codes=codes, label=ident.label))
    return out


def summary_paths(signals, stations, rx_lat, rx_lon, max_hops=3, iono=None, iono_max_km=1500.0):
    """Path rows for the stations the summary's signals point to: one per station and hop count."""
    groups = {}
    for sig in signals:
        for code in sig["codes"]:
            r = path_rows(stations, [(code, sig["el"], None)], rx_lat, rx_lon, max_hops, iono, sig.get("utc"),
                          sig.get("freq_mhz"), iono_max_km)[0]
            g = groups.setdefault((r["code"], r["hops"]), dict(r, signals=[]))
            g["signals"].append(sig)
    rows = []
    for g in groups.values():
        els = [s["el"] for s in g["signals"]]
        if not g.get("ionosonde"):
            best, alts = cs.choose_hops(cs.hop_options(g["distance"], float(np.median(els)), max_hops=max_hops))
            if best is not None and best.hops == g["hops"]:
                g.update(height=best.height_km, layer=best.layer,
                         alternatives="/".join(f"{a.hops}{a.layer}" for a in alts))
        files = sorted({s["file"] for s in g["signals"]})
        g.update(elevation=float(np.median(els)), how=g["how"] + f"; {len(els)} summary signal(s) in {len(files)} file(s)",
                 el_range=(min(els), max(els)), files=files)
        rows.append(g)
    return sorted(rows, key=lambda r: (r["network"], r["code"], r["hops"]))


def extent_of(points, pad=3.0, min_span=12.0):
    lats = np.array([p[0] for p in points])
    lons = np.array([p[1] for p in points])
    lo0, lo1, la0, la1 = lons.min() - pad, lons.max() + pad, lats.min() - pad, lats.max() + pad
    if lo1 - lo0 < min_span * 1.4:
        c = (lo0 + lo1) / 2
        lo0, lo1 = c - min_span * 0.7, c + min_span * 0.7
    if la1 - la0 < min_span:
        c = (la0 + la1) / 2
        la0, la1 = c - min_span / 2, c + min_span / 2
    return lo0, lo1, max(la0, -85), min(la1, 85)


def draw(rows, others, points, rx, title, output, outlines=None, ionosondes=()):
    """``ionosondes``: [(ionosonde.Station, state)], state "used" (its data set a hop count), "data" (data
    loaded, not needed) or "" (none); drawn where they fall inside the map."""
    rx_name, rx_lat, rx_lon = rx
    pts = [(rx_lat, rx_lon)] + [(r["lat"], r["lon"]) for r in rows] + [(p[1], p[2]) for p in points] \
        + [(c.lat_deg, c.lon_deg) for c in others]
    lo0, lo1, la0, la1 = extent_of(pts)
    proj = mu.LambertConformal.for_extent(lo0, lo1, la0, la1)
    # view: every site and path, padded, in projected coordinates
    xs, ys = [], []
    for lat, lon in pts:
        lat_l, lon_l = mu.great_circle(rx_lat, rx_lon, lat, lon, 50)
        x, y = proj(lon_l, lat_l)
        xs += list(x)
        ys += list(y)
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    pad = 0.08 * max(x1 - x0, y1 - y0, 0.15)
    x0, x1, y0, y1 = x0 - pad, x1 + pad, y0 - pad, y1 + pad
    if (y1 - y0) < 0.45 * (x1 - x0):                         # not a thin strip when every path runs one way
        c, half = (y0 + y1) / 2, 0.225 * (x1 - x0)
        y0, y1 = c - half, c + half
    aspect = (y1 - y0) / (x1 - x0)
    n_legend = len({(r["network"], r["hops"]) for r in rows}) + 3 + bool(others) + 2
    fig, ax = plt.subplots(figsize=(12, max(5.0, min(10.0, 12 * aspect)) + 0.22 * ((n_legend + 1) // 2) + 0.6))
    ax.set_aspect("equal")
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    # graticule, labelled where the lines leave the frame (bottom for meridians, left for parallels)
    span = max(lo1 - lo0, la1 - la0)
    step = 10 if span > 25 else 5
    for lon in np.arange(np.floor((lo0 - 30) / step) * step, lo1 + 30 + step, step):
        x, y = proj(np.full(200, lon), np.linspace(-80, 85, 200))
        ax.plot(x, y, color="#e3e3e3", lw=0.5, zorder=0)
        if y[0] <= y0 <= y[-1]:
            xb = np.interp(y0, y, x)
            if x0 < xb < x1:
                ax.annotate(f"{abs(lon):g}°{'W' if lon < 0 else 'E'}", (xb, y0), xytext=(0, 3),
                            textcoords="offset points", ha="center", va="bottom", fontsize=7, color="#888888")
    for lat in np.arange(np.floor((la0 - 20) / step) * step, la1 + 20 + step, step):
        x, y = proj(np.linspace(lo0 - 60, lo1 + 60, 400), np.full(400, lat))
        ax.plot(x, y, color="#e3e3e3", lw=0.5, zorder=0)
        if x[0] <= x0 <= x[-1]:
            yb = np.interp(x0, x, y)
            if y0 < yb < y1:
                ax.annotate(f"{abs(lat):g}°{'N' if lat >= 0 else 'S'}", (x0, yb), xytext=(3, 0),
                            textcoords="offset points", ha="left", va="center", fontsize=7, color="#888888")
    outlines = mu.load_outlines() if outlines is None else outlines
    for name in ("states", "borders", "lakes", "coast"):
        if name in outlines:
            a = outlines[name]
            x, y = proj(a[:, 0], a[:, 1])
            ax.plot(x, y, zorder=1, **OUTLINE_STYLE[name])
    if not outlines:
        ax.text(0.01, 0.99, "resources/map_outlines.npz not found: no coastlines (see map_utils.py)", transform=ax.transAxes,
                fontsize=8, color="#aa0000", va="top")

    for c in others:
        x, y = proj(*mu.great_circle(rx_lat, rx_lon, c.lat_deg, c.lon_deg)[::-1])
        ax.plot(x, y, color="#c8c8c8", lw=0.6, zorder=2)
        sx, sy = proj(c.lon_deg, c.lat_deg)
        ax.plot(sx, sy, "^", color="#b0b0b0", ms=5, zorder=3)
        ax.annotate(c.code, (sx, sy), xytext=(3, -8), textcoords="offset points", fontsize=6, color="#999999")

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    groups = {}                                   # one colour and legend entry per network and hop count
    for r in rows:
        groups.setdefault((r["network"], r["hops"]), []).append(r)
    for i, ((network, hops), grp) in enumerate(groups.items()):
        col = colors[i % len(colors)]
        codes = [r["code"] for r in grp]
        dist = [r["distance"] for r in grp]
        heights = [r["height"] for r in grp if r["height"] is not None]
        what = f"{hops} hop{'s' if hops > 1 else ''}"
        if heights:
            lo, hi = min(heights), max(heights)
            what += f" {grp[0]['layer']} {lo:.0f}" + (f"–{hi:.0f}" if hi - lo >= 1 else "") + " km"
            ions = sorted({r["ionosonde"] for r in grp if r.get("ionosonde")})
            what += f" ({'/'.join(ions)})" if ions else ""
            alts = sorted({a for r in grp for a in r["alternatives"].split("/") if a})
            what += f", or {'/'.join(alts)}" if alts else ""
        els = [e for r in grp for e in (r["el_range"] if r.get("el_range") else (r["elevation"],)) if e is not None]
        if els:
            what += f"; el {min(els):.0f}°" + (f"–{max(els):.0f}°" if max(els) - min(els) >= 0.5 else "")
        n_sig = len({id(s) for r in grp for s in r.get("signals", [])})
        if n_sig:
            what += f", {n_sig} signal{'s' if n_sig > 1 else ''}"
        names = "/".join(codes[:4]) + ("/…" if len(codes) > 4 else "")
        label = (f"{network}: {names} " if network else f"{names} ") + \
            f"({min(dist):.0f}" + (f"–{max(dist):.0f}" if max(dist) - min(dist) >= 1 else "") + f" km, {what})"
        for j, r in enumerate(grp):
            lat, lon = mu.great_circle(rx_lat, rx_lon, r["lat"], r["lon"])
            x, y = proj(lon, lat)
            ax.plot(x, y, color=col, lw=1.4, zorder=4, label=label if j == 0 else None)
            sx, sy = proj(r["lon"], r["lat"])
            ax.plot(sx, sy, "^", color=col, ms=8, mec="k", mew=0.5, zorder=6)
            ax.annotate(r["code"], (sx, sy), xytext=(5, 3), textcoords="offset points", fontsize=7, color=col,
                        weight="bold")
            for la, lo in r["sky"]:
                px, py = proj(lo, la)
                ax.plot(px, py, "o" if r["hops"] == 1 else "s", ms=7, mfc="white", mec=col, mew=1.4, zorder=5)
            for la, lo in r["ground"]:
                px, py = proj(lo, la)
                ax.plot(px, py, "x", ms=6, color=col, mew=1.4, zorder=5)
    shown = {"used": False, "data": False, "": False}
    for st, state in ionosondes:
        px, py = proj(st.lon_deg, st.lat_deg)
        if not (x0 < px < x1 and y0 < py < y1):
            continue
        shown[state] = True
        style = dict(used=dict(mfc="k", mec="k", ms=10), data=dict(mfc="#777777", mec="#777777", ms=8))
        ax.plot(px, py, "*", zorder=6, **style.get(state, dict(mfc="none", mec="#9a9a9a", ms=7)))
        ax.annotate(st.code, (px, py), xytext=(4, -9), textcoords="offset points", fontsize=7 if state else 6,
                    color="k" if state == "used" else "#777777", weight="bold" if state == "used" else None)
    for name, lat, lon in points:
        px, py = proj(lon, lat)
        ax.plot(px, py, "*", color="k", ms=9, zorder=6)
        ax.annotate(name, (px, py), xytext=(4, -9), textcoords="offset points", fontsize=7)
    rxx, rxy = proj(rx_lon, rx_lat)
    ax.plot(rxx, rxy, "D", color="k", ms=8, zorder=7)
    ax.annotate(rx_name, (rxx, rxy), xytext=(6, -12), textcoords="offset points", fontsize=9, weight="bold")

    handles, labels = ax.get_legend_handles_labels()
    from matplotlib.lines import Line2D
    handles += [Line2D([], [], marker="o", ls="", mfc="white", mec="k", label="1-hop reflection (path midpoint)"),
                Line2D([], [], marker="s", ls="", mfc="white", mec="k", label="n-hop sky reflection points"),
                Line2D([], [], marker="x", ls="", color="k", label="ground bounces")]
    if others:
        handles.append(Line2D([], [], color="#c8c8c8", marker="^", mfc="#b0b0b0", mec="#b0b0b0",
                              label="other stations matching the waveform"))
    for state, label, kw in (("used", "GIRO ionosonde used for hop counts", dict(mfc="k", mec="k", ms=10)),
                             ("data", "GIRO ionosonde (data loaded, not needed)", dict(mfc="#777777", mec="#777777")),
                             ("", "GIRO ionosonde (no data loaded)", dict(mfc="none", mec="#9a9a9a"))):
        if shown[state]:
            handles.append(Line2D([], [], marker="*", ls="", label=label, **kw))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=2, fontsize=8, frameon=False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    note = "Outlines: Natural Earth. Spherical Earth, equal hops (mirror model)."
    if any(r.get("ionosonde") for r in rows):
        note += (" Hop counts from GIRO ionosonde profiles where an ionosonde is named (data CC-BY-NC-SA 4.0), "
                 "else fixed layer heights.")
    fig.text(0.99, 0.005, note, ha="right", fontsize=6, color="#888888")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stations", default="", help="CODE[@ELEV|:HOPS],... (see above)")
    p.add_argument("--receiver", default="lwa1", help="lwa1 (default), lwasv, or LAT,LON")
    p.add_argument("--receiver-name", default=None)
    p.add_argument("--band", type=float, default=None, help="Also show stations matching this tuning [MHz] ...")
    p.add_argument("--bandwidth", type=float, default=None, help="... and this signed sweep bandwidth [Hz] ...")
    p.add_argument("--sweep-rate", type=float, default=2.0, help="... and sweep rate [Hz]")
    p.add_argument("--sample-rate", type=float, default=100e3, help="Receive bandwidth [Hz] for --band")
    p.add_argument("--date", default=None, help="Observation date (YYYY-MM-DD) for --band: active stations only")
    p.add_argument("--points", default=None, help='Other sites: "NAME:LAT,LON NAME:LAT,LON" (e.g. ionosondes)')
    p.add_argument("--codar-stations", nargs="*", default=None, metavar="CSV", help="Station tables (default: built-in)")
    p.add_argument("--from-summary", nargs="?", const="auto", default=None, metavar="CSV",
                   help="Take stations and elevations from the processing summary's signals with an array direction "
                        "of arrival (default log: the one next to --output)")
    p.add_argument("--files", default=None, help="With --from-summary: only these recordings (e.g. 4515,5465)")
    p.add_argument("--min-snr", type=float, default=None, help="With --from-summary: only signals with this SNR [dB]")
    p.add_argument("--skip-empty", action="store_true",
                   help="With --from-summary: if no signal has a direction, say so and exit with status 3 "
                        "(skipped) instead of an error")
    p.add_argument("--ionosondes", choices=["giro", "none"], default="giro",
                   help="Draw the GIRO ionosondes inside the map (station list: --giro-stations)")
    p.add_argument("--giro-stations", default=ionosonde.STATION_LIST, help="GIRO station list (default: shipped)")
    p.add_argument("--iono-data", default=ionosonde.GIRO_DATA, metavar="DIR",
                   help="Folder of GIRO FastChar files (<URSI>_<date>.txt): hop counts from the ionosonde profile "
                        "nearest a reflection point instead of fixed layer heights (default: resources/giro; "
                        "'none' for fixed layer heights only)")
    p.add_argument("--iono-max-km", type=float, default=1500.0,
                   help="Use an ionosonde only within this distance of a reflection point")
    p.add_argument("--time", default=None, help="UTC (ISO 8601) of the measurements, for --stations CODE@ELEV with "
                                                "--iono-data (--from-summary uses each file's start)")
    p.add_argument("--freq", type=float, default=None,
                   help="Radar frequency [MHz] for --stations with --iono-data (default: --band, else the station's "
                        "first frequency in the table; --from-summary uses each file's tuning)")
    p.add_argument("--max-hops", type=int, default=3)
    p.add_argument("--title", default=None)
    p.add_argument("--csv", default=None, help="Also write the path table")
    p.add_argument("--output", required=True, help="Output PNG")
    p.add_argument("--summary", default=None, metavar="CSV",
                   help="Processing-summary log (default: $LFM_SUMMARY, else processing_summary.csv next to "
                        "--output; 'none' to skip)")
    args = p.parse_args(argv)

    stations = cs.load_tables(args.codar_stations or None)
    rx_lat, rx_lon = cs.receiver_location(args.receiver)
    rx_name = args.receiver_name or (args.receiver.upper().replace("LWASV", "LWA-SV")
                                     if args.receiver.lower() in ("lwa1", "lwasv") else "receiver")
    iono = None
    if args.iono_data and args.iono_data.lower() != "none" and os.path.isdir(args.iono_data):
        iono = ionosonde.load_soundings(args.iono_data)
        print(f"Ionosonde data: {', '.join(f'{k} ({s.name})' for k, s in iono.items()) or 'none'} in {args.iono_data}")
    when = None
    if args.time:
        when = datetime.fromisoformat(args.time.replace("Z", "+00:00"))
        when = when if when.tzinfo else when.replace(tzinfo=timezone.utc)
    rows = path_rows(stations, parse_stations(args.stations), rx_lat, rx_lon, args.max_hops, iono, when,
                     args.freq or args.band, args.iono_max_km)
    signals = []
    if args.from_summary:
        log = summary_utils.summary_path(args.output, None) if args.from_summary == "auto" else args.from_summary
        if not log or not os.path.exists(log):
            p.error(f"No processing summary at {log}")
        merged = summary_utils.merge(summary_utils.read_log(log))
        signals = summary_signals(merged, stations, rx_lat, rx_lon, args.files, args.min_snr, args.sample_rate)
        print(f"{len(signals)} signal(s) with an array direction of arrival in {log}:")
        for sig in signals:
            print(f"  {sig['file']} {sig['waveform']} delay {sig['delay_ms'] or float('nan'):.2f} ms  "
                  f"az {sig['az']:.1f} el {sig['el']:.1f}" + (f"  SNR {sig['snr']:.0f} dB" if sig["snr"] is not None
                                                             else "") + f"  -> {sig['label']}")
        rows += summary_paths(signals, stations, rx_lat, rx_lon, args.max_hops, iono, args.iono_max_km)
    others = []
    if args.band is not None:
        if args.bandwidth is None:
            p.error("--band needs --bandwidth (signed) and --sweep-rate")
        when = date.fromisoformat(args.date) if args.date else None
        rx = cs.Receiver(rx_lat, rx_lon, args.band * 1e6, args.sample_rate, args.sweep_rate, args.bandwidth, when)
        picked = {r["code"] for r in rows}
        seen = set()
        for c in cs.candidates(stations, rx):
            if c.code not in picked and c.code not in seen:
                seen.add(c.code)
                others.append(c)
    if not rows and not others:
        if args.skip_empty and args.from_summary is not None:
            snr = f" (SNR >= {args.min_snr:g} dB)" if args.min_snr is not None else ""
            print(f"Skipped: no signal with an array direction{snr} for "
                  f"{args.files or 'any file'} in the summary; no map drawn")
            sys.exit(3)
        p.error("Nothing to draw: give --stations, --band, or --from-summary with signals that have a direction")
    points = parse_points(args.points)
    title = args.title or (f"{rx_name}: HF radar paths and reflection points" +
                           (f" ({args.band:g} MHz)" if args.band else ""))
    marks = []
    if args.ionosondes == "giro" and os.path.exists(args.giro_stations):
        used = {r["ionosonde"] for r in rows if r.get("ionosonde")}
        for st in ionosonde.load_station_list(args.giro_stations):
            state = "used" if st.code in used else ("data" if iono and st.code in iono else "")
            same = [i for i, (o, _) in enumerate(marks) if abs(o.lat_deg - st.lat_deg) < 0.3
                    and abs(o.lon_deg - st.lon_deg) < 0.3]
            if same:                                         # one marker per site (WP937 / WI937, IF843 / AC843)
                if state and not marks[same[0]][1]:
                    marks[same[0]] = (st, state)
                continue
            marks.append((st, state))
        marks.sort(key=lambda m: m[1] != "")                 # stations with data drawn last (on top)
    draw(rows, others, points, (rx_name, rx_lat, rx_lon), title, args.output, ionosondes=marks)

    print(f"{'Station':8s} {'dist km':>8s} {'bearing':>8s} {'elev':>6s} {'hops':>5s} {'h km':>6s}  "
          f"reflection points (lat, lon)")
    for r in rows:
        el = f"{r['elevation']:.2f}" if r["elevation"] is not None else "-"
        h = f"{r['height']:.0f}{r['layer']}" if r["height"] is not None else "-"
        refl = "; ".join(f"{la:.2f}, {lo:.2f}" for la, lo in r["sky"])
        alt = f"  (also fits: {r['alternatives']})" if r["alternatives"] else ""
        print(f"{r['code']:8s} {r['distance']:8.0f} {r['bearing']:8.2f} {el:>6s} {r['hops']:5d} {h:>6s}  {refl}{alt}"
              f"\n{'':10s}hop count: {r['how']}")
    if others:
        print(f"Other stations matching the waveform: {', '.join(c.code for c in others)}")
    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["station", "name", "distance_km", "bearing_deg", "elevation_deg", "hops", "hop_from",
                        "virtual_height_km", "layer", "also_fits", "reflection_points", "ground_bounces"])
            for r in rows:
                w.writerow([r["code"], r["name"], f"{r['distance']:.1f}", f"{r['bearing']:.3f}",
                            "" if r["elevation"] is None else r["elevation"], r["hops"], r["how"],
                            "" if r["height"] is None else f"{r['height']:.1f}", r["layer"] or "", r["alternatives"],
                            " ".join(f"{la:.3f},{lo:.3f}" for la, lo in r["sky"]),
                            " ".join(f"{la:.3f},{lo:.3f}" for la, lo in r["ground"])])
    print(f"Wrote {args.output}")
    path = summary_utils.summary_path(args.output, args.summary)
    if path:
        summary_utils.record(path, "plot_station_map", None, args.output, [
            dict(kind="path", station=r["code"], network=r["network"], station_lat=r["lat"], station_lon=r["lon"],
                 distance_km=r["distance"], bearing_deg=r["bearing"], elevation_deg=r["elevation"], hops=r["hops"],
                 virtual_height_km=r["height"], layer=r["layer"], also_fits=r["alternatives"],
                 reflection_points=";".join(f"{la:.3f},{lo:.3f}" for la, lo in r["sky"]),
                 notes=f"receiver {rx_name}; hop count {r['how']}" + (
                     f"; elevation from the summary: {', '.join(r['files'])}" if r.get("files") else "")) for r in rows])


if __name__ == "__main__":
    main()
