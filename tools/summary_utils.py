"""One place for what every run found: a shared log and a merged table with one row per signal.

Commands that record (``scan_lfm.py``, ``plot_delay_doppler.py`` / ``sliding_delay_doppler.py``,
``plot_station_map.py``) append their findings to a CSV log, by default
``processing_summary.csv`` in the folder of their ``--output`` (the parent folder for
videos), so all plots written to one folder share one summary. ``--summary PATH`` or
``$LFM_SUMMARY`` choose another file; ``--summary none`` turns it off. Re-running a
command with the same input and output replaces its earlier rows. Several jobs can
write at once (the file is locked while it is rewritten).

Each log row is one finding (``kind``):

* ``waveform`` (scan_lfm): a bandwidth/sweep rate that compresses and one of its echo
  delays with their SNR, the scan contrast, and the HFRnet stations using that waveform in the band;
* ``echo`` (delay-Doppler / sky maps): one signal over the whole run. The per-frame
  strongest cells and sky-map peaks are consolidated before writing (``consolidate``):
  cells within 0.5 ms of delay of a stronger one are the same echo whatever the frame;
  the row has the median delay, Doppler (and range), SNR and direction (and
  spreads), the station named most often and ``n_frames``. Gating and sweep-wrap
  copies are folded into their echo's ``copies``, echoes seen in fewer than 20% of the
  frames are left out, and at most 5 are kept per file and waveform, so a long video
  adds a few rows. Logs written before this are compacted at the next write;
* ``path`` (station map): a station's path from the receiver, hop count, reflection points.

After every write the log is merged into ``processing_summary_merged.csv`` next to it:
one row per signal, i.e. per file, waveform (bandwidth, sweep rate) and echo (delay
within 1 ms, mod the sweep). Delays are on the UTC-aligned sweep grid (the export's
``--offset`` is added back), so scan and delay-Doppler delays agree. Directions are the
median over frames, with the spread. Station coordinates come from the station table,
and the one-hop midpoint / n-hop reflection points from the hop count. Print it with

    python tools/summary_utils.py ../plots/processing_summary.csv [--min-db 15]
"""

import contextlib
import csv
import fcntl
import math
import os
import re
import time
from collections import Counter
from datetime import datetime, timezone

import numpy as np

LOG_NAME = "processing_summary.csv"
MAX_SIGNALS = 5          # echoes kept per file and waveform from one delay-Doppler run
MIN_FRACTION = 0.2       # ... each seen in at least this fraction of the run's frames
LOG_COLUMNS = [
    "recorded_utc", "script", "input", "output", "file", "utc_start", "frame_t0_s", "frame_t1_s", "tuning_mhz",
    "kind", "bandwidth_hz", "sweep_rate_hz", "snr_db", "snr_kind", "delay_ms", "doppler_hz",
    "look_az_deg", "look_el_deg", "az_deg", "el_deg", "station", "network", "candidate_stations",
    "station_lat", "station_lon", "distance_km", "bearing_deg", "elevation_deg", "hops", "virtual_height_km",
    "layer", "also_fits", "reflection_points", "sample_rate_hz", "copies", "n_frames", "doppler_min_hz",
    "doppler_max_hz", "az_spread_deg", "el_spread_deg", "range_method", "notes"]
MERGED_COLUMNS = [
    "file", "utc_start", "tuning_mhz", "bandwidth_hz", "sweep_rate_hz", "delay_ms", "doppler_hz",
    "scan_snr_db", "dd_snr_db", "az_deg", "el_deg", "az_spread_deg", "el_spread_deg", "n_directions",
    "station", "network", "candidate_stations", "station_lat", "station_lon", "distance_km", "bearing_deg",
    "hops", "virtual_height_km", "layer", "also_fits", "reflection_points", "copies", "frames", "scripts", "plots"]


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def file_id(path):
    """The recording a path belongs to: the LWA name (060287_010274515) if present, else the stem."""
    name = os.path.basename(str(path))
    m = re.search(r"\d{6}_\d{9}", name)
    return m.group(0) if m else os.path.splitext(name)[0]


def summary_path(output, choice=None):
    """Log path for a command writing ``output``; None when turned off."""
    choice = choice or os.environ.get("LFM_SUMMARY")
    if choice and choice.strip().lower() == "none":
        return None
    if choice:
        return os.path.abspath(os.path.expanduser(choice))
    out = os.path.abspath(os.path.expanduser(str(output))).rstrip(os.sep)
    # a plot file: its folder; a video directory: the folder that holds it
    return os.path.join(os.path.dirname(out), LOG_NAME)


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, (float, np.floating)):
        return "" if not np.isfinite(v) else f"{float(v):.10g}"
    if isinstance(v, (list, tuple)):
        return " ".join(_fmt(x) for x in v)
    return str(v)


def record(path, script, input_path, output, rows, total_frames=None, max_signals=MAX_SIGNALS,
           min_fraction=MIN_FRACTION):
    """Replace this command's earlier rows (same script, input and output) with ``rows`` and re-merge.

    Per-frame delay-Doppler findings are first consolidated (``consolidate``): one row per echo
    for the whole run, so a long video adds a few rows, not one per frame."""
    if not path:
        return None
    n_findings = len(rows)
    rows, dropped = consolidate(rows, total_frames, max_signals, min_fraction)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    key = (script, os.path.abspath(str(input_path)) if input_path else "", os.path.abspath(str(output)))
    new = []
    for r in rows:
        row = {c: "" for c in LOG_COLUMNS}
        row.update({k: _fmt(v) for k, v in r.items() if k in row})
        row.update(recorded_utc=now, script=key[0], input=key[1], output=key[2],
                   file=row["file"] or (file_id(input_path) if input_path else ""))
        new.append(row)
    with _locked(path):
        old = read_log(path)
        kept = _compact_old([r for r in old if (r["script"], r["input"], r["output"]) != key])
        tmp = path + ".tmp"
        with open(tmp, "w", newline="") as fh:
            w = csv.DictWriter(fh, LOG_COLUMNS)
            w.writeheader()
            w.writerows(kept + new)
        os.replace(tmp, path)
        merged = merge(kept + new)
        mpath = merged_path(path)
        with open(mpath + ".tmp", "w", newline="") as fh:
            w = csv.DictWriter(fh, MERGED_COLUMNS)
            w.writeheader()
            w.writerows(merged)
        os.replace(mpath + ".tmp", mpath)
    what = (f"{len(new)} signal(s) from {n_findings} per-frame finding(s)" if n_findings != len(new)
            else f"{len(new)} finding(s)")
    extra = f"; {dropped} transient or weaker delay cluster(s) left out" if dropped else ""
    print(f"Summary: {what} recorded in {path}{extra} ({len(merged)} signals in {os.path.basename(mpath)})",
          flush=True)
    return path


@contextlib.contextmanager
def _locked(path, wait_s=120.0, stale_s=600.0):
    """Exclusive lock for rewriting the log: flock where the filesystem supports it, else a lock
    directory (mkdir is atomic on network filesystems too)."""
    with open(path + ".lock", "w") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX)
            locked = True
        except OSError:
            locked = False
        if locked:
            yield
            return
    lockdir = path + ".lockdir"
    start = time.time()
    while True:
        try:
            os.mkdir(lockdir)
            break
        except FileExistsError:
            try:
                if time.time() - os.path.getmtime(lockdir) > stale_s:
                    os.rmdir(lockdir)                 # left behind by a job that died
                    continue
            except OSError:
                pass
            if time.time() - start > wait_s:
                raise TimeoutError(f"Summary log {path} is locked ({lockdir}); remove it if no job is writing")
            time.sleep(0.5)
    try:
        yield
    finally:
        os.rmdir(lockdir)


# ---------------------------------------------------------------------------
# Consolidating one run's per-frame findings
# ---------------------------------------------------------------------------

STATION_COLUMNS = ("station", "network", "station_lat", "station_lon", "distance_km", "bearing_deg",
                   "elevation_deg", "hops", "virtual_height_km", "layer", "also_fits", "reflection_points")


def _num(v):
    return _f(v) if isinstance(v, str) else (None if v is None or not np.isfinite(float(v)) else float(v))


def _spread(values):
    v = np.asarray(values, float)
    return float(np.max(np.abs(v - np.median(v)))) if v.size else None


def consolidate(rows, total_frames=None, max_signals=MAX_SIGNALS, min_fraction=MIN_FRACTION, tol_ms=0.5):
    """One run's 'echo' and 'doa' rows (one per cell per frame) -> one 'echo' row per signal.

    Cells of the same file and waveform within ``tol_ms`` of delay (mod the sweep) of a stronger
    cell are one signal, whatever the frame. Its row has the median delay, Doppler (with its range),
    SNR and direction over the frames (with the spreads), the station named most often, and
    ``n_frames``. Gating and sweep-wrap copies of a stronger signal are folded into its ``copies``.
    Signals seen in fewer than ``min_fraction`` of the frames are left out, and at most
    ``max_signals`` are kept per file and waveform (most frames first, then strongest). Other kinds
    of rows pass through. Returns (rows, number of signals left out)."""
    import lfm_utils
    dd = [r for r in rows if r.get("kind") in ("echo", "doa") and _num(r.get("delay_ms")) is not None]
    if not dd:
        return list(rows), 0
    keep = [r for r in rows if not any(r is x for x in dd)]
    total = total_frames or len({str(r.get("frame_t0_s")) for r in dd})
    # copies within each frame first: a copy can land within tol_ms of another echo (on 060287_010274515
    # the 15.70 ms echo's -5.18 ms copy is 0.57 ms from the 14.54 ms echo)
    per_frame = {}
    for r in dd:
        if r.get("kind") == "echo" and _num(r.get("snr_db")) is not None and _num(r.get("sweep_rate_hz")):
            per_frame.setdefault((r.get("file", ""), str(r.get("frame_t0_s")), round(_num(r.get("bandwidth_hz")) or 0, -1),
                                  _num(r.get("sweep_rate_hz"))), []).append(r)
    copied, moved = set(), {}
    for (file, frame, _, rate), group in per_frame.items():
        if len(group) < 2:
            continue
        peaks = [dict(delay_ms=_num(r["delay_ms"]), db=_num(r["snr_db"]), doppler_hz=_num(r.get("doppler_hz")),
                      r=r) for r in group]
        echoes, copies = lfm_utils.classify_copies(peaks, 1e3 / rate, _num(group[0].get("sample_rate_hz")),
                                                   _num(group[0].get("bandwidth_hz")), rate,
                                                   range_method=group[0].get("range_method") or "dechirp")
        for x in copies:
            main = echoes[x["copy_of"]]
            copied.add(id(x["r"]))
            moved.setdefault((file, frame), []).append((x["delay_ms"], main["delay_ms"]))
            main["r"]["copies"] = _join_copies([main["r"].get("copies"), f"{x['kind']} {x['offset_ms']:+.2f} ms "
                                                                         f"({x['db'] - main['db']:+.1f} dB)"])
    dd = [r for r in dd if id(r) not in copied]
    for r in dd:                     # a sky-map peak on a copy is the echo's direction (same path)
        if r.get("kind") == "doa":
            for c, e in moved.get((r.get("file", ""), str(r.get("frame_t0_s"))), []):
                if abs(_num(r["delay_ms"]) - c) <= 0.1:
                    r["delay_ms"] = e
                    break
    clusters = []
    for r in sorted(dd, key=lambda r: -(_num(r.get("snr_db")) if _num(r.get("snr_db")) is not None
                                        else -1e9)):
        b, rate, d = _num(r.get("bandwidth_hz")), _num(r.get("sweep_rate_hz")), _num(r.get("delay_ms"))
        period = 1e3 / rate if rate else None
        c = next((c for c in clusters if c["file"] == r.get("file", "") and c["rate"] == rate
                  and (b is None) == (c["b"] is None) and (b is None or abs(b - c["b"]) <= max(25.0, 2e-4 * abs(b)))
                  and _same_delay(c["d"], d, period, tol_ms)), None)
        if c is None:
            c = dict(file=r.get("file", ""), b=b, rate=rate, d=d, rows=[])
            clusters.append(c)
        c["rows"].append(r)

    def med(rs, col):
        v = [_num(r.get(col)) for r in rs if _num(r.get(col)) is not None]
        return (float(np.median(v)), v) if v else (None, [])

    for c in clusters:
        rs = c["rows"]
        row = {k: v for k, v in rs[0].items() if k not in STATION_COLUMNS}
        frames = {str(r.get("frame_t0_s")) for r in rs}
        row.update(kind="echo", n_frames=len(frames))
        row["delay_ms"], _ = med(rs, "delay_ms")
        row["doppler_hz"], dop = med(rs, "doppler_hz")
        row["doppler_min_hz"], row["doppler_max_hz"] = (min(dop), max(dop)) if dop else (None, None)
        row["snr_db"], st = med(rs, "snr_db")
        t0 = [_num(r.get("frame_t0_s")) for r in rs if _num(r.get("frame_t0_s")) is not None]
        t1 = [_num(r.get("frame_t1_s")) for r in rs if _num(r.get("frame_t1_s")) is not None]
        row["frame_t0_s"], row["frame_t1_s"] = (min(t0) if t0 else None), (max(t1) if t1 else None)
        for az_col, el_col, sp in (("look_az_deg", "look_el_deg", False), ("az_deg", "el_deg", True)):
            dirs = [(_num(r.get(az_col)), _num(r.get(el_col))) for r in rs]
            dirs = [x for x in dirs if None not in x]
            row[az_col] = row[el_col] = None
            if dirs:
                row[az_col], az_sp = _circ_median_deg([a for a, _ in dirs])
                row[el_col] = float(np.median([e for _, e in dirs]))
                if sp:
                    row["az_spread_deg"], row["el_spread_deg"] = az_sp, _spread([e for _, e in dirs])
        named = Counter(r["station"] for r in rs if r.get("station"))
        if named:
            label = named.most_common(1)[0][0]
            ref = next(r for r in rs if r.get("station") == label)
            row.update({k: ref.get(k) for k in STATION_COLUMNS})
        cands = []
        for r in rs:
            cands += [x for x in str(r.get("candidate_stations") or "").split() if x not in cands]
        row["candidate_stations"] = " ".join(cands)
        row["copies"] = _join_copies(r.get("copies") for r in rs)
        row["notes"] = (f"{len(frames)} of {total} frame(s); SNR median {row['snr_db']:.1f}, "
                        f"max {max(st):.1f} dB" if st else f"{len(frames)} of {total} frame(s)")
        c["row"] = row

    # gating / sweep-wrap copies of a stronger signal
    groups = {}
    for c in clusters:
        groups.setdefault((c["file"], round(c["b"] or 0.0, -1), c["rate"]), []).append(c)
    gone = set()
    for (_, _, rate), group in groups.items():
        if not rate or len(group) < 2:
            continue
        fs = next((_num(c["row"].get("sample_rate_hz")) for c in group if _num(c["row"].get("sample_rate_hz"))), None)
        peaks = [dict(delay_ms=c["row"]["delay_ms"], db=c["row"]["snr_db"] if c["row"]["snr_db"] is not None
                      else -1e9, doppler_hz=c["row"]["doppler_hz"], c=c) for c in group]
        method = next((c["row"].get("range_method") for c in group if c["row"].get("range_method")), "dechirp")
        echoes, copies = lfm_utils.classify_copies(peaks, 1e3 / rate, fs, group[0]["b"], rate, range_method=method)
        for x in copies:
            main, copy = echoes[x["copy_of"]]["c"]["row"], x["c"]
            gone.add(id(copy))
            main["copies"] = _join_copies([main["copies"], f"{x['kind']} {x['offset_ms']:+.2f} ms "
                                                           f"({x['db'] - echoes[x['copy_of']]['db']:+.1f} dB)"])
            if main.get("az_deg") is None and copy["row"].get("az_deg") is not None:
                for col in ("az_deg", "el_deg", "az_spread_deg", "el_spread_deg") + STATION_COLUMNS:
                    main[col] = copy["row"].get(col)
    need = max(1, math.ceil(min_fraction * total - 1e-9))
    out, dropped = [], 0
    for key, group in groups.items():
        good = sorted((c for c in group if id(c) not in gone), key=lambda c: (-c["row"]["n_frames"],
                      -(c["row"]["snr_db"] if c["row"]["snr_db"] is not None else -1e9)))
        kept = [c for c in good if c["row"]["n_frames"] >= need][:max_signals]
        dropped += len(good) - len(kept)
        out += [c["row"] for c in kept]
    return keep + out, dropped


def _compact_old(rows):
    """Consolidate the per-frame rows that runs logged before consolidation existed (no n_frames)."""
    runs, order = {}, []
    for r in rows:
        k = (r["script"], r["input"], r["output"])
        if k not in runs:
            runs[k] = []
            order.append(k)
        runs[k].append(r)
    out = []
    for k in order:
        rs = runs[k]
        if any(r["kind"] in ("echo", "doa") and not r["n_frames"] for r in rs):
            done, _ = consolidate([dict(r) for r in rs])
            out += [{c: _fmt(r.get(c)) if not isinstance(r.get(c), str) else r.get(c) for c in LOG_COLUMNS}
                    for r in done]
        else:
            out += rs
    return out


def _join_copies(texts):
    """Copy descriptions joined, one per kind and offset (the first seen)."""
    seen = {}
    for t in texts:
        for part in str(t or "").split(";"):
            part = part.strip()
            if part:
                seen.setdefault(part.split("(")[0].strip(), part)
    return "; ".join(seen.values())


def merged_path(path):
    root, ext = os.path.splitext(path)
    return root + "_merged" + (ext or ".csv")


def read_log(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as fh:
        return [{c: r.get(c, "") or "" for c in LOG_COLUMNS} for r in csv.DictReader(fh)]


# ---------------------------------------------------------------------------
# Merging: one row per signal
# ---------------------------------------------------------------------------

def _f(text):
    try:
        v = float(text)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _same_delay(a, b, period_ms, tol=1.0):
    d = abs(a - b) % period_ms if period_ms else abs(a - b)
    return min(d, period_ms - d) <= tol if period_ms else d <= tol


def _circ_median_deg(values):
    """Median azimuth near the circular mean, and the spread (max deviation)."""
    v = np.asarray(values, float)
    mean = math.degrees(math.atan2(np.mean(np.sin(np.radians(v))), np.mean(np.cos(np.radians(v)))))
    dev = (v - mean + 180) % 360 - 180
    return (mean + float(np.median(dev)) + 180) % 360 - 180, float(np.max(np.abs(dev - np.median(dev))))


def merge(rows):
    """Log rows -> one dict per signal (file, waveform, echo), in MERGED_COLUMNS."""
    signals = []
    for r in rows:
        if r["kind"] not in ("waveform", "echo", "doa"):
            continue
        b, rate, d = _f(r["bandwidth_hz"]), _f(r["sweep_rate_hz"]), _f(r["delay_ms"])
        period = 1e3 / rate if rate else None
        s = next((s for s in signals if s["file"] == r["file"] and b is not None and s["b"] is not None
                  and abs(s["b"] - b) <= max(25.0, 2e-4 * abs(b)) and rate == s["rate"]
                  and d is not None and _same_delay(s["d"], d, period)), None)
        if s is None:
            s = dict(file=r["file"], b=b, rate=rate, d=d if d is not None else float("nan"), rows=[])
            signals.append(s)
        s["rows"].append(r)
        if r["kind"] == "waveform" and b is not None:
            s["b"] = b                                   # the scan's bandwidth is the precise one
    paths = {}
    for r in rows:
        if r["kind"] == "path":
            paths.setdefault(r["station"], r)
    out = []
    for s in signals:
        rs = s["rows"]
        by = {k: [r for r in rs if r["kind"] == k] for k in ("waveform", "echo", "doa")}
        m = {c: "" for c in MERGED_COLUMNS}
        m.update(file=s["file"], bandwidth_hz=_fmt(s["b"]), sweep_rate_hz=_fmt(s["rate"]))
        m["utc_start"] = min((r["utc_start"] for r in rs if r["utc_start"]), default="")
        m["tuning_mhz"] = next((r["tuning_mhz"] for r in rs if r["tuning_mhz"]), "")
        delays = [_f(r["delay_ms"]) for r in rs if _f(r["delay_ms"]) is not None]
        m["delay_ms"] = _fmt(float(np.median(delays))) if delays else ""
        dop = [_f(r["doppler_hz"]) for r in by["echo"] + by["doa"] if _f(r["doppler_hz"]) is not None]
        m["doppler_hz"] = _fmt(float(np.median(dop))) if dop else ""
        sc = [_f(r["snr_db"]) for r in by["waveform"] if _f(r["snr_db"]) is not None]
        dd = [_f(r["snr_db"]) for r in by["echo"] if _f(r["snr_db"]) is not None]
        m["scan_snr_db"] = _fmt(max(sc)) if sc else ""
        m["dd_snr_db"] = _fmt(float(np.median(dd))) if dd else ""
        dd_rows = by["echo"] + by["doa"]
        dirs = [(_f(r["az_deg"]), _f(r["el_deg"])) for r in dd_rows]
        row_sp = [(_f(r["az_spread_deg"]) or 0.0, _f(r["el_spread_deg"]) or 0.0) for r in dd_rows
                  if None not in (_f(r["az_deg"]), _f(r["el_deg"]))]
        dirs = [x for x in dirs if None not in x]
        if not dirs:                                     # beam look directions as a fallback
            dirs = [(_f(r["look_az_deg"]), _f(r["look_el_deg"])) for r in by["echo"]]
            dirs = [x for x in dirs if None not in x]
            row_sp = []
        if dirs:
            az, az_sp = _circ_median_deg([a for a, _ in dirs])
            el = [e for _, e in dirs]
            el_sp = float(np.max(np.abs(np.asarray(el) - np.median(el))))
            m.update(az_deg=_fmt(az), el_deg=_fmt(float(np.median(el))),
                     az_spread_deg=_fmt(max([az_sp] + [a for a, _ in row_sp])),
                     el_spread_deg=_fmt(max([el_sp] + [e for _, e in row_sp])),
                     n_directions=str(sum(int(_f(r["n_frames"]) or 1) for r in dd_rows
                                          if None not in (_f(r["az_deg"]), _f(r["el_deg"]))) or len(dirs)))
        named = Counter(r["station"] for r in dd_rows if r["station"])
        cands = []
        for r in by["waveform"] + dd_rows:
            cands += [c for c in r["candidate_stations"].split() if c and c not in cands]
        m["candidate_stations"] = " ".join(cands)
        if named:
            label = named.most_common(1)[0][0]
            ref = next(r for r in dd_rows if r["station"] == label)
            m.update(station=label, network=ref["network"])
            for col in ("station_lat", "station_lon", "distance_km", "bearing_deg", "hops", "virtual_height_km",
                        "layer", "also_fits", "reflection_points"):
                m[col] = ref[col]
            code = label.split()[0]
            if code in paths and not m["reflection_points"]:
                m["reflection_points"] = paths[code]["reflection_points"]
        m["copies"] = _join_copies(r["copies"] for r in rs)
        m["frames"] = str(sum(int(_f(r["n_frames"])) for r in rs if _f(r["n_frames"]))
                          + len({(r["output"], r["frame_t0_s"]) for r in rs if not _f(r["n_frames"])}))
        m["scripts"] = " ".join(sorted({r["script"] for r in rs}))
        m["plots"] = " ".join(sorted({os.path.basename(r["output"]) for r in rs}))
        out.append(m)
    out = _drop_copies(out, signals)
    used = {m["station"].split()[0] for m in out if m["station"]}
    for code, r in paths.items():                         # mapped paths not tied to a recorded signal
        if code not in used:
            m = {c: "" for c in MERGED_COLUMNS}
            m.update(station=code, network=r["network"], candidate_stations="", scripts=r["script"],
                     plots=os.path.basename(r["output"]), el_deg=r["elevation_deg"])
            for col in ("station_lat", "station_lon", "distance_km", "bearing_deg", "hops", "virtual_height_km",
                        "layer", "also_fits", "reflection_points"):
                m[col] = r[col]
            out.append(m)
    out.sort(key=lambda m: (m["file"] == "", m["file"], m["tuning_mhz"], m["bandwidth_hz"], _f(m["delay_ms"]) or 0))
    return out


DIRECTION_COLUMNS = ("az_deg", "el_deg", "az_spread_deg", "el_spread_deg", "n_directions", "station", "network",
                     "station_lat", "station_lon", "distance_km", "bearing_deg", "hops", "virtual_height_km", "layer",
                     "also_fits", "reflection_points")


def _drop_copies(out, signals):
    """Fold gating and sweep-wrap copies (lfm_utils.classify_copies) into the echo they copy.

    Within one file and waveform, a weaker signal at the sweep-wrap offset from a stronger one, or one of
    a symmetric pair around it, is a copy: its row is removed, the echo's ``copies`` column says where it
    was, and the echo takes the copy's direction/station if it had none (a copy travels the same path)."""
    import lfm_utils
    groups = {}
    for m, s in zip(out, signals):
        groups.setdefault((m["file"], round(s["b"] or 0.0, -1), s["rate"]), []).append((m, s))
    drop = set()
    for (_, _, rate), group in groups.items():
        if not rate or len(group) < 2:
            continue
        fs = next((_f(r["sample_rate_hz"]) for _, s in group for r in s["rows"] if _f(r["sample_rate_hz"])), None)
        use_scan = all(m["scan_snr_db"] for m, _ in group)
        peaks = []
        for m, _ in group:
            db = _f(m["scan_snr_db"] if use_scan else m["dd_snr_db"] or m["scan_snr_db"])
            if db is not None and m["delay_ms"]:
                peaks.append(dict(delay_ms=_f(m["delay_ms"]), db=db, doppler_hz=_f(m["doppler_hz"]), row=m))
        method = next((r.get("range_method") for _, s in group for r in s["rows"] if r.get("range_method")),
                      "dechirp")
        echoes, copies = lfm_utils.classify_copies(peaks, 1e3 / rate, fs, group[0][1]["b"], rate,
                                                   range_method=method)
        for x in copies:
            main, copy = echoes[x["copy_of"]], x["row"]
            drop.add(id(copy))
            row = main["row"]
            if f"{x['offset_ms']:+.2f} ms" not in row["copies"]:
                row["copies"] = "; ".join(filter(None, [row["copies"], f"{x['kind']} {x['offset_ms']:+.2f} ms "
                                                                       f"({x['db'] - main['db']:+.1f} dB)"]))
            if not row["az_deg"] and copy["az_deg"]:
                for col in DIRECTION_COLUMNS:
                    row[col] = copy[col]
    return [m for m in out if id(m) not in drop]


# ---------------------------------------------------------------------------
# Helpers for the recording commands
# ---------------------------------------------------------------------------

def station_columns(match, elevation=None):
    """Log columns for a codar_stations.Match (named station, path geometry, hop count)."""
    c = match.config
    cols = dict(station_lat=c.lat_deg, station_lon=c.lon_deg, distance_km=match.distance_km,
                bearing_deg=match.bearing_deg, hops=match.hops, virtual_height_km=match.height_km,
                layer=match.layer, also_fits="/".join(f"{a.hops}{a.layer}" for a in match.alternatives),
                network=c.network, elevation_deg=elevation)
    return cols


def reflection_points(rx_lat, rx_lon, lat, lon, hops):
    import map_utils
    (sky_lat, sky_lon), _ = map_utils.hop_points(rx_lat, rx_lon, lat, lon, int(hops or 1))
    return ";".join(f"{a:.3f},{b:.3f}" for a, b in zip(sky_lat, sky_lon))


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="Print the merged processing summary (one row per signal).")
    p.add_argument("log", nargs="?", default=os.environ.get("LFM_SUMMARY") or LOG_NAME,
                   help="processing_summary.csv (default: $LFM_SUMMARY or ./processing_summary.csv)")
    p.add_argument("--min-db", type=float, default=None,
                   help="Only signals whose scan or delay-Doppler SNR reaches this [dB]")
    p.add_argument("--csv", default=None, help="Also write the (filtered) merged table here")
    args = p.parse_args(argv)
    rows = merge(read_log(args.log))
    if args.min_db is not None:
        rows = [m for m in rows if max([_f(m["scan_snr_db"]) or -1e9, _f(m["dd_snr_db"]) or -1e9])
                >= args.min_db or not m["file"]]
    cols = [("file", 16), ("tuning_mhz", 7), ("bandwidth_hz", 11), ("sweep_rate_hz", 5), ("delay_ms", 8),
            ("doppler_hz", 7), ("scan_snr_db", 6), ("dd_snr_db", 6), ("az_deg", 7), ("el_deg", 6),
            ("station", 18), ("hops", 4), ("virtual_height_km", 6), ("candidate_stations", 30)]
    heads = ["file", "MHz", "B [Hz]", "rate", "delay", "Dopp", "SNRsc", "SNRdd", "az", "el", "station", "hops", "h'",
             "stations with this waveform"]
    print("  ".join(f"{h:<{w}}" for h, (_, w) in zip(heads, cols)))
    for m in rows:
        cells = []
        for col, w in cols:
            v = m[col]
            f = _f(v)
            if f is not None and col not in ("file",):
                v = f"{f:.4g}" if abs(f) < 1e4 else f"{f:.1f}"
            cells.append(f"{str(v)[:w]:<{w}}")
        print("  ".join(cells))
    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, MERGED_COLUMNS)
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()
