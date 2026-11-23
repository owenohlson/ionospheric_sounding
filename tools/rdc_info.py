"""What is inside RDC (and calibration) HDF5 files, one line each; and which existing file can be reused.

    python rdc_info.py ../data/*.h5                 # summary table
    python rdc_info.py ../data/060287_010274515_rdc2.h5 --full     # every metadata key, and the history
    python rdc_info.py --find ../data --file 060287_010274515 --bandwidth -99259.377 --sweep-rate 2 \\
        --echoes 15.696,19.145                      # an existing file to reuse (prints its path, or nothing)

Only h5py is needed (no LSL), so it runs anywhere. A file is reused for a waveform when it is
the same recording, holds per-sweep RDC1 (an RDC1, or an RDC2 that kept it), was dechirped with
the same bandwidth (within 1 Hz) and sweep rate, holds the echoes (or the requested delays), has x-pol channels
and no time cut, and was made with the station description in use ($LWA_SSMIF).
The smallest such file wins.
"""

import argparse
import glob
import json
import os
import re

STAGES = ("rdc0", "rdc1", "rdc2", "rdc2p", "rdc3")


def _h5py():
    import h5py
    return h5py


def read_info(path):
    """dict describing one file, or dict(error=...) if it is not readable."""
    out = dict(path=path, name=os.path.basename(path), size_gb=os.path.getsize(path) / 1e9)
    try:
        with _h5py().File(path, "r") as h5:
            meta = json.loads(h5.attrs.get("metadata_json", "{}"))
            stages = [s for s in STAGES if s in h5]
            out["stages"] = stages
            out["meta"] = meta
            out["kind"] = "calibration" if meta.get("kind") == "array_calibration" or "gain" in h5 else "rdc"
            if out["kind"] == "calibration":
                out["pols"] = sorted({int(p) for p in h5["pol"][...]}) if "pol" in h5 else None
                out["n_channels"] = int(len(h5["stand"])) if "stand" in h5 else None
                out["n_flagged"] = int(h5["flagged"][...].sum()) if "flagged" in h5 else None
            shapes = {s: tuple(h5[s]["real"].shape) for s in stages if "real" in h5[s]}
            out["shapes"] = shapes
            rate = meta.get("sweep_frequency_hz")
            if rate is None and "rdc1" in stages:
                rate = h5["rdc1"].attrs.get("sweep_frequency_hz")
            out["sweep_rate_hz"] = None if rate is None else float(rate)
            for s in stages:
                if "delay_ms" in h5[s]:
                    d = h5[s]["delay_ms"][...]
                    out["delay_ms"] = (float(d[0]), float(d[-1])) if len(d) else None
                    break
            pols = out.get("pols")
            for s in stages:
                if "channel_pols" in h5[s]:
                    pols = sorted({int(p) for p in h5[s]["channel_pols"][...]})
                    out["n_channels"] = int(len(h5[s]["channel_pols"]))
                    break
            out["pols"] = pols
    except Exception as exc:                            # noqa: BLE001 - report anything unreadable
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def tag(bandwidth_hz, rate_hz):
    return f"bw{abs(bandwidth_hz) / 1e3:.3f}".replace(".", "p") + f"_fsweep{rate_hz:g}".replace(".", "p")


def window(d_min, d_max):
    return f"d{d_min:.1f}-{d_max:.1f}ms".replace(".", "p")


def standard_name(i, by_name=None):
    """The standard file name for ``i`` (read_info result), or None when it can't be named."""
    if "error" in i:
        return None
    m = i["meta"]
    if i["kind"] == "calibration":
        src = os.path.basename(str(m.get("source", m.get("input", ""))))
        s = (by_name or {}).get(src)
        fid = file_id(src) or file_id(i["name"])
        pol = {(0,): "x", (1,): "y"}.get(tuple(i.get("pols") or ()))
        if not (s and fid and pol and s.get("meta", {}).get("bandwidth_hz") and s.get("sweep_rate_hz")):
            return None
        return f"{fid}_{pol}_cal_{tag(float(s['meta']['bandwidth_hz']), s['sweep_rate_hz'])}.h5"
    fid = file_id(i["name"]) or file_id(str(m.get("source", "")))
    b, rate, d = m.get("bandwidth_hz"), i.get("sweep_rate_hz"), i.get("delay_ms")
    if not (fid and b and rate and d) or "rdc0" in i["stages"]:
        return None
    stage = "rdc2" if "rdc2" in i["stages"] else i["stages"][-1]
    name = f"{fid}_{stage}_{tag(float(b), rate)}_{window(*d)}"
    if stage == "rdc2" and "integration_time_s" in m:
        name += f"_int{m['integration_time_s']:g}hop{m['hop_time_s']:g}".replace(".", "p")
    return name + ".h5"


def file_id(name):
    m = re.search(r"\d{6}_\d{9}", name)
    return m.group(0) if m else None


def _mhz(hz):
    return "?" if hz is None else f"{float(hz) / 1e6:.4f}"


def describe(i):
    if "error" in i:
        return f"{i['name']:42s} {i['size_gb']:7.2f} GB  UNREADABLE ({i['error'][:60]})"
    m = i["meta"]
    if i["kind"] == "calibration":
        pol = {(0,): "x", (1,): "y"}.get(tuple(i.get("pols") or ()), "?")
        src = os.path.basename(str(m.get("source", m.get("input", "?"))))
        return (f"{i['name']:42s} {i['size_gb']:7.2f} GB  calibration  {_mhz(m.get('frequency_hz'))} MHz"
                f"  pol {pol}  {i.get('n_channels', '?')} stands ({i.get('n_flagged', '?')} flagged)  from {src}  "
                f"station {os.path.basename(str(m.get('ssmif', m.get('station', '?'))))}")
    b = m.get("bandwidth_hz")
    rate = i.get("sweep_rate_hz")
    d = i.get("delay_ms")
    stages = "+".join(i["stages"])
    keeps = " (keeps RDC1)" if "rdc2" in i["stages"] and "rdc1" in i["stages"] else ""
    pols = {(0,): "x", (1,): "y", (0, 1): "x+y"}.get(tuple(i["pols"] or ()), "?")
    t = ""
    if m.get("tstart") is not None and m.get("tend") is not None:
        t = f"  t {m.get('tstart') or 0:.0f}-{m.get('tend') or 0:.0f} s"
    integ = f"  int {m['integration_time_s']:g}/hop {m['hop_time_s']:g} s" if "integration_time_s" in m else ""
    return (f"{i['name']:42s} {i['size_gb']:7.2f} GB  {stages}{keeps}  "
            f"bw {'?' if b is None else f'{b / 1e3:+.6f} kHz'}  sweep {'?' if rate is None else f'{rate:g} Hz'}  "
            f"delay {'all' if d is None else f'{d[0]:.1f}-{d[1]:.1f} ms'}{t}{integ}  "
            f"{i.get('n_channels', '?')} ch ({pols})  tune {_mhz(m.get('tuning_frequency_hz'))} MHz  "
            f"range {m.get('range_method') or ('dechirp' if 'rdc1' in i['stages'] or 'rdc2' in i['stages'] else '-')}"
            f"{' +sideband CLEAN' if m.get('sideband_suppression') is not None else ''}  "
            f"station {os.path.basename(str(m.get('ssmif', '?')))}")


def reusable(i, fid, bandwidth, rate, d_min=None, d_max=None, pol=0, ssmif=None, min_seconds=120.0,
             echoes=None, echo_margin=0.5, range_method="matched", sidebands=None):
    """Why this file cannot be reused for that waveform ('' if it can).

    ``sidebands``: True = it must have been exported with --suppress-sidebands, False = without,
    None = either.

    With ``echoes`` (delays, ms) the file's delay window must hold each of them at least
    ``echo_margin`` ms from its edges; otherwise it must cover ``d_min``..``d_max``."""
    if "error" in i:
        return "unreadable"
    m = i["meta"]
    if file_id(i["name"]) != fid and file_id(str(m.get("source", ""))) != fid:
        return "another recording"
    if "rdc1" not in i["stages"]:
        return "no per-sweep RDC1"
    if m.get("bandwidth_hz") is None or abs(float(m["bandwidth_hz"]) - bandwidth) > 1.0:
        return "other bandwidth"
    if i.get("sweep_rate_hz") is None or abs(i["sweep_rate_hz"] - rate) > 1e-6 * rate:
        return "other sweep rate"
    made_with = m.get("range_method") or "dechirp"          # files before 2026-11-21 were dechirped
    if range_method and made_with != range_method:
        return f"range-compressed with {made_with}, not {range_method}"
    cleaned = m.get("sideband_suppression") is not None
    if sidebands is not None and cleaned != bool(sidebands):
        return "exported with --suppress-sidebands" if cleaned else "exported without --suppress-sidebands"
    d = i.get("delay_ms")
    if d is not None and echoes:
        out = [e for e in echoes if not d[0] + echo_margin <= e <= d[1] - echo_margin]
        if out:
            return f"delays {d[0]:.1f}-{d[1]:.1f} ms miss the echo(es) at {', '.join(f'{e:.2f}' for e in out)} ms"
    elif d is not None and ((d_min is not None and d[0] > d_min + 1e-6) or (d_max is not None and d[1] < d_max - 1e-6)):
        return f"delays {d[0]:.1f}-{d[1]:.1f} ms do not cover {d_min}-{d_max} ms"
    if pol not in (i.get("pols") or []):
        return "no x-pol channels" if pol == 0 else "no y-pol channels"
    if m.get("tend") is not None and m.get("tstart") is not None and float(m["tend"]) - float(m["tstart"]) < min_seconds:
        return "a short time cut"
    if ssmif and m.get("ssmif") and os.path.basename(str(m["ssmif"])) != os.path.basename(ssmif):
        return f"made with station description {m['ssmif']}"
    return ""


def find(directory, fid, bandwidth, rate, d_min=None, d_max=None, pol=0, ssmif=None, verbose=False, echoes=None,
         echo_margin=0.5, range_method="matched", sidebands=None):
    best = None
    for path in sorted(glob.glob(os.path.join(directory, f"*{fid}*.h5"))):
        i = read_info(path)
        why = reusable(i, fid, bandwidth, rate, d_min, d_max, pol, ssmif, echoes=echoes, echo_margin=echo_margin,
                       range_method=range_method, sidebands=sidebands)
        if verbose:
            print(f"  {i['name']}: {why or 'reusable'}", flush=True)
        if not why and (best is None or i["size_gb"] < best["size_gb"]):
            best = i
    return best


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="*", help="HDF5 files to describe")
    p.add_argument("--full", action="store_true", help="Print every metadata key and the processing history")
    p.add_argument("--find", metavar="DIR", help="Look in DIR for a file to reuse (prints its path, or nothing)")
    p.add_argument("--file", help="--find: recording id, e.g. 060287_010274515")
    p.add_argument("--bandwidth", type=float, help="--find: signed bandwidth [Hz]")
    p.add_argument("--sweep-rate", type=float, help="--find: sweep rate [Hz]")
    p.add_argument("--d-min", type=float, default=None)
    p.add_argument("--d-max", type=float, default=None)
    p.add_argument("--pol", choices=["x", "y"], default="x", help="--find: polarization it must hold")
    p.add_argument("--echoes", default=None, metavar="MS,MS",
                   help="--find: echo delays the file must hold (instead of --d-min/--d-max)")
    p.add_argument("--echo-margin", type=float, default=0.5, help="--find: echoes at least this far inside [ms]")
    p.add_argument("--verbose", action="store_true", help="--find: say why each candidate is or is not reusable")
    p.add_argument("--range-method", choices=["matched", "dechirp"], default="matched",
                   help="--find: range compression the file must have been made with")
    p.add_argument("--sidebands", choices=["cleaned", "kept", "any"], default="any",
                   help="--find: cleaned = it must have been exported with --suppress-sidebands, kept = without")
    p.add_argument("--rename", action="store_true",
                   help="Print mv commands that give the files their standard names (nothing is moved)")
    args = p.parse_args(argv)
    if args.rename:
        infos = [read_info(path) for path in args.files]
        by_name = {i["name"]: i for i in infos}
        targets = {}
        for i in infos:
            new = standard_name(i, by_name)
            if new is None:
                print(f"# {i['name']}: no standard name (not a waveform-specific RDC or calibration of a recording)")
                continue
            if new == i["name"]:
                continue
            if new in targets or (new in by_name and new != i["name"]):
                print(f"# {i['name']}: would become {new}, which is taken: left as it is")
                continue
            targets[new] = i
            d = os.path.dirname(i["path"])
            print(f"mv -n {i['path']} {os.path.join(d, new)}")
            csv_old = i["path"][:-3] + ".csv"
            if i["kind"] == "calibration" and os.path.exists(csv_old):
                print(f"mv -n {csv_old} {os.path.join(d, new[:-3] + '.csv')}")
        return
    if args.find:
        if None in (args.file, args.bandwidth, args.sweep_rate):
            p.error("--find needs --file, --bandwidth and --sweep-rate")
        echoes = [float(v) for v in args.echoes.split(",") if v.strip()] if args.echoes else None
        best = find(args.find, args.file, args.bandwidth, args.sweep_rate, args.d_min, args.d_max,
                    0 if args.pol == "x" else 1, os.environ.get("LWA_SSMIF"), args.verbose, echoes, args.echo_margin,
                    args.range_method, {"cleaned": True, "kept": False}.get(args.sidebands))
        if best:
            print(best["path"])
        return
    for path in args.files:
        i = read_info(path)
        print(describe(i))
        if args.full and "meta" in i:
            history = i["meta"].pop("history", [])
            for k in sorted(i["meta"]):
                v = json.dumps(i["meta"][k])
                print(f"    {k}: {v if len(v) < 200 else v[:200] + ' ...'}")
            for h in history:
                print(f"    history: {h.get('time_utc', '')} {h.get('step', '')} {json.dumps(h.get('params', {}))[:300]}")
            print(f"    shapes: {i.get('shapes')}")


if __name__ == "__main__":
    main()
