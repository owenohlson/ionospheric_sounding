"""The strong waveforms of a scan_lfm --json result, one line per delay window, for shell pipelines.

    python strong_waveforms.py 060287_010274515_lfm_scan.json [--max 3] [--min-snr 15]
                               [--min-snr-untabled 25] [--before-ms 1.5] [--after-ms 2.5]
                               [--window bw51p468_fsweep2:12:30 ...]

prints, tab separated, strongest first:

    bandwidth_hz  sweep_rate_hz  d_min_ms  d_max_ms  tag  snr_db  echo_delays_ms  window  label

(echo delays inside the window, strongest first, comma separated; "-" when a hand-set window holds none)

* a waveform is kept when its strongest echo has SNR >= --min-snr, or >= --min-snr-untabled when
  scan_lfm found it in no station's table ("not in table": noise picks have that kind); at most --max;
* its window runs from --before-ms ahead of its strongest echo to --after-ms after it (clipped to
  the sweep): the other modes of the same transmitter (E / F1 / F2, high rays, 2-3 hops) arrive
  within ~0.2-1.5 ms of each other, up to ~3 ms for long multi-hop paths. Echoes outside it (most
  likely other stations sending the same waveform at another sweep offset) are left out and listed
  on stderr;
* --window TAG:D_MIN:D_MAX sets a waveform's window by hand instead (any width; repeat it for more
  windows). TAG may be the start of a tag (bw51 matches bw51p468_fsweep2);
* tag names files made with the waveform: bw99p259_fsweep2 for -99.259 kHz at 2 Hz;
* window names the delays a file holds: d14p2-18p2ms (see rdc_info.py for the file names);
* label names its plots: the tag, or tag_window when the waveform has more than one window.
"""

import argparse
import json
import math
import sys


def tag(bandwidth_hz, rate_hz):
    return f"bw{abs(bandwidth_hz) / 1e3:.3f}".replace(".", "p") + f"_fsweep{rate_hz:g}".replace(".", "p")


def window_name(d_min, d_max):
    return f"d{d_min:.1f}-{d_max:.1f}ms".replace(".", "p")


def _snr(e):
    v = e.get("snr_db", e.get("db"))
    return v if v is not None and math.isfinite(v) else None


def parse_windows(specs):
    """['bw51:12:30', ...] -> [('bw51', 12.0, 30.0), ...]"""
    out = []
    for spec in specs or []:
        try:
            t, lo, hi = spec.split(":")
            out.append((t, float(lo), float(hi)))
        except ValueError:
            raise ValueError(f"--window {spec!r}: expected TAG:D_MIN:D_MAX") from None
        if out[-1][2] <= out[-1][1]:
            raise ValueError(f"--window {spec!r}: D_MAX must be above D_MIN")
    return out


def untabled(w):
    return str(w.get("kind", "")).startswith("not in table")


def strong(result, max_n=3, min_snr=15.0, before_ms=1.5, after_ms=2.5, windows=None, min_snr_untabled=25.0,
           skipped=None):
    """One dict per delay window of the strongest waveforms (see the module docstring).

    ``skipped`` (a list), if given, collects (tag, delay_ms, snr_db) of echoes left outside the windows."""
    waves = []
    for w in result.get("waveforms", []):
        echoes = [dict(e, delay_ms=float(e["delay_ms"])) for e in w.get("echoes", []) if e.get("delay_ms") is not None]
        snrs = [s for s in map(_snr, echoes) if s is not None]
        need = min_snr_untabled if untabled(w) else min_snr
        if not echoes or not snrs or max(snrs) < need:
            continue
        waves.append((max(snrs), w, echoes))
    waves.sort(key=lambda x: -x[0])
    out = []
    for best, w, echoes in waves[:max_n]:
        b, rate = float(w["bandwidth_hz"]), float(w["sweep_rate_hz"])
        period, t = 1e3 / rate, tag(b, rate)
        echoes = sorted(echoes, key=lambda e: -(_snr(e) if _snr(e) is not None else -1e9))     # strongest first
        manual = [(lo, hi) for prefix, lo, hi in (windows or []) if t.startswith(prefix)]
        if not manual:
            d0 = echoes[0]["delay_ms"]
            manual = [(max(0.0, d0 - before_ms), min(period, d0 + after_ms))]
        found = []
        used = set()
        for lo, hi in manual:
            inside = [e for e in echoes if lo <= e["delay_ms"] <= hi]
            used.update(id(e) for e in inside)
            s = [x for x in map(_snr, inside) if x is not None]
            found.append((lo, hi, max(s) if s else best, inside))
        if skipped is not None:
            skipped.extend((t, e["delay_ms"], _snr(e)) for e in echoes if id(e) not in used)
        found.sort(key=lambda x: -x[2])
        for lo, hi, s, es in found:
            win = window_name(lo, hi)
            out.append(dict(bandwidth_hz=b, sweep_rate_hz=rate, d_min_ms=lo, d_max_ms=hi, tag=t, snr_db=s,
                            echoes_ms=[e["delay_ms"] for e in es], window=win,
                            label=t if len(found) == 1 else f"{t}_{win}"))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("json")
    p.add_argument("--max", type=int, default=3)
    p.add_argument("--min-snr", type=float, default=15.0)
    p.add_argument("--min-snr-untabled", type=float, default=25.0,
                   help="SNR a waveform in no station's table needs [dB]")
    p.add_argument("--before-ms", type=float, default=1.5, help="Window start ahead of the strongest echo [ms]")
    p.add_argument("--after-ms", type=float, default=2.5, help="Window end after the strongest echo [ms]")
    p.add_argument("--window", action="append", default=[], metavar="TAG:D_MIN:D_MAX",
                   help="Hand-set window for the waveform(s) whose tag starts with TAG (any width)")
    args = p.parse_args(argv)
    try:
        windows = parse_windows(args.window)
    except ValueError as exc:
        p.error(str(exc))
    with open(args.json) as fh:
        result = json.load(fh)
    skipped = []
    got = strong(result, args.max, args.min_snr, args.before_ms, args.after_ms, windows, args.min_snr_untabled,
                 skipped)
    for t, d, s in skipped:
        print(f"    (left out: {t} echo at {d:.3f} ms, SNR {s if s is not None else float('nan'):.1f} dB: "
              f"outside the delay window; add it to WINDOWS to keep it)", file=sys.stderr)
    for w in got:
        echoes = ",".join(f"{d:.3f}" for d in w["echoes_ms"]) or "-"
        print(f"{w['bandwidth_hz']:.6f}\t{w['sweep_rate_hz']:g}\t{w['d_min_ms']:.3f}\t{w['d_max_ms']:.3f}\t"
              f"{w['tag']}\t{w['snr_db']:.1f}\t{echoes}\t{w['window']}\t{w['label']}")


if __name__ == "__main__":
    main()
