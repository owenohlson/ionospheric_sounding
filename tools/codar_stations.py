"""Which HF radar (CODAR/WERA/LERA) station sent the sweeps we see?

Station tables are HFRnet station lists (CSV export: Station, "Station Name",
Network, Frequency [MHz], Bandwidth [kHz, signed: the sign is the sweep
direction], SweepRate [Hz], "Date Added", "Last Radial File", Latitude,
Longitude, Manufacturer). A merged table ships with the code
(``resources/hfrnet_stations.csv`` next to this file) and is used by default. It keeps
every distinct configuration seen in any HFRnet export ("Snapshots" column),
because the configuration in force on an observation date is not recorded, so
every configuration a station has had is a candidate. To fold in a newer
export:

    python tools/codar_stations.py --add HFRnet-station-list-<date>.csv

A detection is a measured direction of arrival (azimuth, elevation) in an
array product made with a known receive band (tuning, sample rate) and a known
dechirp waveform (sweep rate, signed bandwidth). Candidates are whittled down
by tests that need no free parameters, then by direction:

1. **Active**: added on or before the observation date and, if a later "Last
   Radial File" date is known for the station, still producing data then.
2. **Waveform**: same sweep rate and same signed bandwidth as the dechirp
   reference, or the opposite sign (flagged: the HFRnet lists disagree about
   some stations' sign). Only a matching sweep compresses into an echo: a chirp-rate error
   dk smears the echo over |dk| T^2 delay bins, so the test allows at most one
   bin of smear (|dB| <= 1/T for equal sweep rates), i.e. effectively exact.
3. **Band**: at least half of the station's sweep band inside the receive band.
4. **Azimuth**: |measured - great-circle bearing from the array| <= az_tol.
5. **Elevation**: the measured elevation fits a 1-3 hop sky-wave path with a
   virtual reflection height of 90-450 km (spherical Earth), within el_tol.

**Hop count.** For each candidate, the measured elevation and ground distance
give, through the mirror model, the virtual height each hop count would need
(``hop_options``). More hops means a lower reflection. The hop count chosen is
the one whose height is plausible, in this order: F region (180-450 km)
without help from the elevation tolerance; E region / low F1 (90-180 km)
without it; then the same two only within ``el_tol``. Other hop counts that
also fit exactly are kept as alternatives (``Match.alternatives``): elevation
alone cannot tell, for example, one F hop at 256 km from two E hops at
110 km. Example (LWA1 to FOCF, 3,325 km, at 10.3 deg elevation): one hop would
need 560 km, two hops 212 km, three hops 127 km, so it is two F hops with a
three-hop E alternative. ``python tools/codar_stations.py --hops CODE ELEV``
prints this table for any station.

A detection is labelled only when the survivors are unambiguous, i.e. every
other candidate fits at least ``margin`` degrees worse in azimuth than the best
survivor:

* one station survives  ->  ``station``;
* several survive, all of one network (stations sharing a frequency in a
  network are GPS-synchronised and often only a few degrees apart as seen
  from far away)  ->  ``network``;
* anything else  ->  ``ambiguous`` or ``none``; nothing is drawn.

Delay cannot be used: our delay axis is relative to our own sweep timing, and
the tables do not list the stations' sweep timing offsets.
"""

import csv
import math
import os
from dataclasses import dataclass, field, replace
from datetime import date, datetime

import numpy as np

RESOURCES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources")
BUILTIN_TABLE = os.path.join(RESOURCES, "hfrnet_stations.csv")
COLUMNS = ["Station", "Station Name", "Network", "Frequency", "Bandwidth", "SweepRate", "Date Added",
           "Last Radial File", "Latitude", "Longitude", "Manufacturer", "Snapshots"]
EARTH_RADIUS_KM = 6371.0
C_KM_S = 299_792.458
HEIGHT_RANGE_KM = (90.0, 450.0)    # virtual heights accepted for any sky-wave reflection
F_LAYER_KM = (180.0, 450.0)        # F-region virtual heights (preferred); below is E / low F1


@dataclass(frozen=True)
class StationConfig:
    code: str
    name: str
    network: str
    frequency_hz: float
    bandwidth_hz: float          # signed; sign = sweep direction (same convention as --bandwidth)
    sweep_rate_hz: float
    lat_deg: float
    lon_deg: float
    added: date = None
    last_radial: date = None
    manufacturer: str = ""
    sources: tuple = field(default=(), compare=False)


@dataclass
class Match:
    config: StationConfig
    bearing_deg: float
    distance_km: float
    az_residual_deg: float
    hops: int = None              # most plausible hop count (see choose_hops), if any fits
    height_km: float = None       # virtual reflection height for that hop count (clipped to the range)
    path_delay_ms: float = None   # sky-wave group delay for that path (for reference only)
    layer: str = None             # "F" or "E" (E region or low F1) for that height
    within_tol: bool = False      # the hop count fits only after moving the elevation by up to el_tol
    alternatives: list = field(default_factory=list)   # other HopOptions that fit exactly
    options: list = field(default_factory=list)        # every HopOption, 1..max_hops
    passed: bool = False


@dataclass
class HopOption:
    """One hop count for a measured elevation over a known ground distance (mirror model)."""
    hops: int
    height_km: float              # virtual height that reproduces the elevation exactly (inf: none)
    fits: bool                    # height inside HEIGHT_RANGE_KM
    within_tol: bool              # fits only when the elevation is moved by up to el_tol
    layer: str                    # "F", "E", or "" when it does not fit at all
    path_delay_ms: float          # group delay of that path at the (clipped) height

    def label(self):
        return f"{self.hops}{self.layer or '-'} {self.height_km:3.0f}km" if np.isfinite(self.height_km) \
            else f"{self.hops}- no refl."


@dataclass
class Identification:
    status: str                   # station | network | ambiguous | none
    label: str
    matches: list                 # Match, best first (survivors first)
    candidates: int               # active + waveform + band matches before direction tests

    @property
    def confident(self):
        return self.status in ("station", "network")


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def _date(text):
    text = (text or "").strip()
    try:
        return datetime.strptime(text, "%Y-%m-%d").date()
    except ValueError:
        return None


def _float(text):
    try:
        v = float(text)
    except (TypeError, ValueError):
        return None
    return None if v <= -9998 else v


def _snapshot_name(path):
    return os.path.splitext(os.path.basename(str(path)))[0]


def load_tables(paths=None, keep_unknown=False):
    """All distinct station configurations in one or more HFRnet CSV tables (default: the built-in one).

    Rows with an unknown value (HFRnet's -9999 placeholder, e.g. the WERA/LERA stations that
    list no bandwidth) cannot be matched, so they are left out unless ``keep_unknown`` (used when
    merging tables, so the built-in table keeps every station); their unknown fields are None.
    The latest "Last Radial File" date seen for a station applies to all its rows.
    """
    paths = [BUILTIN_TABLE] if not paths else ([paths] if isinstance(paths, str) else list(paths))
    rows, last = {}, {}
    for path in paths:
        with open(path, newline="", encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                code = r["Station"].strip()
                f, bw, rate = _float(r["Frequency"]), _float(r["Bandwidth"]), _float(r["SweepRate"])
                lat, lon = _float(r["Latitude"]), _float(r["Longitude"])
                lr = _date(r.get("Last Radial File"))
                if lr and (code not in last or lr > last[code]):
                    last[code] = lr
                if None in (f, bw, rate, lat, lon) and not keep_unknown:
                    continue
                snaps = [x for x in (r.get("Snapshots") or "").split(";") if x] or [_snapshot_name(path)]
                key = (code, _round(f, 1e6, 0), _round(bw, 1e3, 1), _round(rate, 1, 6))
                prev = rows.get(key)
                added = _date(r.get("Date Added"))
                if prev is not None:
                    added = min(d for d in (prev.added, added) if d) if (prev.added or added) else None
                    snaps = sorted(set(prev.sources) | set(snaps))
                rows[key] = StationConfig(code, r["Station Name"].strip(), r["Network"].strip(), _scale(f, 1e6),
                                          _scale(bw, 1e3), rate, lat, lon, added, lr,
                                          (r.get("Manufacturer") or "").strip(), tuple(sorted(snaps)))
    out = [StationConfig(**{**c.__dict__, "last_radial": last.get(c.code, c.last_radial)}) for c in rows.values()]
    return sorted(out, key=table_order)


def _round(v, scale, digits):
    return None if v is None else round(v * scale, digits)


def _scale(v, scale):
    return None if v is None else v * scale


def table_order(c):
    """Sort key of the built-in table: network, station, frequency, bandwidth (unknowns first)."""
    return (c.network, c.code, -1e30 if c.frequency_hz is None else c.frequency_hz,
            -1e30 if c.bandwidth_hz is None else c.bandwidth_hz)


def write_table(stations, path):
    """Write configurations in HFRnet column order, plus the snapshots each was seen in."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(COLUMNS)
        for c in stations:
            w.writerow([c.code, c.name, c.network, _text(c.frequency_hz, 1e6, 5), _text(c.bandwidth_hz, 1e3, 8),
                        _text(c.sweep_rate_hz, 1, 8), c.added.isoformat() if c.added else "",
                        c.last_radial.isoformat() if c.last_radial else "", _text(c.lat_deg, 1, 6),
                        _text(c.lon_deg, 1, 6), c.manufacturer, ";".join(c.sources)])


def _text(v, scale, digits):
    return f"{-9999:.{digits}f}" if v is None else f"{v / scale:.{digits}f}"


def merge_into_builtin(paths, table=BUILTIN_TABLE):
    """Add HFRnet exports to the built-in table (union of configurations). Returns (before, after) counts."""
    before = load_tables([table], keep_unknown=True) if os.path.exists(table) else []
    merged = load_tables(([table] if os.path.exists(table) else []) + list(paths), keep_unknown=True)
    write_table(merged, table)
    return len(before), len(merged)


# ---------------------------------------------------------------------------
# Geometry (spherical Earth)
# ---------------------------------------------------------------------------

def bearing_distance(lat1, lon1, lat2, lon2):
    """Initial great-circle bearing [deg, clockwise from north] and ground distance [km] from 1 to 2."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return math.degrees(math.atan2(y, x)) % 360.0, 2 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(a)))


def skywave(distance_km, height_km, hops=1):
    """Elevation [deg] and group path [km] of an n-hop mirror-reflection path."""
    phi = distance_km / EARTH_RADIUS_KM / (2 * hops)          # half the angle subtended by one hop
    r, rh = EARTH_RADIUS_KM, EARTH_RADIUS_KM + height_km
    elev = math.degrees(math.atan2(math.cos(phi) - r / rh, math.sin(phi)))
    leg = math.sqrt(r ** 2 + rh ** 2 - 2 * r * rh * math.cos(phi))
    return elev, 2 * hops * leg


def height_for_elevation(distance_km, elev_deg, hops=1):
    """Reflection height [km] that gives ``elev_deg`` for an n-hop path (inverse of ``skywave``)."""
    phi = distance_km / EARTH_RADIUS_KM / (2 * hops)
    b = math.radians(elev_deg)
    # law of sines in the triangle (Earth centre, ground point, reflection point)
    denom = math.cos(b + phi)
    if denom <= 0:
        return math.inf
    return EARTH_RADIUS_KM * (math.cos(b) / denom - 1.0)


def hop_options(distance_km, elev_deg, el_tol=0.0, max_hops=3, heights=HEIGHT_RANGE_KM, f_layer=F_LAYER_KM):
    """Virtual height, layer and group delay for 1..max_hops hops at this elevation and distance."""
    out = []
    for hops in range(1, max_hops + 1):
        h = height_for_elevation(distance_km, elev_deg, hops)
        exact = heights[0] <= h <= heights[1]
        tol = False
        if not exact and el_tol > 0:
            lo = height_for_elevation(distance_km, max(elev_deg - el_tol, 0.0), hops)
            hi = height_for_elevation(distance_km, elev_deg + el_tol, hops)
            tol = lo <= heights[1] and hi >= heights[0]
        used = min(max(h, heights[0]), heights[1])
        layer = "" if not (exact or tol) else ("F" if used >= f_layer[0] else "E")
        delay = skywave(distance_km, used, hops)[1] / C_KM_S * 1e3
        out.append(HopOption(hops, h, exact, tol, layer, delay))
    return out


def choose_hops(options):
    """(best, alternatives): the most plausible HopOption and the other exact fits (None, [] if none fit).

    Order: F without tolerance, E without tolerance, F within tolerance, E within tolerance;
    ties go to fewer hops.
    """
    tiers = [(True, "F"), (True, "E"), (False, "F"), (False, "E")]
    ranked = [o for exact, layer in tiers for o in options
              if o.layer == layer and (o.fits if exact else o.within_tol)]
    if not ranked:
        return None, []
    best = ranked[0]
    return best, [o for o in options if o.fits and o is not best]


def _wrap(deg):
    return (deg + 180.0) % 360.0 - 180.0


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

@dataclass
class Receiver:
    """What the array product was made from."""
    lat_deg: float
    lon_deg: float
    tuning_hz: float
    sample_rate_hz: float
    sweep_rate_hz: float
    bandwidth_hz: float            # signed, as used to dechirp
    obs_date: date = None


def band_overlap(cfg, rx):
    """Overlap of the sweep and the receive band, as a fraction of the narrower of the two (a sweep
    wider than the receive band that covers all of it counts as 1)."""
    lo, hi = cfg.frequency_hz - abs(cfg.bandwidth_hz) / 2, cfg.frequency_hz + abs(cfg.bandwidth_hz) / 2
    rlo, rhi = rx.tuning_hz - rx.sample_rate_hz / 2, rx.tuning_hz + rx.sample_rate_hz / 2
    return max(0.0, min(hi, rhi) - max(lo, rlo)) / max(min(hi - lo, rhi - rlo), 1e-9)


SIGN_FLIPPED = "sign flipped vs table"     # in StationConfig.sources of a candidate matched with the opposite sign


def waveform_match(cfg, rx, flip=False):
    """Same sweep rate and bandwidth (``flip``: the table's bandwidth with its sign reversed)."""
    if not math.isclose(cfg.sweep_rate_hz, rx.sweep_rate_hz, rel_tol=1e-6, abs_tol=1e-9):
        return False
    b = -cfg.bandwidth_hz if flip else cfg.bandwidth_hz
    if np.sign(b) != np.sign(rx.bandwidth_hz):
        return False
    period = 1.0 / rx.sweep_rate_hz
    return abs(b - rx.bandwidth_hz) * period <= 1.0     # <= one delay bin of smear


def sign_flipped(cfg):
    return SIGN_FLIPPED in cfg.sources


def active(cfg, obs_date):
    if obs_date is None:
        return True
    if cfg.added and cfg.added > obs_date:
        return False
    return not (cfg.last_radial and cfg.last_radial < obs_date)


def candidates(stations, rx, min_overlap=0.5):
    """Stations that could have produced a compressed echo in this product, regardless of direction.

    Stations listed with the opposite bandwidth sign are included too, after the others (with that
    sign corrected and SIGN_FLIPPED in their sources): the HFRnet lists disagree about the sign of
    some stations' sweeps (FOCF and VATK share 220.576 kHz at 4 Hz, 26.31 MHz, with opposite signs)."""
    def ok(c, flip):
        return active(c, rx.obs_date) and waveform_match(c, rx, flip) and band_overlap(c, rx) >= min_overlap
    same = [c for c in stations if ok(c, False)]
    return same + [replace(c, bandwidth_hz=-c.bandwidth_hz, sources=tuple(c.sources) + (SIGN_FLIPPED,))
                   for c in stations if ok(c, True)]


def identify(az_deg, el_deg, cands, rx, az_tol=4.0, el_tol=5.0, margin=3.0, heights=(90.0, 450.0),
             max_hops=3):
    """Identify the source of one detection at (az_deg, el_deg) among ``candidates``.

    Survivors: |azimuth residual| <= az_tol and the elevation fits a 1..max_hops
    path. The label is given only if every station outside the survivors' group
    fits clearly worse: its residual is at least ``margin`` degrees larger than
    the best survivor's (azimuth errors of a few degrees are normal: ionospheric
    tilts, calibration, beam width). Stations listed with the opposite bandwidth sign
    (SIGN_FLIPPED) only count for the label when no station with the sign survives.
    """
    matches = []
    for c in cands:
        bearing, dist = bearing_distance(rx.lat_deg, rx.lon_deg, c.lat_deg, c.lon_deg)
        m = Match(c, bearing, dist, _wrap(az_deg - bearing))
        m.options = hop_options(dist, el_deg, el_tol, max_hops, heights)
        best, m.alternatives = choose_hops(m.options)
        if best is not None:
            m.hops, m.layer, m.within_tol = best.hops, best.layer, best.within_tol
            m.height_km = min(max(best.height_km, heights[0]), heights[1])
            m.path_delay_ms = best.path_delay_ms
        m.passed = abs(m.az_residual_deg) <= az_tol and m.hops is not None
        matches.append(m)
    matches.sort(key=lambda m: (not m.passed, abs(m.az_residual_deg)))
    passed = [m for m in matches if m.passed]
    if not passed:
        return Identification("none", "no match", matches, len(cands))
    # stations listed with the opposite sign rank behind: they decide the label only when no station
    # with the sign fits
    pool = matches
    if any(not sign_flipped(m.config) for m in passed):
        pool = [m for m in matches if not sign_flipped(m.config)]
        for m in passed:
            if sign_flipped(m.config):
                m.passed = False                  # fits, but a station with the sign fits too
        matches.sort(key=lambda m: (not m.passed, abs(m.az_residual_deg)))
        passed = [m for m in matches if m.passed]
    best = abs(passed[0].az_residual_deg)
    stations = {m.config.code for m in passed}
    networks = {m.config.network for m in passed}

    def clear(group):
        return all(abs(m.az_residual_deg) >= best + margin for m in pool if not group(m))

    if len(stations) == 1 and clear(lambda m: m.config.code in stations):
        c = passed[0].config
        return Identification("station", f"{c.code} ({c.network})", matches, len(cands))
    if len(networks) == 1 and clear(lambda m: m.config.network in networks):
        net = next(iter(networks))
        return Identification("network", f"{net}: {'/'.join(sorted(stations))}", matches, len(cands))
    return Identification("ambiguous", f"ambiguous ({len(stations)} stations, {len(networks)} networks)",
                          matches, len(cands))


def describe(ident, max_rows=4):
    """Short lines for a side panel: the decision, then the best candidates (* = passed)."""
    lines = ["-> " + ident.label if ident.confident else "-> not labelled: " + ident.label]
    for m in ident.matches[:max_rows]:
        if m.hops:
            hop = f"{m.hops}{m.layer} {m.height_km:3.0f}km{'?' if m.within_tol else ''}"
            if m.alternatives:
                hop += " or " + "/".join(f"{a.hops}{a.layer}" for a in m.alternatives)
        else:
            hop = "elev. no fit"
        lines.append(f" {'*' if m.passed else ' '}{m.config.code + ('~' if sign_flipped(m.config) else ''):5s} "
                     f"brg{_wrap(m.bearing_deg):6.1f} "
                     f"daz{m.az_residual_deg:+5.1f} {m.distance_km:5.0f}km {hop}")
    return lines


def panel_header(rx, cands, n_configs, tables=None):
    source = "built-in table" if not tables else ", ".join(os.path.basename(t) for t in tables)
    when = rx.obs_date.isoformat() if rx.obs_date else "unknown date"
    lines = [f"HF radar sources: HFRnet {source}",
             f" ({n_configs} station configurations)",
             f"Waveform {rx.sweep_rate_hz:g} Hz, {rx.bandwidth_hz / 1e3:+.3f} kHz,",
             f"band {(rx.tuning_hz - rx.sample_rate_hz / 2) / 1e6:.4f}-{(rx.tuning_hz + rx.sample_rate_hz / 2) / 1e6:.4f}"
             f" MHz, {when}:",
             f" {len(cands)} active matching configurations"
             + (f" ({sum(sign_flipped(c) for c in cands)} listed with the opposite sign, marked ~)"
                if any(sign_flipped(c) for c in cands) else "")]
    if cands:
        nets = {}
        for c in cands:
            nets.setdefault(c.network, set()).add(c.code)
        for net, codes in sorted(nets.items()):
            lines.append(f"  {net}: {', '.join(sorted(codes))}")
    return lines


RECEIVERS = {"lwa1": (34.068894, -107.628350), "lwasv": (34.348358, -106.885783)}   # as in lwa_utils


def receiver_location(spec="lwa1"):
    """(lat, lon) from a receiver name (lwa1, lwasv) or "LAT,LON"."""
    key = str(spec).strip().lower().replace("-", "")
    if key in RECEIVERS:
        return RECEIVERS[key]
    try:
        lat, lon = (float(v) for v in str(spec).split(","))
    except ValueError as exc:
        raise ValueError(f"Receiver must be lwa1, lwasv or LAT,LON, not {spec!r}") from exc
    return lat, lon


def hop_table(code, elev_deg, stations, receiver="lwa1", el_tol=0.0, max_hops=3):
    """Text lines: the virtual height, layer and delay each hop count needs for one station."""
    configs = [c for c in stations if c.code.upper() == code.upper()]
    if not configs:
        raise ValueError(f"Station {code} is not in the table")
    c = configs[0]
    lat, lon = receiver_location(receiver)
    bearing, dist = bearing_distance(lat, lon, c.lat_deg, c.lon_deg)
    opts = hop_options(dist, elev_deg, el_tol, max_hops)
    best, alts = choose_hops(opts)
    lines = [f"{c.code} ({c.name}, {c.network}) from {receiver}: bearing {bearing:.2f} deg, {dist:.0f} km; "
             f"elevation {elev_deg:g} deg" + (f" +/- {el_tol:g}" if el_tol else ""),
             " hops  virtual height  layer  group delay  verdict"]
    for o in opts:
        verdict = "chosen" if o is best else ("also fits" if o in alts else
                                              ("within tolerance" if o.within_tol else "no"))
        h = f"{o.height_km:8.0f} km" if np.isfinite(o.height_km) else "      none"
        lines.append(f" {o.hops:4d}  {h:>14s}  {o.layer or '-':>5s}  {o.path_delay_ms:8.3f} ms  {verdict}")
    if best is None:
        lines.append(" No hop count gives a virtual height of "
                     f"{HEIGHT_RANGE_KM[0]:.0f}-{HEIGHT_RANGE_KM[1]:.0f} km at this elevation.")
    return lines


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="Show or update the built-in HFRnet station table, "
                                            "or estimate the hop count of one station's echo.")
    p.add_argument("--add", nargs="+", metavar="CSV", help="HFRnet station-list exports to merge into the table")
    p.add_argument("--table", default=BUILTIN_TABLE, help="Table to update (default: the built-in one)")
    p.add_argument("--hops", nargs=2, metavar=("STATION", "ELEVATION"),
                   help="Virtual height each hop count needs for this station at this elevation [deg]")
    p.add_argument("--receiver", default="lwa1", help="lwa1 (default), lwasv, or LAT,LON")
    p.add_argument("--el-tol", type=float, default=0.0, help="Elevation tolerance [deg] for --hops")
    p.add_argument("--max-hops", type=int, default=3)
    args = p.parse_args(argv)
    if args.add:
        before, after = merge_into_builtin(args.add, args.table)
        print(f"{args.table}: {before} -> {after} station configurations")
    stations = load_tables([args.table])
    if args.hops:
        print("\n".join(hop_table(args.hops[0], float(args.hops[1]), stations, args.receiver, args.el_tol,
                                   args.max_hops)))
        return
    snaps = sorted({s for c in stations for s in c.sources})
    print(f"{len(stations)} configurations of {len({c.code for c in stations})} stations, "
          f"{len({c.network for c in stations})} networks; snapshots: {', '.join(snaps)}")


if __name__ == "__main__":
    main()
