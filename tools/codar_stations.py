"""Which HF radar (CODAR/WERA/LERA) station sent the sweeps we see?

Station tables are HFRnet station lists (CSV export: Station, "Station Name",
Network, Frequency [MHz], Bandwidth [kHz, signed: the sign is the sweep
direction], SweepRate [Hz], "Date Added", "Last Radial File", Latitude,
Longitude, Manufacturer). Several tables (snapshots taken at different times)
can be given; every distinct configuration a station has had is a candidate,
because the configuration in force on the observation date is not recorded.

A detection is a measured direction of arrival (azimuth, elevation) in an
array product made with a known receive band (tuning, sample rate) and a known
dechirp waveform (sweep rate, signed bandwidth). Candidates are whittled down
by tests that need no free parameters, then by direction:

1. **Active**: added on or before the observation date and, if a later "Last
   Radial File" date is known for the station, still producing data then.
2. **Waveform**: same sweep rate and same signed bandwidth as the dechirp
   reference. Only a matching sweep compresses into an echo: a chirp-rate error
   dk smears the echo over |dk| T^2 delay bins, so the test allows at most one
   bin of smear (|dB| <= 1/T for equal sweep rates), i.e. effectively exact.
3. **Band**: at least half of the station's sweep band inside the receive band.
4. **Azimuth**: |measured - great-circle bearing from the array| <= az_tol.
5. **Elevation**: the measured elevation fits a 1-3 hop sky-wave path with a
   virtual reflection height of 90-450 km (spherical Earth), within el_tol.

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
from dataclasses import dataclass, field
from datetime import date, datetime

import numpy as np

EARTH_RADIUS_KM = 6371.0
C_KM_S = 299_792.458


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
    hops: int = None              # hop count whose elevation range fits, if any
    height_km: float = None       # reflection height that reproduces the measured elevation
    path_delay_ms: float = None   # sky-wave group delay for that path (for reference only)
    passed: bool = False


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


def load_tables(paths):
    """All distinct station configurations in one or more HFRnet CSV tables.

    Rows with an unknown waveform (-9999) are kept out: they cannot be matched.
    The latest "Last Radial File" date seen for a station applies to all its rows.
    """
    rows = {}
    last = {}
    for path in ([paths] if isinstance(paths, str) else paths):
        with open(path, newline="", encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                code = r["Station"].strip()
                f, bw, rate = _float(r["Frequency"]), _float(r["Bandwidth"]), _float(r["SweepRate"])
                lat, lon = _float(r["Latitude"]), _float(r["Longitude"])
                lr = _date(r.get("Last Radial File"))
                if lr and (code not in last or lr > last[code]):
                    last[code] = lr
                if None in (f, bw, rate, lat, lon):
                    continue
                cfg = StationConfig(code, r["Station Name"].strip(), r["Network"].strip(), f * 1e6, bw * 1e3,
                                    rate, lat, lon, _date(r.get("Date Added")), lr,
                                    r.get("Manufacturer", "").strip())
                key = (code, round(cfg.frequency_hz), round(cfg.bandwidth_hz, 1), round(rate, 6))
                prev = rows.get(key)
                rows[key] = cfg if prev is None else prev
                rows[key] = StationConfig(**{**rows[key].__dict__,
                                             "sources": tuple(sorted(set(rows[key].sources) | {str(path)}))})
    out = []
    for cfg in rows.values():
        out.append(StationConfig(**{**cfg.__dict__, "last_radial": last.get(cfg.code, cfg.last_radial)}))
    return sorted(out, key=lambda c: (c.network, c.code, c.frequency_hz))


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
    lo, hi = cfg.frequency_hz - abs(cfg.bandwidth_hz) / 2, cfg.frequency_hz + abs(cfg.bandwidth_hz) / 2
    rlo, rhi = rx.tuning_hz - rx.sample_rate_hz / 2, rx.tuning_hz + rx.sample_rate_hz / 2
    return max(0.0, min(hi, rhi) - max(lo, rlo)) / max(hi - lo, 1e-9)


def waveform_match(cfg, rx):
    if not math.isclose(cfg.sweep_rate_hz, rx.sweep_rate_hz, rel_tol=1e-6, abs_tol=1e-9):
        return False
    if np.sign(cfg.bandwidth_hz) != np.sign(rx.bandwidth_hz):
        return False
    period = 1.0 / rx.sweep_rate_hz
    return abs(cfg.bandwidth_hz - rx.bandwidth_hz) * period <= 1.0     # <= one delay bin of smear


def active(cfg, obs_date):
    if obs_date is None:
        return True
    if cfg.added and cfg.added > obs_date:
        return False
    return not (cfg.last_radial and cfg.last_radial < obs_date)


def candidates(stations, rx, min_overlap=0.5):
    """Stations that could have produced a compressed echo in this product, regardless of direction."""
    return [c for c in stations if active(c, rx.obs_date) and waveform_match(c, rx)
            and band_overlap(c, rx) >= min_overlap]


def identify(az_deg, el_deg, cands, rx, az_tol=4.0, el_tol=5.0, margin=3.0, heights=(90.0, 450.0),
             max_hops=3):
    """Identify the source of one detection at (az_deg, el_deg) among ``candidates``.

    Survivors: |azimuth residual| <= az_tol and the elevation fits a 1..max_hops
    path. The label is given only if every station outside the survivors' group
    fits clearly worse: its residual is at least ``margin`` degrees larger than
    the best survivor's (azimuth errors of a few degrees are normal: ionospheric
    tilts, calibration, beam width).
    """
    matches = []
    for c in cands:
        bearing, dist = bearing_distance(rx.lat_deg, rx.lon_deg, c.lat_deg, c.lon_deg)
        m = Match(c, bearing, dist, _wrap(az_deg - bearing))
        for hops in range(1, max_hops + 1):
            lo = skywave(dist, heights[0], hops)[0]
            hi = skywave(dist, heights[1], hops)[0]
            if lo - el_tol <= el_deg <= hi + el_tol:
                h = min(max(height_for_elevation(dist, el_deg, hops), heights[0]), heights[1])
                m.hops, m.height_km = hops, h
                m.path_delay_ms = skywave(dist, h, hops)[1] / C_KM_S * 1e3
                break
        m.passed = abs(m.az_residual_deg) <= az_tol and m.hops is not None
        matches.append(m)
    matches.sort(key=lambda m: (not m.passed, abs(m.az_residual_deg)))
    passed = [m for m in matches if m.passed]
    if not passed:
        return Identification("none", "no match", matches, len(cands))
    best = abs(passed[0].az_residual_deg)
    stations = {m.config.code for m in passed}
    networks = {m.config.network for m in passed}

    def clear(group):
        return all(abs(m.az_residual_deg) >= best + margin for m in matches if not group(m))

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
        hop = f"{m.hops}hop {m.height_km:3.0f}km" if m.hops else "elev. no fit"
        lines.append(f" {'*' if m.passed else ' '}{m.config.code:5s} brg{_wrap(m.bearing_deg):6.1f} "
                     f"daz{m.az_residual_deg:+5.1f} {m.distance_km:5.0f}km {hop}")
    return lines


def panel_header(rx, cands, n_configs, tables):
    import os
    when = rx.obs_date.isoformat() if rx.obs_date else "unknown date"
    lines = ["HF radar sources (HFRnet tables:", *[f"  {os.path.basename(t)}" for t in tables],
             f" {n_configs} station configurations)",
             f"Waveform {rx.sweep_rate_hz:g} Hz, {rx.bandwidth_hz / 1e3:+.3f} kHz,",
             f"band {(rx.tuning_hz - rx.sample_rate_hz / 2) / 1e6:.4f}-{(rx.tuning_hz + rx.sample_rate_hz / 2) / 1e6:.4f}"
             f" MHz, {when}:",
             f" {len(cands)} active matching configurations"]
    if cands:
        nets = {}
        for c in cands:
            nets.setdefault(c.network, set()).add(c.code)
        for net, codes in sorted(nets.items()):
            lines.append(f"  {net}: {', '.join(sorted(codes))}")
    return lines
