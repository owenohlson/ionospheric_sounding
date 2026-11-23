"""Ionosonde data (GIRO / DIDBase) for hop-count estimates: profiles, virtual heights, oblique paths.

Station list
    ``resources/giro_stations.txt`` (next to this file, or --giro-stations): the DIDBase Fast
    Station List pasted as text, one station per line: index, URSI code, name,
    latitude, longitude [deg E] (tab or 2+ spaces between columns).
Measurements
    One file per station and day from GIRO's FastChar service (DIDBase "getbest"),
    kept in ``resources/giro/`` (the default), e.g. ``resources/giro/BC840_2023-12-09.txt``:

        https://lgdc.uml.edu/fastchar/getbest?ursiCode=BC840&charName=foF2,foF1,foE,foEs,
        MUF%28D%29,M%28D%29,hmF2,hmF1,hmE,B0,B1&DMUF=3000&fromDate=2023%2F12%2F09+15%3A30%3A00
        &toDate=2023%2F12%2F09+19%3A00%3A00

    Values are autoscaled. Rows with a confidence score below 50, and rows whose
    hmF2 or M(D) jump away from their neighbours (a mis-scaled trace), are dropped;
    each characteristic is interpolated linearly in time (no farther than 20 min).
Profile
    Electron plasma frequency vs height from the characteristics, no magnetic
    field (ordinary ray, no collisions): a Chapman E layer (foE, hmE, 10 km scale
    height), an F1 parabola when foF1/hmF1 are scaled, and the IRI bottomside of
    F2, fN^2 = foF2^2 exp(-x^B1) / cosh(x), x = (hmF2 - h) / B0, up to hmF2.
    Virtual height h'(fv) = integral of the group index 1 / sqrt(1 - fN^2/fv^2).
Oblique hop
    Spherical Earth; a hop of ground range d reflects (mirror model) at the virtual
    height h' at which the vertical ionogram has the equivalent vertical frequency
    fv = f cos(phi), phi the angle of incidence at h' (secant law with Martyn's
    theorem, no curvature correction factor). Each hop count n has 0, 1 or 2
    solutions (low and high ray); none when f is above the hop's MUF. The elevation
    of each solution is compared with the measured one.

GIRO data are CC-BY-NC-SA 4.0 and need acknowledgement of each station's data
provider: every result names the station it used.
"""

import argparse
import glob
import math
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

R_EARTH_KM = 6371.0
RESOURCES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources")
STATION_LIST = os.path.join(RESOURCES, "giro_stations.txt")
GIRO_DATA = os.path.join(RESOURCES, "giro")            # FastChar files used by default
PARAMS = ("foF2", "foF1", "foE", "foEs", "MUF(D)", "M(D)", "hmF2", "hmF1", "hmE", "B0", "B1")


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------

@dataclass
class Station:
    code: str
    name: str
    lat_deg: float
    lon_deg: float          # -180..180


def load_station_list(path=None):
    """GIRO stations: [Station]; longitudes converted from deg E to -180..180."""
    out = []
    with open(path or STATION_LIST, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = [p.strip() for p in re.split(r"\t|\s{2,}", line.strip()) if p.strip()]
            if len(parts) >= 4 and re.fullmatch(r"\d+", parts[0]):
                parts = parts[1:]
            if len(parts) < 4:
                continue
            try:
                lat, lon = float(parts[-2]), float(parts[-1])
            except ValueError:
                continue
            out.append(Station(parts[0].upper(), " ".join(parts[1:-2]), lat, ((lon + 180) % 360) - 180))
    return out


@dataclass
class Sounding:
    """One station's characteristics over time (NaN where not scaled)."""
    code: str
    name: str
    lat_deg: float
    lon_deg: float
    times: np.ndarray                      # POSIX seconds
    values: dict                           # name -> array
    confidence: np.ndarray
    model: str = ""
    dropped: list = field(default_factory=list)     # times of rows rejected as mis-scaled

    def at(self, when, max_gap_s=1200.0):
        """Characteristics interpolated to ``when`` (datetime or POSIX s); NaN where too far from data."""
        t = when.timestamp() if isinstance(when, datetime) else float(when)
        out = {}
        for k, v in self.values.items():
            ok = np.isfinite(v)
            tt, vv = self.times[ok], v[ok]
            if not tt.size or t < tt[0] - max_gap_s or t > tt[-1] + max_gap_s:
                out[k] = float("nan")
                continue
            i = np.searchsorted(tt, t)
            if i == 0 or i == tt.size:
                j = 0 if i == 0 else tt.size - 1
                out[k] = float(vv[j]) if abs(tt[j] - t) <= max_gap_s else float("nan")
            elif tt[i] - tt[i - 1] > 2 * max_gap_s:
                j = i if tt[i] - t < t - tt[i - 1] else i - 1
                out[k] = float(vv[j]) if abs(tt[j] - t) <= max_gap_s else float("nan")
            else:
                out[k] = float(np.interp(t, tt, vv))
        return out


def read_fastchar(path, min_confidence=50):
    """A GIRO FastChar (getbest) text file -> Sounding, with mis-scaled rows removed."""
    code = name = model = ""
    lat = lon = float("nan")
    cols = None
    rows = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("#"):
                m = re.search(r"GEO \(\s*([-\d.]+)\s*([NS])\s+([-\d.]+)\s*E\s*\),\s*URSI-Code\s+(\S+),\s*(.*)", line)
                if m:
                    lat = float(m.group(1)) * (1 if m.group(2) == "N" else -1)
                    lon = ((float(m.group(3)) + 180) % 360) - 180
                    code, name = m.group(4).strip(), m.group(5).strip()
                m = re.search(r"Model:\s*\[(.*?)\]", line)
                if m:
                    model = m.group(1)
                if line.lstrip("# ").startswith("Time") and "CS" in line:
                    cols = line.lstrip("#").split()[2:]           # after Time, CS: name, QD, name, QD ...
                continue
            parts = line.split()
            if len(parts) < 3 or cols is None:
                continue
            t = datetime.strptime(parts[0].rstrip("Z")[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
            vals = {}
            for i, c in enumerate(cols):
                if c == "QD":
                    continue
                tok = parts[2 + i] if 2 + i < len(parts) else "---"
                try:
                    vals[c] = float(tok)
                except ValueError:
                    vals[c] = float("nan")
            rows.append((t.timestamp(), float(parts[1]), vals))
    if not rows:
        raise ValueError(f"No measurements in {path}")
    rows.sort(key=lambda r: r[0])
    times = np.array([r[0] for r in rows])
    cs = np.array([r[1] for r in rows])
    names = sorted({k for r in rows for k in r[2]})
    values = {k: np.array([r[2].get(k, np.nan) for r in rows]) for k in names}
    bad = (cs >= 0) & (cs < min_confidence) & (cs != 999)
    for key, limit in (("hmF2", 40.0), ("M(D)", 0.5)):          # jumps a mis-scaled trace makes
        v = values.get(key)
        if v is None:
            continue
        med = np.array([np.nanmedian(v[max(0, i - 3):i + 4]) for i in range(v.size)])
        bad |= np.isfinite(v) & (np.abs(v - med) > limit)
    for v in values.values():
        v[bad] = np.nan
    return Sounding(code, name, lat, lon, times, values, cs, model, list(times[bad]))


def load_soundings(directory):
    """Every FastChar file in ``directory`` ({URSI}_*.txt), merged per station: {code: Sounding}."""
    out = {}
    for path in sorted(glob.glob(os.path.join(directory, "*.txt"))):
        try:
            s = read_fastchar(path)
        except (ValueError, OSError):
            continue
        if s.code in out:                                   # several days: concatenate
            a = out[s.code]
            t = np.r_[a.times, s.times]
            order = np.argsort(t)
            keys = set(a.values) | set(s.values)
            vals = {k: np.r_[a.values.get(k, np.full(a.times.size, np.nan)),
                             s.values.get(k, np.full(s.times.size, np.nan))][order] for k in keys}
            out[s.code] = Sounding(a.code, a.name, a.lat_deg, a.lon_deg, t[order], vals,
                                   np.r_[a.confidence, s.confidence][order], a.model, a.dropped + s.dropped)
        else:
            out[s.code] = s
    return out


# ---------------------------------------------------------------------------
# Profile and virtual height
# ---------------------------------------------------------------------------

def plasma_frequency_sq(h_km, p):
    """fN^2 [MHz^2] at heights ``h_km`` from characteristics ``p`` (dict as Sounding.at returns)."""
    h = np.asarray(h_km, float)
    f2 = np.zeros_like(h)
    foE, hmE = p.get("foE", np.nan), p.get("hmE", np.nan)
    if np.isfinite(foE):
        hmE = hmE if np.isfinite(hmE) else 110.0
        z = (h - hmE) / 10.0
        f2 = np.maximum(f2, foE ** 2 * np.exp(1 - z - np.exp(-z)))      # Chapman alpha (in N)
    foF2, hmF2 = p.get("foF2", np.nan), p.get("hmF2", np.nan)
    if np.isfinite(foF2) and np.isfinite(hmF2):
        b0 = p.get("B0", np.nan)
        b1 = p.get("B1", np.nan)
        b0 = b0 if np.isfinite(b0) and b0 > 5 else 0.3 * hmF2 - 20.0
        b1 = b1 if np.isfinite(b1) and b1 > 0.5 else 2.0
        x = np.clip((hmF2 - h) / b0, 0.0, None)
        bottom = foF2 ** 2 * np.exp(-x ** b1) / np.cosh(x)
        f2 = np.maximum(f2, np.where(h <= hmF2, bottom, foF2 ** 2))
        foF1, hmF1 = p.get("foF1", np.nan), p.get("hmF1", np.nan)
        if np.isfinite(foF1) and np.isfinite(hmF1) and hmF1 < hmF2:
            ym = max(20.0, hmF1 - (hmE if np.isfinite(hmE) else 110.0))
            f1 = foF1 ** 2 * (1 - ((h - hmF1) / ym) ** 2)
            f2 = np.maximum(f2, np.where((h <= hmF1) & (f1 > 0), f1, 0.0))
    return f2


def virtual_height(fv_mhz, p, h0=60.0, h_top=None, dh=0.05):
    """Virtual height [km] of vertical reflection at ``fv_mhz``; NaN if it penetrates."""
    top = h_top or (p.get("hmF2", 400.0) if np.isfinite(p.get("hmF2", np.nan)) else 400.0) + 1.0
    h = np.arange(h0, top, dh)
    x = plasma_frequency_sq(h, p) / fv_mhz ** 2
    above = np.flatnonzero(x >= 1.0)
    if not above.size:
        return float("nan")
    k = above[0]
    if k == 0:
        return float(h0)
    xm = 0.5 * (x[:k - 1] + x[1:k]) if k > 1 else np.zeros(0)
    total = h0 + dh * float(np.sum(1.0 / np.sqrt(np.clip(1 - xm, 1e-12, None))))
    # last cell, X rising linearly to 1: integral of dh/sqrt(1-X) = 2 L / sqrt(1 - X_a)
    xa = x[k - 1]
    L = dh * (1 - xa) / max(x[k] - xa, 1e-12)
    return total + 2 * L / math.sqrt(max(1 - xa, 1e-12))


def hop_geometry(hop_range_km, h_km):
    """Elevation and incidence angle [deg] of a mirror reflection at ``h_km`` for one hop."""
    theta = hop_range_km / (2 * R_EARTH_KM)
    r = R_EARTH_KM + h_km
    elev = math.degrees(math.atan2(math.cos(theta) - R_EARTH_KM / r, math.sin(theta)))
    incid = math.degrees(math.asin(min(1.0, R_EARTH_KM * math.cos(math.radians(elev)) / r)))
    return elev, incid


def oblique_solutions(freq_mhz, hop_range_km, p, h_min=80.0, h_max=700.0, step=1.0, min_elevation=1.0):
    """[(virtual height km, elevation deg, equivalent vertical freq MHz, ray)] for one hop of an oblique path.

    The mirror height h solves h = h'(f cos(phi(h))). Per layer, the first solution is the low
    ray and the next the high (Pedersen) ray; none means f is above this hop's MUF. Solutions
    below ``min_elevation`` (the hop is too long for that height) are left out."""
    hs = np.arange(h_min, h_max + step, step)
    g = []
    for h in hs:
        _, inc = hop_geometry(hop_range_km, h)
        fv = freq_mhz * math.cos(math.radians(inc))
        g.append(virtual_height(fv, p, h_top=max(h_max, p.get("hmF2", 400.0) + 1)) - h)
    g = np.array(g)
    out, seen = [], {}
    for i in range(len(hs) - 1):
        if np.isfinite(g[i]) and np.isfinite(g[i + 1]) and (g[i] == 0 or g[i] * g[i + 1] < 0):
            h = hs[i] + step * g[i] / (g[i] - g[i + 1])
            el, inc = hop_geometry(hop_range_km, h)
            if el < min_elevation:
                continue
            layer = layer_of(h, p)
            ray = "low" if layer not in seen else "high"
            seen[layer] = True
            out.append((float(h), el, freq_mhz * math.cos(math.radians(inc)), ray))
    return out


def layer_of(h_km, p):
    hmF1 = p.get("hmF1", np.nan)
    if h_km < 150:
        return "E"
    if np.isfinite(hmF1) and h_km < hmF1 + 20:
        return "F1"
    return "F"


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def _gc_km(lat1, lon1, lat2, lon2):
    p1, p2, dl = math.radians(lat1), math.radians(lat2), math.radians(lon2 - lon1)
    c = math.sin(p1) * math.sin(p2) + math.cos(p1) * math.cos(p2) * math.cos(dl)
    return R_EARTH_KM * math.acos(max(-1.0, min(1.0, c)))


def nearest(soundings, lat, lon, when, max_km=1500.0):
    """(Sounding, distance km, characteristics) of the nearest station with foF2 and hmF2 at ``when``."""
    best = None
    for s in soundings.values():
        p = s.at(when)
        if not (np.isfinite(p.get("foF2", np.nan)) and np.isfinite(p.get("hmF2", np.nan))):
            continue
        d = _gc_km(lat, lon, s.lat_deg, s.lon_deg)
        if d <= max_km and (best is None or d < best[1]):
            best = (s, d, p)
    return best


@dataclass
class HopOption:
    hops: int
    ray: str                  # low / high (per layer)
    height_km: float          # virtual (mirror) height
    elevation_deg: float
    layer: str
    fv_mhz: float
    station: str              # ionosonde used (the one nearest to any of the hop's reflection points)
    station_km: float
    residual_deg: float = None


def predict_hops(rx_lat, rx_lon, tx_lat, tx_lon, freq_mhz, when, soundings, max_hops=3, max_km=1500.0):
    """Every (hop count, ray) the ionosonde profile allows on this path: [HopOption], or None if no
    ionosonde with data is within ``max_km`` of the path's reflection points."""
    import map_utils as mu
    dist = _gc_km(rx_lat, rx_lon, tx_lat, tx_lon)
    out, any_station = [], False
    for n in range(1, max_hops + 1):
        (slat, slon), _ = mu.hop_points(rx_lat, rx_lon, tx_lat, tx_lon, n)
        # the reflection point closest to an ionosonde with data, and that ionosonde
        nears = [x for x in (nearest(soundings, float(a), float(b), when, max_km) for a, b in zip(slat, slon)) if x]
        if not nears:
            continue
        any_station = True
        s, d, p = min(nears, key=lambda x: x[1])
        for h, el, fv, ray in oblique_solutions(freq_mhz, dist / n, p):
            out.append(HopOption(n, ray, h, el, layer_of(h, p), fv, s.code, d))
    return out if any_station else None


def choose(options, elevation_deg, tol_deg=5.0):
    """(best HopOption, others within tol) for a measured elevation; (None, []) if none is within tol."""
    for o in options:
        o.residual_deg = elevation_deg - o.elevation_deg
    fits = sorted((o for o in options if abs(o.residual_deg) <= tol_deg), key=lambda o: abs(o.residual_deg))
    return (fits[0], fits[1:]) if fits else (None, [])


def main(argv=None):
    p = argparse.ArgumentParser(description="Hop options of one path from ionosonde data (see module doc).")
    p.add_argument("--data", required=True, help="Folder of GIRO FastChar files")
    p.add_argument("--path", required=True, metavar="RXLAT,RXLON,TXLAT,TXLON")
    p.add_argument("--freq", type=float, required=True, help="MHz")
    p.add_argument("--time", required=True, help="UTC, ISO 8601")
    p.add_argument("--elevation", type=float, default=None, help="Measured elevation [deg]")
    args = p.parse_args(argv)
    rx_lat, rx_lon, tx_lat, tx_lon = (float(v) for v in args.path.split(","))
    when = datetime.fromisoformat(args.time.replace("Z", "+00:00"))
    when = when if when.tzinfo else when.replace(tzinfo=timezone.utc)
    opts = predict_hops(rx_lat, rx_lon, tx_lat, tx_lon, args.freq, when, load_soundings(args.data))
    if opts is None:
        print("No ionosonde with data near this path")
        return
    if args.elevation is not None:
        choose(opts, args.elevation)
    for o in opts:
        print(f"{o.hops}{o.layer} {o.ray:4s} ray: h' {o.height_km:6.1f} km, elevation {o.elevation_deg:5.1f} deg, "
              f"fv {o.fv_mhz:5.2f} MHz, ionosonde {o.station} {o.station_km:.0f} km"
              + (f", residual {o.residual_deg:+.1f} deg" if o.residual_deg is not None else ""))


if __name__ == "__main__":
    main()
