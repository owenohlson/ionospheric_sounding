"""Map drawing without a mapping library: outlines shipped with the code, a projection, great circles.

Outlines
    ``resources/map_outlines.npz`` (next to this file) holds coastlines, lake shores
    (the Great Lakes: without them Michigan has almost no outline), country borders
    and state/province lines for North America, cut from Natural Earth 1:50m
    (public domain, naturalearthdata.com) and rounded to 0.01 deg. Each layer is
    an (N, 2) int16 array of lon/lat in hundredths of a degree, with rows of
    -32768 between separate lines. Rebuild it (or widen the region) with

        python tools/map_utils.py --build ne_50m_coastline.zip ne_50m_lakes.zip \\
            ne_50m_admin_0_boundary_lines_land.zip ne_50m_admin_1_states_provinces_lines.zip

    which reads Natural Earth shapefiles (.zip or .shp) or GeoJSON.
Projection
    Lambert conformal conic on a sphere, standard parallels chosen from the
    map's latitude range, so great circles over a continent look nearly straight
    and shapes are kept.
Great circles
    Points a fraction of the way along the great circle between two sites:
    the midpoint of a one-hop path, and the reflection points and ground
    bounces of an n-hop path (mirror model: n equal hops).
"""

import io
import json
import math
import os
import struct
import zipfile

import numpy as np

OUTLINES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources", "map_outlines.npz")
LAYERS = ("coast", "lakes", "borders", "states")
REGION = (-172.0, -48.0, 8.0, 76.0)        # lon_min, lon_max, lat_min, lat_max kept in the shipped file
BREAK = -32768


# ---------------------------------------------------------------------------
# Outlines: build from Natural Earth, load
# ---------------------------------------------------------------------------

def _layer_of(name):
    n = os.path.basename(str(name)).lower()
    if "coast" in n:
        return "coast"
    if "lake" in n:
        return "lakes"
    if "admin_0" in n or "boundary_lines_land" in n or "countr" in n:
        return "borders"
    if "admin_1" in n or "states" in n or "province" in n:
        return "states"
    raise ValueError(f"Cannot tell which layer {name} is (coast, lakes, borders or states) from its name")


def read_shp(data):
    """Lines (lists of (lon, lat) arrays) from the bytes of a .shp file (polyline/polygon types)."""
    lines = []
    pos = 100
    while pos + 8 <= len(data):
        length = struct.unpack(">i", data[pos + 4:pos + 8])[0] * 2
        rec = data[pos + 8:pos + 8 + length]
        pos += 8 + length
        if len(rec) < 4:
            continue
        kind = struct.unpack("<i", rec[:4])[0]
        if kind not in (3, 5, 13, 15, 23, 25):
            continue
        n_parts, n_points = struct.unpack("<2i", rec[36:44])
        parts = list(struct.unpack(f"<{n_parts}i", rec[44:44 + 4 * n_parts])) + [n_points]
        pts = np.frombuffer(rec, "<f8", 2 * n_points, 44 + 4 * n_parts).reshape(-1, 2)
        lines += [pts[a:b] for a, b in zip(parts[:-1], parts[1:]) if b - a > 1]
    return lines


def read_geojson(text):
    lines = []

    def geom(g):
        t, c = g.get("type"), g.get("coordinates")
        if t == "LineString":
            lines.append(np.asarray(c, float)[:, :2])
        elif t in ("MultiLineString", "Polygon"):
            lines.extend(np.asarray(x, float)[:, :2] for x in c)
        elif t == "MultiPolygon":
            lines.extend(np.asarray(r, float)[:, :2] for poly in c for r in poly)
        elif t == "GeometryCollection":
            for x in g.get("geometries", []):
                geom(x)

    doc = json.loads(text)
    for f in doc.get("features", [doc]):
        if f.get("geometry"):
            geom(f["geometry"])
    return lines


def read_outline_file(path):
    """Lines from a Natural Earth .zip (with a .shp), .shp, or .geojson/.json file."""
    if path.lower().endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if n.lower().endswith(".shp")]
            if names:
                return read_shp(z.read(names[0]))
            names = [n for n in z.namelist() if n.lower().endswith((".geojson", ".json"))]
            if names:
                return read_geojson(z.read(names[0]).decode("utf-8"))
            raise ValueError(f"{path} holds no .shp or .geojson file")
    if path.lower().endswith(".shp"):
        with open(path, "rb") as fh:
            return read_shp(fh.read())
    with io.open(path, encoding="utf-8") as fh:
        return read_geojson(fh.read())


def _clip_segment(p, q, region):
    """Liang-Barsky: the part of segment p-q inside the box, or None."""
    lo0, lo1, la0, la1 = region
    t0, t1 = 0.0, 1.0
    d = q - p
    for den, num in ((-d[0], p[0] - lo0), (d[0], lo1 - p[0]), (-d[1], p[1] - la0), (d[1], la1 - p[1])):
        if den == 0:
            if num < 0:
                return None
            continue
        t = num / den
        if den < 0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
        if t0 > t1:
            return None
    return p + t0 * d, p + t1 * d, t0 > 0, t1 < 1


def clip_lines(lines, region=REGION):
    """Pieces of each line inside the region; lines leaving it are cut at the edge."""
    lo0, lo1, la0, la1 = region
    out = []
    for ln in lines:
        ln = np.asarray(ln, float)
        inside = (ln[:, 0] >= lo0) & (ln[:, 0] <= lo1) & (ln[:, 1] >= la0) & (ln[:, 1] <= la1)
        if inside.all():
            out.append(ln)
            continue
        if not inside.any() and (ln[:, 0].max() < lo0 or ln[:, 0].min() > lo1 or ln[:, 1].max() < la0
                                 or ln[:, 1].min() > la1):
            continue
        piece = []
        for p, q in zip(ln[:-1], ln[1:]):
            seg = _clip_segment(p, q, region)
            if seg is None:
                continue
            a, b, cut_start, cut_end = seg
            if cut_start or not piece:
                if len(piece) > 1:
                    out.append(np.array(piece))
                piece = [a]
            piece.append(b)
            if cut_end:
                out.append(np.array(piece))
                piece = []
        if len(piece) > 1:
            out.append(np.array(piece))
    return out


def encode(lines):
    """Lines -> one (N, 2) int16 array in hundredths of a degree, BREAK rows between lines."""
    parts = []
    for ln in lines:
        q = np.round(np.asarray(ln) * 100).astype(np.int32)
        keep = np.r_[True, np.any(np.diff(q, axis=0) != 0, axis=1)]      # drop repeats after rounding
        q = q[keep]
        if len(q) > 1:
            parts += [q, np.full((1, 2), BREAK)]
    return np.concatenate(parts).astype(np.int16) if parts else np.zeros((0, 2), np.int16)


LAKE_MIN_DEG = 1.0        # lake outlines whose extent is smaller than this (reservoirs, islands) are left out


def _big(line, min_deg):
    a = np.asarray(line)
    return np.hypot(*(a.max(axis=0) - a.min(axis=0))) >= min_deg


def build_outlines(paths, output=OUTLINES, region=REGION):
    layers = {name: [] for name in LAYERS}
    for path in paths:
        layer = _layer_of(path)
        lines = read_outline_file(path)
        if layer == "lakes":
            lines = [ln for ln in lines if _big(ln, LAKE_MIN_DEG)]
        layers[layer] += clip_lines(lines, region)
    arrays = {name: encode(lines) for name, lines in layers.items()}
    np.savez_compressed(output, region=np.asarray(region, float),
                        source=np.asarray("Natural Earth 1:50m (public domain), rounded to 0.01 deg"), **arrays)
    return {name: int(np.sum(a[:, 0] != BREAK)) for name, a in arrays.items()}


def load_outlines(path=OUTLINES):
    """{layer: (N, 2) float lon/lat with NaN rows between lines}, or {} when the file is missing."""
    if not os.path.exists(path):
        return {}
    with np.load(path) as z:
        out = {}
        for name in LAYERS:
            if name in z.files:
                a = z[name].astype(float)
                a[a[:, 0] == BREAK] = np.nan
                out[name] = a / 100.0
        return out


# ---------------------------------------------------------------------------
# Great circles
# ---------------------------------------------------------------------------

def _xyz(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


def intermediate(lat1, lon1, lat2, lon2, fractions):
    """(lat, lon) arrays of points at ``fractions`` (0 = site 1, 1 = site 2) along the great circle."""
    a, b = _xyz(lat1, lon1), _xyz(lat2, lon2)
    d = math.acos(float(np.clip(a @ b, -1.0, 1.0)))
    f = np.atleast_1d(np.asarray(fractions, float))
    if d < 1e-12:
        p = np.repeat(a[:, None], f.size, axis=1)
    else:
        p = (np.sin((1 - f) * d) * a[:, None] + np.sin(f * d) * b[:, None]) / math.sin(d)
    return np.degrees(np.arcsin(np.clip(p[2], -1, 1))), np.degrees(np.arctan2(p[1], p[0]))


def great_circle(lat1, lon1, lat2, lon2, n=200):
    return intermediate(lat1, lon1, lat2, lon2, np.linspace(0, 1, n))


def hop_points(lat1, lon1, lat2, lon2, hops):
    """Reflection points (sky) and ground bounces of an n-hop path with equal hops."""
    hops = int(hops)
    sky = intermediate(lat1, lon1, lat2, lon2, (2 * np.arange(1, hops + 1) - 1) / (2 * hops))
    ground = intermediate(lat1, lon1, lat2, lon2, np.arange(1, hops) / hops) if hops > 1 else (np.array([]),) * 2
    return sky, ground


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------

class LambertConformal:
    """Spherical Lambert conformal conic; x, y in Earth radii."""

    def __init__(self, lon0, lat0, lat1, lat2):
        self.lon0 = float(lon0)
        p0, p1, p2 = (math.radians(v) for v in (lat0, lat1, lat2))
        t = lambda p: math.tan(math.pi / 4 + p / 2)          # noqa: E731
        self.n = math.sin(p1) if abs(p1 - p2) < 1e-6 else \
            math.log(math.cos(p1) / math.cos(p2)) / math.log(t(p2) / t(p1))
        self.F = math.cos(p1) * t(p1) ** self.n / self.n
        self.rho0 = self.F / t(p0) ** self.n

    @classmethod
    def for_extent(cls, lon_min, lon_max, lat_min, lat_max):
        span = lat_max - lat_min
        return cls((lon_min + lon_max) / 2, (lat_min + lat_max) / 2, lat_min + span / 6, lat_max - span / 6)

    def __call__(self, lon, lat):
        lon, lat = np.asarray(lon, float), np.asarray(lat, float)
        rho = self.F / np.tan(np.pi / 4 + np.radians(lat) / 2) ** self.n
        theta = self.n * np.radians(((lon - self.lon0 + 180) % 360) - 180)
        return rho * np.sin(theta), self.rho0 - rho * np.cos(theta)


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="Build the map outline file from Natural Earth downloads.")
    p.add_argument("--build", nargs="+", required=True, metavar="FILE",
                   help="Natural Earth coastline / admin-0 boundary lines / admin-1 lines (.zip, .shp or .geojson)")
    p.add_argument("--output", default=OUTLINES)
    p.add_argument("--region", default=",".join(f"{v:g}" for v in REGION), help="lon_min,lon_max,lat_min,lat_max")
    args = p.parse_args(argv)
    region = tuple(float(v) for v in args.region.split(","))
    counts = build_outlines(args.build, args.output, region)
    size = os.path.getsize(args.output) / 1024
    print(f"Wrote {args.output} ({size:.0f} KiB): " + ", ".join(f"{k} {v} points" for k, v in counts.items()))


if __name__ == "__main__":
    main()
