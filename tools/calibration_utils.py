"""Per-antenna array self-calibration: estimation, validation and the calibration file.

Model
-----
After the SSMIF cable correction, channel n of a delay-Doppler cell k holding one
plane wave from direction u_k is

    x_kn = g_n * a_n(u_k) * s_k + noise,     a_n(u) = exp(+j 2 pi r_n . u / lambda)

g_n is what the station description does not capture: residual cable or
electronics phase, a mis-surveyed position, a swapped feed. ``self_calibrate``
estimates the phases of g by alternating two steps:

1. with the current g removed, find each cell's best plane-wave direction u_k;
2. with the directions fixed, g_n = phase of sum_k w_k x_kn conj(a_n(u_k) s_k).

Two things cannot be learned from the data and are pinned instead:

* a common phase on all channels (meaningless), and
* a linear phase ramp across the array, which is indistinguishable from moving
  every source by the same amount. It is projected out of g after every
  iteration, so pointing stays tied to the SSMIF geometry and self-calibration
  only removes the part of the error that is *not* a plane wave.

Only phases are estimated. Channel amplitudes are left alone (equalising a
weak channel only amplifies its noise); channels whose power or phase fit is far
from the rest are flagged instead.

Calibration file (HDF5)
-----------------------
Datasets ``stand``, ``pol``, ``digitizer``, ``gain`` (complex, |g| = 1),
``consistency``, ``power_db``, ``flagged``; root attribute ``metadata_json``
with the frequency, SSMIF, cells used, validation results and flag reasons.
Apply with ``ArrayProcessor(..., calibration=ArrayCalibration.load(path))``
or ``--calibration path`` on any array command.
"""

import json
import os
from datetime import datetime, timezone

import numpy as np

import lfm_utils as lfm


# ---------------------------------------------------------------------------
# Plane-wave fit
# ---------------------------------------------------------------------------

class PlaneWaveSearch:
    """Best single plane wave for many snapshots at once (coarse grid + local refinement).

    ``fit(X)`` with X (K, N) returns eta (K,), az (K,), el (K,), where
    eta = |a^H x|^2 / (N x^H x) is the fraction of the snapshot a plane wave explains.
    """

    def __init__(self, positions_m, frequency_hz, coarse_step=2.0, fine_step=0.25):
        self.positions = np.asarray(positions_m, float)
        self.fc = float(frequency_hz)
        self.coarse, self.fine = float(coarse_step), float(fine_step)
        az = np.arange(-180.0, 180.0, self.coarse)
        el = np.arange(0.0, 90.0 + 0.5 * self.coarse, self.coarse)
        AZ, EL = np.meshgrid(az, el)
        self.grid_az, self.grid_el = AZ.ravel(), EL.ravel()
        self.A = steering(self.positions, self.fc, self.grid_az, self.grid_el)       # (D, N)

    def fit(self, X):
        X = np.atleast_2d(np.asarray(X, np.complex128))
        n = X.shape[1]
        energy = np.real(np.sum(X * X.conj(), axis=1))
        best = np.argmax(np.abs(X @ self.A.conj().T) ** 2, axis=1)
        eta, az, el = np.full(len(X), np.nan), np.full(len(X), np.nan), np.full(len(X), np.nan)
        offsets = np.arange(-self.coarse, self.coarse + 1e-9, self.fine)
        for k, i in enumerate(best):
            if not energy[k] > 0:
                continue
            AZ, EL = np.meshgrid(self.grid_az[i] + offsets, np.clip(self.grid_el[i] + offsets, 0.0, 90.0))
            A = steering(self.positions, self.fc, AZ.ravel(), EL.ravel())
            p = np.abs(A.conj() @ X[k]) ** 2
            j = int(np.argmax(p))
            eta[k] = p[j] / (n * energy[k])
            az[k], el[k] = (AZ.ravel()[j] + 180.0) % 360.0 - 180.0, EL.ravel()[j]
        return eta, az, el


def steering(positions_m, frequency_hz, az_deg, el_deg):
    """Rows are plane-wave responses exp(+j 2 pi r.u / lambda) (same convention as rdc_utils)."""
    az, el = np.deg2rad(np.asarray(az_deg, float)), np.deg2rad(np.asarray(el_deg, float))
    u = np.stack([np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)], axis=-1).reshape(-1, 3)
    return np.exp(2j * np.pi * frequency_hz / lfm.C_M_PER_S * (u @ np.asarray(positions_m, float).T))


def unit_phase(x):
    x = np.asarray(x, np.complex128)
    return x / np.maximum(np.abs(x), 1e-30)


# ---------------------------------------------------------------------------
# Cell selection and per-stand timing
# ---------------------------------------------------------------------------

def normalized_power(cube):
    """Per-cell power summed over channels, each channel scaled by its median (noise) power.

    cube: (delay, doppler, channels). Returns (power (delay, doppler), channel noise power (channels,)).
    """
    p = np.abs(np.asarray(cube)) ** 2
    p = np.where(np.isfinite(p), p, np.nan)
    sigma2 = np.nanmedian(p.reshape(-1, p.shape[-1]), axis=0)
    power = np.nansum(p / np.maximum(sigma2, 1e-30), axis=-1)
    power[~np.all(np.isfinite(cube), axis=-1)] = np.nan
    return power, sigma2


def echo_cells(power, n, min_snr_db=15.0, min_contrast_db=6.0, delay_guard=3, doppler_guard=5):
    """Strongest cells that are localised in delay, i.e. echoes rather than delay-spread interference.

    A narrowband (unchirped) interferer is spread over every delay bin of one
    Doppler bin after dechirping. Cells are therefore ranked by their power over
    the median of their own Doppler column ("contrast"), and must also be
    ``min_snr_db`` above the median cell. Returns [(row, col, snr_db, contrast_db)].
    """
    floor = np.nanmedian(power)
    column = np.nanmedian(power, axis=0)
    contrast = power / np.maximum(column, 1e-30)[None, :]
    out = []
    work = np.where(np.isfinite(contrast), contrast, -np.inf)
    for _ in range(int(n)):
        flat = int(np.argmax(work))
        if not np.isfinite(work.flat[flat]):
            break
        r, c = np.unravel_index(flat, work.shape)
        work[max(0, r - delay_guard):r + delay_guard + 1, max(0, c - doppler_guard):c + doppler_guard + 1] = -np.inf
        snr, con = 10 * np.log10(power[r, c] / floor), 10 * np.log10(contrast[r, c])
        if snr >= min_snr_db and con >= min_contrast_db:
            out.append((int(r), int(c), float(snr), float(con)))
    return out


# ---------------------------------------------------------------------------
# Self-calibration
# ---------------------------------------------------------------------------

def remove_ramp(g, positions_m, frequency_hz, use=None, span=0.05, step=0.0025):
    """Remove the common phase and the best-fitting linear phase ramp from g.

    The ramp exp(j 2 pi r.du / lambda) is the horizontal direction-cosine offset du
    maximising |sum_n g_n exp(-j 2 pi r_n.du / lambda)| over the ``use`` channels (a
    beamformer applied to g itself), found on a grid and then refined by a least-
    squares fit of the remaining (small) phases. Returns (g without ramp, du_east,
    du_north). |du| <= span (0.05 is about 3 deg near zenith): at a converged
    plane-wave fit the residual has no first-order ramp, so this only stops drift.
    """
    g = np.asarray(g, complex)
    use = np.ones(g.size, bool) if use is None else np.asarray(use, bool)
    pos = np.asarray(positions_m, float)[:, :2]
    k = 2 * np.pi * frequency_hz / lfm.C_M_PER_S
    grid = np.arange(-span, span + 1e-12, step)
    DX, DY = np.meshgrid(grid, grid)
    d = np.column_stack([DX.ravel(), DY.ravel()])
    du = d[int(np.argmax(np.abs(np.exp(-1j * k * (d @ pos[use].T)) @ g[use])))]
    design = np.column_stack([np.ones(use.sum()), k * pos[use]])
    for _ in range(3):
        h = g[use] * np.exp(-1j * k * (pos[use] @ du))
        phase = np.angle(h * np.conj(np.sum(h)))
        du = du + np.linalg.lstsq(design, phase, rcond=None)[0][1:]
    g = g * np.exp(-1j * k * (pos @ du))
    s = np.sum(g[use])
    return g * np.conj(s) / max(abs(s), 1e-30), float(du[0]), float(du[1])


def self_calibrate(snapshots, positions_m, frequency_hz, use=None, iterations=30, tol_deg=0.05):
    """Phase-only per-channel gains from plane-wave-like snapshots.

    snapshots: (K, N) complex, cable-corrected; use: channels allowed to inform
    the fit (the others still get a gain, estimated against the fitted waves).
    Returns dict with gain (N,), consistency (N,), directions (K, 2), eta (K,),
    iterations, ramp (du_east, du_north) removed in total.
    """
    X = unit_phase(snapshots)                      # phases only: robust to loud or weak channels
    K, N = X.shape
    pos = np.asarray(positions_m, float)
    use = np.ones(N, bool) if use is None else np.asarray(use, bool)
    search = PlaneWaveSearch(pos[use], frequency_hz)
    g = np.ones(N, complex)
    ramp = np.zeros(2)
    it = 0

    def fit(g):
        eta, az, el = search.fit((X * g.conj())[:, use])
        Z = X * steering(pos, frequency_hz, az, el).conj()                # ~ g_n s_k
        s = unit_phase(np.sum(Z[:, use] * g[use].conj(), axis=1))        # per-cell common phase
        w = np.nan_to_num(eta)                                            # plane-wave-like cells count more
        m = np.sum(w[:, None] * Z * s.conj()[:, None], axis=0) / max(np.sum(w), 1e-30)
        return m, eta, az, el

    for it in range(1, int(iterations) + 1):
        m = fit(g)[0]
        g_new, dx, dy = remove_ramp(unit_phase(m), pos, frequency_hz, use)
        change = np.degrees(np.max(np.abs(np.angle(g_new[use] * g[use].conj()))))
        g, ramp = g_new, ramp + (dx, dy)
        if change < tol_deg:
            break
    m, eta, az, el = fit(g)
    return dict(gain=g, consistency=np.abs(m), directions=np.column_stack([az, el]), eta=eta,
                iterations=it, ramp=ramp.tolist())


def plane_wave_eta(snapshots, search, gain=None, use=None):
    """Phase-only eta of each snapshot after removing ``gain`` (None = SSMIF only).

    ``search`` must be built for the ``use`` channels' positions."""
    X = unit_phase(snapshots)
    if gain is not None:
        X = X * np.conj(gain)
    if use is not None:
        X = X[:, np.asarray(use, bool)]
    return search.fit(X)[0]


def class_dependent_stands(X, positions_m, frequency_hz, use, gain, iterations=30, max_diff_deg=60.0,
                           min_consistency=0.3):
    """Stands whose phase differs between two classes of echo cells.

    Class B are the cells the first-pass gains fit poorly: made worse by them, or
    left below 80 % of the median fit. If there are at least 3 of them (and 3
    others), gains are fitted separately to each class; stands whose two gains are
    both well determined (consistency >= ``min_consistency``) yet differ by more
    than ``max_diff_deg`` respond differently to the two classes. On 2023-12-09 the
    classes look like the two magnetoionic modes, so these stands probably have a
    different polarization response, and no single phase can calibrate them for
    both. Returns (mask of such stands, class-B cell mask).
    """
    pos = np.asarray(positions_m, float)
    search = PlaneWaveSearch(pos[use], frequency_hz)
    before = plane_wave_eta(X, search, None, use)
    after = plane_wave_eta(X, search, gain, use)
    poor = (after < before - 0.05) | (after < 0.8 * np.nanmedian(after))
    odd = np.zeros(X.shape[1], bool)
    if poor.sum() < 3 or (~poor).sum() < 3:
        return odd, poor
    ra = self_calibrate(X[~poor], pos, frequency_hz, use, iterations)
    rb = self_calibrate(X[poor], pos, frequency_hz, use, iterations)
    diff = ra["gain"] * np.conj(rb["gain"])
    diff = diff * np.conj(np.sum(diff[use])) / max(abs(np.sum(diff[use])), 1e-30)
    sure = (ra["consistency"] >= min_consistency) & (rb["consistency"] >= min_consistency)
    odd = use & sure & (np.degrees(np.abs(np.angle(diff))) > max_diff_deg)
    return odd, poor


def calibrate(snapshots, positions_m, frequency_hz, channel_power_db, groups=None, max_power_db=10.0,
              min_relative_consistency=0.5, iterations=30):
    """Full procedure: power flags, fit, class and consistency flags, refit, two-fold validation.

    channel_power_db: per-channel noise power re the median (for power flags).
    groups: (K,) fold label per snapshot (e.g. frame parity); default alternates cells.
    """
    X = np.asarray(snapshots, np.complex128)
    K, N = X.shape
    if K < 3:
        raise ValueError(f"Self-calibration needs at least 3 strong cells; found {K}")
    reasons = {}
    power_db = np.asarray(channel_power_db, float)
    for n in np.flatnonzero(np.abs(power_db) > max_power_db):
        reasons[int(n)] = f"power {power_db[n]:+.1f} dB re median"
    use = np.ones(N, bool)
    use[list(reasons)] = False
    power_only = use.copy()

    first = self_calibrate(X, positions_m, frequency_hz, use, iterations)
    odd, class_b = class_dependent_stands(X, positions_m, frequency_hz, use, first["gain"], iterations)
    for n in np.flatnonzero(odd):
        reasons[int(n)] = f"phase differs between echo classes ({int(class_b.sum())} of {K} cells in class B)"
    use[odd] = False
    if odd.any():
        first = self_calibrate(X, positions_m, frequency_hz, use, iterations)
    c = first["consistency"]
    threshold = min_relative_consistency * np.median(c[use])
    for n in np.flatnonzero(use & (c < threshold)):
        reasons[int(n)] = f"phase consistency {c[n]:.2f} < {threshold:.2f}"
    use[list(reasons)] = False
    final = self_calibrate(X, positions_m, frequency_hz, use, iterations)
    search = PlaneWaveSearch(np.asarray(positions_m, float)[use], frequency_hz)
    search_all = PlaneWaveSearch(np.asarray(positions_m, float)[power_only], frequency_hz)

    groups = np.arange(K) % 2 if groups is None or len(np.unique(groups)) < 2 else np.asarray(groups)
    labels = np.unique(groups)
    held_unflagged, held_before, held_after = [], [], []
    for lab in labels[:2] if len(labels) == 2 else labels:
        fit_on, test_on = groups != lab, groups == lab
        if fit_on.sum() < 3 or not test_on.any():
            continue
        r = self_calibrate(X[fit_on], positions_m, frequency_hz, use, iterations)
        held_unflagged.append(plane_wave_eta(X[test_on], search_all, None, power_only))
        held_before.append(plane_wave_eta(X[test_on], search, None, use))
        held_after.append(plane_wave_eta(X[test_on], search, r["gain"], use))
    cat = (lambda h: np.concatenate(h)) if held_before else None
    validation = dict(
        held_out_cells=int(sum(len(h) for h in held_before)),
        eta_power_flags_only_median=float(np.nanmedian(cat(held_unflagged))) if held_before else None,
        eta_before_median=float(np.nanmedian(cat(held_before))) if held_before else None,
        eta_after_median=float(np.nanmedian(cat(held_after))) if held_before else None,
        cells_made_worse=int(np.sum(cat(held_after) < cat(held_before) - 0.05)) if held_before else None,
        in_sample_eta_before_median=float(np.nanmedian(plane_wave_eta(X, search, None, use))),
        in_sample_eta_after_median=float(np.nanmedian(final["eta"])),
        noise_eta=1.0 / max(int(use.sum()), 1))
    flagged = np.zeros(N, bool)
    flagged[list(reasons)] = True
    final.update(flagged=flagged, reasons=reasons, validation=validation, used=use, class_b_cells=class_b,
                 first_pass_consistency=c, consistency_threshold=float(threshold))
    return final


# ---------------------------------------------------------------------------
# Calibration file
# ---------------------------------------------------------------------------

class ArrayCalibration:
    """Per-(stand, pol) complex gains and flags, as written by ``calibrate_array.py``."""

    def __init__(self, stands, pols, gain, flagged, digitizers=None, consistency=None, power_db=None,
                 meta=None, path=None):
        self.stands = np.asarray(stands, int)
        self.pols = np.asarray(pols, int)
        self.gain = np.asarray(gain, complex)
        self.flagged = np.asarray(flagged, bool)
        self.digitizers = None if digitizers is None else np.asarray(digitizers, int)
        self.consistency = None if consistency is None else np.asarray(consistency, float)
        self.power_db = None if power_db is None else np.asarray(power_db, float)
        self.meta = dict(meta or {})
        self.path = path

    @property
    def frequency_hz(self):
        return float(self.meta["frequency_hz"])

    def save(self, path):
        import rdc_io
        with rdc_io.atomic_output(path) as tmp:
            with rdc_io._h5py().File(tmp, "w") as h5:
                h5.create_dataset("stand", data=self.stands)
                h5.create_dataset("pol", data=self.pols)
                h5.create_dataset("gain", data=self.gain.astype(np.complex128))
                h5.create_dataset("flagged", data=self.flagged.astype(np.uint8))
                if self.digitizers is not None:
                    h5.create_dataset("digitizer", data=self.digitizers)
                if self.consistency is not None:
                    h5.create_dataset("consistency", data=self.consistency)
                if self.power_db is not None:
                    h5.create_dataset("power_db", data=self.power_db)

                h5.attrs["metadata_json"] = json.dumps(self.meta, default=_json_default)
        self.path = path
        return path

    @classmethod
    def load(cls, path):
        import rdc_io
        with rdc_io._h5py().File(path, "r") as h5:
            meta = json.loads(h5.attrs["metadata_json"])
            if meta.get("kind") != "array_calibration":
                raise ValueError(f"{path} is not an array calibration file")
            get = lambda k: h5[k][()] if k in h5 else None  # noqa: E731
            return cls(get("stand"), get("pol"), get("gain"), get("flagged").astype(bool), get("digitizer"),
                       get("consistency"), get("power_db"), meta, str(path))

    def factors(self, stands, pols, frequency_hz):
        """Multipliers conj(g) for the given channels and a keep mask (False = flagged or uncalibrated)."""
        if abs(float(frequency_hz) - self.frequency_hz) > 1e3:
            raise ValueError(f"Calibration {os.path.basename(str(self.path))} was made at "
                             f"{self.frequency_hz / 1e6:.6f} MHz, not {float(frequency_hz) / 1e6:.6f} MHz")
        lookup = {(int(s), int(p)): i for i, (s, p) in enumerate(zip(self.stands, self.pols))}
        idx = [lookup.get((int(s), int(p))) for s, p in zip(stands, pols)]
        keep = np.array([i is not None and not self.flagged[i] for i in idx], bool)
        f = np.array([np.conj(self.gain[i]) / max(abs(self.gain[i]), 1e-30) if i is not None else 1.0 for i in idx],
                     complex)
        return f, keep

    def describe(self):
        name = os.path.basename(str(self.path)) if self.path else "in memory"
        return f"self-cal {name} ({int(np.sum(~self.flagged))} stands, {int(np.sum(self.flagged))} flagged)"

    def check_station(self, ssmif_path):
        """Warn when the file was made with a different station description."""
        made = self.meta.get("ssmif")
        if (made or None) != (ssmif_path or None) and os.path.basename(str(made)) != os.path.basename(str(ssmif_path)):
            print(f"WARNING: calibration was made with station description {made or 'LSL bundled'}, "
                  f"now using {ssmif_path or 'LSL bundled'}", flush=True)


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def now_utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
