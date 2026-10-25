"""Wavefront tracking: how one echo's phase across the array changes over time.

For one echo (a delay-Doppler cell followed from frame to frame), channel n of
frame t is, after the SSMIF cable correction and optional self-calibration,

    x_n(t) = A(t) * exp(j [k u(t) . r_n + e_n]) + noise,      k = 2 pi f / c

u(t) is the arrival direction (direction cosines), e_n whatever per-stand phase
error is left. A travelling ionospheric disturbance is hundreds of km long, so
across the ~100 m array it does not make fringes: it tilts and rocks the plane
wavefront, i.e. u(t) wanders slowly. This module measures that wander:

1. Reference plane. Divide every channel by a fixed plane wave from the
   session's median direction u_ref, and by the phase of the beam toward it
   (removes the common phase: Doppler rotation, frame timing). What remains,
   psi_n(t) = k (u(t) - u_ref) . r_n + e_n, is small and changes slowly.
2. Unwrap in time, per stand. Frame-to-frame increments are fitted with a plane
   and each stand's increment is taken on the 2 pi branch closest to that
   plane, so even edge stands whose phase moves fast follow correctly. There is
   no unwrapping *across* stands: neighbours can differ by more than pi at
   13-25 MHz and low elevation, and path-dependent spatial unwrapping spreads one
   bad stand's error to its neighbours.
3. Subtract each stand's time mean. This removes e_n completely (and the mean
   geometry), so the *changes* do not depend on the calibration. The mean itself
   minus its best plane is the static residual: what calibration left over.
4. Fit a plane to the change at every frame: the tilt gives u(t), hence
   az(t), el(t); the rest is the dynamic residual (non-planar structure,
   multipath, noise).

Everything here works on (frames, stands) arrays; reading files and plotting
live in plot_wavefront.py and plotting_utils.py.
"""

import warnings

import numpy as np

import calibration_utils as cu
import lfm_utils as lfm


def wrap(phase):
    return (np.asarray(phase, float) + np.pi) % (2 * np.pi) - np.pi


def wavenumber(frequency_hz):
    return 2 * np.pi * float(frequency_hz) / lfm.C_M_PER_S


def direction_cosines(az_deg, el_deg):
    az, el = np.radians(az_deg), np.radians(el_deg)
    return np.cos(el) * np.sin(az), np.cos(el) * np.cos(az)


def az_el(u, v):
    """Direction cosines (east, north) -> (az [deg, -180..180), el [deg])."""
    u, v = np.asarray(u, float), np.asarray(v, float)
    az = (np.degrees(np.arctan2(u, v)) + 180.0) % 360.0 - 180.0
    return az, np.degrees(np.arccos(np.clip(np.hypot(u, v), 0.0, 1.0)))


def fit_plane(phase, xy, k, use=None, robust=True):
    """Least-squares phase = c + k (du x + dv y) over the ``use`` stands.

    ``robust`` refits once without stands more than 3 MADs (at least 0.5 rad)
    from the first fit. Returns (c, du, dv, residual (N,), used mask).
    """
    phase = np.asarray(phase, float)
    use = np.isfinite(phase) if use is None else (np.asarray(use, bool) & np.isfinite(phase))
    if use.sum() < 4:
        return np.nan, np.nan, np.nan, np.full(phase.shape, np.nan), use
    D = np.column_stack([np.ones(len(xy)), k * xy[:, 0], k * xy[:, 1]])
    for _ in range(2 if robust else 1):
        coef = np.linalg.lstsq(D[use], phase[use], rcond=None)[0]
        res = phase - D @ coef
        if not robust:
            break
        mad = np.median(np.abs(res[use] - np.median(res[use])))
        keep = use & (np.abs(res) <= max(3 * 1.4826 * mad, 0.5))
        if keep.sum() < 4 or keep.sum() == use.sum():
            break
        use = keep
    coef = np.linalg.lstsq(D[use], phase[use], rcond=None)[0]
    return float(coef[0]), float(coef[1]), float(coef[2]), phase - D @ coef, use


def fit_plane_wrapped(phase, xy, k, use=None, iterations=4):
    """Plane fit to phases known only modulo 2 pi, assuming the residual stays within +/-pi.

    Starts from the circular mean (so an offset near +/-pi does not split the
    stands across the wrap), then refits with every stand on the 2 pi branch
    nearest the current plane."""
    phase = np.asarray(phase, float)
    use = np.isfinite(phase) if use is None else (np.asarray(use, bool) & np.isfinite(phase))
    if use.sum() < 4:
        return np.nan, np.nan, np.nan, np.full(phase.shape, np.nan), use
    c0 = float(np.angle(np.mean(np.exp(1j * phase[use]))))
    centred = wrap(phase - c0)
    c, du, dv, res, used = fit_plane(centred, xy, k, use)
    for _ in range(iterations):
        model = c + k * (du * xy[:, 0] + dv * xy[:, 1])
        c, du, dv, res, used = fit_plane(model + wrap(centred - model), xy, k, use)
    return c + c0, du, dv, wrap(res), used


def reference_direction(snapshots, positions_m, frequency_hz, valid_frames=None):
    """Median arrival direction of per-frame plane-wave fits.

    Returns (az_ref, el_ref, per-frame az, el, eta) of the phase-only grid fit
    (calibration_utils.PlaneWaveSearch)."""
    X = cu.unit_phase(np.nan_to_num(snapshots))
    eta, az, el = cu.PlaneWaveSearch(positions_m, frequency_hz).fit(X)
    ok = np.isfinite(az) if valid_frames is None else (np.isfinite(az) & valid_frames)
    if not ok.any():
        raise ValueError("No frame holds a usable plane wave")
    u, v = direction_cosines(az[ok], el[ok])
    a, e = az_el(np.median(u), np.median(v))
    return float(a), float(e), az, el, eta


def track(snapshots, positions_m, frequency_hz, ref_az, ref_el, valid=None, min_stands=8, min_consistency=0.3):
    """Phase change of each stand over time relative to a fixed reference plane.

    snapshots: (T, N) complex, one echo per frame. valid: (T, N) bool (default:
    finite and non-zero). Frames with fewer than ``min_stands`` valid stands are
    skipped. Returns a dict (all angles in radians unless named *_deg):

    ``cum``      (T, N) time-unwrapped phase relative to the reference plane
    ``static``   (N,)   time mean minus its best (wrapped) plane: per-stand error left
    ``change``   (T, N) cum minus each stand's time mean (independent of calibration)
    ``residual`` (T, N) change minus that frame's fitted plane
    ``az_deg``, ``el_deg`` (T,) arrival direction from the phase planes
    ``du``, ``dv`` (T,) direction-cosine change relative to the session mean
    ``coherence`` (T,)  |mean exp(j(static + residual))|: 1 = perfect plane wave
    ``rms_residual_deg`` (T,), ``stands_used`` (T,), ``slips`` (count of 2 pi slips repaired)
    ``noisy`` (N,) stands dropped because their residual phase does not stay put
    (|time mean of exp(j residual)| below ``min_consistency``)
    """
    with warnings.catch_warnings():                  # all-NaN stands/frames are expected (dropped or missing)
        warnings.simplefilter("ignore", RuntimeWarning)
        return _track(snapshots, positions_m, frequency_hz, ref_az, ref_el, valid, min_stands, min_consistency)


def _track(snapshots, positions_m, frequency_hz, ref_az, ref_el, valid, min_stands, min_consistency):
    X = np.asarray(snapshots, np.complex128)
    T, N = X.shape
    pos = np.asarray(positions_m, float)
    xy = pos[:, :2] - pos[:, :2].mean(axis=0)
    k = wavenumber(frequency_hz)
    valid = (np.isfinite(X) & (np.abs(np.nan_to_num(X)) > 0)) if valid is None else np.asarray(valid, bool)
    valid = valid & np.isfinite(X)
    frames = valid.sum(axis=1) >= min_stands
    valid &= frames[:, None]

    a_ref = cu.steering(pos, frequency_hz, [ref_az], [ref_el])[0]
    y = np.where(valid, cu.unit_phase(np.nan_to_num(X)) * a_ref.conj(), 0)
    beam = y.sum(axis=1)
    y *= cu.unit_phase(beam).conj()[:, None]
    psi = np.where(valid, np.angle(y), np.nan)

    cum = _unwrap_in_time(psi, valid, xy, k)
    out = _planes(cum, valid, xy, k)
    consistency = np.abs(np.nanmean(np.where(valid, np.exp(1j * np.nan_to_num(out["residual"])), np.nan), axis=0))
    noisy = np.isfinite(consistency) & (consistency < min_consistency)
    if noisy.any():                                  # stands whose phase does not follow the others
        valid = valid & ~noisy[None, :]
    slips = 0
    for _ in range(3):                               # repair the rare 2 pi slips, then recompute
        out = _planes(cum, valid, xy, k)
        fixed = _repair_slips(cum, out["residual"], valid)
        if not fixed:
            break
        slips += fixed
    out = _planes(cum, valid, xy, k)
    u_ref, v_ref = direction_cosines(ref_az, ref_el)
    az, el = az_el(u_ref + out["du0"] + out["du"], v_ref + out["dv0"] + out["dv"])
    total = out["static"][None, :] + out["residual"]
    coherence = np.abs(np.nanmean(np.where(valid, np.exp(1j * np.nan_to_num(total)), np.nan), axis=1))
    rms = np.degrees(np.sqrt(np.nanmean(np.where(valid, out["residual"] ** 2, np.nan), axis=1)))
    return dict(cum=cum, change=out["change"], residual=out["residual"], static=out["static"],
                du=out["du"], dv=out["dv"], az_deg=np.where(frames, az, np.nan), el_deg=np.where(frames, el, np.nan),
                coherence=np.where(frames, coherence, np.nan), rms_residual_deg=np.where(frames, rms, np.nan),
                stands_used=valid.sum(axis=1), frames=frames, valid=valid, slips=int(slips), noisy=noisy,
                stand_consistency=consistency,
                ref_az_deg=float(ref_az), ref_el_deg=float(ref_el), xy=xy, wavenumber=k,
                mean_az_el_deg=tuple(float(x) for x in az_el(u_ref + out["du0"], v_ref + out["dv0"])))


def _unwrap_in_time(psi, valid, xy, k):
    """Per-stand unwrapping with each increment taken nearest to the plane fitted to all increments."""
    T, N = psi.shape
    cum = np.full((T, N), np.nan)
    last, last_cum = np.full(N, np.nan), np.full(N, np.nan)
    for t in range(T):
        v = valid[t]
        if not v.any():
            continue
        have = v & np.isfinite(last)
        if have.sum() >= 4:
            inc = wrap(psi[t] - last)
            c, du, dv, _, _ = fit_plane(inc, xy, k, have)
            pred = c + k * (du * xy[:, 0] + dv * xy[:, 1])
            cum[t, have] = last_cum[have] + pred[have] + wrap(inc - pred)[have]
            new = v & ~have
            if new.any():                            # stand (re)appearing: nearest branch to the others' plane
                c, du, dv, _, _ = fit_plane(cum[t], xy, k, have)
                model = c + k * (du * xy[:, 0] + dv * xy[:, 1])
                cum[t, new] = model[new] + wrap(psi[t] - model)[new]
        else:
            cum[t, v] = psi[t, v]
        last[v], last_cum[v] = psi[t, v], cum[t, v]
    return cum


def _planes(cum, valid, xy, k):
    T, N = cum.shape
    mean = np.nanmean(np.where(valid, cum, np.nan), axis=0)
    seen = np.isfinite(mean)
    c0, du0, dv0, static, _ = fit_plane_wrapped(mean, xy, k, seen)
    change = np.where(valid, cum - mean[None, :], np.nan)
    du, dv = np.full(T, np.nan), np.full(T, np.nan)
    residual = np.full((T, N), np.nan)
    for t in range(T):
        if valid[t].sum() >= 4:
            _, du[t], dv[t], res, _ = fit_plane(change[t], xy, k, valid[t])
            residual[t] = np.where(valid[t], res, np.nan)
    return dict(change=change, residual=residual, static=np.where(seen, static, np.nan),
                du=du, dv=dv, du0=du0, dv0=dv0)


def _repair_slips(cum, residual, valid):
    """Remove 2 pi steps in a stand's residual time series (an unwrap that took the wrong branch)."""
    fixed = 0
    for n in range(cum.shape[1]):
        t = np.flatnonzero(valid[:, n] & np.isfinite(residual[:, n]))
        if len(t) < 2:
            continue
        step = np.diff(residual[t, n])
        jumps = np.round(step / (2 * np.pi))
        for i in np.flatnonzero(jumps):
            cum[t[i + 1]:, n] -= 2 * np.pi * jumps[i]
            fixed += 1
    return fixed


# ---------------------------------------------------------------------------
# Following one echo through the delay-Doppler frames
# ---------------------------------------------------------------------------

class EchoTracker:
    """Follow one echo: the strongest cell inside a small box around the last position.

    ``start``: (delay_ms, doppler_hz) to start from, or None for the strongest
    delay-localised echo of the first frame that has one."""

    def __init__(self, delay_ms, start=None, delay_bins=3, doppler_bins=3, min_snr_db=10.0):
        self.delay_ms = np.asarray(delay_ms, float)
        self.start, self.box = start, (int(delay_bins), int(doppler_bins))
        self.min_snr_db = float(min_snr_db)
        self.cell = None

    def step(self, power, doppler_hz):
        """power: (delay, doppler) channel-normalised power. Returns (row, col, snr_db) or None."""
        median = np.nanmedian(power)
        if self.cell is None:
            if self.start is not None:
                r = int(np.argmin(np.abs(self.delay_ms - self.start[0])))
                c = int(np.argmin(np.abs(np.asarray(doppler_hz) - self.start[1])))
                self.cell = self._best(power, r, c)
            else:
                cells = cu.echo_cells(power, 1, self.min_snr_db)
                if not cells:
                    return None
                self.cell = cells[0][:2]
        else:
            self.cell = self._best(power, *self.cell)
        r, c = self.cell
        snr = 10 * np.log10(max(power[r, c], 1e-30) / max(median, 1e-30))
        return int(r), int(c), float(snr)

    def _best(self, power, r, c):
        dr, dc = self.box
        r0, c0 = max(r - dr, 0), max(c - dc, 0)
        sub = np.where(np.isfinite(power[r0:r + dr + 1, c0:c + dc + 1]), power[r0:r + dr + 1, c0:c + dc + 1], -np.inf)
        i, j = np.unravel_index(int(np.argmax(sub)), sub.shape)
        return int(r0 + i), int(c0 + j)
