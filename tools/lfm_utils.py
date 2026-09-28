"""Signal-processing mathematics for LFM-CW (chirp) sounding.

Everything here works on plain NumPy arrays and knows nothing about files,
stations, or plotting:

* the reference chirp (``LFMWaveform``) and taper windows,
* dechirping + range FFT (fast time -> delay),
* the Doppler FFT (slow time -> Doppler),
* the sweep-to-sweep phase (pulse-pair) Doppler estimator,
* sweep alignment arithmetic,
* short-time spectra and LFM parameter estimation from a spectrogram.

Array layouts follow the RDC conventions used throughout the tools:
RDC0 = (adc, pulses, channels), RDC1 = (delay, pulses, channels),
RDC2 = (delay, doppler, channels) per frame.
"""

import dataclasses

import numpy as np
from scipy.signal import stft
from scipy.signal.windows import chebwin

C_M_PER_S = 299_792_458.0

WINDOW_CHOICES = ("hann", "hamming", "blackman", "bartlett", "cheb60", "cheb80", "cheb100", "cheb120", "none")


# ---------------------------------------------------------------------------
# Windows and the reference chirp
# ---------------------------------------------------------------------------

def window_from_arg(length, name):
    """Return a taper of ``length`` samples; ``None``/"none" is rectangular."""
    length = int(length)
    if name is None or str(name).lower() in ("none", "boxcar", "rect"):
        return np.ones(length)
    name = str(name).lower()
    if name in ("hann", "hanning"):
        return np.hanning(length)
    if name == "hamming":
        return np.hamming(length)
    if name == "blackman":
        return np.blackman(length)
    if name == "bartlett":
        return np.bartlett(length)
    if name in ("cheb", "chebwin", "dolph-chebyshev"):
        return chebwin(length, at=100)
    if name.startswith("cheb"):
        return chebwin(length, at=float(name[4:]))
    raise ValueError(f"Unknown window type: {name}")


def normalized_window(length, name):
    """Taper divided by its coherent gain, so a tone keeps its amplitude after an FFT."""
    w = window_from_arg(length, name).astype(np.float32)
    gain = float(np.mean(w))
    if gain == 0:
        raise ValueError(f"Window '{name}' has zero coherent gain")
    return w / gain


def parse_reference_gate(spec):
    """Parse ``"FREQ[,DUTY[,PHASE]]"`` (Hz, fraction, seconds) into a tuple or None."""
    if spec is None or str(spec).strip().lower() in ("", "none", "off"):
        return None
    parts = [float(p) for p in str(spec).split(",")]
    if not 1 <= len(parts) <= 3:
        raise ValueError("--reference-gate expects FREQ[,DUTY[,PHASE]]")
    freq, duty, phase = (parts + [0.5, 0.0][len(parts) - 1:])[:3]
    if freq <= 0 or not 0 < duty <= 1:
        raise ValueError("Reference gate needs frequency > 0 and 0 < duty <= 1")
    return float(freq), float(duty), float(phase)


@dataclasses.dataclass
class LFMWaveform:
    """One sweep of the transmitted chirp, centered on 0 Hz at baseband.

    ``bandwidth`` is signed: negative for a down-chirp. The optional reference
    gate multiplies the reference by an on/off pattern (for gated CODAR-like
    transmissions).
    """

    bandwidth: float
    sweep_frequency: float
    sample_rate: float
    reference_gate_frequency: float = None
    reference_gate_duty: float = 0.5
    reference_gate_phase: float = 0.0

    @property
    def period(self):
        return 1.0 / self.sweep_frequency

    @property
    def chirp_rate(self):
        return self.bandwidth * self.sweep_frequency

    @property
    def sweep_length(self):
        return int(round(self.sample_rate / self.sweep_frequency))

    @property
    def waveform(self):
        """Reference sweep with exactly ``sweep_length`` samples."""
        t = np.arange(self.sweep_length) / self.sample_rate
        k = self.chirp_rate
        lfm = np.exp(1j * 2 * np.pi * ((-self.bandwidth / 2) * t + (k / 2) * t ** 2))
        if self.reference_gate_frequency is not None:
            if self.reference_gate_frequency <= 0:
                raise ValueError("reference gate frequency must be > 0")
            if not 0 < self.reference_gate_duty <= 1:
                raise ValueError("reference gate duty must be in (0, 1]")
            period = 1.0 / self.reference_gate_frequency
            gate = ((t - self.reference_gate_phase) % period) < self.reference_gate_duty * period
            lfm = lfm * gate
        return lfm

    def reference(self, adc_start=0, adc_count=None):
        wave = self.waveform
        adc_count = len(wave) - int(adc_start) if adc_count is None else int(adc_count)
        ref = wave[int(adc_start):int(adc_start) + adc_count]
        if ref.size != adc_count:
            raise ValueError("adc_start + adc_count exceeds one sweep")
        return ref

    @classmethod
    def from_metadata(cls, meta, bandwidth=None):
        gate = meta.get("reference_gate_frequency")
        return cls(
            bandwidth=float(bandwidth if bandwidth is not None else meta["bandwidth_hz"]),
            sweep_frequency=float(meta["sweep_frequency_hz"]),
            sample_rate=float(meta["sample_rate_hz"]),
            reference_gate_frequency=None if gate is None else float(gate),
            reference_gate_duty=float(meta.get("reference_gate_duty", 0.5) or 0.5),
            reference_gate_phase=float(meta.get("reference_gate_phase", 0.0) or 0.0),
        )


# ---------------------------------------------------------------------------
# Fast time: dechirp and range FFT
# ---------------------------------------------------------------------------

def delay_axis_ms(lfm, n_bins=None, bandlimit=True):
    """Delay (ms) of each dechirped FFT bin, sorted ascending, and the bin order.

    The beat frequency of an echo at delay tau is ``-k * tau``. Delays are
    wrapped into one sweep period, and only bins within the chirp band are kept
    so every delay appears once.
    """
    n_bins = lfm.sweep_length if n_bins is None else int(n_bins)
    k = lfm.chirp_rate
    if k == 0:
        raise ValueError("Cannot map beat frequency to delay when bandwidth is zero")
    fb = np.fft.fftshift(np.fft.fftfreq(n_bins, d=1.0 / lfm.sample_rate))
    order = np.arange(n_bins)
    if bandlimit:
        half = abs(lfm.bandwidth) / 2.0
        if half < lfm.sample_rate / 2.0:
            keep = np.abs(fb) <= half + 0.5 * lfm.sample_rate / n_bins
            fb, order = fb[keep], order[keep]
    delay = np.mod(-fb / k * 1e3, lfm.period * 1e3)
    sort = np.argsort(delay)
    return delay[sort], order[sort]


def delay_band(lfm, n_bins, d_min=None, d_max=None):
    """FFT-bin indices and delays (ms) inside [d_min, d_max]."""
    delay, order = delay_axis_ms(lfm, n_bins)
    mask = np.ones(delay.shape, bool)
    if d_min is not None:
        mask &= delay >= float(d_min)
    if d_max is not None:
        mask &= delay <= float(d_max)
    if not np.any(mask):
        raise ValueError(f"No delay bins in the requested band; available {delay.min():.3f} to {delay.max():.3f} ms")
    return order[mask], delay[mask]


def dechirp_range_fft(rdc0, lfm, window="hann", adc_start=0, bins=None):
    """RDC0 (adc, pulses, channels) -> RDC1 (delay bins, pulses, channels).

    Multiplies each sweep by the conjugate reference chirp, applies the fast-time
    window, and FFTs along ADC samples. ``bins`` (from ``delay_band``) keeps only
    the requested delay bins, in ascending delay order.
    """
    rdc0 = np.asarray(rdc0)
    if rdc0.ndim != 3:
        raise ValueError("rdc0 must have shape (adc, pulses, channels)")
    n = rdc0.shape[0]
    ref = lfm.reference(adc_start, n).astype(np.complex64)
    w = normalized_window(n, window)
    beat = rdc0 * (np.conj(ref) * w)[:, None, None]
    spectra = np.fft.fftshift(np.fft.fft(beat, axis=0), axes=0)
    if bins is not None:
        spectra = spectra[np.asarray(bins)]
    return spectra.astype(np.complex64)


# ---------------------------------------------------------------------------
# Slow time: Doppler FFT and phase-derived Doppler
# ---------------------------------------------------------------------------

def default_nfft(n_pulses):
    return 1 << int(np.ceil(np.log2(max(int(n_pulses), 1))))


def doppler_axis_hz(nfft, prf):
    return np.fft.fftshift(np.fft.fftfreq(int(nfft), d=1.0 / float(prf)))


def doppler_fft(rdc1, prf, window="hann", nfft=None):
    """RDC1 (delay, pulses, channels) -> RDC2 (delay, doppler, channels), Doppler axis in Hz."""
    rdc1 = np.asarray(rdc1)
    if rdc1.ndim != 3:
        raise ValueError("rdc1 must have shape (delay, pulses, channels)")
    pulses = rdc1.shape[1]
    nfft = default_nfft(pulses) if nfft is None else int(nfft)
    if nfft < pulses:
        raise ValueError(f"Doppler FFT length {nfft} is shorter than the {pulses} sweeps it must cover")
    w = normalized_window(pulses, window)
    rdc2 = np.fft.fftshift(np.fft.fft(rdc1 * w[None, :, None], n=nfft, axis=1), axes=1)
    return rdc2.astype(np.complex64), doppler_axis_hz(nfft, prf)


def pulse_pair(rdc1, estimator_sweeps=1):
    """Sweep-to-sweep phase products, per channel.

    Returns ``(lag1, power, n_valid)``, each (delay, estimates, channels):
    ``lag1`` sums x[n+1]*conj(x[n]) over ``estimator_sweeps`` consecutive pairs,
    ``power`` is the mean pair power and ``n_valid`` counts finite pairs. Summing
    ``lag1`` over channels before taking its angle combines channels without
    needing their relative phases (no calibration required).
    """
    rdc1 = np.asarray(rdc1)
    e = int(estimator_sweeps)
    if e < 1:
        raise ValueError("estimator_sweeps must be >= 1")
    if rdc1.shape[1] < e + 1:
        raise ValueError(f"Need at least {e + 1} sweeps for phase-derived Doppler")
    prod = rdc1[:, 1:, :] * np.conj(rdc1[:, :-1, :])
    ppow = 0.5 * (np.abs(rdc1[:, 1:, :]) ** 2 + np.abs(rdc1[:, :-1, :]) ** 2)
    ok = np.isfinite(prod) & np.isfinite(ppow)
    prod = np.where(ok, prod, 0)
    ppow = np.where(ok, ppow, 0)

    def moving_sum(a):
        c = np.cumsum(a, axis=1)
        c = np.concatenate([np.zeros_like(c[:, :1]), c], axis=1)
        return c[:, e:] - c[:, :-e]

    n_valid = moving_sum(ok.astype(np.int32))
    lag1 = moving_sum(prod).astype(np.complex64)
    power = (moving_sum(ppow) / np.maximum(n_valid, 1)).astype(np.float32)
    return lag1, power, n_valid


def pulse_pair_time_s(n_estimates, estimator_sweeps, prf):
    """Center time of each estimate, relative to the start of the first sweep used."""
    return (np.arange(n_estimates) + (estimator_sweeps + 1) / 2.0) / float(prf)


def phase_doppler_hz(lag1, prf):
    """Doppler (Hz) from summed lag-1 products; unambiguous within +/- prf/2."""
    lag1 = np.asarray(lag1)
    fd = np.full(lag1.shape, np.nan, dtype=np.float32)
    good = np.abs(lag1) > 0
    fd[good] = np.angle(lag1[good]) * float(prf) / (2 * np.pi)
    return fd


# ---------------------------------------------------------------------------
# Sweep alignment
# ---------------------------------------------------------------------------

def sweep_alignment_samples(fractional_second, sweep_frequency, sample_rate, sweep_offset=0.0):
    """Samples to skip so sweeps start at ``sweep_offset`` (mod one period) after a UTC second.

    Returns 0 when the start time is unknown.
    """
    if fractional_second is None:
        return 0
    period = 1.0 / float(sweep_frequency)
    samples = int(round(((float(sweep_offset) - float(fractional_second)) % period) * float(sample_rate)))
    return 0 if samples >= int(round(sample_rate * period)) else samples


# ---------------------------------------------------------------------------
# Spectra of raw samples and LFM parameter estimation
# ---------------------------------------------------------------------------

def stft_power(iq, fs, window="hann", nperseg=1024, hop=512):
    """Power STFT summed over streams. Returns (freq_hz, time_s, power) with power (freq, time)."""
    iq = np.atleast_2d(np.asarray(iq))
    window = "boxcar" if window in (None, "none") else window
    power = None
    for stream in iq:
        f, t, z = stft(stream, fs=fs, window=window, nperseg=nperseg, noverlap=nperseg - hop,
                       return_onesided=False)
        p = np.abs(z) ** 2
        power = p if power is None else power + p
    return np.fft.fftshift(f), t, np.fft.fftshift(power, axes=0)


CODAR_SWEEP_FREQUENCIES_HZ = np.array([0.26, 0.43333, 1.0, 2.0, 2.007735, 3.0, 4.0])
CODAR_BANDWIDTHS_HZ = np.array([
    -22.05764, -24.983, -25.733913, -47.791553, -49.629688, -51.467827, -75.073242,
    -99.259377, -100.178452, -149.808136, -150.001999, -150.727203, -200.356903,
    -219.657333, -299.616272, -600.000854, -601.070679, 24.983, 25.733913, 89.49,
    99.259377, 99.931, 100.0, 101.097519, 220.576399, 600.151611,
]) * 1e3


def estimate_lfm_from_stft(freq_hz, time_s, power_db, utc_fractional_second=None, estimate_gate=True):
    """Estimate sweep rate, bandwidth, sweep start offset and gating from a spectrogram ridge.

    Candidates are snapped to known CODAR sweep settings. Returns a dict or None.
    """
    if power_db.size == 0 or time_s.size < 4 or freq_hz.size < 4:
        return None
    ridge_idx = np.nanargmax(power_db, axis=0)
    ridge_freq = freq_hz[ridge_idx]
    ridge_power = power_db[ridge_idx, np.arange(power_db.shape[1])]
    good = ridge_power >= np.nanpercentile(ridge_power, 55)
    if np.count_nonzero(good) < 4:
        good = np.ones_like(ridge_power, dtype=bool)
    f_good = ridge_freq[good]
    span = float(np.nanpercentile(f_good, 95) - np.nanpercentile(f_good, 5))
    if span <= 0:
        return None

    df = np.diff(ridge_freq)
    jumps = np.flatnonzero(np.abs(df) >= 0.45 * span)
    jumps = jumps[np.argsort(np.abs(df[jumps]))[::-1]]
    resets, min_gap = [], max(1, int(round(0.04 * len(time_s))))
    for idx in jumps:
        if all(abs(int(idx) - r) >= min_gap for r in resets):
            resets.append(int(idx))
    resets = np.array(sorted(resets), dtype=int)

    sweep_period = first_start = None
    direction = "unknown"
    autocorr_period = _period_from_autocorr(time_s, ridge_freq)
    if resets.size >= 2:
        reset_times = time_s[resets + 1]
        periods = np.diff(reset_times)
        periods = periods[periods > 0]
        if periods.size:
            sweep_period = float(np.median(periods))
            direction = "down" if float(np.median(df[resets])) > 0 else "up"
            if autocorr_period is not None and sweep_period / autocorr_period > 1.35:
                sweep_period /= max(1, int(round(sweep_period / autocorr_period)))
            first_start = _circular_mean_phase(reset_times, sweep_period)
    elif autocorr_period is not None:
        sweep_period = autocorr_period

    if direction == "unknown" and sweep_period is not None:
        phase = np.mod(time_s, sweep_period) / sweep_period
        middle = (phase > 0.15) & (phase < 0.85)
        if np.count_nonzero(middle) >= 4:
            direction = "up" if np.polyfit(time_s[middle], ridge_freq[middle], 1)[0] >= 0 else "down"

    raw_bandwidth = -span if direction == "down" else span
    raw_sweep_frequency = None if sweep_period is None else 1.0 / sweep_period
    fit = _fit_codar_candidates(freq_hz, time_s, power_db, ridge_freq, raw_sweep_frequency, direction, first_start)
    if fit:
        bandwidth, sweep_frequency, first_start = fit["bandwidth_hz"], fit["sweep_frequency_hz"], fit["first_sweep_start_s"]
        sweep_period = 1.0 / sweep_frequency
    else:
        bandwidth = _snap_codar_bandwidth(raw_bandwidth)
        sweep_frequency = None
        if raw_sweep_frequency is not None:
            sweep_frequency = float(CODAR_SWEEP_FREQUENCIES_HZ[np.argmin(np.abs(CODAR_SWEEP_FREQUENCIES_HZ - raw_sweep_frequency))])
            sweep_period = 1.0 / sweep_frequency
    half = abs(bandwidth) / 2.0
    f_start, f_end = (half, -half) if bandwidth < 0 else (-half, half)
    utc_offset = None
    if first_start is not None and utc_fractional_second is not None and sweep_period is not None:
        utc_offset = float((utc_fractional_second + first_start) % sweep_period)
    gate = _gate_from_ridge(time_s, ridge_power, sweep_period) if estimate_gate and sweep_period else None
    return dict(bandwidth_hz=bandwidth, raw_bandwidth_hz=raw_bandwidth, sweep_frequency_hz=sweep_frequency,
                raw_sweep_frequency_hz=raw_sweep_frequency, sweep_period_s=sweep_period,
                first_sweep_start_s=first_start, utc_offset_s=utc_offset,
                fit_score=None if not fit else fit["score"], f_start_hz=f_start, f_end_hz=f_end,
                utc_fractional_second=utc_fractional_second, gate=gate)


def _snap_codar_bandwidth(raw):
    if raw is None or not np.isfinite(raw):
        return None
    sign = -1 if raw < 0 else 1
    cands = CODAR_BANDWIDTHS_HZ[np.sign(CODAR_BANDWIDTHS_HZ) == sign]
    target = 2.0 * abs(raw)
    wider = np.abs(cands) >= target
    if np.any(wider):
        return float(cands[wider][np.argmin(np.abs(cands[wider]) - target)])
    return float(cands[np.argmin(np.abs(np.abs(cands) - target))])


def _fit_codar_candidates(freq_hz, time_s, power_db, ridge_freq, raw_sweep_frequency, direction, initial_offset):
    step = abs(float(np.median(np.diff(freq_hz))))
    if not np.isfinite(step) or step == 0:
        return None
    floor = float(np.nanpercentile(power_db, 25))
    z = (power_db - floor) / max(float(np.nanpercentile(power_db, 99) - floor), 1.0)
    if raw_sweep_frequency is None:
        sweeps = CODAR_SWEEP_FREQUENCIES_HZ
    else:
        sweeps = CODAR_SWEEP_FREQUENCIES_HZ[np.argsort(np.abs(CODAR_SWEEP_FREQUENCIES_HZ - raw_sweep_frequency))[:4]]
    bws = {"down": CODAR_BANDWIDTHS_HZ[CODAR_BANDWIDTHS_HZ < 0],
           "up": CODAR_BANDWIDTHS_HZ[CODAR_BANDWIDTHS_HZ > 0]}.get(direction, CODAR_BANDWIDTHS_HZ)
    idx = np.arange(0, time_s.size, max(1, int(np.ceil(time_s.size / 1800))))
    t_eval, ridge_eval = time_s[idx], ridge_freq[idx]
    best = None
    for sf_ in sweeps:
        period = 1.0 / float(sf_)
        if initial_offset is None:
            offsets = np.linspace(0.0, period, 48, endpoint=False)
        else:
            half_window = min(period / 2.0, 0.08)
            offsets = (initial_offset + np.linspace(-half_window, half_window, 41)) % period
        for bw in bws:
            if abs(bw) / 2.0 > max(abs(freq_hz[0]), abs(freq_hz[-1])) * 1.2:
                continue
            for off in offsets:
                pred = -bw / 2.0 + bw * (np.mod(t_eval - off, period) / period)
                inband = (pred >= freq_hz[0]) & (pred <= freq_hz[-1])
                if np.count_nonzero(inband) < max(4, int(0.35 * pred.size)):
                    continue
                pidx = np.clip(np.rint((pred[inband] - freq_hz[0]) / step).astype(int), 0, len(freq_hz) - 1)
                err = np.abs(pred[inband] - ridge_eval[inband]) / max(step, 1.0)
                score = float(np.nanmean(z[pidx, idx[inband]] + 0.6 * np.exp(-0.5 * (err / 2.5) ** 2)))
                if best is None or score > best["score"]:
                    best = dict(score=score, sweep_frequency_hz=float(sf_), bandwidth_hz=float(bw),
                                first_sweep_start_s=float(off))
    return best


def _period_from_autocorr(time_s, ridge_freq):
    if time_s.size > 6000:
        s = int(np.ceil(time_s.size / 6000))
        time_s, ridge_freq = time_s[::s], ridge_freq[::s]
    dt = float(np.median(np.diff(time_s)))
    if not np.isfinite(dt) or dt <= 0 or time_s.size < 8:
        return None
    x = np.nan_to_num(ridge_freq - np.nanmean(ridge_freq))
    if np.std(x) <= 0:
        return None
    nfft = 1 << int(np.ceil(np.log2(2 * len(x) - 1)))
    X = np.fft.fft(x, n=nfft)
    corr = np.fft.ifft(X * np.conj(X)).real[:len(x)]
    if corr[0] <= 0:
        return None
    corr /= corr[0]
    lo = max(3, int(round(0.025 / dt)))
    hi = min(len(corr) - 2, int(round((time_s[-1] - time_s[0]) * 0.75 / dt)))
    cands = [(corr[l], l) for l in range(lo, hi + 1)
             if corr[l] > 0.2 and corr[l] >= corr[l - 1] and corr[l] >= corr[l + 1]]
    if not cands:
        return None
    best = max(c for c, _ in cands)
    return float(min(l for c, l in cands if c >= 0.75 * best) * dt)


def _circular_mean_phase(times, period):
    if period is None or period <= 0 or len(times) == 0:
        return None
    angle = np.angle(np.mean(np.exp(2j * np.pi * np.mod(times, period) / period)))
    return float((angle % (2 * np.pi)) * period / (2 * np.pi))


def _gate_from_ridge(time_s, ridge_power_db, sweep_period):
    if time_s.size > 5000:
        s = int(np.ceil(time_s.size / 5000))
        time_s, ridge_power_db = time_s[::s], ridge_power_db[::s]
    dt = float(np.median(np.diff(time_s)))
    if not np.isfinite(dt) or dt <= 0 or sweep_period <= 0:
        return None
    x = np.nan_to_num(ridge_power_db - np.nanmedian(ridge_power_db))
    if np.std(x) < 1.5:
        return None
    corr = np.correlate(x, x, mode="full")[len(x) - 1:]
    if corr[0] <= 0:
        return None
    corr = corr / corr[0]
    lo, hi = 2, min(len(corr) - 1, int(round(0.5 * sweep_period / dt)))
    if hi <= lo:
        return None
    lag = int(lo + np.argmax(corr[lo:hi + 1]))
    if corr[lag] < 0.25:
        return None
    gate_period = lag * dt
    phase = np.mod(time_s, gate_period)
    nb = max(8, min(64, int(round(gate_period / dt))))
    edges = np.linspace(0, gate_period, nb + 1)
    folded = np.array([np.nanmedian(ridge_power_db[(phase >= edges[i]) & (phase < edges[i + 1])])
                       if np.any((phase >= edges[i]) & (phase < edges[i + 1])) else np.nan for i in range(nb)])
    if np.count_nonzero(np.isfinite(folded)) < 4:
        return None
    lo_p, hi_p = np.nanpercentile(folded, 20), np.nanpercentile(folded, 80)
    if hi_p - lo_p < 4.0:
        return None
    on = folded >= lo_p + 0.45 * (hi_p - lo_p)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return dict(frequency_hz=1.0 / gate_period, period_s=gate_period, duty=float(np.mean(on)),
                phase_s=float(np.mean(centers[on]) % gate_period), confidence=float(corr[lag]))
