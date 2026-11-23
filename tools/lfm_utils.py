"""Signal-processing mathematics for LFM-CW (chirp) sounding.

Everything here works on plain NumPy arrays and knows nothing about files,
stations, or plotting:

* the reference chirp (``LFMWaveform``) and taper windows,
* range compression (fast time -> delay): a circular matched filter against the
  whole periodic sweep (default), or dechirp + range FFT (stretch processing),
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

# Range compression of one sweep.
# * "matched": circular cross-correlation of the sweep with the reference chirp,
#   y[l] = sum_n x[n + l] conj(ref[n]) (computed as IFFT(FFT(x) conj(FFT(ref)))). A CODAR sawtooth
#   repeats every sweep, so one received sweep is the reference rotated by the echo delay and every
#   delay in the sweep compresses fully. The reference is zeroed where its instantaneous frequency is
#   outside +/- INBAND_FRACTION * fs/2 (TBN has filtered that part away) and the fast-time window is
#   applied over the in-band part only.
# * "dechirp": multiply the sweep by conj(ref), window the whole sweep, FFT (stretch processing).
#   An echo at delay tau gives two beat tones, -k tau after tau and B - k tau before it; they share a
#   bin only when |B| = fs, so an echo far into the sweep loses power and leaves a sweep-wrap copy
#   (fs - |B|) / |k| away. Kept for comparison and for files made before 2026-11-21.
RANGE_METHODS = ("matched", "dechirp")
DEFAULT_RANGE_METHOD = "matched"
INBAND_FRACTION = 0.98


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
    transmissions). ``range_method`` ("matched" or "dechirp", see RANGE_METHODS)
    decides how sweeps are range-compressed and so which delay axis applies.
    """

    bandwidth: float
    sweep_frequency: float
    sample_rate: float
    reference_gate_frequency: float = None
    reference_gate_duty: float = 0.5
    reference_gate_phase: float = 0.0
    range_method: str = DEFAULT_RANGE_METHOD

    def __post_init__(self):
        if self.range_method not in RANGE_METHODS:
            raise ValueError(f"range method must be one of {', '.join(RANGE_METHODS)}, not {self.range_method!r}")

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

    def instantaneous_frequency_hz(self):
        """Baseband frequency of the reference at each of its ``sweep_length`` samples: -B/2 + k t."""
        return -self.bandwidth / 2 + self.chirp_rate * (np.arange(self.sweep_length) / self.sample_rate)

    def inband_mask(self, fraction=INBAND_FRACTION):
        """True where the reference lies inside the receive band, |f_inst| < fraction * fs/2.

        f_inst is linear in time, so the in-band samples are one contiguous run (all of the sweep for
        |B| < 0.98 fs; 87.5-412.5 ms of 500 ms for the -150.727 kHz CODARs at 100 kHz)."""
        return np.abs(self.instantaneous_frequency_hz()) < fraction * self.sample_rate / 2

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
            range_method=meta.get("range_method") or (DEFAULT_RANGE_METHOD if meta.get("rdc_stage") in (None, 0)
                                                      else "dechirp"),
        )


# ---------------------------------------------------------------------------
# Fast time: dechirp and range FFT
# ---------------------------------------------------------------------------

def delay_axis_ms(lfm, n_bins=None, bandlimit=True):
    """Delay (ms) of each range bin, sorted ascending, and the bin order.

    Matched filter: bin l is a lag of l samples, so delay = l / fs over the whole sweep.
    Dechirp: the beat frequency of an echo at delay tau is ``-k * tau``. Delays are
    wrapped into one sweep period, and only bins within the chirp band are kept
    so every delay appears once.
    """
    n_bins = lfm.sweep_length if n_bins is None else int(n_bins)
    if lfm.range_method == "matched":
        if n_bins != lfm.sweep_length:
            raise ValueError(f"The matched filter needs whole sweeps ({lfm.sweep_length} samples, not {n_bins}); "
                             "use --range-method dechirp with --adc-start/--adc-count")
        return np.arange(n_bins) / lfm.sample_rate * 1e3, np.arange(n_bins)
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


def delay_spans(delay):
    """The delays an axis reaches, as text: '0.000-28.501 and 471.499-499.999 ms'.

    A chirp wider than the sample rate only reaches the delays whose beat frequency is within
    +/- fs/2, so its axis has a gap."""
    d = np.sort(np.asarray(delay, float))
    if d.size == 0:
        return "none"
    step = np.median(np.diff(d)) if d.size > 1 else 0.0
    cut = np.flatnonzero(np.diff(d) > 3 * step) if step > 0 else np.array([], int)
    lo = np.r_[d[0], d[cut + 1]]
    hi = np.r_[d[cut], d[-1]]
    return " and ".join(f"{a:.3f}-{b:.3f}" for a, b in zip(lo, hi)) + " ms"


def no_delay_bins(delay, d_min, d_max):
    """The error for a delay band holding no bins."""
    want = f"{'-inf' if d_min is None else f'{float(d_min):.3f}'}-{'inf' if d_max is None else f'{float(d_max):.3f}'} ms"
    return ValueError(f"No delay bins in the requested band {want}; this bandwidth and sweep rate reach "
                      f"{delay_spans(delay)}")


def delay_band(lfm, n_bins, d_min=None, d_max=None):
    """FFT-bin indices and delays (ms) inside [d_min, d_max]."""
    delay, order = delay_axis_ms(lfm, n_bins)
    mask = np.ones(delay.shape, bool)
    if d_min is not None:
        mask &= delay >= float(d_min)
    if d_max is not None:
        mask &= delay <= float(d_max)
    if not np.any(mask):
        raise no_delay_bins(delay, d_min, d_max)
    return order[mask], delay[mask]


_MF_CACHE = {}


def matched_reference_spectrum(lfm, window="hann", fraction=INBAND_FRACTION):
    """conj(FFT) of the matched-filter reference: the whole sweep (with its gate, if any), zeroed
    outside the receive band (``LFMWaveform.inband_mask``), times ``window`` stretched over the
    in-band samples only and divided by its mean there, so an in-band echo of unit amplitude
    compresses to a peak equal to the number of in-band samples (the sweep length when |B| < fs,
    like the normalised dechirp window)."""
    key = (lfm.bandwidth, lfm.sweep_frequency, lfm.sample_rate, lfm.reference_gate_frequency,
           lfm.reference_gate_duty, lfm.reference_gate_phase, str(window), float(fraction))
    if key not in _MF_CACHE:
        inband = np.flatnonzero(lfm.inband_mask(fraction))
        if not inband.size:
            raise ValueError("The reference chirp has no samples inside the receive band")
        w = np.zeros(lfm.sweep_length)
        w[inband] = window_from_arg(inband.size, window)
        gain = float(np.mean(w[inband]))
        if gain == 0:
            raise ValueError(f"Window '{window}' has zero coherent gain")
        if len(_MF_CACHE) > 16:
            _MF_CACHE.clear()
        _MF_CACHE[key] = np.conj(np.fft.fft(lfm.waveform * (w / gain)))
    return _MF_CACHE[key]


def matched_filter_range(rdc0, lfm, window="hann", bins=None):
    """RDC0 (adc, pulses, channels) -> RDC1 (lag bins, pulses, channels) by circular matched filtering.

    Each whole sweep is correlated circularly with the in-band, tapered reference:
    y[l] = sum_n x[n + l] conj(ref[n]) = IFFT(FFT(x) * conj(FFT(ref))) (numpy's ifft carries the 1/n).
    A unit echo peaks at the number of in-band samples, as the normalised dechirp does. Bin l is a delay of
    l / fs. ``bins`` (from ``delay_band``) keeps only the requested lags.
    """
    rdc0 = np.asarray(rdc0)
    if rdc0.ndim != 3:
        raise ValueError("rdc0 must have shape (adc, pulses, channels)")
    n = rdc0.shape[0]
    if n != lfm.sweep_length:
        raise ValueError(f"The matched filter needs whole sweeps ({lfm.sweep_length} samples, not {n}); "
                         "use --range-method dechirp with --adc-start/--adc-count")
    spec = matched_reference_spectrum(lfm, window)
    # double precision: the result does not depend on how sweeps and channels are blocked
    out = np.fft.ifft(np.fft.fft(rdc0.astype(np.complex128), axis=0) * spec[:, None, None], axis=0)
    if bins is not None:
        out = out[np.asarray(bins)]
    return out.astype(np.complex64)


def range_compress(rdc0, lfm, window="hann", adc_start=0, bins=None):
    """RDC0 -> RDC1 with the waveform's ``range_method`` (matched filter or dechirp + FFT)."""
    if lfm.range_method == "matched":
        if int(adc_start):
            raise ValueError("The matched filter needs whole sweeps; --adc-start needs --range-method dechirp")
        return matched_filter_range(rdc0, lfm, window, bins)
    return dechirp_range_fft(rdc0, lfm, window, adc_start, bins)


def dechirp_range_fft(rdc0, lfm, window="hann", adc_start=0, bins=None):
    """RDC0 (adc, pulses, channels) -> RDC1 (delay bins, pulses, channels) by stretch processing.

    Multiplies each sweep by the conjugate reference chirp, applies the fast-time
    window, and FFTs along ADC samples. ``bins`` (from ``delay_band`` with a
    "dechirp" waveform) keeps only the requested delay bins, in ascending delay order.
    Use ``range_compress`` to follow the waveform's range method.
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


# ---------------------------------------------------------------------------
# Waveform scan: which (bandwidth, sweep rate) compresses the received echoes?
# ---------------------------------------------------------------------------
# An echo only compresses into a sharp delay peak when it is dechirped with the
# chirp rate and sweep period it was sent with. A chirp-rate error dk smears the
# peak over |dk| T^2 delay bins (T = analysis length), so over a full 0.5 s sweep
# the bandwidth must be right to a few Hz. ``scan_lfm.py`` therefore works in
# steps: chirp rates from the lag-product spectrum (blind), a coarse scan on
# short pieces tiling each sweep (tolerant: 50 Hz steps), the sweep rate from
# those pieces, and a fine scan on whole sweeps.

LAGS_S = (0.005, 0.007, 0.011, 0.013, 0.017)     # ms values with no common factor, so sidelines rarely line up


def lag_product_spectrum(iq, fs, lag_s=LAGS_S, segment_s=0.5):
    """Power spectrum of x(t) x*(t - lag), summed over streams and segments, averaged over lags.

    A chirp of rate k [Hz/s] becomes a tone at k * lag whatever its delay, sweep
    start, carrier offset or Doppler, so each transmitted waveform gives a peak
    before any dechirping (a steady carrier gives a peak at 0). A gated (FMICW)
    transmitter is only seen while it is on at both t and t - lag, so at some lags
    the tone at k is weak and the power sits in sidelines at k +/- n f_gate / lag.
    The sidelines move with the lag and the true chirp rate does not, so ``lag_s``
    may be several lags: each spectrum is divided by its median and put on the
    chirp-rate axis of the longest lag (the finest), and the median over lags is taken.
    Returns (chirp_rate_hz_per_s, power relative to the median), increasing in rate.
    """
    iq = np.atleast_2d(np.asarray(iq))
    lags = sorted(np.atleast_1d(np.asarray(lag_s, float)))
    seg = int(round(segment_s * fs))
    w = normalized_window(seg, "hann")
    f = np.fft.fftshift(np.fft.fftfreq(seg, d=1.0 / fs))
    grid, out = None, None
    for lag_s1 in lags[::-1]:
        lag = int(round(lag_s1 * fs))
        if lag < 1 or iq.shape[1] < lag + seg:
            raise ValueError("Too few samples for the lag-product spectrum; use a longer --duration")
        total = np.zeros(seg)
        for x in iq:
            x = np.nan_to_num(np.asarray(x, np.complex64))
            y = x[lag:] * np.conj(x[:-lag])
            n = y.size // seg
            total += np.sum(np.abs(np.fft.fft(y[:n * seg].reshape(n, seg) * w, axis=1)) ** 2, axis=0)
        total = np.fft.fftshift(total)
        total /= np.median(total[total > 0]) if np.any(total > 0) else 1.0
        k = f / (lag / fs)
        if grid is None:
            grid, out = k, [total]
        else:
            out.append(np.interp(grid, k, total))
    # the median over lags keeps a line present at most lags (the chirp rate) and drops one
    # present at only a few (a gating sideline, which sits elsewhere at every lag)
    return grid, np.median(out, axis=0)


def lag_resolution(fs, lag_s=LAGS_S, segment_s=0.5):
    """Chirp-rate width of one bin of the lag-product spectrum at its shortest lag [Hz/s]."""
    return fs / round(segment_s * fs) / min(np.atleast_1d(np.asarray(lag_s, float)))


def spectrum_peaks(axis, power, n_peaks=4, min_db=8.0, exclude_below=0.0, min_separation=0.0):
    """Strongest local maxima of ``power`` as [(position, dB above the median)], strongest first.

    Positions are refined with a parabola through the peak and its neighbours.
    ``exclude_below`` skips |position| below it; peaks closer than ``min_separation``
    to a stronger one are dropped.
    """
    axis, power = np.asarray(axis, float), np.asarray(power, float)
    med = np.nanmedian(power[power > 0]) if np.any(power > 0) else 1.0
    db = 10 * np.log10(np.maximum(power, 1e-30) / med)
    i = np.arange(1, len(db) - 1)
    local = i[(db[i] >= db[i - 1]) & (db[i] >= db[i + 1]) & (db[i] >= min_db) & (np.abs(axis[i]) >= exclude_below)]
    out = []
    for j in local[np.argsort(db[local])[::-1]]:
        a, b, c = db[j - 1], db[j], db[j + 1]
        den = a - 2 * b + c
        shift = 0.0 if den == 0 else float(np.clip(0.5 * (a - c) / den, -0.5, 0.5))
        pos = float(axis[j] + shift * (axis[j + 1] - axis[j - 1]) / 2)
        if all(abs(pos - p) >= min_separation for p, _ in out):
            out.append((pos, float(b)))
        if len(out) >= n_peaks:
            break
    return out


def sweep_starts(n_samples, fs, sweep_frequency, skip=0):
    """First sample of every complete sweep (period fs/sweep_frequency, need not be an integer) and its length."""
    period = float(fs) / float(sweep_frequency)
    length = int(np.floor(period))
    usable = int(n_samples) - int(skip) - length
    if usable < 0:
        return np.zeros(0, int), length
    return int(skip) + np.round(np.arange(int(usable // period) + 1) * period).astype(int), length


def scan_windows(fs, sweep_frequency, length_s=0.025):
    """(start, length) of equal pieces that tile one sweep, for the coarse scan.

    Short pieces tolerate bandwidth errors (the smear grows with the square of the
    analysis length), and tiling the whole sweep with equal weights makes the
    sweep rate testable: at the true rate every piece puts the echo in the same
    delay bin, at half or double the rate the pieces split between two bins.
    """
    sweep = int(np.floor(fs / sweep_frequency))
    n = max(16, min(int(round(length_s * fs)), sweep))
    return [(i * n, n) for i in range(sweep // n)]


def whole_sweep_method(fs, sweep_frequency, method=DEFAULT_RANGE_METHOD):
    """The range method ``compression_profile`` uses on whole sweeps: the matched filter needs an
    integer number of samples per sweep, otherwise dechirp is used."""
    period = float(fs) / float(sweep_frequency)
    return method if method == "dechirp" or abs(period - round(period)) < 1e-9 else "dechirp"


def compression_profile(iq, fs, bandwidth, sweep_frequency, skip=0, windows=None, window="hann", max_sweeps=None,
                        method=DEFAULT_RANGE_METHOD):
    """Range-compressed power on the delay axis, summed incoherently over sweeps, streams and windows.

    ``windows``: None for whole sweeps (``whole_sweep_method``: matched filter by default), or
    (start, length) pieces of every sweep (equal lengths), each dechirped with the matching piece
    of the reference. Delays are mod one sweep, relative to sweeps starting ``skip`` samples into
    ``iq``. Returns (delay_ms, power).
    """
    iq = np.atleast_2d(np.asarray(iq))
    starts, length = sweep_starts(iq.shape[1], fs, sweep_frequency, skip)
    if max_sweeps:
        starts = starts[:int(max_sweeps)]
    if not starts.size:
        raise ValueError("Not one complete sweep in the data; use a longer --duration")
    if windows is None and whole_sweep_method(fs, sweep_frequency, method) == "matched":
        lfm = LFMWaveform(float(bandwidth), float(sweep_frequency), float(fs), range_method="matched")
        if length == lfm.sweep_length:
            spec = matched_reference_spectrum(lfm, window)
            total = np.zeros(length)
            for x in iq:
                x = np.asarray(x)
                for i in range(0, starts.size, 64):
                    seg = x[starts[i:i + 64, None] + np.arange(length)]
                    seg = np.where(np.isfinite(seg), seg, 0).astype(np.complex128)
                    total += np.sum(np.abs(np.fft.ifft(np.fft.fft(seg, axis=1) * spec, axis=1)) ** 2, axis=0)
            delay, order = delay_axis_ms(lfm, length)
            return delay, total[order]
    lfm = LFMWaveform(float(bandwidth), float(sweep_frequency), float(fs), range_method="dechirp")
    ref = np.conj(lfm.waveform[:length]).astype(np.complex64)
    windows = [(0, length)] if windows is None else windows
    n = windows[0][1]
    w = normalized_window(n, window)
    total = np.zeros(n)
    for x in iq:
        x = np.asarray(x)
        for start, count in windows:
            seg = x[starts[:, None] + start + np.arange(count)]
            seg = np.where(np.isfinite(seg), seg, 0) * (ref[start:start + count] * w)
            total += np.sum(np.abs(np.fft.fft(seg, axis=1)) ** 2, axis=0)
    delay, order = delay_axis_ms(lfm, n)
    return delay, np.fft.fftshift(total)[order]


def compression_profile_clean(iq, fs, bandwidth, sweep_frequency, skip=0, window="hann", max_sweeps=None, block=20,
                              echo_ms=None, n_orders=None, budget_bytes=256 * 2 ** 20):
    """``compression_profile`` on whole sweeps with the gating-sideband comb of the strongest echo removed.

    The comb is found on the matched-filter output (``detect_gating_comb``) and subtracted block by
    block of ``block`` sweeps (``clean_gating_comb``) before the power is summed over sweeps and
    streams. Streams are compressed in groups that fit in ``budget_bytes``; the comb is measured on
    the first group. ``echo_ms``: take the echo within +/-1 ms of this delay instead of the strongest.
    ``n_orders``: orders measured on each side (default: every order that fits in half a sweep, not
    lwa_radar's 8; orders less coherent with the echo than SIDEBAND_MIN_COHERENCE_DB are left alone).
    The CLEAN needs the matched filter's whole lag axis, so with dechirped sweeps the profile is
    returned as it is. Returns (delay_ms, power after the CLEAN, power before it, comb); the two
    powers are the same when no comb was found."""
    iq = np.atleast_2d(np.asarray(iq))
    starts, length = sweep_starts(iq.shape[1], fs, sweep_frequency, skip)
    if max_sweeps:
        starts = starts[:int(max_sweeps)]
    if not starts.size:
        raise ValueError("Not one complete sweep in the data; use a longer --duration")
    lfm = LFMWaveform(float(bandwidth), float(sweep_frequency), float(fs), range_method="matched")
    if whole_sweep_method(fs, sweep_frequency) != "matched" or length != lfm.sweep_length:
        d, p = compression_profile(iq, fs, bandwidth, sweep_frequency, skip, None, window, max_sweeps)
        return d, p, p, dict(detected=False, reason="dechirped sweeps; the CLEAN needs the matched filter")
    spec = matched_reference_spectrum(lfm, window)
    group = max(1, int(budget_bytes // max(length * starts.size * 8, 1)))
    raw, clean, comb = np.zeros(length), np.zeros(length), None
    for g0 in range(0, iq.shape[0], group):
        rows = iq[g0:g0 + group]
        Y = np.empty((length, starts.size, len(rows)), np.complex64)
        for j, x in enumerate(rows):
            seg = np.asarray(x)[starts[:, None] + np.arange(length)]
            seg = np.where(np.isfinite(seg), seg, 0).astype(np.complex128)
            Y[:, :, j] = np.fft.ifft(np.fft.fft(seg, axis=1) * spec, axis=1).T
        raw += np.sum(np.abs(Y.astype(np.complex128)) ** 2, axis=(1, 2))
        if comb is None:
            peak = None
            if echo_ms is not None:
                h = _ms_bins(1.0, fs)
                power = np.sum(np.abs(Y) ** 2, axis=(1, 2))
                idx = (int(round(echo_ms * 1e-3 * fs)) + np.arange(-h, h + 1)) % length
                peak = int(idx[np.argmax(power[idx])])
            orders = n_orders or int(length // (2 * _ms_bins(SIDEBAND_LAGS_MS[0], fs))) + 1
            comb = detect_gating_comb(Y, fs, peak, n_orders=orders)
        if comb.get("detected"):
            Y, _ = clean_gating_comb(Y, comb, fs, n_lags=length, block=block)
            clean += np.sum(np.abs(Y.astype(np.complex128)) ** 2, axis=(1, 2))
    if not comb.get("detected"):
        clean = raw
    delay, order = delay_axis_ms(lfm, length)
    return delay, clean[order], raw[order], comb


SNR_WINDOW_MS = 25.0     # noise taken from delays within this distance of the echo ...
SNR_GUARD_MS = 1.0       # ... but not within this distance of any echo or copy


def local_snr_db(peak, power, delay_ms, at_ms, exclude_ms=(), window_ms=SNR_WINDOW_MS, guard_ms=SNR_GUARD_MS,
                 period_ms=None):
    """SNR of an echo [dB]: 10 log10((P_peak - N) / N), N = mean noise power per bin near it.

    ``power`` is linear power with delay along its first axis (a delay profile, or a delay x
    Doppler map); ``peak`` is the echo's power in the same units. N is the mean of the bins within
    ``window_ms`` of the echo, leaving out every bin within ``guard_ms`` of the echo and of
    ``exclude_ms`` (the other echoes and the copies) and any bin above 10x their median (another
    signal). The floor of a delay profile follows the receive filter, so a local N is used rather
    than the whole profile. Falls back to the whole axis when fewer than 20 bins are left.
    Returns (snr_db, noise)."""
    p = np.asarray(power, float)
    d = np.asarray(delay_ms, float)

    def dist(a, b):
        x = np.abs(a - b)
        return np.minimum(x, period_ms - x) if period_ms else x

    clear = np.ones(d.shape, bool)
    for e in (at_ms,) + tuple(exclude_ms):
        clear &= dist(d, e) > guard_ms
    for rows in (clear & (dist(d, at_ms) <= window_ms), clear):
        v = p[rows].ravel()
        v = v[np.isfinite(v) & (v > 0)]
        if v.size >= 20:
            break
    if not v.size:
        return float("nan"), float("nan")
    noise = float(np.mean(v[v <= 10 * np.median(v)]))
    return float(10 * np.log10(max(float(peak) - noise, 1e-30 * noise) / noise)), noise


def contrast_db(power):
    """Peak over median of a delay profile [dB]: how well an echo compressed."""
    p = np.asarray(power, float)
    p = p[np.isfinite(p) & (p > 0)]
    return float(10 * np.log10(p.max() / np.median(p))) if p.size else float("nan")


def peak_snr_db(power):
    """(peak - median) / median of a delay profile [dB]: the strongest echo's SNR."""
    p = np.asarray(power, float)
    p = p[np.isfinite(p) & (p > 0)]
    if not p.size:
        return float("nan")
    med = np.median(p)
    return float(10 * np.log10(max(p.max() - med, 1e-30 * med) / med))


def parabolic_peak(x, y):
    """(x, y) of the maximum of ``y`` refined with a parabola through its neighbours."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    i = int(np.nanargmax(y))
    if 0 < i < len(y) - 1:
        a, b, c = y[i - 1], y[i], y[i + 1]
        den = a - 2 * b + c
        if den < 0:
            s = 0.5 * (a - c) / den
            return float(x[i] + s * (x[i + 1] - x[i - 1]) / 2), float(b - 0.25 * (a - c) * s)
    return float(x[i]), float(y[i])


# ---------------------------------------------------------------------------
# Sweep rate measured from the resets of the received chirp
# ---------------------------------------------------------------------------
# Dechirping with a continuous chirp of rate k that never resets leaves each
# echo of a k-chirp at one beat frequency for a whole sweep. Every time the
# transmitter starts a new sweep the beat jumps by the sweep bandwidth (mod the
# sample rate), and while the sweep is outside the receive band the echo
# vanishes. So the beat track of short windows is a staircase whose steps are
# the transmitter's resets, and their spacing is the sweep period: measured,
# not assumed. (A sweep of exactly the sample rate jumps by ~0 and cannot be
# measured this way; then only the dechirped sweep-rate test is left.)

def beat_track(iq, fs, chirp_rate, window_s=0.025):
    """(times, strongest beat bin, its dB above the median, bins per window) for a continuous dechirp."""
    n = int(round(window_s * fs))
    total = None
    w = np.hanning(n).astype(np.float32)
    for x in np.atleast_2d(iq):
        x = np.nan_to_num(np.asarray(x, np.complex64))
        m = x.size // n
        t = np.arange(m * n) / fs
        y = (x[:m * n] * np.exp(-1j * np.pi * chirp_rate * t ** 2).astype(np.complex64)).reshape(m, n)
        p = np.abs(np.fft.fft(y * w, axis=1)) ** 2
        total = p if total is None else total + p
    med = np.median(total, axis=1)
    return ((np.arange(total.shape[0]) + 0.5) * window_s, np.argmax(total, axis=1),
            10 * np.log10(total.max(axis=1) / np.maximum(med, 1e-30)), n)


def steady_runs(t, bins, snr_db, n, min_snr_db=6.0, min_len=4, tol=1):
    """[(t_start, t_end, bin)]: stretches where the strongest beat bin stays put (noisy windows skipped)."""
    out, cur = [], None
    for ti, bi, si in zip(t, bins, snr_db):
        if si < min_snr_db:
            continue
        near = (lambda a, b: min(abs(int(a) - int(b)), n - abs(int(a) - int(b))) <= tol)
        if cur and near(bi, cur[2]):
            cur[1], cur[3] = ti, cur[3] + 1
            continue
        if cur and cur[3] < min_len and out and near(bi, out[-1][2]):
            cur = [out[-1][0], ti, out[-1][2], min_len]      # a short glitch inside the previous run
            out.pop()
            continue
        if cur and cur[3] >= min_len:
            out.append(tuple(cur[:3]))
        cur = [ti, ti, int(bi), 1]
    if cur and cur[3] >= min_len:
        out.append(tuple(cur[:3]))
    return out


def reset_times(runs, n, window_s, jump_bins=3):
    """Start times of the runs whose beat differs from the previous run: the transmitter's resets."""
    return np.array([runs[i][0] - window_s / 2 for i in range(1, len(runs))
                     if min(abs(runs[i][2] - runs[i - 1][2]), n - abs(runs[i][2] - runs[i - 1][2])) > jump_bins])


def period_coherence(resets, periods):
    """|mean exp(2 pi i t / T)| over the reset times, for each trial period T (1 = all on one grid)."""
    resets = np.asarray(resets, float)
    if not resets.size:
        return np.zeros(len(periods))
    return np.abs(np.exp(2j * np.pi * resets[None, :] / np.asarray(periods, float)[:, None]).mean(axis=1))


def measure_sweep_rate(iq, fs, chirp_rate, window_s=0.025, rate_min=0.2, rate_max=5.0, min_coherence=0.7,
                       min_resets=5):
    """Sweep rate from the reset times of the echo of one chirp rate.

    Returns dict(rate_hz, rate_err_hz, coherence, resets, rates, coherence_curve, bandwidth_hz) with
    rate_hz None when it cannot be measured. The resets also line up at multiples of the rate
    (every second reset is still on the grid), never at fractions of it, so the rate is the lowest
    one whose coherence is within 80 % of the best; it is then refined and given an error from a
    straight-line fit of reset time against sweep number.
    """
    t, b, snr, n = beat_track(iq, fs, chirp_rate, window_s)
    resets = reset_times(steady_runs(t, b, snr, n), n, window_s)
    rates = np.exp(np.linspace(np.log(rate_min), np.log(rate_max), 3000))
    curve = period_coherence(resets, 1.0 / rates)
    out = dict(rate_hz=None, rate_err_hz=None, coherence=float(curve.max()) if curve.size else 0.0,
               resets=resets, rates=rates, coherence_curve=curve, bandwidth_hz=None)
    if resets.size < min_resets or curve.max() < min_coherence:
        return out
    r0 = rates[curve >= 0.8 * curve.max()].min()
    fine = np.linspace(0.97 * r0, 1.03 * r0, 3001)
    r1 = fine[int(np.argmax(period_coherence(resets, 1.0 / fine)))]
    # sweep number of each reset, then a straight-line fit (resets off the grid by > 1/4 sweep are dropped)
    k = np.round(resets * r1)
    resid = resets - k / r1
    keep = np.abs(resid - np.median(resid)) < 0.25 / r1
    if keep.sum() < min_resets or np.ptp(k[keep]) == 0:
        return out
    A = np.column_stack([k[keep], np.ones(keep.sum())])
    (period, start) = np.linalg.lstsq(A, resets[keep], rcond=None)[0]
    dof = max(keep.sum() - 2, 1)
    sigma = np.sqrt(np.sum((resets[keep] - A @ [period, start]) ** 2) / dof)
    sigma = max(sigma, window_s / np.sqrt(12))                    # reset times are quantised to the windows
    err_period = sigma / np.sqrt(np.sum((k[keep] - k[keep].mean()) ** 2))
    rate = 1.0 / period
    out.update(rate_hz=float(rate), rate_err_hz=float(err_period / period ** 2), coherence=float(
        period_coherence(resets, [period])[0]), bandwidth_hz=float(chirp_rate / rate))
    return out


RARE_SWEEP_FREQUENCIES_HZ = (2.007735,)     # used by a few OSU stations only (12.2 MHz)


def snap_sweep_rate(rate, err, known=CODAR_SWEEP_FREQUENCIES_HZ, n_sigma=3.0, rare=RARE_SWEEP_FREQUENCIES_HZ):
    """The known CODAR sweep rate a measured rate is consistent with (within n_sigma and 2 %), else None.

    When several are consistent the nearest common one is taken: a ``rare`` rate (2.007735 Hz,
    0.4 % from 2 Hz) only when no other rate fits."""
    if rate is None:
        return None
    known = np.asarray(known, float)
    d = np.abs(known - rate)
    ok = (d <= max(n_sigma * (err or 0.0), 1e-9)) & (d <= 0.02 * rate)
    if not ok.any():
        return None
    common = ok & ~np.isin(np.round(known, 6), np.round(np.asarray(rare, float), 6))
    pool = np.flatnonzero(common if common.any() else ok)
    return float(known[pool[np.argmin(d[pool])]])


# ---------------------------------------------------------------------------
# Gating sidebands: detection and CLEAN removal
# ---------------------------------------------------------------------------
# Port of detect_comb and sideband_clean (adaptive) in Peter Driessen's
# lwa_radar.py, on the thesis array layout (lags, pulses, channels) and with the
# bin widths in ms, so they do not depend on the sample rate. Both need the
# matched-filter output on the whole lag axis (bin = lag of 1/fs).

SIDEBAND_LAGS_MS = (1.5, 40.0)    # comb spacings searched
SIDEBAND_ORDERS = 8               # orders measured on each side
SIDEBAND_MIN_DB = 6.0             # weaker of the first-order pair, dB above the profile median
SIDEBAND_LOCAL_MS = 0.02          # each order's peak is taken within this distance of n * spacing
SIDEBAND_WIDTH_MS = 0.04          # bins removed on each side of an order's peak
SIDEBAND_SEARCH_MS = 0.5          # the main echo is looked for this far from its preview delay
SIDEBAND_MIN_COHERENCE_DB = -30.0  # orders whose ratio to the main echo is weaker are left alone


def _ms_bins(ms, fs, minimum=1):
    return max(int(minimum), int(round(float(ms) * 1e-3 * float(fs))))


def detect_gating_comb(Y, sample_rate, peak=None, lag_ms=SIDEBAND_LAGS_MS, n_orders=SIDEBAND_ORDERS,
                       min_db=SIDEBAND_MIN_DB, local_ms=SIDEBAND_LOCAL_MS):
    """Find the comb of gating sidebands around one echo (lwa_radar.detect_comb).

    ``Y``: matched-filter output on the whole lag axis, (lags, pulses) or (lags, pulses, channels).
    ``peak``: lag bin of the echo (default: the strongest bin of the channel-summed mean power).
    The spacing is the lag L in ``lag_ms`` that maximises the weaker of the two levels at peak +/- L
    (a gated transmitter puts copies in symmetric pairs). Each order n is then measured at
    peak + n * spacing (+/- ``local_ms``): its level in dB relative to the echo and its complex
    ratio to the echo, sum Y[b] conj(Y[peak]) / sum |Y[peak]|^2 over sweeps and channels
    (the copies share the echo's path, so the ratio is the same on every channel).
    Returns a dict with ``detected`` and, when detected, peak_bin, peak_ms, spacing_bins,
    spacing_ms, pair_db (first-order pair, dB above the median) and ``orders`` (n, offset_bins,
    offset_ms, level_db, coherence_db, ratio_re, ratio_im).
    """
    from scipy.ndimage import maximum_filter1d
    Y = np.asarray(Y)
    if Y.ndim == 2:
        Y = Y[..., None]
    n = Y.shape[0]
    Yz = np.where(np.isfinite(Y), Y, 0)
    power = np.mean(np.sum(np.abs(Yz) ** 2, axis=2), axis=1)
    med = float(np.median(power))
    if not med > 0:
        return dict(detected=False, reason="no signal")
    pdb = 10 * np.log10(power / med + 1e-12)
    pk = int(np.argmax(power)) if peak is None else int(peak) % n
    h = _ms_bins(local_ms, sample_rate, 0)
    smooth = maximum_filter1d(pdb, 2 * h + 1, mode="wrap")

    def local(c, half=h):
        idx = (int(round(c)) + np.arange(-half, half + 1)) % n
        j = int(np.argmax(pdb[idx]))
        return float(pdb[idx[j]]), int(idx[j])

    def fine(b):
        """Sub-bin position of the peak at bin b (parabola through it and its neighbours, in dB)."""
        y0, y1, y2 = pdb[(b - 1) % n], pdb[b % n], pdb[(b + 1) % n]
        den = y0 - 2 * y1 + y2
        return b + (float(np.clip(0.5 * (y0 - y2) / den, -0.5, 0.5)) if den < 0 else 0.0)

    lags = np.arange(_ms_bins(lag_ms[0], sample_rate), min(_ms_bins(lag_ms[1], sample_rate), n // 2 - h - 1) + 1)
    base = dict(peak_bin=pk, peak_ms=pk / sample_rate * 1e3)
    if not lags.size:
        return dict(base, detected=False, reason="sweep too short for the spacing range")
    pair = np.minimum(smooth[(pk + lags) % n], smooth[(pk - lags) % n])
    best = int(np.argmax(pair))
    if pair[best] < min_db:
        return dict(base, detected=False, reason=f"no symmetric pair {min_db:g} dB above the median",
                    pair_db=float(pair[best]))
    L = int(lags[best])
    # spacing to a fraction of a bin (lwa_radar.detect_comb keeps whole bins): order n is predicted
    # at n x spacing, so a whole-bin error grows to several bins by the higher orders
    plus = (fine(local(pk + L)[1]) - fine(pk)) % n
    minus = (fine(pk) - fine(local(pk - L)[1])) % n
    spacing = 0.5 * (plus + minus)
    ref = Yz[pk]
    e0 = float(np.sum(np.abs(ref) ** 2))
    orders, seen = [], set()
    for k in range(-int(n_orders), int(n_orders) + 1):
        if k == 0 or abs(k) * spacing > n / 2 - h - 1:
            continue
        level, b = local(fine(pk) + k * spacing, max(h, 1))
        off = int((b - pk + n // 2) % n - n // 2)
        if off in seen:
            continue
        seen.add(off)
        ratio = complex(np.sum(Yz[b] * np.conj(ref)) / max(e0, 1e-30))
        orders.append(dict(n=k, offset_bins=off, offset_ms=off / sample_rate * 1e3, level_db=level - float(pdb[pk]),
                           coherence_db=float(20 * np.log10(abs(ratio) + 1e-12)), ratio_re=ratio.real,
                           ratio_im=ratio.imag))
    return dict(base, detected=True, spacing_bins=float(spacing), spacing_ms=float(spacing / sample_rate * 1e3),
                pair_db=float(pair[best]), orders=orders)


def comb_offsets(comb, sample_rate, width_ms=SIDEBAND_WIDTH_MS, min_coherence_db=SIDEBAND_MIN_COHERENCE_DB):
    """Lag offsets (bins, relative to the main echo) that ``clean_gating_comb`` removes."""
    w = _ms_bins(width_ms, sample_rate, 0)
    offs = {o["offset_bins"] + j for o in comb.get("orders", []) if o["coherence_db"] >= min_coherence_db
            for j in range(-w, w + 1)}
    return np.array(sorted(x for x in offs if abs(x) > w), int)


def clean_gating_comb(Y, comb, sample_rate, n_lags=None, lags=None, block=20, search_ms=SIDEBAND_SEARCH_MS,
                      width_ms=SIDEBAND_WIDTH_MS, min_coherence_db=SIDEBAND_MIN_COHERENCE_DB):
    """Subtract the gating sidebands of one echo, block by block of sweeps (adaptive lwa_radar.sideband_clean).

    ``Y``: matched-filter output (rows, pulses[, channels]); row i holds lag ``lags[i]`` of an
    ``n_lags``-bin lag axis (default: all lags, in order). In each block of ``block`` sweeps the
    echo is found within +/- ``search_ms`` of ``comb['peak_bin']`` (channel-summed power), the
    complex ratio of every sideband bin to it is measured over the block, per channel,
    k = sum_p Y[b] conj(m) / sum_p |m|^2, and m * k is subtracted. That removes what moves
    exactly with the echo, sweep by sweep: its copies. Another echo at a sideband delay with a
    different Doppler decorrelates from it over the block and stays; one with the same Doppler
    would be partly removed. An echo exactly n spacings away also loses the fraction r(n) r(-n)
    of itself (r = order ratios), because its own -n copy sits on the main echo: about -0.9 dB
    when the orders are 10 dB down. Sideband bins that are not among the rows are skipped.
    Returns (cleaned copy, info dict).
    """
    Y = np.array(Y, dtype=np.complex64, copy=True)
    squeeze = Y.ndim == 2
    if squeeze:
        Y = Y[..., None]
    rows_n, n_pulses = Y.shape[:2]
    n = rows_n if n_lags is None else int(n_lags)
    lags = np.arange(rows_n) if lags is None else np.asarray(lags, int) % n
    pos = np.full(n, -1, int)
    pos[lags] = np.arange(rows_n)
    offs = comb_offsets(comb, sample_rate, width_ms, min_coherence_db)
    s = _ms_bins(search_ms, sample_rate, 0)
    region = (int(comb["peak_bin"]) + np.arange(-s, s + 1)) % n
    region = region[pos[region] >= 0]
    if not region.size:
        raise ValueError("The rows do not hold the echo whose sidebands are to be removed")
    Yz = np.where(np.isfinite(Y), Y, 0)
    blocks = removed = 0
    for s0 in range(0, n_pulses, int(block)):
        sl = slice(s0, min(s0 + int(block), n_pulses))
        blk = Yz[:, sl]
        pb = int(region[np.argmax(np.sum(np.abs(blk[pos[region]]) ** 2, axis=(1, 2)))])
        rows = np.unique(pos[(pb + offs) % n])
        rows = rows[rows >= 0]
        if not rows.size:
            continue
        m = blk[pos[pb]]                                            # (pulses, channels)
        em = np.sum(np.abs(m) ** 2, axis=0)
        k = np.einsum("rpc,pc->rc", blk[rows], np.conj(m)) / np.maximum(em, 1e-30)
        k[:, em <= 0] = 0
        Y[rows, sl] -= (m[None] * k[:, None, :]).astype(np.complex64)
        blocks += 1
        removed = max(removed, rows.size)
    info = dict(blocks=blocks, sideband_bins=int(removed), block_sweeps=int(block), search_ms=float(search_ms),
                width_ms=float(width_ms))
    return (Y[..., 0] if squeeze else Y), info


def comb_summary(comb, block=None):
    """JSON-ready record of a detected (or not) comb for file metadata."""
    out = dict(method="clean", detected=bool(comb.get("detected")))
    for key in ("peak_ms", "spacing_ms", "pair_db", "reason"):
        if key in comb:
            out[key if key != "peak_ms" else "main_ms"] = comb[key] if isinstance(comb[key], str) else float(comb[key])
    if comb.get("detected"):
        out["orders"] = [dict(n=o["n"], offset_ms=round(o["offset_ms"], 4), level_db=round(o["level_db"], 2),
                              coherence_db=round(o["coherence_db"], 2)) for o in comb["orders"]]
        out.update(block_sweeps=block, search_ms=SIDEBAND_SEARCH_MS, width_ms=SIDEBAND_WIDTH_MS,
                   min_coherence_db=SIDEBAND_MIN_COHERENCE_DB)
    return out


def comb_text(comb):
    if not comb.get("detected"):
        return f"no gating comb found ({comb.get('reason', 'not detected')})"
    first = " ".join(f"{o['n']:+d}:{o['level_db']:+.1f}" for o in comb["orders"] if abs(o["n"]) <= 3)
    return (f"gating comb around the {comb['peak_ms']:.3f} ms echo: spacing {comb['spacing_ms']:.3f} ms, "
            f"order levels re echo [dB] {first}")


# ---------------------------------------------------------------------------
# Copies of an echo
# ---------------------------------------------------------------------------
# Two kinds of weaker copies sit at fixed delay offsets from every strong echo:
# * gating sidebands: CODAR gates its transmitter, which puts copies at
#   +/- (gate frequency / |chirp rate|) and multiples, in symmetric pairs
#   (13.5 MHz on 2023-12-09: +/-5.18 ms; 25.6 MHz: +/-6.82 ms);
# * the sweep-wrap replica (dechirp range method only): an echo far into the sweep
#   is cut in two by the sweep boundary, and with |B| != fs the second piece lands
#   at +(fs - |B|) / |chirp rate| (3.73 ms for -99.26 kHz at 2 Hz). The matched
#   filter has no such copy.
# Real multipath has no reason to sit symmetrically around a stronger echo or at
# the wrap offset, so peaks that do are labelled copies.

def classify_copies(peaks, period_ms, sample_rate=None, bandwidth=None, sweep_rate=None, tol_ms=0.1,
                    min_offset_ms=0.3, min_drop_db=1.0, doppler_tol_hz=0.15, range_method="dechirp"):
    """Split peaks (dicts with delay_ms, db, optional doppler_hz) into echoes and copies.

    Returns (echoes, copies), strongest first. Each copy gets ``copy_of`` (the index of its echo in
    ``echoes``), ``kind`` ("gating" or "sweep wrap") and ``offset_ms``; each echo gets ``copies``
    (a list of those copies). ``range_method``: the method the peaks were found with; only
    "dechirp" data has sweep-wrap copies (the default keeps the behaviour for older results)."""
    def off(a, b):
        d = (a - b) % period_ms
        return d - period_ms if d > period_ms / 2 else d

    wrap = None
    if sample_rate and bandwidth and sweep_rate and range_method == "dechirp":
        w = (float(sample_rate) - abs(float(bandwidth))) / abs(float(bandwidth) * float(sweep_rate)) * 1e3
        wrap = w if abs(w) >= min_offset_ms else None
    ps = sorted((dict(p) for p in peaks), key=lambda p: -p["db"])
    echoes, copies = [], []
    for p in ps:
        p.setdefault("copies", [])
        verdict = None
        for i, m in enumerate(echoes):
            if m["db"] - p["db"] < min_drop_db:
                continue
            if p.get("doppler_hz") is not None and m.get("doppler_hz") is not None and \
                    abs(p["doppler_hz"] - m["doppler_hz"]) > doppler_tol_hz:
                continue
            o = off(p["delay_ms"], m["delay_ms"])
            if abs(o) < min_offset_ms:
                continue
            if wrap is not None and abs(abs(o) - abs(wrap)) <= tol_ms:
                verdict = (i, "sweep wrap", o)
                break
            spacing = m.get("gating_ms")
            if spacing and any(abs(abs(o) - n * spacing) <= tol_ms * n for n in (1, 2, 3)):
                verdict = (i, "gating", o)
                break
            twin = next((q for q in ps if q is not p and m["db"] - q["db"] >= min_drop_db
                         and abs(off(q["delay_ms"], m["delay_ms"]) + o) <= tol_ms
                         and (q.get("doppler_hz") is None or p.get("doppler_hz") is None
                              or abs(q["doppler_hz"] - p["doppler_hz"]) <= doppler_tol_hz)), None)
            if twin is not None:
                m["gating_ms"] = abs(o)
                verdict = (i, "gating", o)
                break
        if verdict is None:
            echoes.append(p)
        else:
            i, kind, o = verdict
            p.update(copy_of=i, kind=kind, offset_ms=o)
            copies.append(p)
            echoes[i]["copies"].append(p)
    return echoes, copies
