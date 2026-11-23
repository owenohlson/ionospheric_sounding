"""Find the chirp waveforms in a recording: bandwidth, sweep rate and echo delays of every CODAR present.

Station-table waveforms come first; a waveform is worked out blind only when
nothing in the table fits. Steps:

1. Blind chirp rates. The lag-product spectrum of x(t) x*(t - lag) turns every
   chirp into a tone at (chirp rate x lag), whatever its delay, sweep start or
   Doppler, so its peaks give the chirp rates k = bandwidth x sweep rate present
   (median over several lags, so the sidelines of gated transmitters fade).
2. Sweep rate from the resets. For each peak the data are dechirped at k and the
   beat followed in 25 ms windows; it jumps when the transmitter resets, and the
   reset times give the sweep period (lfm_utils.measure_sweep_rate).
3. Match to the station table. The HFRnet stations active on the date whose sweep
   overlaps the band (built-in table or --codar-stations), and any --candidates,
   are matched to each peak by chirp rate. A match is used exactly as documented
   (bandwidth and sweep rate); when waveforms of different sweep rates share the
   chirp rate, the measured rate picks between them (or, if the resets cannot be
   measured, the dechirped test below). Waveforms listed with the opposite
   sign match too (the HFRnet lists disagree about some stations' sign, e.g.
   FOCF and VATK at 26.31 MHz): alone they give a "table, sign flipped" match;
   next to a same-sign match, which sets the waveform, they are listed among its
   sources as "(sign flipped)". Documented waveforms that match no peak are
   tested too, in case the lag spectrum missed a weak transmitter.
4. Peaks with no match ("not in the station table"): the measured sweep rate
   (snapped to a CODAR rate; the rare 2.007735 Hz only when 2 Hz does not fit),
   or if it cannot be measured, the dechirped test; then a narrow scan of the
   bandwidth on whole sweeps (+/- --fine-span), since the lag peak fixes it only
   to some tens of Hz and whole sweeps need it to ~1 Hz.
5. Every waveform is dechirped on whole sweeps. It is kept when the delay profile
   peaks (contrast >= --min-contrast-db) and the peak needs that bandwidth
   (contrast drops >= 3 dB at +/- --fine-span / 2), which rejects profiles whose
   floor alone gives a high contrast. Table waveforms found are bandwidth-scanned
   too, as a check drawn in the plot (their documented bandwidth is kept).
   Gating and sweep-wrap copies of an echo are listed, not counted as echoes.
6. Gating sidebands. The comb of copies around each waveform's strongest echo is
   found and subtracted from the matched-filter sweeps before the delay profile is
   drawn (lfm_utils.compression_profile_clean, the CLEAN of the RDC1 export); copies
   still in the profile after that are marked. --keep-sidebands skips this.

Dechirped test (printed as a cross-check for every waveform): when the sweep
is wider than the receive band, the in-band part of a chirp also compresses at
other sweep rates with the bandwidth scaled to keep k; on 25 ms pieces tiling a
sweep, the true rate puts every piece's echo in one delay bin.

Delays are mod one sweep, relative to sweeps that start --offset after each
UTC second (like the rest of the pipeline; without a UTC time, at the first
sample). They are not physical ranges: each transmitter has its own sweep
timing.

Input: a raw TBN/WAV recording (no --sweep-frequency needed) or an RDC0 with
complete sweeps. More stands give more SNR: e.g. --stands 1,2,20,60,100,150,200,250.
"""

import json
import os
from dataclasses import dataclass, field
from datetime import timezone

import numpy as np

import cli
import lfm_utils as lfm
import plotting_utils as plots
import summary_utils


def parse_candidates(text, default_rate=2.0):
    """'B[:RATE],...' (Hz, signed bandwidth) -> [(bandwidth, sweep_rate)]."""
    out = []
    for item in (text or "").split(","):
        item = item.strip()
        if item:
            b, _, rate = item.partition(":")
            out.append((float(b), float(rate) if rate else default_rate))
    return out


def table_candidates(fc, fs, when, tables=None):
    """{(bandwidth, sweep_rate): [codes]} for HFRnet stations whose sweep overlaps the receive band."""
    import codar_stations as cs
    stations = cs.load_tables(tables or None)
    rx = cs.Receiver(0.0, 0.0, float(fc), float(fs), 1.0, 1.0, when)
    out = {}
    for c in stations:
        if cs.active(c, when) and cs.band_overlap(c, rx) >= 0.25:
            out.setdefault((round(c.bandwidth_hz, 3), round(c.sweep_rate_hz, 6)), set()).add(c.code)
    return {k: sorted(v) for k, v in out.items()}


class Candidate:
    """One waveform tried: as documented (station table / --candidates) or worked out from a lag peak."""

    def __init__(self, bandwidth, rate, sources, kind, lag_k=None):
        self.bandwidth, self.rate, self.sources = float(bandwidth), float(rate), list(sources)
        self.kind = kind            # "table" (matched a lag peak), "table, sign flipped" (matched it with the
                                    # opposite sign), "table, no lag peak", or "not in table"
        self.lag_k = lag_k          # chirp rate of the lag peak it came from [Hz/s]
        self.fine = None            # bandwidth scan: (offsets from self.bandwidth [Hz], contrast [dB])
        self.scan_peak_hz = None    # documented waveforms: where the scan peaks, relative to the table bandwidth
        self.best_bandwidth = self.best_db = self.prominence_db = None
        self.delay_ms = self.profile = None
        self.echoes = []            # [(delay ms, SNR dB)], copies excluded (lfm.local_snr_db)
        self.copies = []            # [dict(delay_ms, db, kind, offset_ms, of_delay_ms)]: gating / sweep-wrap copies
        self.skip = self.shift_ms = self.aligned_db = None
        self.rate_test = []         # [dict(rate, bandwidth, snr_db)]: the dechirped sweep-rate test
        self.rate_measurement = None  # lfm.measure_sweep_rate result for this chirp rate
        self.rate_basis = ""
        self.reject = ""
        self.comb = None            # lfm.detect_gating_comb result when the sideband CLEAN ran
        self.raw_profile = None     # delay profile before the CLEAN (same as profile when nothing was removed)

    @property
    def documented(self):
        return self.kind != "not in table"

    @property
    def chirp_rate(self):
        return (self.best_bandwidth if self.best_bandwidth is not None else self.bandwidth) * self.rate

    def label(self):
        b = self.best_bandwidth if self.best_bandwidth is not None else self.bandwidth
        return f"{b / 1e3:+.3f} kHz @ {self.rate:g} Hz"


@dataclass
class ScanOptions:
    offset: float = 0.0                       # sweep start after each UTC second [s]
    sweep_rates: tuple = tuple(float(r) for r in lfm.CODAR_SWEEP_FREQUENCIES_HZ)
    candidates: list = field(default_factory=list)      # [(bandwidth, rate)]
    use_table: bool = True
    tables: list = None
    blind: bool = True
    lags: tuple = lfm.LAGS_S
    lag_peaks: int = 4
    lag_min_db: float = 8.0      # dB over the median of the lag-median spectrum (its noise is tighter than one lag)
    match_tolerance: float = 0.003            # relative chirp-rate tolerance of a table match (or 3 lag bins)
    min_contrast_db: float = 10.0
    min_prominence_db: float = 3.0            # contrast drop at +/- fine_span / 2 from the bandwidth used
    fine_span: float = 100.0                  # bandwidth scan of waveforms not in the table [Hz]
    fine_step: float = 2.0
    fine_seconds: float = 10.0
    suppress_sidebands: bool = True           # CLEAN the gating comb out of the delay profiles
    sideband_block: int = 20                  # sweeps per CLEAN block
    verbose: bool = True


@dataclass
class ScanResult:
    found: list                 # waveforms present (Candidate), strongest first
    rejected: list              # tried and did not compress
    candidates: list            # everything tried
    lag_rate: np.ndarray = None     # lag-product spectrum (chirp rate axis, power)
    lag_power: np.ndarray = None
    lag_peaks: list = field(default_factory=list)
    rate_measurements: list = field(default_factory=list)   # lfm.measure_sweep_rate result per lag peak
    documented: dict = field(default_factory=dict)           # {(bandwidth, rate): [sources]} in the band


def scan(iq, fs, fc=None, utc0=None, opts=None):
    """Run the waveform scan on (streams, samples) ``iq``. See the module docstring."""
    o = opts or ScanOptions()
    say = print if o.verbose else (lambda *a, **k: None)
    iq = np.atleast_2d(iq)
    stamp = None if utc0 is None else utc0.replace(tzinfo=utc0.tzinfo or timezone.utc).timestamp()

    def skip(rate):
        return lfm.sweep_alignment_samples(stamp, rate, fs, o.offset)

    def fits(rate):
        return iq.shape[1] >= 3 * fs / rate            # a whole sweep after aligning to the UTC grid

    # 1. blind chirp rates
    lag_k = lag_p = None
    peaks, res = [], lfm.lag_resolution(fs, o.lags)
    if o.blind:
        lag_k, lag_p = lfm.lag_product_spectrum(iq, fs, o.lags)
        peaks = lfm.spectrum_peaks(lag_k, lag_p, o.lag_peaks, min_db=o.lag_min_db,
                                   exclude_below=max(5 * res, 2e3), min_separation=max(5 * res, 3e3))
        say("Lag-product chirp rates: " + (", ".join(f"{k / 1e3:+.2f} kHz/s ({db:.1f} dB)" for k, db in peaks)
                                           or f"none above {o.lag_min_db:g} dB"), flush=True)

    # 2. the sweep rate of a chirp rate, measured from the transmitter's resets (cached)
    measured = []

    def measurement(k, quiet=False):
        for kk, m in measured:
            if abs(kk - k) <= 0.005 * abs(kk):
                return m
        m = lfm.measure_sweep_rate(iq, fs, k)
        m["snapped_hz"] = lfm.snap_sweep_rate(m["rate_hz"], m["rate_err_hz"], o.sweep_rates)
        m["chirp_rate"] = k
        measured.append((k, m))
        if not quiet:
            say(f"  {k / 1e3:+.2f} kHz/s: " + (
                f"sweep rate not measurable from resets ({len(m['resets'])} resets, coherence {m['coherence']:.2f})"
                if m["rate_hz"] is None else
                f"sweep rate {m['rate_hz']:.4f} +/- {m['rate_err_hz']:.4f} Hz from {len(m['resets'])} resets"),
                flush=True)
        return m

    lag_meas = [measurement(k) for k, _ in peaks]

    # documented waveforms in the band: station table and --candidates
    documented = {}
    for b, rate in o.candidates:
        documented.setdefault((float(b), float(rate)), []).append("user")
    if o.use_table:
        if fc:
            when = utc0.date() if utc0 else None
            for key, codes in table_candidates(fc, fs, when, o.tables).items():
                documented.setdefault(key, []).append("table " + "/".join(codes[:6]) + ("/..." if len(codes) > 6 else ""))
        else:
            say("No tuning frequency known: no station-table waveforms", flush=True)

    def tol(k):
        return max(3 * res, o.match_tolerance * abs(k))

    def tiled_snr(b, rate):
        """Echo SNR on 25 ms pieces tiling a sweep, with the sweep grid moved so the echo starts it."""
        if not fits(rate):
            return float("nan"), None, None
        win = lfm.scan_windows(fs, rate)
        d, pw = lfm.compression_profile(iq, fs, b, rate, skip(rate), win)
        shift = (d[int(np.argmax(pw))] - 2.0) % (1e3 / rate)
        sk = skip(rate) + int(round(shift * 1e-3 * fs))
        return lfm.peak_snr_db(lfm.compression_profile(iq, fs, b, rate, sk, win)[1]), sk, shift

    def choose_rate(k, rates, bandwidth_of, m):
        """Sweep rate among ``rates`` for chirp rate k: the measured one, else the dechirped test."""
        if len(rates) == 1:
            return rates[0], None
        snapped = lfm.snap_sweep_rate(m["rate_hz"], m["rate_err_hz"], rates) if m["rate_hz"] else None
        if snapped:
            return snapped, f"measured from {len(m['resets'])} resets: {m['rate_hz']:.4f} +/- {m['rate_err_hz']:.4f} Hz"
        test = {r: tiled_snr(bandwidth_of(r), r)[0] for r in rates}
        test = {r: v for r, v in test.items() if np.isfinite(v)}
        if not test:
            return None, None
        best = max(test, key=test.get)
        common = [r for r in test if round(r, 6) not in np.round(lfm.RARE_SWEEP_FREQUENCIES_HZ, 6)]
        if common and best not in common and test[best] - max(test[r] for r in common) < 1.0:
            best = max(common, key=test.get)              # a rare rate needs a clear margin
        return best, "dechirped test (resets not measurable): " + ", ".join(
            f"{r:g} Hz {v:.1f} dB" for r, v in sorted(test.items()))

    flipped = {}                                     # (bandwidth, rate) with the table's sign reversed: sources

    def from_documented(k, members, lag, flip=False):
        """The documented waveform for chirp rate k: (bandwidth, rate) pairs whose k matches."""
        m = measurement(k, quiet=lag is None)
        rates = sorted({r for _, r in members})
        rate, how = choose_rate(k, rates, lambda r: next(b for b, rr in members if rr == r), m)
        if rate is None:
            return None
        bs = [b for b, r in members if r == rate]
        if len(bs) > 1 and fits(rate):                   # same rate, nearby bandwidths: the one compressing best
            bs.sort(key=lambda b: -lfm.contrast_db(lfm.compression_profile(
                iq, fs, b, rate, skip(rate), max_sweeps=max(2, int(round(o.fine_seconds * rate))))[1]))
        b = bs[0]
        sources = [x for bb, r in members if r == rate and abs(bb - b) < 1e-6
                   for x in (flipped[(bb, r)] if flip else documented[(bb, r)])]
        kind = "table, no lag peak" if lag is None else ("table, sign flipped" if flip else "table")
        c = Candidate(b, rate, list(dict.fromkeys(sources)), kind, lag)
        c.rate_measurement = m
        if how:
            c.rate_basis = how
        else:
            c.rate_basis = "documented"
            if m["rate_hz"] is not None:
                ok = abs(m["rate_hz"] - rate) <= max(3 * m["rate_err_hz"], 0.005 * rate)
                c.rate_basis += (f"; resets give {m['rate_hz']:.4f} +/- {m['rate_err_hz']:.4f} Hz" +
                                 ("" if ok else " (NOTE: does not agree)"))
        return c

    # 3. lag peaks matched to documented waveforms; 4. the rest worked out blind
    cands, used = [], set()
    for (k, db), m in zip(peaks, lag_meas):
        members = [key for key in documented if abs(key[0] * key[1] - k) <= tol(k)]
        # the table's sign may be wrong (the HFRnet lists disagree): waveforms listed with the opposite
        # sign count too, as a match of their own if nothing matches with the sign, else as extra sources
        opposite = [key for key in documented if abs(key[0] * key[1] + k) <= tol(k)]
        used.update(members + opposite)
        for b0, r0 in opposite:
            flipped[(-b0, r0)] = [f"{x} (sign flipped)" for x in documented[(b0, r0)]]
        flip = not members
        if flip:
            members = [(-b0, r0) for b0, r0 in opposite]
        if members:
            c = from_documented(k, members, k, flip)
            if c is None:
                continue
            if c.kind == "table":
                c.sources += [x for (b0, r0), xs in flipped.items() if r0 == c.rate and abs(b0 - c.bandwidth) <= 25.0
                              for x in xs if x not in c.sources]
        else:
            rate, how = (m["snapped_hz"], f"measured from {len(m['resets'])} resets: {m['rate_hz']:.4f} +/- "
                                          f"{m['rate_err_hz']:.4f} Hz") if m["snapped_hz"] else \
                choose_rate(k, [r for r in o.sweep_rates if 5e3 <= abs(k / r) <= 1.2e6], lambda r: k / r,
                            dict(rate_hz=None))
            if rate is None:
                continue
            if not m["snapped_hz"] and m["rate_hz"] is not None:
                how = f"resets give {m['rate_hz']:.4f} +/- {m['rate_err_hz']:.4f} Hz (no CODAR rate); " + how
            c = Candidate(k / rate, rate, [], "not in table", k)
            c.rate_measurement, c.rate_basis = m, how
        twin = next((x for x in cands if x.rate == c.rate and abs(x.bandwidth - c.bandwidth) <= 25.0), None)
        if twin:
            twin.sources += [s for s in c.sources if s not in twin.sources]
        else:
            if c.lag_k is not None:
                c.sources.append(f"lag {c.lag_k / 1e3:+.1f} kHz/s")
            cands.append(c)
    rest = [key for key in documented if key not in used]
    while rest:                                      # documented waveforms no lag peak matched, grouped by k
        k = rest[0][0] * rest[0][1]
        members = [key for key in rest if abs(key[0] * key[1] - k) <= tol(k)]
        rest = [key for key in rest if key not in members]
        c = from_documented(k, members, None)
        if c is not None:
            cands.append(c)
    if not cands:
        raise ValueError("No candidate waveforms: give --candidates, or allow the table/blind search")
    say(f"{len(cands)} waveform(s) to test: {sum(c.kind == 'table' for c in cands)} lag peak(s) matched to the "
        f"station table, {sum(c.kind == 'table, sign flipped' for c in cands)} matched with the table's sign "
        f"reversed, {sum(c.kind == 'not in table' for c in cands)} not in the table, "
        f"{sum(c.kind == 'table, no lag peak' for c in cands)} documented waveform(s) with no lag peak", flush=True)

    # 5. dechirp each on whole sweeps
    found = []
    for c in cands:
        if not fits(c.rate):
            c.reject = "sweeps longer than a third of the data"
            continue
        c.aligned_db, c.skip, c.shift_ms = tiled_snr(c.bandwidth, c.rate)
        n = max(2, int(round(o.fine_seconds * c.rate)))

        def profile(bandwidth, c=c, n=n):
            return lfm.compression_profile(iq, fs, bandwidth, c.rate, c.skip, max_sweeps=n)

        def bandwidth_scan(c=c, profile=profile):
            """Contrast over +/- fine_span around c.bandwidth, refined near its peak: (offsets, dB), peak offset."""
            outer = np.arange(-o.fine_span, o.fine_span + 1e-9, max(o.fine_step, o.fine_span / 10))
            db = [lfm.contrast_db(profile(c.bandwidth + x)[1]) for x in outer]
            inner = lfm.parabolic_peak(outer, db)[0] + np.arange(-6, 7) * o.fine_step
            db2 = [lfm.contrast_db(profile(c.bandwidth + x)[1]) for x in inner]
            x_all, db_all = np.r_[outer, inner], np.r_[db, db2]
            order = np.argsort(x_all)
            return (x_all[order], db_all[order]), lfm.parabolic_peak(inner, db2)[0]

        if c.kind == "not in table":                  # the bandwidth comes from the scan
            c.fine, peak = bandwidth_scan()
            c.best_bandwidth = c.bandwidth + peak
        else:
            c.best_bandwidth = c.bandwidth                # as documented
        d, c.profile = profile(c.best_bandwidth)
        c.best_db = lfm.contrast_db(c.profile)
        side = max(lfm.contrast_db(profile(c.best_bandwidth + x)[1]) for x in (-o.fine_span / 2, o.fine_span / 2))
        c.prominence_db = c.best_db - side
        if not c.best_db >= o.min_contrast_db:
            c.reject = f"contrast {c.best_db:.1f} dB < {o.min_contrast_db:g} dB"
        elif not c.prominence_db >= o.min_prominence_db:
            c.reject = (f"contrast {c.best_db:.1f} dB but only {c.prominence_db:.1f} dB above +/-{o.fine_span / 2:g} Hz "
                        "away (the profile's floor, not an echo)")
        say(f"  {c.label():24s} {c.kind:19s} contrast {c.best_db:5.1f} dB, {c.prominence_db:+5.1f} dB vs "
            f"+/-{o.fine_span / 2:g} Hz  [{'; '.join(c.sources)}]" + (f"  rejected: {c.reject}" if c.reject else ""),
            flush=True)
        if c.reject:
            continue
        if c.fine is None:                            # documented bandwidth kept; the scan is shown as a check
            c.fine, c.scan_peak_hz = bandwidth_scan()
        period = 1e3 / c.rate
        c.delay_ms = (d + c.shift_ms) % period                  # back on the UTC-aligned sweep grid
        c.raw_profile = c.profile
        if o.suppress_sidebands:
            _, cleaned, raw, c.comb = lfm.compression_profile_clean(iq, fs, c.best_bandwidth, c.rate, c.skip,
                                                                    max_sweeps=n, block=o.sideband_block)
            if c.comb.get("detected"):
                c.profile, c.raw_profile = cleaned, raw
                c.comb["peak_ms"] = (c.comb["peak_ms"] + c.shift_ms) % period      # on the scan's sweep grid
            say(f"  {c.label():24s} sideband CLEAN: {lfm.comb_text(c.comb)}", flush=True)
        order = np.argsort(c.delay_ms)
        c.delay_ms, c.profile, c.raw_profile = c.delay_ms[order], c.profile[order], c.raw_profile[order]
        dpeaks = lfm.spectrum_peaks(c.delay_ms, c.profile, 10, min_db=max(6.0, c.best_db - 25.0), min_separation=0.2)
        c.range_method = lfm.whole_sweep_method(fs, c.rate)
        marked = [dict(delay_ms=d, db=e) for d, e in dpeaks]
        if c.comb and c.comb.get("detected"):           # what is left of the removed comb is still a copy
            main = c.comb["peak_ms"]
            for m in marked:
                if min(abs(m["delay_ms"] - main), period - abs(m["delay_ms"] - main)) <= 0.1:
                    m["gating_ms"] = c.comb["spacing_ms"]
        echoes, copies = lfm.classify_copies(marked, period, fs, c.best_bandwidth, c.rate,
                                             range_method=c.range_method)
        busy = [d for d, _ in dpeaks]

        def snr(delay):
            i = int(np.argmin(np.minimum(np.abs(c.delay_ms - delay), period - np.abs(c.delay_ms - delay))))
            return lfm.local_snr_db(c.profile[i], c.profile, c.delay_ms, delay, busy, period_ms=period)[0]

        c.echoes = [(e["delay_ms"], snr(e["delay_ms"])) for e in echoes
                    if e["db"] >= max(6.0, c.best_db - 15.0)][:3]          # (delay, SNR)
        c.copies = [dict(delay_ms=x["delay_ms"], db=x["db"], kind=x["kind"], offset_ms=x["offset_ms"],
                         of_delay_ms=echoes[x["copy_of"]]["delay_ms"], rel_db=x["db"] - echoes[x["copy_of"]]["db"])
                    for x in copies]
        mark_suppressed(c, period)
        k = c.chirp_rate                                  # the dechirped test across sweep rates, as a cross-check
        for rate in sorted(set(float(r) for r in o.sweep_rates) | {c.rate}):
            snr = c.aligned_db if rate == c.rate else tiled_snr(k / rate, rate)[0]
            c.rate_test.append(dict(rate=rate, bandwidth=k / rate, snr_db=snr))
        found.append(c)
    found.sort(key=lambda c: (-c.best_db))
    # a much weaker result whose chirp rate is close to a strong one is that echo smeared by a
    # wrong waveform (partial compression), not another transmitter
    smeared = [c for i, c in enumerate(found) if any(
        s.best_db - c.best_db >= 10.0 and abs(c.chirp_rate - s.chirp_rate) <= 0.03 * abs(s.chirp_rate)
        for s in found[:i])]
    for c in smeared:
        c.reject = "a weak partial compression near a stronger waveform"
    found = [c for c in found if c not in smeared]
    if smeared:
        say(f"Dropped {len(smeared)} weak partial compression(s) near a stronger waveform: "
            + ", ".join(c.label() for c in smeared), flush=True)
    return ScanResult(found, [c for c in cands if c.reject], cands, lag_k, lag_p, peaks, lag_meas, documented)


def mark_suppressed(c, period):
    """Flag the copies the sideband CLEAN already subtracted from (``suppressed``): what is left of them
    is a residual. A copy is suppressed when it is a gating copy of the echo whose comb was removed,
    on an order the CLEAN measured and subtracted."""
    comb = c.comb or {}
    removed = {o["n"] for o in comb.get("orders", []) if o["coherence_db"] >= lfm.SIDEBAND_MIN_COHERENCE_DB}
    for x in c.copies:
        x["suppressed"] = False
        if not comb.get("detected") or x["kind"] != "gating":
            continue
        gap = abs(x["of_delay_ms"] - comb["peak_ms"]) % period
        if min(gap, period - gap) <= 0.1 and round(x["offset_ms"] / comb["spacing_ms"]) in removed:
            x["suppressed"] = True


def copy_text(x):
    return f"{x['kind']} {x['offset_ms']:+.2f} ms from {x['of_delay_ms']:.2f} ({x['db']:.1f} dB)"


def report_lines(r, min_contrast_db):
    lines = ["Waveforms present (dechirped on whole sweeps):"]
    if not r.found:
        lines.append("  none")
    for i, c in enumerate(r.found, 1):
        echoes = ", ".join(f"{d:.3f} ms (SNR {e:.1f} dB)" for d, e in c.echoes)
        src = "; ".join(s.replace("table ", "") for s in c.sources if not s.startswith("lag"))
        lines.append(f"  #{i} {c.best_bandwidth:+.1f} Hz at {c.rate:g} Hz sweeps [{c.kind}"
                     + (f": {src}" if src else "") + f"]: {c.best_db:.1f} dB contrast; echoes {echoes}")
        if c.lag_k is not None:
            lines.append(f"     lag peak {c.lag_k / 1e3:+.2f} kHz/s")
        if c.comb is not None:
            lines.append("     sideband CLEAN: " + lfm.comb_text(c.comb)
                         + (" (removed)" if c.comb.get("detected") else ""))
        left = [x for x in c.copies if not x.get("suppressed")]
        residual = [x for x in c.copies if x.get("suppressed")]
        if residual:
            lines.append("     suppressed copies (residual after the sideband CLEAN, not counted as echoes): "
                         + ", ".join(copy_text(x) for x in residual))
        if left:
            lines.append("     copies not removed (not counted as echoes): " + ", ".join(copy_text(x) for x in left))
        lines.append(f"     sweep rate {c.rate:g} Hz: {c.rate_basis}")
        tested = [t for t in c.rate_test if np.isfinite(t["snr_db"])]
        if tested:
            lines.append("     cross-check after dechirping (echo SNR on pieces tiling a sweep, same chirp rate): " +
                         ", ".join(f"{t['rate']:g} Hz {t['snr_db']:.1f} dB" + (" <" if t["rate"] == c.rate else "")
                                   for t in tested))
    if r.rejected:
        lines.append(f"{len(r.rejected)} waveform(s) tried did not compress (contrast below {min_contrast_db:g} dB, "
                     "a flat profile, or a partial compression)")
    return lines


def waveform_stations(c, fc, fs, when, tables=None):
    """HFRnet stations whose waveform is this one (same sweep rate, bandwidth within 25 Hz) in the band,
    then those listed with the opposite sign (the HFRnet lists disagree about some stations' sign)."""
    if not fc:
        return []
    same, opposite = set(), set()
    for (b, rate), names in table_candidates(fc, fs, when, tables).items():
        if abs(rate - c.rate) < 1e-6:
            if abs(b - c.best_bandwidth) <= 25.0:
                same.update(names)
            elif abs(b + c.best_bandwidth) <= 25.0:
                opposite.update(names)
    return sorted(same) + sorted(opposite - same)


def record_summary(args, r, view, utc0, fs):
    """One 'waveform' row per echo of every waveform found (see summary_utils)."""
    path = summary_utils.summary_path(args.output, args.summary)
    if not path:
        return
    when = utc0.date() if utc0 else None
    rows = []
    for c in r.found:
        codes = waveform_stations(c, view.fc, fs, when, args.codar_stations or None)
        period = 1e3 / c.rate
        for d, e in c.echoes or [(float("nan"), c.best_db)]:
            rows.append(dict(kind="waveform", utc_start=utc0.isoformat() if utc0 else "",
                             tuning_mhz=view.fc / 1e6 if view.fc else None, bandwidth_hz=c.best_bandwidth,
                             sweep_rate_hz=c.rate, snr_db=e,
                             snr_kind="scan: delay profile on whole sweeps, noise from +/-25 ms of the echo",
                             delay_ms=(d + args.offset * 1e3) % period if np.isfinite(d) else None,
                             sample_rate_hz=fs, range_method=getattr(c, "range_method", None),
                             candidate_stations=" ".join(codes),
                             copies="; ".join(f"{'suppressed ' if x.get('suppressed') else ''}{x['kind']} "
                                              f"{x['offset_ms']:+.2f} ms ({x['rel_db']:+.1f} dB)"
                                              for x in c.copies if abs(x["of_delay_ms"] - d) < 1e-6),
                             notes=f"{c.kind}; contrast {c.best_db:.1f} dB; sweep rate {c.rate_basis}"))
    summary_utils.record(path, "scan_lfm", args.input, args.output, rows)


def main():
    p = cli.parser(__doc__)
    cli.add_input(p)
    cli.add_plot_output(p)
    cli.add_selection(p, default_stands="first")
    d = ScanOptions()
    g = p.add_argument_group("scan")
    g.add_argument("--duration", type=float, default=20.0, help="Seconds of data used, from --tstart")
    g.add_argument("--sweep-rates", default=",".join(f"{r:g}" for r in d.sweep_rates),
                   help="Known CODAR sweep rates [Hz]: a measured rate is snapped to these, and the dechirped test "
                        "tries them")
    g.add_argument("--candidates", default=None, metavar="B[:RATE],...",
                   help="Extra documented waveforms: signed bandwidth [Hz], optional sweep rate (default 2 Hz)")
    g.add_argument("--codar-stations", nargs="*", default=[], metavar="CSV",
                   help="Station tables of documented waveforms (default: the built-in HFRnet table)")
    g.add_argument("--no-table", action="store_true", help="Do not use the station table")
    g.add_argument("--no-blind", action="store_true", help="Skip the lag-product spectrum (blind chirp rates)")
    g.add_argument("--lags", type=lambda v: tuple(float(x) * 1e-3 for x in v.split(",")), default=d.lags,
                   help="Lags of the lag-product spectrum [ms], comma separated; the median over lags is kept so the "
                        "sidelines of a gated transmitter (which move with the lag) fade")
    g.add_argument("--lag-peaks", type=int, default=d.lag_peaks, help="Chirp rates taken from the lag-product spectrum")
    g.add_argument("--match-tolerance", type=float, default=d.match_tolerance,
                   help="Relative chirp-rate tolerance when matching a lag peak to the station table")
    g.add_argument("--min-contrast-db", type=float, default=d.min_contrast_db,
                   help="Contrast [dB] a waveform's delay profile needs to count as present")
    g.add_argument("--fine-span", type=float, default=d.fine_span,
                   help="Bandwidth scan half-width [Hz] for waveforms not in the table")
    g.add_argument("--fine-step", type=float, default=d.fine_step, help="Bandwidth scan step [Hz]")
    g.add_argument("--fine-seconds", type=float, default=d.fine_seconds, help="Seconds of whole sweeps dechirped")
    g.add_argument("--keep-sidebands", action="store_true",
                   help="Do not remove the gating-sideband comb from the delay profiles (copies are only marked)")
    g.add_argument("--sideband-block", type=int, default=d.sideband_block, help="Sweeps per sideband CLEAN block")
    g.add_argument("--json", default=None, help="Also write the results as JSON")
    cli.add_summary(p)
    args = p.parse_args()
    cli.validate(p, args)
    if args.sweep_frequency is not None:
        p.error("--sweep-frequency is not used here; the scan tries --sweep-rates itself")

    if args.tend is None:            # cut only what the scan uses from a raw recording
        args.tend = (args.tstart or 0.0) + args.duration + 1.0
    view = cli.view_from_args(args, raw_stage="rdc0")
    iq, t0, gaps = view.raw_samples()
    fs = float(view.fs)
    iq = iq[:, :int(round(args.duration * fs))]
    if iq.shape[1] < 2 * fs:
        p.error("Less than 2 s of data; use a longer --duration or a different --tstart")
    utc0 = view.utc_at(t0)
    opts = ScanOptions(offset=args.offset, sweep_rates=tuple(float(r) for r in args.sweep_rates.split(",") if r.strip()),
                       candidates=parse_candidates(args.candidates), use_table=not args.no_table,
                       tables=args.codar_stations or None, blind=not args.no_blind, lags=args.lags,
                       lag_peaks=args.lag_peaks, match_tolerance=args.match_tolerance,
                       min_contrast_db=args.min_contrast_db, fine_span=args.fine_span, fine_step=args.fine_step,
                       fine_seconds=args.fine_seconds, suppress_sidebands=not args.keep_sidebands,
                       sideband_block=args.sideband_block)
    r = scan(iq, fs, view.fc, utc0, opts)
    print("\n" + "\n".join(report_lines(r, args.min_contrast_db)))

    rows = [("UTC start", utc0.strftime("%Y-%m-%d %H:%M:%S") if utc0 else "unknown")] + [
        x for x in view.info_rows(t0, t0 + iq.shape[1] / fs) if x[0] in ("File", "Tuning freq", "Channels", "Plot span")]
    notes = "\n".join(f"#{i}: {c.best_bandwidth / 1e3:+.3f} kHz, {c.rate:g} Hz, SNR "
                      f"{max([e for _, e in c.echoes] or [float('nan')]):.0f} dB" for i, c in enumerate(r.found, 1)) \
        or "Found: none"
    if r.found:
        notes += "\nSNR: echo power over the local noise (+/-25 ms of delay, echoes and copies left out)"
        if args.keep_sidebands:
            notes += "\nGating sidebands: kept (--keep-sidebands)"
        else:
            notes += "\nGating sidebands removed: " + "; ".join(
                f"#{i} every {c.comb['spacing_ms']:.2f} ms" if c.comb.get("detected") else f"#{i} none found"
                for i, c in enumerate(r.found, 1) if c.comb is not None)
    tables = [dict(label=f"{b / 1e3:+.3f} kHz @ {rate:g} Hz", k=b * rate) for b, rate in r.documented]
    plots.plot_lfm_scan(r.lag_rate, r.lag_power, r.lag_peaks, tables, r.found, args.title or "Chirp waveform scan",
                        args.output, rows, notes, cli.gap_footer(gaps, view.gap_fill), r.rate_measurements)
    print(f"Wrote {args.output}")
    record_summary(args, r, view, utc0, fs)
    if args.json:
        out = dict(input=os.path.abspath(args.input), utc_start=utc0.isoformat() if utc0 else None,
                   tuning_frequency_hz=view.fc, sample_rate_hz=fs, seconds=iq.shape[1] / fs,
                   lag_peaks=[dict(chirp_rate_hz_per_s=k, db=db) for k, db in r.lag_peaks],
                   waveforms=[dict(bandwidth_hz=c.best_bandwidth, sweep_rate_hz=c.rate, kind=c.kind,
                                   contrast_db=c.best_db, prominence_db=c.prominence_db,
                                   sweep_rate_snr_db=c.aligned_db, chirp_rate_hz_per_s=c.chirp_rate,
                                   lag_chirp_rate_hz_per_s=c.lag_k, bandwidth_scan_peak_offset_hz=c.scan_peak_hz,
                                   sweep_rate_test=c.rate_test, sweep_rate_basis=c.rate_basis,
                                   sweep_rate_measured_hz=(c.rate_measurement or {}).get("rate_hz"),
                                   sweep_rate_measured_err_hz=(c.rate_measurement or {}).get("rate_err_hz"),
                                   copies=c.copies, sideband_clean=None if c.comb is None else
                                   lfm.comb_summary(c.comb, opts.sideband_block),
                                   sources=c.sources, echoes=[dict(delay_ms=d, snr_db=e) for d, e in c.echoes])
                              for c in r.found],
                   rejected=[dict(bandwidth_hz=c.bandwidth, sweep_rate_hz=c.rate, kind=c.kind, contrast_db=c.best_db,
                                  reason=c.reject, sources=c.sources) for c in r.rejected])
        with open(os.path.expanduser(args.json), "w") as fh:
            json.dump(out, fh, indent=2, default=float)
        print(f"Wrote {args.json}")


if __name__ == "__main__":
    cli.run(main)
