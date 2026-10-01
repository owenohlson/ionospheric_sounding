"""Moving data between pipeline stages: raw samples -> RDC0 -> RDC1 -> RDC2 / RDC2P -> RDC3.

Sections
--------
1. Channel selection
2. Raw samples -> RDC0 (sweep packing, TBN/WAV sources)
3. ``RDCView``: bring any stored stage to the product a plot or export needs
4. Exports (streaming, one HDF5 file per stage, metadata + history preserved)
5. Array processing (RDC2 -> RDC3): steering, covariance, beamforming, angle spectra
6. Reductions used by plots (channel combining, Doppler-time, peak picking)

The numerical kernels (dechirp, FFTs, pulse-pair) live in ``lfm_utils``; file
layout lives in ``rdc_io``; LWA station and TBN specifics in ``lwa_utils``.
"""

import dataclasses
import os
import tempfile
from datetime import timedelta

import numpy as np

import lfm_utils as lfm
import rdc_io
import lwa_utils as lwa

CHANNEL_CHUNK = 8     # channels processed together when streaming
PULSE_BLOCK = 64      # sweeps dechirped together when streaming from RDC0


# ===========================================================================
# 1. Channel selection
# ===========================================================================

def select_channels(stands, pols, stand_spec="all", pol="both", bad_stands="none", receiver=False):
    """Increasing channel indices matching a stand list, polarization and bad-stand policy."""
    stands, pols = np.asarray(stands, int), np.asarray(pols, int)
    wanted = lwa.parse_stands(stand_spec, available=np.unique(stands))
    mask = np.isin(stands, wanted)
    if str(pol).lower() != "both":
        mask &= np.isin(pols, lwa.pol_indices(pol))
    if bad_stands != "none" and not receiver:
        candidates = np.flatnonzero(mask)
        mask[candidates] &= ~lwa.bad_stand_mask(stands[candidates], pols[candidates], bad_stands)
    idx = np.flatnonzero(mask)
    if not idx.size:
        raise ValueError("No channels left after --stands/--pol/--bad-stands selection")
    return idx


def polarization_summary(pols):
    unique = sorted(int(p) for p in np.unique(pols))
    mode = {(0,): "x", (1,): "y", (0, 1): "both"}.get(tuple(unique), "unknown")
    return dict(polarization_mode=mode, polarization_streams=unique, n_polarizations=len(unique))


# ===========================================================================
# 2. Raw samples -> RDC0
# ===========================================================================

def pack_sweeps(blocks, sweep_length, skip_samples=0, adc_start=0, adc_count=None, max_pulses=None, gap_fill="nan"):
    """Turn a stream of (channels, n) sample blocks into sweeps.

    Yields ``(pulse_index, adc_block, invalid)``: ``adc_block`` is (adc_count, channels),
    ``invalid`` counts nonfinite samples per channel before any zero fill.
    """
    adc_count = sweep_length - adc_start if adc_count is None else int(adc_count)
    if adc_start < 0 or adc_count < 1 or adc_start + adc_count > sweep_length:
        raise ValueError("--adc-start/--adc-count must select samples inside one sweep")
    buffer = None
    skip = int(skip_samples)
    pulse = 0
    for block, _ in blocks:
        if skip:
            drop = min(skip, block.shape[1])
            block, skip = block[:, drop:], skip - drop
        buffer = block if buffer is None else np.concatenate([buffer, block], axis=1)
        while buffer.shape[1] >= sweep_length:
            sweep = buffer[:, adc_start:adc_start + adc_count].T
            invalid = np.count_nonzero(~np.isfinite(sweep), axis=0)
            if gap_fill == "zero" and invalid.any():
                sweep = np.where(np.isfinite(sweep), sweep, 0)
            yield pulse, sweep.astype(np.complex64), invalid
            buffer = buffer[:, sweep_length:]
            pulse += 1
            if max_pulses is not None and pulse >= max_pulses:
                return


@dataclasses.dataclass
class RawSource:
    """Everything needed to cut a raw recording into sweeps."""
    blocks: object                    # generator of (samples[ch, n], gap_samples)
    sample_rate: float
    skip_samples: int
    channel_stands: np.ndarray
    channel_pols: np.ndarray
    n_samples: int                    # samples available after tstart
    metadata: dict
    close: object = None


def open_raw_source(path, sweep_frequency, tstart=None, tend=None, offset=0.0, timestamp_align=True,
                    stands="all", pol="both", bad_stands="none", start_utc=None, carrier_frequency=None):
    """Open a TBN or WAV recording as a ``RawSource`` aligned to the sweep grid."""
    path = os.path.expanduser(path)
    if path.lower().endswith(".wav"):
        import wav_utils
        info = wav_utils.wav_info(path)
        fs = info["sample_rate_hz"]
        if stands not in (None, "all") or pol != "both" or bad_stands != "none":
            raise ValueError("A WAV file is one receiver; omit --stands/--pol/--bad-stands")
        period = fs / sweep_frequency
        if not np.isclose(period, round(period), atol=1e-6):
            raise ValueError("WAV sample rate / sweep frequency must be an integer number of samples")
        tstart = 0.0 if tstart is None else tstart
        tend = info["duration_s"] if tend is None else tend
        start, stop = int(round(tstart * fs)), int(round(tend * fs))
        if not 0 <= start < stop <= info["n_samples"]:
            raise ValueError("Requested time range lies outside the WAV recording")
        epoch = wav_utils.parse_utc(start_utc)
        clock = start / fs + (epoch.timestamp() if epoch else 0.0)
        skip = 0 if not timestamp_align else lfm.sweep_alignment_samples(clock, sweep_frequency, fs, offset)
        first_utc = epoch + timedelta(seconds=(start + skip) / fs) if epoch else None
        meta = dict(source_format="wav", station=None, channel_id_kind="receiver", wav_layout=info["layout"],
                    tuning_frequency_hz=carrier_frequency, sample_rate_hz=fs)
        return RawSource(wav_utils.read_wav_blocks(path, start, stop - start), fs, skip,
                         np.array([1]), np.array([-1]), stop - start,
                         dict(meta, source_tstart_s=tstart, first_sweep_utc=first_utc))
    idf = lwa.open_tbn(path)
    info = lwa.tbn_info(idf)
    lwa.print_tbn_info(info)
    fs = info["sample_rate_hz"]
    tstart = 0.0 if tstart is None else tstart
    tend = info["usable_duration_s"] if tend is None else tend
    if not 0 <= tstart < tend:
        raise ValueError("--tend must be greater than --tstart")
    stand_list = lwa.parse_stands(stands, available=lwa.station_stands())
    rows, ch_stands, ch_pols = lwa.tbn_stream_indices(stand_list, pol)
    if bad_stands != "none":
        keep = ~lwa.bad_stand_mask(ch_stands, ch_pols, bad_stands)
        rows, ch_stands, ch_pols = rows[keep], ch_stands[keep], ch_pols[keep]
    stamp = lwa.probe_timestamp(path, tstart)
    frac = lwa.fractional_second(stamp) if timestamp_align else None
    skip = lfm.sweep_alignment_samples(frac, sweep_frequency, fs, offset)
    start_dt = lwa.to_datetime(stamp)
    first_utc = start_dt + timedelta(seconds=skip / fs) if start_dt is not None else None
    meta = dict(source_format="tbn", station="lwasv", channel_id_kind="physical_stand",
                channel_digitizers=(rows + 1).tolist(), tuning_frequency_hz=info["tuning_frequency_hz"],
                sample_rate_hz=fs, source_tstart_s=tstart, first_sweep_utc=first_utc)
    blocks = lwa.read_tbn_blocks(idf, tstart, tend - tstart, rows)
    return RawSource(blocks, fs, skip, ch_stands, ch_pols, int(round((tend - tstart) * fs)), meta, idf.close)


# ===========================================================================
# 3. RDCView: any stored stage -> the product that is needed
# ===========================================================================

class RDCView:
    """A channel/time selection of an RDC file that can produce RDC1, RDC2 frames or RDC2P.

    The most-processed stored stage is used when its irreversible settings match
    the request; otherwise the view recomputes from an earlier stored stage
    (retained RDC1, or RDC0). Conflicting requests that cannot be honoured
    raise ``ValueError`` instead of relabelling processed data.

    Times (``tstart``/``tend``) are seconds on the file's time axis: 0 is the
    first sweep of the original RDC0 export.
    """

    def __init__(self, path, stands="all", pol="both", bad_stands="none", tstart=None, tend=None,
                 bandwidth=None, dechirp_window=None, reference_gate=None, gap_fill=None):
        self.file = rdc_io.RDCFile(path)
        self.path = self.file.path
        self.meta = self.file.meta
        m = self.meta
        self.prf = float(m["sweep_frequency_hz"])
        self.fs = float(m["sample_rate_hz"]) if m.get("sample_rate_hz") else None
        self.fc = m.get("tuning_frequency_hz") or m.get("carrier_frequency_hz") or m.get("freq1_hz")
        self.receiver = self.file.is_receiver
        self.gap_fill = gap_fill or m.get("gap_fill", "nan")
        processed = any(self.file.has(s) for s in ("rdc1", "rdc2", "rdc2p", "rdc3"))
        self.bandwidth = _fixed(bandwidth, m.get("bandwidth_hz"), "bandwidth", float)
        self.dechirp_window = _fixed(dechirp_window, m.get("dechirp_window"), "dechirp-window", str,
                                     fixed=processed and not self.file.has("rdc0")) or "hann"
        gate = lfm.parse_reference_gate(reference_gate) if reference_gate else None
        stored_gate = m.get("reference_gate_frequency")
        if gate and processed and not self.file.has("rdc0") and (stored_gate is None or not np.isclose(gate[0], stored_gate)):
            raise ValueError("--reference-gate conflicts with the already-dechirped data; re-export from RDC0")
        self.gate = gate or ((stored_gate, m.get("reference_gate_duty", 0.5), m.get("reference_gate_phase", 0.0))
                             if stored_gate else None)
        self.lfm = None
        if self.bandwidth and self.fs:
            g = self.gate or (None, 0.5, 0.0)
            self.lfm = lfm.LFMWaveform(self.bandwidth, self.prf, self.fs, g[0], g[1], g[2])
        self.selection = dict(stands=stands, pol=pol, bad_stands=bad_stands)
        per_channel = [s for s in self.file.stages if s != "rdc3"]
        self.channels = {s: select_channels(*self.file.channel_ids(s), stands, pol, bad_stands, self.receiver)
                         for s in per_channel}
        if self.file.has("rdc3"):
            self.channels["rdc3"] = np.arange(self.file.shape("rdc3")[-1])
        if not per_channel:
            self.stands = self.pols = np.array([], int)
            self.tstart, self.tend, self.delay_ms, self._band = tstart, tend, None, None
            return
        ids = self.file.channel_ids(per_channel[0])
        base = per_channel
        self.stands = ids[0][self.channels[base[0]]]
        self.pols = ids[1][self.channels[base[0]]]
        self.tstart, self.tend = tstart, tend
        self.delay_ms = None
        self._band = None       # FFT bins (rdc0) or slice (rdc1/rdc2)

    # -- time -------------------------------------------------------------
    def pulse_source(self):
        """The stored per-sweep stage used for recomputation: 'rdc1' if present, else 'rdc0'."""
        for s in ("rdc1", "rdc0"):
            if self.file.has(s):
                return s
        return None

    def pulse_range(self, stage=None):
        stage = stage or self.pulse_source()
        origin = self.file.time_origin_s(stage)
        n = self.file.shape(stage)[1]
        first = 0 if self.tstart is None else int(np.ceil((self.tstart - origin) * self.prf - 1e-8))
        last = n if self.tend is None else int(np.floor((self.tend - origin) * self.prf + 1e-8))
        first, last = max(first, 0), min(last, n)
        if last <= first:
            raise ValueError(f"No complete sweeps in the requested time range "
                             f"(file covers {origin:.3f}-{origin + n / self.prf:.3f} s)")
        return first, last, origin

    def utc_at(self, t):
        zero = self.file.time_zero_utc
        return None if zero is None else zero + timedelta(seconds=float(t))

    # -- delay band -----------------------------------------------------------
    def full_delay_axis(self, stage):
        if stage == "rdc0":
            if self.lfm is None:
                raise ValueError("This RDC0 has no bandwidth; pass --bandwidth")
            return lfm.delay_axis_ms(self.lfm, self.file.shape("rdc0")[0])
        delay = np.asarray(self.file.axis(stage, "delay_ms"))
        return delay, np.arange(delay.size)

    def select_delay(self, d_min=None, d_max=None, full=False, stage=None, half_width_ms=1.5):
        """Choose the delay band; with no bounds, centre +/-1.5 ms on the strongest preview bin."""
        stage = stage or self.pulse_source() or self.file.stages[-1]
        delay, order = self.full_delay_axis(stage)
        if d_min is None and d_max is None and not full:
            peak = self._preview_peak(stage, delay, order)
            d_min, d_max = peak - half_width_ms, peak + half_width_ms
            print(f"Auto delay window: {d_min:.3f} to {d_max:.3f} ms (strongest bin {peak:.3f} ms)", flush=True)
        mask = np.ones(delay.shape, bool)
        if d_min is not None:
            mask &= delay >= d_min
        if d_max is not None:
            mask &= delay <= d_max
        if not mask.any():
            raise ValueError(f"No delay bins in the requested band; available {delay.min():.3f}-{delay.max():.3f} ms")
        self.delay_ms = delay[mask]
        self._band = order[mask]
        self._band_stage = stage
        return self.delay_ms

    def _preview_peak(self, stage, delay, order):
        first, last, _ = self.pulse_range(stage) if stage in ("rdc0", "rdc1") else (0, 1, 0)
        if stage in ("rdc0", "rdc1"):
            last = min(last, first + 32)
            power = np.zeros(delay.size)
            bins = order if stage == "rdc0" else slice(None)
            for _, _, block, _ in self._iter_pulse_stage(stage, self.channels[stage], first, last, bins):
                power += np.nansum(np.abs(block) ** 2, axis=(1, 2))
        elif stage == "rdc2p":
            power = np.nansum(self.file.read_real("rdc2p", "power", self.channels[stage]), axis=(1, 2))
        else:
            frame = self.file.read(stage, channels=self.channels[stage], frames=0)
            power = np.nansum(np.abs(frame) ** 2, axis=(1, 2))
        if not np.any(np.isfinite(power) & (power > 0)):
            raise ValueError("No finite signal in the preview; give --d-min/--d-max or --full-delay")
        return float(delay[int(np.nanargmax(power))])

    def _stage_band(self, stage):
        """Band selection for ``stage``: FFT bins for rdc0, a contiguous slice for stored delay axes."""
        if self.delay_ms is None:
            self.select_delay(full=True, stage=stage)
        if stage == "rdc0":
            if self._band_stage != "rdc0":
                delay, order = self.full_delay_axis("rdc0")
                return order[np.isin(np.round(delay, 9), np.round(self.delay_ms, 9))]
            return self._band
        delay = np.asarray(self.file.axis(stage, "delay_ms"))
        idx = np.flatnonzero((delay >= self.delay_ms[0] - 1e-9) & (delay <= self.delay_ms[-1] + 1e-9))
        if idx.size != self.delay_ms.size:
            raise ValueError(f"The stored {stage.upper()} delay axis does not cover the selected band")
        return slice(int(idx[0]), int(idx[-1]) + 1)

    # -- per-sweep data ---------------------------------------------------------
    def _iter_pulse_stage(self, stage, channels, first, last, bins):
        """Yield (start, stop, rdc1_block (delay, pulses, ch), invalid (pulses, ch)) per channel chunk."""
        for start in range(0, len(channels), CHANNEL_CHUNK):
            stop = min(start + CHANNEL_CHUNK, len(channels))
            ch = channels[start:stop]
            invalid = self.file.invalid_counts(stage, ch, slice(first, last))
            if stage == "rdc1":
                block = self.file.read("rdc1", ch, delay=bins, pulses=slice(first, last))
                bad = ~np.isfinite(block)
                if self.gap_fill == "zero" and bad.any():
                    block[bad] = 0
                nonfinite = np.count_nonzero(bad, axis=0)
            else:
                parts, nonfinite = [], []
                adc_start = int(self.meta.get("adc_start", 0))
                for p in range(first, last, PULSE_BLOCK):
                    q = min(p + PULSE_BLOCK, last)
                    raw = self.file.read("rdc0", ch, pulses=slice(p, q))
                    bad = ~np.isfinite(raw)
                    nonfinite.append(np.count_nonzero(bad, axis=0))
                    if self.gap_fill == "zero" and bad.any():
                        raw[bad] = 0
                    parts.append(lfm.dechirp_range_fft(raw, self.lfm, self.dechirp_window, adc_start, bins))
                block, nonfinite = np.concatenate(parts, axis=1), np.concatenate(nonfinite, axis=0)
            invalid = nonfinite if invalid is None else np.maximum(invalid, nonfinite)
            yield start, stop, block, invalid

    def iter_rdc1(self, channels=None, first=None, last=None):
        """Stream RDC1 for the selected delay band, channel chunk by channel chunk."""
        stage = self.pulse_source()
        if stage is None:
            raise ValueError("Per-sweep data (RDC0 or RDC1) is not stored in this file")
        if stage == "rdc0" and self.lfm is None:
            raise ValueError("Dechirping RDC0 needs a bandwidth; pass --bandwidth")
        f0, l0, _ = self.pulse_range(stage)
        first = f0 if first is None else first
        last = l0 if last is None else last
        channels = self.channels[stage] if channels is None else channels
        yield from self._iter_pulse_stage(stage, channels, first, last, self._stage_band(stage))

    def rdc1(self):
        """(delay, pulses, channels) for the whole selection, plus per-sweep gap flags and time axis."""
        blocks, invalid = [], []
        for _, _, block, inv in self.iter_rdc1():
            blocks.append(block)
            invalid.append(inv)
        first, last, origin = self.pulse_range()
        times = origin + np.arange(first, last) / self.prf
        return np.concatenate(blocks, axis=2), np.any(np.concatenate(invalid, axis=1) > 0, axis=1), times

    # -- Doppler frames ---------------------------------------------------------
    def stored_rdc2_matches(self, integration=None, hop=None, window=None, nfft=None):
        if not self.file.has("rdc2"):
            return False
        m = self.meta
        starts = np.asarray(self.file.axis("rdc2", "frame_tstart_s"))
        ends = np.asarray(self.file.axis("rdc2", "frame_tend_s"))
        stored = dict(integration=float(ends[0] - starts[0]),
                      hop=float(np.diff(starts)[0]) if starts.size > 1 else m.get("hop_time_s"),
                      window=m.get("slow_window"), nfft=m.get("nfft_doppler"))
        for key, value in dict(integration=integration, hop=hop, window=window, nfft=nfft).items():
            if value is None or stored[key] is None:
                continue
            same = np.isclose(float(value), float(stored[key]), atol=1e-6) if key != "window" else value == stored[key]
            if not same:
                return False
        return True

    def rdc2_frames(self, integration=None, hop=None, window=None, nfft=None, fd_min=None, fd_max=None):
        """Iterate ``dict(index, t0, t1, cube (delay, doppler, ch), doppler_hz, gaps)``.

        Uses stored RDC2 frames when compatible, otherwise computes from RDC1/RDC0.
        """
        if self.stored_rdc2_matches(integration, hop, window, nfft) and (
                self.pulse_source() is None or all(v is None for v in (integration, hop, window, nfft))):
            yield from self._stored_frames("rdc2", fd_min, fd_max)
            return
        if self.pulse_source() is None:
            raise ValueError("This RDC2 has no retained RDC1, so integration/hop/window/NFFT are fixed "
                             "at their exported values; re-export to change them")
        window = window or self.meta.get("slow_window") or "hann"
        if window in ("synthetic_rdc2_direct", "unknown"):
            window = "hann"
        rdc1, gaps, times = self.rdc1()
        n = rdc1.shape[1]
        count = n if integration is None else int(round(integration * self.prf))
        stride = count if hop is None else int(round(hop * self.prf))
        if count < 1 or stride < 1 or count > n:
            raise ValueError(f"Integration must cover 1..{n} sweeps and hop at least one sweep")
        nfft = lfm.default_nfft(count) if nfft is None else int(nfft)
        fd = lfm.doppler_axis_hz(nfft, self.prf)
        keep = range_mask(fd, fd_min, fd_max, "Doppler")
        for i, s in enumerate(range(0, n - count + 1, stride)):
            cube, _ = lfm.doppler_fft(rdc1[:, s:s + count], self.prf, window, nfft)
            yield dict(index=i, t0=float(times[s]), t1=float(times[s] + count / self.prf),
                       cube=cube[:, keep], doppler_hz=fd[keep], gaps=bool(gaps[s:s + count].any()),
                       integration=count / self.prf, hop=stride / self.prf, window=window, nfft=nfft)

    def _stored_frames(self, stage, fd_min=None, fd_max=None):
        fd = np.asarray(self.file.axis(stage, "doppler_hz"))
        keep = range_mask(fd, fd_min, fd_max, "Doppler")
        dsel = slice(int(np.flatnonzero(keep)[0]), int(np.flatnonzero(keep)[-1]) + 1)
        starts = np.asarray(self.file.axis(stage, "frame_tstart_s"))
        ends = np.asarray(self.file.axis(stage, "frame_tend_s"))
        band = self._stage_band(stage)
        lo = -np.inf if self.tstart is None else self.tstart - 1e-8
        hi = np.inf if self.tend is None else self.tend + 1e-8
        frames = np.flatnonzero((starts >= lo) & (ends <= hi))
        if not frames.size:
            raise ValueError("No stored frame lies inside the requested time range")
        m = self.meta
        for i, f in enumerate(frames):
            cube = self.file.read(stage, channels=self.channels[stage], frames=int(f), delay=band, doppler=dsel)
            bad = ~np.isfinite(cube)
            if self.gap_fill == "zero" and bad.any():
                cube[bad] = 0
            inv = self.file.invalid_counts(stage, self.channels[stage], int(f)) if stage == "rdc2" else None
            yield dict(index=i, t0=float(starts[f]), t1=float(ends[f]), cube=cube, doppler_hz=fd[dsel],
                       gaps=bool(bad.any() or (inv is not None and np.any(inv > 0))),
                       integration=float(ends[f] - starts[f]), hop=m.get("hop_time_s"),
                       window=m.get("slow_window"), nfft=m.get("nfft_doppler"))

    def rdc3_frames(self, beam=0, fd_min=None, fd_max=None):
        self.channels["rdc3"] = np.array([beam])
        yield from self._stored_frames("rdc3", fd_min, fd_max)

    # -- phase-derived Doppler ------------------------------------------------
    def rdc2p(self, estimator_sweeps=1):
        """Return dict(lag1, power, n_valid (delay, est, ch), times (est,), gaps (est,))."""
        e = int(estimator_sweeps)
        if self.file.has("rdc2p") and int(self.meta.get("estimator_sweeps", 1)) == e and self.pulse_source() is None:
            band = self._stage_band("rdc2p")
            ch = self.channels["rdc2p"]
            lag1 = self.file.read("rdc2p", ch, delay=band)
            power = self.file.read_real("rdc2p", "power", ch, delay=band)
            n_valid = self.file.read_real("rdc2p", "n_valid", ch, delay=band)
            times = np.asarray(self.file.axis("rdc2p", "time_s"))
            keep = np.ones(times.size, bool)
            if self.tstart is not None:
                keep &= times >= self.tstart
            if self.tend is not None:
                keep &= times <= self.tend
            gaps = np.any(n_valid < e, axis=(0, 2))
            return dict(lag1=lag1[:, keep], power=power[:, keep], n_valid=n_valid[:, keep],
                        times=times[keep], gaps=gaps[keep])
        if self.file.has("rdc2p") and self.pulse_source() is None:
            raise ValueError("Stored RDC2P used a different --estimator-sweeps; re-export it")
        rdc1, gaps, times = self.rdc1()
        lag1, power, n_valid = lfm.pulse_pair(rdc1, e)
        t = times[0] + lfm.pulse_pair_time_s(lag1.shape[1], e, self.prf)
        g = np.array([gaps[i:i + e + 1].any() for i in range(lag1.shape[1])])
        return dict(lag1=lag1, power=power, n_valid=n_valid, times=t, gaps=g)

    # -- raw samples ----------------------------------------------------------
    def raw_samples(self, channels=None):
        """(channels, samples) concatenated sweeps for spectra; needs complete RDC0 sweeps."""
        if not self.file.has("rdc0"):
            raise ValueError("Spectra need raw samples (RDC0); this file holds only processed stages")
        if int(self.meta.get("adc_start", 0)) != 0 or self.file.shape("rdc0")[0] != int(round(self.fs / self.prf)):
            raise ValueError("Spectra need complete sweeps; this RDC0 was cropped with --adc-start/--adc-count")
        first, last, origin = self.pulse_range("rdc0")
        channels = self.channels["rdc0"] if channels is None else channels
        cube = self.file.read("rdc0", channels, pulses=slice(first, last))
        gaps = ~np.isfinite(cube)
        if self.gap_fill == "zero":
            cube[gaps] = 0
        return cube.transpose(2, 1, 0).reshape(len(channels), -1), origin + first / self.prf, bool(gaps.any())

    # -- description ------------------------------------------------------------
    def info_rows(self, t0=None, t1=None):
        m = self.meta
        rows = [("File", os.path.basename(self.path)),
                ("Stored stages", ", ".join(s.upper() for s in self.file.stages)),
                ("Tuning freq", "unknown" if not self.fc else f"{float(self.fc) / 1e6:.6f} MHz")]
        if self.receiver:
            rows.append(("Receiver", "WAV I/Q"))
        else:
            n = len(np.unique(self.stands))
            rows.append(("Channels", f"{len(self.stands)} ({n} stand{'s' if n != 1 else ''}, "
                                     f"pol {polarization_summary(self.pols)['polarization_mode']})"))
            if n <= 4:
                rows.append(("Stands", ", ".join(str(s) for s in np.unique(self.stands))))
        if t0 is not None:
            rows.append(("Plot span", f"{t0:.3f}-{t1:.3f} s"))
        if self.fs:
            rows.append(("Sample rate", f"{self.fs / 1e3:.3f} kHz"))
        rows.append(("Sweep freq", f"{self.prf:.8g} Hz"))
        if self.bandwidth:
            rows.append(("Bandwidth", f"{self.bandwidth / 1e3:.6g} kHz"))
        if self.gate:
            rows.append(("Ref. gate", f"{self.gate[0]:.6g} Hz, duty {self.gate[1] * 100:.0f}%, phase {self.gate[2] * 1e3:.2f} ms"))
        rows.append(("Dechirp window", self.dechirp_window))
        if m.get("sweep_offset_s") not in (None, 0, 0.0):
            rows.append(("Sweep offset", f"{float(m['sweep_offset_s']) * 1e3:.3f} ms"))
        rows.append(("Gap fill", self.gap_fill))
        return rows

    def utc_note(self, t0, t1):
        a, b = self.utc_at(t0), self.utc_at(t1)
        if a is None:
            return None
        return f"UTC start: {a.isoformat(timespec='milliseconds')}\nUTC end: {b.isoformat(timespec='milliseconds')}"


def _fixed(requested, stored, flag, cast, fixed=True):
    if requested is not None and stored is not None and fixed:
        same = np.isclose(cast(requested), cast(stored), rtol=1e-10) if cast is float else cast(requested) == cast(stored)
        if not same:
            raise ValueError(f"--{flag} {requested} conflicts with the stored value {stored}; re-export from an earlier stage")
    value = requested if requested is not None else stored
    return None if value is None else cast(value)


def range_mask(values, lo, hi, name):
    mask = np.ones(np.shape(values), bool)
    if lo is not None:
        mask &= values >= lo
    if hi is not None:
        mask &= values <= hi
    if not mask.any():
        raise ValueError(f"No {name} bins in the requested range")
    return mask


# ===========================================================================
# 4. Exports
# ===========================================================================

@dataclasses.dataclass
class ExportSettings:
    # raw input
    sweep_frequency: float = None
    offset: float = 0.0
    timestamp_align: bool = True
    start_utc: str = None
    carrier_frequency: float = None
    adc_start: int = 0
    adc_count: int = None
    # selection
    tstart: float = None
    tend: float = None
    stands: str = "all"
    pol: str = "both"
    bad_stands: str = "none"
    gap_fill: str = None
    # range processing
    bandwidth: float = None
    dechirp_window: str = None
    reference_gate: str = None
    d_min: float = None
    d_max: float = None
    # Doppler processing
    integration_time: float = None
    hop_time: float = None
    slow_window: str = "hann"
    nfft_doppler: int = None
    fd_min: float = None
    fd_max: float = None
    keep_rdc1: bool = True
    estimator_sweeps: int = 1
    # storage
    dtype: str = "float32"
    compression: str = "gzip"
    dry_run: bool = False


def export(input_path, target, output, s, tool=None):
    """Export ``input_path`` (raw TBN/WAV or any RDC file) to ``target`` stage in ``output``."""
    target = rdc_io.stage_name(target)
    input_path = os.path.expanduser(input_path)
    if os.path.realpath(input_path) == os.path.realpath(os.path.expanduser(output)):
        raise ValueError("The output must not overwrite the input")
    tool = tool or f"export_{target}"
    if not input_path.lower().endswith((".h5", ".hdf5")):
        if target == "rdc0":
            return export_rdc0(input_path, output, s, tool)
        with tempfile.TemporaryDirectory(prefix="rdc0_") as tmp:
            temp = os.path.join(tmp, "rdc0.h5")
            quick = dataclasses.replace(s, compression="lzf", dtype="float32", dry_run=False)
            export_rdc0(input_path, temp, quick, tool, temporary=True)
            chained = dataclasses.replace(s, tstart=None, tend=None, stands="all", pol="both", bad_stands="none")
            return export(temp, target, output, chained, tool)
    if target == "rdc0":
        raise ValueError("RDC0 is made from a raw TBN/WAV recording")
    if target == "rdc3":
        raise ValueError("Use export_rdc3.py (array settings) for RDC3")
    view = RDCView(input_path, s.stands, s.pol, s.bad_stands, s.tstart, s.tend,
                   s.bandwidth, s.dechirp_window, s.reference_gate, s.gap_fill)
    if view.pulse_source() is None:
        raise ValueError(f"{target.upper()} needs per-sweep data (RDC0 or RDC1) in the input")
    view.select_delay(s.d_min, s.d_max, full=s.d_min is None and s.d_max is None)
    return _export_processed(view, target, output, s, tool)


def export_rdc0(raw_path, output, s, tool="export_rdc0", temporary=False):
    if not s.sweep_frequency or s.sweep_frequency <= 0:
        raise ValueError("--sweep-frequency is required for raw input")
    src = open_raw_source(raw_path, s.sweep_frequency, s.tstart, s.tend, s.offset, s.timestamp_align,
                          s.stands, s.pol, s.bad_stands, s.start_utc, s.carrier_frequency)
    try:
        fs = src.sample_rate
        sweep = int(round(fs / s.sweep_frequency))
        adc_count = sweep - s.adc_start if s.adc_count is None else s.adc_count
        n_pulses = (src.n_samples - src.skip_samples) // sweep
        if n_pulses < 1:
            raise ValueError("The selection does not contain one complete sweep after alignment")
        n_ch = len(src.channel_stands)
        size = 2 * adc_count * n_pulses * n_ch * np.dtype(s.dtype).itemsize
        print(f"RDC0 shape (adc, pulses, channels) = ({adc_count}, {n_pulses}, {n_ch}); "
              f"{size / 1e9:.2f} GB uncompressed", flush=True)
        if s.dry_run:
            return None
        gap_fill = s.gap_fill or "nan"
        with rdc_io.atomic_output(output) as tmp:
            with rdc_io._h5py().File(tmp, "w") as h5:
                w = rdc_io.StageWriter(h5, "rdc0", (adc_count, n_pulses, n_ch), s.dtype, s.compression,
                                       maxshape=(adc_count, None, n_ch))
                invalid = w.dataset("invalid_counts", shape=(n_pulses, n_ch), dtype="int64", maxshape=(None, n_ch))
                written = 0
                for p, sweep_block, inv in pack_sweeps(src.blocks, sweep, src.skip_samples, s.adc_start,
                                                        adc_count, n_pulses, gap_fill):
                    w.write((slice(None), p, slice(None)), sweep_block)
                    invalid[p] = inv
                    written = p + 1
                    if not temporary and written % 20 == 0:
                        print(f"  RDC0 sweep {written}/{n_pulses}", flush=True)
                if written < n_pulses:
                    print(f"Input ended early: {written} of {n_pulses} sweeps written", flush=True)
                    w.resize((adc_count, written, n_ch))
                    invalid.resize((written, n_ch))
                w.channels(src.channel_stands, src.channel_pols)
                w.dataset("adc_sample_index", np.arange(s.adc_start, s.adc_start + adc_count))
                meta = dict(src.metadata)
                first = meta.pop("first_sweep_utc", None)
                meta.update(
                    format_version=rdc_io.FORMAT_VERSION, rdc_stage=0, sweep_frequency_hz=float(s.sweep_frequency),
                    bandwidth_hz=s.bandwidth, sweep_length_samples=sweep, adc_start=s.adc_start, adc_count=adc_count,
                    start_offset_samples=int(src.skip_samples), sweep_offset_s=float(s.offset or 0.0),
                    timestamp_aligned=bool(s.timestamp_align), gap_fill=gap_fill, tstart=0.0,
                    time_zero_utc=first.isoformat() if first else None,
                    pulse0_utc=first.isoformat() if first else None,
                    source_file=os.path.basename(raw_path), n_channels=n_ch, dtype=s.dtype,
                    **polarization_summary(src.channel_pols),
                    history=[rdc_io.history_entry(tool, stage="rdc0", source=os.path.basename(raw_path),
                                                  tstart=s.tstart, tend=s.tend, offset=s.offset,
                                                  timestamp_align=s.timestamp_align, stands=s.stands, pol=s.pol,
                                                  bad_stands=s.bad_stands, adc_start=s.adc_start,
                                                  adc_count=adc_count, gap_fill=gap_fill)])
                rdc_io.write_metadata(h5, meta)
    finally:
        if src.close:
            src.close()
    if not temporary:
        print(f"Wrote {output}", flush=True)
    return output


def _export_processed(view, target, output, s, tool):
    stage = view.pulse_source()
    first, last, origin = view.pulse_range(stage)
    n_pulses = last - first
    channels = view.channels[stage]
    stands, pols = view.file.channel_ids(stage)
    stands, pols = stands[channels], pols[channels]
    n_ch, n_delay = len(channels), view.delay_ms.size
    prf = view.prf

    if target in ("rdc2",):
        count = n_pulses if s.integration_time is None else int(round(s.integration_time * prf))
        stride = count if s.hop_time is None else int(round(s.hop_time * prf))
        if count < 1 or stride < 1 or count > n_pulses:
            raise ValueError(f"Integration must cover 1..{n_pulses} sweeps and hop at least one sweep")
        nfft = lfm.default_nfft(count) if s.nfft_doppler is None else int(s.nfft_doppler)
        if nfft < count:
            raise ValueError("--nfft-doppler must be >= the sweeps per integration")
        fd = lfm.doppler_axis_hz(nfft, prf)
        fd_keep = range_mask(fd, s.fd_min, s.fd_max, "Doppler")
        starts = np.arange(0, n_pulses - count + 1, stride)
        rdc2_shape = (len(starts), n_delay, int(fd_keep.sum()), n_ch)
    if target == "rdc2p":
        e = int(s.estimator_sweeps)
        n_est = n_pulses - e
        if n_est < 1:
            raise ValueError(f"Need more than {e} sweeps for phase-derived Doppler")

    item = np.dtype(s.dtype).itemsize * 2
    sizes = {"rdc1": n_delay * n_pulses * n_ch * item}
    if target == "rdc2":
        sizes["rdc2"] = int(np.prod(rdc2_shape)) * item
        if not s.keep_rdc1:
            sizes.pop("rdc1")
    if target == "rdc2p":
        sizes = {"rdc2p": n_delay * n_est * n_ch * (item + 8)}
    print(f"{target.upper()} from {stage.upper()}: {n_delay} delay bins "
          f"({view.delay_ms[0]:.3f}-{view.delay_ms[-1]:.3f} ms), {n_pulses} sweeps, {n_ch} channels; "
          f"{sum(sizes.values()) / 1e9:.2f} GB uncompressed", flush=True)
    if s.dry_run:
        return None

    t0 = origin + first / prf
    source_meta = dict(view.meta)
    source_meta.pop("history", None)
    history = list(view.meta.get("history", []))
    with rdc_io.atomic_output(output) as tmp:
        with rdc_io._h5py().File(tmp, "w") as h5:
            w1 = w2 = wp = None
            if "rdc1" in sizes:
                w1 = rdc_io.StageWriter(h5, "rdc1", (n_delay, n_pulses, n_ch), s.dtype, s.compression)
                inv1 = w1.dataset("invalid_counts", shape=(n_pulses, n_ch), dtype="int64")
                w1.group.attrs["tstart_s"] = t0
                w1.group.attrs["sweep_frequency_hz"] = prf
            if target == "rdc2":
                w2 = rdc_io.StageWriter(h5, "rdc2", rdc2_shape, s.dtype, s.compression)
                inv2 = w2.dataset("invalid_counts", shape=(len(starts), n_ch), dtype="int64")
            if target == "rdc2p":
                wp = rdc_io.StageWriter(h5, "rdc2p", (n_delay, n_est, n_ch), s.dtype, s.compression)
                pow_ds = wp.dataset("power", shape=(n_delay, n_est, n_ch), dtype="float32")
                nv_ds = wp.dataset("n_valid", shape=(n_delay, n_est, n_ch), dtype="int16")
                wp.group.attrs["tstart_s"] = t0
            for a, b, block, invalid in view.iter_rdc1(channels, first, last):
                cs = slice(a, b)
                if w1 is not None:
                    w1.write((slice(None), slice(None), cs), block)
                    inv1[:, cs] = invalid
                if w2 is not None:
                    for i, p in enumerate(starts):
                        cube, _ = lfm.doppler_fft(block[:, p:p + count], prf, s.slow_window, nfft)
                        w2.write((i, slice(None), slice(None), cs), cube[:, fd_keep])
                        inv2[i, cs] = invalid[p:p + count].sum(axis=0)
                if wp is not None:
                    lag1, power, n_valid = lfm.pulse_pair(block, e)
                    wp.write((slice(None), slice(None), cs), lag1)
                    pow_ds[:, :, cs] = power
                    nv_ds[:, :, cs] = n_valid
                print(f"  channels {b}/{n_ch}", flush=True)
            meta = dict(source_meta)
            meta.update(format_version=rdc_io.FORMAT_VERSION, rdc_stage=int(target[3]) if target != "rdc2p" else "2p",
                        bandwidth_hz=view.bandwidth,
                        dechirp_window=view.dechirp_window, gap_fill=view.gap_fill,
                        d_min_ms=float(view.delay_ms[0]), d_max_ms=float(view.delay_ms[-1]),
                        tstart=t0, tend=t0 + n_pulses / prf, n_channels=n_ch,
                        selected_stands=stands.tolist(), dtype=s.dtype,
                        time_zero_utc=(view.file.time_zero_utc.isoformat() if view.file.time_zero_utc else None),
                        **polarization_summary(pols))
            if view.gate:
                meta.update(reference_gate_frequency=view.gate[0], reference_gate_duty=view.gate[1],
                            reference_gate_phase=view.gate[2])
            params = dict(stage=target, source=os.path.basename(view.path), source_stage=stage,
                          tstart=t0, tend=t0 + n_pulses / prf, stands=s.stands, pol=s.pol,
                          bad_stands=s.bad_stands, d_min_ms=meta["d_min_ms"], d_max_ms=meta["d_max_ms"],
                          dechirp_window=view.dechirp_window, reference_gate=view.gate, gap_fill=view.gap_fill)
            for writer in (w1, w2, wp):
                if writer is not None:
                    writer.channels(stands, pols)
                    writer.dataset("delay_ms", view.delay_ms)
            if w2 is not None:
                w2.dataset("doppler_hz", fd[fd_keep])
                w2.dataset("frame_tstart_s", t0 + starts / prf)
                w2.dataset("frame_tend_s", t0 + (starts + count) / prf)
                meta.update(integration_time_s=count / prf, hop_time_s=stride / prf, slow_window=s.slow_window,
                            nfft_doppler=nfft, fd_min_hz=s.fd_min, fd_max_hz=s.fd_max, retains_rdc1=w1 is not None)
                params.update(integration_time_s=count / prf, hop_time_s=stride / prf, slow_window=s.slow_window,
                              nfft_doppler=nfft, fd_min_hz=s.fd_min, fd_max_hz=s.fd_max, keep_rdc1=w1 is not None)
            if wp is not None:
                wp.dataset("time_s", t0 + lfm.pulse_pair_time_s(n_est, e, prf))
                meta.update(estimator_sweeps=e)
                params.update(estimator_sweeps=e)
            meta["history"] = history + [rdc_io.history_entry(tool, **params)]
            rdc_io.write_metadata(h5, meta)
    print(f"Wrote {output}", flush=True)
    return output


def export_rdc3(view, output, processor, looks=None, frame_settings=None, dtype="float32",
                compression="gzip", tool="export_rdc3"):
    """Beamform RDC2 frames into RDC3 (frames, delay, doppler, beams).

    ``looks``: list of (az, el); None = one beam auto-steered per frame to the
    strongest signal (its direction is stored per frame).
    """
    frames = list(view.rdc2_frames(**(frame_settings or {})))
    if not frames:
        raise ValueError("No RDC2 frames in the selection")
    n_beams = 1 if looks is None else len(looks)
    shape = (len(frames),) + frames[0]["cube"].shape[:2] + (n_beams,)
    beam_az = np.zeros((len(frames), n_beams))
    beam_el = np.zeros((len(frames), n_beams))
    with rdc_io.atomic_output(output) as tmp:
        with rdc_io._h5py().File(tmp, "w") as h5:
            w = rdc_io.StageWriter(h5, "rdc3", shape, dtype, compression)
            for i, fr in enumerate(frames):
                cube = processor.prepare(fr["cube"])
                directions = looks or [processor.look_direction(cube)]
                for b, (az, el) in enumerate(directions):
                    w.write((i, slice(None), slice(None), b), processor.beamform(cube, az, el))
                    beam_az[i, b], beam_el[i, b] = az, el
                print(f"  frame {i + 1}/{len(frames)}: beams " +
                      ", ".join(f"({a:.1f}, {e:.1f})" for a, e in directions), flush=True)
            w.dataset("delay_ms", view.delay_ms)
            w.dataset("doppler_hz", frames[0]["doppler_hz"])
            w.dataset("frame_tstart_s", [f["t0"] for f in frames])
            w.dataset("frame_tend_s", [f["t1"] for f in frames])
            w.dataset("beam_az_deg", beam_az)
            w.dataset("beam_el_deg", beam_el)
            meta = dict(view.meta)
            history = list(meta.pop("history", []))
            params = dict(stage="rdc3", source=os.path.basename(view.path), beamformer=processor.method,
                          looks="auto per frame" if looks is None else looks, cable_correction=processor.cable_note,
                          carrier_frequency_hz=processor.fc, diagonal_loading=processor.loading,
                          stands=processor.stands.tolist(), pol=int(processor.pols[0]),
                          d_min_ms=float(view.delay_ms[0]), d_max_ms=float(view.delay_ms[-1]),
                          integration_time_s=frames[0]["integration"], hop_time_s=frames[0]["hop"],
                          slow_window=frames[0]["window"], nfft_doppler=frames[0]["nfft"])
            meta.update(format_version=rdc_io.FORMAT_VERSION, rdc_stage=3, beamformer=processor.method,
                        cable_phase_corrected=True, array_stands=processor.stands.tolist(),
                        integration_time_s=frames[0]["integration"], hop_time_s=frames[0]["hop"],
                        slow_window=frames[0]["window"], nfft_doppler=frames[0]["nfft"],
                        d_min_ms=float(view.delay_ms[0]), d_max_ms=float(view.delay_ms[-1]),
                        history=history + [rdc_io.history_entry(tool, **params)])
            rdc_io.write_metadata(h5, meta)
    print(f"Wrote {output}", flush=True)
    return output


# ===========================================================================
# 5. Array processing (RDC2 -> RDC3)
# ===========================================================================

def unit_vectors(az_deg, el_deg):
    az, el = np.deg2rad(np.asarray(az_deg, float)), np.deg2rad(np.asarray(el_deg, float))
    return np.stack([np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)], axis=-1)


def steering_vector(positions_m, wavelength_m, az_deg, el_deg):
    """Plane-wave response exp(+j 2 pi r.u / lambda); u points toward the source (az from north, clockwise)."""
    return np.exp(2j * np.pi / wavelength_m * (np.asarray(positions_m) @ unit_vectors(az_deg, el_deg)))


def steering_matrix(positions_m, frequency_hz, az_deg, el_deg):
    """Rows are steering vectors for each (az, el) pair."""
    u = unit_vectors(az_deg, el_deg).reshape(-1, 3)
    return np.exp(2j * np.pi * frequency_hz / lfm.C_M_PER_S * (u @ np.asarray(positions_m, float).T))


def spatial_covariance(snapshots, loading=0.1, max_power_percentile=None):
    """R = E[x x^H] from (K, N) snapshots, with diagonal loading relative to mean channel power."""
    x = np.asarray(snapshots, np.complex128).reshape(-1, np.shape(snapshots)[-1])
    x = x[np.all(np.isfinite(x), axis=1)]
    if max_power_percentile is not None and len(x) > 2 * x.shape[1]:
        p = np.mean(np.abs(x) ** 2, axis=1)
        quiet = p <= np.percentile(p, float(max_power_percentile))
        if quiet.sum() >= 2:
            x = x[quiet]
    if len(x) < 2:
        raise ValueError("Not enough finite snapshots to estimate a spatial covariance")
    R = x.T @ x.conj() / len(x)
    return R + float(loading) * (np.trace(R).real / R.shape[0] + 1e-30) * np.eye(R.shape[0])


def angle_spectrum(R, positions_m, frequency_hz, az_grid, el_grid, method="conventional", block=2048):
    """(len(el_grid), len(az_grid)) spatial spectrum: conventional a^H R a / N^2 or MVDR 1 / a^H R^-1 a."""
    AZ, EL = np.meshgrid(np.asarray(az_grid, float), np.asarray(el_grid, float))
    az, el = AZ.ravel(), EL.ravel()
    M = np.linalg.pinv(R) if method == "mvdr" else R
    n = R.shape[0]
    out = np.empty(az.size)
    for i in range(0, az.size, block):
        A = steering_matrix(positions_m, frequency_hz, az[i:i + block], el[i:i + block])
        q = np.real(np.sum((A.conj() @ M) * A, axis=1))
        out[i:i + block] = 1.0 / np.maximum(q, 1e-30) if method == "mvdr" else q / n ** 2
    if method not in ("mvdr", "conventional"):
        raise ValueError("angle method must be conventional or mvdr")
    return out.reshape(AZ.shape)


def peak_direction(R, positions_m, frequency_hz, method="conventional", step=2.0, fine=0.1,
                   el_min=0.0, el_max=90.0):
    """Grid search followed by local refinement. Returns (az, el, value)."""
    az = np.arange(-180.0, 180.0, step)
    el = np.arange(el_min, el_max + 0.5 * step, step)
    spec = angle_spectrum(R, positions_m, frequency_hz, az, el, method)
    ei, ai = np.unravel_index(np.argmax(spec), spec.shape)
    az_f = np.arange(az[ai] - step, az[ai] + step + 1e-9, fine)
    el_f = np.clip(np.arange(el[ei] - step, el[ei] + step + 1e-9, fine), el_min, el_max)
    spec_f = angle_spectrum(R, positions_m, frequency_hz, az_f, el_f, method)
    ei, ai = np.unravel_index(np.argmax(spec_f), spec_f.shape)
    return float((az_f[ai] + 180) % 360 - 180), float(el_f[ei]), float(spec_f[ei, ai])


def beam_weights(positions_m, frequency_hz, az_deg, el_deg, method="conventional", R=None):
    a = steering_matrix(positions_m, frequency_hz, [az_deg], [el_deg])[0]
    if method == "conventional":
        return a / a.size
    if method == "mvdr":
        if R is None:
            raise ValueError("MVDR weights need a covariance matrix")
        Ri_a = np.linalg.pinv(R) @ a
        return Ri_a / (a.conj() @ Ri_a)
    raise ValueError("beamformer must be conventional or mvdr")


def beam_pattern(positions_m, frequency_hz, look_az, look_el, az_grid, el_grid):
    """|w^H a|^2 of conventional weights steered to (look_az, look_el); peak = 1."""
    w = beam_weights(positions_m, frequency_hz, look_az, look_el)
    AZ, EL = np.meshgrid(az_grid, el_grid)
    A = steering_matrix(positions_m, frequency_hz, AZ.ravel(), EL.ravel())
    return (np.abs(A @ w.conj()) ** 2).reshape(AZ.shape)


def pick_cells(power, n, delay_guard=3, doppler_guard=5):
    """Strongest (delay, Doppler) cells with non-maximum suppression."""
    work = np.where(np.isfinite(power), power, -np.inf).astype(float)
    cells = []
    for _ in range(int(n)):
        flat = int(np.argmax(work))
        if not np.isfinite(work.flat[flat]):
            break
        r, c = np.unravel_index(flat, work.shape)
        cells.append((int(r), int(c)))
        work[max(0, r - delay_guard):r + delay_guard + 1, max(0, c - doppler_guard):c + doppler_guard + 1] = -np.inf
    return cells


class ArrayProcessor:
    """Calibrated beamforming and angle spectra for one channel set.

    ``positions``: (N, 3) metres; ``fc``: steering frequency (the tuning frequency,
    correct for dechirped data); ``cable``: per-channel correction factors.
    """

    def __init__(self, stands, pols, fc, method="conventional", cable_correction="auto", metadata=None,
                 loading=0.1, background_percentile=80.0):
        metadata = metadata or {}
        if fc is None:
            raise ValueError("Array processing needs the tuning frequency; pass --carrier-frequency")
        if len(np.unique(pols)) != 1:
            raise ValueError("Array processing uses one polarization at a time; pass --pol x or --pol y")
        positions, found = lwa.stand_positions(stands)
        if not found.all():
            print(f"Dropping stands without LSL positions: {np.asarray(stands)[~found].tolist()}", flush=True)
        self.keep = found
        self.stands, self.pols = np.asarray(stands)[found], np.asarray(pols)[found]
        self.positions = positions
        self.fc = float(fc)
        self.method = method
        self.loading = loading
        self.background_percentile = background_percentile
        corrected = metadata.get("cable_phase_corrected", False)
        ideal_sim = metadata.get("synthetic", False) and not metadata.get("cable_delays_injected", False)
        if cable_correction == "off" or corrected or (cable_correction == "auto" and ideal_sim):
            self.cable = np.ones(len(self.stands), complex)
            self.cable_note = "already corrected" if corrected else ("ideal simulation" if ideal_sim else "off")
        else:
            self.cable = lwa.cable_phase_factors(self.stands, self.pols, self.fc)
            self.cable_note = f"LSL LWA-SV at {self.fc / 1e6:.6f} MHz"

    def prepare(self, cube):
        """Drop channels without positions and apply cable correction. cube: (..., channels)."""
        return cube[..., self.keep] * self.cable.astype(np.complex64)

    def signal_covariance(self, cube, top_fraction=0.01, min_cells=16):
        """Covariance from the strongest delay-Doppler cells (the signal), for look-direction search."""
        x = cube.reshape(-1, cube.shape[-1])
        p = np.nansum(np.abs(x) ** 2, axis=1)
        n = max(min_cells, int(top_fraction * len(p)))
        top = np.argsort(np.where(np.isfinite(p), p, -np.inf))[::-1][:n]
        return spatial_covariance(x[top], self.loading)

    def look_direction(self, cube, step=2.0):
        az, el, _ = peak_direction(self.signal_covariance(cube), self.positions, self.fc, "conventional", step)
        return az, el

    def beamform(self, cube, az, el):
        """cube (delay, doppler, channels) -> complex beam (delay, doppler)."""
        R = None
        if self.method == "mvdr":
            R = spatial_covariance(cube.reshape(-1, cube.shape[-1]), self.loading, self.background_percentile)
        w = beam_weights(self.positions, self.fc, az, el, self.method, R)
        return np.tensordot(cube, w.conj(), axes=([-1], [0]))

    def angle_map(self, snapshots, az_grid, el_grid, method=None):
        R = spatial_covariance(snapshots, self.loading)
        return angle_spectrum(R, self.positions, self.fc, az_grid, el_grid, method or self.method), R


# ===========================================================================
# 6. Reductions used by plots
# ===========================================================================

def incoherent_power(cube, combine="sum"):
    """Combine channels (last axis) by power: sum, mean or max."""
    p = np.abs(cube) ** 2
    return {"sum": np.nansum, "mean": np.nanmean, "max": np.nanmax}[combine](p, axis=-1)


def to_db(power):
    return 10.0 * np.log10(np.asarray(power) + 1e-12)


def strongest_delays(power, delay_ms, n=5, min_separation_ms=None):
    """Up to ``n`` strongest delays (ms) of a (..., delay) power array, separated by at least 0.5 ms."""
    power = np.asarray(power, float)
    per_delay = power if power.ndim == 1 else np.nanmax(power.reshape(-1, power.shape[-1]), axis=0)
    if min_separation_ms is None:
        step = float(np.median(np.abs(np.diff(delay_ms)))) if len(delay_ms) > 1 else 0.0
        min_separation_ms = max(0.5, 3 * step)
    work, peaks = per_delay.copy(), []
    for _ in range(n):
        if not np.any(np.isfinite(work)):
            break
        i = int(np.nanargmax(work))
        peaks.append(float(delay_ms[i]))
        work[np.abs(delay_ms - delay_ms[i]) <= min_separation_ms] = np.nan
    return peaks


def doppler_time(rdc1, prf, integration, hop, window="hann", nfft=None, combine="incoherent"):
    """Doppler spectra of a delay band vs time. Returns (times from band start, fd, power (time, fd)).

    combine: incoherent (mean over delay of channel-summed power), coherent
    (sum channels complex first) or peak (strongest delay bin only).
    """
    n = rdc1.shape[1]
    count = int(round(integration * prf))
    stride = int(round(hop * prf))
    if count < 1 or stride < 1 or count > n:
        raise ValueError(f"Integration must cover 1..{n} sweeps and hop at least one sweep")
    starts = np.arange(0, n - count + 1, stride)
    power, fd = [], None
    for s in starts:
        X, fd = lfm.doppler_fft(rdc1[:, s:s + count], prf, window, nfft)
        if combine == "incoherent":
            power.append(np.mean(np.sum(np.abs(X) ** 2, axis=2), axis=0))
        elif combine == "coherent":
            power.append(np.sum(np.abs(np.sum(X, axis=2)) ** 2, axis=0) / X.shape[2])
        elif combine == "peak":
            strongest = int(np.argmax(np.sum(np.abs(rdc1[:, s:s + count]) ** 2, axis=(1, 2))))
            power.append(np.sum(np.abs(X[strongest]) ** 2, axis=1))
        else:
            raise ValueError("combine must be incoherent, coherent or peak")
    return (starts + count / 2.0) / prf, fd, np.array(power)


def phase_doppler_histogram(fd_hz, power_db, fd_edges, combine="max", power_floor_db=None):
    """Bin per-delay phase-Doppler estimates (time, delay) into (time, fd bins)."""
    n_t, n_b = fd_hz.shape[0], len(fd_edges) - 1
    out = np.full((n_t, n_b), np.nan)
    for t in range(n_t):
        ok = np.isfinite(fd_hz[t]) & np.isfinite(power_db[t]) & (fd_hz[t] >= fd_edges[0]) & (fd_hz[t] <= fd_edges[-1])
        if power_floor_db is not None:
            ok &= power_db[t] >= power_floor_db
        if not ok.any():
            continue
        idx = np.clip(np.digitize(fd_hz[t, ok], fd_edges) - 1, 0, n_b - 1)
        if combine == "max":
            np.fmax.at(out[t], idx, power_db[t, ok])
        else:
            lin = np.zeros(n_b)
            cnt = np.zeros(n_b)
            np.add.at(lin, idx, 10 ** (power_db[t, ok] / 10))
            np.add.at(cnt, idx, 1)
            val = lin if combine == "sum" else lin / np.maximum(cnt, 1)
            out[t] = np.where(cnt > 0, to_db(val), np.nan)
    return out
