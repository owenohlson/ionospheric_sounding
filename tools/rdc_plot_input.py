"""Source-independent HDF5 access for the non-array plots."""

from datetime import datetime, timedelta
import os
import warnings

import h5py
import numpy as np

from lfm_utils import LFMWaveform, dechirp_delay_axis_ms
from rdc_utils import (dechirp_range_fft_delay_band_rdc0, load_rdc0_hdf5_metadata,
                       load_rdc1_hdf5_metadata, load_rdc2_hdf5_metadata)


class PlotInput:
    def __init__(self, args, prefer_doppler=False):
        self.path = os.path.expanduser(args.input_file)
        self.args = args
        with h5py.File(self.path, "r") as h5:
            self.stage = 2 if prefer_doppler and "rdc2" in h5 else (1 if "rdc1" in h5 else 0)
            if f"rdc{self.stage}" not in h5:
                raise ValueError("This plot needs RDC0 or per-sweep RDC1; export RDC2 without --no-store-rdc1")
        loader = (load_rdc0_hdf5_metadata, load_rdc1_hdf5_metadata, load_rdc2_hdf5_metadata)[self.stage]
        self.loaded = loader(self.path)
        self.meta = self.loaded["metadata"]
        self.shape = self.loaded["shape"]
        self.prf = self.setting("sweep_frequency", "sweep_frequency_hz", required=True)
        self.fs = float(self.meta["sample_rate_hz"])
        bandwidth = self.setting("bandwidth", "bandwidth_hz", required=args.kind != "spectrogram")
        window = self.setting("dechirp_window", "dechirp_window", default="hann", fixed=self.stage > 0)
        self.window = window
        self.config = LFMWaveform(sample_rate=self.fs, sweep_frequency=self.prf, bandwidth=bandwidth or 1.,
            reference_gate_frequency=self.setting("reference_gate_frequency", "reference_gate_frequency", fixed=self.stage > 0),
            reference_gate_duty=self.setting("reference_gate_duty", "reference_gate_duty", default=.5, fixed=self.stage > 0),
            reference_gate_phase=self.setting("reference_gate_phase", "reference_gate_phase", default=0., fixed=self.stage > 0))
        self.gap_fill = args.gap_fill or self.meta.get("gap_fill", "nan")
        stands, pols = self.loaded["channel_stands"], self.loaded["channel_pols"]
        self.stand = int(stands[0]) if args.stand is None else args.stand
        mask = stands == self.stand
        if getattr(args, "plot", None) == "average" and args.stand is None:
            mask = np.ones(stands.shape, dtype=bool)
        if args.pol != "both":
            mask &= pols == (0 if args.pol in ("x", "0") else 1)
        self.channels = np.flatnonzero(mask)
        if not len(self.channels):
            raise ValueError("Requested stand/polarization is absent from this cache; WAV polarization is unspecified")
        self.pols = pols[self.channels]
        self.epoch = datetime.fromisoformat(self.meta["pulse0_utc"]) if self.meta.get("pulse0_utc") else None
        if self.stage == 2:
            starts = np.asarray(self.loaded["frame_tstart_s"])
            self.origin = float(starts[0])
            self.starts = starts - self.origin
            self.ends = np.asarray(self.loaded["frame_tend_s"]) - self.origin
            self.duration = float(self.ends[-1])
        else:
            self.duration = self.shape[1] / self.prf
        self.start = 0. if args.tstart is None else args.tstart
        self.stop = self.duration if args.tend is None else args.tend
        if not 0 <= self.start < self.stop <= self.duration + 1e-8:
            raise ValueError(f"Time bounds must lie within this cache's {self.duration:g} seconds")
        self.first = int(np.ceil(self.start * self.prf - 1e-8))
        self.last = int(np.floor(self.stop * self.prf + 1e-8))
        self.delay = (dechirp_delay_axis_ms(self.config, n_bins=self.shape[0])[0] if self.stage == 0
                      else np.asarray(self.loaded["delay_ms"]))
        self.delay_slice = slice(None)
        self.warned = False

    def setting(self, flag, key, default=None, required=False, fixed=True):
        requested = getattr(self.args, flag, None)
        stored = self.meta.get(key)
        if fixed and requested is not None and stored is not None:
            equal = np.isclose(requested, stored, rtol=1e-10, atol=1e-12) if isinstance(requested, (float, int)) else requested == stored
            if not equal:
                raise ValueError(f"--{flag.replace('_', '-')} conflicts with cached {key}={stored}; re-export first")
        value = requested if requested is not None else (stored if stored is not None else default)
        if fixed and self.stage > 0 and requested is not None and stored is None and requested != default:
            raise ValueError(f"Cached {key} is unspecified; cannot apply a processing override to an already processed cube")
        if required and value is None:
            raise ValueError(f"Missing {key}; supply --{flag.replace('_', '-')}")
        return value

    def select_delay(self):
        args = self.args
        lo, hi = args.d_min, args.d_max
        if lo is None and hi is None and not args.full_delay:
            if self.stage == 2:
                frames = np.flatnonzero((self.starts >= self.start - 1e-8) & (self.ends <= self.stop + 1e-8))
                if not frames.size:
                    raise ValueError("No saved RDC2 frame inside the requested time range")
                cube, _ = self.doppler(int(frames[0]))
                power = np.sum(np.abs(cube) ** 2, axis=(1, 2))
            else:
                cube, _ = self.spectra(self.first, min(self.last, self.first + args.auto_delay_sweeps))
                valid = np.isfinite(cube)
                power = np.sum(np.where(valid, np.abs(cube) ** 2, 0), axis=(0, 1))
            if not np.any(np.isfinite(power) & (power > 0)):
                raise ValueError("No finite nonzero preview signal; specify delay bounds or use --full-delay")
            peak = self.delay[np.nanargmax(power)]
            lo, hi = peak - args.auto_delay_half_width, peak + args.auto_delay_half_width
            print(f"Auto delay: {lo:.6f} to {hi:.6f} ms (peak {peak:.6f} ms)")
        mask = (self.delay >= (lo if lo is not None else -np.inf)) & (self.delay <= (hi if hi is not None else np.inf))
        idx = np.flatnonzero(mask)
        if not idx.size:
            raise ValueError("No stored delay bins within --d-min/--d-max")
        self.delay_slice = slice(int(idx[0]), int(idx[-1]) + 1)
        self.delay = self.delay[mask]

    def clean(self, cube):
        bad = ~np.isfinite(cube)
        if self.gap_fill == "zero" and np.any(bad):
            if self.stage > 0 and not self.warned:
                warnings.warn("Replacing nonfinite processed bins with zero cannot recover NaN-poisoned FFTs. Re-export from RDC0 with --gap-fill=zero for recovery.")
                self.warned = True
            cube[bad] = 0
        return cube, np.any(bad, axis=(0, 2))

    def spectra(self, first, last):
        """Return (channels, sweeps, delay); limit raw working memory to 32 sweeps."""
        if last <= first:
            raise ValueError("Selected span contains no complete sweeps")
        blocks, gaps = [], []
        with h5py.File(self.path, "r") as h5:
            group = h5[f"rdc{self.stage}"]
            for p in range(first, last, 32):
                q = min(p + 32, last)
                band = slice(None) if self.stage == 0 else self.delay_slice
                cube = group["real"][band, p:q, self.channels] + 1j * group["imag"][band, p:q, self.channels]
                cube, bad = self.clean(cube)
                if "invalid_counts" in group:
                    bad |= np.any(group["invalid_counts"][p:q, self.channels] > 0, axis=1)
                if self.stage == 0:
                    cube, _ = dechirp_range_fft_delay_band_rdc0(
                        cube, self.config, self.delay[0], self.delay[-1], window=self.window,
                        adc_start=self.meta.get("adc_start", 0))
                blocks.append(cube.transpose(2, 1, 0))
                gaps.append(bad)
        return np.concatenate(blocks, axis=1), np.concatenate(gaps)

    def doppler(self, frame):
        with h5py.File(self.path, "r") as h5:
            grp = h5["rdc2"]
            select = (self.delay_slice, slice(None), self.channels)
            if grp["real"].ndim == 4:
                select = (frame,) + select
            cube = grp["real"][select] + 1j * grp["imag"][select]
            cube, bad = self.clean(cube)
            if "invalid_counts" in grp:
                bad = np.r_[bad, grp["invalid_counts"][frame, self.channels] > 0]
            elif "rdc1/invalid_counts" in h5:
                first = int(round(self.starts[frame] * self.prf))
                last = int(round(self.ends[frame] * self.prf))
                bad = np.r_[bad, h5["rdc1/invalid_counts"][first:last, self.channels].ravel() > 0]
            return cube, bad

    def annotation(self, start, stop, gaps):
        fc = self.meta.get("tuning_frequency_hz")
        receiver = self.meta.get("channel_id_kind") == "receiver"
        rows = [("File", os.path.basename(self.path)), ("Input", f"RDC{self.stage}"),
                ("Tuning freq", "unknown" if fc is None else f"{fc / 1e6:.6f} MHz"),
                ("Receiver" if receiver else "Stand/pol", "WAV I/Q" if receiver else f"{self.stand}/{self.pols.tolist()}"),
                ("Plot span", f"{start:.3f}-{stop:.3f} s"), ("Dechirp window", self.window),
                ("Gap fill", self.gap_fill)]
        if self.args.kind == "spectrogram":
            rows.insert(2, ("Sample rate", f"{self.fs / 1000:g} kHz"))
        corner = None
        if self.epoch is not None:
            corner = "UTC start: " + (self.epoch + timedelta(seconds=start)).isoformat() + "\nUTC end: " + (self.epoch + timedelta(seconds=stop)).isoformat()
        note = f"* {'Zeros' if self.gap_fill == 'zero' else 'NaNs'} used for gaps/corrupted data in this view." if np.any(gaps) else None
        return dict(info_rows=rows, corner_note=corner, info_panel_footer_note=note)
