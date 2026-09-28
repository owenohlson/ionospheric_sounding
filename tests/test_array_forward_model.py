"""Physical forward-model checks for the array pipeline.

Unlike simulate_lwa_rdc2.py (which builds RDC2 with the same steering_vector
used for processing), these tests delay a continuous LFM sweep in *time* at each
antenna (geometric delay + cable delay) and apply the matching RF carrier phase.
They then run the real dechirp, Doppler FFT, cable correction and spatial
spectrum code. If the steering sign, the cable-correction sign, or the phase
reference frequency were wrong, the recovered direction would be wrong.

The last test documents the sweep-wrap replica that appears when the echo sits
far into the sweep (for example ~218 ms of a 500 ms sweep).
"""

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch
import importlib

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import rdc_utils as ru
import lfm_utils as lu

C = 299_792_458.0
FS, T, B, FC = 100e3, 0.5, -99.259377e3, 24.55e6


def lfm_periodic(t, bandwidth=B, period=T):
    k = bandwidth / period
    tm = np.mod(t, period)
    return np.exp(1j * 2 * np.pi * ((-bandwidth / 2) * tm + (k / 2) * tm ** 2))


def received(positions, az, el, tau0, cable_s=None, fd=0.3, nsweeps=8, snr_db=10, seed=0):
    rng = np.random.default_rng(seed)
    u = np.array([np.cos(np.deg2rad(el)) * np.sin(np.deg2rad(az)),
                  np.cos(np.deg2rad(el)) * np.cos(np.deg2rad(az)),
                  np.sin(np.deg2rad(el))])
    t = np.arange(int(round(FS * T)) * nsweeps) / FS
    geometric = -(positions @ u) / C          # antennas nearer the source receive earlier
    total = tau0 + geometric + (0.0 if cable_s is None else cable_s)
    x = np.stack([lfm_periodic(t - d) * np.exp(-2j * np.pi * FC * d) for d in total])
    x *= np.exp(2j * np.pi * fd * t)[None, :]
    noise = 10 ** (-snr_db / 20) * (rng.normal(size=x.shape) + 1j * rng.normal(size=x.shape)) / np.sqrt(2)
    return (x + noise).astype(np.complex64)


class ForwardModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        stub = types.ModuleType("tbn_utils")
        for name in ["file_timestamp_range_note", "gap_fill_note", "lsl_open_tbn", "probe_start_timestamp",
                     "lsl_print_metadata", "set_default_tbn_time_bounds"]:
            setattr(stub, name, lambda *a, **kw: None)
        with patch.dict(sys.modules, {"tbn_utils": stub}):
            cls.array = importlib.import_module("array_delay_doppler")
        rng = np.random.default_rng(11)
        cls.positions = np.column_stack([rng.uniform(-50, 50, 40), rng.uniform(-50, 50, 40), rng.normal(0, 1, 40)])
        cls.cable = rng.uniform(150e-9, 900e-9, 40)
        cls.lfm = lu.LFMWaveform(bandwidth=B, sweep_frequency=1 / T, sample_rate=FS)

    def _cube(self, x, d_min, d_max):
        rdc0 = ru.reshape_samples_to_rdc0(x, self.lfm.sweep_length)
        rdc1, delay_ms = ru.dechirp_range_fft_delay_band_rdc0(rdc0, self.lfm, d_min, d_max, window="hann")
        rdc2, _ = ru.doppler_fft_rdc1(rdc1, 1 / T, window="hann")
        return rdc2, delay_ms

    def _peak_direction(self, rdc2):
        power = np.sum(np.abs(rdc2) ** 2, axis=2)
        r, c = np.unravel_index(np.argmax(power), power.shape)
        block = rdc2[max(r - 2, 0):r + 3, max(c - 2, 0):c + 3]
        cov = self.array.estimate_spatial_covariance(block, diagonal_loading=0.01)
        az, el = np.arange(-180, 180, 1.0), np.arange(0, 91, 1.0)
        spec = self.array.spatial_spectrum_from_covariance(cov, self.positions, FC, az, el, "conventional")
        ei, ai = np.unravel_index(np.argmax(spec), spec.shape)
        return az[ai], el[ei]

    def test_time_delayed_plane_wave_is_recovered(self):
        x = received(self.positions, 137.0, 31.0, 0.00437)
        rdc2, _ = self._cube(x, 3.5, 5.5)
        az, el = self._peak_direction(rdc2)
        self.assertLessEqual(abs(az - 137.0), 2.0)
        self.assertLessEqual(abs(el - 31.0), 2.0)

    def test_cable_correction_sign_restores_direction(self):
        x = received(self.positions, -64.0, 47.0, 0.00437, cable_s=self.cable)
        rdc2, _ = self._cube(x, 3.5, 5.5)
        d = self.cable - self.cable.min()
        good = rdc2 * np.exp(2j * np.pi * FC * d)       # same as rdc_utils.cable_phase_factors
        az, el = self._peak_direction(good)
        self.assertLessEqual(abs(az + 64.0), 2.0)
        self.assertLessEqual(abs(el - 47.0), 2.0)
        bad_az, bad_el = self._peak_direction(rdc2 * np.exp(-2j * np.pi * FC * d))
        self.assertFalse(abs(bad_az + 64.0) <= 2.0 and abs(bad_el - 47.0) <= 2.0)

    def test_late_echo_produces_sweep_wrap_replica(self):
        """With |B| < fs, an echo at delay tau leaves a replica at tau + (fs-|B|)/|k|."""
        offset_ms = (FS - abs(B)) / (abs(B) / T) * 1e3
        for tau, expect_replica in ((0.21835, True), (0.002, False)):
            x = received(self.positions[:1], 0.0, 90.0, tau, fd=0.0, nsweeps=3, snr_db=60)
            rdc1 = ru.dechirp_range_fft_rdc0(ru.reshape_samples_to_rdc0(x, self.lfm.sweep_length), self.lfm, window="hann")
            delay_ms, order = lu.dechirp_delay_axis_ms(self.lfm, n_bins=rdc1.shape[0])
            p = np.mean(np.abs(rdc1[order, 1:, 0]) ** 2, axis=1)
            main = p[np.argmin(np.abs(delay_ms - tau * 1e3))]
            replica = p[np.argmin(np.abs(delay_ms - (tau * 1e3 + offset_ms)))]
            rel_db = 10 * np.log10(replica / main)
            if expect_replica:
                self.assertGreater(rel_db, -10.0)
            else:
                self.assertLess(rel_db, -60.0)


if __name__ == "__main__":
    unittest.main()
