import contextlib
import importlib
import io
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import rdc_utils as rdc
from lfm_utils import LFMWaveform


class CableCorrectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Isolate the unavailable raw TBN reader; spatial processing stays real.
        stub = types.ModuleType("tbn_utils")
        for name in ["file_timestamp_range_note", "gap_fill_note", "lsl_open_tbn",
                     "lsl_print_metadata", "set_default_tbn_time_bounds"]:
            setattr(stub, name, lambda *a, **kw: None)
        with patch.dict(sys.modules, {"tbn_utils": stub}):
            cls.array = importlib.import_module("tbn_array_delay_doppler")

    def setUp(self):
        rng = np.random.default_rng(721)
        self.positions = rng.uniform(-45, 45, (16, 3))
        self.positions[:, 2] *= 0.02
        self.delays = rng.uniform(100e-9, 700e-9, 16)
        self.fc = 24.55e6
        self.stands = np.arange(1, 17)
        self.pols = np.zeros(16, dtype=int)
        lookup = {}
        for i, sid in enumerate(self.stands):
            for pol in [0, 1]:
                antenna = types.SimpleNamespace(
                    stand=types.SimpleNamespace(id=sid, x=self.positions[i, 0],
                                               y=self.positions[i, 1], z=self.positions[i, 2]),
                    pol=pol, digitizer=2 * (15 - i) + pol + 1,
                    combined_status=33 if sid != 2 else 0,
                    cable=types.SimpleNamespace(delay=lambda f, d=self.delays[i]: d),
                )
                lookup[(int(sid), pol)] = antenna
        self.patch = patch.object(rdc, "get_lwasv_antennas", return_value=lookup)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_digitizer_mapping_and_legacy_cache(self):
        indices, stands, pols = rdc.channel_selection([1, 16], "both")
        np.testing.assert_array_equal(indices, [30, 31, 0, 1])
        meta = {}
        actual_s, actual_p = rdc.cached_physical_channel_ids([16, 16, 1, 1], [0, 1, 0, 1], meta)
        np.testing.assert_array_equal(actual_s, stands)
        np.testing.assert_array_equal(actual_p, pols)
        self.assertEqual(meta["channel_id_kind"], "physical_stand")

    def test_policy_and_operational_filter(self):
        args = types.SimpleNamespace(cable_correction="auto")
        with contextlib.redirect_stdout(io.StringIO()):
            factors = self.array.array_cable_factors(args, {"synthetic": True}, self.stands, self.pols, self.fc)
            np.testing.assert_array_equal(factors, np.ones(16))
            for mode in ["auto", "on"]:
                args.cable_correction = mode
                factors = self.array.array_cable_factors(args, {"cable_phase_corrected": True}, self.stands, self.pols, self.fc)
                np.testing.assert_array_equal(factors, np.ones(16))
            args.cable_correction = "off"
            factors = self.array.array_cable_factors(args, {}, self.stands, self.pols, self.fc)
            np.testing.assert_array_equal(factors, np.ones(16))
        idx, *_ = self.array.select_array_channels(self.stands, self.pols, "x", drop_bad=True,
                                                  operational_only=True, require_positions=False)
        self.assertNotIn(2, self.stands[idx])
        self.assertFalse(set(self.stands[idx]) & set(rdc.BAD_LWA_STANDS))

    def test_delayed_source_restores_conventional_and_mvdr_direction(self):
        rng = np.random.default_rng(17)
        steering = rdc.steering_vector(self.positions, 299792458.0 / self.fc, 80, 40)
        source = np.exp(1j * rng.uniform(-np.pi, np.pi, 256))
        noise = .02 * (rng.normal(size=(256, 16)) + 1j * rng.normal(size=(256, 16)))
        ideal = source[:, None] * steering + noise
        # Independent forward model: actual cable phase is negative RF phase.
        delayed = ideal * np.exp(-2j * np.pi * self.fc * self.delays)
        args = types.SimpleNamespace(cable_correction="auto")
        with contextlib.redirect_stdout(io.StringIO()):
            factors = self.array.array_cable_factors(args, {"synthetic": True, "cable_delays_injected": True},
                                                     self.stands, self.pols, self.fc)
        corrected = delayed * factors
        global_phase = np.exp(-2j * np.pi * self.fc * self.delays.min())
        np.testing.assert_allclose(corrected, ideal * global_phase, atol=2e-14)
        self.assertGreater(np.mean(abs(delayed / global_phase - ideal)**2), .1)
        reference = self.array.estimate_spatial_covariance(ideal[None], diagonal_loading=.1)
        recovered = self.array.estimate_spatial_covariance(corrected[None], diagonal_loading=.1)
        np.testing.assert_allclose(recovered, reference, atol=1e-13)
        az, el = np.arange(0, 360, 10), np.arange(10, 90, 10)
        for method in ["conventional", "mvdr"]:
            spectrum = self.array.spatial_spectrum_from_covariance(recovered, self.positions, self.fc, az, el, method)
            ei, ai = np.unravel_index(spectrum.argmax(), spectrum.shape)
            self.assertEqual((az[ai], el[ei]), (80, 40))

    def test_correction_commutes_with_range_and_doppler_ffts(self):
        rng = np.random.default_rng(7)
        raw = (rng.normal(size=(64, 8, 16)) + 1j*rng.normal(size=(64, 8, 16))).astype(np.complex64)
        cfg = LFMWaveform(-40, 1, 64)
        factors = rdc.cable_phase_factors(self.stands, self.pols, self.fc).astype(np.complex64)
        def process(x):
            r1 = rdc.dechirp_range_fft_rdc0(x, cfg, window="hann")
            return rdc.doppler_fft_rdc1(r1, 1, window="hann", nfft=16)[0]
        np.testing.assert_allclose(process(raw * factors), process(raw) * factors, rtol=3e-5, atol=2e-5)
        chunks = np.concatenate([process(raw[..., i:i+5]) * factors[i:i+5] for i in range(0, 16, 5)], axis=-1)
        np.testing.assert_allclose(chunks, process(raw) * factors, atol=1e-6)

    def test_cached_frame_correction_preserves_order_and_fills_only_gaps(self):
        raw = np.ones((1, 1, 2, 16), dtype=np.complex64)
        raw[0, 0, 0, 3] = np.nan
        loaded = dict(metadata={"tuning_frequency_hz": self.fc}, shape=raw.shape,
                      delay_ms=np.array([1.0]), doppler_hz=np.array([0.0, 0.1]),
                      channel_stands=self.stands, channel_pols=self.pols,
                      frame_tstart_s=[0], frame_tend_s=[1])
        args = types.SimpleNamespace(rdc2_input="test.h5", sweep_frequency=None,
                                     bandwidth=None, d_min=None, d_max=None, fd_min=None,
                                     fd_max=None, pol="x", drop_bad_stands=False,
                                     channel_chunk_size=5, gap_fill="zero",
                                     carrier_frequency=None, cable_correction="auto")
        chunks = [(i, min(i+5, 16), raw[..., i:i+5].copy()) for i in range(0, 16, 5)]
        with patch.object(self.array, "prepare_rdc2_view", return_value=loaded), \
             patch.object(self.array, "iter_rdc2_view_chunks", return_value=iter(chunks)), \
             contextlib.redirect_stdout(io.StringIO()):
            result = self.array.load_rdc2_frame_for_array(args, require_positions=False)
        expected = np.nan_to_num(raw[0]) * rdc.cable_phase_factors(self.stands, self.pols, self.fc)
        np.testing.assert_allclose(result["rdc2"], expected, atol=1e-7)
        self.assertEqual(result["nonfinite_values_filled"], 1)
        self.assertEqual(np.count_nonzero(result["rdc2"]), 31)
        self.assertTrue(np.isnan(raw[0, 0, 0, 3]))


if __name__ == "__main__":
    unittest.main()
