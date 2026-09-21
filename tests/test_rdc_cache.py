import contextlib
import importlib.util
import io
import json
import shutil
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from lfm_utils import LFMWaveform
from rdc_utils import load_rdc2_hdf5_metadata, iter_rdc2_hdf5_channel_chunks
from rdc_cache import prepare_rdc2_view, iter_rdc2_view_chunks
from plotting_utils import plot_dechirp_profile


class CacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        stub = types.ModuleType("tbn_utils")
        for name in ("lsl_open_tbn", "lsl_print_metadata", "probe_start_timestamp", "set_default_tbn_time_bounds"):
            setattr(stub, name, lambda *a, **kw: None)
        cls.tbn_stub = stub
        spec = importlib.util.spec_from_file_location("export_under_test", Path(__file__).resolve().parents[1] / "tools/rdc_export.py")
        cls.exporter = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"tbn_utils": stub}):
            spec.loader.exec_module(cls.exporter)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.source = Path(self.tmp.name) / "rdc0.h5"
        self.output = Path(self.tmp.name) / "rdc2.h5"
        rng = np.random.default_rng(812)
        self.data = (rng.normal(size=(16, 20, 2)) + 1j * rng.normal(size=(16, 20, 2))).astype(np.complex64)
        self.metadata = dict(sample_rate_hz=64., sweep_frequency_hz=4., adc_start=0,
                             channel_id_kind="physical_stand", tuning_frequency_hz=24.55e6,
                             synthetic=True, start_offset_samples=0)
        with h5py.File(self.source, "w") as h5:
            group = h5.create_group("rdc0")
            group["real"], group["imag"] = self.data.real, self.data.imag
            group["channel_stands"], group["channel_pols"] = [1, 2], [0, 0]
            h5.attrs["metadata_json"] = json.dumps(self.metadata)

    def export(self, *extra, raw=False):
        argv = ["export", "--output", str(self.output), "--bandwidth", "64",
                "--d-min", "0", "--d-max", "250", "--compression", "none",
                "--channel-chunk-size", "1"]
        argv += ["recording.tbn", "--sweep-frequency", "4"] if raw else ["--rdc0-input", str(self.source)]
        with patch.object(sys, "argv", argv + list(extra)), contextlib.redirect_stdout(io.StringIO()):
            self.exporter.main()

    def args(self, **kw):
        defaults = dict(rdc2_input=str(self.output), integration_time=None, hop_time=None,
                        tstart=None, tend=None, duration=None, nfft_doppler=None,
                        slow_window="hann", gap_fill="nan")
        return types.SimpleNamespace(**dict(defaults, **kw))

    def test_default_3d_and_legacy_4d_read_identically(self):
        self.export()
        loaded = load_rdc2_hdf5_metadata(self.output)
        self.assertEqual(len(loaded["storage_shape"]), 3)
        self.assertTrue(loaded["has_rdc1"])
        actual = np.concatenate([c for _, _, c in iter_rdc2_hdf5_channel_chunks(self.output, 1)], axis=-1)
        with h5py.File(self.output, "r+") as h5:
            # Model an existing cache that only retained windowed RDC2.
            del h5["rdc1"]
            for component in ("real", "imag"):
                values = h5["rdc2"][component][...]
                del h5["rdc2"][component]
                h5["rdc2"][component] = values[None]
        legacy = prepare_rdc2_view(self.args())
        expected = np.concatenate([c for _, _, c in iter_rdc2_view_chunks(legacy, 1)], axis=-1)
        np.testing.assert_array_equal(actual, expected)
        self.assertFalse(legacy["has_rdc1"])
        self.assertFalse(prepare_rdc2_view(self.args(integration_time=5, hop_time=5))["recompute"])
        with self.assertRaisesRegex(ValueError, "no retained RDC1"):
            prepare_rdc2_view(self.args(integration_time=1, hop_time=.5))

    def test_later_windows_match_direct_fft_and_preserve_channels(self):
        # Export a narrow Doppler band, then verify retained RDC1 can recreate
        # all Doppler bins and use a different integration/hop.
        self.export("--fd-min", "-.2", "--fd-max", ".2")
        view = prepare_rdc2_view(self.args(integration_time=1.5, hop_time=.5, tstart=.5, tend=4.5))
        np.testing.assert_array_equal(view["frame_tstart_s"], [.5, 1, 1.5, 2, 2.5, 3])
        self.assertEqual(view["shape"][2], 8)
        chunks = list(iter_rdc2_view_chunks(view, 1, channel_indices=[1], delay_selection=slice(1, 5)))
        actual = chunks[0][2]
        with h5py.File(self.output) as h5:
            data = h5["rdc1/real"][...] + 1j * h5["rdc1/imag"][...]
        window = np.hanning(6)
        expected = np.stack([np.fft.fftshift(np.fft.fft(
            data[1:5, first:first + 6, 1:2] * (window / window.mean())[None, :, None],
            n=8, axis=1), axes=1) for first in [2, 4, 6, 8, 10, 12]])
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-5)
        with self.assertRaisesRegex(ValueError, "nfft-doppler"):
            prepare_rdc2_view(self.args(integration_time=2, nfft_doppler=4))
        with self.assertRaisesRegex(ValueError, "outside"):
            prepare_rdc2_view(self.args(tstart=-1))

    def test_explicit_export_timing_and_subselection(self):
        self.export("--integration-time", "1", "--hop-time", ".5", "--tstart", "1", "--tend", "4")
        loaded = load_rdc2_hdf5_metadata(self.output)
        self.assertEqual(len(loaded["storage_shape"]), 4)
        np.testing.assert_array_equal(loaded["frame_tstart_s"], [1, 1.5, 2, 2.5, 3])
        with h5py.File(self.output) as h5:
            self.assertEqual(h5["rdc1/real"].shape[1], 12)
        stored = np.concatenate([c for _, _, c in iter_rdc2_hdf5_channel_chunks(self.output)], axis=-1)
        view = prepare_rdc2_view(self.args(integration_time=1, hop_time=.5))
        rebuilt = np.concatenate([c for _, _, c in iter_rdc2_view_chunks(view)], axis=-1)
        np.testing.assert_allclose(rebuilt, stored, rtol=2e-6, atol=2e-5)

    def test_raw_nonzero_start_is_applied_once(self):
        fake = types.SimpleNamespace(get_info=lambda key: 64., close=lambda: None)
        # Extraction returns an already trimmed temporary RDC0.
        with patch.object(self.tbn_stub, "lsl_open_tbn", return_value=fake), \
             patch.dict(sys.modules, {"tbn_utils": self.tbn_stub}), \
             patch.object(self.exporter, "export_raw_rdc0",
                          side_effect=lambda args, fs, output, **kw: shutil.copyfile(self.source, output)):
            self.export("--tstart", "10", "--tend", "15", raw=True)
        loaded = load_rdc2_hdf5_metadata(self.output)
        np.testing.assert_array_equal(loaded["frame_tstart_s"], [10])
        np.testing.assert_array_equal(loaded["frame_tend_s"], [15])
        with h5py.File(self.output) as h5:
            self.assertEqual(h5["rdc1/real"].shape[1], 20)

    def test_array_plot_commands_accept_new_and_legacy_caches(self):
        import importlib
        stub = types.ModuleType("tbn_utils")
        for name in ("file_timestamp_range_note", "gap_fill_note", "lsl_open_tbn", "probe_start_timestamp",
                     "lsl_print_metadata", "set_default_tbn_time_bounds"):
            setattr(stub, name, lambda *a, **kw: None)
        with patch.dict(sys.modules, {"tbn_utils": stub}):
            array = importlib.import_module("array_delay_doppler")
            sliding = importlib.import_module("sliding_array_delay_doppler")
        self.export()
        image = Path(self.tmp.name) / "array.png"
        common = ["--rdc2-input", str(self.output), "--beamformer", "incoherent",
                  "--cable-correction", "off"]
        argv = ["array", *common, "--output", str(image), "--integration-time", "1",
                "--hop-time", ".5", "--rdc2-frame-index", "2"]
        with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
            array.main()
        self.assertGreater(image.stat().st_size, 1000)
        for legacy in (False, True):
            if legacy:
                self.export("--integration-time", "1", "--hop-time", ".5", "--no-store-rdc1")
            directory = Path(self.tmp.name) / ("legacy" if legacy else "flexible")
            argv = ["sliding", *common, "--output", str(directory), "--max-frames", "2", "--keep-frames"]
            if not legacy:
                argv += ["--integration-time", "1", "--hop-time", ".5"]
            with patch.object(sys, "argv", argv), patch.object(sliding, "make_video", return_value=False), \
                 contextlib.redirect_stdout(io.StringIO()):
                sliding.main()
            self.assertEqual(len(list(directory.glob("frame_*.png"))), 2)

    def test_profile_averages_power_without_phase_cancellation(self):
        spectra = np.array([[[1, 2], [-3, -4], [np.nan, np.nan]],
                            [[2, 1], [-4, -3], [np.nan, np.nan]]], dtype=complex)
        output = Path(self.tmp.name) / "profile.png"
        ax = plot_dechirp_profile([1, 2], spectra, LFMWaveform(64, 4, 64), output_file=output)
        np.testing.assert_allclose(ax.lines[0].get_ydata(), 10 * np.log10([15, 15]))
        self.assertGreater(output.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
