"""RDC interoperability tests using small on-disk HDF5 cubes."""

import contextlib
from datetime import datetime, timedelta
import io
from pathlib import Path
import runpy
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import rdc_export
from lfm_utils import LFMWaveform
from rdc_utils import (
    dechirp_range_fft_delay_band_rdc0, export_tbn_rdc0_hdf5,
    iter_rdc1_hdf5_channel_chunks, iter_rdc2_hdf5_channel_chunks,
    load_rdc0_hdf5_metadata, load_rdc1_hdf5_metadata, load_rdc2_hdf5_metadata,
)


class RDC1ExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.raw = self.root / "raw.h5"
        self.range = self.root / "range.h5"
        rng = np.random.default_rng(19)
        data = (rng.normal(size=(4, 320)) + 1j * rng.normal(size=(4, 320))).astype(np.complex64)

        class Reader:
            closed = False

            def close(self):
                self.closed = True

            def get_info(self, key):
                return {"sample_rate": 64., "freq1": 25.6e6}[key]

            def offset(self, value):
                self.index = int(value * 64)

            def read(self, seconds):
                n = int(round(seconds * 64))
                timestamp = datetime(2023, 12, 9) + timedelta(seconds=self.index / 64)
                chunk = data[:, self.index:self.index + n]
                self.index += n
                return n / 64, timestamp, chunk

        self.reader_type = Reader
        # Exercise the actual RDC0 writer; only the telescope reader/mapping is fake.
        with patch("rdc_utils.channel_selection", return_value=(np.arange(4), np.array([1, 1, 2, 2]), np.array([0, 1, 0, 1]))), \
             contextlib.redirect_stdout(io.StringIO()):
            export_tbn_rdc0_hdf5(Reader(), str(self.raw), 0, 5, 4, [1, 2], "both",
                                 adc_start=2, adc_count=12, compression="none", gap_fill="zero")

    def export(self, output, *options, stage=2):
        chunk_options = ["--channel-chunk-size", "1"] if stage != 0 else []
        with contextlib.redirect_stdout(io.StringIO()):
            rdc_export.main(["--output", str(output), "--compression", "none",
                             *chunk_options, *map(str, options)], stage=stage)

    @contextlib.contextmanager
    def raw_environment(self):
        readers = []

        def open_reader(filename):
            reader = self.reader_type()
            readers.append(reader)
            return reader

        def set_bounds(args, reader):
            if args.tstart is None:
                args.tstart = 0.
            if args.tend is None:
                args.tend = 5.

        stub = types.ModuleType("tbn_utils")
        stub.lsl_open_tbn = open_reader
        stub.lsl_print_metadata = lambda reader: None
        stub.set_default_tbn_time_bounds = set_bounds
        stub.probe_start_timestamp = lambda filename, tstart: datetime(2023, 12, 9, microsecond=125000)
        mapping = (np.arange(4), np.array([1, 1, 2, 2]), np.array([0, 1, 0, 1]))
        with patch.dict(sys.modules, {"tbn_utils": stub}), \
             patch("rdc_utils.channel_selection", return_value=mapping), \
             patch("rdc_export.channel_selection", return_value=mapping):
            yield readers

    def test_shared_raw_extraction_matches_direct_and_chained_stages(self):
        common = ["recording.tbn", "--sweep-frequency", "4", "--no-timestamp-align",
                  "--adc-start", "2", "--adc-count", "12", "--gap-fill", "zero"]
        exported = self.root / "stage0.h5"
        with self.raw_environment() as readers:
            self.export(exported, *common, stage=0)
            self.export(self.range, *common, "--bandwidth=-64", stage=1)
            direct = self.root / "direct.h5"
            self.export(direct, *common, "--bandwidth=-64", stage=2)
        self.assertTrue(all(reader.closed for reader in readers))
        with h5py.File(exported) as actual, h5py.File(self.raw) as expected:
            np.testing.assert_array_equal(actual["rdc0/real"][...], expected["rdc0/real"][...])
            np.testing.assert_array_equal(actual["rdc0/imag"][...], expected["rdc0/imag"][...])
        chained = self.root / "chained.h5"
        self.export(chained, "--rdc1-input", self.range)
        np.testing.assert_allclose(self.cube(direct), self.cube(chained))

    def test_stage_zero_alignment_dry_run_and_main_stage_selection(self):
        output = self.root / "aligned.h5"
        argv = ["--stage", "0", "recording.tbn", "--output", str(output), "--sweep-frequency", "4"]
        with self.raw_environment() as readers, contextlib.redirect_stdout(io.StringIO()):
            rdc_export.main(argv + ["--dry-run-size"])
            self.assertFalse(output.exists())
            rdc_export.main(argv)
        self.assertTrue(all(reader.closed for reader in readers))
        loaded = load_rdc0_hdf5_metadata(output)
        self.assertEqual(loaded["shape"], (16, 19, 4))
        self.assertEqual(loaded["metadata"]["start_offset_samples"], 8)
        self.assertEqual(loaded["metadata"]["tuning_frequency_hz"], 25.6e6)
        for stage in (0, 1, 2):
            with patch("rdc_export.main") as entry:
                runpy.run_path(str(Path(rdc_export.__file__).with_name(f"tbn_export_rdc{stage}.py")), run_name="__main__")
                entry.assert_called_once_with(stage=stage)

    def test_raw_failure_closes_readers_and_cleans_intermediate_cache(self):
        output = self.root / "existing.h5"
        output.write_bytes(b"keep")
        common = ["recording.tbn", "--sweep-frequency", "4", "--no-timestamp-align"]
        with self.raw_environment() as readers, \
             patch("rdc_export.export_tbn_rdc0_hdf5", side_effect=RuntimeError("read failed")), \
             self.assertRaisesRegex(RuntimeError, "read failed"):
            self.export(output, *common, stage=0)
        self.assertTrue(all(reader.closed for reader in readers))
        self.assertEqual(output.read_bytes(), b"keep")
        self.assertEqual(list(self.root.glob("existing.h5.*.tmp")), [])

        directories = []
        temporary_directory = tempfile.TemporaryDirectory

        def track_directory(**kwargs):
            directory = temporary_directory(dir=self.root, **kwargs)
            directories.append(Path(directory.name))
            return directory

        with self.raw_environment() as readers, \
             patch("rdc_export.tempfile.TemporaryDirectory", side_effect=track_directory), \
             patch("rdc_export.doppler_fft_rdc1", side_effect=RuntimeError("FFT failed")), \
             self.assertRaisesRegex(RuntimeError, "FFT failed"):
            self.export(output, *common, "--bandwidth=-64")
        self.assertTrue(all(reader.closed for reader in readers))
        self.assertTrue(directories)
        self.assertTrue(all(not directory.exists() for directory in directories))
        self.assertEqual(output.read_bytes(), b"keep")

    def make_range(self, *options):
        self.export(self.range, "--rdc0-input", self.raw, "--bandwidth=-64",
                    "--dechirp-window", "hamming", *options, stage=1)

    def cube(self, path):
        return np.concatenate([x for _, _, x in iter_rdc2_hdf5_channel_chunks(path, 1)], axis=-1)

    def test_rdc0_rdc1_rdc2_matches_direct_with_metadata(self):
        self.make_range("--pol", "y")
        r0 = load_rdc0_hdf5_metadata(self.raw)
        r1 = load_rdc1_hdf5_metadata(self.range)
        self.assertEqual(r0["metadata"]["rdc_stage"], 0)
        self.assertEqual(r1["metadata"]["rdc_stage"], 1)
        self.assertEqual(r1["metadata"]["pol"], "y")
        self.assertNotIn("nfft_doppler", r1["metadata"])
        np.testing.assert_array_equal(r1["channel_stands"], [1, 2])
        np.testing.assert_array_equal(r1["channel_pols"], [1, 1])
        self.assertEqual(r1["metadata"]["channel_digitizers"], [2, 4])
        # No bandwidth, window, PRF, or ADC flags are needed for RDC1 input.
        chained, direct = self.root / "chained.h5", self.root / "direct.h5"
        timing = ["--integration-time", "1", "--hop-time", ".5", "--nfft-doppler", "8"]
        self.export(chained, "--rdc1-input", self.range, *timing)
        self.export(direct, "--rdc0-input", self.raw, "--bandwidth=-64",
                    "--dechirp-window", "hamming", "--pol", "y", *timing)
        np.testing.assert_allclose(self.cube(chained), self.cube(direct), rtol=1e-6, atol=1e-6)
        r2 = load_rdc2_hdf5_metadata(chained)
        for key in ("tuning_frequency_hz", "sample_rate_hz", "sweep_frequency_hz", "bandwidth_hz",
                    "dechirp_window", "adc_start", "adc_count", "channel_digitizers", "cable_phase_corrected"):
            self.assertEqual(r2["metadata"][key], r1["metadata"][key])
        self.assertEqual(r2["metadata"]["tuning_frequency_hz"], 25.6e6)
        np.testing.assert_array_equal(r2["channel_pols"], [1, 1])

    def test_delay_channel_time_subselection_and_retained_rdc1(self):
        self.make_range("--tstart", "1", "--tend", "4")
        r1 = load_rdc1_hdf5_metadata(self.range)
        lo, hi = r1["delay_ms"][[2, 5]]
        result = self.root / "subset.h5"
        self.export(result, "--rdc1-input", self.range, "--tstart", ".5", "--duration", "1",
                    "--d-min", lo, "--d-max", hi, "--stands", "2", "--pol", "x")
        r2 = load_rdc2_hdf5_metadata(result)
        np.testing.assert_array_equal(r2["frame_tstart_s"], [1.5])
        np.testing.assert_array_equal(r2["frame_tend_s"], [2.5])
        self.assertEqual(r2["metadata"]["channel_digitizers"], [3])
        np.testing.assert_array_equal(r2["delay_ms"], r1["delay_ms"][2:6])
        # The same reader accepts retained RDC1 in older/newer RDC2 files.
        retained = load_rdc1_hdf5_metadata(result)
        self.assertEqual(retained["shape"], (4, 4, 1))
        again = self.root / "again.h5"
        self.export(again, "--rdc1-input", result)
        np.testing.assert_allclose(self.cube(again), self.cube(result))

    def test_zero_fill_before_range_fft_and_no_second_dechirp(self):
        with h5py.File(self.raw, "r+") as h5:
            h5["rdc0/real"][3, 5, 0] = np.nan
        self.make_range("--gap-fill", "zero")
        r1 = load_rdc1_hdf5_metadata(self.range)
        data = np.concatenate([x for _, _, x in iter_rdc1_hdf5_channel_chunks(self.range, 1)], axis=-1)
        self.assertTrue(np.all(np.isfinite(data)))
        self.assertEqual(r1["metadata"]["nonfinite_rdc0_values_filled"], 1)
        with h5py.File(self.raw) as h5:
            raw = h5["rdc0/real"][...] + 1j * h5["rdc0/imag"][...]
        raw[~np.isfinite(raw)] = 0
        expected, _ = dechirp_range_fft_delay_band_rdc0(
            raw, LFMWaveform(-64, 4, 64), r1["delay_ms"][0], r1["delay_ms"][-1],
            window="hamming", adc_start=2,
        )
        np.testing.assert_allclose(data, expected)
        with h5py.File(self.range, "r+") as h5:
            h5["rdc1/real"][0, 0, 0] = np.nan
        output = self.root / "zeros.h5"
        with patch.object(rdc_export, "dechirp_range_fft_delay_band_rdc0", side_effect=AssertionError("double FFT")):
            self.export(output, "--rdc1-input", self.range, "--gap-fill", "zero")
        self.assertTrue(np.all(np.isfinite(self.cube(output))))
        self.assertEqual(load_rdc2_hdf5_metadata(output)["metadata"]["nonfinite_rdc1_values_filled"], 1)

    def test_conflicting_processing_flags_are_rejected(self):
        self.make_range()
        for flag, value in [("--bandwidth", "32"), ("--sweep-frequency", "2"),
                            ("--dechirp-window", "hann"), ("--adc-start", "0"),
                            ("--reference-gate-frequency", "10")]:
            with self.subTest(flag=flag), self.assertRaisesRegex(ValueError, "conflicts"):
                self.export(self.root / "bad.h5", "--rdc1-input", self.range, flag, value)
        with self.assertRaisesRegex(ValueError, "conflicts"):
            self.export(self.root / "bad.h5", "--rdc0-input", self.raw, "--bandwidth=-64", "--sweep-frequency", "2")

    def test_gate_metadata_storage_and_dry_run(self):
        self.make_range("--reference-gate-period", ".1", "--reference-gate-duty", ".3",
                        "--reference-gate-phase", ".01", "--dtype", "float16", "--compression", "gzip")
        meta = load_rdc1_hdf5_metadata(self.range)["metadata"]
        output = self.root / "gate.h5"
        self.export(output, "--rdc1-input", self.range, "--reference-gate-frequency", "10")
        out = load_rdc2_hdf5_metadata(output)["metadata"]
        for key in ("reference_gate_frequency", "reference_gate_duty", "reference_gate_phase"):
            self.assertEqual(meta[key], out[key])
        with h5py.File(self.range) as h5:
            self.assertEqual(h5["rdc1/real"].dtype, np.dtype("float16"))
            self.assertEqual(h5["rdc1/real"].compression, "gzip")
        dry = self.root / "dry.h5"
        self.export(dry, "--rdc0-input", self.raw, "--bandwidth=-64", "--dry-run-size", stage=1)
        self.assertFalse(dry.exists())

    def test_invalid_metadata_and_failed_output_preserve_existing_file(self):
        self.make_range()
        output = self.root / "existing.h5"
        output.write_bytes(b"keep this output")
        with patch.object(rdc_export, "doppler_fft_rdc1", side_effect=RuntimeError("test failure")), \
             self.assertRaisesRegex(RuntimeError, "test failure"):
            self.export(output, "--rdc1-input", self.range)
        self.assertEqual(output.read_bytes(), b"keep this output")
        self.assertEqual(list(self.root.glob("existing.h5.*.tmp")), [])
        with h5py.File(self.range, "r+") as h5:
            h5["rdc1"].attrs["sweep_frequency_hz"] = 2.
        with self.assertRaisesRegex(ValueError, "Conflicting RDC1"):
            load_rdc1_hdf5_metadata(self.range)


if __name__ == "__main__":
    unittest.main()
