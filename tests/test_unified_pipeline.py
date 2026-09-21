import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import subprocess
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import rdc_export
import rdc_plot
from rdc_plot_input import PlotInput
from lfm_utils import LFMWaveform
from rdc_utils import load_rdc0_hdf5, load_rdc1_hdf5_metadata


class UnifiedPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.wav = self.root / "signal.wav"
        config = LFMWaveform(sample_rate=2048, sweep_frequency=4, bandwidth=-2048)
        t = np.arange(16384) / 2048
        self.iq = np.tile(config.waveform, 32) * np.exp(2j * np.pi * 16.1 * t)
        sf.write(self.wav, np.column_stack([self.iq.real, self.iq.imag]), 2048, subtype="FLOAT")
        self.raw = self.root / "rdc0.h5"
        self.export([str(self.wav), "--stage", "0", "--sweep-frequency", "4", "--bandwidth=-2048", "--output", str(self.raw)])

    def export(self, args):
        with contextlib.redirect_stdout(io.StringIO()):
            rdc_export.main(args)

    def test_wav_iq_is_one_unknown_pol_channel_and_samples_unchanged(self):
        loaded = load_rdc0_hdf5(self.raw)
        self.assertEqual(loaded["rdc0"].shape, (512, 32, 1))
        np.testing.assert_allclose(loaded["rdc0"][:, :, 0].T.ravel(), self.iq, atol=1e-7)
        self.assertEqual(loaded["channel_pols"].tolist(), [-1])
        self.assertEqual(loaded["metadata"]["polarization_mode"], "unknown")
        self.assertIsNone(loaded["metadata"]["station"])
        self.assertIsNone(loaded["metadata"]["pulse0_utc"])
        self.assertIsNone(loaded["metadata"]["tuning_frequency_hz"])

    def test_stage_chain_matches_direct(self):
        one, two, direct = [self.root / name for name in ("one.h5", "two.h5", "direct.h5")]
        self.export(["--stage", "1", "--rdc0-input", str(self.raw), "--output", str(one)])
        self.export(["--stage", "2", "--rdc1-input", str(one), "--output", str(two)])
        self.export([str(self.wav), "--stage", "2", "--sweep-frequency", "4", "--bandwidth=-2048", "--output", str(direct)])
        with h5py.File(two) as a, h5py.File(direct) as b:
            np.testing.assert_allclose(a["rdc2/real"], b["rdc2/real"])
            np.testing.assert_allclose(a["rdc2/imag"], b["rdc2/imag"])
            self.assertEqual(json.loads(a.attrs["metadata_json"])["pol"], "unknown")

    def test_utc_and_nonzero_start_propagate(self):
        zero, one = self.root / "timed.h5", self.root / "timed1.h5"
        self.export([str(self.wav), "--stage", "0", "--sweep-frequency", "4", "--bandwidth=-2048",
                     "--start-timestamp", "2023-12-09T16:00:00.125Z", "--tstart", "1",
                     "--carrier-frequency", "25600000", "--output", str(zero)])
        loaded = load_rdc0_hdf5(zero)
        self.assertEqual(loaded["metadata"]["start_offset_samples"], 256)
        self.assertEqual(loaded["metadata"]["pulse0_utc"], "2023-12-09T16:00:01.250000+00:00")
        self.export(["--stage", "1", "--rdc0-input", str(zero), "--tstart", "1", "--output", str(one)])
        meta = load_rdc1_hdf5_metadata(one)["metadata"]
        self.assertEqual(meta["pulse0_utc"], "2023-12-09T16:00:02.250000+00:00")
        self.assertEqual(meta["tuning_frequency_hz"], 25600000)

    def test_plot_loading_preserves_axis_and_gap_location(self):
        with h5py.File(self.raw, "r+") as h5:
            h5["rdc0/real"][10, 5, 0] = np.nan
        args = rdc_plot.parser_for("dechirp").parse_args([str(self.raw), "--gap-fill", "zero", "--full-delay"])
        source = PlotInput(args)
        before, gaps = source.spectra(0, 10)
        self.assertTrue(np.all(np.isfinite(before)))
        self.assertEqual(np.flatnonzero(gaps).tolist(), [5])
        one = self.root / "gap1.h5"
        self.export(["--stage", "1", "--rdc0-input", str(self.raw), "--gap-fill", "zero", "--output", str(one), "--d-min", "1", "--d-max", "4"])
        args.input_file = str(one)
        after_source = PlotInput(args)
        after, gaps = after_source.spectra(0, 10)
        self.assertEqual(np.flatnonzero(gaps).tolist(), [5])
        selected = (source.delay >= 1) & (source.delay <= 4)
        np.testing.assert_allclose(after, before[:, :, selected])
        np.testing.assert_array_equal(after_source.delay, source.delay[selected])
        self.assertIsNone(after_source.annotation(0, 1, gaps[:4])["info_panel_footer_note"])
        self.assertIsNotNone(after_source.annotation(0, 2, gaps)["info_panel_footer_note"])

    def test_render_all_single_receiver_types_and_video_frames(self):
        one, two = self.root / "one.h5", self.root / "two.h5"
        self.export(["--stage", "1", "--rdc0-input", str(self.raw), "--output", str(one)])
        self.export(["--stage", "2", "--rdc1-input", str(one), "--integration-time", "2", "--hop-time", "1", "--output", str(two)])
        for kind in ("dechirp", "filter_output", "doppler_time", "phase_doppler", "phase_doppler_time", "delay_doppler", "spectrogram"):
            with self.subTest(kind=kind):
                path = self.root / (kind + ".png")
                source = self.raw if kind == "spectrogram" else (two if kind == "delay_doppler" else one)
                args = [str(source), "--output", str(path)]
                if kind == "doppler_time":
                    args += ["--integration-time", "2", "--hop-time", "1"]
                if kind == "spectrogram":
                    args += ["--tstart", "1", "--window-size", "128", "--hop-size", "64"]
                with contextlib.redirect_stdout(io.StringIO()):
                    rdc_plot.main(kind, args)
                self.assertGreater(path.stat().st_size, 1000)
        for kind, source in (("sliding_delay_doppler", two), ("sliding_phase_doppler", one)):
            out = self.root / kind
            with patch("rdc_plot.subprocess.run") as encoder:
                rdc_plot.main(kind, [str(source), "--output", str(out), "--max-frames", "2", "--keep-frames"])
                encoder.assert_called_once()
            self.assertEqual(len(list(out.glob("frames_*/*.png"))), 2)

    def test_dry_run_and_invalid_pol_do_not_write(self):
        output = self.root / "dry.h5"
        self.export([str(self.wav), "--stage", "0", "--sweep-frequency", "4", "--dry-run-size", "--output", str(output)])
        self.assertFalse(output.exists())
        with self.assertRaisesRegex(ValueError, "unknown polarization"):
            self.export([str(self.wav), "--stage", "0", "--sweep-frequency", "4", "--pol", "x", "--output", str(output)])
        self.assertFalse(output.exists())

    def test_mono_and_cli_without_lsl(self):
        mono = self.root / "mono.wav"
        sf.write(mono, self.iq.real, 2048, subtype="FLOAT")
        output = self.root / "mono.h5"
        command = [sys.executable, str(Path(rdc_export.__file__)), str(mono), "--stage", "0",
                   "--sweep-frequency", "4", "--bandwidth=-2048", "--output", str(output)]
        subprocess.run(command, check=True, capture_output=True)
        loaded = load_rdc0_hdf5(output)
        self.assertTrue(np.all(loaded["rdc0"].imag == 0))
        self.assertEqual(loaded["metadata"]["wav_layout"], "real")
        for name in ("plot_delay_doppler", "array_delay_doppler", "sliding_array_delay_doppler"):
            subprocess.run([sys.executable, str(Path(rdc_export.__file__).with_name(name + ".py")), "--help"],
                           check=True, capture_output=True)

    def test_shared_reader_matches_original_dechirp_and_explicit_channels(self):
        from lfm_utils import dechirp_fft_complex, dechirp_delay_axis_ms
        args = rdc_plot.parser_for("dechirp").parse_args([str(self.raw), "--full-delay"])
        source = PlotInput(args)
        spectra, _ = source.spectra(0, 32)
        _, original = dechirp_fft_complex(self.iq.astype(np.complex64), source.config, window="hann")
        _, order = dechirp_delay_axis_ms(source.config, n_bins=512)
        np.testing.assert_allclose(spectra[0], original[:, order], rtol=1e-5, atol=1e-4)
        with h5py.File(self.raw, "r+") as h5:
            group = h5["rdc0"]
            raw = group["real"][...] + 1j * group["imag"][...]
            for key in ("real", "imag", "channel_stands", "channel_pols", "invalid_counts"):
                del group[key]
            data = raw * np.array([1., 2j, 3., 4j])[None, None, :]
            group.create_dataset("real", data=data.real)
            group.create_dataset("imag", data=data.imag)
            group.create_dataset("channel_stands", data=[6, 6, 23, 23])
            group.create_dataset("channel_pols", data=[0, 1, 0, 1])
            meta = json.loads(h5.attrs["metadata_json"])
            meta.update(channel_id_kind="physical_stand", source_format="tbn", station="lwasv")
            h5.attrs["metadata_json"] = json.dumps(meta)
        args.stand, args.pol = 23, "y"
        source = PlotInput(args)
        self.assertEqual(source.channels.tolist(), [3])
        selected, _ = source.spectra(0, 32)
        np.testing.assert_allclose(selected, spectra * 4j, rtol=1e-5, atol=1e-4)

    def test_metadata_conflicts_and_average_spectrum(self):
        one = self.root / "one.h5"
        self.export(["--stage", "1", "--rdc0-input", str(self.raw), "--output", str(one)])
        for flag, value in (("--dechirp-window", "blackman"), ("--bandwidth", "-1000"), ("--sweep-frequency", "2"),
                            ("--reference-gate-frequency", "10")):
            args = rdc_plot.parser_for("dechirp").parse_args([str(one), flag, value])
            with self.assertRaises(ValueError):
                PlotInput(args)
        output = self.root / "average.png"
        rdc_plot.main("spectrogram", [str(self.raw), "--plot", "average", "--window-size", "128", "--output", str(output)])
        self.assertGreater(output.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
