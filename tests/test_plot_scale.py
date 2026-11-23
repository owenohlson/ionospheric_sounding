"""Relative power scales of the plots (plotting_utils)."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

import helpers  # noqa: F401  (sets sys.path)
import plotting_utils as plots


class PlotScaleTests(unittest.TestCase):
    def test_lfm_scan_explains_the_lag_peak_markers(self):
        k = np.linspace(-3e6, 3e6, 2001)
        p = np.ones_like(k)
        p[1500] = 1e4
        legends = []
        real_save = plots.save
        plots.save = lambda fig, path, **kw: legends.extend(
            t.get_text() for a in fig.axes if a.get_legend() for t in a.get_legend().get_texts())
        try:
            plots.plot_lfm_scan(k, p, [(k[1500], 40.0)], [], [], "scan", "unused.png")
            self.assertTrue(any("spectrum peaks" in t for t in legends))
            legends.clear()
            plots.plot_lfm_scan(k, p, [], [], [], "scan", "unused.png")
            self.assertFalse(any("spectrum peaks" in t for t in legends))
        finally:
            plots.save = real_save
            plots.plt.close("all")

    def test_sky_map_layout_ignores_the_panel_text(self):
        """Station-panel text of any length leaves the figure size and the maps where they are."""
        az, el = np.arange(0, 361, 10.0), np.arange(0, 91, 10.0)
        maps = [np.random.default_rng(0).random((el.size, az.size)) + 1]
        texts = [["HF radar sources: HFRnet built-in table"],
                 ["x" * 140, "  NET: " + ", ".join(f"S{i:02d}" for i in range(40)), "short"],
                 []]
        layouts = []
        for side in texts:
            captured = {}
            real_save = plots.save
            plots.save = lambda fig, path, **k: captured.update(
                size=tuple(fig.get_size_inches()), axes=[tuple(np.round(a.get_position().bounds, 6)) for a in fig.axes])
            try:
                plots.plot_angle_maps(maps, az, el, ["cell"], "Sky map", "unused.png", side_lines=side)
            finally:
                plots.save = real_save
                plots.plt.close("all")
            layouts.append(captured)
        self.assertTrue(all(lay == layouts[0] for lay in layouts))
        wrapped = plots.wrap_side_lines(texts[1])
        self.assertTrue(all(len(x) <= plots.SIDE_PANEL_CHARS for x in wrapped))
        net = next(i for i, x in enumerate(wrapped) if x.startswith("  NET"))
        self.assertTrue(wrapped[net + 1].startswith("    "))             # continuation of an indented line

    def test_above_median_and_re_peak(self):
        p = np.array([[100.0, 110.0, np.nan], [100.0, 140.0, 100.0]])
        rel = plots.above_median_db(p)
        self.assertEqual(float(np.nanmedian(rel)), 0.0)
        self.assertAlmostEqual(float(np.nanmax(rel)), 40.0)
        self.assertTrue(np.isnan(rel[0, 2]))
        self.assertEqual(float(np.nanmax(plots.re_peak_db(p))), 0.0)
        self.assertEqual(plots.median_limits([rel]), (0.0, 40.0))
        self.assertEqual(plots.median_limits([np.zeros(4)]), (0.0, 10.0))
        self.assertEqual(plots.median_limits([rel], 5, 30), (5.0, 30.0))

    def test_absolute_offset_does_not_change_the_plot(self):
        rng = np.random.default_rng(0)
        p = 10 * np.log10(rng.exponential(size=(40, 30)))
        p[20, 15] += 30
        with tempfile.TemporaryDirectory() as d:
            out = [Path(d) / f"{i}.png" for i in range(2)]
            for path, offset in zip(out, (0.0, 94.0)):          # the same map, 94 dB apart in absolute terms
                plots.plot_delay_doppler(np.arange(30), np.arange(40), p + offset, "t", path)
            import matplotlib.image as mpimg
            a, b = (mpimg.imread(str(x)) for x in out)
            self.assertEqual(a.shape, b.shape)
            self.assertLess(float(np.mean(np.abs(a - b) > 1 / 255)), 1e-3)     # at most a few rounding pixels
        np.testing.assert_allclose(plots.above_median_db(p + 94.0), plots.above_median_db(p), atol=1e-9)


if __name__ == "__main__":
    unittest.main()
