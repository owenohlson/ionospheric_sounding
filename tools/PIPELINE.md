# One HDF5 plotting pipeline

Raw TBN and WAV ingestion belongs in `rdc_export.py`. The paired WAV/TBN plotting
scripts have been replaced by one HDF5-based command per plot type. Existing
HDF5 files remain readable; raw files now need an export before single-receiver
plotting. There is no automatic second timestamp alignment when plotting a cache.

## Quick start

From the repository root:

```bash
python tools/rdc_export.py data/two_path_sim.wav --stage 1 \
  --sweep-frequency 2 --bandwidth=-100e3 --gap-fill zero \
  --output data/sim_rdc1.h5

python tools/dechirp_plot.py data/sim_rdc1.h5 --output figures/dechirp.png

python tools/sliding_delay_doppler.py data/sim_rdc1.h5 \
  --integration-time 120 --hop-time 30 --output figures/sliding_dd
```

The recording must be at least as long as the selected integration. To retain
raw samples instead, export with `--stage 0`; optionally record `--bandwidth`
there so plots need no waveform flags. TBN uses the same workflow, with physical
`--stands`, `--pol`, and optional `--drop-bad-stands` at export. A 30-second example:

```bash
python tools/rdc_export.py recording.dat --stage 2 --duration 30 \
  --sweep-frequency 2 --bandwidth=-99.259377e3 --pol x --drop-bad-stands \
  --d-min 217.5 --d-max 219.5 --gap-fill zero --output recording_rdc2.h5

python tools/array_delay_doppler.py --rdc2-input recording_rdc2.h5 \
  --beamformer mvdr --output figures/array.png
```

## Commands and inputs

| Current command | Supported HDF5 content |
| --- | --- |
| `dechirp_plot.py` | RDC0 or RDC1 (including retained RDC1 in RDC2) |
| `plot_delay_doppler.py` | RDC0, RDC1, or saved RDC2 |
| `sliding_delay_doppler.py` | RDC0, RDC1, or saved RDC2 frames |
| `doppler_time_plot.py` | RDC0 or RDC1, including retained RDC1 |
| `phase_doppler_plot.py` | RDC0 or RDC1, including retained RDC1 |
| `phase_doppler_time_plot.py` | RDC0 or RDC1, including retained RDC1 |
| `sliding_phase_doppler.py` | RDC0 or RDC1, including retained RDC1 |
| `plot_filter_output.py` | RDC0 or RDC1; dechirped power-delay profile, not matched filtering |
| `spectrogram.py` | RDC0 with complete ADC sweeps |
| `array_delay_doppler.py` | LWA RDC0 or RDC2; existing array processing preserved |
| `sliding_array_delay_doppler.py` | LWA RDC0 or RDC2; existing array processing preserved |

All `tbn_` plotting prefixes have been removed. In particular,
`delay_doppler_plot.py` is now `plot_delay_doppler.py`. Old TBN array raw-input
compatibility remains, but exporting first is the recommended common workflow.
Array RDC1 input should be exported to RDC2 first. Array geometry/cable lookup
still needs LSL; ordinary WAV/HDF5 plots do not. Legacy LWA caches that labeled
digitizer pairs instead of physical stands also require LSL to resolve identities.

The single-receiver commands accept the HDF5 file positionally or through
`--rdc0-input`, `--rdc1-input`, or `--rdc2-input` (`--input-rdc2` also works).
They select the first available stand by default; use `--stand` to choose another.
`--pol both` combines received power or phase products, never X/Y voltages.
For WAV it means use the sole receiver of unknown polarization.

## Processing versus display settings

Sample rate, sweep frequency, bandwidth, reference gating, and applied windows
come from the cache. Conflicting overrides for already-applied processing fail
instead of silently relabeling data. RDC0 still needs a bandwidth if none was
recorded at export; fast-time windows/gating can be selected at that stage.
Timing alignment (`--offset`, `--no-timestamp-align`) belongs at raw export.

`--d-min`/`--d-max` remain milliseconds. The default auto window is +/-1.5 ms
around the strongest preview bin (first 32 sweeps, or first selected saved RDC2
frame). `--full-delay` selects all stored bins. Cropped caches cannot recover
discarded delays. Color limits, Doppler zoom, and titles remain plot-time options.

Single-receiver `--tstart`/`--tend` are seconds relative to the cache's first
sweep, rounded inward to complete sweeps. UTC annotations use `pulse0_utc`
where present. Array commands retain their existing cached time-coordinate
conventions. Saving new caches does not alter an older file's metadata.

With saved RDC2, single-receiver DD plots use its existing frames and inherit
integration/hop/slow window/NFFT. A static plot selects the first complete frame
inside the time bounds. To change those FFT settings, use RDC1 as the input or
re-export from the retained RDC1 via `rdc_export.py --rdc1-input ... --stage 2`.
Array plots retain their existing retained-RDC1 rewindowing support.

Videos need ffmpeg; PNGs are removed only after successful encoding unless
`--keep-frames` is given. Each run uses a separate frame directory to avoid
including old PNGs. Gaps are annotated only when present in that view, using
nonfinite samples or saved per-sweep gap counts. Zero filling after a poisoned
FFT cannot recover its lost information; use zero fill before the range FFT.

`spectrogram.py --plot average` averages finite FFT blocks over selected
channels (all cached channels unless a stand is selected). Its optional `--pfb`
uses LSL SpecMaster; the normal spectrum/spectrogram uses SciPy.
