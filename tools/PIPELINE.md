# Processing pipeline

```
raw TBN (preprocess_tbn.py: report / cut, optional) ┐
raw WAV (mono real / stereo I/Q) ───────────────────┤
                                             ▼
                                   export_rdc0.py   RDC0  (adc, pulses, channels)      raw sweeps
                                             ▼
                                   export_rdc1.py   RDC1  (delay, pulses, channels)    dechirp + range FFT
                                    ┌────────┴─────────┐
                  export_rdc2.py    ▼                  ▼   export_rdc2p.py
      RDC2 (frames, delay, doppler, channels)     RDC2P (delay, estimates, channels)
             Doppler FFT                           sweep-to-sweep phase (lag-1) products
                  ▼
      export_rdc3.py  RDC3 (frames, delay, doppler, beams)   array beamforming
                  ▲
      calibrate_array.py  per-stand phase self-calibration (CAL.h5), used via --calibration
```

Every export writes one HDF5 file. Its root attribute `metadata_json` holds
every irreversible choice made so far: sample rate, tuning, sweep rate,
bandwidth, alignment, channel identities, windows, gating, delay/Doppler crops,
integration/hop, and gap policy. It also holds a `history` list with one entry per
export step. Later stages inherit these settings. A request that contradicts an
already-applied setting is an error, never a silent relabel.

Each export command also accepts any earlier input and chains through the missing
stages. From a raw recording, `export_rdc1/2/2p.py` stream sweep by sweep: each
sweep is cut, dechirped and reduced to the delay window before the next is read,
so no RDC0 is ever stored (RDC2/RDC2P go through a temporary RDC1 next to the
output).

**Large recordings.** RDC0 holds every raw sample as complex float32, about 4
times the TBN file (375 GB for 15 minutes of all 520 inputs), so use it only for
short test cuts. Export RDC1/RDC2 straight from the raw file instead; see
"Workflow for an LWA session" below for sizes and commands.
An RDC2 keeps the RDC1 it came from by default, so integration and hop can be
changed later without re-dechirping.

The phase path (RDC2P) is per-channel and combined without phases, so it has no
array stage. Array processing (RDC3, beamformed plots, sky maps) uses the FFT
path.

## Where the code lives

| Module | Contents |
| --- | --- |
| `lfm_utils.py` | Maths only: reference chirp and windows, dechirp/range FFT, delay axis, Doppler FFT, pulse-pair (phase) Doppler, sweep alignment, STFT, LFM estimation from a spectrogram |
| `lwa_utils.py` | Everything LWA-specific: LWA-SV stand positions, cable delays and phase factors, digitizer mapping, bad stands (SSMIF status), legacy channel labels; TBN frame-header scan, quality summary, `TBNReader` (samples placed by timetag, gaps as NaN, exact 196 MHz timestamps; LSL is used only for station metadata) and trimming for `preprocess_tbn.py` |
| `wav_utils.py` | WAV only: info, block reader, UTC parsing |
| `rdc_io.py` | HDF5 layout: `RDCFile` reader (all stages, legacy formats), `StageWriter`, metadata/history, atomic output |
| `rdc_utils.py` | Stage-to-stage processing: channel selection, sweep packing, `RDCView` (brings any stored stage to the product a plot needs), exports, array processing (`ArrayProcessor`, covariance, steering, angle spectra, beam pattern), plot reductions |
| `calibration_utils.py` | Array self-calibration: plane-wave fit, phase-gain estimation (alternating fit, ramp kept out), held-out validation, the calibration file (`ArrayCalibration`) |
| `codar_stations.py` | HF radar source identification: the built-in HFRnet station table (`hfrnet_stations.csv`, updated with `--add`), bearings and sky-wave geometry from the station, waveform/band/date filters, the labelling rule |
| `plotting_utils.py` | Figures only (arrays → PNG) and video assembly |
| `cli.py` | Shared command-line flag groups and the export command line |

## Commands and the inputs they accept

| Command | RDC0 | RDC1 | RDC2 | RDC2P | RDC3 |
| --- | :-: | :-: | :-: | :-: | :-: |
| `plot_spectrogram.py` (`--average` for a mean spectrum) | ✓ | | | | |
| `plot_dechirp.py` (delay vs time) | ✓ | ✓ | ✓ * | | |
| `plot_power_delay.py` | ✓ | ✓ | ✓ * | | |
| `plot_delay_doppler.py` / `sliding_delay_doppler.py` | ✓ | ✓ | ✓ | | ✓ |
| `plot_doppler_time.py` | ✓ | ✓ | ✓ * | | |
| `plot_phase_doppler.py` (`--kind delay-time` or `doppler-time`) / `sliding_phase_doppler.py` | ✓ | ✓ | ✓ * | ✓ | |
| `calibrate_array.py` | ✓ | ✓ | ✓ | | |
| `check_array_calibration.py` | ✓ | ✓ | ✓ | | |

\* uses the RDC1 retained inside the RDC2 file.

Each command uses the most-processed stored stage that still fits the request.
For example, a delay-Doppler plot of an RDC2 uses the stored frames. It recomputes
from the retained RDC1 only if you change `--integration-time`, `--hop-time`,
`--slow-window` or `--nfft-doppler`. Plots read HDF5 only; export raw
recordings first.

## Times, channels and delay windows

* **Time axis.** Seconds from the first sweep of the original RDC0 export. The
  same instant keeps the same time in every later file. `--tstart/--tend` use
  this axis. UTC is shown when known (TBN always; WAV only with `--start-utc`).
* **Channels.** `--stands` accepts `"5"`, `"1-10,20"`, `first` or `all`; `--pol`
  accepts x, y or both; `--bad-stands` accepts none or status (stands flagged in the SSMIF).
  * Single-receiver plots default to the first stand, both pols, incoherently summed.
  * Array processing (`--beamformer conventional|mvdr`, `--angle-output`,
    `--beam-pattern-output`) defaults to all stands, x pol, and drops SSMIF-flagged stands.
    With `--calibration CAL.h5` it also drops the stands that calibration flagged.
* **Station description.** Stand identities (via TBN digitizer), positions,
  cable delays and status come from LSL's bundled LWA-SV SSMIF unless
  `--ssmif FILE` (an SSMIF or an observation metadata `.tgz`) or `$LWA_SSMIF`
  gives one for the observation date. Files store each channel's digitizer, so
  a dated SSMIF re-derives stand identities without re-exporting.
* **Delay window.** The default is ±1.5 ms around the strongest delay in a
  preview; `--d-min/--d-max` set it explicitly and `--full-delay` shows
  everything stored. Delays are relative to the sweep reference timing, not
  physical ranges.
* **Sweep offset.** With the echo far into the sweep (e.g. 218 ms of 500 ms) and
  |B| ≠ fs, dechirping leaves a weaker copy at +(fs−|B|)/|k| and loses about 3 dB.
  Export with `--offset` ≈ observed delay − a few ms to put echoes near 0 ms.
  The offset is recorded in the metadata.

## Array calibration

Phases across the array come from three places, applied in this order:

1. **Station description** (`--ssmif`, or `$LWA_SSMIF`): stand positions and cable
   delays for the observation date. Without the right file the array is not
   coherent (2023-12-09: eta_ph 0.03 with LSL's bundled file, 0.34 with
   `LD021_0001.tgz`).
2. **Self-calibration** (`calibrate_array.py`, then `--calibration CAL.h5`): one
   phase per stand measured from the strongest echo cells, for whatever the SSMIF
   does not capture. The fit alternates between finding each cell's plane-wave
   direction and averaging the per-stand residual phase. Cells are echoes:
   strong, and localised in delay (narrowband interference, which dechirping
   spreads over every delay of one Doppler bin, is skipped). Stands are flagged,
   and dropped when the file is applied, if:
   * their noise power is far from the median (`--max-power-db`, default 10 dB);
   * their phase differs between two classes of echo. Some cells fit the
     first-pass phases poorly; fitted on their own, they need very different
     phases on certain stands. On 2023-12-09 the two classes look like the O and
     X magnetoionic modes, and about 30 stands also differ in amplitude between
     them, which points to a different polarization response on those stands.
     No single phase per stand can serve both modes;
   * their phase does not follow the fit (consistency below half the median). Only phases are corrected; equalising amplitudes
   would amplify the noise of weak channels.

What self-calibration cannot do:

* **Change the pointing.** A linear phase ramp across the array looks exactly
  like a different arrival direction. It is kept out of the gains, so absolute
  directions still rest on the SSMIF geometry. Check them against a known
  transmitter.
* **Transfer between frequencies.** A calibration file is valid at its tuning
  frequency (±1 kHz), and best for the same observation. Cable phases drift with
  temperature, so recalibrate per session.
* **Separate errors that only show up for one direction.** A wrong stand position
  is corrected exactly only near the direction it was calibrated on.

`calibrate_array.py` reports the plane-wave fit on held-out cells (fitted on
alternate frames, or alternate cells for one frame) three ways: SSMIF only,
without the flagged stands, and with self-calibration. It also counts held-out
cells the calibration made worse. The in-sample number is always optimistic.
2023-12-09 (fit on 0–5 s, tested on 5–9.5 s): 0.56 → 0.84 → 0.96. `check_array_calibration.py
--calibration CAL.h5` adds a `selfcal` variant, for testing on another time range.

## Workflow for an LWA session

The commands below use the 2023-12-09 13.5 MHz file. Each TBN file holds one
tuning (13.5, 16.15, 24.55 MHz, ...), so repeat steps 2–6 per file with that
transmitter's sweep settings.

### 0. Station description (every command)

```bash
export LWA_SSMIF=/path/to/LD021_0001.tgz   # once per shell or job script (bash)
LWA_SSMIF=/path/to/LD021_0001.tgz python tools/...   # or for one command
python tools/... --ssmif /path/to/LD021_0001.tgz     # or as a flag
```

The value is an SSMIF text file or an observation metadata `.tgz`. Exports need it
to map digitizers to stands and to apply `--bad-stands status`. Array commands
need it for positions and cable delays. Files store each channel's digitizer, so
setting it later still re-labels the stands correctly.

### 1. Check the file

```bash
python tools/preprocess_tbn.py raw.dat --report-only
```

Read `tuning_mhz`, `start_utc`, the missing captures and the gaps. No trimming is
needed: exports place every frame by its timetag and ignore leftover frames from
the previous tuning.

### 2. Sweep settings and delay window (per frequency)

```bash
python tools/export_rdc0.py raw.dat --sweep-frequency 2 --bandwidth=-99.259377e3 \
    --stands 10 --tend 30 --output test_rdc0.h5
python tools/plot_spectrogram.py test_rdc0.h5 --overlay-lfm --output spec.png
python tools/plot_power_delay.py test_rdc0.h5 --full-delay --output pdp.png
```

The spectrogram estimate confirms the sweep rate and bandwidth. The full-delay
profile shows where the echoes are. Then choose `--d-min/--d-max`, and
`--offset` if the echoes sit far into the sweep (see "Sweep offset").

### 3. Export the session (RDC1)

```bash
python tools/export_rdc1.py raw.dat --sweep-frequency 2 --bandwidth=-99.259377e3 \
    --pol both --bad-stands status --gap-fill zero --d-min 12 --d-max 20 \
    --output s13p5_rdc1.h5 --dry-run        # prints the size; run again without --dry-run
```

RDC1 is the product to keep. Every later step computes its Doppler frames from
it with whatever `--integration-time/--hop-time` it is given: delay-Doppler plots
and videos, Doppler-time, calibration and RDC3. Each frame reads only its own
sweeps, so memory stays at about one frame. In the commands below,
`s13p5_rdc1.h5` can be replaced by an RDC2 file; the commands are the same.

`export_rdc2.py` is optional. It also stores one set of Doppler frames (its
`--integration-time/--hop-time`) as a cache. That saves the FFTs when the same
frames are plotted many times, at the cost of size: overlapping frames are
stored several times over, zero-padded to `nfft`. Plots use the cache only when
their settings match it, and otherwise recompute from the RDC1 the file keeps.
With `--no-keep-rdc1` the cache is all there is, and integration/hop are fixed.

**Size (float32, uncompressed; gzip saves little on noisy data):**

    RDC1 = delay bins x sweeps x channels x 8 bytes
    RDC2 = frames x delay bins x nfft x channels x 8 bytes      (in addition, when RDC1 is kept)

* Delay bins: one per 10.07 us at 100 kHz, about 99 per ms of window.
* Sweeps: 1792 per 15 minutes at 2 Hz.
* Channels: up to 520 inputs.
* Frames: (900 - integration) / hop + 1, i.e. 27 for 120 s / 30 s.
* nfft: the next power of two of the sweeps per frame, i.e. 256 for 120 s at 2 Hz.

For one 15-minute file with all 520 inputs:

| Window | RDC1 (`export_rdc1.py`) | RDC1 + RDC2 cache at 120 s/30 s |
| --- | --- | --- |
| 3 ms | 2.2 GB | 11 GB |
| 10 ms | 7.4 GB | 36 GB |
| 22 ms | 16 GB | 80 GB |
| 108 ms | 80 GB | — |

`--dtype float16` is exact for RDC0 only: dechirped values exceed its range, and
the export stops with an error if they do.

### 4. Single-receiver and incoherent plots

`--stands all` sums the power of every selected input. Without it, the plot uses
the first stand. `--pol x|y|both` selects polarizations.

```bash
python tools/plot_dechirp.py s13p5_rdc1.h5 --stands all --navg 4 --output dechirp.png        # delay vs time
python tools/plot_power_delay.py s13p5_rdc1.h5 --stands all --output pdp.png
python tools/plot_doppler_time.py s13p5_rdc1.h5 --stands all --integration-time 60 --hop-time 10 --output dt.png
python tools/plot_delay_doppler.py s13p5_rdc1.h5 --stands all --tstart 0 --tend 120 --output dd.png
python tools/sliding_delay_doppler.py s13p5_rdc1.h5 --stands all --integration-time 120 --hop-time 30 \
    --output dd_video
python tools/plot_phase_doppler.py s13p5_rdc1.h5 --stands all --kind doppler-time --output pd.png
python tools/sliding_phase_doppler.py s13p5_rdc1.h5 --stands all --output pd_video
```

Delay-Doppler plots and videos default to one frame over the whole selection;
give `--integration-time/--hop-time` (e.g. 120 / 30) for sliding frames, and
`--tstart/--tend` to restrict the time range.

### 5. Calibrate the array (once per file and polarization)

```bash
python tools/calibrate_array.py s13p5_rdc1.h5 --pol x --integration-time 10 --hop-time 10 \
    --output cal_s13p5_x.h5 --csv cal_s13p5_x.csv
python tools/calibrate_array.py s13p5_rdc1.h5 --pol y --integration-time 10 --hop-time 10 \
    --output cal_s13p5_y.h5
```

* Use non-overlapping frames, so the held-out validation really is held out.
* Check the report: the held-out fit should rise (2023-12-09: 0.56 → 0.84 → 0.96)
  and few cells should get worse.
* Recalibrate when the tuning, the session or the SSMIF changes. A file is
  refused at another frequency.
* Each polarization needs its own file.
* For an independent check, calibrate on one time range (`--tend 300`) and test
  another:
  `check_array_calibration.py s13p5_rdc1.h5 --tstart 300 --calibration cal.h5`.

### 6. Array plots (one polarization, calibrated)

```bash
# sliding array delay-Doppler: beam re-aimed each frame at the strongest echo
python tools/sliding_delay_doppler.py s13p5_rdc1.h5 --pol x --beamformer conventional \
    --calibration cal_s13p5_x.h5 --integration-time 120 --hop-time 30 --output dd_beam_video
# ... plus sliding sky (angle) maps: whole window, or the N strongest cells per frame
python tools/sliding_delay_doppler.py s13p5_rdc1.h5 --pol x --beamformer conventional \
    --calibration cal_s13p5_x.h5 --integration-time 120 --hop-time 30 \
    --angle-output sky_video --angle-cells 4 --output dd_beam_video
# fixed beam instead of auto-aim; MVDR; beam pattern
python tools/plot_delay_doppler.py s13p5_rdc1.h5 --pol x --beamformer mvdr --look-az -86 --look-el 22 --tend 120 \
    --calibration cal_s13p5_x.h5 --beam-pattern-output bp.png --output dd_mvdr.png
# stored beams for later (RDC3)
python tools/export_rdc3.py s13p5_rdc1.h5 --pol x --calibration cal_s13p5_x.h5 --look -86,22 \
    --integration-time 120 --hop-time 30 --output s13p5_rdc3.h5
```

**Naming the HF radar behind a peak.** Add `--codar-stations` to any sky-map
command. It uses the HFRnet station table that ships with the code
(`tools/hfrnet_stations.csv`); give CSV paths after the flag to use other tables
instead. The sky map then makes room for a panel on the right that lists, for
each peak, the candidate stations with their bearing, azimuth error and
distance, and the sky-wave path that fits the elevation. Peaks that can be named
get a label and small triangles at the stations' bearings:

```bash
python tools/sliding_delay_doppler.py s13p5_rdc1.h5 --pol x --beamformer conventional \
    --calibration cal_s13p5_x.h5 --integration-time 120 --hop-time 30 --angle-output sky_video \
    --angle-cells 4 --codar-stations --output dd_beam_video
```

The built-in table is the union of every HFRnet export added to it. Each
distinct station configuration is kept, along with the exports it appeared in
(the `Snapshots` column). Stations with no published waveform (-9999) are left
out. To add a newer export (download the CSV from the HFRnet station list):

```bash
python tools/codar_stations.py --add HFRnet-station-list-<date>.csv   # then commit the table
python tools/codar_stations.py                                        # summary of the table
```

How a station is chosen:

1. **Active:** it was added before the observation date and still produced
   radials then.
2. **Waveform:** it uses the same sweep rate and signed bandwidth as the dechirp.
   Other sweeps do not compress into echoes.
3. **Band:** at least half of its sweep band falls inside the receive band.
4. **Direction:** its bearing is within `--station-az-tol` (default 4°) of the
   peak's azimuth, and the peak's elevation fits a 1–3 hop path reflecting at
   90–450 km.

A label is given only when every other candidate fits at least 3° worse in
azimuth. The label is one station, or one network when several of its
GPS-synchronised stations fit equally well. Anything else is listed as
ambiguous and not drawn.

Every configuration in the table is a candidate, because the one in force on the
observation date is not recorded. Exports from close to your observation dates
make the results sharper.
Delay cannot help: our delays are relative to our own sweep timing, and the
tables list no sweep offsets.

On 2023-12-09 at 13.5 MHz:

* the echoes near az −84°, el 19° are the SLO network (ARG1, DCSR, ESTR, FBK1,
  LUIS, PTC1, about 1,260 km away; 1 hop at a virtual height of about 260 km);
* those near az −92° are USC (SCCI, SCDH, about 1,070 km away);
* the interferer at az −65° matches no station.

Auto-aimed videos write `look_directions.csv` (az/el per frame) into the
`--output` directory. Sky maps are in dB relative to their own peak
(`--angle-range-db`).

## Examples

```bash
python tools/preprocess_tbn.py raw.dat --report-only
python tools/preprocess_tbn.py raw.dat --tstart 60 --tend 960 --output clean.dat

python tools/export_rdc2.py clean.dat --sweep-frequency 2 --bandwidth=-99.259377e3 \
  --pol x --bad-stands status --d-min 217.5 --d-max 219.5 --gap-fill zero \
  --integration-time 120 --hop-time 30 --output clean_rdc2.h5

python tools/plot_delay_doppler.py clean_rdc2.h5 --stands 5 --output dd_stand5.png
python tools/sliding_delay_doppler.py clean_rdc2.h5 --beamformer conventional \
  --angle-output sky --output dd_video
python tools/plot_delay_doppler.py clean_rdc2.h5 --integration-time 30 --output dd_30s.png
python tools/export_rdc3.py clean_rdc2.h5 --look 166,58 --output clean_rdc3.h5
python tools/check_array_calibration.py clean_rdc2.h5 --pol x

export LWA_SSMIF=data/LD021_0001.tgz      # station description for 2023-12-09
python tools/calibrate_array.py clean_rdc2.h5 --output cal_13p5.h5 --csv cal_13p5.csv
python tools/plot_delay_doppler.py clean_rdc2.h5 --beamformer conventional \
  --calibration cal_13p5.h5 --angle-output sky_cal.png --output dd_cal.png
```

Tests: `lwa_env/bin/python -m unittest discover -s tests` (needs h5py and
soundfile; LSL is not needed, because a fake station and synthetic TBN bytes are used).

## Migrating from the previous scripts

| Old | New |
| --- | --- |
| `splitTBNfix.py` | `preprocess_tbn.py`. It scans frame headers without LSL and finds captures by timetag (frame order inside a capture varies, and late frames can arrive after the next capture). It reports complete, partial, missing and late captures and gaps (`--report-only`), and cuts time ranges (`--tend` is an end time, `-r/-d` became `--chunk`). Trimming is no longer needed before export: exports read raw TBN with their own reader and place every frame by its timetag. |
| `rdc_export.py --stage N`, `tbn_export_rdcN.py` | `export_rdc0.py`, `export_rdc1.py`, `export_rdc2.py`, new `export_rdc2p.py`, `export_rdc3.py` |
| `dechirp_plot.py`, `plot_filter_output.py`, `doppler_time_plot.py`, `spectrogram.py` | `plot_dechirp.py`, `plot_power_delay.py`, `plot_doppler_time.py`, `plot_spectrogram.py` |
| `phase_doppler_plot.py`, `phase_doppler_time_plot.py` | `plot_phase_doppler.py --kind delay-time / doppler-time` |
| `array_delay_doppler.py`, `sliding_array_delay_doppler.py` | `plot_delay_doppler.py`, `sliding_delay_doppler.py` with `--beamformer` |
| `rdc_cache.py`, `rdc_plot.py`, `rdc_plot_input.py`, `wav_rdc.py` | `rdc_io.py`, `rdc_utils.RDCView`, `wav_utils.py` |
| `array_calibration_check.py` | `check_array_calibration.py` (positional input) |

Renamed flags:
* `--rdc0-input/--rdc1-input/--rdc2-input` are now the positional input.
* `--stand` became `--stands`.
* `--channel-combine` became `--combine`.
* `--drop-bad-stands`/`--keep-bad-stands`/`--operational-only` became
  `--bad-stands none|status`. The hand-made `BAD_LWA_STANDS` list (and
  `--bad-stands list/both`) was removed: it was built under a different station
  description and named the wrong stands under the dated SSMIF. Use the SSMIF
  status flags plus the flags in a calibration file.
* The four `--reference-gate-*` flags became `--reference-gate FREQ[,DUTY[,PHASE]]`.
* `--dry-run-size` became `--dry-run`.
* `--no-store-rdc1` became `--no-keep-rdc1`.
* `--angle-plot-output`/`--angle-output` (HDF5) became `--angle-output` (PNG, or a
  directory for videos). `--angle-top-n`/`--angle-source` became `--angle-cells N`.
* `--angle-delay-ms`/`--angle-doppler-hz` were removed; use `--angle-cells`.

Removed flags:
* `--max-stand`, `--channel-chunk-size`, `--compression-level` (fixed at gzip
  level 4), `--rdc0-dtype/--rdc0-compression/--rdc0-compression-level` (temporary
  RDC0 uses lzf).
* `--rdc0-cache/--keep-temp-rdc0`, `--rdc2-frame-index` (use `--tstart`).
* `--auto-delay-sweeps/--auto-delay-half-width`, `--streaming`, `--interactive`,
  `--units`, `--quiet`.
* `--pfb` (LSL SpecMaster averaged spectrum).
* The six `--lfm-overlay-*` manual overlay settings (`--overlay-lfm` draws the estimate).
* `--no-estimate-lfm-gate`.
* `--az-min/--az-max/--el-min/--el-max` (full sky; step via `--angle-step`).
* `--angle-plot-vmin/vmax` and `--beam-pattern-vmin/vmax` (maps are dB re peak,
  range via `--angle-range-db`).
* `--angle-flat-fallback/--angle-flat-threshold-db`,
  `--angle-delay-radius-bins/--angle-doppler-radius-bins/--angle-min-snapshots`
  (cells use ±1 delay × ±2 Doppler bins).
* `--mvdr-covariance/--mvdr-background-percentile` (MVDR beams always use the
  quietest 80 % of cells).
* `--frame-hop-sweeps` became `--frame-step`.

Behaviour changes:
* Array plots no longer read raw TBN; export first.
* The auto look direction is found from the strongest cells, not the whole
  window, and is logged to `look_directions.csv` in videos.
* The "virtual range" axis and km rows were removed, because delays are relative.
* Phase-Doppler estimate times are now centred on the sweeps they use
  (previously half a sweep early).
* `LFMWaveform.waveform` always has exactly `sweep_length` samples.
