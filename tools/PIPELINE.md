# Processing pipeline

```
raw TBN (preprocess_tbn.py: report / cut, optional) ┐
raw WAV (mono real / stereo I/Q) ───────────────────┤
                                             ▼
                                   export_rdc0.py   RDC0  (adc, pulses, channels)      raw sweeps
                                             ▼
                                   export_rdc1.py   RDC1  (delay, pulses, channels)    matched filter (or dechirp + FFT)
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
| `lfm_utils.py` | Maths only: reference chirp and windows, range compression (circular matched filter, or dechirp + FFT), delay axis, Doppler FFT, pulse-pair (phase) Doppler, sweep alignment, STFT, LFM estimation from a spectrogram, waveform-scan kernels (lag-product spectrum, compression profiles) |
| `lwa_utils.py` | Everything LWA-specific: the station description (LSL's bundled LWA1 by default, `--ssmif` for a dated file or `lwasv`), stand positions, cable delays and phase factors, digitizer mapping, bad stands (SSMIF status), legacy channel labels; TBN frame-header scan, quality summary, `TBNReader` (samples placed by timetag, gaps as NaN, exact 196 MHz timestamps; LSL is used only for station metadata) and trimming for `preprocess_tbn.py` |
| `wav_utils.py` | WAV only: info, block reader, UTC parsing |
| `rdc_io.py` | HDF5 layout: `RDCFile` reader (all stages, legacy formats), `StageWriter`, metadata/history, atomic output |
| `rdc_utils.py` | Stage-to-stage processing: channel selection, sweep packing, `RDCView` (brings any stored stage to the product a plot needs), exports, array processing (`ArrayProcessor`, covariance, steering, angle spectra, beam pattern), plot reductions |
| `calibration_utils.py` | Array self-calibration: plane-wave fit, phase-gain estimation (alternating fit, ramp kept out), held-out validation, the calibration file (`ArrayCalibration`) |
| `codar_stations.py` | HF radar source identification: the built-in HFRnet station table (`resources/hfrnet_stations.csv`, updated with `--add`), bearings and sky-wave geometry from the station, waveform/band/date filters, the labelling rule, hop count from elevation (`--hops CODE ELEV`) |
| `summary_utils.py` | The shared processing summary: the findings log (`processing_summary.csv` next to the plots) and its merge into one row per signal (`processing_summary_merged.csv`) |
| `map_utils.py` | Maps without a mapping library: the shipped outline file (`resources/map_outlines.npz`, Natural Earth; rebuilt with `--build`), Lambert conformal projection, great-circle and n-hop reflection points |
| `plotting_utils.py` | Figures only (arrays → PNG) and video assembly |
| `cli.py` | Shared command-line flag groups and the export command line |


Files the scripts read are in `tools/resources/`, so copying `tools/` is all a
machine needs:

| File | Used by | What |
| --- | --- | --- |
| `hfrnet_stations.csv` | codar_stations, scan_lfm, sky maps, station map | HFRnet station table (`codar_stations.py --add` merges new exports) |
| `map_outlines.npz` | station map | Natural Earth coastlines, borders, states (`map_utils.py --build`) |
| `giro_stations.txt` | station map | GIRO ionosonde list (DIDBase Fast Station List) |
| `giro/<URSI>_<date>.txt` | station map, `ionosonde.py` | GIRO FastChar characteristics for the hop-count estimates |
| `ssmif/LD021_0001.tgz` | every LWA command (`$LWA_SSMIF`) | station description for 2023-12-09 |

## Commands and the inputs they accept

| Command | RDC0 | RDC1 | RDC2 | RDC2P | RDC3 |
| --- | :-: | :-: | :-: | :-: | :-: |
| `plot_spectrogram.py` (`--average` for a mean spectrum) | ✓ | | | | |
| `plot_power_delay_profile.py` (delay vs time, plus its time average) | ✓ | ✓ | ✓ * | | |
| `plot_delay_doppler.py` / `sliding_delay_doppler.py` | ✓ | ✓ | ✓ | | ✓ |
| `plot_doppler_time.py` | ✓ | ✓ | ✓ * | | |
| `plot_phase_doppler.py` (`--kind delay-time` or `doppler-time`) / `sliding_phase_doppler.py` | ✓ | ✓ | ✓ * | ✓ | |
| `calibrate_array.py` | ✓ | ✓ | ✓ | | |
| `check_array_calibration.py` | ✓ | ✓ | ✓ | | |
| `scan_lfm.py` (which chirp bandwidths and sweep rates compress) | ✓ | | | | |

`plot_station_map.py` (receiver, stations, paths, midpoints and reflection points) needs no recording.

\* uses the RDC1 retained inside the RDC2 file.

Each command uses the most-processed stored stage that still fits the request.
For example, a delay-Doppler plot of an RDC2 uses the stored frames. It recomputes
from the retained RDC1 only if you change `--integration-time`, `--hop-time`,
`--slow-window` or `--nfft-doppler`.

**Raw input.** Every plot (and `calibrate_array.py`, `export_rdc3.py`) also takes a
raw TBN/WAV file. It is first cut into a temporary file in `$SLURM_TMPDIR` (else
the system temp directory), deleted when the plot finishes: RDC0 for
`plot_spectrogram.py`, RDC1 for the rest. Give the same raw-input flags as the
exports (`--sweep-frequency`, `--offset`, ...; plus `--bandwidth` to dechirp).
`plot_spectrogram.py` needs no `--sweep-frequency`: it then reads 1 s blocks from
the first sample (the info table shows "Sweep freq 1 Hz", and up to 1 s at the
end is dropped). Each run re-reads the recording, so for repeated plots of the
same data export an RDC1 once.

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
  cable delays and status come from LSL's bundled LWA1 SSMIF unless
  `--ssmif FILE` (an SSMIF or an observation metadata `.tgz`) or `$LWA_SSMIF`
  gives one for the observation date; `--ssmif lwasv` uses LSL's bundled LWA-SV
  file (e.g. for the 2019 Santa Fe recordings, which are LWA-SV). The 2023-12-09
  CODAR session (LD021) is LWA1: its `LD021_0001.tgz` SSMIF is station `VL` at
  34.0689 N, 107.6284 W. Still use the dated file: the bundled one is not specific
  to the observation date. Files store each channel's digitizer, so a dated SSMIF
  re-derives stand identities without re-exporting, and files made before LWA1
  became the default (they record "LSL bundled LWA-SV" or nothing) are re-labelled
  from their digitizers when read.
* **Power scale.** Plots are relative, as in Peter's `lwa_radar.py`: delay-Doppler maps,
  power-delay profiles, Doppler-time and phase-Doppler plots in dB above their own median
  (the noise floor; colour range 0 dB to the map's peak, at least 10 dB), spectrograms
  and spectra in dB below their peak (−40 to 0 dB), sky maps in dB below their peak.
  `--vmin/--vmax` are on those scales. The absolute numbers (counts² times processing gains)
  change with every setting, so they are not drawn; SNRs in the summary are unaffected.
* **Delay window.** The default is ±1.5 ms around the strongest delay in a
  preview; `--d-min/--d-max` set it explicitly and `--full-delay` shows
  everything stored. Delays are relative to the sweep reference timing, not
  physical ranges.
* **Range compression.** By default (`--range-method matched`, since 2026-11-21) each
  whole sweep is correlated circularly with the reference chirp. A CODAR sawtooth
  repeats every sweep, so a received sweep is the reference rotated by the echo delay
  and every delay from 0 to one period compresses fully, also when |B| > fs. The
  reference is zeroed where its instantaneous frequency is outside ±0.98·fs/2 (TBN has
  filtered that part away), and `--dechirp-window` (default Hann) spans only the
  in-band part (all of the sweep at 99.26 kHz; 87.5–412.5 ms of 500 ms at 150.7 kHz).
  Delay bins are 1/fs (10 µs) apart. This is the matched filter of Peter's
  `lwa_radar.py` (same output, up to scale).
  `--range-method dechirp` is the older stretch processing (multiply by the reference,
  FFT). With the echo far into the sweep (e.g. 218 ms of 500 ms) and |B| ≠ fs it loses
  up to ~4 dB and leaves a copy at +(fs−|B|)/|k|; with |B| > fs some delays are not
  reachable at all. Files made before 2026-11-21 are dechirped (no `range_method` in
  their metadata); commands keep the method a file was made with.
* **Sweep offset.** `--offset` moves the sweep grid (e.g. to put echoes near 0 ms). It
  is only needed with `--range-method dechirp`. The offset is recorded in the metadata.

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

The value is an SSMIF text file, an observation metadata `.tgz`, or `lwa1`/`lwasv`
for the files bundled with LSL (default `lwa1`). Exports need it
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
python tools/plot_power_delay_profile.py test_rdc0.h5 --full-delay --output pdp.png
```

The spectrogram estimate confirms the sweep rate and bandwidth of the strongest
ridge. It cannot see a weaker transmitter on the same frequency (24.55 MHz on
2023-12-09 carries a -150.727 kHz and a -99.259 kHz waveform), so scan for every
waveform that compresses:

```bash
python tools/scan_lfm.py raw.dat --stands 1,2,20,60,100,150,200,250 --pol x --duration 20 \
    --output scan.png --json scan.json
```

Station-table waveforms come first; a waveform is worked out blind only when
nothing in the table fits.

1. **Lag-product spectrum (blind).** The lag product x(t)·x*(t−lag) of a chirp is a
   tone at the chirp rate k = bandwidth × sweep rate, so its peaks fix k but not
   the sweep rate. CODARs are gated (FMICW: on about half of every ~0.5–1 ms), and
   the tone only appears while the transmitter is on at both t and t−lag. At some
   lags that overlap is poor and the power moves to sidelines at k ± n·f_gate/lag
   (on 060287_010275465 the 10 ms lag missed PCYC's −301.4 kHz/s entirely). The
   sidelines move with the lag and k does not, so the spectrum is taken at 5, 7,
   11, 13 and 17 ms (`--lags`; no common factor, so sidelines of different lags
   rarely coincide) and the median over lags is kept.
2. **Sweep rate from the resets.** For each lag peak the scan dechirps with a
   continuous chirp at k and follows the beat in 25 ms windows. The beat holds
   steady through a sweep and jumps when the transmitter resets; the coherence of
   the reset times over trial rates of 0.2–5 Hz picks the period (the lowest rate
   within 80% of the best, since multiples line up too), and a line fit of reset
   time against sweep number gives the rate and its error.
3. **Match to the station table.** The HFRnet stations active on the date whose
   sweep overlaps the band (and any `--candidates`) are matched to each peak by
   chirp rate (within 0.3% or 3 lag bins, `--match-tolerance`). A match is used
   exactly as documented: bandwidth and sweep rate. Its bandwidth is still
   scanned ±100 Hz once it is found, but only as a check drawn in the plot (on
   060287_010274515 both scans peak at the table value). The reset measurement is
   then a consistency check, reported in the summary. When documented waveforms
   of different sweep rates share the chirp rate (−99.26 kHz at 2 Hz and
   −49.63 kHz at 4 Hz), the measured rate picks between them, or the dechirped
   test when the resets cannot be measured. Documented waveforms that match no
   lag peak are dechirped too, in case the lag spectrum missed a weak transmitter.
4. **Not in the table.** A lag peak with no match takes its measured sweep rate,
   snapped to a CODAR rate within 3σ and 2% (the rare 2.007735 Hz of the OSU
   12.2 MHz stations only when 2 Hz does not fit), or the dechirped test if the
   resets cannot be measured. Its bandwidth is then scanned ±100 Hz on whole
   sweeps (`--fine-span`), since the lag peak fixes it only to some tens of Hz and
   whole sweeps need it to ~1 Hz. These are flagged "not in table".
5. **Present or not.** Each waveform is dechirped on 10 s of whole sweeps. It is
   present when the delay profile peaks (contrast ≥ `--min-contrast-db`) and the
   peak needs that bandwidth: the contrast must drop ≥ 3 dB at ±50 Hz. That
   rejects profiles whose floor alone gives a high contrast.

The dechirped test: when the sweep is wider than the 100 kHz band, the same chirp
rate also compresses at other sweep rates with the bandwidth scaled to keep k.
On 25 ms pieces tiling a sweep, the true rate puts every piece's echo in one
delay bin; a wrong rate splits it between delays. Its result for every rate is
printed as the "cross-check" line of each waveform and kept in the JSON.

The plot has four panels: the lag-product spectrum (waveforms found in colour,
table waveforms dotted, lag peaks as triangles), the reset coherence against trial
sweep rate, the bandwidth scan, and the delay profile (echo delays labelled,
copies x).

On 060287_010274515 (13.5 MHz) the two lag peaks match the table: −99.259 kHz at
2 Hz (ARG1/DCSR/ESTR/…; resets 1.998 ± 0.002 Hz) and −51.468 kHz at 2 Hz (BML1,
GCVE, PPNS, PREY, SLID; resets 1.98 ± 0.01 Hz). The other two lag peaks and the
seven table waveforms with no lag peak do not compress. Use the bandwidths it
reports for `--bandwidth`.

The whole-sweep profiles use the matched filter, so every delay is reachable. (The
25 ms pieces of the rate test are dechirped, which is what that test needs.)

Copies are not echoes. The compressed delay profile has weaker copies of each
echo at the same Doppler: gating sidebands in symmetric pairs at ±s around it
(±5.18 ms for the 13.5 MHz CODARs, ±10.31 ms for BML), and, in dechirped data only,
the sweep-wrap replica at +(fs−|B|)/|k| when |B| ≠ fs. The scan marks them with an x, lists
them under their echo, and does not count them as echoes; plot_delay_doppler
does the same with the strongest cells, and the summary folds any that slip
through into the echo's `copies` column.

`scan_lfm.py` removes the comb from its delay profiles by default (the same CLEAN on the
whole-sweep matched filter, every order that fits in half a sweep; `--keep-sidebands` turns it
off). The plot draws the profile before the CLEAN faintly behind it, and marks with x only the
copies still in the profile. The comb spacing is measured to a fraction of a lag bin (a parabola
through each first-order peak), so the predicted position of order n stays on its peak; Peter's
`detect_comb` keeps whole bins, which puts the higher orders a few bins off.

To remove the gating comb instead of only labelling it, add `--suppress-sidebands` when
exporting RDC1 from the raw file (a port of `detect_comb` and the adaptive `sideband_clean`
in Peter's `lwa_radar.py`; matched filter only). The first 32 sweeps of 8 channels find the
strongest echo (or the one near `--sideband-echo MS`), the comb spacing from the symmetric
pair around it (1.5–40 ms), and each order's level and coherence. Then, per block of
`--sideband-block` sweeps (default 20) and per channel, every sideband bin's complex ratio to
the echo is measured and the echo times that ratio is subtracted. Only what moves exactly with
that echo goes: another echo at a sideband delay with a different Doppler decorrelates over
the block and stays, one with the same Doppler would be partly removed. An echo exactly n
comb spacings from the cleaned one also loses r(n)·r(−n) of its amplitude (its own copy sits
on the cleaned echo; about −0.9 dB when the orders are 10 dB down). One echo's comb is
removed per export; the comb found is recorded in `sideband_suppression` in the metadata.

The full-delay profile shows where the echoes are. Then choose `--d-min/--d-max`.

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
python tools/plot_power_delay_profile.py s13p5_rdc1.h5 --stands all --navg 4 --output pdp.png  # delay vs time + mean
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
(`tools/resources/hfrnet_stations.csv`); give CSV paths after the flag to use other tables
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

**Hop count.** For each candidate the panel shows the hop count and virtual
height the measured elevation implies (mirror model, spherical Earth), e.g.
`1F 257km or 2E`. More hops mean a lower reflection, so the choice is the hop
count whose height is plausible: F region (180–450 km) first, then E / low F1
(90–180 km), then either only within the elevation tolerance (marked `?`).
Other hop counts that also fit are listed after "or": elevation alone cannot
tell one F hop from two E hops, and the group delays of the alternatives differ
by only ~0.1 ms. For any station and elevation:

```bash
python tools/codar_stations.py --hops FOCF 10.3        # 1 hop needs 560 km -> 2 F hops at 212 km
```

**Map of the paths.** `plot_station_map.py` draws the receiver (LWA1 by default),
the stations, their great-circle paths, the one-hop midpoints, and the sky
reflection points and ground bounces of n-hop paths. With `--from-summary` the
stations and their elevations come from the processing summary: every signal that
array processing gave an azimuth/elevation (sky-map peaks, or the beam look
direction) is matched to the stations whose waveform, band and date fit its file
and whose bearing and a 1–3 hop path fit its direction (the same test as the sky
maps), and each is drawn with the hop count its elevation gives. Paths are
coloured per network and hop count, with the elevation range and number of
signals in the legend. Run it after the delay-Doppler jobs; `--files 4515` and
`--min-snr` narrow it down. Stations can also be given by hand (`CODE@ELEV` or
`CODE:HOPS`):

```bash
python tools/plot_station_map.py --from-summary --min-snr 10 \
    --output ../plots/paths.png --csv ../plots/paths.csv
python tools/plot_station_map.py --from-summary --files 5465 --band 24.55 --bandwidth -150727.2 \
    --sweep-rate 2 --date 2023-12-09 --output ../plots/paths_5465.png    # plus every matching station, in grey
```

**Ionosondes (GIRO).** The GIRO ionosondes inside the map are drawn as stars, from
`tools/resources/giro_stations.txt` (the DIDBase Fast Station List). With GIRO
FastChar files in `tools/resources/giro/` (the default `--iono-data`), the hop count comes from
the measured ionosphere instead of a fixed range of layer heights. Each file is one
station and day, saved as `<URSI>_<date>.txt` from

    https://lgdc.uml.edu/fastchar/getbest?ursiCode=BC840&charName=foF2,foF1,foE,foEs,MUF%28D%29,M%28D%29,hmF2,hmF1,hmE,B0,B1&DMUF=3000&fromDate=2023%2F12%2F09+15%3A30%3A00&toDate=2023%2F12%2F09+19%3A00%3A00

For each path and hop count, `tools/ionosonde.py`:

1. takes the ionosonde nearest any of the hop's reflection points (within 1500 km)
   at the recording's time, interpolated between its 5-minute soundings, dropping
   low-confidence and mis-scaled rows;
2. builds an electron-density profile from its characteristics: a Chapman E layer,
   an F1 parabola, and the IRI F2 bottomside from hmF2, B0 and B1 (no magnetic field);
3. finds the mirror heights where the virtual height of the equivalent vertical
   frequency f·cos(φ) equals the reflection height (secant law with Martyn's theorem,
   spherical Earth). Per layer there is a low ray and a high ray, and none above
   that hop's MUF.

The hop count drawn is the one whose predicted elevation is within 5° of the
measured one, closest first. Without an ionosonde near the path, or when nothing fits,
the fixed-height mirror model is used and the path table says so. The table and the
summary note name the ionosonde used, which is needed for the acknowledgement GIRO's
CC-BY-NC-SA 4.0 terms require. On 2023-12-09 only Boulder (BC840) and Austin
(AU930) have data, about 950–1250 km from most reflection points, so the profiles
are representative rather than local. Run `python tools/ionosonde.py --data
tools/resources/giro --path RXLAT,RXLON,TXLAT,TXLON --freq 13.5 --time 2023-12-09T16:00Z
--elevation 21` for one path's options.

Coastlines, borders and states come from `tools/resources/map_outlines.npz` (Natural
Earth 1:50m, public domain, cut to North America). To rebuild or widen it from
the Natural Earth downloads: `python tools/map_utils.py --build ne_50m_coastline.zip
ne_50m_admin_0_boundary_lines_land.zip ne_50m_admin_1_states_provinces_lines.zip`.

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

## Everything for every file (cluster)

`slurm/slurm_all.sh` reruns the standard plots for all ten 2023-12-09 files, one
after another, in one job: `sbatch slurm_all.sh`. Its log is
`slurm/out/slurm_all.out` and is replaced on each run. Create `slurm/out` once:
SLURM does not create the folder, and a job whose log can't be opened fails without
output. Per file:

1. spectrogram of the first 5 s;
2. `scan_lfm` (with `--json`);
3. for each strong waveform (best echo SNR ≥ 15 dB, at most 3 per file, picked by
   `tools/strong_waveforms.py`): an RDC1 holding only the delays around its echoes,
   named by its waveform (`<file>_rdc1_bw99p259_fsweep2.h5`); the PDP; Doppler vs
   time; the x-pol calibration (`<file>_x_cal_<tag>.h5`); and the sliding array
   delay-Doppler video with sky maps and station identification.

Then it redraws the station maps (all files, and one per file) from the array
directions in the summary, and prints the merged summary. Steps that finished in
an earlier run are skipped, so a job that hits its time limit can simply be
submitted again. To redo steps or run a subset:

    sbatch --export=ALL,SKIP_EXISTING=0 slurm_all.sh
    sbatch --export=ALL,FILES="060287_010274515 060287_010275465" slurm_all.sh

A failed step is reported, the steps that need its output are skipped, and the rest
carry on.

Existing files are reused (`REUSE=0` turns this off). An RDC1, or an RDC2 that kept
its RDC1, already in `../data` is used instead of a new export when it matches. To
match, it must be the same recording with the same bandwidth (within 1 Hz) and sweep
rate, cover the waveform's delays, hold x-pol channels and the whole file (no time
cut), and have been made with the same station description. A file's existing
`<file>_x_cal.h5` is used for every waveform of that file, because the stand gains
do not depend on the waveform. `python tools/rdc_info.py ../data/*.h5` lists what
each file holds (stages, bandwidth, sweep rate, delays, channels, station
description), and `--full` adds every metadata key and the processing history:

    python tools/rdc_info.py ../data/*.h5
    python tools/rdc_info.py --find ../data --file 060287_010274515 --bandwidth -99259.377 \
        --sweep-rate 2 --d-min 12.7 --d-max 22.1 --verbose    # what would be reused, and why not

Outputs (`/project/6003738/onohlson/`):

```
plots/2023-12-09/
  processing_summary.csv, processing_summary_merged.csv   every finding / one row per signal
  station_map_all.png, station_map_all.csv                paths of every file's signals
  .done/                                                  markers of finished steps
  <file>/                                                 e.g. 060287_010274515/
    <file>_spectrogram.png
    <file>_lfm_scan.png, <file>_lfm_scan.json             waveforms, echoes, SNR
    <file>_pdp_<tag>.png                                  one per strong waveform (tag: bw99p259_fsweep2)
    <file>_doppler_time_<tag>.png
    <file>_x_ADD_mvdr_<tag>/                              sliding array delay-Doppler
      delay_doppler.mp4, look_directions.csv
      angle_plot/angle_maps.mp4                           sky maps
    <file>_station_map.png, <file>_station_map.csv
data/
  <file>_rdc1_<tag>.h5                                    RDC1 of one waveform, delays around its echoes
  <file>_x_cal_<tag>.h5, <file>_x_cal_<tag>.csv           x-pol calibration
slurm/out/slurm_all.out                                   the job's log
```

## Processing summary

`scan_lfm.py`, `plot_delay_doppler.py` / `sliding_delay_doppler.py` and
`plot_station_map.py` add what they find to `processing_summary.csv` in the folder
of their `--output` (the parent folder for videos), so every plot written to one
folder shares one summary. `--summary PATH` or `$LFM_SUMMARY` pick another file,
`--summary none` skips it. Re-running a command with the same input and output
replaces its earlier rows; several jobs can write at once.

The log has a few rows per file and command:

| kind | from | what |
| --- | --- | --- |
| `waveform` | scan_lfm | bandwidth, sweep rate (measured, with how), echo delay and SNR, its copies, HFRnet stations with that waveform in the band |
| `echo` | delay-Doppler, sky maps | one row per echo for the whole run: median delay, Doppler (with its range), SNR, direction (sky-map peaks, or the beam look direction) with its spread, the station named most often, `n_frames` |
| `path` | station map | station, coordinates, distance, bearing, hop count, virtual height, reflection points |

SNR (`snr_db`; merged: `scan_snr_db`, `dd_snr_db`) is 10·log10((P − N)/N): P the
echo's power (its delay-profile bin, or its delay-Doppler cell) and N the mean
noise power per bin, taken from the bins within ±25 ms of delay of the echo,
leaving out ±1 ms around every echo and copy and anything above 10× their median.
N is local because the profile's floor follows the receive filter and varies by
several dB across the sweep. The scan's contrast (peak over the profile median) is
kept in its notes.

A delay-Doppler run collects the 3 strongest cells and the sky-map peaks of every
frame, then consolidates them before writing: gating/sweep-wrap copies are
removed frame by frame, cells within 0.5 ms of delay are one echo whatever the
frame (the exact time in the file is not kept, only the first and last frame),
echoes seen in fewer than 20% of the frames are dropped, and at most 5 are kept
per file and waveform. Logs written before this are compacted the next time
anything records into them.

After each write the log is merged into `processing_summary_merged.csv`: one row
per signal (file, bandwidth, sweep rate, echo delay within 1 ms mod the sweep),
with the scan contrast, the median delay-Doppler peak, Doppler, the median
direction and its spread over frames, the station and its path. Delays are put
on the UTC-aligned sweep grid (the export `--offset` is added back), so scan and
delay-Doppler delays agree. Mapped stations not tied to a recording get their
own rows. Gating and sweep-wrap copies (a weaker row at the same Doppler, at a
pair offset ±s or at the wrap offset from a stronger echo of the same waveform)
are folded into that echo's row and listed in its `copies` column rather than
kept as signals of their own. To read it:

```bash
python tools/summary_utils.py ../plots/processing_summary.csv --min-db 15
```

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

export LWA_SSMIF=tools/resources/ssmif/LD021_0001.tgz      # station description for 2023-12-09
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
| `dechirp_plot.py`, `plot_filter_output.py` (and the interim `plot_dechirp.py`, `plot_power_delay.py`) | `plot_power_delay_profile.py`: one figure with delay vs time and the time-averaged profile |
| `doppler_time_plot.py`, `spectrogram.py` | `plot_doppler_time.py`, `plot_spectrogram.py` |
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
