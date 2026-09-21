# Exporting RDC0, RDC1, and RDC2

Use one exporter for TBN (`.dat`/`.tbn`) and WAV recordings:

| Command | Input | Output axes |
| --- | --- | --- |
| `rdc_export.py --stage 0` | TBN or WAV | ADC samples, pulses, channels |
| `rdc_export.py --stage 1` | TBN, WAV or `--rdc0-input` | delay, pulses, channels |
| `rdc_export.py --stage 2` | TBN, WAV, `--rdc0-input`, or `--rdc1-input` | delay, Doppler, channels; explicit integration/hop adds a leading frame axis |

The old `tbn_export_rdcN.py` files remain tiny compatibility launchers, not
separate implementations. The shared entry point defaults to stage 2. Example:

```bash
python tools/rdc_export.py --stage 0 recording.dat \
  --output recording_rdc0.h5 --duration 30 --sweep-frequency 2
```

Only flags relevant to the selected output stage are exposed. Use
`python tools/rdc_export.py --stage 0 --help` for the RDC0 options.

RDC1 includes dechirping, the fast-time window, and the range FFT. It preserves
complex per-sweep values; no Doppler FFT, antenna combining, or cable correction
is added by this export step. RDC2 input processing never dechirps RDC1 again.
These are forward processing stages, not reversible format conversions.

## Example pipeline

Run from the repository root, using your Python environment:

```bash
python tools/rdc_export.py --stage 0 recording.dat \
  --output recording_rdc0.h5 --duration 30 --sweep-frequency 2 \
  --drop-bad-stands --pol x --gap-fill zero

python tools/rdc_export.py --stage 1 --rdc0-input recording_rdc0.h5 \
  --output recording_rdc1.h5 --bandwidth=-99.259377e3 \
  --d-min 217.5 --d-max 219.5 --dechirp-window hann --gap-fill zero

python tools/rdc_export.py --stage 2 --rdc1-input recording_rdc1.h5 \
  --output recording_rdc2.h5 --integration-time 10 --hop-time 5 \
  --slow-window hann --nfft-doppler 64
```

For direct TBN/WAV to RDC1, replace `--rdc0-input ...` with the raw filename and
provide `--sweep-frequency`, `--bandwidth`, and the desired time selection.
This route creates and subsequently removes a temporary RDC0 file.

Omitting `--d-min`/`--d-max` keeps all available delay bins. Narrow selections
reduce storage. Use `--dtype float16` for lossy reduced-precision storage,
`--compression gzip|lzf|none`, and `--compression-level` for gzip. Float32 is
the default. `--dry-run-size` reports uncompressed sizes; for raw TBN it still
needs temporary RDC0 extraction. Channel chunks limit RAM usage.

## Metadata and compatibility

All stages use HDF5 datasets `/rdcN/real` and `/rdcN/imag`, plus explicit
`channel_stands` and `channel_pols` arrays. Never infer polarization or stand
identity from the number or parity of channel indices. MATLAB can read the
datasets with `h5read`; Python can use `h5py`. These are HDF5 files, not MAT
files. Consult dataset shapes and axis vectors when arranging MATLAB arrays.

Root attribute `metadata_json` contains the stage, frequency in Hz, sample
rate, sweep rate, channel identities, polarization, processing choices, and
cable-correction state. RDC1/RDC2 also preserve the source metadata under
`source_metadata`. Existing RDC0 files without a stage attribute still work.

With `--rdc1-input`, the RDC2 exporter inherits bandwidth, sweep rate, ADC
selection, dechirp window, and reference gating. Conflicting overrides raise
an error instead of relabeling previously processed data. You may choose a
smaller delay/channel/time subset and new Doppler processing settings.
Omitted `--gap-fill` inherits the source policy when available, otherwise nan.

Selection times are relative to the input cache's first pulse. An RDC1 time
origin is carried forward when selecting a subinterval for RDC2. The original
RDC0 extraction/alignment metadata remains available in the provenance.

RDC2 retains its selected RDC1 by default, so existing array plotting tools can
change integration/hop later. `--no-store-rdc1` omits it. `--rdc1-input` can also
read retained RDC1 from an existing RDC2 file. Use the resulting RDC2 with the
plotting scripts' `--rdc2-input` flag. Single-receiver plots now also accept
RDC0 and RDC1 directly; see [PIPELINE.md](PIPELINE.md).

## Gaps

Prefer `--gap-fill zero` before the range FFT if you want to keep unaffected
samples in a sweep containing a gap. Changing NaNs to zero after an FFT cannot
recover values already contaminated by that FFT. The exporter records fill
counts at the stage where it replaces nonfinite values and warns when filling
RDC1 values. For best recovery, re-export from RDC0 with zero fill.

## Readers

`rdc_utils.py` provides `load_rdc0_hdf5_metadata`, `load_rdc1_hdf5_metadata`,
and `load_rdc2_hdf5_metadata`, with matching channel-chunk iterators. The RDC1
reader validates axes and metadata and accepts both standalone and retained
RDC1. Use the explicit `delay_ms` axis after cropping, not a reconstructed axis
based on the reduced number of bins.

## Code ownership

- `rdc_export.py`: shared CLI, export orchestration, and atomic output handling.
- `rdc_utils.py`: cube transformations and HDF5 readers/writers.
- `wav_rdc.py`: streaming mono-real/stereo-IQ WAV ingestion; no LSL dependency.
- `rdc_plot_input.py` and `rdc_plot.py`: shared single-receiver HDF5 selection
  and plotting CLI; each plotting script is only a launcher.
- `tbn_utils.py`: the canonical TBN opener and timestamp probe, also used by
  the array plotting commands. The exporter imports these only for raw TBN;
  cached HDF5 processing with physical channel IDs does not require the TBN reader.

Temporary RDC0 files and raw reader handles are closed/removed even if a later
processing stage fails. Final outputs replace existing files only after the
new HDF5 file is written and closed successfully.

## WAV specifics

Stereo means I in track 1 and Q in track 2, producing ONE complex receiver
channel. Mono is stored as real samples with zero imaginary part; no analytic
signal conversion is inferred. Other channel layouts are rejected. SoundFile
decodes PCM to normalized floating-point samples and preserves float WAV values.

WAV receivers use `channel_id_kind=receiver`, polarization `unknown` (ID -1),
and no station/cable metadata. Do not pass LWA stand/pol filters at WAV ingestion.
The virtual channel ID 1 is not physical LWA stand 1. Beamforming requires
multiple physical channels and geometry, so array plotting rejects these files.

Ordinary WAV has no reliable absolute UTC timestamp. `--start-timestamp`
optionally supplies sample-zero UTC (ISO 8601; a timezone-free value means UTC).
`--carrier-frequency` optionally supplies RF frequency in Hz. Missing values
remain unknown, never default to an LWA carrier or invented GPS time. Automatic
WAV alignment without UTC is anchored to sample zero, including when tstart
changes; `--offset` shifts that reference. `--no-timestamp-align` disables this
alignment. The exporter keeps only complete sweeps and records any start shift.
WAV currently requires an integer number of samples per sweep.

New RDC0 and retained RDC1 contain `invalid_counts` by sweep/channel. This
preserves gap annotations even after zero filling. Older zero-filled caches
without this dataset cannot localize gaps reliably; their actual zero samples
are not assumed to be corrupt. New caches also carry `pulse0_utc` when known.
