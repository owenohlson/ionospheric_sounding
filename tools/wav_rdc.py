"""Streaming WAV ingestion. Two audio tracks are I/Q, never X/Y polarization."""

from datetime import datetime, timedelta, timezone
import os

import h5py
import numpy as np
import soundfile as sf

from rdc_utils import estimate_rdc0_bytes, format_bytes, metadata_json, polarization_metadata


def export_wav_rdc0(args, output_file, temporary=False):
    if args.pol != "both" or args.stands != "all" or args.drop_bad_stands:
        raise ValueError("WAV is one receiver with unknown polarization; omit LWA stand/pol selection")
    epoch = None
    if args.start_timestamp:
        epoch = datetime.fromisoformat(args.start_timestamp.replace("Z", "+00:00"))
        epoch = epoch.replace(tzinfo=timezone.utc) if epoch.tzinfo is None else epoch.astimezone(timezone.utc)
    with sf.SoundFile(args.input_file) as wav:
        if wav.channels not in (1, 2):
            raise ValueError("WAV must be mono real samples or stereo I/Q (I first, Q second)")
        fs = wav.samplerate
        period_samples = fs / args.sweep_frequency
        sweep = int(round(period_samples))
        if sweep < 1 or not np.isclose(sweep, period_samples, rtol=0, atol=1e-6):
            raise ValueError("Sample rate / sweep frequency must be an integer number of samples")
        start = int(round(args.tstart * fs))
        stop = int(round(args.tend * fs))
        if not 0 <= start < stop <= len(wav):
            raise ValueError("Requested time range lies outside the WAV recording")
        # Without UTC, alignment remains anchored to WAV sample zero, not tstart.
        clock = start / fs
        if epoch is not None:
            clock += epoch.timestamp()
        shift = 0 if args.no_timestamp_align else int(round((args.offset - clock) % (1 / args.sweep_frequency) * fs)) % sweep
        start += shift
        count = sweep - args.adc_start if args.adc_count is None else args.adc_count
        if args.adc_start < 0 or count < 1 or args.adc_start + count > sweep:
            raise ValueError("--adc-start/--adc-count must select samples within a sweep")
        pulses = (stop - start) // sweep
        if pulses < 1:
            raise ValueError("No complete sweeps remain after alignment")
        dtype = args.rdc0_dtype if temporary else args.dtype
        compression = args.rdc0_compression if temporary else args.compression
        level = args.rdc0_compression_level if temporary else args.compression_level
        print(f"WAV RDC0 shape: ({count}, {pulses}, 1); {format_bytes(estimate_rdc0_bytes(count, pulses, 1, dtype))}", flush=True)
        if args.dry_run_size and not temporary:
            return
        kwargs = dict(compression=None if compression == "none" else compression)
        if compression == "gzip":
            kwargs["compression_opts"] = level
        with h5py.File(output_file, "w") as h5:
            grp = h5.create_group("rdc0")
            real = grp.create_dataset("real", (count, pulses, 1), dtype=dtype, chunks=(min(count, 8192), 1, 1), **kwargs)
            imag = grp.create_dataset("imag", (count, pulses, 1), dtype=dtype, chunks=real.chunks, **kwargs)
            invalid = grp.create_dataset("invalid_counts", (pulses, 1), dtype="int64")
            wav.seek(start)
            for pulse in range(pulses):
                samples = wav.read(sweep, dtype="float32", always_2d=True)
                iq = samples[:, 0].astype(np.complex64)
                if wav.channels == 2:
                    iq.imag = samples[:, 1]
                iq = iq[args.adc_start:args.adc_start + count]
                bad = ~np.isfinite(iq)
                invalid[pulse, 0] = np.count_nonzero(bad)
                if args.gap_fill == "zero":
                    iq[bad] = 0
                real[:, pulse, 0], imag[:, pulse, 0] = iq.real, iq.imag
            grp.create_dataset("channel_stands", data=[1])
            grp.create_dataset("channel_pols", data=[-1])
            grp.create_dataset("adc_sample_index", data=np.arange(args.adc_start, args.adc_start + count))
            grp.create_dataset("pulse_index", data=np.arange(pulses))
            h5.attrs["metadata_json"] = metadata_json(
                rdc_stage=0, dimension_order=["adc", "pulses", "channels"],
                source_format="wav", source_file=os.path.basename(args.input_file),
                wav_layout="iq" if wav.channels == 2 else "real", station=None,
                channel_id_kind="receiver", cable_phase_corrected=False,
                sample_rate_hz=fs, sweep_frequency_hz=args.sweep_frequency,
                bandwidth_hz=args.bandwidth, tuning_frequency_hz=args.carrier_frequency,
                tstart=args.tstart, start_offset_samples=shift,
                pulse0_utc=(epoch + timedelta(seconds=start / fs)).isoformat() if epoch else None,
                sweep_offset_s=args.offset, timestamp_aligned=epoch is not None and not args.no_timestamp_align,
                sweep_length_samples=sweep, adc_start=args.adc_start, adc_count=count,
                duration_requested_s=args.tend - args.tstart, n_channels=1, n_stands=0,
                stands=[], pol="unknown", **polarization_metadata([-1]),
                gap_fill=args.gap_fill, dtype=dtype, compression=compression,
                compression_level=level,
            )

