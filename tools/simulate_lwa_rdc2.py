"""Synthetic LWA-SV RDC2 (frames, delay, doppler, channels) with moving targets and stored truth.

Note: targets are injected with the same steering_vector the processing uses, so
this tests the plotting/tracking chain, not the steering convention itself
(tests/test_array_forward_model.py covers that with true time delays).
"""

import argparse
import json
import os

import numpy as np

from rdc_utils import polarization_summary as polarization_metadata, steering_vector
from lwa_utils import BAD_LWA_STANDS, cable_phase_factors, parse_stands, stand_positions as lookup_positions


def get_lwasv_stand_positions():
    positions, found = lookup_positions(range(1, 257))
    return {s: tuple(p) for s, p in zip(np.arange(1, 257)[found], positions)}


def metadata_json(**kwargs):
    return json.dumps(kwargs, indent=2, sort_keys=True)


TARGET_NAMES = ["box", "radial", "cross_a", "cross_b", "fader"]


def wrap_angle_deg(x):
    return ((x + 180.0) % 360.0) - 180.0


def piecewise_box(u, delay_min, delay_max, fd_min, fd_max):
    u = float(u % 1.0)
    if u < 0.25:
        q = u / 0.25
        return delay_min + q * (delay_max - delay_min), fd_min
    if u < 0.50:
        q = (u - 0.25) / 0.25
        return delay_max, fd_min + q * (fd_max - fd_min)
    if u < 0.75:
        q = (u - 0.50) / 0.25
        return delay_max - q * (delay_max - delay_min), fd_max
    q = (u - 0.75) / 0.25
    return delay_min, fd_max - q * (fd_max - fd_min)


def segment_fraction(frame_idx, n_frames, start, stop):
    if n_frames <= 1:
        return 0.0
    x = frame_idx / float(n_frames - 1)
    if x < start or x > stop:
        return None
    if stop <= start:
        return 0.0
    return (x - start) / (stop - start)


def target_state(name, frame_idx, n_frames, scenario, delay_min, delay_max, fd_min, fd_max):
    mid_delay = 0.5 * (delay_min + delay_max)
    delay_span = delay_max - delay_min
    fd_span = fd_max - fd_min
    use_all = scenario == "showcase"

    if name == "box" and scenario in ("showcase", "box"):
        u = segment_fraction(frame_idx, n_frames, 0.00, 0.36) if use_all else frame_idx / max(n_frames - 1, 1)
        if u is None:
            return None
        d, f = piecewise_box(u, delay_min + 0.15 * delay_span, delay_max - 0.15 * delay_span, fd_min + 0.20 * fd_span, fd_max - 0.20 * fd_span)
        return {
            "delay_ms": d,
            "doppler_hz": f,
            "az_deg": wrap_angle_deg(35.0 + 115.0 * u),
            "el_deg": 22.0 + 38.0 * u,
            "amplitude": 1.0,
        }

    if name == "radial" and scenario in ("showcase", "angle"):
        u = segment_fraction(frame_idx, n_frames, 0.22, 0.62) if use_all else frame_idx / max(n_frames - 1, 1)
        if u is None:
            return None
        return {
            "delay_ms": mid_delay + 0.10 * delay_span * np.sin(2.0 * np.pi * u),
            "doppler_hz": 0.12 * fd_span * np.sin(2.0 * np.pi * u),
            "az_deg": 150.0,
            "el_deg": 15.0 + 65.0 * np.sin(np.pi * u),
            "amplitude": 0.85,
        }

    if name == "cross_a" and scenario in ("showcase", "multi"):
        u = segment_fraction(frame_idx, n_frames, 0.48, 0.86) if use_all else frame_idx / max(n_frames - 1, 1)
        if u is None:
            return None
        return {
            "delay_ms": delay_min + (0.20 + 0.60 * u) * delay_span,
            "doppler_hz": fd_min + (0.25 + 0.50 * u) * fd_span,
            "az_deg": -80.0,
            "el_deg": 42.0,
            "amplitude": 0.95,
        }

    if name == "cross_b" and scenario in ("showcase", "multi"):
        u = segment_fraction(frame_idx, n_frames, 0.48, 0.86) if use_all else frame_idx / max(n_frames - 1, 1)
        if u is None:
            return None
        return {
            "delay_ms": delay_max - (0.20 + 0.60 * u) * delay_span,
            "doppler_hz": fd_max - (0.25 + 0.50 * u) * fd_span,
            "az_deg": wrap_angle_deg(85.0 + 45.0 * u),
            "el_deg": 30.0 + 12.0 * np.sin(2.0 * np.pi * u),
            "amplitude": 0.65,
        }

    if name == "fader" and scenario in ("showcase", "amplitude"):
        u = segment_fraction(frame_idx, n_frames, 0.72, 1.00) if use_all else frame_idx / max(n_frames - 1, 1)
        if u is None:
            return None
        return {
            "delay_ms": mid_delay - 0.18 * delay_span,
            "doppler_hz": 0.32 * fd_span * np.sin(2.0 * np.pi * u),
            "az_deg": wrap_angle_deg(-155.0 + 95.0 * u),
            "el_deg": 38.0 + 18.0 * np.sin(np.pi * u),
            "amplitude": 0.25 + 0.95 * (0.5 + 0.5 * np.sin(6.0 * np.pi * u)),
        }

    return None


def select_stands(stand_spec, max_channels, drop_bad):
    stands = parse_stands(stand_spec)
    if drop_bad:
        stands = [s for s in stands if s not in BAD_LWA_STANDS]
    stand_positions = get_lwasv_stand_positions()
    stands = [stand for stand in stands if int(stand) in stand_positions]
    if max_channels is not None and max_channels > 0:
        stands = stands[: int(max_channels)]
    if not stands:
        raise ValueError("No stands selected for synthetic RDC2 generation")
    return stands


def selected_pols(pol):
    p = str(pol).lower()
    if p in ("x", "0"):
        return [0]
    if p in ("y", "1"):
        return [1]
    if p == "both":
        return [0, 1]
    raise ValueError("--pol must be x, y, 0, 1, or both")


def grid_from_bounds(vmin, vmax, step):
    values = np.arange(float(vmin), float(vmax) + 0.5 * float(step), float(step), dtype=np.float32)
    if values.size < 2:
        raise ValueError("Grid bounds/step must produce at least two bins")
    return values


def gaussian_2d(delay_grid, fd_grid, delay_ms, doppler_hz, sigma_delay_ms, sigma_doppler_hz):
    dd = ((delay_grid[:, None] - delay_ms) / sigma_delay_ms) ** 2
    ff = ((fd_grid[None, :] - doppler_hz) / sigma_doppler_hz) ** 2
    return np.exp(-0.5 * (dd + ff)).astype(np.float32)


def inject_nan_patch(cube, frame_idx, delay_count, doppler_count):
    d0 = max(0, delay_count // 2 - max(1, delay_count // 12))
    d1 = min(delay_count, delay_count // 2 + max(2, delay_count // 12))
    f0 = max(0, doppler_count // 2 - max(1, doppler_count // 12))
    f1 = min(doppler_count, doppler_count // 2 + max(2, doppler_count // 12))
    cube[d0:d1, f0:f1, :] = np.nan + 1j * np.nan
    return int((d1 - d0) * (f1 - f0) * cube.shape[2])


def main():
    parser = argparse.ArgumentParser(
        description="Create a synthetic LWA-style RDC2 HDF5 file for testing array delay-Doppler and angle plotting.",
        formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, max_help_position=34),
    )
    parser.add_argument("--output", required=True, help="Output synthetic RDC2 HDF5 file")
    parser.add_argument("--scenario", choices=["showcase", "box", "angle", "multi", "amplitude"], default="showcase")
    parser.add_argument("--frames", type=int, default=96)
    parser.add_argument("--integration-time", type=float, default=120.0)
    parser.add_argument("--hop-time", type=float, default=2.0)

    parser.add_argument("--delay-min", type=float, default=216.5)
    parser.add_argument("--delay-max", type=float, default=220.5)
    parser.add_argument("--delay-step-ms", type=float, default=0.05)
    parser.add_argument("--fd-min", type=float, default=-1.0)
    parser.add_argument("--fd-max", type=float, default=1.0)
    parser.add_argument("--fd-step-hz", type=float, default=0.025)

    parser.add_argument("--sigma-delay-ms", type=float, default=0.08)
    parser.add_argument("--sigma-doppler-hz", type=float, default=0.045)
    parser.add_argument("--noise-std", type=float, default=0.025)
    parser.add_argument("--seed", type=int, default=12345)

    parser.add_argument("--stands", default="all")
    parser.add_argument("--max-channels", type=int, default=64,
                        help="Cap selected stands for a compact test file; use 0 for every selected stand")
    parser.add_argument("--drop-bad-stands", action="store_true", default=True,
                        help=f"Drop known bad stands: {BAD_LWA_STANDS}")
    parser.add_argument("--keep-bad-stands", action="store_false", dest="drop_bad_stands")
    parser.add_argument("-p", "--pol", choices=["both", "x", "y", "0", "1"], default="x")

    parser.add_argument("--carrier-frequency", type=float, default=24.55e6)
    parser.add_argument("--sample-rate", type=float, default=100e3)
    parser.add_argument("--sweep-frequency", type=float, default=2.0)
    parser.add_argument("--bandwidth", type=float, default=-99.259377e3)

    parser.add_argument("--inject-cable-delays", action="store_true",
                        help="Inject LSL LWA-SV RF-center cable phase for correction tests")
    parser.add_argument("--gain-jitter-db", type=float, default=0.0,
                        help="Per-channel fixed gain error standard deviation")
    parser.add_argument("--phase-jitter-deg", type=float, default=0.0,
                        help="Per-channel fixed phase error standard deviation")
    parser.add_argument("--nan-frame", type=int, default=None,
                        help="Optional frame index where a small RDC2 NaN patch is injected for gap-fill testing")

    parser.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--compression", choices=["lzf", "gzip", "none"], default="lzf")
    parser.add_argument("--compression-level", type=int, default=4)

    args = parser.parse_args()
    if args.frames < 1:
        raise ValueError("--frames must be >= 1")
    if args.integration_time <= 0 or args.hop_time <= 0:
        raise ValueError("--integration-time and --hop-time must be positive")

    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required to write synthetic RDC2 files") from exc

    rng = np.random.default_rng(args.seed)
    delay_ms = grid_from_bounds(args.delay_min, args.delay_max, args.delay_step_ms)
    doppler_hz = grid_from_bounds(args.fd_min, args.fd_max, args.fd_step_hz)
    stands = select_stands(args.stands, args.max_channels, args.drop_bad_stands)
    pols = selected_pols(args.pol)

    channel_stands = np.repeat(np.asarray(stands, dtype=np.int32), len(pols))
    channel_pols = np.tile(np.asarray(pols, dtype=np.int32), len(stands))
    stand_positions = get_lwasv_stand_positions()
    positions = np.asarray([stand_positions[int(stand)] for stand in stands], dtype=np.float64)
    channel_positions = np.repeat(positions, len(pols), axis=0)
    n_channels = channel_stands.size

    gain = np.ones(n_channels, dtype=np.complex64)
    if args.inject_cable_delays:
        gain *= np.conj(cable_phase_factors(channel_stands, channel_pols, args.carrier_frequency)).astype(np.complex64)
    if args.gain_jitter_db > 0:
        gain *= (10.0 ** (rng.normal(0.0, args.gain_jitter_db, n_channels) / 20.0)).astype(np.float32)
    if args.phase_jitter_deg > 0:
        gain *= np.exp(1j * np.deg2rad(rng.normal(0.0, args.phase_jitter_deg, n_channels))).astype(np.complex64)

    frame_tstart = np.arange(args.frames, dtype=np.float64) * float(args.hop_time)
    frame_tend = frame_tstart + float(args.integration_time)
    out_dtype = np.dtype(args.dtype)
    compression_arg = None if args.compression == "none" else args.compression
    compression_opts = None if args.compression in ("none", "lzf") else int(args.compression_level)

    output = os.path.expanduser(args.output)
    os.makedirs(os.path.dirname(os.path.abspath(output)) or ".", exist_ok=True)
    tmp_output = output + ".tmp"
    chunks = (1, delay_ms.size, doppler_hz.size, min(n_channels, 32))

    truth = {
        name: {
            "delay_ms": np.full(args.frames, np.nan, dtype=np.float32),
            "doppler_hz": np.full(args.frames, np.nan, dtype=np.float32),
            "az_deg": np.full(args.frames, np.nan, dtype=np.float32),
            "el_deg": np.full(args.frames, np.nan, dtype=np.float32),
            "amplitude": np.zeros(args.frames, dtype=np.float32),
            "active": np.zeros(args.frames, dtype=np.int8),
        }
        for name in TARGET_NAMES
    }
    nan_values_injected = 0

    with h5py.File(tmp_output, "w") as h5:
        grp = h5.create_group("rdc2")
        dset_real = grp.create_dataset(
            "real",
            shape=(args.frames, delay_ms.size, doppler_hz.size, n_channels),
            dtype=out_dtype,
            chunks=chunks,
            compression=compression_arg,
            compression_opts=compression_opts,
        )
        dset_imag = grp.create_dataset(
            "imag",
            shape=(args.frames, delay_ms.size, doppler_hz.size, n_channels),
            dtype=out_dtype,
            chunks=chunks,
            compression=compression_arg,
            compression_opts=compression_opts,
        )

        wavelength = 299_792_458.0 / float(args.carrier_frequency)
        for frame_idx in range(args.frames):
            noise = args.noise_std * (
                rng.normal(size=(delay_ms.size, doppler_hz.size, n_channels))
                + 1j * rng.normal(size=(delay_ms.size, doppler_hz.size, n_channels))
            ) / np.sqrt(2.0)
            cube = noise.astype(np.complex64)

            for target_idx, name in enumerate(TARGET_NAMES):
                state = target_state(
                    name,
                    frame_idx,
                    args.frames,
                    args.scenario,
                    float(delay_ms[0]),
                    float(delay_ms[-1]),
                    float(doppler_hz[0]),
                    float(doppler_hz[-1]),
                )
                if state is None:
                    continue
                envelope = gaussian_2d(
                    delay_ms,
                    doppler_hz,
                    state["delay_ms"],
                    state["doppler_hz"],
                    args.sigma_delay_ms,
                    args.sigma_doppler_hz,
                )
                array_response = steering_vector(
                    channel_positions,
                    wavelength,
                    state["az_deg"],
                    state["el_deg"],
                ).astype(np.complex64)
                target_phase = np.complex64(np.exp(1j * (0.43 * frame_idx + 1.17 * target_idx)))
                cube += (
                    state["amplitude"]
                    * target_phase
                    * envelope[:, :, None]
                    * array_response[None, None, :]
                    * gain[None, None, :]
                ).astype(np.complex64)

                for key in ("delay_ms", "doppler_hz", "az_deg", "el_deg", "amplitude"):
                    truth[name][key][frame_idx] = state[key]
                truth[name]["active"][frame_idx] = 1

            if args.nan_frame is not None and frame_idx == int(args.nan_frame):
                nan_values_injected += inject_nan_patch(cube, frame_idx, delay_ms.size, doppler_hz.size)

            dset_real[frame_idx] = cube.real.astype(out_dtype, copy=False)
            dset_imag[frame_idx] = cube.imag.astype(out_dtype, copy=False)
            if (frame_idx + 1) % 10 == 0 or frame_idx + 1 == args.frames:
                print(f"Wrote synthetic frame {frame_idx + 1}/{args.frames}", flush=True)

        grp.create_dataset("delay_ms", data=delay_ms)
        grp.create_dataset("doppler_hz", data=doppler_hz)
        grp.create_dataset("frame_tstart_s", data=frame_tstart)
        grp.create_dataset("frame_tend_s", data=frame_tend)
        grp.create_dataset("channel_stands", data=channel_stands)
        grp.create_dataset("channel_pols", data=channel_pols)

        truth_grp = h5.create_group("truth")
        for name, values in truth.items():
            target_grp = truth_grp.create_group(name)
            for key, value in values.items():
                target_grp.create_dataset(key, data=value)

        h5.attrs["metadata_json"] = metadata_json(
            rdc_stage=2,
            synthetic=True,
            station="lwasv",
            channel_id_kind="physical_stand",
            cable_phase_corrected=False,
            cable_delays_injected=bool(args.inject_cable_delays),
            scenario=args.scenario,
            source_file="synthetic_lwa_rdc2",
            shape_frames_delay_doppler_channels=[int(args.frames), int(delay_ms.size), int(doppler_hz.size), int(n_channels)],
            sample_rate_hz=float(args.sample_rate),
            tuning_frequency_hz=float(args.carrier_frequency),
            carrier_frequency_hz=float(args.carrier_frequency),
            sweep_frequency_hz=float(args.sweep_frequency),
            bandwidth_hz=float(args.bandwidth),
            tstart=0.0,
            tend=float(frame_tend[-1]),
            integration_time_s=float(args.integration_time),
            hop_time_s=float(args.hop_time),
            d_min_ms=float(delay_ms[0]),
            d_max_ms=float(delay_ms[-1]),
            fd_min_hz=float(doppler_hz[0]),
            fd_max_hz=float(doppler_hz[-1]),
            dechirp_window="synthetic_rdc2_direct",
            slow_window="synthetic_rdc2_direct",
            nfft_doppler=int(doppler_hz.size),
            adc_start=0,
            adc_count=0,
            gap_fill="nan" if args.nan_frame is not None else "none",
            nan_values_injected=int(nan_values_injected),
            selected_stands=[int(s) for s in channel_stands],
            stands=[int(s) for s in stands],
            pol=str(args.pol),
            drop_bad_stands=bool(args.drop_bad_stands),
            n_channels=int(n_channels),
            n_stands=int(len(np.unique(channel_stands))),
            simulator="simulate_lwa_rdc2.py",
            target_names=TARGET_NAMES,
            sigma_delay_ms=float(args.sigma_delay_ms),
            sigma_doppler_hz=float(args.sigma_doppler_hz),
            noise_std=float(args.noise_std),
            gain_jitter_db=float(args.gain_jitter_db),
            phase_jitter_deg=float(args.phase_jitter_deg),
            **polarization_metadata(channel_pols),
        )

    os.replace(tmp_output, output)
    bytes_uncompressed = args.frames * delay_ms.size * doppler_hz.size * n_channels * np.dtype(args.dtype).itemsize * 2
    print(f"Wrote {output}", flush=True)
    print(
        f"RDC2 shape: (frames={args.frames}, delay={delay_ms.size}, doppler={doppler_hz.size}, channels={n_channels})",
        flush=True,
    )
    print(f"Uncompressed real/imag size: {bytes_uncompressed / 1e6:.1f} MB", flush=True)
    print("Truth trajectories are stored under /truth/<target_name>/ in the HDF5 file.", flush=True)


if __name__ == "__main__":
    main()
