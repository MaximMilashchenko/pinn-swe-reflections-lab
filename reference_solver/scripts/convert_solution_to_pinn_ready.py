from pathlib import Path
import argparse
import json
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--case", default="AH=5e+4")
    parser.add_argument("--raw-dx", type=float, default=400.0)
    parser.add_argument("--raw-output-dt", type=float, default=60.0)

    parser.add_argument("--target-dx", type=float, default=10000.0)
    parser.add_argument("--target-dt", type=float, default=60.0)

    parser.add_argument("--time-start", type=float, default=0.0)
    parser.add_argument("--time-end", type=float, default=270000.0)

    parser.add_argument("--x-min", type=float, default=-1000000.0)
    parser.add_argument("--x-max", type=float, default=1000000.0)

    parser.add_argument("--zeta-start-index", type=int, default=13)
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64"])

    parser.add_argument("--grid-label", default="dt=1s_dx=400m")

    return parser.parse_args()


def require_integer(value, name):
    rounded = round(value)
    if abs(value - rounded) > 1e-9:
        raise ValueError(f"{name} must be integer, got {value}")
    return int(rounded)


def load_frame(path, indices, dtype):
    values = np.loadtxt(path, dtype=dtype)
    return values[indices]


def main():
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    data_root = repo_root / "reference_solver" / "data"

    raw_case_dir = data_root / "raw" / args.case
    u_dir = raw_case_dir / "udata"
    zeta_dir = raw_case_dir / "zdata"

    out_dir = data_root / "pinn_ready" / args.grid_label / args.case
    out_dir.mkdir(parents=True, exist_ok=True)

    dtype = np.float32 if args.dtype == "float32" else np.float64

    space_stride = require_integer(args.target_dx / args.raw_dx, "target_dx / raw_dx")
    time_stride = require_integer(args.target_dt / args.raw_output_dt, "target_dt / raw_output_dt")
    start_frame = require_integer(args.time_start / args.raw_output_dt, "time_start / raw_output_dt")
    n_time = require_integer((args.time_end - args.time_start) / args.target_dt, "time range / target_dt")

    u_x_grid = np.arange(args.x_min, args.x_max + args.target_dx, args.target_dx)
    zeta_x_grid = (u_x_grid - 0.5 * args.target_dx)[1:]

    first_u_path = u_dir / f"u{start_frame:04d}.dat"
    first_zeta_path = zeta_dir / f"z{start_frame:04d}.dat"

    if not first_u_path.exists():
        raise FileNotFoundError(first_u_path)

    if not first_zeta_path.exists():
        raise FileNotFoundError(first_zeta_path)

    raw_u_points = np.loadtxt(first_u_path, dtype=dtype).shape[0]
    raw_zeta_points = np.loadtxt(first_zeta_path, dtype=dtype).shape[0]

    u_indices = np.arange(0, raw_u_points, space_stride)[: len(u_x_grid)]
    zeta_indices = np.arange(args.zeta_start_index, raw_zeta_points, space_stride)[: len(zeta_x_grid)]
    time_indices = start_frame + np.arange(n_time) * time_stride

    if len(u_indices) != len(u_x_grid):
        raise ValueError(
            f"Expected {len(u_x_grid)} u points from target grid, got {len(u_indices)}. "
            f"Check x_min/x_max/target_dx/raw_dx."
        )

    if len(zeta_indices) != len(zeta_x_grid):
        raise ValueError(
            f"Expected {len(zeta_x_grid)} zeta points from target grid, got {len(zeta_indices)}. "
            f"Check zeta_start_index/x_min/x_max/target_dx/raw_dx."
        )

    u_frames = []
    zeta_frames = []

    for k, frame_idx in enumerate(time_indices):
        u_path = u_dir / f"u{frame_idx:04d}.dat"
        zeta_path = zeta_dir / f"z{frame_idx:04d}.dat"

        if not u_path.exists():
            raise FileNotFoundError(u_path)

        if not zeta_path.exists():
            raise FileNotFoundError(zeta_path)

        u_frames.append(load_frame(u_path, u_indices, dtype))
        zeta_frames.append(load_frame(zeta_path, zeta_indices, dtype))

        if k % 100 == 0:
            print(f"converted {k}/{len(time_indices)} frames")

    u_ref = np.asarray(u_frames, dtype=dtype).T
    zeta_ref = np.asarray(zeta_frames, dtype=dtype).T

    expected_u_shape = (len(u_x_grid), n_time)
    expected_zeta_shape = (len(zeta_x_grid), n_time)

    if u_ref.shape != expected_u_shape:
        raise ValueError(f"u_ref shape {u_ref.shape}, expected {expected_u_shape}")

    if zeta_ref.shape != expected_zeta_shape:
        raise ValueError(f"zeta_ref shape {zeta_ref.shape}, expected {expected_zeta_shape}")

    np.save(out_dir / "Zonal_Velocity.npy", u_ref)
    np.save(out_dir / "Sea_Level_Elevation.npy", zeta_ref)

    meta = {
        "case": args.case,
        "raw_case_dir": str(raw_case_dir),
        "output_dir": str(out_dir),
        "raw_dx_m": args.raw_dx,
        "raw_output_dt_s": args.raw_output_dt,
        "target_dx_m": args.target_dx,
        "target_dt_s": args.target_dt,
        "time_start_s": args.time_start,
        "time_end_s": args.time_end,
        "n_time": n_time,
        "x_min_m": args.x_min,
        "x_max_m": args.x_max,
        "u_shape": list(u_ref.shape),
        "zeta_shape": list(zeta_ref.shape),
        "u_indices": [int(u_indices[0]), int(u_indices[-1]), int(space_stride)],
        "zeta_indices": [int(zeta_indices[0]), int(zeta_indices[-1]), int(space_stride)],
        "time_indices": [int(time_indices[0]), int(time_indices[-1]), int(time_stride)],
        "dtype": args.dtype,
    }

    with open(out_dir / "reference_preprocess_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("saved:", out_dir / "Zonal_Velocity.npy")
    print("saved:", out_dir / "Sea_Level_Elevation.npy")
    print("u_ref shape:", u_ref.shape)
    print("zeta_ref shape:", zeta_ref.shape)


if __name__ == "__main__":
    main()