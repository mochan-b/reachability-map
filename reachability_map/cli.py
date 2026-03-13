from __future__ import annotations

import argparse
import sys

from .config import ReachabilityConfig, WorkspaceBounds
from .reachability import run


def _parse_bounds(value: str) -> WorkspaceBounds:
    """Parse a comma-separated bounds string "xmin,xmax,ymin,ymax,zmin,zmax"."""
    parts = value.split(",")
    if len(parts) != 6:
        raise argparse.ArgumentTypeError(
            f"--bounds requires exactly 6 comma-separated floats, got {len(parts)}: '{value}'"
        )
    try:
        xmin, xmax, ymin, ymax, zmin, zmax = (float(p) for p in parts)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"--bounds contains non-numeric value: {e}") from e
    return WorkspaceBounds(xmin, xmax, ymin, ymax, zmin, zmax)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="reachability-map",
        description=(
            "Compute a 6-D reachability index map for a robot arm from its URDF. "
            "Samples the 3-D workspace and SO(3) orientations, runs parallel IK, "
            "and writes results to an HDF5 file."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    p.add_argument("--urdf", required=True, metavar="PATH",
                   help="Path to the robot URDF file.")
    p.add_argument("--gripper-link", required=True, metavar="LINK",
                   help="Name of the end-effector / gripper link in the URDF.")
    p.add_argument("--output", required=True, metavar="PATH",
                   help="Output HDF5 file path (e.g. panda_reach.h5).")

    # Grid resolution
    p.add_argument("--xyz-delta", type=float, default=0.04, metavar="M",
                   help="Voxel side length in metres.")

    # Orientation resolution — mutually exclusive
    ori = p.add_mutually_exclusive_group()
    ori.add_argument("--quat-delta", type=float, default=None, metavar="D",
                     help="Quaternion-space resolution; determines number of SO(3) samples "
                          "(default 0.5 → ~237 orientations, matching OpenRAVE).")
    ori.add_argument("--n-orientations", type=int, default=None, metavar="N",
                     help="Explicit number of SO(3) orientations to sample.")

    # Workspace bounds
    p.add_argument("--bounds", type=_parse_bounds, default=None,
                   metavar="xmin,xmax,ymin,ymax,zmin,zmax",
                   help="Workspace bounding box in metres. "
                        "Auto-estimated from FK at q_max if omitted.")

    # Mode
    p.add_argument("--mode", choices=["positional", "full6d"], default="positional",
                   help="'positional' stores only the per-voxel index; "
                        "'full6d' also stores every successful (pose, orientation) pair.")

    # IK solver
    p.add_argument("--ik-solver", choices=["LM", "GN", "NR"], default="LM",
                   help="Numerical IK algorithm: Levenberg-Marquardt, Gauss-Newton, "
                        "or Newton-Raphson.")
    p.add_argument("--ik-slimit", type=int, default=100, metavar="N",
                   help="Number of random restarts per IK call.")
    p.add_argument("--ik-ilimit", type=int, default=30, metavar="N",
                   help="Max iterations per IK restart.")
    p.add_argument("--ik-tol", type=float, default=1e-6, metavar="TOL",
                   help="IK convergence tolerance.")

    # Parallelism
    p.add_argument("--workers", type=int, default=None, metavar="N",
                   help="Number of worker processes (default: os.cpu_count()).")

    # GPU
    p.add_argument("--use-gpu", action="store_true",
                   help="Use cuRobo for GPU-accelerated batched IK.")
    p.add_argument("--gpu-batch-size", type=int, default=2000, metavar="N",
                   help="Number of target poses to batch in each GPU call.")

    # Tight bounds pre-pass
    p.add_argument(
        "--no-tight-bounds",
        action="store_false",
        dest="tight_bounds",
        help="Skip the tight-bounds pre-pass and use auto-estimated bounds directly.",
    )

    # Misc
    p.add_argument("--quiet", action="store_true",
                   help="Suppress progress output.")

    return p


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Resolve orientation defaults: if neither flag given, fall back to quat_delta=0.5
    quat_delta = args.quat_delta if args.quat_delta is not None else (
        0.5 if args.n_orientations is None else 0.5  # n_orientations takes precedence
    )

    config = ReachabilityConfig(
        urdf_path=args.urdf,
        gripper_link=args.gripper_link,
        output_path=args.output,
        xyz_delta=args.xyz_delta,
        quat_delta=quat_delta,
        n_orientations=args.n_orientations,
        bounds=args.bounds,
        mode=args.mode,
        ik_solver=args.ik_solver,
        ik_slimit=args.ik_slimit,
        ik_ilimit=args.ik_ilimit,
        ik_tol=args.ik_tol,
        n_workers=args.workers,
        quiet=args.quiet,
        use_gpu=args.use_gpu,
        gpu_batch_size=args.gpu_batch_size,
        tight_bounds=args.tight_bounds,
    )

    try:
        run(config)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
