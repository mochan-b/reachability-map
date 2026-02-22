from __future__ import annotations

import multiprocessing as mp
import os
from math import pi
from typing import Optional

import numpy as np
from tqdm import tqdm

from .config import ReachabilityConfig
from .hdf5_io import write_hdf5
from .orientation_sampler import estimate_n_orientations, sample_orientations
from .robot_loader import get_ets, load_robot
from .voxel_grid import auto_bounds, build_voxel_grid
from .worker import compute_voxel_reachability, worker_init


def run(config: ReachabilityConfig) -> None:
    """Run the full reachability mapping pipeline.

    Phase 1 — Setup:
        Load the robot, build the voxel grid, sample SO(3) orientations.

    Phase 2 — Parallel IK:
        Distribute voxel tasks across a multiprocessing pool; each worker
        attempts IK for every sampled orientation at a given voxel centre.

    Phase 3 — Aggregate & Write:
        Collect per-voxel results, fill the 3-D reachability array, and
        write the HDF5 output file.

    Parameters
    ----------
    config:
        Fully populated ReachabilityConfig dataclass.
    """

    # -----------------------------------------------------------------------
    # Phase 1 — Setup
    # -----------------------------------------------------------------------
    robot = load_robot(config.urdf_path, config.gripper_link)
    qlim = robot.qlim  # shape (2, n_joints)
    n_joints = int(qlim.shape[1])
    robot_name: str = robot.name

    bounds = config.bounds if config.bounds is not None else auto_bounds(robot, qlim)
    centers_flat, grid_shape = build_voxel_grid(bounds, config.xyz_delta)
    nx, ny, nz = grid_shape
    n_voxels = len(centers_flat)

    if config.n_orientations is not None:
        n_orientations = config.n_orientations
        # Back-compute the effective quat_delta for metadata
        actual_quat_delta = (3.0 * pi**2 / n_orientations) ** (1.0 / 3.0)
    else:
        n_orientations = estimate_n_orientations(config.quat_delta)
        actual_quat_delta = config.quat_delta

    orientations = sample_orientations(n_orientations)  # (N, 4) float32

    n_workers = config.n_workers if config.n_workers is not None else (os.cpu_count() or 1)
    chunksize = max(1, n_voxels // (n_workers * 16))

    solver_config = {
        "solver_name": config.ik_solver,
        "ilimit": config.ik_ilimit,
        "slimit": config.ik_slimit,
        "tol": config.ik_tol,
    }

    if not config.quiet:
        print(f"Robot      : {robot_name} ({n_joints} joints)")
        print(f"Workspace  : x=[{bounds.x_min:.2f}, {bounds.x_max:.2f}]  "
              f"y=[{bounds.y_min:.2f}, {bounds.y_max:.2f}]  "
              f"z=[{bounds.z_min:.2f}, {bounds.z_max:.2f}]")
        print(f"Grid       : {nx}×{ny}×{nz} = {n_voxels:,} voxels  (δ={config.xyz_delta} m)")
        print(f"Orientations: {n_orientations}  (quat_delta ≈ {actual_quat_delta:.3f})")
        print(f"Workers    : {n_workers}  chunksize={chunksize}")
        print(f"IK solver  : {config.ik_solver}  slimit={config.ik_slimit}  "
              f"ilimit={config.ik_ilimit}  tol={config.ik_tol}")

    # -----------------------------------------------------------------------
    # Phase 2 — Parallel IK
    # -----------------------------------------------------------------------
    # Pass xyz as plain Python tuples (picklable primitives).
    voxel_tasks = [
        (i, (float(centers_flat[i, 0]), float(centers_flat[i, 1]), float(centers_flat[i, 2])))
        for i in range(n_voxels)
    ]

    results: list[dict] = []
    with mp.Pool(
        processes=n_workers,
        initializer=worker_init,
        initargs=(
            os.path.abspath(config.urdf_path),
            config.gripper_link,
            orientations,
            solver_config,
            config.mode,
        ),
    ) as pool:
        for result in tqdm(
            pool.imap_unordered(compute_voxel_reachability, voxel_tasks, chunksize=chunksize),
            total=n_voxels,
            desc="Voxels",
            unit="vox",
            disable=config.quiet,
        ):
            results.append(result)

    # -----------------------------------------------------------------------
    # Phase 3 — Aggregate & Write
    # -----------------------------------------------------------------------
    reachability_3d = np.zeros(grid_shape, dtype=np.float32)
    poses_parts: list[np.ndarray] | None = [] if config.mode == "full6d" else None

    for r in results:
        ix, iy, iz = np.unravel_index(r["flat_idx"], grid_shape)
        reachability_3d[ix, iy, iz] = r["reach_index"]
        if poses_parts is not None and r["poses"] is not None and len(r["poses"]) > 0:
            poses_parts.append(r["poses"])

    # Reshape flat centers back to (nx, ny, nz, 3) — safe because
    # build_voxel_grid uses indexing='ij' and ravel() with default C order.
    voxel_centers_3d = centers_flat.reshape(nx, ny, nz, 3)

    poses_array: Optional[np.ndarray] = None
    if poses_parts is not None:
        poses_array = (
            np.concatenate(poses_parts, axis=0)
            if poses_parts
            else np.empty((0, 8), dtype=np.float32)
        )

    grid_meta = {
        "origin": [bounds.x_min, bounds.y_min, bounds.z_min],
        "shape": list(grid_shape),
        "delta": config.xyz_delta,
        "robot_name": robot_name,
        "n_joints": n_joints,
        "actual_quat_delta": actual_quat_delta,
    }

    write_hdf5(
        path=config.output_path,
        reachability_3d=reachability_3d,
        voxel_centers=voxel_centers_3d,
        orientations=orientations,
        config=config,
        grid_meta=grid_meta,
        poses=poses_array,
    )

    if not config.quiet:
        n_reachable = int((reachability_3d > 0).sum())
        print(f"\nDone. Output: {config.output_path}")
        print(f"Reachable voxels: {n_reachable:,} / {n_voxels:,} "
              f"({100.0 * n_reachable / n_voxels:.1f}%)")
