from __future__ import annotations

import multiprocessing as mp
import os
from math import pi

import numpy as np
from tqdm import tqdm

from .config import ReachabilityConfig
from .hdf5_io import init_hdf5_writer, stream_voxel_result
from .orientation_sampler import estimate_n_orientations, sample_orientations
from .robot_loader import get_ets, load_robot
from .voxel_grid import auto_bounds, build_voxel_grid
from .worker import compute_voxel_reachability, worker_init


def run(config: ReachabilityConfig) -> None:
    # ... (Phase 1 Setup stays mostly same, but we check use_gpu)
    robot = load_robot(config.urdf_path, config.gripper_link)
    qlim = robot.qlim
    n_joints = int(qlim.shape[1])
    robot_name: str = robot.name

    bounds = config.bounds if config.bounds is not None else auto_bounds(robot, qlim)
    centers_flat, grid_shape = build_voxel_grid(bounds, config.xyz_delta)
    nx, ny, nz = grid_shape
    n_voxels = len(centers_flat)

    if config.n_orientations is not None:
        n_orientations = config.n_orientations
        actual_quat_delta = (3.0 * pi**2 / n_orientations) ** (1.0 / 3.0)
    else:
        n_orientations = estimate_n_orientations(config.quat_delta)
        actual_quat_delta = config.quat_delta

    orientations = sample_orientations(n_orientations)  # (N, 4) float32

    if not config.quiet:
        print(f"Robot      : {robot_name} ({n_joints} joints)")
        print(f"Workspace  : x=[{bounds.x_min:.2f}, {bounds.x_max:.2f}]  "
              f"y=[{bounds.y_min:.2f}, {bounds.y_max:.2f}]  "
              f"z=[{bounds.z_min:.2f}, {bounds.z_max:.2f}]")
        print(f"Grid       : {nx}×{ny}×{nz} = {n_voxels:,} voxels  (δ={config.xyz_delta} m)")
        print(f"Orientations: {n_orientations}  (quat_delta ≈ {actual_quat_delta:.3f})")
        if config.use_gpu:
            print(f"IK solver  : cuRobo (GPU)  batch_size={config.gpu_batch_size}")
        else:
            n_workers = config.n_workers if config.n_workers is not None else (os.cpu_count() or 1)
            print(f"Workers    : {n_workers}")
            print(f"IK solver  : {config.ik_solver}  slimit={config.ik_slimit}  "
                  f"ilimit={config.ik_ilimit}  tol={config.ik_tol}")

    voxel_centers_3d = centers_flat.reshape(nx, ny, nz, 3)
    grid_meta = {
        "origin": [bounds.x_min, bounds.y_min, bounds.z_min],
        "shape": list(grid_shape),
        "delta": config.xyz_delta,
        "robot_name": robot_name,
        "n_joints": n_joints,
        "actual_quat_delta": actual_quat_delta,
    }

    if config.use_gpu:
        run_gpu(config, centers_flat, grid_shape, orientations, grid_meta, voxel_centers_3d)
    else:
        run_cpu(config, centers_flat, grid_shape, orientations, grid_meta, voxel_centers_3d)


def run_cpu(config, centers_flat, grid_shape, orientations, grid_meta, voxel_centers_3d) -> None:
    n_voxels = len(centers_flat)
    n_workers = config.n_workers if config.n_workers is not None else (os.cpu_count() or 1)
    chunksize = max(1, n_voxels // (n_workers * 16))

    solver_config = {
        "solver_name": config.ik_solver,
        "ilimit": config.ik_ilimit,
        "slimit": config.ik_slimit,
        "tol": config.ik_tol,
    }

    voxel_tasks = [
        (i, (float(centers_flat[i, 0]), float(centers_flat[i, 1]), float(centers_flat[i, 2])))
        for i in range(n_voxels)
    ]

    n_reachable = 0
    with init_hdf5_writer(
        path=config.output_path,
        reachability_shape=grid_shape,
        voxel_centers=voxel_centers_3d,
        orientations=orientations,
        config=config,
        grid_meta=grid_meta,
    ) as f:
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
                desc="Voxels (CPU)",
                unit="vox",
                disable=config.quiet,
            ):
                stream_voxel_result(
                    f,
                    result["flat_idx"],
                    grid_shape,
                    result["reach_index"],
                    result.get("poses"),
                )
                if result["reach_index"] > 0:
                    n_reachable += 1
    
    _print_summary(config, n_reachable, n_voxels)


def run_gpu(config, centers_flat, grid_shape, orientations, grid_meta, voxel_centers_3d) -> None:
    from .curobo_solver import CuroboIKSolver
    
    solver = CuroboIKSolver(config.urdf_path, config.gripper_link)
    n_voxels = len(centers_flat)
    n_orientations = len(orientations)
    
    # Batch by voxels
    voxels_per_batch = max(1, config.gpu_batch_size // n_orientations)
    n_batches = (n_voxels + voxels_per_batch - 1) // voxels_per_batch
    
    n_reachable = 0
    with init_hdf5_writer(
        path=config.output_path,
        reachability_shape=grid_shape,
        voxel_centers=voxel_centers_3d,
        orientations=orientations,
        config=config,
        grid_meta=grid_meta,
    ) as f:
        for b in tqdm(range(n_batches), desc="Batches (GPU)", disable=config.quiet):
            start_idx = b * voxels_per_batch
            end_idx = min(start_idx + voxels_per_batch, n_voxels)
            batch_size = end_idx - start_idx
            
            batch_voxels = centers_flat[start_idx:end_idx] # (batch_size, 3)
            
            # Repeat voxels for each orientation
            # [v1, v1, ..., v1, v2, v2, ..., v2, ...]
            xyz_batch = np.repeat(batch_voxels, n_orientations, axis=0) # (batch_size * n_ori, 3)
            # Repeat orientations for each voxel
            # [o1, o2, ..., oN, o1, o2, ..., oN, ...]
            ori_batch = np.tile(orientations, (batch_size, 1)) # (batch_size * n_ori, 4)
            
            success_mask = solver.solve_batch(xyz_batch, ori_batch) # (batch_size * n_ori,)
            success_mask = success_mask.reshape(batch_size, n_orientations)
            
            for i in range(batch_size):
                flat_idx = start_idx + i
                n_success = np.sum(success_mask[i])
                reach_index = n_success / n_orientations
                
                poses = None
                if config.mode == "full6d":
                    # Filter successful orientations for this voxel
                    valid_ori = orientations[success_mask[i]]
                    v_xyz = batch_voxels[i]
                    # Create (K, 8) array: [qx, qy, qz, qw, x, y, z, 1.0]
                    k = len(valid_ori)
                    poses = np.zeros((k, 8), dtype=np.float32)
                    poses[:, :4] = valid_ori
                    poses[:, 4:7] = v_xyz
                    poses[:, 7] = 1.0
                
                stream_voxel_result(
                    f,
                    flat_idx,
                    grid_shape,
                    reach_index,
                    poses,
                )
                if reach_index > 0:
                    n_reachable += 1

    _print_summary(config, n_reachable, n_voxels)


def _print_summary(config, n_reachable, n_voxels) -> None:
    if not config.quiet:
        print(f"\nDone. Output: {config.output_path}")
        print(f"Reachable voxels: {n_reachable:,} / {n_voxels:,} "
              f"({100.0 * n_reachable / n_voxels:.1f}%)")
