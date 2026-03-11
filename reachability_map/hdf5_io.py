from __future__ import annotations

import datetime
from typing import Optional

import h5py
import numpy as np

from .config import ReachabilityConfig

VERSION = "1.0"


def write_hdf5(
    path: str,
    reachability_3d: np.ndarray,
    voxel_centers: np.ndarray,
    orientations: np.ndarray,
    config: ReachabilityConfig,
    grid_meta: dict,
    poses: Optional[np.ndarray] = None,
) -> None:
    """Write a reachability map to an HDF5 file.

    Parameters
    ----------
    path:
        Output file path.
    reachability_3d:
        Float32 array of shape (nx, ny, nz) with per-voxel reachability index.
    voxel_centers:
        Float32 array of shape (nx, ny, nz, 3) with voxel center coordinates.
    orientations:
        Float32 array of shape (N, 4) with sampled quaternions [qx, qy, qz, qw].
    config:
        ReachabilityConfig used for the computation.
    grid_meta:
        Dict with keys: 'origin' (array[3]), 'shape' (tuple), 'delta' (float),
        'robot_name' (str), 'n_joints' (int), 'actual_quat_delta' (float).
    poses:
        Optional float32 array of shape (M, 8) [qx,qy,qz,qw,tx,ty,tz,ik_count]
        written only when mode="full6d".
    """
    with h5py.File(path, "w") as f:
        # Root attributes
        f.attrs["version"] = VERSION
        f.attrs["urdf_path"] = config.urdf_path
        f.attrs["gripper_link"] = config.gripper_link
        f.attrs["robot_name"] = grid_meta.get("robot_name", "")
        f.attrs["n_joints"] = grid_meta.get("n_joints", 0)
        f.attrs["xyz_delta"] = config.xyz_delta
        f.attrs["actual_quat_delta"] = grid_meta.get("actual_quat_delta", config.quat_delta)
        f.attrs["n_orientations"] = len(orientations)
        f.attrs["mode"] = config.mode
        f.attrs["ik_solver"] = config.ik_solver
        f.attrs["ik_slimit"] = config.ik_slimit
        f.attrs["ik_ilimit"] = config.ik_ilimit
        f.attrs["ik_tol"] = config.ik_tol
        f.attrs["n_workers"] = config.n_workers if config.n_workers is not None else 0
        f.attrs["timestamp"] = datetime.datetime.utcnow().isoformat()

        # grid/ group
        grp = f.create_group("grid")
        grp.attrs["origin"] = np.asarray(grid_meta["origin"], dtype=np.float32)
        grp.attrs["shape"] = np.asarray(grid_meta["shape"], dtype=np.int64)
        grp.attrs["delta"] = config.xyz_delta

        nx, ny, nz = reachability_3d.shape
        chunks = (min(32, nx), min(32, ny), min(32, nz))
        grp.create_dataset(
            "reachability_index",
            data=reachability_3d.astype(np.float32),
            compression="gzip",
            compression_opts=4,
            chunks=chunks,
        )
        grp.create_dataset(
            "voxel_centers",
            data=voxel_centers.astype(np.float32),
        )

        # orientations/ group
        ori_grp = f.create_group("orientations")
        ds = ori_grp.create_dataset("samples", data=orientations.astype(np.float32))
        ds.attrs["method"] = "super_fibonacci"
        ds.attrs["quaternion_convention"] = "xyzw"

        # poses/ group (full6d only)
        if poses is not None:
            poses_grp = f.create_group("poses")
            ds2 = poses_grp.create_dataset(
                "reachability_stats",
                data=poses.astype(np.float32),
            )
            ds2.attrs["columns"] = ["qx", "qy", "qz", "qw", "tx", "ty", "tz", "ik_count"]
            ds2.attrs["quaternion_convention"] = "xyzw"


def init_hdf5_writer(
    path: str,
    reachability_shape: tuple,
    voxel_centers: np.ndarray,
    orientations: np.ndarray,
    config: ReachabilityConfig,
    grid_meta: dict,
) -> "h5py.File":
    """Create an HDF5 file for streaming voxel writes.

    Writes static data (voxel_centers, orientations, metadata) immediately and
    pre-allocates the reachability_index dataset so individual voxel results can
    be written incrementally without buffering all results in memory first.

    For full6d mode, creates an empty resizable poses dataset that grows as
    results stream in.

    Returns the open h5py.File handle; use as a context manager (``with f:``).
    """
    f = h5py.File(path, "w")

    f.attrs["version"] = VERSION
    f.attrs["urdf_path"] = config.urdf_path
    f.attrs["gripper_link"] = config.gripper_link
    f.attrs["robot_name"] = grid_meta.get("robot_name", "")
    f.attrs["n_joints"] = grid_meta.get("n_joints", 0)
    f.attrs["xyz_delta"] = config.xyz_delta
    f.attrs["actual_quat_delta"] = grid_meta.get("actual_quat_delta", config.quat_delta)
    f.attrs["n_orientations"] = len(orientations)
    f.attrs["mode"] = config.mode
    f.attrs["ik_solver"] = config.ik_solver
    f.attrs["ik_slimit"] = config.ik_slimit
    f.attrs["ik_ilimit"] = config.ik_ilimit
    f.attrs["ik_tol"] = config.ik_tol
    f.attrs["n_workers"] = config.n_workers if config.n_workers is not None else 0
    f.attrs["timestamp"] = datetime.datetime.utcnow().isoformat()

    nx, ny, nz = reachability_shape
    grp = f.create_group("grid")
    grp.attrs["origin"] = np.asarray(grid_meta["origin"], dtype=np.float32)
    grp.attrs["shape"] = np.asarray(grid_meta["shape"], dtype=np.int64)
    grp.attrs["delta"] = config.xyz_delta

    chunks = (min(32, nx), min(32, ny), min(32, nz))
    grp.create_dataset(
        "reachability_index",
        shape=reachability_shape,
        dtype=np.float32,
        compression="gzip",
        compression_opts=4,
        chunks=chunks,
        fillvalue=0.0,
    )
    grp.create_dataset("voxel_centers", data=voxel_centers.astype(np.float32))

    ori_grp = f.create_group("orientations")
    ds = ori_grp.create_dataset("samples", data=orientations.astype(np.float32))
    ds.attrs["method"] = "super_fibonacci"
    ds.attrs["quaternion_convention"] = "xyzw"

    if config.mode == "full6d":
        poses_grp = f.create_group("poses")
        ds2 = poses_grp.create_dataset(
            "reachability_stats",
            shape=(0, 8),
            maxshape=(None, 8),
            dtype=np.float32,
            chunks=(1024, 8),
        )
        ds2.attrs["columns"] = ["qx", "qy", "qz", "qw", "tx", "ty", "tz", "ik_count"]
        ds2.attrs["quaternion_convention"] = "xyzw"

    return f


def stream_voxel_result(
    f: "h5py.File",
    flat_idx: int,
    grid_shape: tuple,
    reach_index: float,
    poses: Optional[np.ndarray] = None,
) -> None:
    """Write one voxel result into the pre-allocated HDF5 datasets.

    Parameters
    ----------
    f:
        Open h5py.File created by :func:`init_hdf5_writer`.
    flat_idx:
        Flat voxel index from the worker result.
    grid_shape:
        (nx, ny, nz) grid dimensions.
    reach_index:
        Per-voxel reachability index (0.0–1.0).
    poses:
        Optional (K, 8) float32 array of successful poses (full6d mode only).
    """
    ix, iy, iz = np.unravel_index(flat_idx, grid_shape)
    f["grid/reachability_index"][ix, iy, iz] = reach_index

    if poses is not None and len(poses) > 0:
        ds = f["poses/reachability_stats"]
        n_existing = ds.shape[0]
        n_new = len(poses)
        ds.resize(n_existing + n_new, axis=0)
        ds[n_existing:] = poses


def read_hdf5(path: str) -> dict:
    """Read a reachability HDF5 file and return its contents as a dict.

    Returns
    -------
    dict with keys:
        'reachability_index'  — (nx, ny, nz) float32
        'voxel_centers'       — (nx, ny, nz, 3) float32
        'orientations'        — (N, 4) float32
        'poses'               — (M, 8) float32, or None
        'attrs'               — root metadata dict
        'grid_attrs'          — grid group metadata dict
    """
    result: dict = {}
    with h5py.File(path, "r") as f:
        result["attrs"] = dict(f.attrs)
        result["grid_attrs"] = dict(f["grid"].attrs)
        result["reachability_index"] = f["grid/reachability_index"][:]
        result["voxel_centers"] = f["grid/voxel_centers"][:]
        result["orientations"] = f["orientations/samples"][:]
        result["poses"] = f["poses/reachability_stats"][:] if "poses" in f else None
    return result
