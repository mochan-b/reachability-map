from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

# ---------------------------------------------------------------------------
# Module-level globals — set once per worker process by worker_init()
# ---------------------------------------------------------------------------
_ets = None
_orientations: np.ndarray | None = None
_solver = None
_mode: str = "positional"


def worker_init(
    urdf_path: str,
    gripper_link: str,
    orientations: np.ndarray,
    solver_config: dict,
    mode: str,
) -> None:
    """Initialise per-process globals for the multiprocessing pool.

    Called once in each worker process before any tasks are dispatched.
    Re-loading the robot here (rather than forking a pre-loaded one) avoids
    fork-safety issues with the underlying C++ robot libraries.

    Parameters
    ----------
    urdf_path:
        Absolute path to the robot URDF.
    gripper_link:
        End-effector link name.
    orientations:
        (N, 4) float32 array of quaternions [qx, qy, qz, qw].
    solver_config:
        Dict with keys: solver_name, ilimit, slimit, tol.
    mode:
        "positional" or "full6d".
    """
    global _ets, _orientations, _solver, _mode

    from .robot_loader import load_robot, get_ets
    from .ik_solver import IKSolverWrapper

    robot = load_robot(urdf_path, gripper_link)
    _ets = get_ets(robot, gripper_link)
    _orientations = orientations
    _solver = IKSolverWrapper(
        solver_name=solver_config["solver_name"],
        ilimit=solver_config["ilimit"],
        slimit=solver_config["slimit"],
        tol=solver_config["tol"],
    )
    _mode = mode


def compute_voxel_reachability(task: tuple) -> dict:
    """Compute the reachability index for a single voxel.

    This function must be a module-level callable so that it is picklable
    across multiprocessing boundaries.

    Parameters
    ----------
    task:
        Tuple of (flat_idx: int, xyz: tuple[float, float, float]).

    Returns
    -------
    dict with keys:
        'flat_idx'    — int, the flat voxel index
        'reach_index' — float in [0.0, 1.0]
        'n_success'   — int, number of orientations for which IK succeeded
        'poses'       — (K, 8) float32 array of successful poses
                        [qx,qy,qz,qw,tx,ty,tz,1.0] if mode=="full6d", else None
    """
    flat_idx, xyz = task
    orientations = _orientations
    n = len(orientations)
    n_success = 0

    T = np.eye(4, dtype=np.float64)
    T[0, 3], T[1, 3], T[2, 3] = float(xyz[0]), float(xyz[1]), float(xyz[2])

    poses_list: list | None = [] if _mode == "full6d" else None

    for i in range(n):
        quat_xyzw = orientations[i]  # [qx, qy, qz, qw]
        T[:3, :3] = Rotation.from_quat(quat_xyzw).as_matrix()
        if _solver.solve(_ets, T):
            n_success += 1
            if poses_list is not None:
                poses_list.append([*quat_xyzw, *xyz, 1.0])

    reach_index = n_success / n if n > 0 else 0.0

    poses: np.ndarray | None = None
    if poses_list is not None:
        poses = (
            np.array(poses_list, dtype=np.float32)
            if poses_list
            else np.empty((0, 8), dtype=np.float32)
        )

    return {
        "flat_idx": flat_idx,
        "reach_index": reach_index,
        "n_success": n_success,
        "poses": poses,
    }
