from __future__ import annotations

import warnings

import numpy as np
from scipy.spatial.transform import Rotation

from .config import WorkspaceBounds


def find_tight_bounds(
    initial_bounds: WorkspaceBounds,
    delta: float,
    urdf_path: str,
    gripper_link: str,
    orientations: np.ndarray,  # (N, 4) scalar-last quaternions
    solver_config: dict,
    use_gpu: bool = False,
    gpu_solver=None,
    gpu_batch_size: int = 2000,
    quiet: bool = False,
) -> WorkspaceBounds:
    """Binary-search for tight workspace bounds.

    For each of the 6 axis-aligned faces, finds the tightest boundary where
    the outermost face layer is all-zero but the adjacent interior layer has
    at least one reachable voxel.

    Parameters
    ----------
    initial_bounds:
        Conservative bounds to search within (e.g. from auto_bounds).
    delta:
        Voxel side length in metres.
    urdf_path:
        Absolute path to the robot URDF.
    gripper_link:
        End-effector link name.
    orientations:
        (N, 4) float32 quaternion array [qx, qy, qz, qw].  First min(20, N)
        are used for probing.
    solver_config:
        Dict with keys: solver_name, ilimit, slimit, tol.
    use_gpu:
        Whether to use the GPU solver for probing.
    gpu_solver:
        Pre-built CuroboIKSolver instance (only used when use_gpu=True).
    quiet:
        Suppress progress messages.

    Returns
    -------
    WorkspaceBounds
        Tightened bounds.
    """
    # GPU: use all orientations for correctness; stride=2 to cut slice work by 4×.
    # CPU: cap at 20 orientations to keep the pre-pass fast; stride=1 (already cheap).
    if use_gpu:
        probe_orientations = orientations
        checker = _GPUChecker(gpu_solver, probe_orientations, batch_size=gpu_batch_size)
        stride = 2
    else:
        probe_orientations = orientations[:min(20, len(orientations))]
        checker = _CPUChecker(urdf_path, gripper_link, probe_orientations, solver_config)
        stride = 1

    xs = np.arange(initial_bounds.x_min + delta / 2, initial_bounds.x_max, delta)
    ys = np.arange(initial_bounds.y_min + delta / 2, initial_bounds.y_max, delta)
    zs = np.arange(initial_bounds.z_min + delta / 2, initial_bounds.z_max, delta)

    x_min, x_max = _tighten_axis(
        axis=0, coords=xs, other_coords=(ys, zs),
        checker=checker, delta=delta, stride=stride,
        orig_min=initial_bounds.x_min, orig_max=initial_bounds.x_max,
        quiet=quiet,
    )
    y_min, y_max = _tighten_axis(
        axis=1, coords=ys, other_coords=(xs, zs),
        checker=checker, delta=delta, stride=stride,
        orig_min=initial_bounds.y_min, orig_max=initial_bounds.y_max,
        quiet=quiet,
    )
    z_min, z_max = _tighten_axis(
        axis=2, coords=zs, other_coords=(xs, ys),
        checker=checker, delta=delta, stride=stride,
        orig_min=initial_bounds.z_min, orig_max=initial_bounds.z_max,
        quiet=quiet,
    )

    return WorkspaceBounds(x_min, x_max, y_min, y_max, z_min, z_max)


# ---------------------------------------------------------------------------
# Per-axis search
# ---------------------------------------------------------------------------

def _tighten_axis(
    axis: int,
    coords: np.ndarray,
    other_coords: tuple[np.ndarray, np.ndarray],
    checker,
    delta: float,
    stride: int,
    orig_min: float,
    orig_max: float,
    quiet: bool,
) -> tuple[float, float]:
    """Return tightened (min, max) for one axis."""
    axis_names = ["x", "y", "z"]
    name = axis_names[axis]

    new_min = _find_min_face(axis, coords, other_coords, checker, delta, stride, orig_min, name, quiet)
    new_max = _find_max_face(axis, coords, other_coords, checker, delta, stride, orig_max, name, quiet)
    return new_min, new_max


def _find_min_face(axis, coords, other_coords, checker, delta, stride, orig_min, name, quiet):
    """Binary search for the minimum face: find leftmost reachable slice."""
    # Safety check: is the initial (leftmost) slice already reachable?
    if _slice_has_reachable_voxel(axis, coords[0], other_coords, checker, stride):
        if not quiet:
            warnings.warn(
                f"Tight bounds: {name}_min initial face is already reachable; "
                f"keeping original bound {orig_min:.4f}.",
                stacklevel=3,
            )
        return orig_min

    lo, hi = 0, len(coords) - 1
    found = None
    while lo <= hi:
        mid = (lo + hi) // 2
        if _slice_has_reachable_voxel(axis, coords[mid], other_coords, checker, stride):
            found = mid
            hi = mid - 1  # try even smaller index (closer to min face)
        else:
            lo = mid + 1

    if found is None:
        if not quiet:
            warnings.warn(
                f"Tight bounds: no reachable voxels found along {name} axis; "
                f"keeping original {name}_min bound {orig_min:.4f}.",
                stacklevel=3,
            )
        return orig_min

    # Place boundary so the outer face aligns with coords[found-1] (confirmed unreachable).
    # Using 1.5*delta ensures new_bound + delta/2 == coords[found] - delta, which is an
    # initial-grid center that was probed and found unreachable.
    return coords[found] - 1.5 * delta


def _find_max_face(axis, coords, other_coords, checker, delta, stride, orig_max, name, quiet):
    """Binary search for the maximum face: find rightmost reachable slice."""
    if _slice_has_reachable_voxel(axis, coords[-1], other_coords, checker, stride):
        if not quiet:
            warnings.warn(
                f"Tight bounds: {name}_max initial face is already reachable; "
                f"keeping original bound {orig_max:.4f}.",
                stacklevel=3,
            )
        return orig_max

    lo, hi = 0, len(coords) - 1
    found = None
    while lo <= hi:
        mid = (lo + hi) // 2
        if _slice_has_reachable_voxel(axis, coords[mid], other_coords, checker, stride):
            found = mid
            lo = mid + 1  # try even larger index (closer to max face)
        else:
            hi = mid - 1

    if found is None:
        if not quiet:
            warnings.warn(
                f"Tight bounds: no reachable voxels found along {name} axis; "
                f"keeping original {name}_max bound {orig_max:.4f}.",
                stacklevel=3,
            )
        return orig_max

    # Place boundary so the outer face aligns with coords[found+1] (confirmed unreachable).
    return coords[found] + 1.5 * delta


def _slice_has_reachable_voxel(
    axis: int,
    pos: float,
    other_coords: tuple[np.ndarray, np.ndarray],
    checker,
    stride: int = 1,
) -> bool:
    """Return True if any voxel in the 2-D slice at (axis=pos) is reachable."""
    a_coords, b_coords = other_coords
    if stride > 1:
        a_coords = a_coords[::stride]
        b_coords = b_coords[::stride]
    ga, gb = np.meshgrid(a_coords, b_coords, indexing='ij')
    n_pts = ga.size
    pts = np.empty((n_pts, 3), dtype=np.float32)

    # axis=0 → pos is x, a=y, b=z
    # axis=1 → pos is y, a=x, b=z
    # axis=2 → pos is z, a=x, b=y
    if axis == 0:
        pts[:, 0] = pos
        pts[:, 1] = ga.ravel()
        pts[:, 2] = gb.ravel()
    elif axis == 1:
        pts[:, 0] = ga.ravel()
        pts[:, 1] = pos
        pts[:, 2] = gb.ravel()
    else:
        pts[:, 0] = ga.ravel()
        pts[:, 1] = gb.ravel()
        pts[:, 2] = pos

    return checker.any_reachable(pts)


# ---------------------------------------------------------------------------
# Checker implementations
# ---------------------------------------------------------------------------

class _CPUChecker:
    """IK checker using a single in-process IKSolverWrapper (main process only)."""

    def __init__(self, urdf_path, gripper_link, probe_orientations, solver_config):
        from .robot_loader import load_robot, get_ets
        from .ik_solver import IKSolverWrapper

        robot = load_robot(urdf_path, gripper_link)
        self._ets = get_ets(robot, gripper_link)
        self._orientations = probe_orientations
        self._solver = IKSolverWrapper(
            solver_name=solver_config["solver_name"],
            ilimit=min(solver_config.get("ilimit", 30), 10),
            slimit=min(solver_config.get("slimit", 100), 5),
            tol=solver_config.get("tol", 1e-6),
        )

    def any_reachable(self, pts: np.ndarray) -> bool:
        """Return True as soon as any (point, orientation) IK succeeds."""
        T = np.eye(4, dtype=np.float64)
        for pt in pts:
            T[0, 3], T[1, 3], T[2, 3] = float(pt[0]), float(pt[1]), float(pt[2])
            for quat in self._orientations:
                T[:3, :3] = Rotation.from_quat(quat).as_matrix()
                if self._solver.solve(self._ets, T):
                    return True
        return False


class _GPUChecker:
    """IK checker using a pre-built CuroboIKSolver."""

    def __init__(self, gpu_solver, probe_orientations, batch_size: int = 2000):
        self._solver = gpu_solver
        self._orientations = probe_orientations
        self._batch_size = batch_size

    def any_reachable(self, pts: np.ndarray) -> bool:
        n_ori = len(self._orientations)
        voxels_per_chunk = max(1, self._batch_size // n_ori)
        for start in range(0, len(pts), voxels_per_chunk):
            chunk = pts[start:start + voxels_per_chunk]
            xyz_batch = np.repeat(chunk, n_ori, axis=0)
            ori_batch = np.tile(self._orientations, (len(chunk), 1))
            if bool(np.any(self._solver.solve_batch(xyz_batch, ori_batch))):
                return True
        return False
