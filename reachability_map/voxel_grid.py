from __future__ import annotations

import numpy as np

from .config import WorkspaceBounds


def build_voxel_grid(bounds: WorkspaceBounds, delta: float) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Build a voxel grid over the workspace and return voxel centers.

    Parameters
    ----------
    bounds:
        WorkspaceBounds dataclass defining the axis-aligned bounding box.
    delta:
        Voxel side length in metres.

    Returns
    -------
    centers_flat:
        Array of shape (N, 3) containing the (x, y, z) centre of every voxel,
        dtype float32.
    grid_shape:
        Tuple (nx, ny, nz) giving the number of voxels along each axis.
    """
    xs = np.arange(bounds.x_min + delta / 2, bounds.x_max, delta)
    ys = np.arange(bounds.y_min + delta / 2, bounds.y_max, delta)
    zs = np.arange(bounds.z_min + delta / 2, bounds.z_max, delta)

    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing='ij')

    grid_shape: tuple[int, int, int] = (int(gx.shape[0]), int(gx.shape[1]), int(gx.shape[2]))

    centers_flat = np.stack(
        [gx.ravel(), gy.ravel(), gz.ravel()], axis=1
    ).astype(np.float32)

    return centers_flat, grid_shape


def auto_bounds(robot, qlim: np.ndarray, margin: float = 0.1) -> WorkspaceBounds:
    """Estimate workspace bounds by summing link translation magnitudes.

    Walks the kinematic chain from the default end-effector back to the root,
    summing the Euclidean length of each link's fixed translation offset.  This
    gives the theoretical maximum reach (all links fully extended), which is a
    reliable upper bound regardless of joint configuration.

    Parameters
    ----------
    robot:
        An ERobot instance with ``ee_links`` and per-link ``ets``.
    qlim:
        Joint limits array of shape (2, n_joints).  Unused; kept for API
        compatibility.
    margin:
        Extra distance added to the estimated reach (metres).

    Returns
    -------
    WorkspaceBounds
        A symmetric bounding box of side 2 * reach centred at the origin.
    """
    link = robot.ee_links[0]
    reach = 0.0
    while link is not None:
        n = link.ets.n
        T = link.ets.fkine(np.zeros(n))
        reach += float(np.linalg.norm(T.t))
        link = link.parent
    reach += margin
    return WorkspaceBounds(-reach, reach, -reach, reach, -reach, reach)
