from __future__ import annotations

from math import sqrt, pi
import numpy as np


def sample_orientations(n: int) -> np.ndarray:
    """Return *n* orientations distributed uniformly over SO(3).

    Uses the Super-Fibonacci spiral (Marc Alexa, CVPR 2022) to produce a
    near-optimal, deterministic sampling of unit quaternions.

    Parameters
    ----------
    n:
        Number of orientations to sample.

    Returns
    -------
    np.ndarray
        Array of shape (n, 4), dtype float32.  Columns are
        ``[qx, qy, qz, qw]`` (scalar-last / scipy convention).
    """
    phi = sqrt(2.0)
    psi = 1.533751168755204288118041

    s = (np.arange(n) + 0.5) / n
    r = np.sqrt(s)
    R = np.sqrt(1.0 - s)
    alpha = 2 * pi * np.arange(n) / phi
    beta = 2 * pi * np.arange(n) / psi

    q = np.stack(
        [r * np.sin(alpha), r * np.cos(alpha), R * np.sin(beta), R * np.cos(beta)],
        axis=1,
    )
    return q.astype(np.float32)


def estimate_n_orientations(quat_delta: float) -> int:
    """Estimate the number of orientations needed for a given angular resolution.

    Parameters
    ----------
    quat_delta:
        Desired quaternion-space resolution.  A value of 0.5 yields ~237
        orientations, matching the OpenRAVE default.

    Returns
    -------
    int
        Number of orientations, clamped to a minimum of 1.
    """
    n = int(pi**2 / 2 / (quat_delta**3 / 6))
    return max(n, 1)
