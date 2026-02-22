from __future__ import annotations

import numpy as np
from spatialmath import SE3


_SOLVER_CLASSES = {
    "LM": "IK_LM",
    "GN": "IK_GN",
    "NR": "IK_NR",
}


class IKSolverWrapper:
    """Thin wrapper around roboticstoolbox numerical IK solvers.

    Parameters
    ----------
    solver_name:
        One of "LM" (Levenberg-Marquardt), "GN" (Gauss-Newton),
        or "NR" (Newton-Raphson).
    ilimit:
        Maximum iterations per restart.
    slimit:
        Number of random restarts (handled internally by rtb).
    tol:
        Convergence tolerance.
    """

    def __init__(self, solver_name: str, ilimit: int, slimit: int, tol: float) -> None:
        if solver_name not in _SOLVER_CLASSES:
            raise ValueError(
                f"Unknown solver '{solver_name}'. Choose from: {list(_SOLVER_CLASSES)}"
            )
        # Import here so the module can be imported without rtb installed at
        # import time (e.g. during packaging / type-checking).
        import roboticstoolbox.robot.IK as _IK

        cls = getattr(_IK, _SOLVER_CLASSES[solver_name])
        self._solver = cls(ilimit=ilimit, slimit=slimit, tol=tol, joint_limits=True)

    def solve(self, ets, T: np.ndarray) -> bool:
        """Attempt IK for target pose T.

        Parameters
        ----------
        ets:
            ETS kinematic chain (from robot.ets()).
        T:
            4×4 homogeneous transformation matrix (float64).

        Returns
        -------
        bool
            True if IK converged to a valid solution.
        """
        sol = self._solver.solve(ets, SE3(T, check=False))
        return bool(sol.success)
