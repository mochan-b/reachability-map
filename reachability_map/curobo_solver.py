from __future__ import annotations

import os
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)

try:
    from curobo.types.base import TensorDeviceType
    from curobo.types.math import Pose
    from curobo.types.robot import RobotConfig
    from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
    CUROBO_AVAILABLE = True
except ImportError as e:
    # Use print here because logging might not be configured to show debug
    print(f"cuRobo import failed: {e}")
    CUROBO_AVAILABLE = False


class CuroboIKSolver:
    """GPU-accelerated batched IK solver using cuRobo."""

    def __init__(
        self,
        urdf_path: str,
        gripper_link: str,
        base_link: str | None = None,
        device: str = "cuda:0",
    ) -> None:
        if not CUROBO_AVAILABLE:
            raise ImportError(
                "cuRobo is not installed or version is incompatible. Please check your installation."
            )

        self.device = torch.device(device)
        self.tensor_args = TensorDeviceType(device=self.device)

        # Ensure absolute path for URDF
        abs_urdf_path = os.path.abspath(urdf_path)

        # For panda, use panda_link0 as base if not provided.
        if base_link is None:
            if "panda" in urdf_path.lower():
                base_link = "panda_link0"

        # Load robot configuration
        robot_cfg = RobotConfig.from_basic(
            abs_urdf_path,
            base_link,
            gripper_link,
            tensor_args=self.tensor_args,
        )
        
        # cuRobo's IKSolverConfig.
        ik_config = IKSolverConfig.load_from_robot_config(
            robot_cfg,
            None, # world_config
            rotation_threshold=0.05,
            position_threshold=0.005,
            num_seeds=20,
            self_collision_check=True,
            collision_checker_type="mesh",
            use_cuda_graph=True,
        )
        self._solver = IKSolver(ik_config)

    def solve_batch(self, xyz: np.ndarray, orientations: np.ndarray) -> np.ndarray:
        """Attempt IK for a batch of target poses.

        Parameters
        ----------
        xyz:
            (N, 3) float32 array of positions.
        orientations:
            (N, 4) float32 array of quaternions [qx, qy, qz, qw].

        Returns
        -------
        mask:
            (N,) bool array indicating IK success for each pose.
        """
        n = len(xyz)
        if n == 0:
            return np.zeros(0, dtype=bool)

        # Convert to torch and change quat from [x,y,z,w] to [w,x,y,z]
        xyz_tensor = torch.as_tensor(xyz, device=self.device, dtype=torch.float32)
        quat_xyzw = torch.as_tensor(orientations, device=self.device, dtype=torch.float32)
        quat_wxyz = torch.empty_like(quat_xyzw)
        quat_wxyz[:, 0] = quat_xyzw[:, 3]
        quat_wxyz[:, 1:] = quat_xyzw[:, :3]

        target_pose = Pose(position=xyz_tensor, quaternion=quat_wxyz)

        # Use solve_batch instead of the deprecated solve()
        # solve_batch expects a batch of goals
        result = self._solver.solve_batch(target_pose)
        
        # result.success is a boolean tensor of shape (N,)
        return result.success.cpu().numpy()
