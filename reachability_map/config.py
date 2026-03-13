from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class WorkspaceBounds:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float


@dataclass
class ReachabilityConfig:
    urdf_path: str
    gripper_link: str
    output_path: str
    xyz_delta: float = 0.04
    quat_delta: float = 0.5
    n_orientations: Optional[int] = None
    bounds: Optional[WorkspaceBounds] = None
    mode: str = "positional"
    ik_solver: str = "LM"
    ik_slimit: int = 100
    ik_ilimit: int = 30
    ik_tol: float = 1e-6
    n_workers: Optional[int] = None
    quiet: bool = False
    use_gpu: bool = False
    gpu_batch_size: int = 2000
    tight_bounds: bool = True
