# reachability-map

A pure-Python tool that computes **6D reachability index maps** for robot arms from URDF files. It samples the 3D workspace and the full SO(3) orientation space, runs parallel inverse kinematics (IK) at each point, and writes the results to an HDF5 file.

The output is a per-voxel **reachability index** (0.0–1.0): the fraction of sampled orientations for which IK succeeds at that position. This is conceptually equivalent to OpenRAVE's `kinematicreachability` plugin.

### Definition of reachability

The 6D workspace is discretised into two parts:

- **Position space** — a regular 3D voxel grid covering the robot's workspace, with voxel side length `xyz_delta` (metres).
- **Orientation space** — a finite set of *N* rotations sampled uniformly from SO(3) using the Super-Fibonacci spiral.

For each voxel centre **p** and each sampled orientation **R**, the tool attempts to solve the inverse kinematics problem: find a joint configuration **q** such that the end-effector reaches the pose **(R | p)**. The **reachability index** of a voxel is then:

> **reachability(p) = (number of orientations for which IK succeeds) / N**

A value of `1.0` means the end-effector can reach that position in every sampled orientation; `0.0` means it cannot reach it in any. The index is therefore a measure of **manipulability diversity** at each point in the workspace — not just whether a position is reachable, but how flexibly it can be approached.

---

## Installation

Requires Python ≥ 3.10.

```bash
pip install -e .
```

The project ships a `.venv/` directory. To use it directly:

```bash
.venv/bin/pip install -e .
.venv/bin/reachability-map --help
```

### Dependencies

| Package | Purpose |
|---|---|
| `roboticstoolbox-python` | URDF loading, forward/inverse kinematics |
| `spatialmath-python` | SE(3) pose mathematics |
| `numpy` | Voxel grid and array operations |
| `scipy` | `Rotation` for quaternion handling |
| `h5py` | HDF5 output |
| `tqdm` | Progress bar |

---

## Usage

```bash
reachability-map \
  --urdf panda.urdf \
  --gripper-link panda_hand \
  --output panda_reach.h5
```

### All options

| Flag | Default | Description |
|---|---|---|
| `--urdf PATH` | *(required)* | Path to the robot URDF file |
| `--gripper-link LINK` | *(required)* | End-effector link name in the URDF |
| `--output PATH` | *(required)* | Output HDF5 file path |
| `--xyz-delta M` | `0.04` | Voxel side length in metres |
| `--quat-delta D` | `0.5` | Quaternion-space resolution (~237 orientations at 0.5, matching OpenRAVE default) |
| `--n-orientations N` | — | Explicit SO(3) sample count (mutually exclusive with `--quat-delta`) |
| `--bounds xmin,xmax,ymin,ymax,zmin,zmax` | auto | Workspace bounding box in metres; estimated from FK at `q_max` if omitted |
| `--mode {positional\|full6d}` | `positional` | `positional`: per-voxel index only; `full6d`: also stores every successful (pose, orientation) pair |
| `--ik-solver {LM\|GN\|NR}` | `LM` | IK algorithm: Levenberg-Marquardt, Gauss-Newton, or Newton-Raphson |
| `--ik-slimit N` | `100` | Random restarts per IK call |
| `--ik-ilimit N` | `30` | Max iterations per restart |
| `--ik-tol TOL` | `1e-6` | IK convergence tolerance |
| `--workers N` | `os.cpu_count()` | Worker processes for parallel IK |
| `--quiet` | off | Suppress progress output |

### Examples

See `examples/compute_panda.sh` for ready-to-run invocations for the Franka Panda arm, including a quick low-resolution sanity check, a full positional run, and a full6d run.

```bash
# Quick sanity check
reachability-map \
  --urdf assets/panda.urdf --gripper-link panda_hand \
  --output assets/quick.h5 --xyz-delta 0.1 --n-orientations 20 --workers 4

# Full run
reachability-map \
  --urdf assets/panda.urdf --gripper-link panda_hand \
  --output assets/panda_reach.h5 \
  --xyz-delta 0.04 --quat-delta 0.5 \
  --mode positional --ik-solver LM
```

---

## Visualization

```bash
# Basic view — reachability voxels + robot in home position
python visualize.py assets/quick.h5

# Only show voxels with reachability ≥ 0.5
python visualize.py assets/quick.h5 --threshold 0.5

# Interactive clip-plane mode — drag the plane to slice through the volume
python visualize.py assets/quick.h5 --clip

# Hide the robot geometry
python visualize.py assets/quick.h5 --no-robot

# Save a screenshot before the interactive window opens
python visualize.py assets/quick.h5 --screenshot reach.png

# Different colormap
python visualize.py assets/quick.h5 --cmap plasma
```

Uses PyVista (`pip install pyvista`) and trimesh + pycollada (`pip install trimesh pycollada`) for loading the robot meshes. The window is interactive — rotate, zoom, and pan with the mouse.

The robot geometry is loaded from the `.dae` visual mesh files bundled with `rtbdata` (installed alongside `roboticstoolbox-python`) and positioned using forward kinematics at the home position (all joints at zero).

---

## Output Format (HDF5)

```
output.h5
├── /  (root attributes)
│   ├── version, timestamp
│   ├── urdf_path, gripper_link, robot_name, n_joints
│   ├── xyz_delta, actual_quat_delta, n_orientations
│   ├── mode, ik_solver, ik_slimit, ik_ilimit, ik_tol, n_workers
│
├── grid/
│   ├── (attrs) origin[3], shape[3], delta
│   ├── reachability_index   float32 (nx, ny, nz)     gzip-4, chunks (32,32,32)
│   └── voxel_centers        float32 (nx, ny, nz, 3)
│
├── orientations/
│   └── samples              float32 (N, 4)  [qx, qy, qz, qw]
│       (attrs) method="super_fibonacci", quaternion_convention="xyzw"
│
└── poses/                   (mode="full6d" only)
    └── reachability_stats   float32 (M, 8)  [qx, qy, qz, qw, tx, ty, tz, ik_count]
        (attrs) columns, quaternion_convention="xyzw"
```

All quaternions use scalar-last convention `[qx, qy, qz, qw]` (SciPy convention).

### Reading output in Python

```python
from reachability_map.hdf5_io import read_hdf5

data = read_hdf5("panda_reach.h5")
r = data["reachability_index"]        # (nx, ny, nz) float32
centers = data["voxel_centers"]       # (nx, ny, nz, 3) float32
orientations = data["orientations"]   # (N, 4) float32
poses = data["poses"]                 # (M, 8) float32, or None
meta = data["attrs"]                  # root metadata dict
```

---

## Project Structure

```
reachability/
├── pyproject.toml                  # Package metadata, dependencies, CLI entry point
├── visualize.py                    # PyVista visualization script (threshold, clip-plane, screenshot)
├── assets/                         # Robot models and computed maps
│   ├── panda.urdf                  # Franka Panda arm+hand URDF (from corlab/cogimon-gazebo-models)
│   └── *.h5                        # Generated reachability map outputs
├── reachability_map/               # Main Python package
│   ├── __init__.py
│   ├── cli.py                      # Argument parsing; builds ReachabilityConfig and calls run()
│   ├── config.py                   # ReachabilityConfig and WorkspaceBounds dataclasses
│   ├── reachability.py             # Three-phase pipeline orchestrator: setup → parallel IK → write
│   ├── robot_loader.py             # Loads URDF via ERobot; extracts picklable ETS chain
│   ├── voxel_grid.py               # Builds 3D voxel grid (numpy meshgrid, indexing='ij')
│   ├── orientation_sampler.py      # Samples SO(3) via Super-Fibonacci spiral (deterministic)
│   ├── ik_solver.py                # IKSolverWrapper around roboticstoolbox IK_LM/GN/NR
│   ├── worker.py                   # Multiprocessing worker: init globals, run IK per voxel
│   └── hdf5_io.py                  # write_hdf5() and read_hdf5()
├── examples/
│   └── compute_panda.sh            # Reference invocations for Franka Panda
└── plans/
    └── reachability_plan.md        # Original implementation plan and algorithm notes
```

### Obtaining robot URDFs

`assets/panda.urdf` is the Franka Panda arm with hand (gripper) and was downloaded from [corlab/cogimon-gazebo-models](https://github.com/corlab/cogimon-gazebo-models/blob/master/franka/robots/panda_arm_hand.urdf). It includes the `panda_hand` gripper link and requires no ROS tooling.

```bash
wget -O assets/panda.urdf \
  "https://raw.githubusercontent.com/corlab/cogimon-gazebo-models/master/franka/robots/panda_arm_hand.urdf"
```

The URDF references mesh files via `package://franka_description/meshes/...` paths, but these are only needed for visualisation — the reachability tool uses only the kinematic chain and loads cleanly without them.

### How the modules connect

```
cli.py
  └─→ reachability.py::run()
        ├─ robot_loader.py      load_robot()       URDF → ERobot; get_ets() → picklable ETS
        ├─ voxel_grid.py        build_voxel_grid() workspace → flat list of voxel centres
        ├─ orientation_sampler.py  sample_orientations()  Super-Fibonacci → (N,4) quaternions
        ├─ multiprocessing.Pool
        │    worker.py          worker_init()      re-loads robot per process (fork safety)
        │    worker.py          compute_voxel_reachability()  IK over all orientations
        │         ik_solver.py  IKSolverWrapper    wraps rtb IK_LM / IK_GN / IK_NR
        └─ hdf5_io.py           write_hdf5()       aggregated results → HDF5
```

**Key design decisions:**

- **Fork safety.** `ERobot` (wraps C++ kinematics) is not safely picklable. Each worker process re-loads the robot from the URDF path inside `worker_init()`; only the `ETS` kinematic chain (pure Python, picklable) crosses the process boundary.
- **Deterministic orientation sampling.** The Super-Fibonacci spiral (Alexa, CVPR 2022) requires no random seed and produces a near-optimal, repeatable set of SO(3) samples.
- **Two output modes.** `positional` writes only the `(nx, ny, nz)` reachability index — compact and sufficient for workspace analysis. `full6d` additionally appends every successful `(pose, orientation)` pair, enabling downstream tasks such as grasp planning.
