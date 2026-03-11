#!/usr/bin/env python3
"""Visualize a reachability map HDF5 file using PyVista.

Usage:
    python visualize.py assets/quick.h5
    python visualize.py assets/quick.h5 --threshold 0.3
    python visualize.py assets/quick.h5 --clip
    python visualize.py assets/quick.h5 --no-robot
    python visualize.py assets/quick.h5 --screenshot out.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pyvista as pv

from reachability_map.hdf5_io import read_hdf5

# Visual mesh files bundled with rtbdata (roboticstoolbox's data package).
_RTBDATA_XACRO = (
    Path(__file__).parent
    / ".venv/lib/python3.12/site-packages/rtbdata/xacro"
)

# Panda: maps URDF link name → mesh filename relative to the meshes/visual dir.
# finger.dae is shared by both fingers (they are mirror images).
_PANDA_MESH_DIR = _RTBDATA_XACRO / "franka_description/meshes/visual"
_PANDA_LINK_MESHES: dict[str, str] = {
    "panda_link0":       "link0.dae",
    "panda_link1":       "link1.dae",
    "panda_link2":       "link2.dae",
    "panda_link3":       "link3.dae",
    "panda_link4":       "link4.dae",
    "panda_link5":       "link5.dae",
    "panda_link6":       "link6.dae",
    "panda_link7":       "link7.dae",
    "panda_hand":        "hand.dae",
    "panda_leftfinger":  "finger.dae",
    "panda_rightfinger": "finger.dae",
}

# UR10: maps URDF link name → mesh filename relative to the meshes/ur10/visual dir.
_UR10_MESH_DIR = _RTBDATA_XACRO / "ur_description/meshes/ur10/visual"
_UR10_LINK_MESHES: dict[str, str] = {
    "base_link":     "base.dae",
    "shoulder_link": "shoulder.dae",
    "upper_arm_link": "upperarm.dae",
    "forearm_link":  "forearm.dae",
    "wrist_1_link":  "wrist1.dae",
    "wrist_2_link":  "wrist2.dae",
    "wrist_3_link":  "wrist3.dae",
}


def _dae_to_polydata(path: Path) -> pv.PolyData:
    """Load a Collada (.dae) mesh file and return a PyVista PolyData.

    Uses scene.to_geometry() so that intra-file scene-graph transforms are
    applied before the sub-meshes are combined.  Without this, each sub-mesh
    is placed at the origin instead of its correct position within the link,
    causing all pieces to visually float disconnected from each other.
    """
    import trimesh
    scene = trimesh.load(str(path), force="scene")
    combined = scene.to_geometry()
    faces = np.hstack([
        np.full((len(combined.faces), 1), 3, dtype=np.int_),
        combined.faces,
    ]).ravel()
    return pv.PolyData(combined.vertices.astype(np.float32), faces)


def load_robot_meshes(robot_name: str, q: np.ndarray | None = None) -> list[pv.PolyData]:
    """Return a list of PyVista meshes for each robot link at joint config *q*.

    Uses roboticstoolbox for forward kinematics and the DAE mesh files
    bundled with rtbdata for geometry. If *q* is None, the home position
    (all joints at zero) is used.

    Parameters
    ----------
    robot_name:
        Robot model name as stored in the HDF5 attrs (e.g. "Panda", "UR10").
    """
    import roboticstoolbox as rtb

    robot_name_upper = robot_name.upper()
    if robot_name_upper == "UR10":
        robot = rtb.models.UR10()
        mesh_dir = _UR10_MESH_DIR
        link_meshes = _UR10_LINK_MESHES
    else:
        robot = rtb.models.Panda()
        mesh_dir = _PANDA_MESH_DIR
        link_meshes = _PANDA_LINK_MESHES

    if q is None:
        q = np.zeros(robot.n)

    # Collect all links: arm links + gripper links
    all_links: list[str] = [l.name for l in robot.links]
    for g in robot.grippers:
        all_links.extend(gl.name for gl in g.links)

    meshes: list[pv.PolyData] = []
    for link_name in all_links:
        mesh_file = link_meshes.get(link_name)
        if mesh_file is None:
            continue

        mesh_path = mesh_dir / mesh_file
        if not mesh_path.exists():
            print(f"Warning: mesh not found for {link_name}: {mesh_path}", file=sys.stderr)
            continue

        # Forward kinematics → 4×4 world transform for this link
        T = robot.fkine(q, end=link_name).A  # .A gives the numpy ndarray

        mesh = _dae_to_polydata(mesh_path)
        mesh.transform(T, inplace=True)
        meshes.append(mesh)

    return meshes


def build_grid(data: dict) -> pv.ImageData:
    """Convert HDF5 data to a PyVista ImageData (uniform voxel grid)."""
    r = data["reachability_index"]          # (nx, ny, nz) float32
    origin = data["grid_attrs"]["origin"]   # [x0, y0, z0]
    delta = float(data["grid_attrs"]["delta"])

    grid = pv.ImageData()
    grid.dimensions = np.array(r.shape) + 1   # cell data → n+1 points per axis
    grid.origin = origin - delta / 2
    grid.spacing = (delta, delta, delta)
    grid.cell_data["reachability"] = r.flatten(order="F")
    return grid


def make_title(data: dict) -> str:
    a = data["attrs"]
    return (
        f"{a.get('robot_name', 'robot')}  |  "
        f"gripper: {a.get('gripper_link', '?')}  |  "
        f"δ={a.get('xyz_delta', '?')} m  |  "
        f"N orientations: {a.get('n_orientations', '?')}"
    )


def visualize(
    hdf5_path: str,
    threshold: float = 0.0,
    clip: bool = False,
    show_robot: bool = True,
    screenshot: str | None = None,
    cmap: str = "viridis",
) -> None:
    data = read_hdf5(hdf5_path)
    grid = build_grid(data)

    # Apply threshold — always strip unreachable (zero) voxels
    thresh_val = max(threshold, 1e-6)
    threshed = grid.threshold(thresh_val, scalars="reachability")

    if threshed.n_cells == 0:
        print(f"No voxels above threshold {threshold:.2f}. Try a lower value.")
        sys.exit(1)

    n_total = grid.n_cells
    n_shown = threshed.n_cells
    print(f"Showing {n_shown:,} / {n_total:,} voxels  (threshold ≥ {thresh_val:.3f})")

    scalar_bar_args = dict(
        title="Reachability",
        title_font_size=14,
        label_font_size=12,
        n_labels=5,
        fmt="%.2f",
    )

    pl = pv.Plotter(title=make_title(data))

    if clip:
        pl.add_mesh_clip_plane(
            threshed,
            scalars="reachability",
            cmap=cmap,
            clim=[0, 1],
            opacity=0.9,
            scalar_bar_args=scalar_bar_args,
        )
    else:
        pl.add_mesh(
            threshed,
            scalars="reachability",
            cmap=cmap,
            clim=[0, 1],
            opacity=0.6,
            scalar_bar_args=scalar_bar_args,
        )

    if show_robot:
        robot_name = data["attrs"].get("robot_name", "Panda")
        print(f"Loading robot geometry ({robot_name})...")
        robot_meshes = load_robot_meshes(robot_name)
        for mesh in robot_meshes:
            pl.add_mesh(mesh, color="lightgrey", opacity=1.0, smooth_shading=True)

    pl.add_axes()
    pl.show_grid()

    if screenshot:
        pl.screenshot(screenshot)

    pl.show()


def main() -> None:
    p = argparse.ArgumentParser(
        description="Visualize a reachability map HDF5 file with PyVista.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("hdf5", metavar="FILE", help="Path to the reachability HDF5 file.")
    p.add_argument(
        "--threshold", type=float, default=0.0, metavar="T",
        help="Hide voxels with reachability index below this value (0–1).",
    )
    p.add_argument(
        "--clip", action="store_true",
        help="Enable interactive clip-plane mode to slice through the volume.",
    )
    p.add_argument(
        "--no-robot", action="store_true",
        help="Hide the robot geometry (shown by default at home position).",
    )
    p.add_argument(
        "--screenshot", metavar="PATH", default=None,
        help="Save a screenshot to this path before showing the interactive window.",
    )
    p.add_argument(
        "--cmap", default="viridis", metavar="NAME",
        help="Matplotlib colormap name for the reachability values.",
    )
    args = p.parse_args()

    visualize(
        hdf5_path=args.hdf5,
        threshold=args.threshold,
        clip=args.clip,
        show_robot=not args.no_robot,
        screenshot=args.screenshot,
        cmap=args.cmap,
    )


if __name__ == "__main__":
    main()
