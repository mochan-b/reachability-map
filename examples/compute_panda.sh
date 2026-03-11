#!/usr/bin/env bash
# compute_panda.sh — Example reachability map computation for the Franka Panda arm.
#
# Prerequisites:
#   1. Install the package into the venv:
#        plans/.venv/bin/pip install -e .
#   2. Have a panda.urdf available (e.g. from franka_description or pybullet_data).
#      Export PANDA_URDF to its path, or edit the --urdf argument below.
#
# Usage:
#   bash examples/compute_panda.sh
#   PANDA_URDF=/path/to/panda.urdf bash examples/compute_panda.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${SCRIPT_DIR}/../plans/.venv"
PYTHON="${VENV}/bin/python"
CMD="${VENV}/bin/reachability-map"

PANDA_URDF="${PANDA_URDF:-panda.urdf}"
OUTPUT="${OUTPUT:-panda_reach.h5}"

# ---------------------------------------------------------------------------
# Quick sanity check — low resolution, few orientations, few workers.
# Useful for verifying the pipeline end-to-end before a full run.
# ---------------------------------------------------------------------------
echo "=== Quick sanity check (low resolution) ==="
"${CMD}" \
  --urdf "${PANDA_URDF}" \
  --gripper-link panda_hand \
  --output "quick_${OUTPUT}" \
  --xyz-delta 0.1 \
  --n-orientations 20 \
  --workers 4

echo ""
echo "=== Validating quick output ==="
"${PYTHON}" - <<'EOF'
import h5py, numpy as np, sys
path = "quick_panda_reach.h5"
with h5py.File(path) as f:
    r = f["grid/reachability_index"][:]
    print(f"Shape          : {r.shape}")
    print(f"Min / Max      : {r.min():.4f} / {r.max():.4f}")
    print(f"Reachable voxels: {(r > 0).sum()} / {r.size}")
    print(f"Version        : {f.attrs['version']}")
    print(f"Timestamp      : {f.attrs['timestamp']}")
EOF

# ---------------------------------------------------------------------------
# Full-resolution positional run (~4 cm voxels, ~237 orientations).
# ---------------------------------------------------------------------------
echo ""
echo "=== Full-resolution positional run ==="
"${CMD}" \
  --urdf "${PANDA_URDF}" \
  --gripper-link panda_hand \
  --output "${OUTPUT}" \
  --xyz-delta 0.04 \
  --quat-delta 0.5 \
  --mode positional \
  --ik-solver LM \
  --ik-slimit 100 \
  --ik-ilimit 30 \
  --ik-tol 1e-6

# ---------------------------------------------------------------------------
# Full-resolution full6d run — also writes per-pose raw data.
# ---------------------------------------------------------------------------
echo ""
echo "=== Full-resolution full6d run ==="
"${CMD}" \
  --urdf "${PANDA_URDF}" \
  --gripper-link panda_hand \
  --output "full6d_${OUTPUT}" \
  --xyz-delta 0.04 \
  --quat-delta 0.5 \
  --mode full6d \
  --ik-solver LM

echo ""
echo "All done."
