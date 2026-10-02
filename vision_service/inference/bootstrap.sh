#!/usr/bin/env bash
#
# Bootstrap script for the ActualPlate vision inference service.
#
# Runs on any GPU host that starts from a bare Python environment:
# Colab, RunPod, GCP, Lambda, Vast.ai, or a local GPU workstation.
#
# Usage:
#
#     bash bootstrap.sh              start uvicorn only
#     bash bootstrap.sh --tunnel     start uvicorn + Cloudflare tunnel
#
# Environment overrides:
#
#     REPO_DIR        default /content/ActualPlate
#     DEPTH_PRO_DIR   default /content/ml-depth-pro
#     VISION_PORT     default 8000 (consumed by server.py)
#
# Verbosity:
#
#   This script is run by a human, in an interactive session.
#   pip and wget therefore run WITHOUT -q so their progress is
#   visible. For CI use, wrap the script in a redirect and grep
#   the output; do not add -q here, because silence during a
#   10-minute install reads as a hang.
#
# What it does, in order:
#
#   1. Installs the inference service's Python deps
#   2. Clones and installs Apple ml-depth-pro WITHOUT its deps
#      (its pyproject.toml pins numpy<2, which breaks Colab)
#   3. Installs depth-pro's real runtime deps that pip would
#      have installed if step 2 had been a normal install
#   4. Downloads the depth-pro checkpoint (~1.8 GB) once
#   5. Launches the FastAPI service via vision_service.inference.server

set -euo pipefail

REPO_DIR="${REPO_DIR:-/content/ActualPlate}"
DEPTH_PRO_DIR="${DEPTH_PRO_DIR:-/content/ml-depth-pro}"
DEPTH_PRO_CHECKPOINT_URL="${DEPTH_PRO_CHECKPOINT_URL:-https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt}"

TUNNEL="false"
if [[ "${1:-}" == "--tunnel" ]]; then
    TUNNEL="true"
fi

echo "============================================================"
echo "ActualPlate inference service — bootstrap"
echo "============================================================"
echo "REPO_DIR:       $REPO_DIR"
echo "DEPTH_PRO_DIR:  $DEPTH_PRO_DIR"
echo "TUNNEL:         $TUNNEL"
echo "============================================================"
echo

# ------------------------------------------------------------
# 1. Inference service Python deps
# ------------------------------------------------------------

echo "[1/5] Installing inference service dependencies."
echo "      This may take several minutes on a cold host."
echo

pip install -r "$REPO_DIR/vision_service/inference/requirements.txt"

echo
echo "[1/5] Inference service dependencies installed."
echo

# ------------------------------------------------------------
# 2. Apple Depth Pro, WITHOUT its pinned deps
#
# ml-depth-pro's pyproject.toml pins numpy<2. A normal
# `pip install -e` there downgrades the whole environment
# and breaks Colab's numpy 2.x stack (scipy, opencv, jax,
# and many others). The --no-deps flag installs only the
# depth_pro package itself; its real deps are installed
# explicitly in step 3.
# ------------------------------------------------------------

if [[ ! -d "$DEPTH_PRO_DIR" ]]; then
    echo "[2/5] Cloning Apple ml-depth-pro..."
    git clone https://github.com/apple/ml-depth-pro.git "$DEPTH_PRO_DIR"
else
    echo "[2/5] ml-depth-pro already cloned at $DEPTH_PRO_DIR."
fi

echo
echo "[2/5] Installing depth_pro with --no-deps."
echo

pip install --no-deps -e "$DEPTH_PRO_DIR"

echo
echo "[2/5] depth_pro installed."
echo

# ------------------------------------------------------------
# 3. depth_pro's real runtime dependencies
#
# These are what --no-deps skipped. The list is the residue
# of testing on a fresh Colab runtime: each entry was a
# ModuleNotFoundError discovered during that test.
# ------------------------------------------------------------

echo "[3/5] Installing depth_pro runtime deps: timm, scikit-image, pillow-heif."
echo

pip install timm scikit-image pillow-heif

echo
echo "[3/5] depth_pro runtime deps installed."
echo

# ------------------------------------------------------------
# 4. depth_pro checkpoint (idempotent)
#
# wget's default progress bar is disabled when stdout is not
# a TTY. Colab captures stdout, so plain wget would be silent
# for a 1.8 GB download. --progress=bar:force:noscroll forces
# the progress bar regardless of TTY detection.
# ------------------------------------------------------------

CHECKPOINT_PATH="$DEPTH_PRO_DIR/checkpoints/depth_pro.pt"
if [[ ! -f "$CHECKPOINT_PATH" ]]; then
    echo "[4/5] Downloading depth_pro checkpoint (~1.8 GB)."
    echo "      This takes several minutes; progress bar below."
    echo
    mkdir -p "$(dirname "$CHECKPOINT_PATH")"
    wget --progress=bar:force:noscroll \
        "$DEPTH_PRO_CHECKPOINT_URL" \
        -O "$CHECKPOINT_PATH"
    echo
    echo "[4/5] Checkpoint downloaded to $CHECKPOINT_PATH."
else
    echo "[4/5] Checkpoint already present at $CHECKPOINT_PATH."
fi
echo

# ------------------------------------------------------------
# 5. Launch the service
# ------------------------------------------------------------

echo "[5/5] Starting inference service."
echo

export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export REPO_DIR
export TUNNEL

python -c "
import os, sys, time
sys.path.insert(0, os.environ['REPO_DIR'])
from vision_service.inference.server import launch

tunnel = os.environ.get('TUNNEL', 'false').lower() == 'true'
public_url = launch(tunnel=tunnel)

if public_url:
    print()
    print('Service is running at: %s' % public_url)
else:
    print()
    print('Service is running on port %s.' % os.environ.get('VISION_PORT', '8000'))

print('The cell will stay open until you stop it (Colab stop button or Ctrl+C).')
sys.stdout.flush()

while True:
    time.sleep(60)
"