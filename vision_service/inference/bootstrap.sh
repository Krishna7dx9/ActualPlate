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
echo

echo "[1/5] Installing inference service dependencies..."
pip install -q -r "$REPO_DIR/vision_service/inference/requirements.txt"

if [[ ! -d "$DEPTH_PRO_DIR" ]]; then
    echo "[2/5] Cloning Apple ml-depth-pro..."
    git clone -q https://github.com/apple/ml-depth-pro.git "$DEPTH_PRO_DIR"
else
    echo "[2/5] ml-depth-pro already cloned."
fi

echo "[2/5] Installing depth_pro (no deps)..."
pip install -q --no-deps -e "$DEPTH_PRO_DIR"

echo "[3/5] Installing depth_pro runtime deps..."
pip install -q timm scikit-image pillow-heif

CHECKPOINT_PATH="$DEPTH_PRO_DIR/checkpoints/depth_pro.pt"
if [[ ! -f "$CHECKPOINT_PATH" ]]; then
    echo "[4/5] Downloading depth_pro checkpoint (~1.8 GB)..."
    mkdir -p "$(dirname "$CHECKPOINT_PATH")"
    wget -q "$DEPTH_PRO_CHECKPOINT_URL" -O "$CHECKPOINT_PATH"
else
    echo "[4/5] depth_pro checkpoint already present."
fi

echo "[5/5] Starting inference service..."
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export REPO_DIR
export TUNNEL

python -c "
import os, sys
sys.path.insert(0, os.environ['REPO_DIR'])
from vision_service.inference.server import launch
tunnel = os.environ.get('TUNNEL', 'false').lower() == 'true'
launch(tunnel=tunnel)
"