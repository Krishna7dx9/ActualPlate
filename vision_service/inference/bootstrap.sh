#!/usr/bin/env bash
#
# Bootstrap script for the ActualPlate vision inference service.
#
# Runs on any GPU host that starts from a bare Python environment.
#
# Usage:
#     bash bootstrap.sh              start uvicorn only
#     bash bootstrap.sh --tunnel     start uvicorn + Cloudflare tunnel
#
# Environment overrides:
#     REPO_DIR        default /content/ActualPlate
#     VISION_PORT     default 8000 (consumed by server.py)
#

set -euo pipefail

REPO_DIR="${REPO_DIR:-/content/ActualPlate}"

TUNNEL="false"
if [[ "${1:-}" == "--tunnel" ]]; then
    TUNNEL="true"
fi

echo "============================================================"
echo "ActualPlate inference service — bootstrap"
echo "============================================================"
echo "REPO_DIR:       $REPO_DIR"
echo "TUNNEL:         $TUNNEL"
echo "============================================================"
echo

# ------------------------------------------------------------
# 1. Inference service Python deps
# ------------------------------------------------------------

echo "[1/2] Installing inference service dependencies."
pip install -r "$REPO_DIR/vision_service/inference/requirements.txt"
echo "[1/2] Inference service dependencies installed."
echo

# ------------------------------------------------------------
# 2. Launch the service
# ------------------------------------------------------------

echo "[2/2] Starting inference service."
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

print('The cell will stay open until you stop it.')
sys.stdout.flush()

while True:
    time.sleep(60)
"