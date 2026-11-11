#!/usr/bin/env bash
set -Eeuo pipefail

VISION_HOST="${VISION_HOST:-0.0.0.0}"
VISION_PORT="${VISION_PORT:-8001}"
BACKEND_HOST="${BACKEND_HOST:-0.0.0.0}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
VISION_STARTUP_TIMEOUT_SECONDS="${VISION_STARTUP_TIMEOUT_SECONDS:-60}"

export VISION_HOST VISION_PORT BACKEND_HOST BACKEND_PORT
export VISION_STARTUP_TIMEOUT_SECONDS


vision_pid=""
backend_pid=""

cleanup() {
    local exit_code=$?
    if [[ -n "$backend_pid" ]]; then
        kill "$backend_pid" 2>/dev/null || true
    fi
    if [[ -n "$vision_pid" ]]; then
        kill "$vision_pid" 2>/dev/null || true
    fi
    wait 2>/dev/null || true
    exit "$exit_code"
}
trap cleanup EXIT INT TERM

echo "Starting vision worker on $VISION_HOST:$VISION_PORT"
uvicorn vision_service.inference.api:app --host "$VISION_HOST" --port "$VISION_PORT" &
vision_pid=$!

# Do not accept backend traffic until the vision HTTP process is reachable.
python - <<'PY'
import os
import sys
import time

import requests

port = os.environ["VISION_PORT"]
timeout_seconds = float(os.environ["VISION_STARTUP_TIMEOUT_SECONDS"])
if timeout_seconds <= 0:
    raise ValueError("VISION_STARTUP_TIMEOUT_SECONDS must be positive.")

url = f"http://127.0.0.1:{port}/health"
deadline = time.monotonic() + timeout_seconds
while time.monotonic() < deadline:
    try:
        response = requests.get(url, timeout=2)
        if response.status_code == 200:
            break
    except requests.RequestException:
        pass
    time.sleep(1)
else:
    sys.exit(f"Vision worker did not become reachable at {url}")
PY

echo "Starting backend on $BACKEND_HOST:$BACKEND_PORT"
uvicorn backend.app.main:app --host "$BACKEND_HOST" --port "$BACKEND_PORT" &
backend_pid=$!

# If either process exits, stop the other process and preserve its exit code.
set +e
wait -n "$vision_pid" "$backend_pid"
status=$?
set -e
exit "$status"
