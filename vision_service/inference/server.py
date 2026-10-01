"""
Runtime entrypoint for the ActualPlate vision inference service.

Starts uvicorn serving vision_service.inference.api:app, optionally
exposing it via a Cloudflare quick tunnel.

Two deployment modes:

    launch(tunnel=False)    uvicorn only. Use on RunPod, GCP, etc.
    launch(tunnel=True)     uvicorn + Cloudflare quick tunnel.
                            Use on Colab, which has no public IP.

The tunnel is a transport detail. The app itself (api.py) is unchanged
by the choice of mode.
"""

from __future__ import annotations

import os
import re
import subprocess
import threading
import time

import nest_asyncio
import requests
import uvicorn


HOST = os.environ.get("VISION_HOST", "0.0.0.0")
PORT = int(os.environ.get("VISION_PORT", "8000"))

CLOUDFLARED_BIN = os.environ.get("CLOUDFLARED_BIN", "cloudflared")


def _wait_for_local_health(timeout_seconds: int = 30) -> bool:
    """Poll local /health until it responds 200 or timeout."""
    url = f"http://127.0.0.1:{PORT}/health"
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            if requests.get(url, timeout=2).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def _start_uvicorn_thread() -> threading.Thread:
    """
    Start uvicorn in a daemon thread.

    The thread dies when the process exits; no explicit shutdown is
    required. Nest asyncio patch is applied so this works inside a
    running event loop (Colab, notebooks).
    """
    nest_asyncio.apply()

    def run() -> None:
        uvicorn.run(
            "vision_service.inference.api:app",
            host=HOST,
            port=PORT,
            log_level="info",
        )

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def _start_cloudflare_tunnel() -> str:
    """
    Start cloudflared quick tunnel pointed at the local uvicorn port.

    Blocks until the tunnel URL is detected in cloudflared's output,
    then returns it. Raises RuntimeError if no URL is seen within 60s.
    """
    proc = subprocess.Popen(
        [CLOUDFLARED_BIN, "tunnel", "--url", f"http://127.0.0.1:{PORT}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    url_pattern = re.compile(r"https://[a-zA-Z0-9.-]+\.trycloudflare\.com")
    deadline = time.time() + 60

    while time.time() < deadline:
        line = proc.stdout.readline() if proc.stdout else ""
        if not line:
            time.sleep(0.1)
            continue
        match = url_pattern.search(line)
        if match:
            return match.group(0).rstrip("/")

    proc.kill()
    raise RuntimeError("Cloudflare tunnel URL was not detected within 60s.")


def _verify_tunnel(url: str, attempts: int = 10) -> bool:
    """
    Poll the public tunnel URL until /health returns 200.

    Cloudflare quick tunnels announce their URL before DNS propagates,
    so the first few requests will fail with NameResolutionError.
    """
    health_url = f"{url}/health"
    for attempt in range(1, attempts + 1):
        try:
            if requests.get(health_url, timeout=10).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(2 * attempt)
    return False


def launch(tunnel: bool = False) -> str | None:
    """
    Start the inference service. Returns the public tunnel URL if
    tunnel=True, else None.

    This is the single entrypoint a host calls to run the service.
    """
    _start_uvicorn_thread()

    if not _wait_for_local_health():
        raise RuntimeError("Uvicorn did not become healthy on local port.")

    print(f"Local service ready on http://{HOST}:{PORT}")

    if not tunnel:
        return None

    print("Starting Cloudflare tunnel...")
    public_url = _start_cloudflare_tunnel()
    print(f"Tunnel URL: {public_url}")

    if not _verify_tunnel(public_url):
        raise RuntimeError(f"Tunnel not reachable at {public_url}/health")

    print("=" * 70)
    print("ACTUALPLATE VISION SERVICE READY")
    print("=" * 70)
    print(f"VISION_INFERENCE_ENDPOINT={public_url}")
    print("=" * 70)

    return public_url