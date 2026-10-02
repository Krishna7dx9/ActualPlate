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
import socket
import subprocess
import threading
import time

import nest_asyncio
import psutil
import requests
import uvicorn


HOST = os.environ.get("VISION_HOST", "0.0.0.0")
PORT = int(os.environ.get("VISION_PORT", "8000"))

CLOUDFLARED_URL = (
    "https://github.com/cloudflare/cloudflared/releases/latest/download/"
    "cloudflared-linux-amd64"
)

CLOUDFLARED_CANDIDATES = [
    os.environ.get("CLOUDFLARED_BIN"),
    "cloudflared",
    "/usr/local/bin/cloudflared",
    "/content/cloudflared-linux-amd64",
    "./cloudflared-linux-amd64",
]


def _find_or_install_cloudflared() -> str:
    """
    Return a path to the cloudflared binary.

    Checks (in order): env override, PATH, common install locations,
    the current directory. If none exist, downloads the Linux amd64
    binary to /content/ and returns that path.
    """
    for candidate in CLOUDFLARED_CANDIDATES:
        if not candidate:
            continue
        if os.path.isabs(candidate) and os.path.isfile(candidate):
            return candidate
        found = shutil_which(candidate)
        if found:
            return found

    target = "/content/cloudflared-linux-amd64"
    if not os.path.isfile(target):
        subprocess.run(["wget", "-q", CLOUDFLARED_URL, "-O", target], check=True)
        os.chmod(target, 0o755)
    return target


def shutil_which(name: str) -> str | None:
    """Minimal which(): return full path if executable is on PATH."""
    from shutil import which
    return which(name)


def _port_in_use(port: int) -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.5)
    try:
        return s.connect_ex(("127.0.0.1", port)) == 0
    finally:
        s.close()


def _kill_stale_uvicorn(port: int) -> None:
    """
    Kill any uvicorn process listening on the target port.

    Colab keeps a kernel alive across cell re-runs; the previous
    uvicorn from an earlier launch() call survives and holds the
    port. Without this, the new uvicorn silently fails to bind and
    health checks pass against the stale server.
    """
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmdline = " ".join(proc.info["cmdline"] or [])
            if "uvicorn" in cmdline:
                proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    for _ in range(15):
        if not _port_in_use(port):
            return
        time.sleep(1)

    raise RuntimeError(
        f"Port {port} is still occupied after killing uvicorn processes. "
        "Restart the runtime and try again."
    )


def _wait_for_local_health(timeout_seconds: int = 30) -> bool:
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
    binary = _find_or_install_cloudflared()

    proc = subprocess.Popen(
        [binary, "tunnel", "--url", f"http://127.0.0.1:{PORT}"],
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
    """
    _kill_stale_uvicorn(PORT)
    print(f"Port {PORT} free. Starting uvicorn...")

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