"""Stop only WebShop environment processes recorded by its PID file."""

from __future__ import annotations

import argparse
import json
import os
import signal
import time


def cmdline(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as handle:
            return handle.read().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return ""


def stop(pid: int, name: str) -> None:
    command = cmdline(pid)
    allowed = ("env/webshop/worker_server.py", "env/webshop/router_server.py")
    if not command:
        return
    if not any(marker in command for marker in allowed):
        print(f"[webshop/stop] refusing unrelated pid={pid}: {command}")
        return
    print(f"[webshop/stop] stopping {name} pid={pid}")
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.time() + 5
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except OSError:
            return
        time.sleep(0.1)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid-file", required=True)
    args = parser.parse_args()
    if not os.path.isfile(args.pid_file):
        return
    payload = json.loads(open(args.pid_file).read())
    router = payload.get("router")
    if isinstance(router, dict) and isinstance(router.get("pid"), int):
        stop(router["pid"], "router")
    for worker in payload.get("workers", []):
        if isinstance(worker, dict) and isinstance(worker.get("pid"), int):
            stop(worker["pid"], str(worker.get("worker_id", "worker")))
    try:
        os.unlink(args.pid_file)
    except FileNotFoundError:
        pass


if __name__ == "__main__":
    main()
