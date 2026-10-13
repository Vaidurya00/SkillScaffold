"""Launch a ScienceWorld HTTP router and process-isolated workers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import urlopen


def _wait(url: str, proc: subprocess.Popen, timeout: float) -> None:
    deadline = time.time() + timeout
    last_error = None
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"service exited early ({proc.returncode}): {url}")
        try:
            with urlopen(url, timeout=5) as response:
                if 200 <= response.status < 300:
                    return
        except Exception as exc:
            last_error = exc
        time.sleep(0.5)
    raise RuntimeError(f"health timeout for {url}: {last_error}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", type=int, default=64)
    parser.add_argument("--router-host", default="0.0.0.0")
    parser.add_argument("--router-port", type=int, required=True)
    parser.add_argument("--worker-host", default="127.0.0.1")
    parser.add_argument("--worker-start-port", type=int, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--worker-script", required=True)
    parser.add_argument("--router-script", required=True)
    parser.add_argument("--simplification", default="easy")
    parser.add_argument("--default-max-steps", type=int, default=30)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--pid-file", required=True)
    parser.add_argument("--startup-timeout", type=float, default=600)
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    Path(args.pid_file).parent.mkdir(parents=True, exist_ok=True)
    processes = []
    workers = []
    try:
        for index in range(args.num_workers):
            port = args.worker_start_port + index
            url = f"http://{args.worker_host}:{port}"
            log = open(log_dir / f"w{index}.log", "w")
            cmd = [
                args.python,
                args.worker_script,
                "--host", args.worker_host,
                "--port", str(port),
                "--worker-id", f"w{index}",
                "--simplification", args.simplification,
                "--default-max-steps", str(args.default_max_steps),
            ]
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
            processes.append((f"w{index}", proc, log))
            workers.append(
                {"worker_id": f"w{index}", "pid": proc.pid, "url": url}
            )
        for worker, (_, proc, _) in zip(workers, processes):
            _wait(f"{worker['url']}/health", proc, args.startup_timeout)

        router_log = open(log_dir / "router.log", "w")
        router_cmd = [
            args.python,
            args.router_script,
            "--host", args.router_host,
            "--port", str(args.router_port),
            "--workers", ",".join(worker["url"] for worker in workers),
        ]
        router = subprocess.Popen(
            router_cmd, stdout=router_log, stderr=subprocess.STDOUT
        )
        processes.append(("router", router, router_log))
        payload = {
            "router": {"pid": router.pid, "port": args.router_port},
            "workers": workers,
        }
        Path(args.pid_file).write_text(json.dumps(payload, indent=2))
        _wait(f"http://127.0.0.1:{args.router_port}/health", router, args.startup_timeout)
        print(f"ScienceWorld services ready: {args.router_port}", flush=True)
        router.wait()
    finally:
        for _, proc, handle in reversed(processes):
            if proc.poll() is None:
                proc.terminate()
            handle.close()


if __name__ == "__main__":
    main()
