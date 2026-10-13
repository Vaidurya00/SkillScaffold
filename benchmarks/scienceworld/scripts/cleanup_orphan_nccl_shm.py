#!/usr/bin/env python3
"""Remove stale NCCL shared-memory files when the node has no GPU workload."""

from __future__ import annotations

import glob
import os
from pathlib import Path
import subprocess


def gpu_compute_pids() -> list[str]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def open_targets(targets: set[str]) -> set[str]:
    held: set[str] = set()
    for fd in glob.glob("/proc/[0-9]*/fd/*"):
        try:
            target = os.readlink(fd)
        except (FileNotFoundError, PermissionError, OSError):
            continue
        target = target.removesuffix(" (deleted)")
        if target in targets:
            held.add(target)
    return held


def main() -> None:
    pids = gpu_compute_pids()
    if pids:
        raise SystemExit(f"refusing NCCL cleanup while GPU compute PIDs exist: {pids}")
    paths = sorted(Path("/dev/shm").glob("nccl-*"))
    targets = {str(path) for path in paths}
    held = open_targets(targets)
    if held:
        raise SystemExit(f"refusing to remove open NCCL shm files: {sorted(held)}")
    total = sum(path.stat().st_size for path in paths)
    for path in paths:
        path.unlink()
    print(f"removed_files={len(paths)} removed_bytes={total}", flush=True)


if __name__ == "__main__":
    main()
