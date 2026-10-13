#!/usr/bin/env python3
"""Build and validate a clean source release archive."""

from __future__ import annotations

import argparse
import compileall
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
SKIP_PARTS = {
    ".git",
    ".venv",
    "venv",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "__pycache__",
    "logs",
    "outputs",
    "runs",
    "third_party",
    "wandb",
}
SKIP_ROOTS = {
    "checkpoints",
    "data",
    "dist",
    "log",
    "logs",
    "models",
    "outputs",
    "runs",
    "third_party",
    "trajectories",
    "wandb",
}
SKIP_SUFFIXES = {
    ".arrow",
    ".ckpt",
    ".distcp",
    ".log",
    ".npy",
    ".npz",
    ".parquet",
    ".pid",
    ".pt",
    ".pth",
    ".pyc",
    ".safetensors",
}


def excluded(relative: Path) -> bool:
    if relative.name in {".DS_Store", ".copy_quarantine_path"}:
        return True
    if relative == Path("benchmarks/alfworld/configs/alfworld.yaml"):
        return True
    if any(part in SKIP_PARTS for part in relative.parts):
        return True
    if relative.parts and relative.parts[0] in SKIP_ROOTS:
        return True
    if (
        len(relative.parts) >= 3
        and relative.parts[0] == "benchmarks"
        and relative.parts[2] == "data"
    ):
        return True
    if relative.name == ".env" or (
        relative.name.startswith(".env.") and relative.name != ".env.example"
    ):
        return True
    return relative.suffix.lower() in SKIP_SUFFIXES


def copy_source(destination: Path) -> None:
    for source in ROOT.rglob("*"):
        relative = source.relative_to(ROOT)
        if excluded(relative) or source.is_symlink():
            continue
        target = destination / relative
        if source.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def run_checks(release_root: Path) -> None:
    subprocess.run(
        [sys.executable, "scripts/audit_release.py"], cwd=release_root, check=True
    )
    if not compileall.compile_dir(release_root / "benchmarks", quiet=1):
        raise SystemExit("benchmark syntax check failed")
    if not compileall.compile_dir(release_root / "integrations", quiet=1):
        raise SystemExit("integration syntax check failed")
    if not compileall.compile_dir(release_root / "framework", quiet=1):
        raise SystemExit("framework syntax check failed")
    subprocess.run(["bash", "scripts/check_shell.sh"], cwd=release_root, check=True)
    subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        cwd=release_root,
        check=True,
    )


def remove_check_artifacts(release_root: Path) -> None:
    """Remove bytecode/cache files created while validating the staged tree."""
    for directory in sorted(
        release_root.rglob("__pycache__"), key=lambda path: len(path.parts), reverse=True
    ):
        shutil.rmtree(directory)
    for suffix in ("*.pyc", "*.pyo"):
        for artifact in release_root.rglob(suffix):
            artifact.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "dist" / "SkillScaffold-release.zip",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="skillscaffold-release-") as temporary:
        release_root = Path(temporary) / "SkillScaffold"
        release_root.mkdir()
        copy_source(release_root)
        run_checks(release_root)
        remove_check_artifacts(release_root)
        archive = Path(
            shutil.make_archive(
                str(output.with_suffix("")), "zip", release_root.parent, release_root.name
            )
        )

    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    checksum = archive.with_suffix(archive.suffix + ".sha256")
    checksum.write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
    print(f"archive={archive}")
    print(f"sha256={digest}")


if __name__ == "__main__":
    main()
