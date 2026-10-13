#!/usr/bin/env python3
"""Fail closed on common accidental-release hazards."""

from __future__ import annotations

import os
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()
TEXT_LIMIT = 5 * 1024 * 1024
FILE_LIMIT = 20 * 1024 * 1024

# Split strings so the scanner source does not contain the forbidden literal.
FORBIDDEN = {
    "internal storage prefix": "/wl_" + "intelligent",
    "corporate domain": "kwai" + "dc.com",
    "known secret prefix": "wandb_" + "v1_",
}
SECRET_PATTERNS = {
    "OpenAI-style secret": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}
PRIVATE_IP = re.compile(
    r"\b(?:10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})\b"
)
ALLOWED_PRIVATE = {"10.0.0.0", "192.168.0.0", "172.16.0.0"}
FORBIDDEN_ARTIFACT_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors", ".distcp", ".parquet"}
SKIP_DIRS = {".git", "__pycache__", ".venv", "venv"}


def files():
    for path in ROOT.rglob("*"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.is_file() or path.is_symlink():
            yield path


def main() -> int:
    errors: list[str] = []
    for path in files():
        rel = path.relative_to(ROOT)
        if path.is_symlink():
            errors.append(f"symlink must be reviewed or materialized: {rel} -> {os.readlink(path)}")
            continue
        size = path.stat().st_size
        if size > FILE_LIMIT:
            errors.append(f"large file ({size} bytes): {rel}")
        if path.suffix.lower() in FORBIDDEN_ARTIFACT_SUFFIXES:
            errors.append(f"training/model artifact committed: {rel}")
        if path == SELF or size > TEXT_LIMIT:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for label, value in FORBIDDEN.items():
            if value in text:
                errors.append(f"{label} in {rel}")
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                errors.append(f"{label} in {rel}")
        for match in PRIVATE_IP.finditer(text):
            if match.group(0) not in ALLOWED_PRIVATE:
                errors.append(f"private host {match.group(0)} in {rel}")

    if errors:
        print("Release audit failed:", file=sys.stderr)
        for error in sorted(set(errors)):
            print(f"  - {error}", file=sys.stderr)
        return 1
    print("Release audit passed: no private paths/hosts, secrets, symlinks, or large artifacts found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
