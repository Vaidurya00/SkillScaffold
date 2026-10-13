#!/usr/bin/env python3
"""Build a round-robin, oversampled prompt file with equal domain counts."""

from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
from collections import Counter, defaultdict
from pathlib import Path


def _domain(row: dict) -> str:
    metadata = row.get("metadata") or {}
    return str(metadata.get("base_domain") or metadata.get("domain") or "unknown")


def build_balanced_rows(rows: list[dict], domains: list[str], seed: int) -> list[dict]:
    by_domain: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_domain[_domain(row)].append(row)
    missing = [domain for domain in domains if not by_domain[domain]]
    if missing:
        raise ValueError(f"No prompts found for domains: {', '.join(missing)}")
    unexpected = sorted(set(by_domain) - set(domains))
    if unexpected:
        raise ValueError("Input contains domains not included in the balance set: " + ", ".join(unexpected))
    target = max(len(by_domain[domain]) for domain in domains)
    expanded: dict[str, list[dict]] = {}
    for index, domain in enumerate(domains):
        source = list(by_domain[domain])
        random.Random(seed + index).shuffle(source)
        expanded[domain] = [source[i % len(source)] for i in range(target)]
    return [expanded[domain][i] for i in range(target) for domain in domains]


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--domains", nargs="+", default=["airline", "retail", "telecom"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    source_rows = _read_jsonl(args.input)
    balanced_rows = build_balanced_rows(source_rows, args.domains, args.seed)
    _write_jsonl_atomic(args.output, balanced_rows)
    source_counts = Counter(_domain(row) for row in source_rows)
    balanced_counts = Counter(_domain(row) for row in balanced_rows)
    print(f"Domain-balanced prompt data: {args.output}")
    print(f"  source: {dict(sorted(source_counts.items()))}")
    print(f"  balanced: {dict(sorted(balanced_counts.items()))}")


if __name__ == "__main__":
    main()
