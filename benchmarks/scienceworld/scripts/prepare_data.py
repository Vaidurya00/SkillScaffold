#!/usr/bin/env python3
"""Materialize the deterministic ScienceWorld train and evaluation manifests."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SPLITS = {
    "train": (
        ("train", "power-component", range(0, 10)),
        ("train", "test-conductivity", range(0, 450)),
    ),
    "dev": (
        ("dev", "power-component-renewable-vs-nonrenewable-energy", range(10, 15)),
        ("dev", "test-conductivity-of-unknown-substances", range(300, 450)),
    ),
    "test_id_dev230": (
        ("dev", "power-component", range(10, 15)),
        ("dev", "test-conductivity", range(450, 675)),
    ),
    "test_ood155": (
        ("test", "power-component-renewable-vs-nonrenewable-energy", range(15, 20)),
        ("test", "test-conductivity-of-unknown-substances", range(450, 600)),
    ),
}
EXPECTED_ROWS = {
    "train": 460,
    "dev": 155,
    "test_id_dev230": 230,
    "test_ood155": 155,
}


def build(specs) -> pd.DataFrame:
    rows = []
    for split, task_name, variations in specs:
        for variation_idx in variations:
            record_id = f"{split}:{task_name}:{variation_idx}"
            metadata = {
                "record_id": record_id,
                "task_name": task_name,
                "variation_idx": int(variation_idx),
                "split": split,
                "max_steps": 30,
                "task_description": "",
                "category": task_name,
                "task_type": task_name,
            }
            rows.append(
                {
                    "id": record_id,
                    "question": record_id,
                    "prompt": [{"role": "user", "content": record_id}],
                    "metadata": metadata,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "train": args.output_dir / "train.parquet",
        "dev": args.output_dir / "dev.parquet",
        "test_id_dev230": args.output_dir / "test_id_dev230.parquet",
        "test_ood155": args.output_dir / "test_ood155.parquet",
    }
    keys: dict[str, set[tuple[str, int]]] = {}
    for name, specs in SPLITS.items():
        frame = build(specs)
        if len(frame) != EXPECTED_ROWS[name]:
            raise RuntimeError(f"{name}: expected {EXPECTED_ROWS[name]}, got {len(frame)}")
        keys[name] = {
            (item["task_name"], int(item["variation_idx"]))
            for item in frame["metadata"]
        }
        frame.to_parquet(outputs[name], index=False)
        print(f"{name}={len(frame)} output={outputs[name]}")
    if keys["train"] & keys["dev"]:
        raise RuntimeError("ScienceWorld training and development tasks overlap")
    if keys["test_id_dev230"] & keys["test_ood155"]:
        raise RuntimeError("ScienceWorld ID and OOD evaluation tasks overlap")


if __name__ == "__main__":
    main()
