#!/usr/bin/env python3
"""Build the exact ALFWorld train split and all evaluation views used by training."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import pandas as pd


EXPECTED = {
    "pick_and_place_simple": 790,
    "pick_clean_then_place_in_recep": 650,
    "pick_cool_then_place_in_recep": 533,
    "pick_heat_then_place_in_recep": 459,
    "pick_two_obj_and_place": 813,
    "look_at_obj_in_light": 308,
}
EXPECTED_SOURCE_SHA256 = "9056ef9f4df1e2c3355c3f96313d5daef96fffb0b8170d085b959f84662c8279"
TRAIN_TYPES = {
    "pick_and_place_simple",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_clean_then_place_in_recep",
}
HELDOUT_TYPES = {"pick_two_obj_and_place", "look_at_obj_in_light"}
EXPECTED_ROWS = {
    "train": 3553,
    "train_seen": 2432,
    "train_unseen": 1121,
    "valid_seen": 140,
    "valid_seen_id_types": 103,
    "valid_seen_heldout_types": 37,
    "valid_unseen": 134,
    "valid_unseen_id_types": 99,
    "valid_unseen_heldout_types": 35,
    "valid_train": 200,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        help="Optional prebuilt parquet. If omitted, rebuild from the official task tree.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="ALFWorld json_2.1.1 root containing train/.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _converter():
    module_path = Path(__file__).resolve().parents[1] / "src/data/alfworld/extract.py"
    spec = importlib.util.spec_from_file_location("skillscaffold_alfworld_extract", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load ALFWorld converter: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _extract_from_task_tree(
    data_root: Path, output_dir: Path, *, include_train: bool
) -> None:
    module = _converter()
    splits = ["valid_seen", "valid_unseen", "valid_train"]
    if include_train:
        splits.insert(0, "train")
    for split in splits:
        module.convert_one_split(str(data_root), split, str(output_dir))


def _task_type(metadata: object) -> str:
    return str(metadata.get("task_type") or "") if isinstance(metadata, dict) else ""


def _write_views(output_dir: Path) -> dict[str, pd.DataFrame]:
    train = pd.read_parquet(output_dir / "train.parquet")
    valid_seen = pd.read_parquet(output_dir / "valid_seen.parquet")
    valid_unseen = pd.read_parquet(output_dir / "valid_unseen.parquet")
    frames = {
        "train": train,
        "train_seen": train[train["metadata"].map(_task_type).isin(TRAIN_TYPES)].copy(),
        "train_unseen": train[train["metadata"].map(_task_type).isin(HELDOUT_TYPES)].copy(),
        "valid_seen": valid_seen,
        "valid_seen_id_types": valid_seen[
            valid_seen["metadata"].map(_task_type).isin(TRAIN_TYPES)
        ].copy(),
        "valid_seen_heldout_types": valid_seen[
            valid_seen["metadata"].map(_task_type).isin(HELDOUT_TYPES)
        ].copy(),
        "valid_unseen": valid_unseen,
        "valid_unseen_id_types": valid_unseen[
            valid_unseen["metadata"].map(_task_type).isin(TRAIN_TYPES)
        ].copy(),
        "valid_unseen_heldout_types": valid_unseen[
            valid_unseen["metadata"].map(_task_type).isin(HELDOUT_TYPES)
        ].copy(),
        "valid_train": pd.read_parquet(output_dir / "valid_train.parquet"),
    }
    derived = {
        "train_seen",
        "train_unseen",
        "valid_seen_id_types",
        "valid_seen_heldout_types",
        "valid_unseen_id_types",
        "valid_unseen_heldout_types",
    }
    for name in derived:
        frames[name].to_parquet(output_dir / f"{name}.parquet", index=False)
    return frames


def _validate_train(frame: pd.DataFrame, data_root: Path) -> list[Path]:
    required = {"id", "question", "prompt", "metadata"}
    if set(frame.columns) != required:
        raise ValueError(f"unexpected columns: {list(frame.columns)}")
    if len(frame) != 3553:
        raise ValueError(f"expected 3,553 rows, found {len(frame)}")
    rel_dirs = frame["metadata"].map(lambda item: item["rel_dir"])
    if rel_dirs.nunique() != 3553:
        raise ValueError("ALFWorld rel_dir values are not unique")
    if rel_dirs.tolist() != sorted(rel_dirs.tolist()):
        raise ValueError("ALFWorld rows must use deterministic sorted task-directory order")
    counts = Counter(item["task_type"] for item in frame["metadata"])
    if dict(counts) != EXPECTED:
        raise ValueError(f"task-type counts differ: {dict(counts)} != {EXPECTED}")
    paths = [data_root / "train" / rel_dir / "game.tw-pddl" for rel_dir in rel_dirs]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"missing {len(missing)} game files; first entries:\n" + "\n".join(missing[:10])
        )
    return paths


def _validate_views(frames: dict[str, pd.DataFrame], data_root: Path) -> None:
    actual_rows = {name: len(frame) for name, frame in frames.items()}
    if actual_rows != EXPECTED_ROWS:
        raise ValueError(f"ALFWorld split sizes differ: {actual_rows} != {EXPECTED_ROWS}")
    for name, frame in frames.items():
        if frame["id"].astype(str).nunique() != len(frame):
            raise ValueError(f"duplicate IDs in {name}")
        for metadata in frame["metadata"]:
            game = (
                data_root
                / str(metadata["split"])
                / str(metadata["rel_dir"])
                / "game.tw-pddl"
            )
            if not game.is_file():
                raise FileNotFoundError(f"missing ALFWorld game file for {name}: {game}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _extract_from_task_tree(
        args.data_root, args.output_dir, include_train=args.source is None
    )
    if args.source:
        digest = hashlib.sha256(args.source.read_bytes()).hexdigest()
        if digest != EXPECTED_SOURCE_SHA256:
            raise ValueError(f"unexpected source SHA-256: {digest}")
        shutil.copyfile(args.source, args.output_dir / "train.parquet")
        frame = pd.read_parquet(args.source)
    else:
        frame = pd.read_parquet(args.output_dir / "train.parquet")

    paths = _validate_train(frame, args.data_root)
    frames = _write_views(args.output_dir)
    _validate_views(frames, args.data_root)
    allowlist = args.output_dir / "train_full3553_gamefiles.txt"
    allowlist.write_text("".join(f"{path}\n" for path in paths), encoding="utf-8")
    manifest = {
        "rows": {name: len(view) for name, view in frames.items()},
        "train_task_types": sorted(TRAIN_TYPES),
        "heldout_task_types": sorted(HELDOUT_TYPES),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"rows={len(frame)}")
    print(f"train_parquet={args.output_dir / 'train.parquet'}")
    print(f"views={json.dumps(manifest['rows'], sort_keys=True)}")
    print(f"game_allowlist={allowlist}")


if __name__ == "__main__":
    main()
