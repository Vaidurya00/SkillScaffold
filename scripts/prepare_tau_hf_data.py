#!/usr/bin/env python3
"""Build the public SkillScaffold Tau RL dataset in AReaL-compatible form."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


SOURCE_POOL_SHA256 = "9880517777ffbf14ad714b262a9feaeb2416956e5359113f2c5020da2399e600"
SOURCE_POOL_FILE = Path("jsonl/three_domain_joint_family_conservative_round_robin.jsonl")

DB_BY_TASK_SET = {
    "areal_airline_db0": "tau2_airline_db.json",
    "areal_airline_db1": "tau2_airline_new_db_1.json",
    "areal_airline_db2": "tau2_airline_new_db_2.json",
    "areal_airline_db3": "tau2_airline_new_db_3.json",
    "areal_retail_db0": "tau2_retail_new_db_1.json",
    "areal_retail_db1": "tau2_retail_new_db_2.json",
    "areal_retail_db2": "tau2_retail_new_db_3.json",
    "areal_retail_db3": "tau2_retail_new_db_4.json",
    "tau2_telecom_non_base": "tau2_telecom_db.toml",
}

DATABASE_SHA256 = {
    "tau2_airline_db.json": "1af9fea6e03ca7ca15a22bb3fcaf3e351393e3fc9070b6777947da8996f7531b",
    "tau2_airline_new_db_1.json": "b69dbaee93456de6a612bd2ea5174225afe746be4e109c075a9f4db15aaa56c3",
    "tau2_airline_new_db_2.json": "a8a5201ee4f4959bd38787089957029846f79760bf26a9756f996aab166cf1ad",
    "tau2_airline_new_db_3.json": "82255bf2b6c92de4328b401c1a7d8b844d4bdd15746fe00575ee36b2044732b1",
    "tau2_retail_new_db_1.json": "785b3fe40f8f9e9ac033bf3071c3f3bd27da4108ce68532fb4170be5d3a1c07e",
    "tau2_retail_new_db_2.json": "4cfa2f845239052f0e062328dbfb2b41477639079c123e957a473f9e192c0e4a",
    "tau2_retail_new_db_3.json": "2d872f166fa11da78df4dba12905e8dcf2e71d1873ec2a8f0c26c7d2a686e6e7",
    "tau2_retail_new_db_4.json": "00b2095d6351c958b546561f791663c413e83346b1becf59cd55a1c3a49b749b",
    "tau2_telecom_db.toml": "562d647ef9d7df8df91eafd8ee76036e707c8f9e32aaedc4a7fde06975aea2c0",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_empty(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise SystemExit(f"refusing non-empty output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def build_train(source_pool: Path, destination: Path) -> int:
    rows = 0
    with source_pool.open(encoding="utf-8") as reader, destination.open("w", encoding="utf-8") as writer:
        for line_number, line in enumerate(reader, 1):
            source = json.loads(line)
            metadata = source.get("metadata")
            if not isinstance(metadata, dict) or not isinstance(metadata.get("task"), dict):
                raise SystemExit(f"invalid source task at line {line_number}")

            task_set = str(metadata.get("source_task_set"))
            database_name = DB_BY_TASK_SET.get(task_set)
            if database_name is None:
                raise SystemExit(f"unknown source_task_set at line {line_number}: {task_set}")

            # Match the upstream AReaL RL layout: one complete task per line and
            # a repository-relative database path. Tau stores criteria as an
            # object internally; AReaL serializes that object as a JSON string.
            task = dict(metadata["task"])
            criteria = task.get("evaluation_criteria")
            if isinstance(criteria, dict):
                task["evaluation_criteria"] = json.dumps(
                    criteria, ensure_ascii=False, separators=(",", ":")
                )
            elif not isinstance(criteria, str):
                raise SystemExit(f"invalid evaluation_criteria at line {line_number}")
            task["db_path"] = f"tau2_rl_database/{database_name}"
            writer.write(json.dumps(task, ensure_ascii=False, separators=(",", ":")) + "\n")
            rows += 1
    return rows


def copy_databases(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for filename, expected_hash in DATABASE_SHA256.items():
        source_file = source / filename
        if not source_file.is_file():
            raise SystemExit(f"missing source database: {source_file}")
        actual_hash = sha256(source_file)
        if actual_hash != expected_hash:
            raise SystemExit(
                f"database hash mismatch for {source_file}: {actual_hash} != {expected_hash}"
            )
        shutil.copy2(source_file, destination / filename)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rl-source-root", type=Path, required=True)
    parser.add_argument("--database-source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source_pool = args.rl_source_root / SOURCE_POOL_FILE
    if not source_pool.is_file():
        raise SystemExit(f"missing source training pool: {source_pool}")
    actual_source_hash = sha256(source_pool)
    if actual_source_hash != SOURCE_POOL_SHA256:
        raise SystemExit(
            f"source training-pool hash mismatch: {actual_source_hash} != {SOURCE_POOL_SHA256}"
        )

    require_empty(args.output_dir)
    train_file = args.output_dir / "train.jsonl"
    rows = build_train(source_pool, train_file)
    if rows != 900:
        raise SystemExit(f"generated {rows} rows, expected 900")
    copy_databases(args.database_source_root, args.output_dir / "tau2_rl_database")

    train_hash = sha256(train_file)
    repo_root = Path(__file__).resolve().parents[1]
    card = (repo_root / "hf_dataset_cards/tau_rl_README.md").read_text(encoding="utf-8")
    card = card.replace("__TRAIN_ROWS__", str(rows)).replace("__TRAIN_SHA256__", train_hash)
    (args.output_dir / "README.md").write_text(card, encoding="utf-8")

    print(json.dumps({"rows": rows, "train_sha256": train_hash}, indent=2))


if __name__ == "__main__":
    main()
