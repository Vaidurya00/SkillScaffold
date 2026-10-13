#!/usr/bin/env python3
"""Audit the exact Tau RL directory before uploading it to Hugging Face."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


EXPECTED_ROWS = 900
EXPECTED_UNIQUE_TASKS = 838
EXPECTED_TRAIN_SHA256 = "d32ece1d3085afac2212a24913b81f1efcf799195d2c79a861b2ef44d21958ce"
EXPECTED_DOMAINS = Counter({"airline": 300, "retail": 300, "telecom": 300})
EXPECTED_DATABASE_COUNTS = Counter(
    {
        "tau2_rl_database/tau2_airline_db.json": 179,
        "tau2_rl_database/tau2_airline_new_db_1.json": 33,
        "tau2_rl_database/tau2_airline_new_db_2.json": 20,
        "tau2_rl_database/tau2_airline_new_db_3.json": 68,
        "tau2_rl_database/tau2_retail_new_db_1.json": 56,
        "tau2_rl_database/tau2_retail_new_db_2.json": 39,
        "tau2_rl_database/tau2_retail_new_db_3.json": 90,
        "tau2_rl_database/tau2_retail_new_db_4.json": 115,
        "tau2_rl_database/tau2_telecom_db.toml": 300,
    }
)
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
FORBIDDEN_BYTES = (
    b"/wl_" + b"intelligent",
    b"wandb_" + b"v1_",
    b"-----BEGIN " + b"PRIVATE KEY-----",
)
PRIVATE_IP = re.compile(rb"\b10(?:\.\d{1,3}){3}\b")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_bytes(path: Path, errors: list[str]) -> None:
    data = path.read_bytes()
    for marker in FORBIDDEN_BYTES:
        if marker in data:
            errors.append(f"private marker in {path}")
    if PRIVATE_IP.search(data):
        errors.append(f"private host in {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    errors: list[str] = []

    allowed_root = {"README.md", "train.jsonl", "tau2_rl_database"}
    actual_root = {path.name for path in root.iterdir()}
    if actual_root != allowed_root:
        errors.append(f"unexpected root entries: {sorted(actual_root ^ allowed_root)}")

    train_file = root / "train.jsonl"
    domains: Counter[str] = Counter()
    database_counts: Counter[str] = Counter()
    task_ids: set[tuple[str, str]] = set()
    rows = 0
    if not train_file.is_file():
        errors.append(f"missing training file: {train_file}")
    else:
        with train_file.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                rows += 1
                try:
                    task = json.loads(line)
                except json.JSONDecodeError as exc:
                    errors.append(f"invalid JSON at line {line_number}: {exc}")
                    continue
                required = {"id", "description", "user_scenario", "evaluation_criteria", "db_path"}
                if not required.issubset(task):
                    errors.append(f"missing task fields at line {line_number}")
                    continue
                instructions = task.get("user_scenario", {}).get("instructions", {})
                domain = str(instructions.get("domain"))
                domains[domain] += 1
                task_ids.add((domain, str(task["id"])))
                db_path = str(task["db_path"])
                database_counts[db_path] += 1
                try:
                    criteria = json.loads(task["evaluation_criteria"])
                    if not isinstance(criteria, dict):
                        raise TypeError
                except (json.JSONDecodeError, TypeError):
                    errors.append(f"invalid serialized evaluation_criteria at line {line_number}")

        if rows != EXPECTED_ROWS:
            errors.append(f"row count is {rows}, expected {EXPECTED_ROWS}")
        if len(task_ids) != EXPECTED_UNIQUE_TASKS:
            errors.append(f"unique task count is {len(task_ids)}, expected {EXPECTED_UNIQUE_TASKS}")
        if domains != EXPECTED_DOMAINS:
            errors.append(f"domain counts are {dict(domains)}, expected {dict(EXPECTED_DOMAINS)}")
        if database_counts != EXPECTED_DATABASE_COUNTS:
            errors.append("database-reference counts do not match the audited training pool")
        actual_hash = sha256(train_file)
        if actual_hash != EXPECTED_TRAIN_SHA256:
            errors.append(f"train SHA-256 is {actual_hash}, expected {EXPECTED_TRAIN_SHA256}")

    database_dir = root / "tau2_rl_database"
    actual_databases = {path.name for path in database_dir.iterdir()} if database_dir.is_dir() else set()
    if actual_databases != set(DATABASE_SHA256):
        errors.append("database file set does not contain exactly the nine audited files")
    for filename, expected_hash in DATABASE_SHA256.items():
        path = database_dir / filename
        if path.is_file() and sha256(path) != expected_hash:
            errors.append(f"database SHA-256 mismatch: {filename}")

    for path in root.rglob("*"):
        if path.is_file():
            audit_bytes(path, errors)

    if errors:
        print("Tau HF data audit failed:")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("Tau HF data audit passed.")
    print(f"train.jsonl: {EXPECTED_ROWS} rows, {EXPECTED_UNIQUE_TASKS} unique tasks")
    print(f"domains: {dict(EXPECTED_DOMAINS)}")
    print(f"SHA-256: {EXPECTED_TRAIN_SHA256}")
    print("databases: 9/9 hashes verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
