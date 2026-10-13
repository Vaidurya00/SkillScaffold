#!/usr/bin/env python3
"""Download public Tau RL data and build the local SkillScaffold runtime view."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download


TASK_SET_BY_DATABASE = {
    "tau2_airline_db.json": ("areal_airline_db0", "airline"),
    "tau2_airline_new_db_1.json": ("areal_airline_db1", "airline"),
    "tau2_airline_new_db_2.json": ("areal_airline_db2", "airline"),
    "tau2_airline_new_db_3.json": ("areal_airline_db3", "airline"),
    "tau2_retail_new_db_1.json": ("areal_retail_db0", "retail"),
    "tau2_retail_new_db_2.json": ("areal_retail_db1", "retail"),
    "tau2_retail_new_db_3.json": ("areal_retail_db2", "retail"),
    "tau2_retail_new_db_4.json": ("areal_retail_db3", "retail"),
    "tau2_telecom_db.toml": (
        "areal_telecom_nonbase_qwen3_8b_rc1_train300_eval50",
        "telecom",
    ),
}


def prepare_runtime(public_root: Path, runtime_root: Path) -> Path:
    if runtime_root.exists() and any(runtime_root.iterdir()):
        raise SystemExit(f"refusing non-empty runtime directory: {runtime_root}")
    runtime_root.mkdir(parents=True, exist_ok=True)
    prompt_dir = runtime_root / "jsonl"
    prompt_dir.mkdir()

    tasks_by_set: dict[str, dict[str, dict]] = {}
    prompt_sequence: list[tuple[str, str, str, dict]] = []
    with (public_root / "train.jsonl").open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            public_task = json.loads(line)
            database_name = Path(str(public_task["db_path"])).name
            if database_name not in TASK_SET_BY_DATABASE:
                raise SystemExit(f"unknown database at line {line_number}: {database_name}")
            task_set, base_domain = TASK_SET_BY_DATABASE[database_name]

            task = dict(public_task)
            task.pop("db_path", None)
            criteria = task.get("evaluation_criteria")
            if isinstance(criteria, str):
                task["evaluation_criteria"] = json.loads(criteria)
            task_id = str(task["id"])

            registry = tasks_by_set.setdefault(task_set, {})
            if task_id in registry and registry[task_id] != task:
                raise SystemExit(f"conflicting duplicate task: {base_domain}/{task_id}")
            registry.setdefault(task_id, task)

            prompt_sequence.append((task_set, base_domain, task_id, task))

    manifest: dict[str, dict] = {}
    for database_name, (task_set, base_domain) in TASK_SET_BY_DATABASE.items():
        database_file = (public_root / "tau2_rl_database" / database_name).resolve()
        if not database_file.is_file():
            raise SystemExit(f"missing database: {database_file}")
        task_map = tasks_by_set.get(task_set, {})
        if not task_map:
            raise SystemExit(f"no tasks found for runtime task set: {task_set}")
        task_dir = runtime_root / task_set
        task_dir.mkdir()
        task_ids = list(task_map)
        (task_dir / "tasks.json").write_text(
            json.dumps(list(task_map.values()), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (task_dir / "split_tasks.json").write_text(
            json.dumps({"train": task_ids}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        manifest[task_set] = {
            "domain": base_domain,
            "db_file": str(database_file),
            "num_tasks": len(task_ids),
        }

    task_index = {
        task_set: {task_id: index for index, task_id in enumerate(task_map)}
        for task_set, task_map in tasks_by_set.items()
    }
    prompt_rows = [
        {
            "text": [{"role": "user", "content": "task"}],
            "metadata": {
                "domain": task_set,
                "base_domain": base_domain,
                "split": "train",
                "task_id": task_id,
                "task_index": task_index[task_set][task_id],
                "task": task,
                "source_task_set": task_set,
            },
        }
        for task_set, base_domain, task_id, task in prompt_sequence
    ]
    prompt_file = prompt_dir / "three_domain_joint_family_conservative_round_robin.jsonl"
    with prompt_file.open("w", encoding="utf-8") as handle:
        for row in prompt_rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    (runtime_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return prompt_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="HF dataset repo containing Tau RL data")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output", type=Path, default=Path("data/tau/rl"))
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    downloaded = Path(
        snapshot_download(
            repo_id=args.repo,
            repo_type="dataset",
            revision=args.revision,
            local_dir=args.output,
        )
    ).resolve()
    prompt_file = prepare_runtime(downloaded, downloaded / "runtime")
    print(f"Tau RL data: {downloaded}")
    print(f"TAU2_TRAIN_POOL_ROOT={prompt_file.parents[1]}")
    print(f"TAU2_TRAIN_TASKS_JSONL={prompt_file}")


if __name__ == "__main__":
    main()
