"""Compact, durable artifacts for post-hoc WebShop co-evolution analysis."""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
from typing import Any

from slime.utils.types import Sample


_TEXT_KEYS = {
    "skill",
    "skill_text",
    "raw_skill_output",
    "skill_parse_error",
    "source_task_description",
    "source_task_type",
    "retrieved_query",
    "retrieved_query_key",
    "retrieved_category",
    "scored_query",
    "scored_query_key",
    "scored_category",
    "skill_diff_scoring_source",
    "skill_diff_skip_reason",
    "skill_language_audit_judgment",
    "skill_language_audit_reason",
    "executor_skill_mode",
    "selected_skill",
    "category",
    "task_description",
    "downstream_task_description",
    "downstream_task_type",
}

_SCALAR_PREFIXES = (
    "skill_diff_",
    "generator_",
    "raw_",
    "normalized_",
    "reward_",
    "retrieval_",
    "retrieved_",
    "trajectory_buffer_",
    "executor_",
    "selected_",
    "source_episode_",
    "format_",
    "skill_language_audit_",
    "downstream_",
)


def _jsonable_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _sample_role(sample: Sample) -> str:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    train_metadata = sample.train_metadata if isinstance(sample.train_metadata, dict) else {}
    return str(train_metadata.get("sample_role") or metadata.get("sample_role") or "")


def _compact_sample(sample: Sample) -> dict[str, Any]:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    selected = {
        key: value
        for key, value in metadata.items()
        if _jsonable_scalar(value)
        and (key in _TEXT_KEYS or key.startswith(_SCALAR_PREFIXES))
    }
    prompt = str(sample.prompt or "")
    response = str(sample.response or "")
    compact = {
        "role": _sample_role(sample),
        "index": sample.index,
        "group_index": sample.group_index,
        "goal_idx": (sample.label or {}).get("goal_idx") if isinstance(sample.label, dict) else None,
        "status": str(sample.status),
        "reward": sample.reward,
        "response_length": sample.response_length,
        "effective_response_length": sample.effective_response_length,
        "prompt_has_reference_skill": "Reference Skill" in prompt,
        "prompt_has_closed_empty_think": "<think>\n\n</think>" in prompt,
        "response_begins_think": response.lstrip().startswith("<think>"),
        "response_begins_json": response.lstrip().startswith("{"),
        "response_prefix": response[:256],
        "metadata": selected,
    }
    candidate_rows = metadata.get("feature3_skill_candidates")
    executor_group_position = metadata.get("executor_group_position")
    if executor_group_position == 0 and isinstance(candidate_rows, list):
        compact["feature3_skill_candidates"] = [
            {
                str(key): value
                for key, value in row.items()
                if _jsonable_scalar(value)
            }
            for row in candidate_rows
            if isinstance(row, dict)
        ]
    return compact


def persist_rollout_dynamics(
    args,
    *,
    rollout_id: int,
    samples: list[Sample],
    metrics: dict[str, Any],
    buffer_snapshot: dict | None = None,
) -> None:
    root_text = os.environ.get("DYNAMICS_DIR", "").strip()
    if not root_text:
        save_root = str(getattr(args, "save", "") or "").strip()
        if not save_root:
            return
        root_text = str(Path(save_root) / "dynamics")
    root = Path(root_text)
    sample_dir = root / "rollout_samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "rollout_id": int(rollout_id),
        "metrics": metrics,
        "samples": [_compact_sample(sample) for sample in samples],
    }
    sample_path = sample_dir / f"rollout_{rollout_id:06d}.json.gz"
    with gzip.open(sample_path, "wt", encoding="utf-8") as output:
        json.dump(payload, output, ensure_ascii=False, default=str)

    interval = max(1, int(os.environ.get("DYNAMICS_BUFFER_SNAPSHOT_INTERVAL", "25")))
    if buffer_snapshot is None or (rollout_id % interval != 0 and rollout_id > 2):
        return
    compact_records = []
    for category, records in buffer_snapshot.items():
        for record in records.values():
            compact_records.append(
                {
                    "category": category,
                    "goal_idx": record.get("goal_idx"),
                    "query_key": record.get("query_key"),
                    "query": record.get("query"),
                    "retrieval_eligible": bool(record.get("retrieval_eligible")),
                    "selection_reason": record.get("selection_reason"),
                    "buffer_insert_step": record.get("buffer_insert_step"),
                    "update_sequence": record.get("update_sequence"),
                    "won": record.get("won"),
                }
            )
    snapshot_dir = root / "buffer_snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(snapshot_dir / f"buffer_{rollout_id:06d}.json.gz", "wt", encoding="utf-8") as output:
        json.dump({"rollout_id": rollout_id, "records": compact_records}, output, ensure_ascii=False)
