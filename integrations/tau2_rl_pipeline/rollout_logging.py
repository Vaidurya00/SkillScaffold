"""Human-readable trajectory snapshots for Tau2 RL rollouts."""

from __future__ import annotations

import json
import os
from enum import Enum
from pathlib import Path
from typing import Any


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump())
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    return str(value)


def _first_user_message(messages: Any) -> Any:
    if not isinstance(messages, list):
        return None
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "user":
            return _jsonable(message.get("content"))
    return None


def _task_instructions(metadata: dict[str, Any]) -> Any:
    task = metadata.get("task") or {}
    scenario = task.get("user_scenario") or {}
    instructions = scenario.get("instructions") or {}
    return _jsonable(instructions.get("task_instructions"))


def _hard_reward(sample: Any) -> float:
    if getattr(sample, "remove_sample", False):
        return 0.0
    try:
        return 1.0 if float(getattr(sample, "reward", 0.0) or 0.0) >= 1.0 else 0.0
    except (TypeError, ValueError):
        return 0.0


def _sample_sort_key(sample: Any) -> tuple[int, int]:
    metadata = getattr(sample, "metadata", None) or {}
    reason = metadata.get("termination_reason")
    severe = int(reason not in {"format_error", "invalid_action", "repeated_action", "context_limit"})
    return severe, int(getattr(sample, "index", 0) or 0)


def _base_domain(sample: Any) -> str:
    metadata = getattr(sample, "metadata", None) or {}
    return str(metadata.get("base_domain") or metadata.get("domain") or "unknown")


def _select_samples_by_domain(
    samples: list[Any],
    *,
    failures_per_domain: int,
    successes_per_domain: int,
) -> tuple[list[Any], dict[str, dict[str, int]]]:
    """Select fixed failure/success controls independently for each domain."""
    if failures_per_domain < 0 or successes_per_domain < 0:
        raise ValueError("Per-domain trajectory dump quotas must be >= 0")

    grouped: dict[str, list[Any]] = {}
    for sample in samples:
        grouped.setdefault(_base_domain(sample), []).append(sample)

    preferred_order = {"airline": 0, "retail": 1, "telecom": 2}
    domains = sorted(grouped, key=lambda domain: (preferred_order.get(domain, 3), domain))
    selected: list[Any] = []
    counts: dict[str, dict[str, int]] = {}
    for domain in domains:
        domain_samples = grouped[domain]
        failures = sorted(
            (sample for sample in domain_samples if _hard_reward(sample) == 0.0),
            key=_sample_sort_key,
        )
        successes = sorted(
            (sample for sample in domain_samples if _hard_reward(sample) == 1.0),
            key=_sample_sort_key,
        )
        selected_failures = failures[:failures_per_domain]
        selected_successes = successes[:successes_per_domain]
        selected.extend(selected_failures)
        selected.extend(selected_successes)
        counts[domain] = {
            "available_failure": len(failures),
            "available_success": len(successes),
            "selected_failure": len(selected_failures),
            "selected_success": len(selected_successes),
        }
    return selected, counts


def _sample_record(sample: Any, *, include_tokens: bool) -> dict[str, Any]:
    metadata = getattr(sample, "metadata", None) or {}
    prompt = getattr(sample, "prompt", None)
    loss_mask = list(getattr(sample, "loss_mask", None) or [])
    tokens = list(getattr(sample, "tokens", None) or [])
    status = getattr(sample, "status", None)

    record = {
        "sample_index": getattr(sample, "index", None),
        "group_index": getattr(sample, "group_index", None),
        "task_id": metadata.get("task_id"),
        "domain": metadata.get("domain"),
        "base_domain": metadata.get("base_domain"),
        "query": {
            "task_instructions": _task_instructions(metadata),
            "initial_user_message": metadata.get("initial_user_message")
            or _first_user_message(prompt),
        },
        "environment_reward": getattr(sample, "reward", None),
        "hard_reward": _hard_reward(sample),
        "status": status.value if isinstance(status, Enum) else status,
        "remove_sample": bool(getattr(sample, "remove_sample", False)),
        "termination_reason": metadata.get("termination_reason"),
        "tau2_termination_reason": metadata.get("tau2_termination_reason"),
        "format_error": metadata.get("format_error"),
        "num_env_steps": metadata.get("num_env_steps"),
        "tool_sequence": metadata.get("tool_sequence"),
        "response": getattr(sample, "response", None),
        "prompt": _jsonable(prompt),
        "messages": _jsonable(prompt) if isinstance(prompt, list) else None,
        "reward_info": _jsonable(metadata.get("reward_info")),
        "mask": {
            "total_tokens": len(tokens),
            "response_length": int(getattr(sample, "response_length", 0) or 0),
            "trainable_tokens": sum(loss_mask),
            "mode": metadata.get("loss_mask_mode"),
            "alignment_error": metadata.get("mask_alignment_error"),
            "token_mismatch": metadata.get("token_mismatch", False),
        },
        "timing": {
            "env_wait_seconds": metadata.get("tau2_env_wait_seconds"),
            "agent_generation_seconds": metadata.get("agent_generation_seconds"),
        },
        "metadata": _jsonable(metadata),
    }
    if include_tokens:
        record["token_ids"] = tokens
        record["loss_mask"] = loss_mask
    return record


def dump_rollout_trajectories(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    """Write sampled trajectories and continue slime's default rollout logging."""
    output_dir = os.environ.get("TAU2_TRAJECTORY_DUMP_DIR")
    interval = int(os.environ.get("TAU2_TRAJECTORY_DUMP_INTERVAL", "10"))
    if not output_dir or interval <= 0 or int(rollout_id) % interval != 0:
        return False

    failures_per_domain = int(os.environ.get("TAU2_TRAJECTORY_DUMP_FAILURES_PER_DOMAIN", "4"))
    successes_per_domain = int(os.environ.get("TAU2_TRAJECTORY_DUMP_SUCCESSES_PER_DOMAIN", "2"))
    include_tokens = os.environ.get("TAU2_TRAJECTORY_DUMP_INCLUDE_TOKENS", "0") == "1"
    selected, selection_counts = _select_samples_by_domain(
        list(samples),
        failures_per_domain=failures_per_domain,
        successes_per_domain=successes_per_domain,
    )
    rewards = [_hard_reward(sample) for sample in samples]
    payload = {
        "schema_version": 2,
        "rollout_id": int(rollout_id),
        "rollout_time_seconds": float(rollout_time),
        "num_samples": len(samples),
        "num_success": int(sum(rewards)),
        "num_failure": len(rewards) - int(sum(rewards)),
        "rollout_extra_metrics": _jsonable(rollout_extra_metrics),
        "sample_selection": "domain_stratified_failure_success_controls",
        "sample_selection_config": {
            "failures_per_domain": failures_per_domain,
            "successes_per_domain": successes_per_domain,
        },
        "sample_selection_counts": selection_counts,
        "samples": [_sample_record(sample, include_tokens=include_tokens) for sample in selected],
    }

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"rollout_{int(rollout_id):06d}.json"
    temporary = directory / f".{path.name}.{os.getpid()}.tmp"
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)
    return False
