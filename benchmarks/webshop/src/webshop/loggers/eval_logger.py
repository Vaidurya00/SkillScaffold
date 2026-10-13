import logging
from typing import Any

from slime.utils import logging_utils
from slime.utils.metric_utils import compute_rollout_step, compute_statistics, dict_add_prefix
from slime.utils.types import Sample

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_task_type(sample: Sample) -> str:
    label = getattr(sample, "label", None)
    if isinstance(label, dict):
        return str(label.get("task_type", "unknown"))
    return "unknown"


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _meta_float(sample: Sample, key: str, default: float = 0.0) -> float:
    m = sample.metadata
    if isinstance(m, dict):
        v = m.get(key)
        if v is not None:
            try:
                return float(v)
            except Exception:
                pass
    return default


def _won(sample: Sample) -> bool:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    return bool(metadata.get("won", False))


# ---------------------------------------------------------------------------
# Main log function
# ---------------------------------------------------------------------------


def log_eval_by_task_type(
    rollout_id: int,
    args,
    data: dict[str, dict[str, Any]],
    extra_metrics: dict[str, Any] | None,
) -> bool:
    """Custom eval log function with per-task_type win rate for WebShop."""
    log_dict: dict[str, Any] = extra_metrics.copy() if extra_metrics else {}

    step = compute_rollout_step(args, rollout_id)
    log_dict["eval/step"] = step

    for dataset_name, dataset_data in data.items():
        samples: list[Sample] = dataset_data.get("samples") or []
        if not samples:
            continue

        prefix = f"eval/{dataset_name}"

        _log_baseline_eval(log_dict, prefix, samples, dataset_data)

    logger.info(f"eval {rollout_id}: {log_dict}")
    logging_utils.log(args, log_dict, step_key="eval/step")
    return True


# ---------------------------------------------------------------------------
# Baseline eval
# ---------------------------------------------------------------------------


def _log_baseline_eval(
    log_dict: dict,
    prefix: str,
    samples: list[Sample],
    dataset_data: dict,
) -> None:
    won_flags = [1.0 if _won(sample) else 0.0 for sample in samples]
    num_steps = [_meta_float(s, "num_steps") for s in samples]

    log_dict[f"{prefix}/win_rate"] = _mean(won_flags)
    log_dict[f"{prefix}/win_rate_no_skill"] = _mean(won_flags)
    log_dict[f"{prefix}/count"] = len(samples)
    log_dict |= dict_add_prefix(compute_statistics(num_steps), f"{prefix}/num_steps/")

    if "truncated" in dataset_data:
        truncated = dataset_data["truncated"]
        log_dict[f"{prefix}/truncated_ratio"] = sum(truncated) / len(truncated)

    task_type_groups: dict[str, list[int]] = {}
    for i, s in enumerate(samples):
        tt = _get_task_type(s)
        task_type_groups.setdefault(tt, []).append(i)

    for tt, indices in sorted(task_type_groups.items()):
        tt_won = [won_flags[i] for i in indices]
        tt_steps = [num_steps[i] for i in indices]
        log_dict[f"{prefix}/{tt}/win_rate"] = _mean(tt_won)
        log_dict[f"{prefix}/{tt}/win_rate_no_skill"] = _mean(tt_won)
        log_dict[f"{prefix}/{tt}/count"] = len(indices)
        log_dict |= dict_add_prefix(compute_statistics(tt_steps), f"{prefix}/{tt}/num_steps/")
