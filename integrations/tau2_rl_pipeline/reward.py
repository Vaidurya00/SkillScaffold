"""Binary task reward for tau2-bench rollouts.

Entry point: `--custom-reward-post-process-path reward.tau2_reward_post_process`

The environment's final task reward is thresholded to exactly 0 or 1. Format,
termination reason, partial checks, and no-progress tiers never override or
shape that value. Infrastructure-aborted samples remain masked out. GRPO may
still center the binary rewards within each prompt group when reward
normalization is enabled; that is advantage construction, not reward shaping.
"""

from __future__ import annotations

import os
import threading
from collections import defaultdict
from functools import lru_cache
from typing import Any

import torch

from slime.utils.types import Sample
from tau2_rl_pipeline.termination import TAU2_TERMINATION_REASONS

_SAFETY_LOCK = threading.Lock()
_SAFETY_LAST_ROLLOUT: object | None = None
_SAFETY_ROLLOUT_COUNT = 0
_SAFETY_BAD_STREAK = 0
_REWARD_METRIC_DEFINED = False
BEHAVIOR_FAILURE_REASONS = {
    "repeated_action",
    "max_steps",
    "env_truncated",
    "context_limit",
}
_AIRLINE_WRITE_TOOL_PREFIXES = (
    "update_",
    "cancel_",
    "book_",
    "create_",
    "issue_",
)
_AIRLINE_WRITE_TOOL_NAMES = {"send_certificate"}
TASK_ACTION_CATEGORIES = ("write", "read_only", "no_op")
_KNOWN_TOOL_TYPES = {
    ("airline", "assistant"): {
        **dict.fromkeys(
            (
                "get_flight_status",
                "get_reservation_details",
                "get_user_details",
                "list_all_airports",
                "search_direct_flight",
                "search_onestop_flight",
            ),
            "read",
        ),
        **dict.fromkeys(
            (
                "book_reservation",
                "cancel_reservation",
                "send_certificate",
                "update_reservation_baggages",
                "update_reservation_flights",
                "update_reservation_passengers",
            ),
            "write",
        ),
        "calculate": "generic",
        "transfer_to_human_agents": "generic",
    },
    ("retail", "assistant"): {
        **dict.fromkeys(
            (
                "find_user_id_by_email",
                "find_user_id_by_name_zip",
                "get_item_details",
                "get_order_details",
                "get_product_details",
                "get_user_details",
                "list_all_product_types",
            ),
            "read",
        ),
        **dict.fromkeys(
            (
                "cancel_pending_order",
                "exchange_delivered_order_items",
                "modify_pending_order_address",
                "modify_pending_order_items",
                "modify_pending_order_payment",
                "modify_user_address",
                "return_delivered_order_items",
            ),
            "write",
        ),
        "calculate": "generic",
        "transfer_to_human_agents": "generic",
    },
    ("telecom", "assistant"): {
        **dict.fromkeys(
            (
                "get_bills_for_customer",
                "get_customer_by_id",
                "get_customer_by_name",
                "get_customer_by_phone",
                "get_data_usage",
                "get_details_by_id",
            ),
            "read",
        ),
        **dict.fromkeys(
            (
                "disable_roaming",
                "enable_roaming",
                "refuel_data",
                "resume_line",
                "send_payment_request",
                "suspend_line",
            ),
            "write",
        ),
        "transfer_to_human_agents": "generic",
    },
    ("telecom", "user"): {
        **dict.fromkeys(
            (
                "can_send_mms",
                "check_apn_settings",
                "check_app_permissions",
                "check_app_status",
                "check_data_restriction_status",
                "check_installed_apps",
                "check_network_mode_preference",
                "check_network_status",
                "check_payment_request",
                "check_sim_status",
                "check_status_bar",
                "check_vpn_status",
                "check_wifi_calling_status",
                "check_wifi_status",
                "run_speed_test",
            ),
            "read",
        ),
        **dict.fromkeys(
            (
                "connect_vpn",
                "disconnect_vpn",
                "grant_app_permission",
                "make_payment",
                "reboot_device",
                "reseat_sim_card",
                "reset_apn_settings",
                "set_apn_settings",
                "set_network_mode_preference",
                "toggle_airplane_mode",
                "toggle_data",
                "toggle_data_saver_mode",
                "toggle_roaming",
                "toggle_wifi",
                "toggle_wifi_calling",
            ),
            "write",
        ),
    },
}


def _flatten(samples: list[Sample] | list[list[Sample]]) -> list[Sample]:
    if not samples:
        return []
    if isinstance(samples[0], list):
        return [s for group in samples for s in group]
    return list(samples)


def _no_write_reward_violation(metadata: dict[str, Any]) -> bool:
    """Whether a marked synthetic no-write task attempted a mutation.

    The prompt dataset owns the opt-in marker. Pools without these tasks do
    not carry ``no_write_reward_gate`` and retain their original reward path.
    Prefix matching also catches hallucinated mutation tools such as
    ``update_reservation_insurance``.
    """
    if metadata.get("no_write_reward_gate") is not True:
        return False
    sequence = metadata.get("tool_sequence") or []
    if not isinstance(sequence, list):
        return False
    return any(
        str(name) in _AIRLINE_WRITE_TOOL_NAMES
        or str(name).startswith(_AIRLINE_WRITE_TOOL_PREFIXES)
        for name in sequence
    )


def _domain_env_step_metrics(flat: list[Sample]) -> dict[str, float]:
    domain_steps: dict[str, list[float]] = defaultdict(list)
    for sample in flat:
        if sample.remove_sample:
            continue
        meta = sample.metadata or {}
        num_env_steps = meta.get("num_env_steps")
        if not isinstance(num_env_steps, (int, float)) or isinstance(num_env_steps, bool):
            continue
        domain = str(meta.get("base_domain") or meta.get("domain") or "unknown")
        domain_steps[domain].append(float(num_env_steps))

    metrics: dict[str, float] = {}
    for domain, values in domain_steps.items():
        prefix = f"reward/env_steps/{domain}"
        metrics[f"{prefix}/mean"] = sum(values) / len(values)
        metrics[f"{prefix}/max"] = max(values)
        metrics[f"{prefix}/min"] = min(values)
    return metrics


@lru_cache(maxsize=None)
def _tool_type(domain: str, requestor: str, tool_name: str) -> str | None:
    """Resolve Tau2's conceptual tool type without constructing an environment.

    The checked-in map keeps logging usable in lightweight metric workers where
    importing the full Tau2 runner dependency tree is unavailable. When Tau2
    can be imported, its decorator metadata remains authoritative.
    """
    fallback = _KNOWN_TOOL_TYPES.get((domain, requestor), {}).get(tool_name)
    try:
        if domain == "airline" and requestor == "assistant":
            from tau2.domains.airline.tools import AirlineTools

            toolkit_class = AirlineTools
        elif domain == "retail" and requestor == "assistant":
            from tau2.domains.retail.tools import RetailTools

            toolkit_class = RetailTools
        elif domain == "telecom" and requestor == "assistant":
            from tau2.domains.telecom.tools import TelecomTools

            toolkit_class = TelecomTools
        elif domain == "telecom" and requestor == "user":
            from tau2.domains.telecom.user_tools import TelecomUserTools

            toolkit_class = TelecomUserTools
        else:
            return fallback
    except ImportError:
        return fallback

    method = getattr(toolkit_class, tool_name, None)
    value = getattr(method, "__tool_type__", None)
    if value is None:
        return fallback
    return str(getattr(value, "value", value)).lower()


def _task_action_category(metadata: dict[str, Any]) -> str | None:
    """Classify a task from its expected actions.

    The prompt JSONL does not carry a task-type label. An empty expected-action
    list is no-op; any expected WRITE action (including telecom user tools)
    makes the task write; a fully resolved non-empty list without WRITE is
    read-only/non-write. Unknown tools are left unclassified rather than being
    silently treated as reads.
    """
    task = metadata.get("task")
    if not isinstance(task, dict):
        return None
    criteria = task.get("evaluation_criteria") or {}
    actions = criteria.get("actions") or []
    if not actions:
        return "no_op"

    domain = str(metadata.get("base_domain") or metadata.get("domain") or "unknown")
    action_types: list[str] = []
    for action in actions:
        if not isinstance(action, dict) or not action.get("name"):
            return None
        requestor = str(action.get("requestor") or "assistant")
        action_type = _tool_type(domain, requestor, str(action["name"]))
        if action_type is None:
            return None
        action_types.append(action_type)
    return "write" if "write" in action_types else "read_only"


def _sample_tool_stats(metadata: dict[str, Any], domain: str) -> dict[str, float]:
    sequence = metadata.get("tool_sequence") or []
    if not isinstance(sequence, list):
        sequence = []
    tool_names = [str(name) for name in sequence]
    write_calls = sum(
        _tool_type(domain, "assistant", name) == "write" for name in tool_names
    )
    transfer_calls = tool_names.count("transfer_to_human_agents")
    return {
        "tool_calls": float(len(tool_names)),
        "write_calls": float(write_calls),
        "transfer_calls": float(transfer_calls),
    }


def _task_type_metrics(
    flat: list[Sample],
    *,
    n_samples_per_prompt: int,
) -> dict[str, float]:
    """Build online task-type, group-signal, and overactivity metrics."""
    buckets: dict[tuple[str, str], list[Sample]] = defaultdict(list)
    for sample in flat:
        metadata = sample.metadata or {}
        category = _task_action_category(metadata)
        if category is None:
            continue
        domain = str(metadata.get("base_domain") or metadata.get("domain") or "unknown")
        buckets[(domain, category)].append(sample)

    metrics: dict[str, float] = {}
    for (domain, category), samples in buckets.items():
        prefix = f"{domain}/{category}"
        valid = [sample for sample in samples if not sample.remove_sample]
        metrics[f"reward/task_type_sample_count/{prefix}"] = float(len(samples))
        metrics[f"reward/task_type_valid_sample_count/{prefix}"] = float(len(valid))
        metrics[f"reward/success_rate/{prefix}"] = sum(
            float((sample.metadata or {}).get("raw_reward", 0.0)) >= 1.0
            for sample in samples
        ) / len(samples)

        if valid:
            tool_stats = [
                _sample_tool_stats(sample.metadata or {}, domain) for sample in valid
            ]
            metrics[f"reward/tool_calls_mean/{prefix}"] = sum(
                stat["tool_calls"] for stat in tool_stats
            ) / len(tool_stats)
            metrics[f"reward/transfer_to_human_agents_frac/{prefix}"] = sum(
                stat["transfer_calls"] > 0 for stat in tool_stats
            ) / len(tool_stats)
            metrics[f"reward/transfer_to_human_agents_calls_mean/{prefix}"] = sum(
                stat["transfer_calls"] for stat in tool_stats
            ) / len(tool_stats)
            if category in {"read_only", "no_op"}:
                metrics[f"reward/violating_write_frac/{prefix}"] = sum(
                    stat["write_calls"] > 0 for stat in tool_stats
                ) / len(tool_stats)
                metrics[f"reward/violating_write_calls_mean/{prefix}"] = sum(
                    stat["write_calls"] for stat in tool_stats
                ) / len(tool_stats)

    if n_samples_per_prompt <= 0 or len(flat) % n_samples_per_prompt != 0:
        return metrics

    group_signal: dict[tuple[str, str], list[float]] = defaultdict(list)
    for start in range(0, len(flat), n_samples_per_prompt):
        group = flat[start : start + n_samples_per_prompt]
        first_meta = group[0].metadata or {}
        domain = str(
            first_meta.get("base_domain") or first_meta.get("domain") or "unknown"
        )
        category = _task_action_category(first_meta)
        if category is None:
            continue
        if any(
            str(
                (sample.metadata or {}).get("base_domain")
                or (sample.metadata or {}).get("domain")
                or "unknown"
            )
            != domain
            or _task_action_category(sample.metadata or {}) != category
            for sample in group
        ):
            continue
        valid_rewards = [
            float((sample.metadata or {}).get("raw_reward", 0.0))
            for sample in group
            if not sample.remove_sample
        ]
        has_signal = (
            len(valid_rewards) >= 2
            and max(valid_rewards) - min(valid_rewards) > 1e-8
        )
        group_signal[(domain, category)].append(float(has_signal))

    for (domain, category), values in group_signal.items():
        prefix = f"{domain}/{category}"
        metrics[f"reward/prompt_group_count/{prefix}"] = float(len(values))
        metrics[f"reward/nonzero_std_prompt_group_frac/{prefix}"] = (
            sum(values) / len(values)
        )

    # Domain-level overactivity summaries use non-write tasks as denominator.
    for domain in {key[0] for key in buckets}:
        nonwrite = [
            sample
            for category in ("read_only", "no_op")
            for sample in buckets.get((domain, category), [])
            if not sample.remove_sample
        ]
        if nonwrite:
            stats = [
                _sample_tool_stats(sample.metadata or {}, domain) for sample in nonwrite
            ]
            metrics[f"reward/violating_write_frac/{domain}"] = sum(
                stat["write_calls"] > 0 for stat in stats
            ) / len(stats)
            metrics[f"reward/violating_write_calls_mean/{domain}"] = sum(
                stat["write_calls"] for stat in stats
            ) / len(stats)

        domain_valid = [
            sample
            for category in TASK_ACTION_CATEGORIES
            for sample in buckets.get((domain, category), [])
            if not sample.remove_sample
        ]
        if domain_valid:
            stats = [
                _sample_tool_stats(sample.metadata or {}, domain)
                for sample in domain_valid
            ]
            metrics[f"reward/transfer_to_human_agents_frac/{domain}"] = sum(
                stat["transfer_calls"] > 0 for stat in stats
            ) / len(stats)
            metrics[f"reward/transfer_to_human_agents_calls_mean/{domain}"] = sum(
                stat["transfer_calls"] for stat in stats
            ) / len(stats)

    return metrics


def _normalize_rewards(
    rewards: list[float],
    *,
    valid_mask: list[float],
    n_samples_per_prompt: int,
    apply_std: bool,
) -> list[float]:
    if not rewards:
        return []
    if n_samples_per_prompt <= 0:
        raise ValueError(f"n_samples_per_prompt must be >= 1 (got {n_samples_per_prompt})")
    if len(rewards) % n_samples_per_prompt != 0:
        raise ValueError(
            "Reward count must be a multiple of n_samples_per_prompt "
            f"(count={len(rewards)}, n_samples_per_prompt={n_samples_per_prompt})."
        )

    rewards_t = torch.tensor(rewards, dtype=torch.float).view(-1, n_samples_per_prompt)
    mask_t = torch.tensor(valid_mask, dtype=torch.float).view(-1, n_samples_per_prompt)

    denom = mask_t.sum(dim=-1, keepdim=True).clamp(min=1.0)
    mean = (rewards_t * mask_t).sum(dim=-1, keepdim=True) / denom
    centered = (rewards_t - mean) * mask_t

    if apply_std:
        var = (centered**2).sum(dim=-1, keepdim=True) / denom
        centered = centered / (torch.sqrt(var) + 1e-6)

    return centered.flatten().tolist()


def _log_reward_component_metrics(args, flat: list[Sample]) -> None:
    """Push per-component reward breakdown to wandb.

    Runs inside the RolloutManager actor, where slime has already initialized
    wandb, so a plain wandb.log is enough. Uses the same rollout/step axis as
    slime's own rollout metrics.
    """
    if not flat or not getattr(args, "use_wandb", False):
        return
    try:
        import wandb
        from slime.utils.metric_utils import compute_rollout_step
    except ImportError:
        return
    if wandb.run is None:
        return

    # Bind reward/* to the rollout/step axis. slime's wandb init only defines
    # its own namespaces (train/*, rollout/*, ...), so without this the charts
    # fall back to the global _step axis (one tick per wandb.log call) and
    # drift far ahead of the rollout counter. Idempotent, so safe to repeat.
    global _REWARD_METRIC_DEFINED
    if not _REWARD_METRIC_DEFINED:
        wandb.define_metric("reward/*", step_metric="rollout/step")
        _REWARD_METRIC_DEFINED = True

    valid = [s for s in flat if not s.remove_sample]
    n = len(flat)

    def _mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    metrics: dict[str, float] = {
        "reward/task_reward_mean": _mean([(s.metadata or {}).get("raw_reward", 0.0) for s in flat]),
        "reward/success_rate": _mean([1.0 if (s.metadata or {}).get("raw_reward", 0.0) >= 1.0 else 0.0 for s in flat]),
        # Kept for dashboard compatibility; shaped_reward is now identical to
        # the binary task reward.
        "reward/shaped_reward_mean": _mean([(s.metadata or {}).get("shaped_reward", 0.0) for s in valid]),
        "reward/binary_reward_mean": _mean([(s.metadata or {}).get("binary_reward", 0.0) for s in valid]),
        "reward/partial_score_mean": _mean([(s.metadata or {}).get("partial_score", 0.0) for s in valid]),
        "reward/behavior_failure_frac": _mean(
            [1.0 if (s.metadata or {}).get("behavior_failure") else 0.0 for s in valid]
        ),
        "reward/valid_frac": len(valid) / n,
        "reward/truncated_frac": _mean(
            [0.0 if (s.metadata or {}).get("terminated") else 1.0 for s in valid]
        ),
        "reward/tau2_env_wait_seconds_mean": _mean(
            [float((s.metadata or {}).get("tau2_env_wait_seconds", 0.0)) for s in valid]
        ),
        "reward/agent_generation_seconds_mean": _mean(
            [float((s.metadata or {}).get("agent_generation_seconds", 0.0)) for s in valid]
        ),
        # Aggregate retained for dashboards that need the old broad category;
        # reward/termination/* below carries Tau2's native detailed reasons.
        "reward/termination/env_terminated_total": _mean(
            [
                1.0
                if (s.metadata or {}).get(
                    "env_terminated",
                    (s.metadata or {}).get("termination_reason") == "env_terminated",
                )
                else 0.0
                for s in flat
            ]
        ),
    }
    metrics.update(_domain_env_step_metrics(flat))
    task_type_metrics = _task_type_metrics(
        flat,
        n_samples_per_prompt=int(getattr(args, "n_samples_per_prompt", 0) or 0),
    )
    metrics.update(task_type_metrics)
    task_type_counts = {
        key.removeprefix("reward/task_type_sample_count/"): int(value)
        for key, value in task_type_metrics.items()
        if key.startswith("reward/task_type_sample_count/")
    }
    if task_type_counts:
        task_type_summaries = []
        for prefix, count in sorted(task_type_counts.items()):
            fields = [
                f"n={count}",
                f"ok={task_type_metrics[f'reward/success_rate/{prefix}']:.3f}",
            ]
            for label, metric_name in (
                ("nzstd", "nonzero_std_prompt_group_frac"),
                ("tools", "tool_calls_mean"),
                ("transfer", "transfer_to_human_agents_frac"),
                ("badwrite", "violating_write_frac"),
            ):
                metric_key = f"reward/{metric_name}/{prefix}"
                if metric_key in task_type_metrics:
                    fields.append(f"{label}={task_type_metrics[metric_key]:.3f}")
            task_type_summaries.append(f"{prefix}:" + ",".join(fields))
        print(
            "[tau2 task types] "
            + " ".join(task_type_summaries)
        )

    # Per-reason termination breakdown: wandb series + one greppable log line.
    reason_counts: dict[str, int] = defaultdict(int)
    for s in flat:
        reason_counts[str((s.metadata or {}).get("termination_reason") or "unknown")] += 1
    # Log zeros for known Tau2 reasons as well, so W&B produces continuous
    # per-rollout series instead of omitting steps where a reason did not occur.
    all_reasons = set(TAU2_TERMINATION_REASONS) | set(reason_counts)
    for reason in all_reasons:
        metrics[f"reward/termination/{reason}"] = reason_counts.get(reason, 0) / n
    print(
        "[tau2 termination] "
        + " ".join(f"{k}={v}" for k, v in sorted(reason_counts.items(), key=lambda kv: -kv[1]))
    )

    group_size = int(getattr(args, "n_samples_per_prompt", 0) or 0)
    if group_size > 0 and len(flat) % group_size == 0:
        groups = [flat[i : i + group_size] for i in range(0, len(flat), group_size)]
        task_signal_groups = 0
        proxy_only_groups = 0
        for group in groups:
            raw = [float((sample.metadata or {}).get("raw_reward", 0.0)) for sample in group]
            shaped = [float((sample.metadata or {}).get("shaped_reward", 0.0)) for sample in group]
            if max(raw) - min(raw) > 1e-8:
                task_signal_groups += 1
            elif max(shaped) - min(shaped) > 1e-8:
                proxy_only_groups += 1
        metrics["reward/task_signal_group_frac"] = task_signal_groups / len(groups)
        metrics["reward/proxy_only_group_frac"] = proxy_only_groups / len(groups)

    if valid:
        safety_meta = valid[0].metadata or {}
        metrics["reward/safety_bad_rollout"] = float(safety_meta.get("safety_bad_rollout", 0.0))
        metrics["reward/safety_bad_streak"] = float(safety_meta.get("safety_bad_streak", 0.0))

    # Retain component logging for compatibility with externally populated
    # metadata, although partial scores do not affect the training reward.
    comp_values: dict[str, list[float]] = defaultdict(list)
    for s in valid:
        for key, value in ((s.metadata or {}).get("partial_components") or {}).items():
            comp_values[key].append(float(value))
    for key, values in comp_values.items():
        metrics[f"reward/partial/{key}"] = _mean(values)
        metrics[f"reward/partial_coverage/{key}"] = len(values) / max(len(valid), 1)

    # Per-domain success rate (airline / retail / banking_knowledge).
    domain_success: dict[str, list[float]] = defaultdict(list)
    for s in flat:
        meta = s.metadata or {}
        domain = meta.get("base_domain") or meta.get("domain") or "unknown"
        domain_success[domain].append(1.0 if meta.get("raw_reward", 0.0) >= 1.0 else 0.0)
    for domain, values in domain_success.items():
        metrics[f"reward/success_rate/{domain}"] = _mean(values)

    domain_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"samples": 0, "success": 0, "format_error": 0, "max_steps": 0}
    )
    for sample in flat:
        meta = sample.metadata or {}
        domain = str(meta.get("base_domain") or meta.get("domain") or "unknown")
        stats = domain_counts[domain]
        stats["samples"] += 1
        stats["success"] += int(float(meta.get("raw_reward", 0.0)) >= 1.0)
        reason = meta.get("termination_reason")
        stats["format_error"] += int(reason == "format_error")
        stats["max_steps"] += int(reason == "max_steps")
    print(
        "[tau2 domains] "
        + " ".join(
            f"{domain}:n={stats['samples']},ok={stats['success']},"
            f"fmt={stats['format_error']},max={stats['max_steps']}"
            for domain, stats in sorted(domain_counts.items())
        )
    )
    for domain, stats in domain_counts.items():
        metrics[f"reward/sample_frac/{domain}"] = stats["samples"] / n
        metrics[f"reward/format_error_frac/{domain}"] = (
            stats["format_error"] / stats["samples"]
        )
        metrics[f"reward/max_steps_frac/{domain}"] = (
            stats["max_steps"] / stats["samples"]
        )

    format_error_counts: dict[str, int] = defaultdict(int)
    for sample in flat:
        error = str((sample.metadata or {}).get("format_error") or "")
        if not error:
            continue
        if "not canonical" in error:
            category = "noncanonical_serialization"
        elif "Invalid JSON" in error:
            category = "invalid_json"
        elif "Missing </tool_call>" in error:
            category = "missing_tool_call_end"
        elif "Expected exactly one" in error:
            category = "multiple_tool_calls"
        elif "Unexpected text after" in error:
            category = "trailing_text"
        elif "Empty assistant output" in error:
            category = "empty_output"
        else:
            category = "other"
        format_error_counts[category] += 1
    if format_error_counts:
        print(
            "[tau2 format errors] "
            + " ".join(
                f"{category}={count}"
                for category, count in sorted(
                    format_error_counts.items(), key=lambda item: -item[1]
                )
            )
        )
    for category, count in format_error_counts.items():
        metrics[f"reward/format_error_reason/{category}"] = count / n

    termination_counts: dict[str, int] = defaultdict(int)
    for sample in valid:
        termination_counts[str((sample.metadata or {}).get("termination_reason", "unknown"))] += 1
    for reason in set(TAU2_TERMINATION_REASONS) | set(termination_counts):
        metrics[f"reward/termination/{reason}_frac"] = (
            termination_counts.get(reason, 0) / len(valid)
        )

    rollout_id = (flat[0].metadata or {}).get("rollout_id")
    if rollout_id is not None:
        metrics["rollout/step"] = compute_rollout_step(args, rollout_id)
    wandb.log(metrics)


def _update_safety_state(flat: list[Sample]) -> str | None:
    """Return an error message when consecutive collapsed rollouts trip the guard."""
    global _SAFETY_BAD_STREAK, _SAFETY_LAST_ROLLOUT, _SAFETY_ROLLOUT_COUNT

    if os.environ.get("TAU2_SAFETY_ENABLED", "1") != "1":
        return None

    valid = [sample for sample in flat if not sample.remove_sample]
    if not valid:
        return None

    rollout_id = (valid[0].metadata or {}).get("rollout_id")
    rollout_key: object = ("rollout", rollout_id) if rollout_id is not None else object()
    success_rate = sum(
        1.0 if float((sample.metadata or {}).get("raw_reward", 0.0)) >= 1.0 else 0.0
        for sample in valid
    ) / len(valid)
    truncated_frac = sum(
        0.0 if (sample.metadata or {}).get("terminated") else 1.0 for sample in valid
    ) / len(valid)

    min_success_rate = float(os.environ.get("TAU2_ABORT_MIN_SUCCESS_RATE", "0.02"))
    max_truncated_frac = float(os.environ.get("TAU2_ABORT_TRUNCATED_FRAC", "0.60"))
    patience = int(os.environ.get("TAU2_ABORT_PATIENCE", "2"))
    warmup = int(os.environ.get("TAU2_ABORT_WARMUP_ROLLOUTS", "3"))
    bad_rollout = success_rate < min_success_rate and truncated_frac >= max_truncated_frac

    with _SAFETY_LOCK:
        if rollout_key == _SAFETY_LAST_ROLLOUT:
            return None
        _SAFETY_LAST_ROLLOUT = rollout_key
        _SAFETY_ROLLOUT_COUNT += 1
        if _SAFETY_ROLLOUT_COUNT <= warmup:
            _SAFETY_BAD_STREAK = 0
        elif bad_rollout:
            _SAFETY_BAD_STREAK += 1
        else:
            _SAFETY_BAD_STREAK = 0
        bad_streak = _SAFETY_BAD_STREAK

    for sample in flat:
        metadata = sample.metadata or {}
        sample.metadata = {
            **metadata,
            "safety_bad_rollout": bad_rollout,
            "safety_bad_streak": bad_streak,
        }

    if patience > 0 and bad_streak >= patience:
        return (
            "Tau2 rollout safety circuit breaker triggered: "
            f"success_rate={success_rate:.4f} < {min_success_rate:.4f}, "
            f"truncated_frac={truncated_frac:.4f} >= {max_truncated_frac:.4f}, "
            f"bad_streak={bad_streak}. Refusing to optimize another collapsed rollout."
        )
    return None


def tau2_reward_post_process(args, samples: list[Sample] | list[list[Sample]]) -> tuple[list[float], list[float]]:
    flat = _flatten(samples)

    raw_rewards: list[float] = []
    binary_rewards: list[float] = []
    valid_mask: list[float] = []

    for sample in flat:
        metadata = sample.metadata or {}
        env_task_reward = float(sample.get_reward_value(args))
        is_valid = not sample.remove_sample
        task_reward = (1.0 if env_task_reward >= 1.0 else 0.0) if is_valid else 0.0
        no_write_violation = _no_write_reward_violation(metadata)
        if no_write_violation:
            task_reward = 0.0
        behavior_failure = metadata.get("termination_reason") in BEHAVIOR_FAILURE_REASONS

        raw_rewards.append(task_reward)
        valid_mask.append(1.0 if is_valid else 0.0)
        binary_rewards.append(task_reward)

        sample.metadata = {
            **metadata,
            "env_task_reward": env_task_reward,
            "no_write_reward_violation": no_write_violation,
            "raw_reward": task_reward,
            "partial_score": 0.0,
            "partial_components": {},
            "progress_score": task_reward,
            "behavior_failure": behavior_failure,
            "shaped_reward": task_reward,
            "binary_reward": task_reward,
        }

    safety_error = _update_safety_state(flat)
    _log_reward_component_metrics(args, flat)
    if safety_error is not None:
        raise RuntimeError(safety_error)

    if (
        args.advantage_estimator in {"grpo", "gspo", "reinforce_plus_plus_baseline"}
        and args.rewards_normalization
        and binary_rewards
    ):
        rewards = _normalize_rewards(
            binary_rewards,
            valid_mask=valid_mask,
            n_samples_per_prompt=args.n_samples_per_prompt,
            apply_std=(args.advantage_estimator in {"grpo", "gspo"} and args.grpo_std_normalization),
        )
        return raw_rewards, rewards

    return raw_rewards, binary_rewards
