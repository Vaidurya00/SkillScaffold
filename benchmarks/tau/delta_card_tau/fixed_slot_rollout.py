"""Fixed-query rollout collection for fair Tau2 E7 comparisons.

The stock asynchronous collector admits the first ``rollout_batch_size``
groups that pass its dynamic filter.  That is useful for throughput, but it can
bias a mixed-domain batch toward faster environments.  This collector instead
reserves the query slots up front, retries a failed slot with the *same* query,
and waits for every reserved slot before returning a training batch.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from copy import deepcopy
import os
from typing import Any

from slime.rollout.base_types import RolloutFnTrainOutput
from slime.rollout.filter_hub.base_types import MetricGatherer, call_dynamic_filter
from slime.rollout.sglang_rollout import (
    GenerateState,
    generate_and_rm_group,
    generate_rollout as stock_generate_rollout,
)
from slime.utils.async_utils import run
from slime.utils.misc import load_function
from slime.utils.types import Sample


def _flatten(group: Any) -> list[Sample]:
    values = list(group if isinstance(group, list) else [group])
    samples: list[Sample] = []
    while values:
        value = values.pop(0)
        if isinstance(value, list):
            values[0:0] = value
        else:
            samples.append(value)
    return samples


def _head(group: Any) -> Sample:
    samples = _flatten(group)
    if not samples:
        raise ValueError("empty fixed-slot group")
    return samples[0]


def _domain(group: Any) -> str:
    metadata = _head(group).metadata or {}
    return str(
        metadata.get("base_domain")
        or metadata.get("domain")
        or metadata.get("fixed_query_domain")
        or "unknown"
    ).strip().lower()


def _task_id(group: Any) -> str:
    metadata = _head(group).metadata or {}
    return str(
        metadata.get("task_id")
        or metadata.get("id")
        or metadata.get("fixed_query_task_id")
        or "unknown"
    )


def _slot(group: Any) -> int:
    metadata = _head(group).metadata or {}
    return int(metadata.get("fixed_query_slot", _head(group).index or 0))


def _failure_detail(group: Any) -> str:
    details = []
    for sample in _flatten(group):
        metadata = sample.metadata or {}
        for key in ("source_episode_error", "termination_reason", "error"):
            value = metadata.get(key)
            if value:
                details.append(f"{key}={value}")
    return " | ".join(dict.fromkeys(details))


def _fresh_attempt(original_group: list[Sample], *, rollout_id: int, slot: int, attempt: int) -> list[Sample]:
    group = deepcopy(original_group)
    for sample in group:
        sample.session_id = None
        metadata = dict(sample.metadata or {})
        metadata.update(
            {
                "training_rollout_id": rollout_id,
                "fixed_query_slot": slot,
                "fixed_query_attempt": attempt,
            }
        )
        sample.metadata = metadata
    return group


async def _run_slot(
    args,
    state: GenerateState,
    original_group: list[Sample],
    dynamic_filter,
    metric_gatherer: MetricGatherer,
    *,
    rollout_id: int,
    slot: int,
    max_retries: int,
) -> tuple[Any, int]:
    """Generate one reserved slot, retrying only its original query."""

    domain = _domain(original_group)
    task_id = _task_id(original_group)
    last_reason = "unknown"
    last_detail = ""
    for attempt in range(max_retries + 1):
        attempt_group = _fresh_attempt(original_group, rollout_id=rollout_id, slot=slot, attempt=attempt)
        try:
            generated = await generate_and_rm_group(
                args,
                attempt_group,
                sampling_params=state.sampling_params.copy(),
                evaluation=False,
            )
            decision = call_dynamic_filter(dynamic_filter, args, generated)
            if decision.keep:
                for sample in _flatten(generated):
                    metadata = dict(sample.metadata or {})
                    metadata.update(
                        {
                            "fixed_query_slot": slot,
                            "fixed_query_attempt": attempt,
                            "fixed_query_retry_count": attempt,
                            "fixed_query_domain": domain,
                            "fixed_query_task_id": task_id,
                        }
                    )
                    sample.metadata = metadata
                return generated, attempt
            last_reason = str(decision.reason or "rejected")
            last_detail = _failure_detail(generated)
        except Exception as exc:  # retry the reserved query; never draw a replacement
            last_reason = f"exception_{type(exc).__name__}"
            last_detail = str(exc)

        metric_gatherer.on_dynamic_filter_drop(last_reason)
        action = f"retry={attempt + 1}/{max_retries}" if attempt < max_retries else "retries_exhausted"
        detail_suffix = f" detail={last_detail}" if last_detail else ""
        print(
            "[tau/e7-fixed-slot] rejected "
            f"rollout={rollout_id} slot={slot} {action} "
            f"domain={domain} task_id={task_id} reason={last_reason}{detail_suffix}",
            flush=True,
        )

    raise RuntimeError(
        "fixed query slot exhausted retries: "
        f"rollout={rollout_id} slot={slot} domain={domain} task_id={task_id} "
        f"max_retries={max_retries} last_reason={last_reason} last_detail={last_detail}"
    )


async def _generate_fixed_rollout(args, rollout_id: int, data_source) -> RolloutFnTrainOutput:
    state = GenerateState(args)
    state.reset()
    max_retries = int(os.environ.get("TAU_E7_FIXED_SLOT_RETRIES", "2"))
    if max_retries < 0:
        raise ValueError(f"TAU_E7_FIXED_SLOT_RETRIES must be non-negative, got {max_retries}")

    # This is the only draw from the data source for the step.  Rejected slots
    # are retried from a deep copy of their own original query.
    reserved = data_source.get_samples(args.rollout_batch_size)
    if len(reserved) != args.rollout_batch_size:
        raise RuntimeError(
            f"fixed-slot sampler reserved {len(reserved)} groups, expected {args.rollout_batch_size}"
        )

    selected_domains = Counter(_domain(group) for group in reserved)
    selected_tasks = [_task_id(group) for group in reserved]
    print(
        "[tau/e7-fixed-slot] reserved "
        f"rollout={rollout_id} groups={len(reserved)} domains={dict(sorted(selected_domains.items()))}",
        flush=True,
    )

    dynamic_filter = (
        load_function(args.dynamic_sampling_filter_path)
        if args.dynamic_sampling_filter_path is not None
        else None
    )
    metric_gatherer = MetricGatherer()
    outcomes = await asyncio.gather(
        *[
            _run_slot(
                args,
                state,
                group,
                dynamic_filter,
                metric_gatherer,
                rollout_id=rollout_id,
                slot=slot,
                max_retries=max_retries,
            )
            for slot, group in enumerate(reserved)
        ],
        return_exceptions=True,
    )

    failures = [value for value in outcomes if isinstance(value, BaseException)]
    if failures:
        state.reset()
        details = "; ".join(str(exc) for exc in failures[:4])
        raise RuntimeError(f"{len(failures)} fixed query slot(s) failed; {details}")

    accepted_with_retries = sorted(
        [(value[0], int(value[1])) for value in outcomes],
        key=lambda value: _slot(value[0]),
    )
    accepted = [value[0] for value in accepted_with_retries]
    retry_counts = [value[1] for value in accepted_with_retries]
    trained_domains = Counter(_domain(group) for group in accepted)
    if trained_domains != selected_domains:
        raise RuntimeError(
            f"fixed-slot domain mismatch: selected={dict(selected_domains)} trained={dict(trained_domains)}"
        )
    # Use a multiset because a training pool may intentionally repeat task ids.
    if Counter(_task_id(group) for group in accepted) != Counter(selected_tasks):
        raise RuntimeError("fixed-slot task multiset changed between reservation and training")

    if args.rollout_sample_filter_path is not None:
        load_function(args.rollout_sample_filter_path)(args, accepted)
    if args.rollout_all_samples_process_path is not None:
        load_function(args.rollout_all_samples_process_path)(args, accepted, data_source)

    metrics = metric_gatherer.collect()
    metrics.update(
        {
            "rollout/fixed_slot/selected_groups": len(reserved),
            "rollout/fixed_slot/trained_groups": len(accepted),
            "rollout/fixed_slot/retries_total": sum(retry_counts),
            "rollout/fixed_slot/retried_groups": sum(value > 0 for value in retry_counts),
            "rollout/fixed_slot/max_retry_used": max(retry_counts, default=0),
        }
    )
    for domain in sorted(set(selected_domains) | set(trained_domains)):
        metrics[f"rollout/fixed_slot/selected_{domain}"] = selected_domains[domain]
        metrics[f"rollout/fixed_slot/trained_{domain}"] = trained_domains[domain]
        metrics[f"rollout/fixed_slot/retries_{domain}"] = sum(
            retry for retry, group in zip(retry_counts, accepted) if _domain(group) == domain
        )

    print(
        "[tau/e7-fixed-slot] complete "
        f"rollout={rollout_id} domains={dict(sorted(trained_domains.items()))} "
        f"retries={sum(retry_counts)}",
        flush=True,
    )
    state.reset()
    return RolloutFnTrainOutput(samples=accepted, metrics=metrics)


def generate_rollout(args, rollout_id: int, data_source: Any, evaluation: bool = False):
    if evaluation:
        return stock_generate_rollout(args, rollout_id, data_source, evaluation=True)
    return run(_generate_fixed_rollout(args, rollout_id, data_source))
