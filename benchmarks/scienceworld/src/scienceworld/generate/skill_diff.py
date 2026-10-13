"""ScienceWorld replay skill rewards from paired teacher-forced forwards."""

from __future__ import annotations

import asyncio
import math
from typing import Any

from slime.utils.http_utils import post
from slime.utils.types import Sample

from src.scienceworld.prompts import (
    format_executor_messages,
    format_executor_with_skill_messages,
)
from src.webshop.utils.common import ensure_metadata


def _source_identity(sample: Sample) -> tuple[str, int | str | None, str]:
    metadata = ensure_metadata(sample)
    label = sample.label if isinstance(sample.label, dict) else {}
    category = str(
        metadata.get("category")
        or label.get("category")
        or label.get("task_type")
        or "unknown"
    ).strip().lower()
    goal_idx = metadata.get("goal_idx", label.get("goal_idx"))
    query = str(
        metadata.get("task_description")
        or label.get("task_description")
        or ""
    ).strip()
    return category or "unknown", goal_idx, query


async def retrieve_successful_buffer_trajectory(
    args,
    source_scout: Sample,
) -> tuple[dict[str, Any] | None, str]:
    """Retrieve the nearest eligible canonical trajectory in the same category."""

    del args
    # Legacy compatibility only.  Delta-RL v2 never calls this path; keeping
    # the import local prevents its runtime from constructing or consulting a
    # cross-query Buffer.
    from src.webshop.trajectory_buffer import get_trajectory_buffer

    category, source_goal_idx, _ = _source_identity(source_scout)
    return get_trajectory_buffer().nearest_successful_trajectory(
        category=category,
        source_goal_idx=source_goal_idx,
    )


def build_model_token_mask(record: dict[str, Any]) -> list[bool]:
    """Build a response-length mask containing only model-generated turns."""

    response_token_ids = record.get("response_token_ids")
    turns = record.get("turns")
    if not isinstance(response_token_ids, list) or not response_token_ids:
        raise ValueError("retrieved trajectory has no response token IDs")
    if not isinstance(turns, list) or not turns:
        raise ValueError("retrieved trajectory has no structured turns")

    mask = [False] * len(response_token_ids)
    for turn in turns:
        if not isinstance(turn, dict):
            raise ValueError("retrieved trajectory contains a malformed turn")
        try:
            start = int(turn["response_token_start"])
            end = int(turn["response_token_end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("retrieved turn has no valid response token span") from exc
        if start < 0 or end <= start or end > len(mask):
            raise ValueError(
                "retrieved turn response span is out of bounds: "
                f"start={start}, end={end}, response_len={len(mask)}"
            )
        if any(mask[start:end]):
            raise ValueError("retrieved trajectory has overlapping response token spans")
        mask[start:end] = [True] * (end - start)

    if not any(mask):
        raise ValueError("retrieved trajectory has no model-generated response tokens")
    return mask


def _prompt_token_ids(
    args,
    tokenizer,
    *,
    query: str,
    skill: str | None,
) -> list[int]:
    messages = (
        format_executor_with_skill_messages(query, skill)
        if skill is not None
        else format_executor_messages(query)
    )
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **(getattr(args, "apply_chat_template_kwargs", None) or {}),
    )
    return tokenizer(prompt_text, add_special_tokens=False)["input_ids"]


def _input_logprob_value(row: Any) -> float:
    if isinstance(row, dict):
        value = row.get("logprob", row.get("log_prob"))
    elif isinstance(row, (list, tuple)) and row:
        value = row[0]
    else:
        value = None
    try:
        logprob = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("SGLang returned a malformed input-token logprob") from exc
    if not math.isfinite(logprob):
        raise ValueError("SGLang returned a non-finite input-token logprob")
    return logprob


def extract_response_logprobs(
    output: dict[str, Any],
    *,
    response_token_count: int,
) -> list[float]:
    """Align SGLang input-token logprobs with the fixed response suffix."""

    meta_info = output.get("meta_info")
    rows = meta_info.get("input_token_logprobs") if isinstance(meta_info, dict) else None
    if not isinstance(rows, list):
        raise ValueError("SGLang response has no input_token_logprobs")
    if response_token_count <= 0 or len(rows) < response_token_count:
        raise ValueError(
            "SGLang input logprob length is shorter than the fixed response: "
            f"rows={len(rows)}, response={response_token_count}"
        )
    return [_input_logprob_value(row) for row in rows[-response_token_count:]]


async def _teacher_forced_logprobs(
    args,
    *,
    prompt_token_ids: list[int],
    response_token_ids: list[int],
) -> list[float]:
    total_length = len(prompt_token_ids) + len(response_token_ids)
    max_context_length = int(getattr(args, "rollout_max_context_len", 0) or 0)
    if max_context_length > 0 and total_length > max_context_length:
        raise ValueError(
            "teacher-forcing sequence exceeds rollout context: "
            f"{total_length}>{max_context_length}"
        )

    output = await post(
        f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate",
        {
            "input_ids": prompt_token_ids + response_token_ids,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 0,
                "skip_special_tokens": False,
            },
            "return_logprob": True,
            "logprob_start_len": 0,
        },
    )
    return extract_response_logprobs(
        output,
        response_token_count=len(response_token_ids),
    )


def _masked_mean(values: list[float], mask: list[bool]) -> float:
    if len(values) != len(mask):
        raise ValueError(f"logprob/mask mismatch: {len(values)}!={len(mask)}")
    selected = [value for value, keep in zip(values, mask) if keep]
    if not selected:
        raise ValueError("teacher-forcing mask selects no model tokens")
    return float(sum(selected) / len(selected))


def _empty_result(reason: str) -> dict[str, Any]:
    return {
        "available": False,
        "skip_reason": str(reason),
        "base_mean_logprob": None,
        "skill_mean_logprob": None,
        "diff_reward": 0.0,
        "scored_token_count": 0,
    }


def build_successful_scout_scoring_record(source_scout: Sample) -> dict[str, Any]:
    """Build a prompt-neutral scoring record directly from one successful scout."""

    metadata = ensure_metadata(source_scout)
    if not bool(metadata.get("won", False)):
        raise ValueError("Delta scoring requires an environment-confirmed success")
    response_length = int(source_scout.response_length or 0)
    if response_length <= 0 or not isinstance(source_scout.tokens, list):
        raise ValueError("successful scout has no response token suffix")
    response_token_ids = [int(token_id) for token_id in source_scout.tokens[-response_length:]]
    turns = metadata.get("turns")
    if not isinstance(turns, list) or not turns:
        raise ValueError("successful scout has no structured response turns")

    label = source_scout.label if isinstance(source_scout.label, dict) else {}
    record_id = str(metadata.get("record_id") or label.get("record_id") or "")
    query = str(
        metadata.get("task_description")
        or label.get("task_description")
        or ""
    ).strip()
    if not query:
        raise ValueError("successful scout has no task description")
    return {
        "scoring_source": "selected_successful_scout",
        "goal_idx": record_id,
        "category": str(metadata.get("category") or label.get("category") or ""),
        "query": query,
        "query_key": f"selected-scout:{record_id}",
        "response_token_ids": response_token_ids,
        "turns": turns,
        "turn_count": len(turns),
        "provenance": {
            "trajectory_role": "selected_successful_scout",
            "executor_group_id": str(metadata.get("executor_group_id") or ""),
            "executor_group_position": int(
                metadata.get("executor_group_position", source_scout.index or 0)
            ),
        },
    }


async def score_candidate_skills_on_retrieved_trajectory(
    args,
    *,
    source_scout: Sample,
    generator_samples: list[Sample],
    retrieved_record: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Compute four skill rewards using one shared no-skill forward."""

    if retrieved_record is None:
        record, retrieval_error = await retrieve_successful_buffer_trajectory(
            args,
            source_scout,
        )
    else:
        record = retrieved_record
        retrieval_error = ""
    if record is None:
        return [_empty_result(retrieval_error) for _ in generator_samples]

    try:
        response_token_ids = [int(token_id) for token_id in record["response_token_ids"]]
        model_token_mask = build_model_token_mask(record)
        from slime.rollout.sglang_rollout import GenerateState

        tokenizer = GenerateState(args).tokenizer
        base_prompt_token_ids = _prompt_token_ids(
            args,
            tokenizer,
            query=str(record["query"]),
            skill=None,
        )
    except Exception as exc:
        return [_empty_result(f"trajectory_preparation_failed:{exc}") for _ in generator_samples]

    async def _score_skill(generator_sample: Sample) -> dict[str, Any]:
        metadata = ensure_metadata(generator_sample)
        skill = str(metadata.get("skill") or "").strip()
        if float(metadata.get("format_score", 0.0) or 0.0) < 1.0 or not skill:
            return _empty_result("invalid_skill_output")
        try:
            skill_prompt_token_ids = _prompt_token_ids(
                args,
                tokenizer,
                query=str(record["query"]),
                skill=skill,
            )
            logprobs = await _teacher_forced_logprobs(
                args,
                prompt_token_ids=skill_prompt_token_ids,
                response_token_ids=response_token_ids,
            )
            return {
                "available": True,
                "skip_reason": "",
                "skill_logprobs": logprobs,
            }
        except Exception as exc:
            return _empty_result(f"with_skill_forward_failed:{exc}")

    try:
        forward_results = await asyncio.gather(
            _teacher_forced_logprobs(
                args,
                prompt_token_ids=base_prompt_token_ids,
                response_token_ids=response_token_ids,
            ),
            *[_score_skill(sample) for sample in generator_samples],
        )
        base_logprobs = forward_results[0]
        base_mean = _masked_mean(base_logprobs, model_token_mask)
    except Exception as exc:
        return [_empty_result(f"no_skill_forward_failed:{exc}") for _ in generator_samples]

    scoring_source = str(record.get("scoring_source") or "retrieved_buffer_trajectory")
    shared_metadata = {
        "skill_diff_scoring_source": scoring_source,
        "scored_goal_idx": record.get("goal_idx"),
        "scored_category": record.get("category"),
        "scored_query": record.get("query"),
        "scored_query_key": record.get("query_key"),
        "scored_trajectory_turn_count": int(record.get("turn_count", 0)),
        "scored_trajectory_provenance": record.get("provenance", {}),
    }
    if scoring_source == "retrieved_buffer_trajectory":
        shared_metadata.update(
            {
                "retrieved_goal_idx": record.get("goal_idx"),
                "retrieved_category": record.get("category"),
                "retrieved_query": record.get("query"),
                "retrieved_query_key": record.get("query_key"),
                "retrieved_similarity": float(record.get("retrieval_similarity", 0.0)),
                "retrieval_candidate_count": int(
                    record.get("retrieval_source_candidate_count", 0)
                ),
                "retrieved_trajectory_turn_count": int(record.get("turn_count", 0)),
                "retrieved_trajectory_provenance": record.get("provenance", {}),
            }
        )
    scored_results = []
    skill_results = forward_results[1:]
    for skill_result in skill_results:
        if not skill_result.get("available"):
            scored_results.append({**skill_result, **shared_metadata})
            continue
        try:
            skill_mean = _masked_mean(
                skill_result["skill_logprobs"],
                model_token_mask,
            )
        except Exception as exc:
            scored_results.append(
                {
                    **_empty_result(f"skill_logprob_alignment_failed:{exc}"),
                    **shared_metadata,
                }
            )
            continue
        scored_results.append(
            {
                "available": True,
                "skip_reason": "",
                "base_mean_logprob": base_mean,
                "skill_mean_logprob": skill_mean,
                "diff_reward": float(skill_mean - base_mean),
                "scored_token_count": sum(model_token_mask),
                **shared_metadata,
            }
        )
    return scored_results


async def score_candidate_skills_on_successful_scout(
    args,
    *,
    source_scout: Sample,
    generator_samples: list[Sample],
) -> list[dict[str, Any]]:
    """Score four skills against the exact selected scout, without retrieval."""

    try:
        record = build_successful_scout_scoring_record(source_scout)
    except Exception as exc:
        return [_empty_result(f"selected_scout_preparation_failed:{exc}") for _ in generator_samples]
    return await score_candidate_skills_on_retrieved_trajectory(
        args,
        source_scout=source_scout,
        generator_samples=generator_samples,
        retrieved_record=record,
    )


def apply_skill_diff_rewards(
    generator_samples: list[Sample],
    scoring_results: list[dict[str, Any]],
    *,
    reward_source: str = "retrieved_trajectory_logprob_diff",
) -> None:
    if len(generator_samples) != len(scoring_results):
        raise ValueError(
            "generator/scoring result count mismatch: "
            f"{len(generator_samples)}!={len(scoring_results)}"
        )

    for sample, result in zip(generator_samples, scoring_results):
        metadata = ensure_metadata(sample)
        raw_diff = float(result.get("diff_reward", 0.0) or 0.0)
        format_ok = float(metadata.get("format_score", 0.0) or 0.0) >= 1.0
        audit_applied = float(
            metadata.get("skill_language_audit_applied", 0.0) or 0.0
        ) > 0.5
        audit_pass = float(
            metadata.get("skill_language_audit_pass", 0.0) or 0.0
        ) > 0.5
        audit_zeroed = float(
            metadata.get("skill_language_audit_reward_zeroed", 0.0) or 0.0
        ) > 0.5
        diff_is_finite = math.isfinite(raw_diff)
        reward_before_audit = (
            raw_diff
            if result.get("available") and format_ok and diff_is_finite
            else 0.0
        )
        reward_zeroed_by_audit = audit_zeroed or (audit_applied and not audit_pass)
        audit_eligible_reward = (
            reward_before_audit if not reward_zeroed_by_audit else 0.0
        )
        # The no-skill forward is the explicit null baseline. A candidate that
        # is worse than that baseline must not receive a negative raw reward:
        # invalid/audit-failed candidates and non-positive diffs all enter the
        # within-group normalization at the same zero floor.
        reward = max(0.0, float(audit_eligible_reward))

        for key in (
            "skill_diff_scoring_source",
            "scored_goal_idx",
            "scored_category",
            "scored_query",
            "scored_query_key",
            "scored_trajectory_turn_count",
            "scored_trajectory_provenance",
            "retrieved_goal_idx",
            "retrieved_category",
            "retrieved_query",
            "retrieved_query_key",
            "retrieved_similarity",
            "retrieval_candidate_count",
            "retrieved_trajectory_turn_count",
            "retrieved_trajectory_provenance",
        ):
            if key in result:
                metadata[key] = result[key]
        metadata.update(
            {
                "generator_reward_source": str(reward_source),
                "skill_diff_available": 1.0 if result.get("available") else 0.0,
                "skill_diff_skip_reason": str(result.get("skip_reason") or ""),
                "skill_diff_base_mean_logprob": result.get("base_mean_logprob"),
                "skill_diff_with_skill_mean_logprob": result.get(
                    "skill_mean_logprob"
                ),
                "skill_diff_reward": raw_diff,
                "skill_diff_scored_token_count": int(
                    result.get("scored_token_count", 0) or 0
                ),
                "raw_generator_base_reward": raw_diff,
                "raw_generator_reward_before_language_audit": reward_before_audit,
                "generator_reward_after_null_clip": float(reward),
                "reward_clipped_by_nonpositive_diff": (
                    1.0
                    if result.get("available")
                    and format_ok
                    and not reward_zeroed_by_audit
                    and raw_diff <= 0.0
                    else 0.0
                ),
                "reward_zeroed_by_nonfinite_diff": (
                    1.0 if result.get("available") and not diff_is_finite else 0.0
                ),
                "reward_zeroed_by_format": 0.0 if format_ok else 1.0,
                "reward_zeroed_by_language_audit": (
                    1.0 if reward_zeroed_by_audit else 0.0
                ),
                "raw_training_reward": float(reward),
            }
        )
        sample.reward = float(reward)
