"""Four-card by four-scout teacher-forced Delta credit for ALFWorld."""

from __future__ import annotations

import asyncio
import math
import os

import numpy as np

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.types import Sample

from src.alfworld.generate.delta_credit import (
    _masked_mean,
    _model_turn_mask,
    _teacher_forced,
)
from src.alfworld.utils.common import ensure_metadata
from src.scienceworld.generate.skill_diff import _prompt_token_ids


def _prompt_ids(args, tokenizer, task, skill):
    return _prompt_token_ids(args, tokenizer, query=task, skill=skill)


SCOUT_COUNT = 4
_FORWARD_SEMAPHORE: asyncio.Semaphore | None = None


def _forward_semaphore() -> asyncio.Semaphore:
    global _FORWARD_SEMAPHORE
    if _FORWARD_SEMAPHORE is None:
        _FORWARD_SEMAPHORE = asyncio.Semaphore(
            int(os.environ.get("ALFWORLD_GROUP_DELTA_FORWARD_CONCURRENCY", "48"))
        )
    return _FORWARD_SEMAPHORE


async def _bounded_teacher_forced(args, prompt_ids: list[int], response_ids: list[int]) -> list[float]:
    async with _forward_semaphore():
        return await _teacher_forced(args, prompt_ids, response_ids)


def aggregate_group_delta(lifts: list[float], wins: list[bool]) -> dict:
    """Give success/failure classes equal weight when both are present.

    A one-class group is a complete group: its observed class receives weight 1.
    Positive lift means with-skill minus no-skill.  Failures reverse the sign.
    """

    if len(lifts) != SCOUT_COUNT or len(wins) != SCOUT_COUNT:
        raise ValueError("group Delta requires exactly four lifts and four labels")
    if not all(math.isfinite(float(value)) for value in lifts):
        raise ValueError("group Delta received a non-finite lift")
    success = [float(value) for value, won in zip(lifts, wins) if won]
    failure = [float(value) for value, won in zip(lifts, wins) if not won]
    success_mean = float(np.mean(success)) if success else None
    failure_mean = float(np.mean(failure)) if failure else None
    if success and failure:
        score = 0.5 * success_mean - 0.5 * failure_mean
        mode = "mixed_equal_class_weight"
    elif success:
        score = success_mean
        mode = "all_success_full_weight"
    else:
        score = -failure_mean
        mode = "all_failure_full_weight"
    return {
        "diff_reward": float(score),
        "aggregation_mode": mode,
        "success_count": len(success),
        "failure_count": len(failure),
        "success_mean_lift": success_mean,
        "failure_mean_lift": failure_mean,
    }


def _empty(reason: str) -> dict:
    return {
        "available": False,
        "skip_reason": reason,
        "diff_reward": 0.0,
        "aggregation_mode": "unavailable",
        "success_count": 0,
        "failure_count": 0,
        "success_mean_lift": None,
        "failure_mean_lift": None,
        "trajectory_lifts": [],
        "scored_token_count": 0,
        "model_token_count": 0,
        "environment_token_count": 0,
        "scored_environment_token_count": 0,
        "loss_mask_mismatch_count": 0,
        "credit_scope": "all_model_tokens",
    }


async def score_cards_on_scout_group(
    args,
    scouts: list[Sample],
    cards: list[Sample],
) -> list[dict]:
    """Score every valid card on every scout while caching four baselines."""

    if len(scouts) != SCOUT_COUNT:
        return [_empty("incomplete_scout_group") for _ in cards]

    tokenizer = GenerateState(args).tokenizer
    prepared = []
    try:
        for position, scout in enumerate(scouts):
            metadata = ensure_metadata(scout)
            response_length = int(scout.response_length or 0)
            if response_length <= 0:
                raise ValueError(f"scout {position} has no response tokens")
            response_ids = [int(value) for value in scout.tokens[-response_length:]]
            mask, audit = _model_turn_mask(scout, response_length)
            if int(audit["environment_token_count"]) != response_length - int(sum(mask)):
                raise ValueError(f"scout {position} mask audit is inconsistent")
            prepared.append(
                {
                    "task": str(metadata.get("task_description") or ""),
                    "won": bool(metadata.get("won", False)),
                    "response_ids": response_ids,
                    "mask": mask,
                    "audit": audit,
                }
            )
        tasks = {item["task"] for item in prepared}
        if len(tasks) != 1 or not next(iter(tasks), ""):
            raise ValueError("scouts do not share one non-empty task description")
    except Exception as exc:
        return [_empty(f"scout_group_invalid:{exc}") for _ in cards]

    task = prepared[0]["task"]
    base_prompt = _prompt_ids(args, tokenizer, task, None)
    try:
        base_values = await asyncio.gather(
            *[
                _bounded_teacher_forced(args, base_prompt, item["response_ids"])
                for item in prepared
            ]
        )
    except Exception as exc:
        return [_empty(f"no_skill_forward_failed:{exc}") for _ in cards]

    async def score_one(card: Sample) -> dict:
        metadata = ensure_metadata(card)
        skill = str(metadata.get("skill") or "").strip()
        if float(metadata.get("format_score", 0.0) or 0.0) < 1.0 or not skill:
            return _empty("invalid_or_abstained_card")
        try:
            skill_prompt = _prompt_ids(args, tokenizer, task, skill)
            skill_values = await asyncio.gather(
                *[
                    _bounded_teacher_forced(args, skill_prompt, item["response_ids"])
                    for item in prepared
                ]
            )
            lifts = [
                float(
                    _masked_mean(with_skill, item["mask"])
                    - _masked_mean(no_skill, item["mask"])
                )
                for item, no_skill, with_skill in zip(prepared, base_values, skill_values)
            ]
            aggregate = aggregate_group_delta(lifts, [item["won"] for item in prepared])
            scored_tokens = [int(sum(item["mask"])) for item in prepared]
            model_tokens = [int(item["audit"]["model_token_count"]) for item in prepared]
            environment_tokens = [int(item["audit"]["environment_token_count"]) for item in prepared]
            mismatches = [int(item["audit"]["loss_mask_mismatch_count"]) for item in prepared]
            return {
                "available": True,
                "skip_reason": "",
                **aggregate,
                "trajectory_lifts": lifts,
                "scored_token_count": int(sum(scored_tokens)),
                "scored_token_count_by_trajectory": scored_tokens,
                "model_token_count": int(sum(model_tokens)),
                "environment_token_count": int(sum(environment_tokens)),
                "scored_environment_token_count": 0,
                "loss_mask_mismatch_count": int(sum(mismatches)),
                "credit_scope": "all_model_tokens",
            }
        except Exception as exc:
            return _empty(f"with_skill_forward_failed:{exc}")

    return list(await asyncio.gather(*[score_one(card) for card in cards]))
