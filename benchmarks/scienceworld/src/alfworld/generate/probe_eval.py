"""Static candidate-skill utility probe for ALFWorld.

Each parquet row is one no-skill baseline or one frozen candidate skill.  Slime
repeats every row K times.  Candidate repeat zero additionally teacher-forces
three frozen scout trajectories to compute the preregistered Delta estimators.
"""

from __future__ import annotations

import asyncio
from copy import deepcopy
import math
import os
from pathlib import Path
from typing import Any

import torch

from slime.utils.types import Sample

from src.alfworld.generate.delta_credit import _prompt_ids, _teacher_forced
from src.alfworld.generate.episode import run_episode
from src.alfworld.utils.common import ensure_metadata


_ANCHOR_CASES: dict[str, dict] | None = None
_ANCHOR_LOCK = asyncio.Lock()


def _load_cases() -> dict[str, dict]:
    global _ANCHOR_CASES
    if _ANCHOR_CASES is None:
        path = Path(os.environ["ALFWORLD_STATIC_PROBE_ANCHORS"])
        payload = torch.load(path, map_location="cpu", weights_only=False)
        _ANCHOR_CASES = payload["cases"]
    return _ANCHOR_CASES


def _response(anchor: dict) -> list[int]:
    length = int(anchor["response_length"])
    return [int(value) for value in anchor["tokens"][-length:]]


def _action_masks(anchor: dict) -> tuple[list[bool], list[bool]]:
    length = int(anchor["response_length"])
    all_action = [False] * length
    first_action = [False] * length
    for turn in anchor["turns"]:
        start = int(turn["action_token_start"])
        end = int(turn["action_token_end"])
        if start < 0 or end <= start or end > length or any(all_action[start:end]):
            raise ValueError(f"invalid action span {(start, end)}")
        all_action[start:end] = [True] * (end - start)
        first_action[start] = True
    if not any(all_action) or not any(first_action):
        raise ValueError("empty action mask")
    return all_action, first_action


def _mean(values: list[float], mask: list[bool]) -> float:
    selected = [value for value, keep in zip(values, mask) if keep]
    if len(values) != len(mask) or not selected:
        raise ValueError("invalid masked mean")
    return float(sum(selected) / len(selected))


def _weighted_first_action_lift(
    base: list[float], skill: list[float], first_action: list[bool]
) -> tuple[float, float, int]:
    """Uncertainty-weighted action-decision Delta.

    Full entropy and the unconditioned argmax are not returned for SGLang input
    tokens.  The predeclared proxy uses the first content token of every action
    and weights it by 1-p_theta(anchor token | no-skill prefix).
    """

    rows = []
    for base_lp, skill_lp, keep in zip(base, skill, first_action):
        if not keep:
            continue
        weight = 1.0 - math.exp(min(0.0, float(base_lp)))
        rows.append((weight, float(skill_lp) - float(base_lp)))
    denom = sum(weight for weight, _ in rows)
    if not rows:
        raise ValueError("empty first-action mask")
    if denom <= 1e-12:
        return float(sum(delta for _, delta in rows) / len(rows)), 0.0, len(rows)
    return (
        float(sum(weight * delta for weight, delta in rows) / denom),
        float(denom / len(rows)),
        len(rows),
    )


async def _score_anchor(args, tokenizer, anchor: dict, skill: str) -> dict[str, Any]:
    task = str(anchor["task_description"])
    response = _response(anchor)
    action_mask, first_action_mask = _action_masks(anchor)
    base, conditioned = await asyncio.gather(
        _teacher_forced(args, _prompt_ids(args, tokenizer, task, None), response),
        _teacher_forced(args, _prompt_ids(args, tokenizer, task, skill), response),
    )
    high_leverage, mean_weight, decision_count = _weighted_first_action_lift(
        base, conditioned, first_action_mask
    )
    return {
        "all_action_delta": _mean(conditioned, action_mask) - _mean(base, action_mask),
        "high_leverage_delta": high_leverage,
        "high_leverage_mean_weight": mean_weight,
        "high_leverage_decision_count": decision_count,
        "all_action_token_count": int(sum(action_mask)),
    }


async def _score_candidate(args, case_id: str, skill: str) -> dict[str, Any]:
    # Limit only the short critical section that initializes the shared CPU
    # payload.  Model HTTP calls remain fully concurrent.
    async with _ANCHOR_LOCK:
        case = _load_cases()[case_id]
    from slime.rollout.sglang_rollout import GenerateState

    tokenizer = GenerateState(args).tokenizer
    positive_a, positive_b, negative = await asyncio.gather(
        _score_anchor(args, tokenizer, case["positive_a"], skill),
        _score_anchor(args, tokenizer, case["positive_b"], skill),
        _score_anchor(args, tokenizer, case["negative"], skill),
    )
    return {
        "delta_current": positive_a["all_action_delta"],
        "delta_negative_anchor": (
            positive_a["all_action_delta"] - negative["all_action_delta"]
        ),
        "delta_crossfit": positive_b["all_action_delta"],
        "delta_high_leverage": positive_a["high_leverage_delta"],
        "positive_anchor": positive_a,
        "crossfit_anchor": positive_b,
        "negative_anchor": negative,
        "negative_anchor_lambda": 1.0,
        "high_leverage_definition": "action_first_content_token_weighted_by_1_minus_anchor_probability",
    }


async def run_eval(args, sample: Sample, sampling_params: dict) -> Sample:
    label = sample.label if isinstance(sample.label, dict) else {}
    condition = str(label.get("probe_condition") or "")
    skill = str(label.get("probe_skill") or "") if condition == "skill" else ""
    repeat = int(sample.index) % int(getattr(args, "n_samples_per_eval_prompt", 1) or 1)

    score_task = None
    if condition == "skill" and repeat == 0:
        score_task = asyncio.create_task(
            _score_candidate(args, str(label["probe_case_id"]), skill)
        )
    result = await run_episode(
        args,
        sample=deepcopy(sample),
        skill=skill or None,
        sampling_params=sampling_params,
        evaluation=True,
    )
    metadata = ensure_metadata(result)
    won = bool(metadata.get("won", False))
    metadata.update(
        {
            "evaluation_protocol": "static_delta_utility_probe",
            "probe_case_id": str(label["probe_case_id"]),
            "probe_condition": condition,
            "probe_candidate_index": label.get("probe_candidate_index"),
            "probe_repeat": repeat,
            "probe_skill": skill,
            "probe_sampling_seed": sampling_params.get("sampling_seed"),
            "executor_skill_mode": "with_skill" if skill else "no_skill",
            "raw_env_reward": 10.0 * float(won),
        }
    )
    if score_task is not None:
        try:
            metadata["probe_delta_scores"] = await score_task
            metadata["probe_delta_score_error"] = ""
        except Exception as exc:
            metadata["probe_delta_scores"] = {}
            metadata["probe_delta_score_error"] = f"{type(exc).__name__}:{exc}"
    result.reward = 10.0 * float(won)
    return result
