"""Paired teacher-forced card credit for an exact successful ALFWorld scout."""

from __future__ import annotations

import asyncio
import math
import os
from typing import Any

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

from src.alfworld.prompts import format_executor_messages, format_executor_with_skill_messages
from src.alfworld.utils.common import ensure_metadata


_POLICY_ACTION_PHASES = {
    "inspect": "inspect",
    "go": "navigate",
    "take": "take",
    "open": "open",
    "close": "close",
    "clean": "clean",
    "heat": "heat",
    "cool": "cool",
    "toggle": "toggle",
    "place": "place",
}

_ENV_ACTION_PHASES = {
    "look": "inspect",
    "examine": "inspect",
    "go": "navigate",
    "take": "take",
    "open": "open",
    "close": "close",
    "clean": "clean",
    "heat": "heat",
    "cool": "cool",
    "toggle": "toggle",
    "put": "place",
    "move": "place",
}

# Local Delta credit is intentionally limited to state-changing decisions.
# Navigation, acquisition, inspection, and final placement often occur many
# times in one trajectory, so a phase-only mask would credit a generic card on
# unrelated turns instead of the single reusable decision promised by V3.
_CREDITABLE_ACTION_PHASES = {
    "open",
    "close",
    "clean",
    "heat",
    "cool",
    "toggle",
}

_CREDIT_SCOPE_ENV = "ALFWORLD_DELTA_CREDIT_SCOPE"
_CREDIT_SCOPES = {
    "target_action_turn",
    "all_action_tokens",
    "all_reasoning_tokens",
    "all_model_tokens",
}


def _configured_credit_scope() -> str:
    scope = os.environ.get(_CREDIT_SCOPE_ENV, "target_action_turn").strip()
    if scope not in _CREDIT_SCOPES:
        raise ValueError(
            f"unsupported {_CREDIT_SCOPE_ENV}={scope!r}; expected one of {sorted(_CREDIT_SCOPES)}"
        )
    return scope


def _prompt_ids(args, tokenizer, task_description: str, skill: str | None) -> list[int]:
    messages = (
        format_executor_with_skill_messages(task_description, skill)
        if skill
        else format_executor_messages(task_description)
    )
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **(getattr(args, "apply_chat_template_kwargs", None) or {}),
    )
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def _row_logprob(row: Any) -> float:
    if isinstance(row, dict):
        value = row.get("logprob", row.get("log_prob"))
    elif isinstance(row, (list, tuple)) and row:
        value = row[0]
    else:
        value = None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("non-finite input-token logprob")
    return result


async def _teacher_forced(args, prompt_ids: list[int], response_ids: list[int]) -> list[float]:
    total = len(prompt_ids) + len(response_ids)
    max_context = int(getattr(args, "rollout_max_context_len", 0) or 0)
    if max_context and total > max_context:
        raise ValueError(f"teacher-forcing sequence exceeds context: {total}>{max_context}")
    output = await post(
        f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate",
        {
            "input_ids": prompt_ids + response_ids,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 0,
                "skip_special_tokens": False,
            },
            "return_logprob": True,
            "logprob_start_len": 0,
        },
    )
    rows = output.get("meta_info", {}).get("input_token_logprobs")
    if not isinstance(rows, list) or len(rows) < len(response_ids):
        raise ValueError("SGLang did not return aligned input-token logprobs")
    return [_row_logprob(row) for row in rows[-len(response_ids):]]


def _masked_mean(values: list[float], mask: list[bool]) -> float:
    selected = [value for value, keep in zip(values, mask) if keep]
    if len(values) != len(mask) or not selected:
        raise ValueError("invalid Delta model-token mask")
    return float(sum(selected) / len(selected))


def _model_turn_mask(successful_scout: Sample, response_length: int) -> tuple[list[bool], dict]:
    """Rebuild the Delta mask from recorded model-turn token spans.

    The episode response interleaves environment observations and model turns.
    Reconstructing the mask from the model spans makes the no-environment-credit
    contract explicit and independently checks the upstream training loss mask.
    """

    upstream = [bool(value) for value in (successful_scout.loss_mask or [])]
    if len(upstream) != response_length:
        raise ValueError("selected scout loss mask length mismatch")

    turns = ensure_metadata(successful_scout).get("turns")
    if not isinstance(turns, list) or not turns:
        raise ValueError("selected scout has no model-turn spans")

    explicit = [False] * response_length
    for turn in turns:
        if not isinstance(turn, dict):
            raise ValueError("selected scout contains an invalid model-turn span")
        start = int(turn.get("response_token_start", -1))
        end = int(turn.get("response_token_end", -1))
        if start < 0 or end <= start or end > response_length:
            raise ValueError("selected scout contains an out-of-range model-turn span")
        if any(explicit[start:end]):
            raise ValueError("selected scout contains overlapping model-turn spans")
        explicit[start:end] = [True] * (end - start)

    mismatch = sum(left != right for left, right in zip(explicit, upstream))
    if mismatch:
        raise ValueError(f"model-turn spans disagree with loss mask at {mismatch} tokens")
    if not any(explicit):
        raise ValueError("selected scout model-turn mask is empty")
    return explicit, {
        "mask_source": "explicit_model_turn_spans",
        "model_turn_count": len(turns),
        "model_token_count": int(sum(explicit)),
        "environment_token_count": int(response_length - sum(explicit)),
        "loss_mask_mismatch_count": int(mismatch),
    }


def _policy_action_phase(policy_action: str) -> str:
    first = str(policy_action or "").strip().lower().split(maxsplit=1)
    return _POLICY_ACTION_PHASES.get(first[0], "") if first else ""


def _environment_action_phase(action: str) -> str:
    first = str(action or "").strip().lower().split(maxsplit=1)
    return _ENV_ACTION_PHASES.get(first[0], "") if first else ""


def _validate_policy_action_phase(successful_scout: Sample, policy_action: str) -> str:
    target_phase = _policy_action_phase(policy_action)
    if not target_phase:
        raise ValueError("card policy action has no supported action phase")
    if target_phase not in _CREDITABLE_ACTION_PHASES:
        raise ValueError(
            f"card action phase {target_phase!r} is not a local state transition"
        )
    if not any(
        _environment_action_phase(turn.get("action", "")) == target_phase
        for turn in ensure_metadata(successful_scout).get("turns", [])
    ):
        raise ValueError(f"card action phase {target_phase!r} is absent from successful scout")
    return target_phase


def _decision_token_mask(
    successful_scout: Sample,
    response_length: int,
    policy_action: str,
) -> tuple[list[bool], dict]:
    """Score only model turns that execute the card's single action phase."""

    model_mask, audit = _model_turn_mask(successful_scout, response_length)
    target_phase = _validate_policy_action_phase(successful_scout, policy_action)

    decision_mask = [False] * response_length
    matched_turn_count = 0
    for turn in ensure_metadata(successful_scout)["turns"]:
        if _environment_action_phase(turn.get("action", "")) != target_phase:
            continue
        start = int(turn["response_token_start"])
        end = int(turn["response_token_end"])
        decision_mask[start:end] = model_mask[start:end]
        matched_turn_count += 1

    if not any(decision_mask):
        raise ValueError(f"card action phase {target_phase!r} is absent from successful scout")
    scored_environment_token_count = sum(
        keep and not is_model
        for keep, is_model in zip(decision_mask, model_mask)
    )
    if scored_environment_token_count:
        raise ValueError(
            "Delta decision mask includes environment-feedback tokens: "
            f"{scored_environment_token_count}"
        )
    audit.update(
        {
            "credit_scope": "matching_model_action_phase",
            "policy_action_phase": target_phase,
            "scored_turn_count": int(matched_turn_count),
            "scored_token_count": int(sum(decision_mask)),
            "scored_environment_token_count": int(scored_environment_token_count),
        }
    )
    return decision_mask, audit


def _tag_content_mask(
    successful_scout: Sample,
    response_length: int,
    policy_action: str,
    tag: str,
) -> tuple[list[bool], dict]:
    """Score all exact reasoning or action content spans in the trajectory."""

    model_mask, audit = _model_turn_mask(successful_scout, response_length)
    target_phase = _validate_policy_action_phase(successful_scout, policy_action)
    turns = ensure_metadata(successful_scout)["turns"]
    mask = [False] * response_length
    start_key = f"{tag}_token_start"
    end_key = f"{tag}_token_end"
    scored_turn_count = 0
    missing_tag_span_count = 0
    for turn in turns:
        if start_key not in turn or end_key not in turn:
            missing_tag_span_count += 1
            continue
        start = int(turn.get(start_key, -1))
        end = int(turn.get(end_key, -1))
        if start < 0 or end <= start or end > response_length:
            raise ValueError(f"selected scout has invalid {tag} token span")
        if any(mask[start:end]):
            raise ValueError(f"selected scout has overlapping {tag} token spans")
        if not all(model_mask[start:end]):
            raise ValueError(f"selected scout {tag} span includes environment tokens")
        mask[start:end] = [True] * (end - start)
        scored_turn_count += 1
    if not any(mask):
        raise ValueError(f"selected scout {tag} token mask is empty")
    scored_environment_token_count = sum(
        keep and not is_model for keep, is_model in zip(mask, model_mask)
    )
    if scored_environment_token_count:
        raise ValueError(
            f"Delta {tag} mask includes environment-feedback tokens: "
            f"{scored_environment_token_count}"
        )
    audit.update(
        {
            "mask_source": f"explicit_{tag}_content_spans",
            "credit_scope": f"all_{tag}_tokens",
            "policy_action_phase": target_phase,
            "scored_turn_count": int(scored_turn_count),
            "missing_tag_span_count": int(missing_tag_span_count),
            "scored_token_count": int(sum(mask)),
            "scored_environment_token_count": int(scored_environment_token_count),
        }
    )
    return mask, audit


def _credit_token_mask(
    successful_scout: Sample,
    response_length: int,
    policy_action: str,
) -> tuple[list[bool], dict]:
    """Dispatch the single controlled variable for Delta credit scope."""

    scope = _configured_credit_scope()
    if scope == "target_action_turn":
        mask, audit = _decision_token_mask(
            successful_scout,
            response_length,
            policy_action,
        )
        audit["credit_scope"] = scope
        return mask, audit
    if scope == "all_action_tokens":
        return _tag_content_mask(
            successful_scout,
            response_length,
            policy_action,
            "action",
        )
    if scope == "all_reasoning_tokens":
        return _tag_content_mask(
            successful_scout,
            response_length,
            policy_action,
            "reasoning",
        )

    model_mask, audit = _model_turn_mask(successful_scout, response_length)
    target_phase = _validate_policy_action_phase(successful_scout, policy_action)
    audit.update(
        {
            "credit_scope": "all_model_tokens",
            "policy_action_phase": target_phase,
            "scored_turn_count": int(audit["model_turn_count"]),
            "scored_token_count": int(sum(model_mask)),
            "scored_environment_token_count": 0,
        }
    )
    return model_mask, audit


def _empty(reason: str) -> dict:
    return {
        "available": False,
        "skip_reason": reason,
        "base_mean_logprob": None,
        "skill_mean_logprob": None,
        "diff_reward": 0.0,
        "scored_token_count": 0,
        "scored_turn_count": 0,
        "mask_source": "explicit_model_turn_spans",
        "credit_scope": os.environ.get(_CREDIT_SCOPE_ENV, "target_action_turn"),
        "policy_action_phase": "",
        "model_token_count": 0,
        "environment_token_count": 0,
        "scored_environment_token_count": 0,
        "loss_mask_mismatch_count": 0,
    }


async def score_cards(args, successful_scout: Sample, card_samples: list[Sample]) -> list[dict]:
    metadata = ensure_metadata(successful_scout)
    if not bool(metadata.get("won", False)):
        return [_empty("selected_scout_not_successful") for _ in card_samples]
    response_length = int(successful_scout.response_length or 0)
    response_ids = [int(x) for x in successful_scout.tokens[-response_length:]]
    try:
        _model_turn_mask(successful_scout, response_length)
    except Exception as exc:
        return [_empty(f"selected_scout_token_mask_invalid:{exc}") for _ in card_samples]
    task_description = str(metadata.get("task_description") or "").strip()
    tokenizer = GenerateState(args).tokenizer
    base_prompt = _prompt_ids(args, tokenizer, task_description, None)

    async def _score_one(card_sample: Sample) -> dict:
        card_meta = ensure_metadata(card_sample)
        card = str(card_meta.get("skill") or "").strip()
        if float(card_meta.get("format_score", 0.0) or 0.0) < 1.0 or not card:
            return _empty("invalid_card")
        try:
            mask, audit = _credit_token_mask(
                successful_scout,
                response_length,
                str(card_meta.get("skill_policy_action") or ""),
            )
            values = await _teacher_forced(
                args,
                _prompt_ids(args, tokenizer, task_description, card),
                response_ids,
            )
            return {
                "available": True,
                "values": values,
                "mask": mask,
                "skip_reason": "",
                **audit,
            }
        except Exception as exc:
            return _empty(f"with_card_forward_failed:{exc}")

    try:
        outputs = await asyncio.gather(
            _teacher_forced(args, base_prompt, response_ids),
            *[_score_one(sample) for sample in card_samples],
        )
    except Exception as exc:
        return [_empty(f"no_card_forward_failed:{exc}") for _ in card_samples]

    results = []
    for output in outputs[1:]:
        if not output.get("available"):
            results.append(output)
            continue
        mask = output["mask"]
        base_mean = _masked_mean(outputs[0], mask)
        skill_mean = _masked_mean(output["values"], mask)
        results.append(
            {
                "available": True,
                "skip_reason": "",
                "base_mean_logprob": base_mean,
                "skill_mean_logprob": skill_mean,
                "diff_reward": float(skill_mean - base_mean),
                "scored_token_count": int(sum(mask)),
                "scored_turn_count": output["scored_turn_count"],
                "missing_tag_span_count": output.get("missing_tag_span_count", 0),
                "mask_source": output["mask_source"],
                "credit_scope": output["credit_scope"],
                "policy_action_phase": output["policy_action_phase"],
                "model_token_count": output["model_token_count"],
                "environment_token_count": output["environment_token_count"],
                "scored_environment_token_count": output[
                    "scored_environment_token_count"
                ],
                "loss_mask_mismatch_count": output["loss_mask_mismatch_count"],
            }
        )
    return results
