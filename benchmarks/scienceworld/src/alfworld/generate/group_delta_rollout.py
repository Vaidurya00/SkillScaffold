"""Four-scout, four-card contrastive Delta training variants for ALFWorld."""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import math
import os
import random
from copy import deepcopy
from functools import partial
from pathlib import Path
import re

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.types import Sample

from src.alfworld.generate.delta_credit import _model_turn_mask
from src.alfworld.generate.delta_rollout import _is_trainable, _run_executor, _sampling_params_for_position
from src.alfworld.generate.episode import run_episode
from src.alfworld.generate.generator import generate_generator_sample
from src.alfworld.generate.group_delta_credit import score_cards_on_scout_group
from src.alfworld.reward.functions import grpo_normalize
from src.alfworld.utils.common import ensure_metadata, require_arg


SCOUT_COUNT = 4
POST_SCOUT_COUNT = 4
EXECUTOR_GROUP_SIZE = 8
PROMPT_FILE = Path(
    os.environ.get(
        "ALFWORLD_GROUP_GENERATOR_PROMPT_FILE",
        str(Path(__file__).resolve().parents[1] / "group_generator_prompts.py"),
    )
)
VARIANTS = {
    "round_robin",
    "max_signed",
    "positive_gate",
    "positive_gate_trainext_raw",
    "random_one",
}
MAX_TRAJECTORY_TOKENS = 2500
MAX_GENERATOR_INPUT_TOKENS = 15000


def _variant() -> str:
    value = os.environ.get("ALFWORLD_GROUP_DELTA_VARIANT", "positive_gate").strip().lower()
    if value not in VARIANTS:
        raise ValueError(f"unknown ALFWORLD_GROUP_DELTA_VARIANT={value!r}")
    return value


def _load_prompt_module():
    spec = importlib.util.spec_from_file_location("alfworld_generator_prompts_v3_user", PROMPT_FILE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load prompt module: {PROMPT_FILE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("GENERATOR_SYSTEM_PROMPT", "build_generator_user_prompt", "select_example_key"):
        if not hasattr(module, name):
            raise RuntimeError(f"prompt module is missing {name}")
    return module


def _strip_admissible_actions(text: str) -> str:
    marker = "Admissible actions:"
    result: list[str] = []
    position = 0
    while True:
        marker_index = text.find(marker, position)
        if marker_index < 0:
            result.append(text[position:])
            break
        line_start = text.rfind("\n", 0, marker_index) + 1
        result.append(text[position:line_start])
        bracket_start = text.find("[", marker_index)
        if bracket_start < 0:
            line_end = text.find("\n", marker_index)
            if line_end < 0:
                break
            position = line_end + 1
            result.append("[admissible actions omitted]\n")
            continue
        depth = 0
        end = bracket_start
        while end < len(text):
            if text[end] == "[":
                depth += 1
            elif text[end] == "]":
                depth -= 1
                if depth == 0:
                    end += 1
                    break
            end += 1
        if depth:
            result.append(text[marker_index:])
            break
        while end < len(text) and text[end] in " \t\r\n":
            end += 1
        result.append("[admissible actions omitted]\n")
        position = end
    return "\n".join(line.rstrip() for line in "".join(result).splitlines()).strip()


def _compact_trajectory(tokenizer, scout: Sample, ordinal: int) -> tuple[str, dict]:
    cleaned = _strip_admissible_actions(str(scout.response or ""))
    token_ids = tokenizer(cleaned, add_special_tokens=False)["input_ids"]
    original_tokens = len(token_ids)
    truncated = original_tokens > MAX_TRAJECTORY_TOKENS
    if truncated:
        head = int(MAX_TRAJECTORY_TOKENS * 0.58)
        tail = MAX_TRAJECTORY_TOKENS - head
        cleaned = (
            tokenizer.decode(token_ids[:head], skip_special_tokens=True)
            + "\n[deterministic middle truncation]\n"
            + tokenizer.decode(token_ids[-tail:], skip_special_tokens=True)
        )
    won = bool(ensure_metadata(scout).get("won", False))
    outcome = "SUCCESS" if won else "FAILURE"
    rendered = (
        f"<<<BEGIN ATTEMPT {ordinal}; OUTCOME={outcome}>>>\n"
        f"{cleaned}\n"
        f"<<<END ATTEMPT {ordinal}>>>"
    )
    return rendered, {
        "ordinal": ordinal,
        "outcome": outcome,
        "tokens_before_cap": original_tokens,
        "truncated": truncated,
    }


def _parse_skill(args, response_text: str) -> dict:
    del args
    text = str(response_text or "").strip()
    for marker in ("<|im_end|>", "<|endoftext|>"):
        if text.endswith(marker):
            text = text[: -len(marker)].rstrip()
            break
    match = re.fullmatch(r"<skill>\s*(.*?)\s*</skill>", text, flags=re.I | re.S)
    if match is None:
        return {
            "skill": "", "raw_skill_output": text, "format_score": 0.0,
            "format_penalty": -1.0, "skill_parse_error": "expected exactly one skill tag pair",
        }
    skill = match.group(1).strip()
    headings = (
        "# Skill:",
        "## When to apply",
        "## Success-preserving strategy",
        "## Failure patterns to suppress",
        "## Procedure",
        "## Decision rules",
        "## Verification and recovery",
        "## Stop conditions",
    )
    positions = [skill.lower().find(item.lower()) for item in headings]
    word_count = len(re.findall(r"[A-Za-z][A-Za-z'-]*", skill))
    numbered_id = re.search(
        r"\b(?:cabinet|drawer|shelf|countertop|microwave|fridge|sinkbasin|toilet|dresser|desk|safe|object)\s*\d+\b",
        skill,
        flags=re.I,
    )
    forbidden = re.search(r"\b(?:trajectory|trajectories|reward|delta|demonstration)\b", skill, flags=re.I)
    valid = (
        bool(skill)
        and all(position >= 0 for position in positions)
        and positions == sorted(positions)
        and len(set(positions)) == len(positions)
        and word_count <= 400
        and numbered_id is None
        and forbidden is None
    )
    reason = "" if valid else "v3 skill structure, length, or leakage audit failed"
    return {
        "skill": skill if valid else "",
        "raw_skill_output": text,
        "skill_word_count": word_count,
        "skill_prompt_contract": "alfworld_four_trajectory_v3_fewshot",
        "skill_parser_contract": "single_skill_tags_seven_sections_max400",
        "format_score": 1.0 if valid else 0.0,
        "format_penalty": 0.0 if valid else -1.0,
        "skill_parse_error": reason,
        "skill_language_audit_applied": 1.0,
        "skill_language_audit_pass": 1.0 if valid else 0.0,
    }


async def _generate_cards(args, scouts: list[Sample], params: dict) -> list[Sample]:
    prompt_module = _load_prompt_module()
    tokenizer = GenerateState(args).tokenizer
    task_descriptions = {str(ensure_metadata(scout).get("task_description") or "") for scout in scouts}
    if len(task_descriptions) != 1 or not next(iter(task_descriptions), ""):
        raise ValueError("scouts do not share one task description")
    task = next(iter(task_descriptions))
    rendered, audits = [], []
    for index, scout in enumerate(scouts):
        text, audit = _compact_trajectory(tokenizer, scout, index)
        rendered.append(text)
        audits.append(audit)
    labels = [bool(ensure_metadata(scout).get("won", False)) for scout in scouts]
    example_key = prompt_module.select_example_key(labels)
    messages = [
        {"role": "system", "content": prompt_module.GENERATOR_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": prompt_module.build_generator_user_prompt(
                task,
                "\n\n".join(rendered),
                example_keys=(example_key,),
            ),
        },
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **(getattr(args, "apply_chat_template_kwargs", None) or {}),
    )
    input_tokens = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
    if input_tokens > MAX_GENERATOR_INPUT_TOKENS:
        raise ValueError(f"generator input exceeds cap: {input_tokens}>{MAX_GENERATOR_INPUT_TOKENS}")
    card_params = dict(params)
    card_params.update({"max_new_tokens": 900, "temperature": 1.0, "top_p": 0.85})
    generated = await asyncio.gather(
        *[
            generate_generator_sample(
                args,
                messages=messages,
                sampling_params=_sampling_params_for_position(card_params, index),
                n_idx=index,
                requested_mode="four_scout_group_contrastive_v3",
                actual_mode="four_scout_group_contrastive_v3",
                parser_fn=_parse_skill,
                base_metadata={
                    "generator_candidate_index": index,
                    "generator_candidate_count": SCOUT_COUNT,
                    "generator_visible_success_count": int(sum(labels)),
                    "generator_fewshot_key": example_key,
                    "generator_input_token_count": input_tokens,
                    "generator_trajectory_audits": audits,
                },
            )
            for index in range(SCOUT_COUNT)
        ]
    )
    if len(generated) != SCOUT_COUNT:
        raise RuntimeError("generator did not return exactly four candidate samples")
    return list(generated)


def _annotate_card(card: Sample, score: dict, variant: str) -> None:
    metadata = ensure_metadata(card)
    delta = float(score.get("diff_reward", 0.0) or 0.0)
    metadata.update(
        {
            "method": "group_delta_v3",
            "trajectory_role": "generator",
            "group_delta_variant": variant,
            "skill_diff_available": 1.0 if score.get("available") else 0.0,
            "skill_diff_skip_reason": str(score.get("skip_reason") or ""),
            "skill_diff_reward": delta,
            "skill_diff_aggregation_mode": str(score.get("aggregation_mode") or ""),
            "skill_diff_success_count": int(score.get("success_count", 0) or 0),
            "skill_diff_failure_count": int(score.get("failure_count", 0) or 0),
            "skill_diff_success_mean_lift": score.get("success_mean_lift"),
            "skill_diff_failure_mean_lift": score.get("failure_mean_lift"),
            "skill_diff_trajectory_lifts": list(score.get("trajectory_lifts") or []),
            "skill_diff_scored_token_count": int(score.get("scored_token_count", 0) or 0),
            "skill_diff_model_token_count": int(score.get("model_token_count", 0) or 0),
            "skill_diff_environment_token_count": int(score.get("environment_token_count", 0) or 0),
            "skill_diff_scored_environment_token_count": int(
                score.get("scored_environment_token_count", 0) or 0
            ),
            "skill_diff_loss_mask_mismatch_count": int(score.get("loss_mask_mismatch_count", 0) or 0),
            "skill_diff_credit_scope": "all_model_tokens",
            "generator_reward_clipped": 0.0,
            "raw_training_reward": delta,
        }
    )


def _random_selection_seed(args, sample: Sample) -> int:
    """Derive a stable per-rollout, per-group seed without using global RNG state."""
    fields = (
        "group_delta_v3_random_one",
        int(getattr(args, "rollout_seed", 0) or 0),
        int(getattr(args, "current_rollout_id", 0) or 0),
        int(sample.group_index) if isinstance(sample.group_index, int) else 0,
        int(sample.index) if isinstance(sample.index, int) else 0,
    )
    digest = hashlib.sha256("|".join(map(str, fields)).encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _select_routes(
    variant: str,
    cards: list[Sample],
    scores: list[dict],
    *,
    random_seed: int | None = None,
) -> tuple[list[str | None], dict]:
    eligible = []
    for index, (card, score) in enumerate(zip(cards, scores)):
        metadata = ensure_metadata(card)
        delta = float(score.get("diff_reward", 0.0) or 0.0)
        if (
            bool(score.get("available"))
            and float(metadata.get("format_score", 0.0) or 0.0) >= 1.0
            and bool(metadata.get("skill"))
            and math.isfinite(delta)
        ):
            eligible.append((delta, index, str(metadata["skill"])))

    if variant == "round_robin":
        routes = [
            str(ensure_metadata(card).get("skill") or "") or None
            if any(item[1] == index for item in eligible)
            else None
            for index, card in enumerate(cards)
        ]
        return routes, {
            "selected_skill_candidate_index": None,
            "selected_skill_delta": None,
            "executor_skill_mode": "round_robin_four_cards",
        }

    if variant == "random_one":
        if random_seed is None:
            raise ValueError("random_one requires a deterministic random_seed")
        selected = random.Random(random_seed).choice(eligible) if eligible else None
    else:
        selected = max(eligible, key=lambda item: (item[0], -item[1])) if eligible else None
    if variant in {"positive_gate", "positive_gate_trainext_raw"} and (
        selected is None or selected[0] <= 0
    ):
        selected = None
    skill = selected[2] if selected else None
    selection = {
        "selected_skill_candidate_index": selected[1] if selected else None,
        "selected_skill_delta": selected[0] if selected else None,
        "executor_skill_mode": (
            "max_signed_skill"
            if variant == "max_signed" and selected
            else "random_one_skill"
            if variant == "random_one" and selected
            else "positive_delta_skill"
            if selected
            else "no_skill_fallback"
        ),
    }
    if variant == "random_one":
        selection.update(
            {
                "random_selection_pool_size": len(eligible),
                "random_selection_seed": random_seed,
            }
        )
    return [skill] * POST_SCOUT_COUNT, selection


async def run_rollout(args, sample: Sample, sampling_params: dict, evaluation: bool = False) -> list[Sample]:
    if evaluation:
        raise ValueError("Use the dedicated ALFWorld evaluator for evaluation")
    if int(require_arg(args, "n_experiences")) != SCOUT_COUNT:
        raise ValueError("group Delta requires n_experiences=4")
    variant = _variant()
    executor_params = dict(sampling_params)
    executor_params["temperature"] = float(require_arg(args, "executor_temperature"))
    if getattr(args, "executor_top_p", None) is not None:
        executor_params["top_p"] = float(args.executor_top_p)
    template = deepcopy(sample)

    try:
        scouts = list(
            await asyncio.gather(
                *[_run_executor(args, template, executor_params, index, None) for index in range(SCOUT_COUNT)]
            )
        )
        cards = await _generate_cards(args, scouts, sampling_params)
        scores = await score_cards_on_scout_group(args, scouts, cards)
    except Exception as exc:
        print(f"[alfworld/group_delta] dropping incomplete pre-scout group: {exc}")
        return []

    for card, score in zip(cards, scores):
        _annotate_card(card, score, variant)

    random_seed = _random_selection_seed(args, sample) if variant == "random_one" else None
    routes, selection = _select_routes(variant, cards, scores, random_seed=random_seed)
    try:
        post = list(
            await asyncio.gather(
                *[
                    _run_executor(args, template, executor_params, SCOUT_COUNT + index, skill)
                    for index, skill in enumerate(routes)
                ]
            )
        )
    except Exception as exc:
        print(f"[alfworld/group_delta] dropping incomplete post-scout group: {exc}")
        return []

    executors = scouts + post
    group_metadata = {
        "method": "group_delta_v3",
        "rollout_variant": "four_scout_four_card_group_balanced_delta_v3",
        "group_delta_variant": variant,
        "initial_scout_success_count": sum(bool(ensure_metadata(item).get("won", False)) for item in scouts),
        "generator_candidate_count": len(cards),
        "generator_valid_count": sum(bool(ensure_metadata(item).get("skill")) for item in cards),
        "generator_available_delta_count": sum(bool(item.get("available")) for item in scores),
        "generator_positive_delta_count": sum(
            bool(item.get("available")) and float(item.get("diff_reward", 0.0) or 0.0) > 0
            for item in scores
        ),
        "group_delta_scores": [
            float(item["diff_reward"]) if item.get("available") else None for item in scores
        ],
        "group_delta_aggregation_modes": [str(item.get("aggregation_mode") or "") for item in scores],
        "group_delta_success_mean_lifts": [item.get("success_mean_lift") for item in scores],
        "group_delta_failure_mean_lifts": [item.get("failure_mean_lift") for item in scores],
        "group_delta_scored_environment_token_count": int(
            sum(int(item.get("scored_environment_token_count", 0) or 0) for item in scores)
        ),
        "group_delta_scored_token_count": int(
            sum(int(item.get("scored_token_count", 0) or 0) for item in scores)
        ),
        "group_delta_model_token_count": int(
            sum(int(item.get("model_token_count", 0) or 0) for item in scores)
        ),
        "group_delta_environment_token_count": int(
            sum(int(item.get("environment_token_count", 0) or 0) for item in scores)
        ),
        "group_delta_loss_mask_mismatch_count": int(
            sum(int(item.get("loss_mask_mismatch_count", 0) or 0) for item in scores)
        ),
        "selected_skill_injected": 1.0 if any(routes) else 0.0,
        **selection,
    }
    for position, executor in enumerate(executors):
        metadata = ensure_metadata(executor)
        raw_reward = 10.0 * float(bool(metadata.get("won", False)))
        metadata.update(group_metadata)
        metadata.update(
            {
                "trajectory_role": "executor",
                "executor_group_position": position,
                "executor_group_size": EXECUTOR_GROUP_SIZE,
                "executor_phase": "scout" if position < SCOUT_COUNT else "post_scout",
                "executor_route_candidate_index": (
                    position - SCOUT_COUNT
                    if variant == "round_robin" and position >= SCOUT_COUNT
                    else selection.get("selected_skill_candidate_index")
                    if position >= SCOUT_COUNT
                    else None
                ),
                "executor_route_skill_injected": 1.0
                if position >= SCOUT_COUNT and routes[position - SCOUT_COUNT]
                else 0.0,
                "executor_route_delta": (
                    float(scores[position - SCOUT_COUNT]["diff_reward"])
                    if variant == "round_robin"
                    and position >= SCOUT_COUNT
                    and scores[position - SCOUT_COUNT].get("available")
                    else selection.get("selected_skill_delta")
                    if position >= SCOUT_COUNT
                    else None
                ),
                "raw_env_reward": raw_reward,
                "raw_training_reward": raw_reward,
            }
        )
        executor.reward = raw_reward
    grpo_normalize(executors)

    if variant != "positive_gate_trainext_raw":
        return executors

    all_delta_valid = (
        len(cards) == SCOUT_COUNT
        and len(scores) == SCOUT_COUNT
        and all(_is_trainable(card) for card in cards)
        and all(bool(score.get("available")) for score in scores)
        and all(bool(ensure_metadata(card).get("skill")) for card in cards)
    )
    if not all_delta_valid:
        return executors
    source_group = sample.group_index if isinstance(sample.group_index, int) else 0
    for position, (card, score) in enumerate(zip(cards, scores)):
        card.group_index = -(int(source_group) + 1)
        card.index = position
        card.reward = float(score["diff_reward"])
        metadata = ensure_metadata(card)
        metadata["raw_training_reward"] = float(score["diff_reward"])
        train_metadata = dict(card.train_metadata or {})
        train_metadata.update({"training_stage": "generator", "sample_role": "generator"})
        card.train_metadata = train_metadata
    grpo_normalize(cards)
    weight = float(require_arg(args, "generator_reward_weight"))
    for card in cards:
        card.reward = float(card.reward or 0.0) * weight
        ensure_metadata(card)["reward_weight_applied"] = weight
    return cards + executors
