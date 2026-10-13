"""ALFWorld-E7-style group-Delta training adapted to Tau2's three domains."""

from __future__ import annotations

import asyncio
from copy import copy
from functools import partial
import json
import math
import os
from pathlib import Path
import re
from typing import Any

from slime.rollout.filter_hub.base_types import DynamicFilterOutput
from slime.utils.types import Sample

from delta_card_tau.e7_prompts import build_messages, normalize_domain
from delta_card_tau.pipeline import (
    POST_SCOUT_COUNT,
    SCOUT_COUNT,
    EXECUTOR_GROUP_SIZE,
    _failed_source_sample,
    _generator_chat_template_kwargs,
    _inject_skill_into_rendered_prompt,
    _masked_mean,
    _raw_env_success,
    _raw_success,
    _run_executor,
    _run_tau_episode,
    _sampling_for_position,
    _teacher_forced_logprobs,
    _without_skill,
    attach_train_metadata,
    ensure_metadata,
    generate_generator_sample,
    grpo_normalize,
)


MAX_VISIBLE_CONTEXT_TOKENS = int(os.environ.get("TAU2_E7_VISIBLE_CONTEXT_TOKENS", "4800"))
MAX_ATTEMPT_TOKENS = int(os.environ.get("TAU2_E7_ATTEMPT_TOKENS", "2400"))
MAX_GENERATOR_INPUT_TOKENS = int(os.environ.get("TAU2_E7_GENERATOR_INPUT_TOKENS", "15000"))
_FORWARD_LIMIT = int(os.environ.get("TAU2_E7_FORWARD_CONCURRENCY", "20"))
_FORWARD_SEMAPHORE: asyncio.Semaphore | None = None
_WANDB_DEFINED = False


def _forward_semaphore() -> asyncio.Semaphore:
    global _FORWARD_SEMAPHORE
    if _FORWARD_SEMAPHORE is None:
        _FORWARD_SEMAPHORE = asyncio.Semaphore(_FORWARD_LIMIT)
    return _FORWARD_SEMAPHORE


async def _bounded_teacher_forced(args, prompt_ids: list[int], response_ids: list[int]) -> list[float]:
    async with _forward_semaphore():
        return await _teacher_forced_logprobs(args, prompt_ids, response_ids)


def _clip_tokens(tokenizer, text: str, cap: int, *, head_fraction: float = 0.60) -> tuple[str, dict[str, Any]]:
    ids = tokenizer.encode(str(text or ""), add_special_tokens=False)
    original = len(ids)
    if original <= cap:
        return str(text or ""), {"tokens_before_cap": original, "truncated": False}
    head = max(1, int(cap * head_fraction))
    tail = max(1, cap - head)
    clipped = (
        tokenizer.decode(ids[:head], skip_special_tokens=False)
        + "\n[deterministic middle omission]\n"
        + tokenizer.decode(ids[-tail:], skip_special_tokens=False)
    )
    return clipped, {"tokens_before_cap": original, "truncated": True}


def _domain(sample: Sample) -> str:
    metadata = ensure_metadata(sample)
    return normalize_domain(str(metadata.get("base_domain") or metadata.get("domain") or ""))


def _visible_context(sample: Sample, tokenizer) -> tuple[str, dict[str, Any]]:
    # `prompt` is the rendered system policy, tool schema and initial user turn.
    # Never read metadata.task: it contains hidden user instructions and evaluator criteria.
    return _clip_tokens(tokenizer, str(sample.prompt or ""), MAX_VISIBLE_CONTEXT_TOKENS, head_fraction=0.82)


def _compact_attempt(sample: Sample, tokenizer, position: int) -> tuple[str, dict[str, Any]]:
    text, audit = _clip_tokens(tokenizer, str(sample.response or ""), MAX_ATTEMPT_TOKENS)
    metadata = ensure_metadata(sample)
    tools = metadata.get("tool_sequence")
    if isinstance(tools, list) and tools:
        text += "\nObserved tool sequence: " + " -> ".join(str(x) for x in tools)
    audit.update(
        {
            "position": position,
            "outcome": "SUCCESS" if _raw_env_success(sample) > 0.5 else "FAILURE",
            "termination_reason": str(metadata.get("termination_reason") or ""),
        }
    )
    return text, audit


_LEAK_PATTERNS = (
    re.compile(r"\b[A-Z0-9]{6}\b"),
    re.compile(r"\b(?:credit_card|gift_card|payment|order|account|user)_[A-Za-z0-9_-]+\b", re.I),
    re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"),
    re.compile(r"\$\s?\d"),
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
)
_FORBIDDEN = re.compile(r"\b(?:trajector(?:y|ies)|rewards?|delta|demonstrations?|grader|grading)\b", re.I)
_HEADINGS = (
    "# Skill:",
    "## When to apply",
    "## Policy and tool checks",
    "## Procedure",
    "## Decision rules",
    "## Verification and recovery",
    "## Stop or escalation conditions",
)


def _parse_skill(args, response_text: str, *, domain: str = "") -> dict[str, Any]:
    source = str(response_text or "").strip()
    source = re.sub(r"(?:<\|im_end\|>|<\|endoftext\|>)+\s*$", "", source).strip()
    payload = None
    mode = "whole_response_json"
    try:
        payload = json.loads(source)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        candidates = []
        for match in re.finditer(r"\{", source):
            try:
                value, _ = decoder.raw_decode(source[match.start():])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                candidates.append(value)
        if candidates:
            payload = candidates[-1]
            mode = "last_complete_json_object"
    if isinstance(payload, dict) and set(payload) == {"skill"}:
        skill = str(payload.get("skill") or "").strip()
    else:
        cleaned = source
        if cleaned.startswith("```markdown") and cleaned.endswith("```"):
            cleaned = cleaned[len("```markdown"):-3].strip()
            mode = "markdown_fence"
        elif cleaned.startswith("```") and cleaned.endswith("```"):
            cleaned = cleaned[3:-3].strip()
            mode = "generic_fence"
        elif cleaned.startswith('{"skill":"') and cleaned.endswith('"}'):
            # Qwen often emits literal newlines inside the JSON string. Preserve the
            # intended card while still requiring a complete wrapper.
            cleaned = cleaned[len('{"skill":"'):-2].replace('\\"', '"').strip()
            mode = "literal_newline_json_wrapper"
        else:
            mode = "direct_markdown"
        if not cleaned.startswith("# Skill:"):
            return _invalid_skill(args, source, "expected a direct Markdown skill card")
        skill = cleaned
    words = re.findall(r"\b[A-Za-z][A-Za-z'-]*\b", skill)
    positions = [skill.lower().find(h.lower()) for h in _HEADINGS]
    if not 70 <= len(words) <= 400:
        return _invalid_skill(args, source, f"skill word count {len(words)} outside 70..400")
    if any(position < 0 for position in positions) or positions != sorted(positions):
        return _invalid_skill(args, source, "missing or misordered required headings")
    if domain == "telecom":
        procedure_start = positions[3] + len(_HEADINGS[3])
        procedure_end = positions[4]
        procedure = skill[procedure_start:procedure_end]
        bullets = re.findall(r"(?im)^\s*[-*]\s*(.+)$", procedure)
        exact_roles = (
            len(bullets) == 2
            and re.match(r"(?i)^Agent\s*:", bullets[0])
            and re.match(r"(?i)^Ask user\s*:", bullets[1])
        )
        wait_clause = "; wait for the user's reported result before continuing."
        if exact_roles and wait_clause not in bullets[1].lower():
            procedure, repaired = re.subn(
                r"(?im)^(\s*[-*]\s*Ask user\s*:[^\n]*?)\.?\s*$",
                lambda match: match.group(1).rstrip().rstrip(".") + wait_clause,
                procedure,
                count=1,
            )
            if repaired:
                skill = skill[:procedure_start] + procedure + skill[procedure_end:]
                words = re.findall(r"\b[A-Za-z][A-Za-z'-]*\b", skill)
                positions = [skill.lower().find(h.lower()) for h in _HEADINGS]
                bullets = re.findall(r"(?im)^\s*[-*]\s*(.+)$", procedure)
                mode += "+telecom_wait_guardrail"
        agent_device_action = re.search(
            r"(?i)\b(?:sim|apn|network mode|permission|airplane mode|wi-?fi|reboot|restart device)\b",
            bullets[0] if bullets else "",
        )
        user_action = bullets[1] if len(bullets) > 1 else ""
        action_without_wait = user_action.lower().replace(wait_clause, "")
        batched_user_actions = re.search(r"(?i)\b(?:and then|then)\b", action_without_wait)
        if (
            not exact_roles
            or wait_clause not in user_action.lower()
            or agent_device_action
            or batched_user_actions
        ):
            return _invalid_skill(
                args,
                source,
                "telecom procedure violates Agent/Ask user single-action wait contract",
            )
    if _FORBIDDEN.search(skill):
        return _invalid_skill(args, source, "forbidden evaluator or trajectory language")
    if any(pattern.search(skill) for pattern in _LEAK_PATTERNS):
        return _invalid_skill(args, source, "task-specific entity leakage")
    return {
        "skill": skill,
        "format_score": 1.0,
        "format_penalty": 0.0,
        "skill_word_count": len(words),
        "skill_schema": "tau_e7_domain_v1",
        "skill_parse_mode": mode,
        "skill_language_audit_pass": 1.0,
    }


def _invalid_skill(args, source: str, reason: str) -> dict[str, Any]:
    return {
        "skill": "",
        "raw_skill_output": source,
        "format_score": 0.0,
        "format_penalty": -abs(float(getattr(args, "skill_format_penalty", 1.0))),
        "skill_parse_error": reason,
        "skill_schema": "tau_e7_domain_v1",
        "skill_language_audit_pass": 0.0,
    }


async def _generate_skills(args, scouts: list[Sample], sampling_params: dict[str, Any]) -> list[Sample]:
    from slime.rollout.sglang_rollout import GenerateState

    tokenizer = GenerateState(args).tokenizer
    domains = {_domain(scout) for scout in scouts}
    if len(domains) != 1:
        raise ValueError(f"scouts disagree on domain: {domains}")
    domain = next(iter(domains))
    context, context_audit = _visible_context(scouts[0], tokenizer)
    attempts, audits, outcomes = [], [], []
    for position, scout in enumerate(scouts):
        attempt, audit = _compact_attempt(scout, tokenizer, position)
        attempts.append(attempt)
        audits.append(audit)
        outcomes.append(_raw_env_success(scout) > 0.5)
    messages = build_messages(
        domain=domain,
        visible_context=context,
        attempts=attempts,
        outcomes=outcomes,
    )
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **_generator_chat_template_kwargs(args),
    )
    input_tokens = len(tokenizer.encode(prompt_text, add_special_tokens=False))
    if input_tokens > MAX_GENERATOR_INPUT_TOKENS:
        raise ValueError(f"generator input {input_tokens}>{MAX_GENERATOR_INPUT_TOKENS}")

    params = dict(sampling_params)
    params.update({"max_new_tokens": 700, "temperature": 1.0, "top_p": 0.85})
    metadata = ensure_metadata(scouts[0])
    return list(
        await asyncio.gather(
            *[
                generate_generator_sample(
                    args,
                    messages=messages,
                    sampling_params=_sampling_for_position(params, index),
                    n_idx=index,
                    requested_mode=f"tau_e7_{domain}_four_scout",
                    actual_mode=f"tau_e7_{domain}_four_scout",
                    parser_fn=partial(_parse_skill, domain=domain),
                    base_metadata={
                        "training_stage": "experience",
                        "sample_role": "generator",
                        "source_task_id": metadata.get("task_id"),
                        "source_domain": domain,
                        "generator_candidate_index": index,
                        "generator_candidate_count": SCOUT_COUNT,
                        "generator_visible_success_count": int(sum(outcomes)),
                        "generator_input_token_count": input_tokens,
                        "generator_context_audit": context_audit,
                        "generator_attempt_audits": audits,
                        "generator_hidden_metadata_exposed": 0.0,
                    },
                )
                for index in range(SCOUT_COUNT)
            ]
        )
    )


def aggregate_group_delta(lifts: list[float], wins: list[bool]) -> dict[str, Any]:
    if len(lifts) != SCOUT_COUNT or len(wins) != SCOUT_COUNT:
        raise ValueError("Tau2 E7 group Delta requires four lifts and labels")
    if not all(math.isfinite(float(value)) for value in lifts):
        raise ValueError("non-finite group Delta lift")
    success = [float(v) for v, won in zip(lifts, wins) if won]
    failure = [float(v) for v, won in zip(lifts, wins) if not won]
    success_mean = sum(success) / len(success) if success else None
    failure_mean = sum(failure) / len(failure) if failure else None
    if success and failure:
        score = 0.5 * success_mean - 0.5 * failure_mean
        mode = "mixed_equal_class_weight"
    elif success:
        score = success_mean
        mode = "all_success_full_weight"
    else:
        score = -float(failure_mean)
        mode = "all_failure_full_weight"
    return {
        "available": True,
        "diff_reward": float(score),
        "aggregation_mode": mode,
        "success_count": len(success),
        "failure_count": len(failure),
        "success_mean_lift": success_mean,
        "failure_mean_lift": failure_mean,
        "trajectory_lifts": [float(x) for x in lifts],
    }


def _empty_delta(reason: str) -> dict[str, Any]:
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
        "loss_mask_mismatch_count": 0,
    }


async def _score_skills_on_group(args, scouts: list[Sample], skills: list[Sample]) -> list[dict[str, Any]]:
    from slime.rollout.sglang_rollout import GenerateState

    if len(scouts) != SCOUT_COUNT or len(skills) != SCOUT_COUNT:
        return [_empty_delta("incomplete_four_by_four_group") for _ in skills]
    tokenizer = GenerateState(args).tokenizer
    prepared = []
    try:
        for position, scout in enumerate(scouts):
            response_len = int(scout.response_length or 0)
            if response_len <= 0 or not isinstance(scout.tokens, list):
                raise ValueError(f"scout {position} missing response tokens")
            response_ids = [int(x) for x in scout.tokens[-response_len:]]
            base_prompt_ids = [int(x) for x in scout.tokens[:-response_len]]
            mask = [bool(x) for x in (scout.loss_mask or [])]
            if len(mask) != response_len or not any(mask):
                raise ValueError(f"scout {position} invalid assistant loss mask")
            prepared.append(
                {
                    "response_ids": response_ids,
                    "base_prompt_ids": base_prompt_ids,
                    "mask": mask,
                    "won": _raw_env_success(scout) > 0.5,
                    "model_tokens": int(sum(mask)),
                    "environment_tokens": response_len - int(sum(mask)),
                    "prompt": str(scout.prompt or ""),
                }
            )
        base_values = await asyncio.gather(
            *[
                _bounded_teacher_forced(args, row["base_prompt_ids"], row["response_ids"])
                for row in prepared
            ]
        )
        base_means = [
            _masked_mean(values, row["mask"])
            for values, row in zip(base_values, prepared)
        ]
    except Exception as exc:
        return [_empty_delta(f"base_group_forward_failed:{exc}") for _ in skills]

    async def score_one(skill_sample: Sample) -> dict[str, Any]:
        metadata = ensure_metadata(skill_sample)
        skill = str(metadata.get("skill") or "").strip()
        if float(metadata.get("format_score", 0.0) or 0.0) < 1.0 or not skill:
            return _empty_delta("invalid_skill_output")
        try:
            prompt_ids = [
                tokenizer.encode(
                    _inject_skill_into_rendered_prompt(row["prompt"], skill),
                    add_special_tokens=False,
                )
                for row in prepared
            ]
            skill_values = await asyncio.gather(
                *[
                    _bounded_teacher_forced(args, ids, row["response_ids"])
                    for ids, row in zip(prompt_ids, prepared)
                ]
            )
            skill_means = [
                _masked_mean(values, row["mask"])
                for values, row in zip(skill_values, prepared)
            ]
            lifts = [skill_mean - base_mean for skill_mean, base_mean in zip(skill_means, base_means)]
            result = aggregate_group_delta(lifts, [bool(row["won"]) for row in prepared])
            result.update(
                {
                    "skip_reason": "",
                    "base_mean_logprob": sum(base_means) / len(base_means),
                    "skill_mean_logprob": sum(skill_means) / len(skill_means),
                    "scored_token_count": sum(int(row["model_tokens"]) for row in prepared),
                    "model_token_count": sum(int(row["model_tokens"]) for row in prepared),
                    "environment_token_count": sum(int(row["environment_tokens"]) for row in prepared),
                    "loss_mask_mismatch_count": 0,
                }
            )
            return result
        except Exception as exc:
            return _empty_delta(f"skill_group_forward_failed:{exc}")

    return list(await asyncio.gather(*[score_one(skill) for skill in skills]))


def _annotate_and_train_generators(args, skills: list[Sample], scores: list[dict[str, Any]]) -> list[Sample]:
    all_valid = (
        len(skills) == SCOUT_COUNT
        and len(scores) == SCOUT_COUNT
        and all(bool(score.get("available")) for score in scores)
        and all(float(ensure_metadata(skill).get("format_score", 0.0) or 0.0) >= 1.0 for skill in skills)
    )
    for skill, score in zip(skills, scores):
        delta = float(score.get("diff_reward", 0.0) or 0.0)
        metadata = ensure_metadata(skill)
        metadata.update(
            {
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
                "skill_diff_loss_mask_mismatch_count": int(score.get("loss_mask_mismatch_count", 0) or 0),
                "generator_reward_clipped": 0.0,
                "generator_optimizer_included": 1.0 if all_valid else 0.0,
                "raw_training_reward": delta,
            }
        )
        skill.reward = delta
    if not all_valid:
        return []
    grpo_normalize(skills)
    weight = float(args.generator_reward_weight)
    for skill in skills:
        metadata = ensure_metadata(skill)
        metadata["generator_reward_before_weight"] = float(skill.reward or 0.0)
        skill.reward = float(skill.reward or 0.0) * weight
        metadata["reward_weight_applied"] = weight
        metadata["effective_training_reward"] = float(skill.reward)
    attach_train_metadata(skills, sample_role="generator")
    return skills


def _select_positive_skill(skills: list[Sample], scores: list[dict[str, Any]]) -> tuple[str | None, int | None, float | None]:
    eligible = []
    for index, (sample, score) in enumerate(zip(skills, scores)):
        skill = str(ensure_metadata(sample).get("skill") or "").strip()
        delta = float(score.get("diff_reward", 0.0) or 0.0)
        if score.get("available") and skill and math.isfinite(delta) and delta > 0:
            eligible.append((delta, -index, skill, index))
    if not eligible:
        return None, None, None
    delta, _, skill, index = max(eligible)
    return skill, index, delta


def _dump_generator_audit(scouts: list[Sample], skills: list[Sample], scores: list[dict[str, Any]]) -> None:
    """Persist raw generator outputs so smoke prompt failures are diagnosable."""
    dump_root = os.environ.get("TAU2_TRAJECTORY_DUMP_DIR")
    if not dump_root:
        return
    source = ensure_metadata(scouts[0])
    record = {
        "task_id": source.get("task_id"),
        "domain": _domain(scouts[0]),
        "scout_success_count": int(sum(_raw_env_success(x) > 0.5 for x in scouts)),
        "candidates": [],
    }
    for index, (skill_sample, score) in enumerate(zip(skills, scores)):
        metadata = ensure_metadata(skill_sample)
        record["candidates"].append(
            {
                "index": index,
                "format_score": metadata.get("format_score"),
                "parse_error": metadata.get("skill_parse_error"),
                "word_count": metadata.get("skill_word_count"),
                "skill": metadata.get("skill"),
                "raw_output": metadata.get("raw_skill_output", skill_sample.response),
                "delta_available": bool(score.get("available")),
                "delta": score.get("diff_reward") if score.get("available") else None,
                "delta_skip_reason": score.get("skip_reason"),
            }
        )
    path = Path(dump_root) / "e7_generator_audit.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


async def tau_e7_pipeline(
    args,
    sample: Sample,
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> list[Sample] | Sample:
    if evaluation:
        return await _run_tau_episode(args, sample, sampling_params)
    template = _without_skill(sample)
    rollout_id = sample.rollout_id if sample.rollout_id is not None else sample.index
    if rollout_id is None:
        rollout_id = int(sample.group_index or 0)
    try:
        scouts = list(
            await asyncio.gather(
                *[
                    _run_executor(
                        args,
                        template,
                        sampling_params,
                        position=position,
                        skill=None,
                        phase="initial_scout",
                        mode="initial_scout_no_skill",
                    )
                    for position in range(SCOUT_COUNT)
                ]
            )
        )
        skills = await _generate_skills(args, scouts, sampling_params)
        scores = await _score_skills_on_group(args, scouts, skills)
        _dump_generator_audit(scouts, skills, scores)
    except Exception as exc:
        return [_failed_source_sample(template, exc)]

    selected_skill, selected_index, selected_delta = _select_positive_skill(skills, scores)
    mode = "positive_delta_skill" if selected_skill else "no_positive_delta_no_skill"
    try:
        post = list(
            await asyncio.gather(
                *[
                    _run_executor(
                        args,
                        template,
                        sampling_params,
                        position=SCOUT_COUNT + offset,
                        skill=selected_skill,
                        phase="post_scout",
                        mode=mode,
                    )
                    for offset in range(POST_SCOUT_COUNT)
                ]
            )
        )
    except Exception as exc:
        return [_failed_source_sample(template, exc)]

    executors = scouts + post
    group_metadata = {
        "method": "tau_group_delta_e7",
        "rollout_variant": "tau_four_scout_four_skill_group_delta_e7_nokl",
        "initial_scout_success_count": sum(_raw_env_success(x) > 0.5 for x in scouts),
        "scout_outcome_group": (
            "all_wrong" if all(_raw_env_success(x) <= 0.5 for x in scouts)
            else "all_correct" if all(_raw_env_success(x) > 0.5 for x in scouts)
            else "mixed"
        ),
        "generator_candidate_count": len(skills),
        "generator_valid_count": sum(float(ensure_metadata(x).get("format_score", 0.0) or 0.0) >= 1.0 for x in skills),
        "generator_delta_available_count": sum(bool(x.get("available")) for x in scores),
        "generator_positive_delta_count": sum(bool(x.get("available")) and float(x.get("diff_reward", 0.0) or 0.0) > 0 for x in scores),
        "group_delta_scores": [float(x["diff_reward"]) if x.get("available") else None for x in scores],
        "selected_skill_injected": 1.0 if selected_skill else 0.0,
        "selected_skill_candidate_index": selected_index,
        "selected_skill_delta": selected_delta,
        "executor_skill_mode": mode,
    }
    for item in executors + skills:
        item.rollout_id = int(rollout_id)
        item.group_index = sample.group_index
        ensure_metadata(item).update(group_metadata)

    grpo_normalize(executors)
    executor_weight = float(args.executor_reward_weight)
    for executor in executors:
        executor.reward = float(executor.reward or 0.0) * executor_weight
        ensure_metadata(executor)["reward_weight_applied"] = executor_weight
    attach_train_metadata(executors, sample_role="executor")
    optimizer_skills = _annotate_and_train_generators(args, skills, scores)

    print(
        "[tau/e7] group complete "
        f"task_id={ensure_metadata(scouts[0]).get('task_id')} domain={_domain(scouts[0])} "
        f"scout_successes={group_metadata['initial_scout_success_count']} "
        f"skill_valid={group_metadata['generator_valid_count']}/4 "
        f"delta_available={group_metadata['generator_delta_available_count']}/4 "
        f"positive_delta={group_metadata['generator_positive_delta_count']} "
        f"injected={group_metadata['selected_skill_injected']:.0f} "
        f"delta={group_metadata['group_delta_scores']}",
        flush=True,
    )
    return optimizer_skills + executors


def _sample_role(sample: Sample) -> str:
    train = sample.train_metadata if isinstance(sample.train_metadata, dict) else {}
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    return str(train.get("sample_role") or metadata.get("sample_role") or "")


def keep_valid_tau_e7_group(args, group, **kwargs) -> DynamicFilterOutput:
    del args, kwargs
    samples: list[Sample] = []
    stack = list(group if isinstance(group, list) else [group])
    while stack:
        value = stack.pop(0)
        if isinstance(value, list):
            stack[0:0] = value
        else:
            samples.append(value)
    if not samples:
        return DynamicFilterOutput(keep=False, reason="empty_tau_e7_group")
    executors = [sample for sample in samples if _sample_role(sample) == "executor"]
    if len(executors) != EXECUTOR_GROUP_SIZE:
        return DynamicFilterOutput(keep=False, reason=f"expected_8_executors_got_{len(executors)}")
    smoke_domain = os.environ.get("TAU_E7_SMOKE_DOMAIN", "").strip().lower()
    if smoke_domain:
        group_domains = {_domain(sample) for sample in executors}
        if group_domains != {smoke_domain}:
            actual = ",".join(sorted(group_domains)) or "unknown"
            return DynamicFilterOutput(
                keep=False,
                reason=f"smoke_domain_{smoke_domain}_got_{actual}",
            )
    for sample in samples:
        if sample.remove_sample or int(sample.response_length or 0) <= 0:
            reason = ensure_metadata(sample).get("termination_reason", "invalid_sample")
            return DynamicFilterOutput(keep=False, reason=str(reason))
    return DynamicFilterOutput(keep=True, reason=None)


def _prepare_official_metrics(executors: list[Sample]) -> None:
    for sample in executors:
        metadata = ensure_metadata(sample)
        reward = _raw_env_success(sample)
        metadata.update(
            {
                "env_task_reward": reward,
                "raw_reward": reward,
                "shaped_reward": reward,
                "binary_reward": reward,
                "progress_score": reward,
                "behavior_failure": metadata.get("termination_reason") in {"format_error", "max_steps", "too_many_errors"},
            }
        )


def _log_metrics(args, executors: list[Sample], generators: list[Sample]) -> None:
    if not getattr(args, "use_wandb", False) or not executors:
        return
    try:
        import wandb
        from slime.utils.metric_utils import compute_rollout_step
    except ImportError:
        return
    if wandb.run is None:
        return

    global _WANDB_DEFINED
    if not _WANDB_DEFINED:
        wandb.define_metric("tau_e7/*", step_metric="rollout/step")
        _WANDB_DEFINED = True

    def mean(values):
        values = list(values)
        return float(sum(values) / len(values)) if values else 0.0

    scouts = [x for x in executors if ensure_metadata(x).get("executor_rollout_phase") == "initial_scout"]
    post = [x for x in executors if ensure_metadata(x).get("executor_rollout_phase") == "post_scout"]
    available = [x for x in generators if float(ensure_metadata(x).get("skill_diff_available", 0.0) or 0.0) > 0.5]
    deltas = [float(ensure_metadata(x).get("skill_diff_reward", 0.0) or 0.0) for x in available]
    metrics = {
        "tau_e7/scout_success_rate": mean(_raw_env_success(x) for x in scouts),
        "tau_e7/post_scout_success_rate": mean(_raw_env_success(x) for x in post),
        "tau_e7/post_minus_scout_success_rate": mean(_raw_env_success(x) for x in post) - mean(_raw_env_success(x) for x in scouts),
        "tau_e7/skill_injected_group_frac": mean(float(ensure_metadata(x).get("selected_skill_injected", 0.0) or 0.0) for x in scouts),
        "tau_e7/generator_count": float(len(generators)),
        "tau_e7/generator_format_valid_frac": mean(float(ensure_metadata(x).get("format_score", 0.0) or 0.0) >= 1.0 for x in generators),
        "tau_e7/generator_delta_available_frac": len(available) / max(len(generators), 1),
        "tau_e7/generator_raw_delta_mean": mean(deltas),
        "tau_e7/generator_raw_delta_abs_mean": mean(abs(x) for x in deltas),
        "tau_e7/generator_positive_delta_frac": mean(float(x > 0.0) for x in deltas),
        "tau_e7/generator_scored_tokens_mean": mean(float(ensure_metadata(x).get("skill_diff_scored_token_count", 0.0) or 0.0) for x in available),
    }
    for domain in ("airline", "retail", "telecom"):
        ds = [x for x in scouts if _domain(x) == domain]
        dp = [x for x in post if _domain(x) == domain]
        metrics[f"tau_e7/{domain}/scout_success_rate"] = mean(_raw_env_success(x) for x in ds)
        metrics[f"tau_e7/{domain}/post_scout_success_rate"] = mean(_raw_env_success(x) for x in dp)
    rid = executors[0].rollout_id
    if rid is not None:
        metrics["rollout/step"] = compute_rollout_step(args, rid)
    wandb.log(metrics)


def tau_e7_reward_post_process(args, samples):
    flat = [item for group in samples for item in (group if isinstance(group, list) else [group])]
    executors = [item for item in flat if _sample_role(item) == "executor"]
    generators = [item for item in flat if _sample_role(item) == "generator"]
    _prepare_official_metrics(executors)
    if executors:
        from tau2_rl_pipeline.reward import _log_reward_component_metrics

        metric_args = copy(args)
        metric_args.n_samples_per_prompt = EXECUTOR_GROUP_SIZE
        _log_reward_component_metrics(metric_args, executors)
    _log_metrics(args, executors, generators)
    raw = [float(ensure_metadata(x).get("raw_env_reward", ensure_metadata(x).get("raw_training_reward", 0.0)) or 0.0) for x in flat]
    rewards = [float(x.get_reward_value(args)) for x in flat]
    return raw, rewards
