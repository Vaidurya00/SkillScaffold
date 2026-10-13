"""Four-scout Delta-card training pipeline for Tau2."""

from __future__ import annotations

import asyncio
from copy import copy
import json
import math
import os
import re
from typing import Any

from slime.utils.http_utils import post
from slime.rollout.filter_hub.base_types import DynamicFilterOutput
from slime.utils.types import Sample

from delta_card_tau.prompts import format_tau_delta_card_messages


SCOUT_COUNT = 4
POST_SCOUT_COUNT = 4
EXECUTOR_GROUP_SIZE = SCOUT_COUNT + POST_SCOUT_COUNT
_DELTA_WANDB_METRICS_DEFINED = False
SKILL_BLOCK = """

## Retrieved reusable skill
Use the following learned skill as guidance when it applies. The environment
policy and tool schemas remain authoritative; ignore conflicting or
task-specific claims in the skill.

{skill}"""


def ensure_metadata(sample: Sample) -> dict[str, Any]:
    if not isinstance(sample.metadata, dict):
        sample.metadata = {}
    return sample.metadata


def _generator_chat_template_kwargs(args) -> dict[str, Any]:
    """Keep Tau skill generation in the default no-native-thinking mode."""

    kwargs = dict(getattr(args, "apply_chat_template_kwargs", None) or {})
    kwargs["enable_thinking"] = False
    kwargs.pop("preserve_thinking", None)
    return kwargs


def _ensure_train_placeholder_sample(sample: Sample, *, reason: str) -> Sample:
    tokens = list(sample.tokens or [])
    response_length = int(sample.response_length or 0)
    if response_length >= 1 and len(tokens) - response_length >= 1:
        return sample
    metadata = ensure_metadata(sample)
    metadata.update({"train_placeholder": 1.0, "train_placeholder_reason": reason})
    sample.remove_sample = True
    response_length = max(1, response_length)
    tokens = ([0] * max(0, response_length + 1 - len(tokens))) + tokens
    sample.tokens = tokens
    sample.response_length = response_length
    sample.loss_mask = [0] * response_length
    sample.rollout_log_probs = [0.0] * response_length
    sample.reward = float(sample.reward or 0.0)
    return sample


def _task_description(metadata: dict[str, Any]) -> str:
    task = metadata.get("task")
    if isinstance(task, dict):
        scenario = task.get("user_scenario")
        if isinstance(scenario, dict):
            instructions = scenario.get("instructions")
            if isinstance(instructions, dict):
                value = instructions.get("task_instructions")
                if isinstance(value, str) and value.strip():
                    return value.strip()
        description = task.get("description")
        if isinstance(description, dict):
            value = description.get("purpose")
            if isinstance(value, str) and value.strip():
                return value.strip()
    value = metadata.get("initial_user_message")
    return value.strip() if isinstance(value, str) and value.strip() else str(metadata.get("task_id") or "Tau2 customer-service task")


def _failed_source_sample(source: Sample, exc: Exception) -> Sample:
    failed = Sample(
        group_index=source.group_index,
        index=source.index,
        rollout_id=source.rollout_id,
        status=Sample.Status.FAILED,
    )
    failed.prompt = source.prompt
    failed.label = dict(source.label) if isinstance(source.label, dict) else source.label
    failed.reward = 0.0
    failed.tokens = []
    failed.loss_mask = []
    failed.response = ""
    failed.response_length = 0
    failed.metadata = dict(source.metadata) if isinstance(source.metadata, dict) else {}
    failed.metadata.update({
        "training_stage": "experience",
        "sample_role": "generator",
        "source_episode_error": f"{type(exc).__name__}: {exc}",
    })
    return _ensure_train_placeholder_sample(failed, reason="tau_source_episode_error")


async def _run_tau_episode(args, sample: Sample, sampling_params: dict[str, Any]) -> Sample:
    from tau2_rl_pipeline.rollout import generate as tau_generate

    timeout_seconds = float(os.environ.get("TAU2_EPISODE_TIMEOUT_SECONDS", "0"))
    episode = tau_generate(args, sample, sampling_params)
    if timeout_seconds <= 0:
        return await episode
    try:
        return await asyncio.wait_for(episode, timeout=timeout_seconds)
    except asyncio.TimeoutError as exc:
        raise TimeoutError(f"Tau2 episode exceeded {timeout_seconds:.1f}s wall-clock limit") from exc


async def generate_generator_sample(
    args,
    *,
    messages: list[dict],
    sampling_params: dict,
    n_idx: int,
    requested_mode: str,
    actual_mode: str,
    parser_fn,
    base_metadata: dict | None = None,
) -> Sample:
    from slime.rollout.sglang_rollout import GenerateState

    tokenizer = GenerateState(args).tokenizer
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **_generator_chat_template_kwargs(args),
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    try:
        output = await post(
            f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate",
            {"text": prompt_text, "sampling_params": sampling_params, "return_logprob": True},
        )
    except Exception as exc:
        failed = Sample(group_index=0, index=n_idx, status=Sample.Status.FAILED)
        failed.reward = 0.0
        failed.metadata = dict(base_metadata or {})
        failed.metadata.update({
            "skill": "",
            "format_score": 0.0,
            "format_penalty": 0.0,
            "skill_parse_error": f"generator_post_failed:{exc}",
            "requested_skill_source_mode": requested_mode,
            "skill_source_mode": actual_mode,
        })
        return _ensure_train_placeholder_sample(failed, reason="tau_generator_post_failed")

    meta_info = output.get("meta_info") or {}
    finish_type = ((meta_info.get("finish_reason") or {}).get("type") or "stop")
    if finish_type == "abort":
        failed = Sample(group_index=0, index=n_idx, status=Sample.Status.ABORTED)
        failed.reward = 0.0
        failed.metadata = dict(base_metadata or {})
        failed.metadata.update({
            "skill": "",
            "format_score": 0.0,
            "format_penalty": 0.0,
            "skill_parse_error": "generator_aborted",
            "requested_skill_source_mode": requested_mode,
            "skill_source_mode": actual_mode,
        })
        return _ensure_train_placeholder_sample(failed, reason="tau_generator_aborted")

    response_text = str(output.get("text") or "")
    token_logprobs = meta_info.get("output_token_logprobs") or []
    if token_logprobs:
        response_ids = [int(row[1]) for row in token_logprobs]
        response_logprobs = [float(row[0]) for row in token_logprobs]
    else:
        response_ids = tokenizer(response_text, add_special_tokens=False)["input_ids"]
        response_logprobs = [0.0] * len(response_ids)
    parsed = dict(base_metadata or {})
    parsed.update(parser_fn(args, response_text))
    parsed.update({"requested_skill_source_mode": requested_mode, "skill_source_mode": actual_mode})

    generated = Sample(group_index=0, index=n_idx)
    generated.prompt = prompt_text
    generated.tokens = list(prompt_ids) + list(response_ids)
    generated.response = response_text
    generated.response_length = len(response_ids)
    generated.loss_mask = [1] * len(response_ids)
    generated.rollout_log_probs = response_logprobs
    generated.status = Sample.Status.TRUNCATED if finish_type == "length" else Sample.Status.COMPLETED
    generated.metadata = parsed
    return _ensure_train_placeholder_sample(generated, reason="tau_generator_generated")


def grpo_normalize(samples: list[Sample]) -> None:
    if len(samples) <= 1:
        return
    rewards = [float(sample.reward or 0.0) for sample in samples]
    mean = sum(rewards) / len(rewards)
    std = (sum((reward - mean) ** 2 for reward in rewards) / len(rewards)) ** 0.5
    for sample, reward in zip(samples, rewards):
        sample.reward = (reward - mean) / (std + 1e-8)


def apply_reward_weights(args, *, generator_samples: list[Sample], downstream_samples: list[Sample]) -> None:
    for sample in generator_samples:
        if isinstance(sample.reward, (int, float)):
            sample.reward *= float(args.generator_reward_weight)
            ensure_metadata(sample)["reward_weight_applied"] = float(args.generator_reward_weight)
    for sample in downstream_samples:
        if isinstance(sample.reward, (int, float)):
            sample.reward *= float(args.executor_reward_weight)
            ensure_metadata(sample)["reward_weight_applied"] = float(args.executor_reward_weight)


def attach_train_metadata(samples: list[Sample], *, sample_role: str) -> None:
    for sample in samples:
        metadata = dict(sample.train_metadata or {})
        metadata.update({"training_stage": "experience", "sample_role": sample_role})
        sample.train_metadata = metadata


def keep_valid_tau_delta_group(args, group, **kwargs) -> DynamicFilterOutput:
    """Validate compact Delta-card groups without changing Tau rewards.

    Slime wraps the custom pipeline result once for ``n_samples_per_prompt``;
    Delta-card itself returns a list of generator and executor samples.  Flatten
    only that wrapper, then apply the same infra/zero-token validity rule as
    Tau's stock filter.  Content failures with sampled tokens stay as negative
    training examples.
    """
    samples: list[Sample] = []
    stack = list(group if isinstance(group, list) else [group])
    while stack:
        value = stack.pop(0)
        if isinstance(value, list):
            stack[0:0] = value
        else:
            samples.append(value)
    if not samples:
        return DynamicFilterOutput(keep=False, reason="empty_delta_card_group")
    for sample in samples:
        if sample.remove_sample or int(sample.response_length or 0) <= 0:
            reason = (sample.metadata or {}).get("termination_reason", "invalid_sample")
            return DynamicFilterOutput(keep=False, reason=str(reason))
    return DynamicFilterOutput(keep=True, reason=None)


def parse_tau_delta_card(args, response_text: str) -> dict[str, Any]:
    source = str(response_text or "").strip()
    payload = None
    parse_mode = "whole_response_json"
    try:
        payload = json.loads(source)
    except json.JSONDecodeError:
        # Qwen may retain a hidden-thought wrapper even when thinking is
        # disabled, followed by the final JSON answer.  Deterministically take
        # the final complete JSON object; schema and content checks below stay
        # unchanged.
        decoder = json.JSONDecoder()
        candidates = []
        for match in re.finditer(r"\{", source):
            try:
                value, _ = decoder.raw_decode(source[match.start():])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                candidates.append(value)
        if not candidates:
            return _invalid_skill(args, source, "No complete JSON object found")
        payload = candidates[-1]
        parse_mode = "last_complete_json_object"
    if not isinstance(payload, dict) or set(payload) != {"skill"}:
        return _invalid_skill(args, source, "Expected exactly one JSON key: skill")
    skill = str(payload.get("skill") or "").strip()
    words = re.findall(r"\b[A-Za-z0-9][A-Za-z0-9'-]*\b", skill)
    if not skill or not 20 <= len(words) <= 220:
        return _invalid_skill(args, source, f"Skill word count out of range: {len(words)}")
    return {
        "skill": skill,
        "format_score": 1.0,
        "format_penalty": 0.0,
        "skill_word_count": len(words),
        "skill_schema": "tau_delta_card_v1",
        "skill_parse_mode": parse_mode,
    }


def _invalid_skill(args, source: str, error: str) -> dict[str, Any]:
    return {
        "skill": "",
        "raw_skill_output": source,
        "format_score": 0.0,
        "format_penalty": -abs(float(getattr(args, "skill_format_penalty", 1.0))),
        "skill_parse_error": error,
        "skill_schema": "tau_delta_card_v1",
    }


def _raw_success(sample: Sample) -> float:
    if sample.remove_sample:
        return 0.0
    return 1.0 if isinstance(sample.reward, (int, float)) and float(sample.reward) >= 1.0 else 0.0


def _raw_env_success(sample: Sample) -> float:
    """Return the immutable pre-normalization Tau task success label.

    Delta-card normalizes ``sample.reward`` within each eight-executor group for
    optimization, so dashboard metrics must never infer task success from the
    normalized value.  ``raw_env_reward`` is written immediately after each
    Tau episode and remains the source of truth thereafter.
    """
    if sample.remove_sample:
        return 0.0
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    value = metadata.get("raw_env_reward")
    if isinstance(value, (int, float)):
        return 1.0 if float(value) >= 1.0 else 0.0
    # Compatibility fallback for synthetic/unit-test samples that have not yet
    # passed through _run_executor. Production executor samples always take the
    # immutable metadata branch above.
    return _raw_success(sample)


def _sampling_for_position(sampling_params: dict[str, Any], position: int) -> dict[str, Any]:
    params = dict(sampling_params)
    if params.get("sampling_seed") is not None:
        params["sampling_seed"] = int(params["sampling_seed"]) + int(position)
    return params


def _clone_sample_for_tau_episode(sample: Sample) -> Sample:
    """Clone only mutable rollout state, not Tau's large nested task payload.

    Tau rollout replaces prompt/tokens/response and only updates metadata at the
    top level.  Deep-copying the nested task/environment record is both
    unnecessary and extremely expensive on the shared filesystem.
    """
    value = copy(sample)
    value.tokens = list(sample.tokens or [])
    value.weight_versions = list(sample.weight_versions or [])
    value.loss_mask = list(sample.loss_mask) if sample.loss_mask is not None else None
    value.rollout_log_probs = (
        list(sample.rollout_log_probs) if sample.rollout_log_probs is not None else None
    )
    value.label = dict(sample.label) if isinstance(sample.label, dict) else sample.label
    value.metadata = dict(sample.metadata) if isinstance(sample.metadata, dict) else {}
    value.spec_info = copy(sample.spec_info)
    value.prefix_cache_info = copy(sample.prefix_cache_info)
    return value


def _without_skill(sample: Sample) -> Sample:
    value = _clone_sample_for_tau_episode(sample)
    if isinstance(value.label, dict):
        value.label.pop("evolving_skill", None)
    if isinstance(value.metadata, dict):
        value.metadata.pop("evolving_skill", None)
    return value


async def _run_executor(
    args,
    template: Sample,
    sampling_params: dict[str, Any],
    *,
    position: int,
    skill: str | None,
    phase: str,
    mode: str,
) -> Sample:
    sample = _without_skill(template)
    sample.index = position
    if skill:
        if not isinstance(sample.label, dict):
            sample.label = {}
        sample.label["evolving_skill"] = skill
        sample.metadata["evolving_skill"] = skill
    result = await _run_tau_episode(
        args,
        sample,
        _sampling_for_position(sampling_params, position),
    )
    metadata = ensure_metadata(result)
    raw_reward = _raw_success(result)
    metadata.update({
        "training_stage": "experience",
        "sample_role": "executor",
        "executor_rollout_phase": phase,
        "executor_group_position": position,
        "executor_group_size": EXECUTOR_GROUP_SIZE,
        "executor_skill_mode": mode,
        "raw_env_reward": raw_reward,
        "raw_training_reward": raw_reward,
        "delta_card_tau": 1.0,
    })
    result.reward = raw_reward
    return result


def _inject_skill_into_rendered_prompt(prompt: str, skill: str) -> str:
    marker = "<|im_end|>"
    start = prompt.find("<|im_start|>system")
    end = prompt.find(marker, max(start, 0))
    if start < 0 or end < 0:
        raise ValueError("Tau prompt has no Qwen3 system-message boundary")
    return prompt[:end] + SKILL_BLOCK.format(skill=skill.strip()) + prompt[end:]


def _input_logprob(row: Any) -> float:
    value = row.get("logprob", row.get("log_prob")) if isinstance(row, dict) else row[0]
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("non-finite input logprob")
    return value


async def _teacher_forced_logprobs(args, prompt_ids: list[int], response_ids: list[int]) -> list[float]:
    if len(prompt_ids) + len(response_ids) > int(getattr(args, "rollout_max_context_len", 0) or 10**9):
        raise ValueError("teacher-forced Tau trajectory exceeds rollout context")
    output = await post(
        f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate",
        {
            "input_ids": prompt_ids + response_ids,
            "sampling_params": {"temperature": 0, "max_new_tokens": 0, "skip_special_tokens": False},
            "return_logprob": True,
            "logprob_start_len": 0,
        },
    )
    rows = (output.get("meta_info") or {}).get("input_token_logprobs")
    if not isinstance(rows, list) or len(rows) < len(response_ids):
        raise ValueError("missing or short teacher-forced input logprobs")
    return [_input_logprob(row) for row in rows[-len(response_ids):]]


def _masked_mean(values: list[float], mask: list[bool]) -> float:
    selected = [value for value, keep in zip(values, mask) if keep]
    if len(values) != len(mask) or not selected:
        raise ValueError("invalid Tau response logprob mask")
    return float(sum(selected) / len(selected))


async def _score_skills(args, source: Sample, generators: list[Sample]) -> list[dict[str, Any]]:
    response_len = int(source.response_length or 0)
    if response_len <= 0 or not isinstance(source.tokens, list):
        return [_empty_delta("source_has_no_response_tokens") for _ in generators]
    response_ids = [int(x) for x in source.tokens[-response_len:]]
    base_prompt_ids = [int(x) for x in source.tokens[:-response_len]]
    mask = [bool(x) for x in (source.loss_mask or [])]
    if len(mask) != response_len or not any(mask):
        return [_empty_delta("source_has_no_trainable_assistant_tokens") for _ in generators]

    from slime.rollout.sglang_rollout import GenerateState
    tokenizer = GenerateState(args).tokenizer

    async def score_one(generator: Sample) -> dict[str, Any]:
        metadata = ensure_metadata(generator)
        skill = str(metadata.get("skill") or "").strip()
        if float(metadata.get("format_score", 0.0) or 0.0) < 1.0 or not skill:
            return _empty_delta("invalid_skill_output")
        try:
            rendered = _inject_skill_into_rendered_prompt(str(source.prompt or ""), skill)
            skill_prompt_ids = tokenizer.encode(rendered, add_special_tokens=False)
            values = await _teacher_forced_logprobs(args, skill_prompt_ids, response_ids)
            return {"available": True, "skill_values": values, "skip_reason": ""}
        except Exception as exc:
            return _empty_delta(f"skill_forward_failed:{exc}")

    try:
        outputs = await asyncio.gather(
            _teacher_forced_logprobs(args, base_prompt_ids, response_ids),
            *[score_one(generator) for generator in generators],
        )
        base_mean = _masked_mean(outputs[0], mask)
    except Exception as exc:
        return [_empty_delta(f"base_forward_failed:{exc}") for _ in generators]

    results = []
    for output in outputs[1:]:
        if not output.get("available"):
            results.append(output)
            continue
        skill_mean = _masked_mean(output["skill_values"], mask)
        results.append({
            "available": True,
            "skip_reason": "",
            "base_mean_logprob": base_mean,
            "skill_mean_logprob": skill_mean,
            "diff_reward": float(skill_mean - base_mean),
            "scored_token_count": sum(mask),
        })
    return results


def _empty_delta(reason: str) -> dict[str, Any]:
    return {"available": False, "skip_reason": reason, "diff_reward": 0.0, "scored_token_count": 0}


def _apply_delta_rewards(generators: list[Sample], results: list[dict[str, Any]]) -> None:
    for generator, result in zip(generators, results):
        metadata = ensure_metadata(generator)
        raw_delta = float(result.get("diff_reward", 0.0) or 0.0)
        valid = bool(result.get("available")) and float(metadata.get("format_score", 0.0) or 0.0) >= 1.0 and math.isfinite(raw_delta)
        reward = max(0.0, raw_delta) if valid else 0.0
        generator.reward = reward
        metadata.update({
            "generator_reward_source": "same_successful_tau_scout_logprob_delta",
            "skill_diff_available": 1.0 if result.get("available") else 0.0,
            "skill_diff_skip_reason": str(result.get("skip_reason") or ""),
            "skill_diff_base_mean_logprob": result.get("base_mean_logprob"),
            "skill_diff_with_skill_mean_logprob": result.get("skill_mean_logprob"),
            "skill_diff_reward": raw_delta,
            "skill_diff_scored_token_count": int(result.get("scored_token_count", 0) or 0),
            "generator_reward_after_null_clip": reward,
            "raw_training_reward": reward,
        })


def _select_positive_skill(generators: list[Sample], results: list[dict[str, Any]]) -> tuple[str | None, int | None, float | None]:
    candidates = []
    for index, (generator, result) in enumerate(zip(generators, results)):
        skill = str(ensure_metadata(generator).get("skill") or "").strip()
        delta = float(result.get("diff_reward", 0.0) or 0.0)
        if result.get("available") and skill and math.isfinite(delta) and delta > 0:
            candidates.append((delta, -index, skill, index))
    if not candidates:
        return None, None, None
    delta, _, skill, index = max(candidates)
    return skill, index, delta


async def tau_delta_card_pipeline(
    args,
    sample: Sample,
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> list[Sample] | Sample:
    if evaluation:
        return await _run_tau_episode(args, sample, sampling_params)

    template = _without_skill(sample)
    compact_rollout_id = sample.rollout_id if sample.rollout_id is not None else sample.index
    if compact_rollout_id is None:
        compact_rollout_id = int(sample.group_index or 0)
    try:
        scouts = list(await asyncio.gather(*[
            _run_executor(args, template, sampling_params, position=position, skill=None,
                        phase="initial_scout", mode="initial_scout_no_skill")
            for position in range(SCOUT_COUNT)
        ]))
    except Exception as exc:
        return [_failed_source_sample(template, exc)]

    successful = next((scout for scout in scouts if _raw_success(scout) > 0.5), None)
    generators: list[Sample] = []
    results: list[dict[str, Any]] = []
    selected_skill = None
    selected_index = None
    selected_delta = None

    if successful is not None:
        metadata = ensure_metadata(successful)
        task_description = _task_description(metadata)
        generators = list(await asyncio.gather(*[
            generate_generator_sample(
                args,
                messages=format_tau_delta_card_messages(task_description, str(successful.response or ""), 1.0),
                sampling_params=_sampling_for_position(sampling_params, index),
                n_idx=index,
                requested_mode="tau_delta_card",
                actual_mode="tau_delta_card",
                parser_fn=parse_tau_delta_card,
                base_metadata={
                    "training_stage": "experience",
                    "sample_role": "generator",
                    "source_task_id": metadata.get("task_id"),
                    "source_domain": metadata.get("base_domain") or metadata.get("domain"),
                    "source_episode_won": 1.0,
                },
            )
            for index in range(SCOUT_COUNT)
        ]))
        results = await _score_skills(args, successful, generators)
        _apply_delta_rewards(generators, results)
        selected_skill, selected_index, selected_delta = _select_positive_skill(generators, results)

    mode = "positive_delta_skill" if selected_skill else (
        "successful_scout_no_positive_delta" if successful is not None else "no_successful_scout"
    )
    try:
        post_scout = list(await asyncio.gather(*[
            _run_executor(args, template, sampling_params, position=SCOUT_COUNT + offset,
                        skill=selected_skill, phase="post_scout", mode=mode)
            for offset in range(POST_SCOUT_COUNT)
        ]))
    except Exception as exc:
        return [_failed_source_sample(template, exc)]

    executors = scouts + post_scout
    group_metadata = {
        "rollout_id": int(compact_rollout_id),
        "rollout_variant": "tau_four_scout_direct_delta_card_v1",
        "initial_scout_success_count": sum(_raw_success(x) > 0.5 for x in scouts),
        "generator_candidate_count": len(generators),
        "generator_delta_available_count": sum(bool(x.get("available")) for x in results),
        "generator_positive_delta_count": sum(float(x.get("diff_reward", 0.0) or 0.0) > 0 for x in results if x.get("available")),
        "selected_skill_available": 1.0 if selected_skill else 0.0,
        "selected_skill_injected": 1.0 if selected_skill else 0.0,
        "selected_skill_candidate_index": selected_index,
        "selected_skill_delta": selected_delta,
        "executor_skill_mode": mode,
    }
    for item in executors + generators:
        item.rollout_id = int(compact_rollout_id)
        item.group_index = sample.group_index
        ensure_metadata(item).update(group_metadata)

    grpo_normalize(executors)
    if generators:
        grpo_normalize(generators)
    apply_reward_weights(args, generator_samples=generators, downstream_samples=executors)
    attach_train_metadata(generators, sample_role="generator")
    attach_train_metadata(executors, sample_role="executor")
    print(
        "[tau/delta-card] group complete "
        f"task_id={ensure_metadata(scouts[0]).get('task_id')} "
        f"scout_successes={group_metadata['initial_scout_success_count']} "
        f"delta_available={group_metadata['generator_delta_available_count']} "
        f"positive_delta={group_metadata['generator_positive_delta_count']} "
        f"skill_injected={group_metadata['selected_skill_injected']:.0f} "
        f"format_scores={[ensure_metadata(x).get('format_score') for x in generators]} "
        f"skill_words={[ensure_metadata(x).get('skill_word_count') for x in generators]} "
        f"delta_skips={[x.get('skip_reason') for x in results]}",
        flush=True,
    )
    return generators + executors


def _sample_role(sample: Sample) -> str:
    train_metadata = sample.train_metadata if isinstance(sample.train_metadata, dict) else {}
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    return str(train_metadata.get("sample_role") or metadata.get("sample_role") or "")


def _prepare_official_tau_metrics(executors: list[Sample]) -> None:
    """Populate the binary fields consumed by Tau's official metric logger.

    Training rewards have already been normalized independently inside every
    eight-executor group.  This function only restores the raw binary task view
    used by the established Tau dashboards; it does not alter optimization.
    """
    for sample in executors:
        metadata = ensure_metadata(sample)
        task_reward = _raw_env_success(sample)
        metadata.update({
            "env_task_reward": float(metadata.get("raw_env_reward", task_reward) or 0.0),
            "raw_reward": task_reward,
            "shaped_reward": task_reward,
            "binary_reward": task_reward,
            "progress_score": task_reward,
            "behavior_failure": metadata.get("termination_reason") in {
                "format_error", "max_steps", "too_many_errors",
            },
        })


def _log_delta_card_metrics(args, executors: list[Sample], generators: list[Sample]) -> None:
    if not getattr(args, "use_wandb", False) or not executors:
        return
    try:
        import wandb
        from slime.utils.metric_utils import compute_rollout_step
    except ImportError:
        return
    if wandb.run is None:
        return

    global _DELTA_WANDB_METRICS_DEFINED
    if not _DELTA_WANDB_METRICS_DEFINED:
        wandb.define_metric("delta_card/*", step_metric="rollout/step")
        _DELTA_WANDB_METRICS_DEFINED = True

    def mean(values: list[float]) -> float:
        return float(sum(values) / len(values)) if values else 0.0

    scouts = [x for x in executors if (x.metadata or {}).get("executor_rollout_phase") == "initial_scout"]
    post_scout = [x for x in executors if (x.metadata or {}).get("executor_rollout_phase") == "post_scout"]
    skill_executors = [x for x in post_scout if bool((x.metadata or {}).get("evolving_skill"))]
    no_skill_executors = [x for x in executors if not bool((x.metadata or {}).get("evolving_skill"))]
    available = [
        x for x in generators
        if float((x.metadata or {}).get("skill_diff_available", 0.0) or 0.0) > 0.5
    ]
    raw_deltas = [float((x.metadata or {}).get("skill_diff_reward", 0.0) or 0.0) for x in available]
    clipped = [float((x.metadata or {}).get("generator_reward_after_null_clip", 0.0) or 0.0) for x in available]

    # Every executor group contains exactly eight samples, so averaging the
    # repeated group flag yields the exact fraction of groups using a skill.
    group_skill_frac = mean([
        float((x.metadata or {}).get("selected_skill_injected", 0.0) or 0.0)
        for x in scouts
    ])
    metrics = {
        "delta_card/executor_success_rate": mean([_raw_env_success(x) for x in executors]),
        "delta_card/scout_success_rate": mean([_raw_env_success(x) for x in scouts]),
        "delta_card/post_scout_success_rate": mean([_raw_env_success(x) for x in post_scout]),
        "delta_card/skill_executor_success_rate": mean([_raw_env_success(x) for x in skill_executors]),
        "delta_card/no_skill_executor_success_rate": mean([_raw_env_success(x) for x in no_skill_executors]),
        "delta_card/skill_injected_group_frac": group_skill_frac,
        "delta_card/skill_injected_executor_frac": len(skill_executors) / max(len(executors), 1),
        "delta_card/generator_count": float(len(generators)),
        "delta_card/generator_available_frac": len(available) / max(len(generators), 1),
        "delta_card/generator_format_valid_frac": mean([
            float((x.metadata or {}).get("format_score", 0.0) or 0.0) >= 1.0
            for x in generators
        ]),
        "delta_card/generator_raw_delta_mean": mean(raw_deltas),
        "delta_card/generator_raw_delta_abs_mean": mean([abs(x) for x in raw_deltas]),
        "delta_card/generator_positive_delta_frac": mean([float(x > 0.0) for x in raw_deltas]),
        "delta_card/generator_clipped_reward_mean": mean(clipped),
        "delta_card/generator_scored_tokens_mean": mean([
            float((x.metadata or {}).get("skill_diff_scored_token_count", 0.0) or 0.0)
            for x in available
        ]),
    }
    rollout_id = executors[0].rollout_id
    if rollout_id is not None:
        metrics["rollout/step"] = compute_rollout_step(args, rollout_id)
    wandb.log(metrics)


def tau_delta_card_reward_post_process(args, samples):
    flat = [item for group in samples for item in (group if isinstance(group, list) else [group])]
    executors = [item for item in flat if _sample_role(item) == "executor"]
    generators = [item for item in flat if _sample_role(item) == "generator"]
    _prepare_official_tau_metrics(executors)

    # Match the established qwen36 Tau run's reward/* dashboard without
    # contaminating its denominators with generator language-model samples.
    if executors:
        from tau2_rl_pipeline.reward import _log_reward_component_metrics

        metric_args = copy(args)
        metric_args.n_samples_per_prompt = EXECUTOR_GROUP_SIZE
        _log_reward_component_metrics(metric_args, executors)
    _log_delta_card_metrics(args, executors, generators)

    rewards = [float(item.get_reward_value(args)) for item in flat]
    raw = [float((item.metadata or {}).get("raw_env_reward", (item.metadata or {}).get("raw_training_reward", 0.0))) for item in flat]
    return raw, rewards
