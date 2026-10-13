"""Generator helpers: skill extraction from WebShop episode trajectories."""

from __future__ import annotations

import ast
import os
import re

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

from src.webshop.generate.episode import run_episode
from src.webshop.prompts import skill_json_to_text
from src.webshop.utils.common import ensure_metadata, maybe_print_random_sample, require_arg


def task_info_to_sample(task_info: dict) -> Sample:
    downstream = Sample()
    label = task_info.get("metadata") or task_info.get("label") or {}
    downstream.label = dict(label)
    downstream.prompt = task_info.get("prompt")
    downstream.id = task_info.get("id")
    return downstream


def resolve_task_description(label: dict | None) -> str:
    if not isinstance(label, dict):
        return ""
    ann = label.get("ann")
    if isinstance(ann, dict) and ann.get("task_desc"):
        return str(ann.get("task_desc") or "")
    return str(label.get("task_description") or "")


def resolve_task_type(label: dict | None) -> str:
    if not isinstance(label, dict):
        return ""
    return str(label.get("task_type") or label.get("category") or "")


def _extract_question_text(prompt: str) -> str:
    """Extract the retrieval query text from a prompt string."""

    stop_markers = [
        "<|im_end|>",
        "<|im_start|>assistant",
        "[/INST]",
        "\nAssistant:",
        "\n\nAssistant:",
    ]

    def _cut_at_stop(text: str) -> str:
        indices = [text.find(marker) for marker in stop_markers if marker in text]
        if not indices:
            return text
        return text[: min(index for index in indices if index != -1)]

    def _extract_after_question(text: str) -> str | None:
        match = re.search(r"(?is)\bquestion\s*:\s*(.+)", text)
        if not match:
            return None
        question = _cut_at_stop(match.group(1)).strip()
        return question if question else None

    if "<|im_start|>user" in prompt:
        after = prompt.split("<|im_start|>user", 1)[-1]
        user_block = after.split("<|im_end|>", 1)[0]
        question = _extract_after_question(user_block)
        return question if question is not None else user_block.strip()

    if "[INST]" in prompt:
        after = prompt.split("[INST]", 1)[-1]
        inst_block = after.split("[/INST]", 1)[0]
        question = _extract_after_question(inst_block)
        return question if question is not None else inst_block.strip()

    for marker in ["\n\nUser:", "\nHuman:", "User:", "Human:"]:
        if marker in prompt:
            after = _cut_at_stop(prompt.split(marker, 1)[-1]).strip()
            question = _extract_after_question(after)
            return question if question is not None else after

    question = _extract_after_question(prompt)
    if question is not None:
        return question

    return prompt.strip()


def _message_content_to_text(content) -> str:
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                if item.strip():
                    parts.append(item.strip())
                continue
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if text is not None and str(text).strip():
                parts.append(str(text).strip())
        return "\n".join(parts).strip()

    if content is None:
        return ""
    return str(content).strip()


def _maybe_parse_message_list(prompt) -> list[dict] | None:
    if isinstance(prompt, list):
        return [item for item in prompt if isinstance(item, dict)]

    if not isinstance(prompt, str):
        return None

    stripped = prompt.strip()
    if not stripped.startswith("["):
        return None

    try:
        parsed = ast.literal_eval(stripped)
    except Exception:
        return None

    if not isinstance(parsed, list):
        return None
    return [item for item in parsed if isinstance(item, dict)]


def resolve_retrieval_query(
    sample: Sample,
    fallback_task_description: str = "",
) -> str:
    """Resolve a retrieval query consistently for both rollout and eval.

    We prefer the original dataset prompt/question because the retrieval cache
    is built from parquet `question` values. Task description is only used as a
    fallback when no prompt-derived query is available.
    """

    prompt = getattr(sample, "prompt", None)
    message_list = _maybe_parse_message_list(prompt)
    if message_list:
        for message in message_list:
            role = str(message.get("role") or "").lower()
            if role not in {"user", "human"}:
                continue
            text = _message_content_to_text(message.get("content"))
            if text:
                query = _extract_question_text(text).strip()
                if query:
                    return query
        merged_text = "\n".join(
            _message_content_to_text(message.get("content"))
            for message in message_list
            if isinstance(message, dict)
        ).strip()
        if merged_text:
            query = _extract_question_text(merged_text).strip()
            if query:
                return query

    prompt_text = _message_content_to_text(prompt)
    if prompt_text:
        query = _extract_question_text(prompt_text).strip()
        if query:
            return query

    return str(fallback_task_description or "").strip()


def parse_default_skill_output(
    args,
    response_text: str,
    *,
    min_words: int = 20,
    max_words: int = 100,
    min_sentences: int = 1,
    max_sentences: int = 2,
) -> dict:
    skill_text, error_msg, word_count = skill_json_to_text(
        response_text,
        min_words=min_words,
        max_words=max_words,
        min_sentences=min_sentences,
        max_sentences=max_sentences,
    )
    skill_format_penalty = abs(float(require_arg(args, "skill_format_penalty")))
    if skill_text is None:
        return {
            "skill": "",
            "raw_skill_output": str(response_text or "").strip(),
            "skill_word_count": int(word_count),
            "skill_schema": "single_short_text_v1",
            "format_penalty": -skill_format_penalty,
            "format_score": 0.0,
            "skill_parse_error": error_msg,
        }

    return {
        "skill": skill_text,
        "skill_word_count": int(word_count),
        "skill_schema": "single_short_text_v1",
        "format_penalty": 0.0,
        "format_score": 1.0,
    }


def parse_successful_scout_skill_output(args, response_text: str) -> dict:
    """Parse the reviewed Feature-2 schema with a permissive 20--200 word bound."""

    parsed = parse_default_skill_output(
        args,
        response_text,
        min_words=20,
        max_words=200,
        min_sentences=1,
        max_sentences=6,
    )
    parsed["skill_schema"] = "successful_scout_single_skill_v2"
    if float(parsed.get("format_score", 0.0) or 0.0) <= 0.0:
        return parsed

    word_count = int(parsed.get("skill_word_count", 0) or 0)
    if 20 <= word_count <= 200:
        return parsed

    parsed.update(
        {
            "skill": "",
            "raw_skill_output": str(response_text or "").strip(),
            "format_penalty": -abs(float(require_arg(args, "skill_format_penalty"))),
            "format_score": 0.0,
            "skill_parse_error": (
                f"Feature-2 skill must contain 20-200 words, got {word_count}"
            ),
        }
    )
    return parsed


def _make_failed_generator_sample(
    *,
    n_idx: int,
    status,
    requested_mode: str,
    actual_mode: str,
    base_metadata: dict | None = None,
) -> Sample:
    sample = Sample(group_index=0, index=n_idx, status=status)
    sample.reward = 0.0
    metadata = ensure_metadata(sample)
    metadata.update(base_metadata or {})
    metadata.setdefault("skill", "")
    metadata.setdefault("format_penalty", 0.0)
    metadata["requested_skill_source_mode"] = requested_mode
    metadata["skill_source_mode"] = actual_mode
    return sample


def _external_generator_config(args) -> tuple[str, str, str] | None:
    """Return an optional OpenAI-compatible generator endpoint configuration."""
    api_base = str(
        getattr(args, "webshop_generator_api_base", "")
        or os.environ.get("WEBSHOP_GENERATOR_API_BASE", "")
    ).strip()
    if not api_base:
        return None

    model = str(
        getattr(args, "webshop_generator_model", "")
        or os.environ.get("WEBSHOP_GENERATOR_MODEL", "")
        or "evolving-generator"
    ).strip()
    api_key = str(
        getattr(args, "webshop_generator_api_key", "")
        or os.environ.get("WEBSHOP_GENERATOR_API_KEY", "")
        or "EMPTY"
    )
    return api_base.rstrip("/"), model, api_key


async def _generate_generator_text(
    args,
    *,
    prompt_text: str,
    sampling_params: dict,
) -> tuple[str, str, str]:
    """Generate generator text locally or through a separate model server."""
    generator_max_response_len = int(
        getattr(args, "webshop_generator_max_response_len", 256)
    )
    if generator_max_response_len <= 0:
        raise ValueError(
            "webshop_generator_max_response_len must be positive, got "
            f"{generator_max_response_len}"
        )
    generator_sampling_params = dict(sampling_params)
    requested_max_new_tokens = int(
        generator_sampling_params.get("max_new_tokens", generator_max_response_len)
        or generator_max_response_len
    )
    generator_sampling_params["max_new_tokens"] = min(
        requested_max_new_tokens,
        generator_max_response_len,
    )

    external = _external_generator_config(args)
    if external is None:
        url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
        output = await post(
            url,
            {"text": prompt_text, "sampling_params": generator_sampling_params},
        )
        finish_type = output["meta_info"]["finish_reason"]["type"]
        return output["text"], finish_type, "policy_model"

    api_base, model, api_key = external
    request = {
        "model": model,
        "prompt": prompt_text,
        "temperature": float(generator_sampling_params.get("temperature", 1.0)),
        "top_p": float(generator_sampling_params.get("top_p", 1.0)),
        "max_tokens": int(generator_sampling_params["max_new_tokens"]),
    }
    if generator_sampling_params.get("stop") is not None:
        request["stop"] = generator_sampling_params["stop"]
    output = await post(
        f"{api_base}/completions",
        request,
        headers={"Authorization": f"Bearer {api_key}"},
    )
    choice = output["choices"][0]
    finish_reason = str(choice.get("finish_reason") or "stop")
    finish_type = "length" if finish_reason == "length" else "stop"
    return str(choice.get("text") or ""), finish_type, f"external:{model}"


async def generate_generator_sample(
    args,
    *,
    messages: list[dict],
    sampling_params: dict,
    n_idx: int,
    requested_mode: str = "generate",
    actual_mode: str = "generate",
    parser_fn=None,
    base_metadata: dict | None = None,
) -> Sample:
    parser_fn = parser_fn or parse_default_skill_output

    state = GenerateState(args)
    tokenizer = state.tokenizer
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **(getattr(args, "apply_chat_template_kwargs", None) or {}),
    )
    prompt_token_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

    try:
        response_text, finish_type, generator_name = await _generate_generator_text(
            args,
            prompt_text=prompt_text,
            sampling_params=sampling_params,
        )
    except Exception as exc:
        print(
            f"[webshop/generate_generator_sample] POST error "
            f"({actual_mode}, n={n_idx}): {exc}"
        )
        return _make_failed_generator_sample(
            n_idx=n_idx,
            status=Sample.Status.FAILED,
            requested_mode=requested_mode,
            actual_mode=actual_mode,
            base_metadata=base_metadata,
        )

    if finish_type == "abort":
        return _make_failed_generator_sample(
            n_idx=n_idx,
            status=Sample.Status.ABORTED,
            requested_mode=requested_mode,
            actual_mode=actual_mode,
            base_metadata=base_metadata,
        )

    response_token_ids = tokenizer(response_text, add_special_tokens=False)["input_ids"]

    parsed_metadata = dict(base_metadata or {})
    try:
        parsed_metadata.update(parser_fn(args, response_text))
    except Exception as exc:
        print(
            f"[webshop/generate_generator_sample] parse error "
            f"({actual_mode}, n={n_idx}): {exc}"
        )
        parsed_metadata.update(
            {
                "skill": "",
                "raw_skill_output": str(response_text or "").strip(),
                "skill_schema": "single_short_text_v1",
                "format_penalty": -abs(float(require_arg(args, "skill_format_penalty"))),
                "format_score": 0.0,
                "skill_parse_error": str(exc),
            }
        )

    sample = Sample()
    sample.prompt = prompt_text
    sample.tokens = prompt_token_ids + response_token_ids
    sample.response = response_text
    sample.response_length = len(response_token_ids)
    sample.loss_mask = [1] * len(response_token_ids)
    sample.group_index = 0
    sample.index = n_idx
    sample.status = (
        Sample.Status.TRUNCATED if finish_type == "length" else Sample.Status.COMPLETED
    )

    metadata = ensure_metadata(sample)
    metadata.update(parsed_metadata)
    metadata["requested_skill_source_mode"] = requested_mode
    metadata["skill_source_mode"] = actual_mode
    metadata["skill_generator"] = generator_name

    return sample


async def generate_generator_sample_from_trajectory(
    args,
    *,
    task_description: str,
    trajectory: str,
    won_score: float,
    sampling_params: dict,
    n_idx: int,
    parser_fn=None,
    base_metadata: dict | None = None,
) -> Sample:
    from src.webshop.prompts import format_generator_messages

    return await generate_generator_sample(
        args,
        messages=format_generator_messages(task_description, trajectory, won_score),
        sampling_params=sampling_params,
        n_idx=n_idx,
        requested_mode="generate",
        actual_mode="generate",
        parser_fn=parser_fn or parse_default_skill_output,
        base_metadata=base_metadata,
    )


async def generate_generator_sample_from_outcome_context(
    args,
    *,
    task_description: str,
    full_trajectory: str | None,
    high_entropy_turns: list[dict],
    won_score: float,
    sampling_params: dict,
    n_idx: int,
    candidate_count: int,
    parser_fn=None,
    base_metadata: dict | None = None,
) -> Sample:
    """Generate one skill from outcome-conditioned source evidence."""

    from src.webshop.prompts import format_outcome_conditioned_generator_messages

    return await generate_generator_sample(
        args,
        messages=format_outcome_conditioned_generator_messages(
            task_description,
            won_score,
            full_trajectory=full_trajectory,
            high_entropy_turns=high_entropy_turns,
            candidate_index=n_idx,
            candidate_count=candidate_count,
        ),
        sampling_params=sampling_params,
        n_idx=n_idx,
        requested_mode="generate_from_outcome_context",
        actual_mode="generate_from_outcome_context",
        parser_fn=parser_fn or parse_default_skill_output,
        base_metadata=base_metadata,
    )


async def build_generator_sample(
    args,
    *,
    task_description: str,
    full_trajectory: str | None,
    high_entropy_turns: list[dict],
    won_score: float,
    sampling_params: dict,
    n_idx: int,
    candidate_count: int,
    parser_fn=None,
    base_metadata: dict | None = None,
) -> Sample:
    parser_fn = parser_fn or parse_default_skill_output

    return await generate_generator_sample_from_outcome_context(
        args,
        task_description=task_description,
        full_trajectory=full_trajectory,
        high_entropy_turns=high_entropy_turns,
        won_score=won_score,
        sampling_params=sampling_params,
        n_idx=n_idx,
        candidate_count=candidate_count,
        parser_fn=parser_fn,
        base_metadata=base_metadata,
    )


async def build_successful_scout_generator_sample(
    args,
    *,
    task_description: str,
    trajectory: str,
    sampling_params: dict,
    n_idx: int,
    candidate_count: int,
    parser_fn=None,
    base_metadata: dict | None = None,
) -> Sample:
    """Generate one of exactly four candidates from one successful scout.

    Candidate position is metadata only.  It is deliberately absent from the
    prompt so all four stochastic generations receive byte-identical prompts.
    """

    if candidate_count != 4 or not 0 <= n_idx < candidate_count:
        raise ValueError(
            "Feature-2 requires candidate_count=4 and indices 0..3, got "
            f"index={n_idx}, count={candidate_count}"
        )

    from src.webshop.prompts import format_successful_scout_generator_messages

    metadata = dict(base_metadata or {})
    metadata.update(
        {
            "generator_candidate_index": int(n_idx),
            "generator_candidate_count": int(candidate_count),
            "generator_optimizer_enabled": 0.0,
            "training_stage": "feature2_observation_only",
        }
    )
    return await generate_generator_sample(
        args,
        messages=format_successful_scout_generator_messages(
            task_description,
            trajectory,
        ),
        sampling_params=sampling_params,
        n_idx=n_idx,
        requested_mode="generate_from_successful_scout_v2",
        actual_mode="generate_from_successful_scout_v2",
        parser_fn=parser_fn or parse_successful_scout_skill_output,
        base_metadata=metadata,
    )


async def evaluate_downstream(
    args,
    *,
    task_info: dict,
    skill_sample: Sample,
    source_task_description: str,
    source_task_type: str,
    n_idx: int,
    k_idx: int,
    sampling_params: dict,
    training_stage: str,
    message_builder=None,
    executor_reward_fn=None,
) -> Sample:
    downstream_sample = task_info_to_sample(task_info)
    downstream_label = downstream_sample.label if isinstance(downstream_sample.label, dict) else {}
    downstream_task_description = resolve_task_description(downstream_label)
    downstream_task_type = resolve_task_type(downstream_label)
    skill_metadata = skill_sample.metadata if isinstance(skill_sample.metadata, dict) else {}
    skill_text = str(skill_metadata.get("skill") or skill_sample.response or "").strip()
    workflow_expected_labels = None

    try:
        sample = await run_episode(
            args,
            sample=downstream_sample,
            skill=skill_text if skill_text else None,
            sampling_params=sampling_params,
            evaluation=False,
            message_builder=message_builder,
            workflow_expected_labels=workflow_expected_labels,
        )
    except Exception as exc:
        print(
            f"[webshop/evaluate_downstream] error "
            f"({training_stage}, k={k_idx}, n={n_idx}): {exc}"
        )
        sample = Sample(status=Sample.Status.FAILED)
        sample.reward = 0.0
        sample.tokens = []
        sample.loss_mask = []
        sample.response = ""
        sample.label = downstream_label
        sample.metadata = {
            "task_description": downstream_task_description,
            "won": False,
            "done": False,
            "num_steps": 0,
        }

    sample.group_index = k_idx + 1
    sample.index = n_idx

    metadata = ensure_metadata(sample)
    won = bool(metadata.get("won", False))
    metadata["training_stage"] = training_stage
    metadata["skill_text"] = skill_text
    metadata["source_task_description"] = str(source_task_description or "")
    metadata["source_task_type"] = str(source_task_type or "")
    metadata["downstream_task_description"] = str(
        metadata.get("task_description") or downstream_task_description or ""
    )
    metadata["downstream_task_type"] = str(downstream_task_type or "")
    metadata["requested_skill_source_mode"] = str(
        skill_metadata.get("requested_skill_source_mode") or ""
    )
    metadata["skill_source_mode"] = str(skill_metadata.get("skill_source_mode") or "")
    metadata["raw_env_reward"] = 10.0 * float(won)
    for audit_key in [
        "skill_language_audit_enabled",
        "skill_language_audit_applied",
        "skill_language_audit_pass",
        "skill_language_audit_has_uncommon_characters",
        "skill_language_audit_has_non_english_characters",
        "skill_language_audit_has_repetition",
        "skill_language_audit_penalty",
        "skill_language_audit_status",
        "skill_language_audit_judgment",
        "skill_language_audit_reason",
        "skill_language_audit_reward_zeroed",
    ]:
        if audit_key in skill_metadata:
            metadata[audit_key] = skill_metadata[audit_key]
    if "workflow_labels" in skill_metadata:
        metadata["workflow_labels"] = list(skill_metadata.get("workflow_labels") or [])
        metadata["workflow_step_count"] = len(metadata["workflow_labels"])

    training_reward = (
        executor_reward_fn(sample, skill_sample, args)
        if executor_reward_fn is not None
        else metadata["raw_env_reward"]
    )
    sample.reward = training_reward
    metadata["raw_training_reward"] = (
        float(training_reward) if isinstance(training_reward, (int, float)) else 0.0
    )
    return sample


def mark_source_episode_won(
    generator_samples: list[Sample],
    *,
    won_score: float,
    training_stage: str,
    source_task_description: str,
    source_task_type: str,
) -> None:
    for sample in generator_samples:
        metadata = ensure_metadata(sample)
        metadata["training_stage"] = training_stage
        metadata["source_episode_won"] = float(bool(won_score))
        metadata["source_task_description"] = source_task_description
        metadata["source_task_type"] = source_task_type
