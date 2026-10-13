"""WebShop evaluation pipeline: skill-conditioned episode evaluation.

Retrieves a task semantically similar to the eval sample, extracts a skill,
then evaluates the target with and without that skill.
Metadata keys: won_with_skill, won_no_skill,
               num_steps_with_skill, num_steps_no_skill.
"""

import asyncio
from copy import deepcopy
import json
import os
import re

from slime.utils.types import Sample

from src.webshop.generate.episode import run_episode
from src.webshop.generate.retrieval import retrieve_tasks
from src.webshop.generate.generator import (
    generate_generator_sample,
    resolve_retrieval_query,
    resolve_task_description,
    task_info_to_sample,
)
from src.webshop.utils.common import ensure_metadata, maybe_print_random_sample, require_arg


FORMAL_GENERATOR_OUTPUT_PROMPT = """Return ONLY one valid JSON object using exactly this schema:

{"skill": "one transferable policy-facing WebShop skill"}

Requirements:
- The object must contain exactly one key: "skill".
- The skill must contain 2-4 action-oriented sentences and 50-150 English words.
- State an observable trigger, the concrete action or action ordering to use, and the verification or stopping condition.
- If the trajectory succeeded, retain only a reusable rule supported by its observations and actions; success does not prove that every action was useful or necessary.
- If the trajectory failed, extract only a grounded recovery or avoidance rule; never describe the failed behavior as successful.
- Abstract away product IDs, brands, exact prices, exact sizes, exact colors, and copied task-specific query phrases.
- Do not invent filters, controls, product facts, or actions absent from the task and trajectory.
- The task instruction, current page observations, and admissible actions always take priority over the skill.
- Do not output analysis, Markdown, headings, lists, code fences, or text outside the JSON object.
"""

_TRAILING_MODEL_TERMINATORS = (
    "<|im_end|>",
    "<|endoftext|>",
    "<|eot_id|>",
    "</s>",
)


def _format_formal_generator_messages(
    task_description: str,
    trajectory: str,
    won_score: float,
) -> list[dict]:
    """Build the method-independent long-skill prompt used by formal eval."""

    success = float(won_score) >= 0.9
    return [
        {
            "role": "user",
            "content": (
                f"Task: {str(task_description or '').strip()}\n\n"
                "Agent's Trajectory:\n"
                "<<<BEGIN_TRAJECTORY>>>\n"
                f"{str(trajectory or '').strip()}\n"
                "<<<END_TRAJECTORY>>>\n\n"
                f"True Environment Success: {success}\n\n"
                f"{FORMAL_GENERATOR_OUTPUT_PROMPT}"
            ),
        }
    ]


def _parse_formal_skill_output(args, response_text: str) -> dict:
    """Strict JSON parser with a permissive safety envelope around the prompt."""

    source = str(response_text or "").strip()
    while True:
        for terminator in _TRAILING_MODEL_TERMINATORS:
            if source.endswith(terminator):
                source = source[: -len(terminator)].rstrip()
                break
        else:
            break

    error = ""
    payload = None
    try:
        payload = json.loads(source)
    except json.JSONDecodeError as exc:
        error = f"Invalid JSON: {exc.msg}"
    if payload is not None and not isinstance(payload, dict):
        error = "Skill output must be one JSON object"
        payload = None
    if payload is not None and set(payload) != {"skill"}:
        error = "Skill JSON must contain exactly one key: skill"
        payload = None

    skill_text = str((payload or {}).get("skill") or "").strip()
    word_count = len(re.findall(r"\b[A-Za-z0-9][A-Za-z0-9'-]*\b", skill_text))
    if payload is not None and not skill_text:
        error = "Skill text is empty"
    if payload is not None and skill_text and not 20 <= word_count <= 200:
        error = f"Skill must contain 20-200 words, got {word_count}"

    sentences = []
    if payload is not None and skill_text and not error:
        sentences = [
            part
            for part in re.split(
                r"(?:[!?]+|(?<!\d)\.(?!\d))"
                r"(?=(?:[\"')\]]*)?(?:\s+[A-Z0-9]|\s*$))",
                skill_text,
            )
            if part.strip()
        ]
        if not 1 <= len(sentences) <= 6:
            error = f"Skill must contain 1-6 sentences, got {len(sentences)}"

    penalty = abs(float(require_arg(args, "skill_format_penalty")))
    if error or payload is None:
        return {
            "skill": "",
            "raw_skill_output": str(response_text or "").strip(),
            "skill_word_count": int(word_count),
            "skill_sentence_count": int(len(sentences)),
            "skill_schema": "formal_long_skill_v2",
            "format_penalty": -penalty,
            "format_score": 0.0,
            "skill_parse_error": error or "Invalid skill output",
        }
    return {
        "skill": skill_text,
        "skill_word_count": int(word_count),
        "skill_sentence_count": int(len(sentences)),
        "skill_schema": "formal_long_skill_v2",
        "format_penalty": 0.0,
        "format_score": 1.0,
    }


async def _extract_skill_from_task_info(
    args,
    task_info: dict,
    sampling_params: dict,
    log_tag: str = "eval_pipeline",
) -> str:
    """Run Executor (no skill) on a retrieved task, then Generator → skill text.

    Returns an empty string on any failure.
    """
    retrieved_sample = task_info_to_sample(task_info)

    # Step 2: Executor episode on retrieved task
    try:
        src_episode = await run_episode(
            args,
            sample=retrieved_sample,
            skill=None,
            sampling_params=sampling_params,
        )
    except Exception as exc:
        print(f"[{log_tag}] Executor error on retrieved task: {exc}")
        return ""

    # Step 3: Generator → skill text
    src_metadata = src_episode.metadata if isinstance(src_episode.metadata, dict) else {}
    retrieved_task_desc = src_metadata.get("task_description", "")
    trajectory = src_episode.response
    won_score = 1.0 if src_metadata.get("won", False) else 0.0

    max_attempts = max(
        1, int(os.environ.get("WEBSHOP_EVAL_GENERATOR_MAX_ATTEMPTS", "3"))
    )
    for attempt in range(1, max_attempts + 1):
        try:
            exp_sample = await generate_generator_sample(
                args,
                messages=_format_formal_generator_messages(
                    retrieved_task_desc,
                    trajectory,
                    won_score,
                ),
                sampling_params=sampling_params,
                n_idx=attempt - 1,
                requested_mode="formal_long_skill_v2",
                actual_mode="formal_long_skill_v2",
                parser_fn=_parse_formal_skill_output,
                base_metadata={
                    "training_stage": "eval",
                    "formal_skill_prompt_contract": "2-4_sentences_50-150_words",
                    "formal_skill_parser_contract": "1-6_sentences_20-200_words",
                },
            )
            metadata = exp_sample.metadata if isinstance(exp_sample.metadata, dict) else {}
            skill_text = str(metadata.get("skill") or "").strip()
            if skill_text:
                print(
                    f"[{log_tag}] [generator_outcome] success "
                    f"attempt={attempt}/{max_attempts}"
                )
                if attempt > 1:
                    print(f"[{log_tag}] Generator succeeded on attempt {attempt}/{max_attempts}")
                return skill_text
            print(
                f"[{log_tag}] Empty/invalid Generator output on attempt "
                f"{attempt}/{max_attempts}: "
                f"{metadata.get('skill_parse_error') or exp_sample.status or 'unknown'}"
            )
        except Exception as exc:
            print(f"[{log_tag}] Generator error on attempt {attempt}/{max_attempts}: {exc}")
    print(f"[{log_tag}] [generator_outcome] exhausted attempts={max_attempts}")
    return ""


async def _eval_with_skill_text(
    args,
    sample: Sample,
    sampling_params: dict,
    skill_text: str,
    skill_meta_suffix: str,
) -> Sample:
    """Steps 4-6: run target with/without skill and package results.

    Args:
        skill_meta_suffix: metadata key suffix for the skill run (e.g. "with_skill").
                           The no-skill run always uses "no_skill".
    """
    with_sample = deepcopy(sample)
    no_sample = deepcopy(sample)

    try:
        if skill_text:
            with_result, no_result = await asyncio.gather(
                run_episode(args, sample=with_sample, skill=skill_text, sampling_params=sampling_params),
                run_episode(args, sample=no_sample, skill=None, sampling_params=sampling_params),
            )
        else:
            # A failed Generator is an end-to-end with-skill failure, not a
            # baseline win.  Still run and retain the real no-skill episode.
            no_result = await run_episode(
                args, sample=no_sample, skill=None, sampling_params=sampling_params
            )
            with_result = None
    except Exception as exc:
        print(f"[eval_pipeline] Episode error ({skill_meta_suffix}): {exc}")
        sample.reward = 0.0
        sample.tokens = []
        sample.loss_mask = []
        sample.response = ""
        sample.metadata = {
            f"won_{skill_meta_suffix}": 0.0,
            "won_no_skill": 0.0,
            f"num_steps_{skill_meta_suffix}": 0,
            "num_steps_no_skill": 0,
        }
        return sample

    with_metadata = (
        with_result.metadata
        if with_result is not None and isinstance(with_result.metadata, dict)
        else {}
    )
    no_metadata = no_result.metadata if isinstance(no_result.metadata, dict) else {}
    won_with = 1.0 if with_metadata.get("won", False) else 0.0
    won_no = 1.0 if no_metadata.get("won", False) else 0.0

    result = with_result if with_result is not None else no_result
    result.reward = 10.0 * won_with
    if not isinstance(result.metadata, dict):
        result.metadata = {}
    result.metadata[f"won_{skill_meta_suffix}"] = won_with
    result.metadata["won_no_skill"] = won_no
    result.metadata[f"num_steps_{skill_meta_suffix}"] = with_metadata.get("num_steps", 0)
    result.metadata["num_steps_no_skill"] = no_metadata.get("num_steps", 0)
    result.metadata["skill_extraction_succeeded"] = bool(skill_text)
    return result


# ---------------------------------------------------------------------------
# Public eval functions
# ---------------------------------------------------------------------------


def _resolve_eval_query(args, sample: Sample) -> str:
    label = sample.label if isinstance(sample.label, dict) else {}
    query = resolve_retrieval_query(
        sample,
        fallback_task_description=resolve_task_description(label),
    )
    if query:
        return query
    return str(label.get("task_type") or label.get("category") or "")


async def run_eval(
    args,
    sample: Sample,
    sampling_params: dict,
) -> Sample:
    """Evaluate one WebShop sample with a semantically similar retrieved skill.

    Step 1 queries the retrieve server (/search) with the eval sample's task
    description to find the most relevant task for skill generation.
    Metadata keys written:
      won_with_skill, won_no_skill,
      num_steps_with_skill, num_steps_no_skill.
    """
    retrieve_url_test = require_arg(args, "webshop_retrieve_url_test")
    query = _resolve_eval_query(args, sample)
    label = sample.label if isinstance(sample.label, dict) else {}
    current_goal_idx = label.get("goal_idx")
    exclude_self = str(
        os.environ.get(
            "WEBSHOP_EVAL_EXCLUDE_SELF",
            getattr(args, "webshop_eval_exclude_self", "0"),
        )
    ).strip().lower() in {"1", "true", "yes", "on"}
    if exclude_self and current_goal_idx is None:
        raise ValueError("Self-excluding WebShop eval requires metadata.goal_idx")

    # ------------------------------------------------------------------
    # Similar retrieved skill
    # ------------------------------------------------------------------
    retrieved = await retrieve_tasks(
        query=query,
        K=1,
        url=retrieve_url_test,
        exclude_goal_idx=int(current_goal_idx) if exclude_self else None,
    )

    if retrieved:
        retrieved_metadata = retrieved[0].get("metadata") or retrieved[0].get("label") or {}
        retrieved_goal_idx = retrieved_metadata.get("goal_idx")
        if exclude_self and int(retrieved_goal_idx) == int(current_goal_idx):
            raise RuntimeError(f"Self retrieval escaped filtering: goal_idx={current_goal_idx}")
        skill_text = await _extract_skill_from_task_info(
            args, retrieved[0], sampling_params, log_tag="eval_pipeline/similar"
        )
    else:
        retrieved_goal_idx = None
        print(
            f"[eval_pipeline] WARNING: /search returned no results for '{query[:60]}'. "
            "Skill will be empty; only baseline episode is run."
        )
        skill_text = ""

    require_nonempty_skill = str(
        os.environ.get("WEBSHOP_EVAL_REQUIRE_NONEMPTY_SKILL", "0")
    ).strip().lower() in {"1", "true", "yes", "on"}
    empty_skill_policy = str(
        os.environ.get("WEBSHOP_EVAL_EMPTY_SKILL_POLICY", "fail")
    ).strip().lower()
    if require_nonempty_skill and not str(skill_text or "").strip() and empty_skill_policy == "fail":
        raise RuntimeError(
            "skill-conditioned evaluation produced an empty skill; refusing "
            "to silently copy the no-skill trajectory"
        )
    if not str(skill_text or "").strip():
        print(
            "[eval_pipeline] Generator exhausted all attempts; scoring the "
            "with-skill branch as zero and retaining the independently sampled "
            "no-skill branch."
        )

    result = await _eval_with_skill_text(
        args, sample, sampling_params, skill_text, skill_meta_suffix="with_skill"
    )
    result.metadata["skill_retrieval_self_excluded"] = bool(exclude_self)
    result.metadata["skill_retrieval_query_goal_idx"] = current_goal_idx
    result.metadata["skill_retrieval_source_goal_idx"] = retrieved_goal_idx
    result.metadata["skill_retrieval_is_self"] = (
        retrieved_goal_idx is not None
        and current_goal_idx is not None
        and int(retrieved_goal_idx) == int(current_goal_idx)
    )

    maybe_print_random_sample(result, tag="eval/with_experience")

    return result
