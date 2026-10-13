"""Training-like per-task evaluation with two scouts and one shared skill.

For each evaluation task this protocol runs exactly two no-skill scout episodes,
asks the checkpoint's own generator for one skill grounded in those two
trajectories, and then runs K fresh answers with that one shared skill.  Only
the K final answers are returned to the evaluator; complete scout and generator
evidence is retained once in the first answer's metadata.
"""

from __future__ import annotations

import asyncio
from copy import deepcopy
import os

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.types import Sample

from src.alfworld.generate.delta_rollout import _run_executor, _sampling_params_for_position
from src.alfworld.generate.generator import generate_generator_sample
from src.alfworld.generate.group_delta_rollout import (
    _compact_trajectory,
    _load_prompt_module,
    _parse_skill,
)
from src.alfworld.utils.common import ensure_metadata


SCOUT_COUNT = 2


def _two_attempt_system_prompt(text: str) -> str:
    replacements = (
        ("from four independent\nattempts", "from two independent\nattempts"),
        ("Inspect all four attempts", "Inspect both attempts"),
        ("all four attempts", "both attempts"),
    )
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def _two_attempt_user_prompt(prompt_module, task: str, trajectories: str) -> str:
    text = prompt_module.build_generator_user_prompt(
        task,
        trajectories,
        example_keys=(),
    )
    return text.replace(
        "Four independent no-skill attempts",
        "Two independent no-skill attempts",
    )


async def _extract_one_skill(args, scouts: list[Sample], sampling_params: dict) -> Sample:
    if len(scouts) != SCOUT_COUNT:
        raise ValueError(f"expected {SCOUT_COUNT} scouts, found {len(scouts)}")
    prompt_module = _load_prompt_module()
    tokenizer = GenerateState(args).tokenizer
    task_descriptions = {
        str(ensure_metadata(scout).get("task_description") or "") for scout in scouts
    }
    if len(task_descriptions) != 1 or not next(iter(task_descriptions), ""):
        raise ValueError("two scouts do not share one non-empty task description")
    task = next(iter(task_descriptions))
    rendered = []
    audits = []
    for index, scout in enumerate(scouts):
        record, audit = _compact_trajectory(tokenizer, scout, index)
        rendered.append(record)
        audits.append(audit)
    labels = [bool(ensure_metadata(scout).get("won", False)) for scout in scouts]
    messages = [
        {
            "role": "system",
            "content": _two_attempt_system_prompt(prompt_module.GENERATOR_SYSTEM_PROMPT),
        },
        {
            "role": "user",
            "content": _two_attempt_user_prompt(
                prompt_module,
                task,
                "\n\n".join(rendered),
            ),
        },
    ]
    params = dict(sampling_params)
    params.update({"max_new_tokens": 900, "temperature": 1.0, "top_p": 0.85})
    params = _sampling_params_for_position(params, 100)
    return await generate_generator_sample(
        args,
        messages=messages,
        sampling_params=params,
        n_idx=0,
        requested_mode="self_extract2_same_task",
        actual_mode="self_extract2_same_task",
        parser_fn=_parse_skill,
        base_metadata={
            "training_stage": "eval_self_extract2",
            "generator_candidate_count": 1,
            "generator_visible_success_count": int(sum(labels)),
            "generator_trajectory_audits": audits,
            "generator_prompt_attempt_count": SCOUT_COUNT,
            "generator_prompt_contract_override": "alfworld_two_trajectory_v3_zero_shot",
        },
    )


async def run_eval(
    args,
    sample: Sample,
    sampling_params: dict,
) -> list[Sample]:
    """Return K final answer samples sharing one skill extracted from two scouts."""
    answer_count = int(os.environ.get("ALFWORLD_SELF_EXTRACT_ANSWER_COUNT", "8"))
    if answer_count <= 0:
        raise ValueError("ALFWORLD_SELF_EXTRACT_ANSWER_COUNT must be positive")
    template = deepcopy(sample)
    base_index = int(sample.index or 0)
    executor_params = dict(sampling_params)
    executor_params["temperature"] = float(getattr(args, "executor_temperature", 1.0))
    if getattr(args, "executor_top_p", None) is not None:
        executor_params["top_p"] = float(args.executor_top_p)

    scouts = list(
        await asyncio.gather(
            *[
                _run_executor(args, template, executor_params, position, None)
                for position in range(SCOUT_COUNT)
            ]
        )
    )
    generator = await _extract_one_skill(args, scouts, sampling_params)
    generator_metadata = ensure_metadata(generator)
    skill = str(generator_metadata.get("skill") or "").strip()

    answers = list(
        await asyncio.gather(
            *[
                _run_executor(
                    args,
                    template,
                    executor_params,
                    SCOUT_COUNT + position,
                    skill or None,
                )
                for position in range(answer_count)
            ]
        )
    )
    scout_wins = [bool(ensure_metadata(item).get("won", False)) for item in scouts]
    group_id = str((template.label or {}).get("rel_dir") if isinstance(template.label, dict) else "")
    for position, answer in enumerate(answers):
        answer.index = base_index * answer_count + position
        metadata = ensure_metadata(answer)
        won = bool(metadata.get("won", False))
        metadata.update(
            {
                "evaluation_protocol": "self_extract2_one_shared_skill_k",
                "self_extract_group_id": group_id,
                "self_extract_source_scout_count": SCOUT_COUNT,
                "self_extract_answer_count": answer_count,
                "self_extract_scout_wins": scout_wins,
                "self_extract_skill_valid": bool(skill),
                "self_extract_skill": skill,
                "self_extract_format_score": float(
                    generator_metadata.get("format_score", 0.0) or 0.0
                ),
                "self_extract_parse_error": str(
                    generator_metadata.get("skill_parse_error") or ""
                ),
                "self_extract_answer_position": position,
                "executor_skill_mode": "self_extracted_shared_skill" if skill else "no_skill_fallback",
                "executor_route_skill_injected": 1.0 if skill else 0.0,
                "raw_env_reward": 10.0 * float(won),
            }
        )
        answer.reward = 10.0 * float(won)
        if position == 0:
            metadata["self_extract_scout_records"] = [
                {
                    "won": bool(ensure_metadata(scout).get("won", False)),
                    "num_steps": int(ensure_metadata(scout).get("num_steps", 0) or 0),
                    "response": str(scout.response or ""),
                }
                for scout in scouts
            ]
            metadata["self_extract_raw_output"] = str(generator.response or "")
            metadata["self_extract_prompt"] = str(generator.prompt or "")
    return answers
