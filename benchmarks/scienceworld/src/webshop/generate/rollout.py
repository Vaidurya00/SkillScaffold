"""Delta-RL v2 WebShop rollout: four scouts plus four post-scout executors."""

from __future__ import annotations

import asyncio
import math
from copy import deepcopy

from slime.utils.types import Sample

from src.webshop.generate.episode import run_episode
from src.webshop.generate.generator import (
    build_successful_scout_generator_sample,
    resolve_task_description,
)
from src.webshop.generate.skill_diff import (
    apply_skill_diff_rewards,
    score_candidate_skills_on_successful_scout,
)
from src.webshop.reward.functions import apply_reward_weights, grpo_normalize
from src.webshop.trajectory import (
    finalize_source_scout,
    mark_executor_group_member,
    mark_source_scout,
)
from src.webshop.utils.common import ensure_metadata, require_arg


_SCOUT_COUNT = 4
_POST_SCOUT_COUNT = 4
_EXECUTOR_GROUP_SIZE = _SCOUT_COUNT + _POST_SCOUT_COUNT


def _episode_is_trainable(sample: Sample) -> bool:
    return (
        isinstance(sample.tokens, list)
        and bool(sample.tokens)
        and int(sample.response_length or 0) > 0
        and isinstance(sample.loss_mask, list)
        and any(bool(value) for value in sample.loss_mask)
    )


def _executor_sampling_params(args, sampling_params: dict) -> dict:
    params = dict(sampling_params)
    params["temperature"] = float(require_arg(args, "executor_temperature"))
    executor_top_p = getattr(args, "executor_top_p", None)
    if executor_top_p is not None:
        params["top_p"] = float(executor_top_p)
    return params


def _annotate_samples(samples: list[Sample], metadata_update: dict) -> None:
    for item in samples:
        ensure_metadata(item).update(metadata_update)


def _sampling_params_for_position(
    sampling_params: dict,
    position: int,
) -> dict:
    params = dict(sampling_params)
    if params.get("sampling_seed") is not None:
        params["sampling_seed"] = int(params["sampling_seed"]) + int(position)
    return params


async def _run_initial_scouts(
    args,
    *,
    source_template: Sample,
    sampling_params: dict,
) -> list[Sample]:
    """Generate four same-query, no-skill scouts as one future GRPO group."""

    async def _run_one(position: int) -> Sample:
        scout = deepcopy(source_template)
        result = await run_episode(
            args,
            sample=scout,
            skill=None,
            sampling_params=_sampling_params_for_position(sampling_params, position),
        )
        if not _episode_is_trainable(result):
            raise RuntimeError(
                f"initial scout position={position} produced no trainable model tokens"
            )
        result.index = int(position)
        return result

    scouts = list(
        await asyncio.gather(*[_run_one(position) for position in range(_SCOUT_COUNT)])
    )
    source_scout = scouts[0]
    mark_source_scout(source_scout)
    ensure_metadata(source_scout)["executor_rollout_phase"] = "initial_scout"
    for position, scout in enumerate(scouts[1:], start=1):
        mark_executor_group_member(
            scout,
            source_scout=source_scout,
            position=position,
            skill=None,
            executor_skill_mode="initial_four_scouts",
            executor_group_size=_EXECUTOR_GROUP_SIZE,
        )
        ensure_metadata(scout)["executor_rollout_phase"] = "initial_scout"
    return scouts


def _select_successful_scout(scouts: list[Sample]) -> tuple[Sample | None, int | None]:
    """Select the first successful scout in stable sampling-position order."""

    for position, scout in enumerate(scouts):
        if bool(ensure_metadata(scout).get("won", False)):
            return scout, position
    return None, None


async def _generate_successful_scout_skill_candidates(
    args,
    *,
    selected_success: Sample,
    selected_position: int,
    sampling_params: dict,
) -> tuple[list[Sample], list[dict], bool]:
    """Generate four observation-only candidates from the selected success."""

    selected_metadata = ensure_metadata(selected_success)
    task_description = resolve_task_description(
        selected_success.label if isinstance(selected_success.label, dict) else None
    ) or str(selected_metadata.get("task_description") or "")
    trajectory = str(selected_success.response or "").strip()
    base_metadata = {
        "selected_success_scout_position": int(selected_position),
        "source_goal_idx": selected_metadata.get("goal_idx"),
        "source_category": selected_metadata.get("category"),
        "source_task_type": selected_metadata.get("task_type"),
    }

    generated = await asyncio.gather(
        *[
            build_successful_scout_generator_sample(
                args,
                task_description=task_description,
                trajectory=trajectory,
                sampling_params=_sampling_params_for_position(
                    sampling_params,
                    candidate_index,
                ),
                n_idx=candidate_index,
                candidate_count=_SCOUT_COUNT,
                base_metadata=base_metadata,
            )
            for candidate_index in range(_SCOUT_COUNT)
        ],
        return_exceptions=True,
    )

    candidates: list[Sample] = []
    summaries: list[dict] = []
    prompts: list[str] = []
    for candidate_index, result in enumerate(generated):
        if isinstance(result, BaseException):
            summaries.append(
                {
                    "candidate_index": candidate_index,
                    "status": "generation_exception",
                    "format_score": 0.0,
                    "skill": "",
                    "error": str(result),
                }
            )
            continue

        candidates.append(result)
        prompts.append(str(result.prompt or ""))
        metadata = ensure_metadata(result)
        summaries.append(
            {
                "candidate_index": candidate_index,
                "status": str(result.status),
                "format_score": float(metadata.get("format_score", 0.0) or 0.0),
                "skill": str(metadata.get("skill") or ""),
                "error": str(metadata.get("skill_parse_error") or ""),
            }
        )

    prompts_identical = len(prompts) == _SCOUT_COUNT and len(set(prompts)) == 1
    return candidates, summaries, prompts_identical


def _select_positive_delta_skill(
    generator_samples: list[Sample],
    scoring_results: list[dict],
) -> tuple[str | None, int | None, float | None]:
    """Select the stable argmax among valid candidates with finite Delta > 0."""

    eligible: list[tuple[float, int, str]] = []
    for sample, result in zip(generator_samples, scoring_results):
        metadata = ensure_metadata(sample)
        skill = str(metadata.get("skill") or "").strip()
        raw_diff = float(result.get("diff_reward", 0.0) or 0.0)
        if (
            result.get("available")
            and float(metadata.get("format_score", 0.0) or 0.0) >= 1.0
            and skill
            and math.isfinite(raw_diff)
            and raw_diff > 0.0
        ):
            eligible.append((raw_diff, int(sample.index), skill))
    if not eligible:
        return None, None, None
    selected_diff, selected_index, selected_skill = max(
        eligible,
        key=lambda item: (item[0], -item[1]),
    )
    return selected_skill, selected_index, selected_diff


def _prepare_generator_optimizer_group(
    source_scout: Sample,
    generator_samples: list[Sample],
) -> list[Sample]:
    """Return an all-or-none four-sample Generator GRPO group."""

    if len(generator_samples) != _SCOUT_COUNT or not all(
        _episode_is_trainable(sample) for sample in generator_samples
    ):
        return []
    source_group_index = source_scout.group_index
    if not isinstance(source_group_index, int) or source_group_index < 0:
        raise ValueError(
            "Executor source requires a nonnegative integer group_index, got "
            f"{source_group_index!r}"
        )
    generator_group_index = -(source_group_index + 1)
    generator_group_id = f"webshop:generator:{source_group_index}"
    ordered = sorted(generator_samples, key=lambda sample: int(sample.index))
    for position, sample in enumerate(ordered):
        sample.group_index = generator_group_index
        sample.index = position
        metadata = ensure_metadata(sample)
        metadata.update(
            {
                "trajectory_role": "generator",
                "generator_group_id": generator_group_id,
                "generator_group_position": position,
                "generator_group_size": _SCOUT_COUNT,
                "generator_optimizer_enabled": 1.0,
            }
        )
        train_metadata = dict(sample.train_metadata or {})
        train_metadata.update(
            {
                "training_stage": "generator",
                "sample_role": "generator",
                "generator_group_id": generator_group_id,
                "generator_group_position": position,
                "generator_group_size": _SCOUT_COUNT,
            }
        )
        sample.train_metadata = train_metadata
    return ordered


def _summarize_skill_candidates(
    generator_samples: list[Sample],
    *,
    generation_summaries: list[dict],
    optimizer_sample_ids: set[int],
) -> list[dict]:
    summaries_by_index = {
        int(item["candidate_index"]): dict(item)
        for item in generation_summaries
        if item.get("candidate_index") is not None
    }
    for sample in sorted(generator_samples, key=lambda item: int(item.index)):
        metadata = ensure_metadata(sample)
        candidate_index = int(
            metadata.get("generator_candidate_index", sample.index)
        )
        summaries_by_index[candidate_index] = {
            "candidate_index": candidate_index,
            "status": str(sample.status),
            "format_score": float(metadata.get("format_score", 0.0) or 0.0),
            "skill": str(metadata.get("skill") or ""),
            "error": str(metadata.get("skill_parse_error") or ""),
            "delta_available": float(
                metadata.get("skill_diff_available", 0.0) or 0.0
            ),
            "delta": metadata.get("skill_diff_reward"),
            "positive_clipped_reward": float(
                metadata.get("generator_reward_after_null_clip", 0.0) or 0.0
            ),
            "optimizer_included": 1.0 if id(sample) in optimizer_sample_ids else 0.0,
        }
    return [summaries_by_index[index] for index in sorted(summaries_by_index)]


async def _run_post_scout_executors(
    args,
    *,
    source_template: Sample,
    source_scout: Sample,
    skill: str | None,
    executor_skill_mode: str,
    sampling_params: dict,
) -> list[Sample]:
    """Generate positions 4--7 for the same-query Executor GRPO group."""

    async def _run_one(position: int) -> Sample:
        sibling = deepcopy(source_template)
        result = await run_episode(
            args,
            sample=sibling,
            skill=skill,
            sampling_params=_sampling_params_for_position(sampling_params, position),
        )
        if not _episode_is_trainable(result):
            raise RuntimeError(
                f"post-scout executor position={position} produced no trainable model tokens"
            )
        mark_executor_group_member(
            result,
            source_scout=source_scout,
            position=position,
            skill=skill,
            executor_skill_mode=executor_skill_mode,
            executor_group_size=_EXECUTOR_GROUP_SIZE,
        )
        ensure_metadata(result)["executor_rollout_phase"] = "post_scout"
        return result

    return list(
        await asyncio.gather(
            *[
                _run_one(position)
                for position in range(_SCOUT_COUNT, _EXECUTOR_GROUP_SIZE)
            ]
        )
    )


def _normalize_executor_group(executor_samples: list[Sample]) -> None:
    for item in executor_samples:
        metadata = ensure_metadata(item)
        raw_outcome = float(metadata.get("outcome_reward", 0.0) or 0.0)
        metadata["raw_env_reward"] = raw_outcome
        metadata["raw_training_reward"] = raw_outcome
    grpo_normalize(executor_samples)
    for item in executor_samples:
        ensure_metadata(item)["normalized_executor_reward"] = float(item.reward or 0.0)


async def run_rollout(
    args,
    sample: Sample,
    sampling_params: dict,
    evaluation: bool = False,
) -> list[Sample]:
    """Run the reviewed Feature-3 direct-Delta training pipeline.

    A selected successful scout supplies both the evidence for four Generator
    candidates and the fixed trajectory for paired no-skill/with-skill scoring.
    The maximum finite positive Delta skill conditions Executor positions 4--7;
    all four trainable Generator candidates form a separate GRPO group.
    """

    if evaluation:
        raise ValueError("training run_rollout does not handle evaluation; use run_eval")
    num_experiences = int(require_arg(args, "n_experiences"))
    if num_experiences != 4:
        raise ValueError(
            "Delta-RL v2 reserves exactly four generator candidates, got "
            f"n_experiences={num_experiences}"
        )

    source_template = deepcopy(sample)
    executor_sampling_params = _executor_sampling_params(args, sampling_params)
    try:
        scouts = await _run_initial_scouts(
            args,
            source_template=source_template,
            sampling_params=executor_sampling_params,
        )
    except Exception as exc:
        print(f"[webshop/rollout/v2] four-scout group error: {exc}")
        return []

    source_scout = scouts[0]
    selected_success, selected_position = _select_successful_scout(scouts)
    success_count = sum(bool(ensure_metadata(item).get("won", False)) for item in scouts)
    if selected_success is None:
        executor_skill_mode = "scout_no_success_no_skill"
        skill_stage = "not_applicable_no_success"
        generator_candidates: list[Sample] = []
        generation_summaries: list[dict] = []
        scoring_results: list[dict] = []
        generator_optimizer_samples: list[Sample] = []
        selected_skill = None
        selected_candidate_index = None
        selected_delta = None
        prompts_identical = False
    else:
        generator_candidates, generation_summaries, prompts_identical = (
            await _generate_successful_scout_skill_candidates(
                args,
                selected_success=selected_success,
                selected_position=int(selected_position),
                sampling_params=sampling_params,
            )
        )
        scoring_results = await score_candidate_skills_on_successful_scout(
            args,
            source_scout=selected_success,
            generator_samples=generator_candidates,
        )
        apply_skill_diff_rewards(
            generator_candidates,
            scoring_results,
            reward_source="selected_successful_scout_logprob_delta",
        )
        selected_skill, selected_candidate_index, selected_delta = (
            _select_positive_delta_skill(generator_candidates, scoring_results)
        )
        try:
            generator_optimizer_samples = _prepare_generator_optimizer_group(
                source_scout,
                generator_candidates,
            )
        except Exception as exc:
            print(f"[webshop/rollout/v2] generator group disabled: {exc}")
            generator_optimizer_samples = []
        if selected_skill is None:
            executor_skill_mode = "scout_success_no_positive_delta_no_skill"
            skill_stage = "feature3_no_positive_delta"
        else:
            executor_skill_mode = "scout_success_positive_delta_skill"
            skill_stage = "feature3_positive_delta_skill_selected"

    valid_candidate_count = sum(
        float(ensure_metadata(item).get("format_score", 0.0) or 0.0) > 0.0
        for item in generator_candidates
    )
    delta_available_count = sum(bool(item.get("available")) for item in scoring_results)
    positive_delta_count = sum(
        bool(result.get("available"))
        and float(ensure_metadata(candidate).get("format_score", 0.0) or 0.0)
        >= 1.0
        and bool(str(ensure_metadata(candidate).get("skill") or "").strip())
        and math.isfinite(float(result.get("diff_reward", 0.0) or 0.0))
        and float(result.get("diff_reward", 0.0) or 0.0) > 0.0
        for candidate, result in zip(generator_candidates, scoring_results)
    )
    optimizer_sample_ids = {id(item) for item in generator_optimizer_samples}
    candidate_summaries = _summarize_skill_candidates(
        generator_candidates,
        generation_summaries=generation_summaries,
        optimizer_sample_ids=optimizer_sample_ids,
    )

    group_metadata = {
        "rollout_variant": "four_scout_direct_delta_v3",
        "cross_query_memory_used": 0.0,
        "similarity_gate_used": 0.0,
        "initial_scout_count": _SCOUT_COUNT,
        "initial_scout_success_count": int(success_count),
        "initial_scout_success_fraction": float(success_count / _SCOUT_COUNT),
        "initial_scout_any_success": 1.0 if selected_success is not None else 0.0,
        "selected_success_scout_position": selected_position,
        "success_skill_pipeline_stage": skill_stage,
        "executor_skill_mode": executor_skill_mode,
        "generator_candidate_attempt_count": (
            _SCOUT_COUNT if selected_success is not None else 0
        ),
        "generator_candidate_sample_count": len(generator_candidates),
        "generator_valid_candidate_count": int(valid_candidate_count),
        "generator_prompts_byte_identical": 1.0 if prompts_identical else 0.0,
        "generator_delta_available_count": int(delta_available_count),
        "generator_positive_delta_count": int(positive_delta_count),
        "generator_optimizer_enabled": (
            1.0 if generator_optimizer_samples else 0.0
        ),
        "generator_optimizer_group_size": len(generator_optimizer_samples),
        "generator_delta_attempted": 1.0 if scoring_results else 0.0,
        "generator_delta_computed": 1.0 if delta_available_count > 0 else 0.0,
        "selected_skill_injected": 1.0 if selected_skill else 0.0,
        "selected_skill_candidate_index": selected_candidate_index,
        "selected_skill_delta": selected_delta,
        "selected_skill": str(selected_skill or ""),
        "feature3_skill_candidates": candidate_summaries,
    }
    _annotate_samples(scouts, group_metadata)

    try:
        post_scout_executors = await _run_post_scout_executors(
            args,
            source_template=source_template,
            source_scout=source_scout,
            skill=selected_skill,
            executor_skill_mode=executor_skill_mode,
            sampling_params=executor_sampling_params,
        )
    except Exception as exc:
        print(f"[webshop/rollout/v2] post-scout executor group error: {exc}")
        return []

    executor_samples = [*scouts, *post_scout_executors]
    _annotate_samples(executor_samples, group_metadata)
    finalize_source_scout(
        source_scout,
        executor_skill_mode=executor_skill_mode,
        executor_group_size=_EXECUTOR_GROUP_SIZE,
    )
    _normalize_executor_group(executor_samples)
    if generator_optimizer_samples:
        grpo_normalize(generator_optimizer_samples)
        for item in generator_optimizer_samples:
            ensure_metadata(item)["normalized_generator_reward"] = float(
                item.reward or 0.0
            )
    apply_reward_weights(
        args,
        generator_samples=generator_optimizer_samples,
        downstream_samples=executor_samples,
    )
    return [*generator_optimizer_samples, *executor_samples]
