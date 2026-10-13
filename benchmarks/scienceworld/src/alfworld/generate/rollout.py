"""Rollout: experience training."""

from __future__ import annotations

import asyncio
from copy import deepcopy
import os

from slime.utils.types import Sample

from src.alfworld.generate.episode import run_episode
from src.alfworld.generate.retrieval import retrieve_tasks
from src.alfworld.generate.generator import (
    build_generator_sample,
    evaluate_downstream,
    mark_source_episode_won,
    parse_default_skill_output,
    resolve_task_description,
    resolve_retrieval_query,
    resolve_task_type,
)
from src.alfworld.generate.skill_audit import audit_generator_samples
from src.alfworld.reward.functions import (
    _downstream_flat_index,
    _require_num_repeat,
    apply_reward_weights,
    attach_train_metadata,
    filter_high_mean_success_samples,
    finalize_generator_rewards,
    grpo_normalize,
)
from src.alfworld.utils.common import ensure_metadata, require_arg


def _parse_skill_output(args, response_text: str) -> dict:
    return parse_default_skill_output(args, response_text)


async def _evaluate_downstream_episode_repeat(
    args,
    *,
    task_info: dict,
    skill_sample: Sample,
    source_task_description: str,
    source_task_type: str,
    n_idx: int,
    k_idx: int,
    repeat_idx: int,
    num_repeat: int,
    sampling_params: dict,
) -> Sample:
    sample = await evaluate_downstream(
        args,
        task_info=task_info,
        skill_sample=skill_sample,
        source_task_description=source_task_description,
        source_task_type=source_task_type,
        n_idx=n_idx,
        k_idx=k_idx,
        sampling_params=sampling_params,
        training_stage="experience",
    )
    sample.index = n_idx * num_repeat + repeat_idx
    metadata = ensure_metadata(sample)
    metadata["experience_index"] = n_idx
    metadata["downstream_task_index"] = k_idx
    metadata["downstream_repeat_index"] = repeat_idx
    metadata["downstream_eval_repeats"] = num_repeat
    return sample


async def run_rollout(
    args,
    sample: Sample,
    sampling_params: dict,
    evaluation: bool = False,
) -> list[Sample]:
    num_experiences = int(require_arg(args, "n_experiences"))
    retrieval_topk = int(require_arg(args, "retrieval_topk"))
    retrieve_url_train = require_arg(args, "alfworld_retrieve_url_train")
    num_repeat = _require_num_repeat(args)
    label = sample.label if isinstance(sample.label, dict) else {}
    source_task_type = resolve_task_type(label)
    retrieval_query = resolve_retrieval_query(
        sample,
        fallback_task_description=resolve_task_description(label),
    )
    data_root = str(os.environ.get("ALFWORLD_DATA_ROOT", "")).strip()
    source_task_path = None
    if data_root and label.get("rel_dir"):
        source_task_path = os.path.join(
            data_root,
            str(label.get("split") or "train"),
            str(label["rel_dir"]),
        )

    try:
        src_sample = await run_episode(
            args,
            sample=sample,
            skill=None,
            sampling_params=sampling_params,
        )
    except Exception as exc:
        print(f"[alfworld/rollout] source episode error: {exc}")
        return []

    src_metadata = ensure_metadata(src_sample)
    trajectory = src_sample.response
    won_score = 1.0 if src_metadata.get("won", False) else 0.0
    task_description = src_metadata.get(
        "task_description",
        label.get("task_description", ""),
    )
    generator_samples = list(
        await asyncio.gather(
            *[
                build_generator_sample(
                    args,
                    task_description=task_description,
                    trajectory=trajectory,
                    won_score=won_score,
                    sampling_params=sampling_params,
                    n_idx=n_idx,
                    parser_fn=_parse_skill_output,
                    base_metadata={"training_stage": "experience"},
                )
                for n_idx in range(num_experiences)
            ]
        )
    )
    mark_source_episode_won(
        generator_samples,
        won_score=won_score,
        training_stage="experience",
        source_task_description=task_description,
        source_task_type=source_task_type,
    )
    skill_language_audit_task = asyncio.create_task(
        audit_generator_samples(args, generator_samples)
    )

    downstream_tasks = await retrieve_tasks(
        query=retrieval_query or task_description,
        K=retrieval_topk,
        url=retrieve_url_train,
        exclude_path=source_task_path,
    )

    if not downstream_tasks:
        print(
            "[alfworld/rollout] WARNING: no downstream tasks retrieved. "
            "Generator samples will receive reward=0."
        )
        await skill_language_audit_task
        for generator_sample in generator_samples:
            metadata = ensure_metadata(generator_sample)
            audit_applied = float(metadata.get("skill_language_audit_applied", 0.0) or 0.0) > 0.5
            audit_pass = float(metadata.get("skill_language_audit_pass", 0.0) or 0.0) > 0.5
            audit_reward_zeroed = float(
                metadata.get("skill_language_audit_reward_zeroed", 0.0) or 0.0
            ) > 0.5
            metadata["reward_zeroed_by_language_audit"] = 1.0 if (
                audit_reward_zeroed or (audit_applied and not audit_pass)
            ) else 0.0
            generator_sample.reward = 0.0
            metadata["raw_training_reward"] = 0.0
            metadata["raw_mean_downstream_executor_reward"] = 0.0
            metadata["raw_generator_base_reward"] = 0.0
            metadata["raw_generator_reward_before_language_audit"] = 0.0
            metadata["downstream_win_rate"] = 0.0
            metadata["raw_downstream_win_reward"] = 0.0
            metadata["downstream_eval_repeats"] = num_repeat
            metadata["raw_training_reward"] = float(generator_sample.reward)
        attach_train_metadata(
            generator_samples,
            sample_role="generator",
        )
        if not evaluation:
            grpo_normalize(generator_samples)
        return generator_samples

    executor_sampling_params = deepcopy(sampling_params)
    executor_sampling_params["temperature"] = float(require_arg(args, "executor_temperature"))
    executor_top_p = getattr(args, "executor_top_p", None)
    if executor_top_p is not None:
        executor_sampling_params["top_p"] = float(executor_top_p)
    print(executor_sampling_params["top_p"])
    downstream_samples = list(
        await asyncio.gather(
            *[
                _evaluate_downstream_episode_repeat(
                    args,
                    task_info=downstream_tasks[k_idx],
                    skill_sample=generator_samples[n_idx],
                    source_task_description=task_description,
                    source_task_type=source_task_type,
                    n_idx=n_idx,
                    k_idx=k_idx,
                    repeat_idx=repeat_idx,
                    num_repeat=num_repeat,
                    sampling_params=executor_sampling_params,
                )
                for k_idx in range(len(downstream_tasks))
                for n_idx in range(num_experiences)
                for repeat_idx in range(num_repeat)
            ]
        )
    )
    await skill_language_audit_task

    finalize_generator_rewards(
        generator_samples,
        downstream_samples,
        num_experiences=num_experiences,
        num_tasks=len(downstream_tasks),
        num_repeat=num_repeat,
    )

    generator_samples = filter_high_mean_success_samples(
        generator_samples,
        downstream_samples,
        num_experiences=num_experiences,
        num_tasks=len(downstream_tasks),
        num_repeat=num_repeat,
    )
    # if generator_samples:
    #     generator_samples = _filter_unreliable_generator_samples(
    #         generator_samples,
    #         downstream_samples,
    #         num_experiences=num_experiences,
    #         num_tasks=len(downstream_tasks),
    #         num_repeat=num_repeat,
    #     )

    if not evaluation:
        if generator_samples:
            grpo_normalize(generator_samples)
        for k_idx in range(len(downstream_tasks)):
            grpo_normalize(
                [
                    downstream_samples[
                        _downstream_flat_index(
                            num_experiences=num_experiences,
                            num_repeat=num_repeat,
                            k_idx=k_idx,
                            n_idx=n_idx,
                            repeat_idx=repeat_idx,
                        )
                    ]
                    for n_idx in range(num_experiences)
                    for repeat_idx in range(num_repeat)
                ]
            )

    apply_reward_weights(
        args,
        generator_samples=generator_samples,
        downstream_samples=downstream_samples,
    )
    attach_train_metadata(
        generator_samples,
        sample_role="generator",
    )
    attach_train_metadata(downstream_samples, sample_role="downstream_executor")
    return generator_samples + downstream_samples
