"""Fair eight-episode no-skill GRPO baseline for ALFWorld."""

from __future__ import annotations

import asyncio
from copy import deepcopy

from slime.utils.types import Sample

from src.alfworld.generate.episode import run_episode
from src.alfworld.reward.functions import grpo_normalize
from src.alfworld.utils.common import ensure_metadata, require_arg


def _sampling_params_for_position(params: dict, position: int) -> dict:
    result = dict(params)
    if result.get("sampling_seed") is not None:
        result["sampling_seed"] = int(result["sampling_seed"]) + int(position)
    return result


def _is_trainable(sample: Sample) -> bool:
    return (
        isinstance(sample.tokens, list)
        and bool(sample.tokens)
        and int(sample.response_length or 0) > 0
        and isinstance(sample.loss_mask, list)
        and any(bool(value) for value in sample.loss_mask)
    )


async def run_rollout(args, sample: Sample, sampling_params: dict, evaluation: bool = False) -> list[Sample]:
    if evaluation:
        raise ValueError("Use the dedicated ALFWorld evaluator for evaluation")
    group_size = int(getattr(args, "alfworld_executor_group_size", 8) or 8)
    if group_size != 8:
        raise ValueError(f"The fair baseline requires eight executor episodes, got {group_size}")

    params = dict(sampling_params)
    params["temperature"] = float(require_arg(args, "executor_temperature"))
    if getattr(args, "executor_top_p", None) is not None:
        params["top_p"] = float(args.executor_top_p)

    source = deepcopy(sample)

    async def _run(position: int) -> Sample:
        result = await run_episode(
            args,
            sample=deepcopy(source),
            skill=None,
            sampling_params=_sampling_params_for_position(params, position),
        )
        if not _is_trainable(result):
            raise RuntimeError(f"vanilla executor position {position} is not trainable")
        result.index = position
        metadata = ensure_metadata(result)
        raw_reward = 10.0 * float(bool(metadata.get("won", False)))
        metadata.update(
            {
                "method": "vanilla_grpo",
                "trajectory_role": "executor",
                "executor_group_position": position,
                "executor_group_size": group_size,
                "executor_skill_mode": "no_skill",
                "raw_env_reward": raw_reward,
                "raw_training_reward": raw_reward,
            }
        )
        result.reward = raw_reward
        train_metadata = dict(result.train_metadata or {})
        train_metadata.update({"training_stage": "executor", "sample_role": "executor"})
        result.train_metadata = train_metadata
        return result

    try:
        results = list(await asyncio.gather(*[_run(i) for i in range(group_size)]))
    except Exception as exc:
        print(f"[alfworld/vanilla] dropping incomplete group: {exc}")
        return []
    grpo_normalize(results)
    return results
