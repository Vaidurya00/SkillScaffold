"""Matched no-skill ALFWorld evaluation shared by all training methods."""

from __future__ import annotations

from copy import deepcopy

from slime.utils.types import Sample

from src.alfworld.generate.episode import run_episode
from src.alfworld.utils.common import ensure_metadata


async def run_eval(args, sample: Sample, sampling_params: dict) -> Sample:
    result = await run_episode(
        args,
        sample=deepcopy(sample),
        skill=None,
        sampling_params=sampling_params,
        evaluation=True,
    )
    metadata = ensure_metadata(result)
    won = bool(metadata.get("won", False))
    metadata.update(
        {
            "evaluation_protocol": "matched_no_skill_pass_at_1",
            "executor_skill_mode": "no_skill",
            "raw_env_reward": 10.0 * float(won),
        }
    )
    result.reward = 10.0 * float(won)
    return result
