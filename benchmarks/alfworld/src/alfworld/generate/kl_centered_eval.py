"""Rollout-only evaluator for the audited KL-centered Delta experiment."""

from __future__ import annotations

from copy import deepcopy
import os

from slime.utils.types import Sample

from src.alfworld.generate.episode import run_episode
from src.alfworld.utils.common import ensure_metadata


async def run_eval(args, sample: Sample, sampling_params: dict) -> Sample:
    label = sample.label if isinstance(sample.label, dict) else {}
    condition = str(label.get("kl_condition") or "no_skill")
    skill = str(label.get("kl_skill") or "") if condition != "no_skill" else ""
    repeat_count = int(getattr(args, "n_samples_per_eval_prompt", 1) or 1)
    repeat = int(sample.index) % repeat_count
    result = await run_episode(
        args,
        sample=deepcopy(sample),
        skill=skill or None,
        sampling_params=sampling_params,
        evaluation=True,
    )
    metadata = ensure_metadata(result)
    won = bool(metadata.get("won", False))
    metadata.update(
        {
            "evaluation_protocol": "kl_centered_delta_rollout_v1",
            "kl_checkpoint": os.environ.get("ALFWORLD_KL_CHECKPOINT", ""),
            "kl_stage": os.environ.get("ALFWORLD_KL_STAGE", ""),
            "kl_query_id": str(label["kl_query_id"]),
            "kl_condition": condition,
            "kl_card_id": str(label.get("kl_card_id") or ""),
            "kl_skill": skill,
            "kl_repeat": repeat,
            "kl_sampling_seed": sampling_params.get("sampling_seed"),
            "executor_skill_mode": "with_skill" if skill else "no_skill",
            "raw_env_reward": 10.0 * float(won),
        }
    )
    result.reward = 10.0 * float(won)
    return result
