"""Entry points for ScienceWorld Delta-card training."""

from __future__ import annotations


async def scienceworld_pipeline(args, sample, sampling_params: dict, evaluation=False):
    from src.scienceworld.e7_audit import save_group, save_eval
    if evaluation:
        from src.scienceworld.generate.episode import run_episode
        result = await run_episode(args, sample, sampling_params, skill=None, evaluation=True)
        result.metadata = result.metadata or {}
        result.reward = 10.0 * float(result.metadata.get("won", False))
        result.metadata["evaluation_protocol"] = "matched_no_skill_pass_at_1"
        save_eval(args, result)
        return result
    from src.scienceworld.generate.group_delta_rollout import run_rollout
    result = await run_rollout(args, sample, sampling_params, evaluation=False)
    save_group(args, sample, result)
    return result


def scienceworld_reward_post_process(args, samples: list):
    rewards = [sample.get_reward_value(args) for sample in samples]
    return rewards, rewards
