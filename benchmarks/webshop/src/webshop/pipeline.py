"""Entry points for WebShop training."""

from __future__ import annotations


async def webshop_pipeline(
    args,
    sample,
    sampling_params: dict,
    evaluation: bool = False,
) -> list:
    if evaluation:
        from src.webshop.generate.eval import run_eval

        return await run_eval(args, sample, sampling_params)

    from src.webshop.generate.rollout import run_rollout

    return await run_rollout(args, sample, sampling_params, evaluation)


def webshop_reward_post_process(
    args,
    samples: list,
) -> tuple[list[float], list[float]]:
    rewards = [sample.get_reward_value(args) for sample in samples]
    return rewards, rewards


webshop_stage2_pipeline = webshop_pipeline
webshop_stage2_reward_post_process = webshop_reward_post_process
