"""Entry points for ALFWorld training."""

from __future__ import annotations

import os


async def alfworld_pipeline(
    args,
    sample,
    sampling_params: dict,
    evaluation: bool = False,
) -> list:
    if evaluation:
        from src.alfworld.generate.eval import run_eval

        return await run_eval(args, sample, sampling_params)

    from src.alfworld.generate.rollout import run_rollout

    return await run_rollout(args, sample, sampling_params, evaluation)


def alfworld_reward_post_process(
    args,
    samples: list,
) -> tuple[list[float], list[float]]:
    rewards = [sample.get_reward_value(args) for sample in samples]
    return rewards, rewards


alfworld_stage2_pipeline = alfworld_pipeline
alfworld_stage2_reward_post_process = alfworld_reward_post_process


async def alfworld_three_method_pipeline(
    args,
    sample,
    sampling_params: dict,
    evaluation: bool = False,
) -> list:
    """Dispatch one of the three matched ALFWorld training methods."""

    method = os.environ.get("ALFWORLD_METHOD", "evolving").strip().lower()
    if evaluation:
        protocol = os.environ.get(
            "ALFWORLD_EVAL_PROTOCOL", "matched_no_skill"
        ).strip().lower()
        if protocol == "matched_no_skill":
            from src.alfworld.generate.common_eval import run_eval
        elif protocol == "static_delta_utility_probe":
            from src.alfworld.generate.probe_eval import run_eval
        elif protocol == "kl_centered_delta_rollout":
            from src.alfworld.generate.kl_centered_eval import run_eval
        elif protocol == "local_intervention_route_a":
            from src.alfworld.generate.local_intervention_eval import run_eval
        elif protocol == "native_skill":
            from src.alfworld.generate.eval import run_eval
        elif protocol == "self_extract2_skill":
            from src.alfworld.generate.self_extract_eval import run_eval
        else:
            raise ValueError(f"Unknown ALFWORLD_EVAL_PROTOCOL={protocol!r}")

        return await run_eval(args, sample, sampling_params)
    if method == "vanilla":
        from src.alfworld.generate.vanilla_rollout import run_rollout
    elif method == "delta":
        from src.alfworld.generate.delta_rollout import run_rollout
    elif method == "group_delta":
        from src.alfworld.generate.group_delta_rollout import run_rollout
    elif method == "evolving":
        from src.alfworld.generate.rollout import run_rollout
    else:
        raise ValueError(f"Unknown ALFWORLD_METHOD={method!r}")
    return await run_rollout(args, sample, sampling_params, evaluation)
