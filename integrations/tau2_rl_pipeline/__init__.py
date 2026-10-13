"""Tau2 RL training pipeline for slime."""

from typing import Any

__all__ = ["generate", "tau2_reward_post_process"]


def __getattr__(name: str) -> Any:
    """Keep the package import lightweight while preserving public exports."""
    if name == "generate":
        from tau2_rl_pipeline.rollout import generate

        return generate
    if name == "tau2_reward_post_process":
        from tau2_rl_pipeline.reward import tau2_reward_post_process

        return tau2_reward_post_process
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
