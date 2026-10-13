"""Always-on validity filtering for Tau2 rollout groups."""

from __future__ import annotations

import os

from slime.rollout.filter_hub.base_types import DynamicFilterOutput
from slime.utils.types import Sample


def keep_valid_tau2_group(args, samples: list[Sample], **kwargs) -> DynamicFilterOutput:
    """Resample groups containing infrastructure failures or zero-token samples.

    Content/format failures that contain real sampled tokens remain valid
    negative examples.  Optional DAPO zero-std filtering retains the launcher's
    historical ``USE_DYNAMIC_FILTER=1`` behavior.
    """

    for sample in samples:
        if sample.remove_sample or sample.response_length <= 0:
            reason = (sample.metadata or {}).get("termination_reason", "invalid_sample")
            return DynamicFilterOutput(keep=False, reason=str(reason))

    if os.environ.get("TAU2_DROP_ZERO_STD_GROUPS", "0") == "1":
        rewards = [float(sample.get_reward_value(args)) for sample in samples]
        if max(rewards) - min(rewards) <= 1e-6:
            return DynamicFilterOutput(
                keep=False,
                reason=f"zero_std_{round(rewards[0], 1)}",
            )

    return DynamicFilterOutput(keep=True, reason=None)
