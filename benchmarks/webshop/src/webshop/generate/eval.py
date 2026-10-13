"""WebShop evaluation without retrieval or skill injection."""

from slime.utils.types import Sample

from src.webshop.generate.episode import run_episode
from src.webshop.utils.common import ensure_metadata, maybe_print_random_sample


async def run_eval(args, sample: Sample, sampling_params: dict) -> Sample:
    """Run one target episode and score its actual environment outcome."""
    result = await run_episode(
        args, sample=sample, sampling_params=sampling_params,
        skill=None, evaluation=True,
    )
    metadata = ensure_metadata(result)
    won = 1.0 if metadata.get("won", False) else 0.0
    result.reward = 10.0 * won
    metadata["won_no_skill"] = won
    metadata["num_steps_no_skill"] = metadata.get("num_steps", 0)
    metadata["evaluation_protocol"] = "no_skill"
    maybe_print_random_sample(result, tag="eval/no_skill")
    return result
