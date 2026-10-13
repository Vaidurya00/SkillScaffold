"""Generation module for WebShop training and evaluation."""

from .episode import run_episode
from .rollout import run_rollout
from .eval import run_eval
from .generator import (
    build_generator_sample,
    evaluate_downstream,
    generate_generator_sample,
    generate_generator_sample_from_outcome_context,
    generate_generator_sample_from_trajectory,
    resolve_retrieval_query,
    resolve_task_description,
    resolve_task_type,
    task_info_to_sample,
    mark_source_episode_won,
)
from .retrieval import retrieve_tasks
