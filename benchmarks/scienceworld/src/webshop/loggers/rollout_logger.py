"""Shared rollout logger for WebShop training."""

import logging

import numpy as np
from argparse import Namespace

from slime.utils import logging_utils
from slime.utils.metric_utils import (
    compute_rollout_step,
    compute_statistics,
    dict_add_prefix,
    has_repetition,
)
from slime.utils.misc import group_by
from slime.utils.types import Sample
from src.webshop.dynamics import persist_rollout_dynamics

logger = logging.getLogger(__name__)


def _meta_float(sample: Sample, key: str, default: float | None = None) -> float | None:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    value = metadata.get(key)
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        return default


def _training_reward(sample: Sample) -> float | None:
    value = _meta_float(sample, "raw_training_reward")
    if value is not None:
        return value
    reward = sample.reward
    return float(reward) if isinstance(reward, (int, float)) else None


def _env_reward(sample: Sample) -> float:
    value = _meta_float(sample, "raw_env_reward")
    if value is not None:
        return value
    won = bool((sample.metadata or {}).get("won", False)) if isinstance(sample.metadata, dict) else False
    return 10.0 * float(won)


def _won(sample: Sample) -> bool:
    return bool((sample.metadata or {}).get("won", False)) if isinstance(sample.metadata, dict) else False


def _sample_role(sample: Sample) -> str:
    train_metadata = sample.train_metadata if isinstance(sample.train_metadata, dict) else {}
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    return str(
        train_metadata.get("sample_role") or metadata.get("sample_role") or ""
    ).strip()


def _safe_fraction(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _feature3_candidate_rows(source_scout: Sample) -> list[dict]:
    metadata = source_scout.metadata if isinstance(source_scout.metadata, dict) else {}
    rows = metadata.get("feature3_skill_candidates")
    if not isinstance(rows, list):
        return []
    return [dict(row) for row in rows if isinstance(row, dict)]


def _add_direct_delta_v3_metrics(
    log_dict: dict,
    source_scouts: list[Sample],
    executor: list[Sample],
) -> None:
    """Aggregate Feature-3 dynamics with explicit query/candidate denominators."""

    if not source_scouts:
        return

    prefix = "rollout/v3"
    query_count = len(source_scouts)
    scout_success = [
        sample
        for sample in source_scouts
        if (_meta_float(sample, "initial_scout_any_success", 0.0) or 0.0) > 0.5
    ]
    skill_generation_success = [
        sample
        for sample in source_scouts
        if (_meta_float(sample, "generator_valid_candidate_count", 0.0) or 0.0)
        > 0.0
    ]
    delta_computed = [
        sample
        for sample in source_scouts
        if (_meta_float(sample, "generator_delta_computed", 0.0) or 0.0) > 0.5
    ]
    positive_delta_queries = [
        sample
        for sample in source_scouts
        if (_meta_float(sample, "generator_positive_delta_count", 0.0) or 0.0)
        > 0.0
    ]
    adopted = [
        sample
        for sample in source_scouts
        if (_meta_float(sample, "selected_skill_injected", 0.0) or 0.0) > 0.5
    ]

    log_dict[f"{prefix}/query/total_count"] = query_count
    log_dict[f"{prefix}/query/scout_success_count"] = len(scout_success)
    log_dict[f"{prefix}/query/scout_success_frac_all_queries"] = _safe_fraction(
        len(scout_success), query_count
    )
    log_dict[f"{prefix}/query/skill_generation_success_count"] = len(
        skill_generation_success
    )
    log_dict[f"{prefix}/query/skill_generation_success_frac_all_queries"] = (
        _safe_fraction(len(skill_generation_success), query_count)
    )
    log_dict[
        f"{prefix}/query/skill_generation_success_frac_scout_success_queries"
    ] = _safe_fraction(len(skill_generation_success), len(scout_success))
    log_dict[f"{prefix}/query/delta_computed_count"] = len(delta_computed)
    log_dict[f"{prefix}/query/delta_computed_frac_all_queries"] = _safe_fraction(
        len(delta_computed), query_count
    )
    log_dict[f"{prefix}/query/delta_computed_frac_scout_success_queries"] = (
        _safe_fraction(len(delta_computed), len(scout_success))
    )
    log_dict[f"{prefix}/query/positive_delta_count"] = len(
        positive_delta_queries
    )
    log_dict[f"{prefix}/query/positive_delta_frac_all_queries"] = _safe_fraction(
        len(positive_delta_queries), query_count
    )
    log_dict[f"{prefix}/query/positive_delta_frac_delta_computed_queries"] = (
        _safe_fraction(len(positive_delta_queries), len(delta_computed))
    )
    log_dict[f"{prefix}/query/skill_adopted_count"] = len(adopted)
    log_dict[f"{prefix}/query/skill_adopted_frac_all_queries"] = _safe_fraction(
        len(adopted), query_count
    )
    log_dict[f"{prefix}/query/skill_adopted_frac_delta_computed_queries"] = (
        _safe_fraction(len(adopted), len(delta_computed))
    )

    attempted_count = int(
        sum(_meta_float(sample, "generator_candidate_attempt_count", 0.0) or 0.0 for sample in source_scouts)
    )
    generated_count = int(
        sum(_meta_float(sample, "generator_candidate_sample_count", 0.0) or 0.0 for sample in source_scouts)
    )
    valid_count = int(
        sum(_meta_float(sample, "generator_valid_candidate_count", 0.0) or 0.0 for sample in source_scouts)
    )
    delta_available_count = int(
        sum(_meta_float(sample, "generator_delta_available_count", 0.0) or 0.0 for sample in source_scouts)
    )
    positive_candidate_count = int(
        sum(_meta_float(sample, "generator_positive_delta_count", 0.0) or 0.0 for sample in source_scouts)
    )
    log_dict[f"{prefix}/candidate/attempt_count"] = attempted_count
    log_dict[f"{prefix}/candidate/generated_count"] = generated_count
    log_dict[f"{prefix}/candidate/generated_frac_attempted"] = _safe_fraction(
        generated_count, attempted_count
    )
    log_dict[f"{prefix}/candidate/format_valid_count"] = valid_count
    log_dict[f"{prefix}/candidate/format_valid_frac_attempted"] = _safe_fraction(
        valid_count, attempted_count
    )
    log_dict[f"{prefix}/candidate/delta_available_count"] = delta_available_count
    log_dict[f"{prefix}/candidate/delta_available_frac_valid_candidates"] = (
        _safe_fraction(delta_available_count, valid_count)
    )
    log_dict[f"{prefix}/candidate/positive_delta_count"] = positive_candidate_count
    log_dict[f"{prefix}/candidate/positive_delta_frac_available_candidates"] = (
        _safe_fraction(positive_candidate_count, delta_available_count)
    )

    prompt_checks = [
        _meta_float(sample, "generator_prompts_byte_identical", 0.0) or 0.0
        for sample in scout_success
    ]
    log_dict[f"{prefix}/candidate/byte_identical_prompt_frac_scout_success_queries"] = (
        float(np.mean(prompt_checks)) if prompt_checks else 0.0
    )

    all_deltas: list[float] = []
    per_query_max_deltas: list[float] = []
    selected_deltas: list[float] = []
    optimizer_group_reward_stds: list[float] = []
    optimizer_group_count = 0
    all_zero_group_count = 0
    zero_std_group_count = 0
    for source in source_scouts:
        rows = _feature3_candidate_rows(source)
        query_deltas = []
        clipped_rewards = []
        for row in rows:
            if float(row.get("delta_available", 0.0) or 0.0) > 0.5:
                try:
                    delta = float(row.get("delta"))
                except (TypeError, ValueError):
                    continue
                if np.isfinite(delta):
                    query_deltas.append(delta)
                    all_deltas.append(delta)
            if row.get("positive_clipped_reward") is not None:
                try:
                    clipped = float(row["positive_clipped_reward"])
                except (TypeError, ValueError):
                    continue
                if np.isfinite(clipped):
                    clipped_rewards.append(clipped)
        if query_deltas:
            per_query_max_deltas.append(max(query_deltas))
        selected_delta = _meta_float(source, "selected_skill_delta")
        if selected_delta is not None and np.isfinite(selected_delta):
            selected_deltas.append(selected_delta)
        optimizer_enabled = (
            _meta_float(source, "generator_optimizer_enabled", 0.0) or 0.0
        ) > 0.5
        if optimizer_enabled and len(clipped_rewards) == 4:
            optimizer_group_count += 1
            reward_std = float(np.std(clipped_rewards))
            optimizer_group_reward_stds.append(reward_std)
            if all(abs(value) <= 1e-12 for value in clipped_rewards):
                all_zero_group_count += 1
            if reward_std <= 1e-12:
                zero_std_group_count += 1

    if all_deltas:
        log_dict |= dict_add_prefix(
            compute_statistics(all_deltas),
            f"{prefix}/delta/all_available_candidates/",
        )
    if per_query_max_deltas:
        log_dict |= dict_add_prefix(
            compute_statistics(per_query_max_deltas),
            f"{prefix}/delta/per_query_max/",
        )
    if selected_deltas:
        log_dict |= dict_add_prefix(
            compute_statistics(selected_deltas),
            f"{prefix}/delta/selected_positive/",
        )
    log_dict[f"{prefix}/generator_signal/optimizer_group_count"] = optimizer_group_count
    log_dict[f"{prefix}/generator_signal/all_zero_group_count"] = all_zero_group_count
    log_dict[f"{prefix}/generator_signal/all_zero_group_frac_optimizer_groups"] = (
        _safe_fraction(all_zero_group_count, optimizer_group_count)
    )
    log_dict[f"{prefix}/generator_signal/zero_std_group_count"] = zero_std_group_count
    log_dict[f"{prefix}/generator_signal/zero_std_group_frac_optimizer_groups"] = (
        _safe_fraction(zero_std_group_count, optimizer_group_count)
    )
    if optimizer_group_reward_stds:
        log_dict |= dict_add_prefix(
            compute_statistics(optimizer_group_reward_stds),
            f"{prefix}/generator_signal/raw_group_reward_std/",
        )

    initial = [
        sample
        for sample in executor
        if str((sample.metadata or {}).get("executor_rollout_phase") or "")
        == "initial_scout"
    ]
    post = [
        sample
        for sample in executor
        if str((sample.metadata or {}).get("executor_rollout_phase") or "")
        == "post_scout"
    ]
    post_skill = [
        sample
        for sample in post
        if str((sample.metadata or {}).get("executor_variant") or "") == "with_skill"
    ]
    post_no_skill = [
        sample
        for sample in post
        if str((sample.metadata or {}).get("executor_variant") or "")
        != "with_skill"
    ]
    for name, subset in (
        ("initial_scout", initial),
        ("post_scout", post),
        ("post_with_skill", post_skill),
        ("post_no_skill", post_no_skill),
    ):
        log_dict[f"{prefix}/executor/{name}_count"] = len(subset)
        log_dict[f"{prefix}/executor/{name}_win_rate"] = (
            float(np.mean([1.0 if _won(sample) else 0.0 for sample in subset]))
            if subset
            else 0.0
        )


def log_rollout_data(
    rollout_id: int,
    args: Namespace,
    samples: list[Sample],
    rollout_extra_metrics: dict | None,
    rollout_time: float,
) -> bool:
    if any((sample.metadata or {}).get("method") == "group_delta_v3" for sample in samples):
        from src.alfworld.loggers.rollout_logger import log_rollout_data as log_e7
        return log_e7(rollout_id, args, samples, rollout_extra_metrics, rollout_time)
    generator = [sample for sample in samples if _sample_role(sample) == "generator"]
    executor = [
        sample
        for sample in samples
        if _sample_role(sample) in {"executor", "downstream_executor"}
    ]

    log_dict: dict = {}
    if rollout_extra_metrics:
        log_dict.update(rollout_extra_metrics)

    log_dict["rollout/final_sample_count"] = len(samples)
    log_dict["rollout/final_generator_count"] = len(generator)
    log_dict["rollout/final_executor_count"] = len(executor)

    if generator:
        ext_training_rewards = [
            reward for sample in generator if (reward := _training_reward(sample)) is not None
        ]
        ext_normalized_rewards = [
            reward
            for sample in generator
            if (reward := _meta_float(sample, "normalized_generator_reward"))
            is not None
        ]
        ext_downstream_executor_rewards = [
            reward
            for sample in generator
            if (reward := _meta_float(sample, "raw_mean_downstream_executor_reward")) is not None
        ]
        ext_win_rates = [
            win_rate
            for sample in generator
            if (win_rate := _meta_float(sample, "downstream_win_rate")) is not None
        ]
        ext_format_scores = [
            score for sample in generator if (score := _meta_float(sample, "format_score")) is not None
        ]
        ext_format_penalties = [
            penalty for sample in generator if (penalty := _meta_float(sample, "format_penalty")) is not None
        ]
        ext_skill_diff_rewards = [
            reward
            for sample in generator
            if (reward := _meta_float(sample, "skill_diff_reward")) is not None
        ]
        ext_skill_diff_available = [
            available
            for sample in generator
            if (available := _meta_float(sample, "skill_diff_available")) is not None
        ]
        ext_skill_diff_base_logprobs = [
            value
            for sample in generator
            if (value := _meta_float(sample, "skill_diff_base_mean_logprob")) is not None
        ]
        ext_skill_diff_with_logprobs = [
            value
            for sample in generator
            if (value := _meta_float(sample, "skill_diff_with_skill_mean_logprob")) is not None
        ]
        ext_skill_diff_similarities = [
            value
            for sample in generator
            if (value := _meta_float(sample, "retrieved_similarity")) is not None
        ]
        ext_skill_diff_token_counts = [
            value
            for sample in generator
            if (value := _meta_float(sample, "skill_diff_scored_token_count")) is not None
        ]
        ext_skill_diff_positive = [
            1.0 if reward > 0.0 else 0.0
            for sample in generator
            if (_meta_float(sample, "skill_diff_available", 0.0) or 0.0) > 0.5
            if (reward := _meta_float(sample, "skill_diff_reward")) is not None
        ]
        ext_null_clipped_rewards = [
            value
            for sample in generator
            if (value := _meta_float(sample, "generator_reward_after_null_clip"))
            is not None
        ]
        ext_nonpositive_clipped = [
            value
            for sample in generator
            if (value := _meta_float(sample, "reward_clipped_by_nonpositive_diff"))
            is not None
        ]
        ext_group_update_skipped = [
            value
            for sample in generator
            if (value := _meta_float(sample, "generator_group_update_skipped"))
            is not None
        ]
        ext_skill_language_audit_applied = [
            applied
            for sample in generator
            if (applied := _meta_float(sample, "skill_language_audit_applied")) is not None
        ]
        ext_skill_language_audit_pass = [
            passed
            for sample in generator
            if (_meta_float(sample, "skill_language_audit_applied", 0.0) or 0.0) > 0.5
            if (passed := _meta_float(sample, "skill_language_audit_pass")) is not None
        ]
        ext_skill_language_audit_penalties = [
            penalty
            for sample in generator
            if (penalty := _meta_float(sample, "skill_language_audit_penalty")) is not None
            if abs(penalty) > 1e-8
        ]
        ext_skill_language_audit_reward_zeroed = [
            zeroed
            for sample in generator
            if (zeroed := _meta_float(sample, "skill_language_audit_reward_zeroed")) is not None
        ]
        ext_skill_language_audit_uncommon_char = [
            flag
            for sample in generator
            if (_meta_float(sample, "skill_language_audit_applied", 0.0) or 0.0) > 0.5
            if (flag := _meta_float(sample, "skill_language_audit_has_uncommon_characters")) is not None
        ]
        ext_skill_language_audit_non_english_char = [
            flag
            for sample in generator
            if (_meta_float(sample, "skill_language_audit_applied", 0.0) or 0.0) > 0.5
            if (flag := _meta_float(sample, "skill_language_audit_has_non_english_characters")) is not None
        ]
        ext_resp_lens = [sample.effective_response_length for sample in generator]
        ext_valid_skill = [
            1.0
            if str((sample.metadata or {}).get("skill") or "").strip()
            and float((sample.metadata or {}).get("format_score", 0.0) or 0.0) >= 1.0
            else 0.0
            for sample in generator
        ]
        ext_positive_diff_counts = [
            value
            for sample in generator
            if (value := _meta_float(sample, "generator_positive_diff_count")) is not None
        ]
        ext_update_sample_counts = [
            value
            for sample in generator
            if (value := _meta_float(sample, "generator_update_sample_count")) is not None
        ]

        if ext_training_rewards:
            log_dict |= dict_add_prefix(
                compute_statistics(ext_training_rewards),
                "rollout/generator/training_reward/",
            )
        if ext_normalized_rewards:
            log_dict |= dict_add_prefix(
                compute_statistics(ext_normalized_rewards),
                "rollout/generator/normalized_reward/",
            )
        if ext_downstream_executor_rewards:
            log_dict |= dict_add_prefix(
                compute_statistics(ext_downstream_executor_rewards),
                "rollout/generator/mean_downstream_executor_reward/",
            )
        if ext_win_rates:
            log_dict |= dict_add_prefix(
                compute_statistics(ext_win_rates),
                "rollout/generator/downstream_win_rate/",
            )
        if ext_format_scores:
            log_dict |= dict_add_prefix(
                compute_statistics(ext_format_scores),
                "rollout/generator/format_score/",
            )
        if ext_format_penalties:
            log_dict |= dict_add_prefix(
                compute_statistics(ext_format_penalties),
                "rollout/generator/format_penalty/",
            )
        if ext_skill_diff_rewards:
            log_dict |= dict_add_prefix(
                compute_statistics(ext_skill_diff_rewards),
                "rollout/generator/skill_diff/reward/",
            )
        if ext_skill_diff_available:
            log_dict["rollout/generator/skill_diff/available_frac"] = float(
                np.mean(ext_skill_diff_available)
            )
        if ext_skill_diff_positive:
            log_dict["rollout/generator/skill_diff/positive_count"] = int(
                sum(ext_skill_diff_positive)
            )
            log_dict["rollout/generator/skill_diff/positive_frac"] = float(
                np.mean(ext_skill_diff_positive)
            )
        log_dict["rollout/generator/valid_skill_frac"] = float(np.mean(ext_valid_skill))
        if ext_positive_diff_counts:
            log_dict["rollout/generator/positive_diff_per_group_mean"] = float(
                np.mean(ext_positive_diff_counts)
            )
        if ext_update_sample_counts:
            log_dict["rollout/generator/update_sample_count_mean"] = float(
                np.mean(ext_update_sample_counts)
            )
        if ext_null_clipped_rewards:
            log_dict |= dict_add_prefix(
                compute_statistics(ext_null_clipped_rewards),
                "rollout/generator/null_clipped_reward/",
            )
        if ext_nonpositive_clipped:
            log_dict["rollout/generator/skill_diff/nonpositive_clipped_frac"] = float(
                np.mean(ext_nonpositive_clipped)
            )
        if ext_group_update_skipped:
            log_dict["rollout/generator/group_update_skipped_frac"] = float(
                np.mean(ext_group_update_skipped)
            )
        if ext_skill_diff_base_logprobs:
            log_dict |= dict_add_prefix(
                compute_statistics(ext_skill_diff_base_logprobs),
                "rollout/generator/skill_diff/base_mean_logprob/",
            )
        if ext_skill_diff_with_logprobs:
            log_dict |= dict_add_prefix(
                compute_statistics(ext_skill_diff_with_logprobs),
                "rollout/generator/skill_diff/with_skill_mean_logprob/",
            )
        if ext_skill_diff_similarities:
            log_dict |= dict_add_prefix(
                compute_statistics(ext_skill_diff_similarities),
                "rollout/generator/skill_diff/retrieval_similarity/",
            )
        if ext_skill_diff_token_counts:
            log_dict |= dict_add_prefix(
                compute_statistics(ext_skill_diff_token_counts),
                "rollout/generator/skill_diff/scored_token_count/",
            )
        if ext_skill_language_audit_applied:
            log_dict["rollout/generator/skill_language_audit/applied_frac"] = float(
                np.mean(ext_skill_language_audit_applied)
            )
        if ext_skill_language_audit_pass:
            log_dict["rollout/generator/skill_language_audit/pass_rate"] = float(
                np.mean(ext_skill_language_audit_pass)
            )
        if ext_skill_language_audit_penalties:
            log_dict |= dict_add_prefix(
                compute_statistics(ext_skill_language_audit_penalties),
                "rollout/generator/skill_language_audit/penalty/",
            )
        if ext_skill_language_audit_reward_zeroed:
            log_dict["rollout/generator/skill_language_audit/reward_zeroed_frac"] = float(
                np.mean(ext_skill_language_audit_reward_zeroed)
            )
        if ext_skill_language_audit_uncommon_char:
            log_dict["rollout/generator/skill_language_audit/uncommon_char_rate"] = float(
                np.mean(ext_skill_language_audit_uncommon_char)
            )
        if ext_skill_language_audit_non_english_char:
            log_dict["rollout/generator/skill_language_audit/non_english_char_rate"] = float(
                np.mean(ext_skill_language_audit_non_english_char)
            )
        log_dict |= dict_add_prefix(
            compute_statistics(ext_resp_lens),
            "rollout/generator/response_len/",
        )
        log_dict["rollout/generator/count"] = len(generator)
        log_dict["rollout/generator/truncated_ratio"] = float(
            np.mean([int(sample.status == Sample.Status.TRUNCATED) for sample in generator])
        )
        log_dict["rollout/generator/repetition_frac"] = float(
            np.mean([int(has_repetition(sample.response)) for sample in generator])
        )

    if executor:
        executor_training_rewards = [
            reward for sample in executor if (reward := _training_reward(sample)) is not None
        ]
        executor_env_rewards = [_env_reward(sample) for sample in executor]
        executor_resp_lens = [sample.response_length for sample in executor]
        executor_eff_resp_lens = [sample.effective_response_length for sample in executor]
        executor_num_steps = [_meta_float(sample, "num_steps", 0.0) or 0.0 for sample in executor]
        executor_workflow_compliance = [
            float(_meta_float(sample, "workflow_compliance", 0.0) or 0.0)
            for sample in executor
            if _meta_float(sample, "workflow_compliance") is not None
        ]
        executor_skill_language_audit_applied = [
            applied
            for sample in executor
            if (applied := _meta_float(sample, "skill_language_audit_applied")) is not None
        ]
        executor_skill_language_audit_pass = [
            passed
            for sample in executor
            if (_meta_float(sample, "skill_language_audit_applied", 0.0) or 0.0) > 0.5
            if (passed := _meta_float(sample, "skill_language_audit_pass")) is not None
        ]
        executor_skill_language_audit_penalty = [
            penalty
            for sample in executor
            if (penalty := _meta_float(sample, "skill_language_audit_penalty")) is not None
            if abs(penalty) > 1e-8
        ]
        executor_skill_language_audit_reward_zeroed = [
            zeroed
            for sample in executor
            if (zeroed := _meta_float(sample, "skill_language_audit_reward_zeroed")) is not None
        ]
        executor_skill_language_audit_uncommon_char = [
            flag
            for sample in executor
            if (_meta_float(sample, "skill_language_audit_applied", 0.0) or 0.0) > 0.5
            if (flag := _meta_float(sample, "skill_language_audit_has_uncommon_characters")) is not None
        ]

        if executor_training_rewards:
            log_dict |= dict_add_prefix(
                compute_statistics(executor_training_rewards),
                "rollout/executor/training_reward/",
            )
        if executor_env_rewards:
            log_dict |= dict_add_prefix(
                compute_statistics(executor_env_rewards),
                "rollout/executor/env_reward/",
            )
        if executor_workflow_compliance:
            log_dict |= dict_add_prefix(
                compute_statistics(executor_workflow_compliance),
                "rollout/executor/workflow_compliance/",
            )
        if executor_skill_language_audit_applied:
            log_dict["rollout/executor/skill_language_audit/applied_frac"] = float(
                np.mean(executor_skill_language_audit_applied)
            )
        if executor_skill_language_audit_pass:
            log_dict["rollout/executor/skill_language_audit/pass_rate"] = float(
                np.mean(executor_skill_language_audit_pass)
            )
        if executor_skill_language_audit_penalty:
            log_dict |= dict_add_prefix(
                compute_statistics(executor_skill_language_audit_penalty),
                "rollout/executor/skill_language_audit/penalty/",
            )
        if executor_skill_language_audit_reward_zeroed:
            log_dict["rollout/executor/skill_language_audit/reward_zeroed_frac"] = float(
                np.mean(executor_skill_language_audit_reward_zeroed)
            )
        if executor_skill_language_audit_uncommon_char:
            log_dict["rollout/executor/skill_language_audit/uncommon_char_rate"] = float(
                np.mean(executor_skill_language_audit_uncommon_char)
            )
        log_dict |= dict_add_prefix(compute_statistics(executor_resp_lens), "rollout/executor/response_len/")
        log_dict |= dict_add_prefix(
            compute_statistics(executor_eff_resp_lens),
            "rollout/executor/effective_response_len/",
        )
        log_dict |= dict_add_prefix(compute_statistics(executor_num_steps), "rollout/executor/num_steps/")
        log_dict["rollout/executor/count"] = len(executor)
        log_dict["rollout/executor/win_rate"] = float(np.mean([1.0 if _won(sample) else 0.0 for sample in executor]))
        log_dict["rollout/executor/truncated_ratio"] = float(
            np.mean([int(sample.status == Sample.Status.TRUNCATED) for sample in executor])
        )
        log_dict["rollout/executor/repetition_frac"] = float(
            np.mean([int(has_repetition(sample.response)) for sample in executor])
        )

        source_scouts = [
            sample
            for sample in executor
            if _meta_float(sample, "executor_group_position", -1.0) == 0.0
        ]
        if source_scouts:
            direct_v3 = any(
                str((sample.metadata or {}).get("rollout_variant") or "")
                == "four_scout_direct_delta_v3"
                for sample in source_scouts
            )
            if direct_v3:
                _add_direct_delta_v3_metrics(log_dict, source_scouts, executor)
            mode_names = (
                (
                    "scout_no_success_no_skill",
                    "scout_success_no_positive_delta_no_skill",
                    "scout_success_positive_delta_skill",
                )
                if direct_v3
                else (
                    "cold_no_candidate",
                    "cold_below_threshold",
                    "warm_with_skill",
                    "all_diff_nonpositive",
                    "skill_scoring_failed",
                )
            )
            modes = [
                str((sample.metadata or {}).get("executor_skill_mode") or "")
                for sample in source_scouts
            ]
            for mode_name in mode_names:
                count = sum(mode == mode_name for mode in modes)
                log_dict[f"rollout/executor/mode/{mode_name}_count"] = count
                log_dict[f"rollout/executor/mode/{mode_name}_frac"] = (
                    float(count) / len(source_scouts)
                )
            if not direct_v3:
                gate_similarities = [
                    value
                    for sample in source_scouts
                    if (value := _meta_float(sample, "retrieved_similarity"))
                    is not None
                ]
                if gate_similarities:
                    log_dict |= dict_add_prefix(
                        compute_statistics(gate_similarities),
                        "rollout/executor/retrieval_gate/similarity/",
                    )
                gate_open = [
                    float(_meta_float(sample, "retrieval_gate_open", 0.0) or 0.0)
                    for sample in source_scouts
                ]
                log_dict["rollout/executor/retrieval_gate/open_frac"] = float(
                    np.mean(gate_open)
                )
                category_scouts: dict[str, list[Sample]] = {}
                for sample in source_scouts:
                    metadata = (
                        sample.metadata if isinstance(sample.metadata, dict) else {}
                    )
                    category = str(metadata.get("category") or "unknown")
                    category_scouts.setdefault(category, []).append(sample)
                for category, category_samples in sorted(category_scouts.items()):
                    category_open = [
                        float(
                            _meta_float(sample, "retrieval_gate_open", 0.0) or 0.0
                        )
                        for sample in category_samples
                    ]
                    log_dict[
                        f"rollout/executor/retrieval_gate/category/{category}/open_frac"
                    ] = float(np.mean(category_open))

        executor_groups = group_by(executor, lambda sample: sample.group_index)
        zero_std_groups = 0
        for group in executor_groups.values():
            rewards = [_training_reward(sample) for sample in group]
            rewards = [reward for reward in rewards if reward is not None]
            if len(rewards) > 1 and len(set(rewards)) == 1:
                zero_std_groups += 1
        log_dict["rollout/executor/zero_std_group_count"] = zero_std_groups

        task_type_groups: dict[str, list[Sample]] = {}
        for sample in executor:
            task_type = (
                str(sample.label.get("task_type", "unknown"))
                if isinstance(sample.label, dict)
                else "unknown"
            )
            task_type_groups.setdefault(task_type, []).append(sample)
        for task_type, task_samples in sorted(task_type_groups.items()):
            log_dict[f"rollout/executor/{task_type}/count"] = len(task_samples)
            log_dict[f"rollout/executor/{task_type}/win_rate"] = float(
                np.mean([1.0 if _won(sample) else 0.0 for sample in task_samples])
            )
            task_rewards = [
                reward for sample in task_samples if (reward := _training_reward(sample)) is not None
            ]
            if task_rewards:
                log_dict |= dict_add_prefix(
                    compute_statistics(task_rewards),
                    f"rollout/executor/{task_type}/training_reward/",
                )
            task_steps = [_meta_float(sample, "num_steps", 0.0) or 0.0 for sample in task_samples]
            if task_steps:
                log_dict |= dict_add_prefix(
                    compute_statistics(task_steps),
                    f"rollout/executor/{task_type}/num_steps/",
                )

    source_episode_won = [
        float(_meta_float(sample, "source_episode_won", 0.0) or 0.0)
        for sample in generator
        if _meta_float(sample, "source_episode_won") is not None
    ]
    if source_episode_won:
        log_dict["rollout/source_episode/win_rate"] = float(np.mean(source_episode_won))

    log_dict["perf/rollout_time"] = rollout_time
    all_resp_lens = [sample.response_length for sample in samples]
    eff_resp_lens = [sample.effective_response_length for sample in samples]
    non_gen_times = [sample.non_generation_time for sample in samples]

    if non_gen_times and max(non_gen_times) > 0:
        log_dict |= dict_add_prefix(
            compute_statistics(non_gen_times),
            "perf/non_generation_time/",
        )

    if hasattr(args, "rollout_num_gpus") and args.rollout_num_gpus is not None:
        log_dict["perf/tokens_per_gpu_per_sec"] = (
            sum(all_resp_lens) / rollout_time / args.rollout_num_gpus
        )
        log_dict["perf/effective_tokens_per_gpu_per_sec"] = (
            sum(eff_resp_lens) / rollout_time / args.rollout_num_gpus
        )

    if all_resp_lens:
        log_dict["perf/longest_sample_tokens_per_sec"] = max(all_resp_lens) / rollout_time

    if executor:
        log_dict["rollout/executor/reference_skill_prompt_frac"] = float(
            np.mean(["Reference Skill" in str(sample.prompt or "") for sample in executor])
        )
        log_dict["rollout/executor/closed_empty_think_prompt_frac"] = float(
            np.mean(["<think>\n\n</think>" in str(sample.prompt or "") for sample in executor])
        )
    if generator:
        log_dict["rollout/generator/response_begins_think_frac"] = float(
            np.mean(
                [str(sample.response or "").lstrip().startswith("<think>") for sample in generator]
            )
        )
        log_dict["rollout/generator/response_begins_json_frac"] = float(
            np.mean([str(sample.response or "").lstrip().startswith("{") for sample in generator])
        )

    step = compute_rollout_step(args, rollout_id)
    log_dict["rollout/step"] = step

    logger.info(f"rollout {rollout_id}: {log_dict}")
    logging_utils.log(args, log_dict, step_key="rollout/step")
    persist_rollout_dynamics(
        args,
        rollout_id=rollout_id,
        samples=samples,
        metrics=log_dict,
        buffer_snapshot=None,
    )
    return True
