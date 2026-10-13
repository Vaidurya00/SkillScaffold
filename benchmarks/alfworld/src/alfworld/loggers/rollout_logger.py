"""Shared rollout logger for ALFWorld training."""

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
        train_metadata.get("sample_role")
        or metadata.get("trajectory_role")
        or ("generator" if sample.group_index == 0 else "executor")
    )


def log_rollout_data(
    rollout_id: int,
    args: Namespace,
    samples: list[Sample],
    rollout_extra_metrics: dict | None,
    rollout_time: float,
) -> bool:
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

        if ext_training_rewards:
            log_dict |= dict_add_prefix(
                compute_statistics(ext_training_rewards),
                "rollout/generator/training_reward/",
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
        delta_values = [
            value
            for sample in generator
            if (value := _meta_float(sample, "skill_diff_reward")) is not None
        ]
        if delta_values:
            log_dict |= dict_add_prefix(
                compute_statistics(delta_values),
                "rollout/generator/delta/",
            )
            log_dict["rollout/generator/delta_positive_frac"] = float(
                np.mean([value > 0.0 for value in delta_values])
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
        selected_skill_flags = [
            value
            for sample in executor
            if (value := _meta_float(sample, "selected_skill_injected")) is not None
        ]
        if selected_skill_flags:
            log_dict["rollout/delta/selected_skill_injected_rate"] = float(
                np.mean(selected_skill_flags)
            )
        scout_success_counts = [
            value
            for sample in executor
            if (value := _meta_float(sample, "initial_scout_success_count")) is not None
        ]
        if scout_success_counts:
            log_dict["rollout/delta/initial_scout_success_count_mean"] = float(
                np.mean(scout_success_counts)
            )

        executor_groups = group_by(executor, lambda sample: sample.group_index)
        zero_std_groups = 0
        for group in executor_groups.values():
            rewards = [_training_reward(sample) for sample in group]
            rewards = [reward for reward in rewards if reward is not None]
            if len(rewards) > 1 and len(set(rewards)) == 1:
                zero_std_groups += 1
        log_dict["rollout/executor/zero_std_group_count"] = zero_std_groups

        scout_executor = [
            sample for sample in executor
            if str((sample.metadata or {}).get("executor_phase") or "") == "scout"
        ]
        post_executor = [
            sample for sample in executor
            if str((sample.metadata or {}).get("executor_phase") or "") == "post_scout"
        ]
        if scout_executor:
            log_dict["rollout/group_delta/scout_win_rate"] = float(
                np.mean([1.0 if _won(sample) else 0.0 for sample in scout_executor])
            )
        if post_executor:
            log_dict["rollout/group_delta/post_scout_win_rate"] = float(
                np.mean([1.0 if _won(sample) else 0.0 for sample in post_executor])
            )
            injected_post = [
                sample for sample in post_executor
                if (_meta_float(sample, "executor_route_skill_injected", 0.0) or 0.0) > 0.5
            ]
            no_skill_post = [
                sample for sample in post_executor
                if (_meta_float(sample, "executor_route_skill_injected", 0.0) or 0.0) <= 0.5
            ]
            if injected_post:
                log_dict["rollout/group_delta/injected_post_win_rate"] = float(
                    np.mean([1.0 if _won(sample) else 0.0 for sample in injected_post])
                )
            if no_skill_post:
                log_dict["rollout/group_delta/no_skill_post_win_rate"] = float(
                    np.mean([1.0 if _won(sample) else 0.0 for sample in no_skill_post])
                )

        # Group-level fields are repeated on eight executor samples.  Read one
        # representative per group so W&B statistics are query-weighted.
        group_representatives = [group[0] for group in executor_groups.values() if group]
        group_delta_values = []
        success_lifts = []
        failure_lifts = []
        aggregation_modes = []
        mask_leaks = []
        mask_mismatches = []
        scored_tokens = []
        model_tokens = []
        environment_tokens = []
        valid_counts = []
        available_counts = []
        positive_counts = []
        optimizer_card_counts = []
        all_nonpositive_flags = []
        for representative in group_representatives:
            metadata = representative.metadata if isinstance(representative.metadata, dict) else {}
            if metadata.get("method") != "group_delta_v3":
                continue
            group_delta_values.extend(
                float(value) for value in (metadata.get("group_delta_scores") or []) if value is not None
            )
            success_lifts.extend(
                float(value)
                for value in (metadata.get("group_delta_success_mean_lifts") or [])
                if value is not None
            )
            failure_lifts.extend(
                float(value)
                for value in (metadata.get("group_delta_failure_mean_lifts") or [])
                if value is not None
            )
            aggregation_modes.extend(metadata.get("group_delta_aggregation_modes") or [])
            mask_leaks.append(float(metadata.get("group_delta_scored_environment_token_count", 0) or 0))
            mask_mismatches.append(float(metadata.get("group_delta_loss_mask_mismatch_count", 0) or 0))
            scored_tokens.append(float(metadata.get("group_delta_scored_token_count", 0) or 0))
            model_tokens.append(float(metadata.get("group_delta_model_token_count", 0) or 0))
            environment_tokens.append(float(metadata.get("group_delta_environment_token_count", 0) or 0))
            valid_counts.append(float(metadata.get("generator_valid_count", 0) or 0))
            available_counts.append(float(metadata.get("generator_available_delta_count", 0) or 0))
            positive_counts.append(float(metadata.get("generator_positive_delta_count", 0) or 0))
            if metadata.get("generator_optimizer_card_count") is not None:
                optimizer_card_counts.append(
                    float(metadata.get("generator_optimizer_card_count", 0) or 0)
                )
            if metadata.get("generator_all_nonpositive") is not None:
                all_nonpositive_flags.append(
                    float(metadata.get("generator_all_nonpositive", 0.0) or 0.0)
                )
        if group_delta_values:
            log_dict |= dict_add_prefix(
                compute_statistics(group_delta_values), "rollout/group_delta/card_delta/"
            )
            log_dict["rollout/group_delta/card_delta_positive_frac"] = float(
                np.mean([value > 0 for value in group_delta_values])
            )
        if success_lifts:
            log_dict |= dict_add_prefix(
                compute_statistics(success_lifts), "rollout/group_delta/success_mean_lift/"
            )
        if failure_lifts:
            log_dict |= dict_add_prefix(
                compute_statistics(failure_lifts), "rollout/group_delta/failure_mean_lift/"
            )
        if aggregation_modes:
            for mode in (
                "mixed_equal_class_weight",
                "all_success_full_weight",
                "all_failure_full_weight",
            ):
                log_dict[f"rollout/group_delta/aggregation/{mode}_frac"] = float(
                    np.mean([value == mode for value in aggregation_modes])
                )
        if valid_counts:
            log_dict["rollout/group_delta/valid_card_count_mean"] = float(np.mean(valid_counts))
            log_dict["rollout/group_delta/available_delta_count_mean"] = float(np.mean(available_counts))
            log_dict["rollout/group_delta/positive_delta_count_mean"] = float(np.mean(positive_counts))
            if optimizer_card_counts:
                log_dict["rollout/group_delta/generator_optimizer_card_count_mean"] = float(
                    np.mean(optimizer_card_counts)
                )
            if all_nonpositive_flags:
                log_dict["rollout/group_delta/generator_all_nonpositive_frac"] = float(
                    np.mean(all_nonpositive_flags)
                )
            log_dict["rollout/group_delta/scored_environment_token_count_sum"] = float(sum(mask_leaks))
            log_dict["rollout/group_delta/loss_mask_mismatch_count_sum"] = float(sum(mask_mismatches))
            log_dict["rollout/group_delta/scored_token_count_sum"] = float(sum(scored_tokens))
            log_dict["rollout/group_delta/model_token_count_sum"] = float(sum(model_tokens))
            log_dict["rollout/group_delta/environment_token_count_sum"] = float(sum(environment_tokens))

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

    step = compute_rollout_step(args, rollout_id)
    log_dict["rollout/step"] = step

    logger.info(f"rollout {rollout_id}: {log_dict}")
    logging_utils.log(args, log_dict, step_key="rollout/step")
    return True
