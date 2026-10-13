"""Structured trajectory records for the Delta-RL WebShop pipeline.

The source scout is generated before the skill-conditioned executor siblings.  It
is kept as member zero of that future same-query executor GRPO group, but it must
not be optimized on its own before the rest of the group exists.
"""

from __future__ import annotations

import math
from typing import Any

from slime.utils.types import Sample


def build_turn_record(
    *,
    turn_index: int,
    observation: str,
    admissible_actions: list[str],
    response_text: str,
    action: str,
    model_action_valid: bool,
    response_token_start: int,
    response_token_end: int,
) -> dict[str, Any]:
    """Build a JSON-serializable record for one model response turn."""

    if response_token_start < 0 or response_token_end < response_token_start:
        raise ValueError(
            "invalid response token span: "
            f"start={response_token_start}, end={response_token_end}"
        )

    return {
        "turn_index": int(turn_index),
        "observation": str(observation),
        "admissible_actions": [str(item) for item in admissible_actions],
        "response_text": str(response_text),
        "action": str(action),
        "model_action_valid": bool(model_action_valid),
        "response_token_start": int(response_token_start),
        "response_token_end": int(response_token_end),
        "response_token_count": int(response_token_end - response_token_start),
        # Feature 2 fills these from SGLang's per-token top-k logprobs.  Keep
        # unavailable distinct from zero entropy: zero is a valid value for a
        # fully certain model response, while None means no usable statistics.
        "turn_entropy": None,
        "entropy_available": False,
        "entropy_topk": None,
        "entropy_token_count": 0,
        "topk_probability_mass_mean": None,
        "entropy_error": None,
        # Filled after the environment step.  None distinguishes an unavailable
        # result (generation truncation or request failure) from a false result.
        "environment_action_valid": None,
        "done_after_turn": None,
        "won_after_turn": None,
        "task_score_after_turn": None,
    }


def attach_turn_outcome(
    turn_record: dict[str, Any],
    *,
    environment_action_valid: bool,
    done: bool,
    won: bool,
    task_score: float,
) -> None:
    """Attach the environment result observed after a response turn."""

    turn_record["environment_action_valid"] = bool(environment_action_valid)
    turn_record["done_after_turn"] = bool(done)
    turn_record["won_after_turn"] = bool(won)
    turn_record["task_score_after_turn"] = float(task_score)


def _top_logprob_row_values(row: Any) -> list[float]:
    """Extract logprob values from the SGLang top-logprob row formats.

    SGLang releases have exposed each token row as tuples such as
    ``(logprob, token_id, token_text)`` or as dictionaries.  Accept both while
    rejecting non-finite values instead of silently converting malformed data
    into a low-entropy turn.
    """

    if isinstance(row, dict):
        for key in ("logprob", "log_prob"):
            if key in row:
                try:
                    value = float(row[key])
                except (TypeError, ValueError):
                    return []
                return [value] if math.isfinite(value) else []

        values: list[float] = []
        for item in row.values():
            if isinstance(item, (int, float)):
                value = float(item)
                if math.isfinite(value):
                    values.append(value)
            else:
                values.extend(_top_logprob_row_values(item))
        return values

    if not isinstance(row, (list, tuple)):
        return []

    # One tuple/list entry: (logprob, token_id[, token_text]).
    if row and isinstance(row[0], (int, float)):
        value = float(row[0])
        return [value] if math.isfinite(value) else []

    values = []
    for item in row:
        if isinstance(item, (int, float)):
            value = float(item)
            if math.isfinite(value):
                values.append(value)
        else:
            values.extend(_top_logprob_row_values(item))
    return values


def compute_turn_entropy(
    raw_top_logprobs: Any,
    *,
    expected_token_count: int,
    vocab_size: int,
    topk: int,
) -> dict[str, Any]:
    """Compute ARPO-style normalized top-k entropy for one response turn.

    For token ``t`` this computes the partial-vocabulary approximation
    ``-sum_j exp(logp_tj) * logp_tj / log(vocab_size)`` over the returned top-k
    candidates.  The turn entropy is the mean over every generated response
    token (ARPO monitors only an initial prefix).
    """

    result: dict[str, Any] = {
        "turn_entropy": None,
        "entropy_available": False,
        "entropy_topk": int(topk),
        "entropy_token_count": 0,
        "topk_probability_mass_mean": None,
        "entropy_error": None,
    }
    if expected_token_count <= 0:
        result["entropy_error"] = "empty_response"
        return result
    if vocab_size <= 1:
        result["entropy_error"] = f"invalid_vocab_size:{vocab_size}"
        return result
    if not isinstance(raw_top_logprobs, (list, tuple)):
        result["entropy_error"] = "missing_top_logprobs"
        return result
    if len(raw_top_logprobs) != expected_token_count:
        result["entropy_error"] = (
            "top_logprob_token_count_mismatch:"
            f"{len(raw_top_logprobs)}!={expected_token_count}"
        )
        return result

    normalization = math.log(vocab_size)
    token_entropies: list[float] = []
    probability_masses: list[float] = []
    for row in raw_top_logprobs:
        logprobs = _top_logprob_row_values(row)[:topk]
        if not logprobs:
            result["entropy_error"] = "empty_top_logprob_row"
            return result

        # Log probabilities should be <= 0.  Clamp tiny positive numerical
        # noise so a malformed score cannot create probability mass above one.
        stable_logprobs = [min(logprob, 0.0) for logprob in logprobs]
        probabilities = [math.exp(logprob) for logprob in stable_logprobs]
        token_entropies.append(
            -sum(
                probability * logprob
                for probability, logprob in zip(probabilities, stable_logprobs)
            )
            / normalization
        )
        probability_masses.append(min(1.0, sum(probabilities)))

    result.update(
        {
            "turn_entropy": float(sum(token_entropies) / len(token_entropies)),
            "entropy_available": True,
            "entropy_token_count": len(token_entropies),
            "topk_probability_mass_mean": float(
                sum(probability_masses) / len(probability_masses)
            ),
        }
    )
    return result


def attach_turn_entropy(
    turn_record: dict[str, Any],
    raw_top_logprobs: Any,
    *,
    vocab_size: int,
    topk: int,
) -> None:
    """Attach top-k response entropy statistics to one turn record."""

    turn_record.update(
        compute_turn_entropy(
            raw_top_logprobs,
            expected_token_count=int(turn_record["response_token_count"]),
            vocab_size=vocab_size,
            topk=topk,
        )
    )


def summarize_trajectory_entropy(
    turn_records: list[dict[str, Any]],
    *,
    topk: int,
) -> dict[str, Any]:
    """Return token-weighted trajectory entropy and a full turn ranking."""

    available = [record for record in turn_records if record.get("entropy_available")]
    ranked = sorted(
        available,
        key=lambda record: (-float(record["turn_entropy"]), int(record["turn_index"])),
    )
    token_count = sum(int(record["entropy_token_count"]) for record in available)
    trajectory_entropy = None
    if token_count:
        trajectory_entropy = sum(
            float(record["turn_entropy"]) * int(record["entropy_token_count"])
            for record in available
        ) / token_count

    return {
        "trajectory_entropy": (
            float(trajectory_entropy) if trajectory_entropy is not None else None
        ),
        "entropy_ranked_turn_indices": [int(record["turn_index"]) for record in ranked],
        "entropy_turn_count": len(available),
        "entropy_token_count": token_count,
        "entropy_topk": int(topk),
        "entropy_method": "arpo_topk_normalized",
        "entropy_token_aggregation": "mean_all_response_tokens",
    }


def select_high_entropy_turns(
    trajectory_metadata: dict[str, Any],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """Select the highest-entropy response turns for generator conditioning.

    The returned order is entropy rank (highest first), not episode chronology.
    Missing entropy is treated as unavailable rather than as zero uncertainty.
    """

    if limit <= 0:
        raise ValueError(f"high-entropy turn limit must be positive, got {limit}")

    raw_turns = trajectory_metadata.get("turns")
    if not isinstance(raw_turns, list):
        raise ValueError("source trajectory metadata has no turn records")

    available: list[dict[str, Any]] = []
    for record in raw_turns:
        if not isinstance(record, dict) or not record.get("entropy_available"):
            continue
        entropy = record.get("turn_entropy")
        if not isinstance(entropy, (int, float)) or not math.isfinite(float(entropy)):
            continue
        available.append(record)

    if not available:
        raise ValueError("source trajectory has no response turn with usable entropy")

    ranked = sorted(
        available,
        key=lambda record: (-float(record["turn_entropy"]), int(record["turn_index"])),
    )
    return [dict(record) for record in ranked[:limit]]


def mark_source_scout(sample: Sample) -> None:
    """Mark the no-skill source trajectory as executor-group member zero.

    The sample is intentionally marked pending until skill-conditioned siblings
    are generated.  A later executor-group assembly step will clear this flag and
    return the complete group to the optimizer.
    """

    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    sample.metadata = metadata

    label = sample.label if isinstance(sample.label, dict) else {}
    goal_idx = metadata.get("goal_idx", label.get("goal_idx"))
    category = str(metadata.get("category") or label.get("category") or "unknown")
    group_index = sample.group_index
    executor_group_id = f"webshop:{group_index}" if group_index is not None else f"webshop:goal:{goal_idx}"
    outcome_reward = float(sample.reward) if isinstance(sample.reward, (int, float)) else 0.0

    metadata.update(
        {
            "trajectory_role": "source_scout",
            "executor_variant": "no_skill",
            "executor_group_id": executor_group_id,
            "executor_group_position": 0,
            "executor_group_pending": True,
            "outcome_reward": outcome_reward,
            "category": category,
        }
    )

    train_metadata = dict(sample.train_metadata or {})
    train_metadata.update(
        {
            "training_stage": "source_scout",
            "sample_role": "executor",
            "executor_variant": "no_skill",
            "executor_group_id": executor_group_id,
            "executor_group_position": 0,
            "executor_group_pending": True,
        }
    )
    sample.train_metadata = train_metadata


def finalize_source_scout(
    sample: Sample,
    *,
    executor_skill_mode: str,
    executor_group_size: int,
) -> None:
    """Make the already-generated scout a trainable member of its executor group."""

    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    sample.metadata = metadata
    metadata.update(
        {
            "executor_group_pending": False,
            "executor_group_size": int(executor_group_size),
            "executor_skill_mode": str(executor_skill_mode),
        }
    )
    train_metadata = dict(sample.train_metadata or {})
    train_metadata.update(
        {
            "training_stage": "executor",
            "sample_role": "executor",
            "executor_group_pending": False,
            "executor_group_size": int(executor_group_size),
            "executor_skill_mode": str(executor_skill_mode),
        }
    )
    sample.train_metadata = train_metadata


def mark_executor_group_member(
    sample: Sample,
    *,
    source_scout: Sample,
    position: int,
    skill: str | None,
    executor_skill_mode: str,
    executor_group_size: int,
) -> None:
    """Attach stable GRPO-group identity to a post-scout executor trajectory."""

    if position <= 0:
        raise ValueError(f"post-scout executor position must be positive, got {position}")
    source_metadata = (
        source_scout.metadata if isinstance(source_scout.metadata, dict) else {}
    )
    executor_group_id = str(source_metadata.get("executor_group_id") or "")
    normalized_skill = str(skill or "").strip()
    executor_variant = "with_skill" if normalized_skill else "no_skill"

    sample.group_index = source_scout.group_index
    sample.index = int(position)
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    sample.metadata = metadata
    metadata.update(
        {
            "trajectory_role": f"{executor_variant}_executor",
            "executor_variant": executor_variant,
            "executor_group_id": executor_group_id,
            "executor_group_position": int(position),
            "executor_group_pending": False,
            "executor_group_size": int(executor_group_size),
            "executor_skill_mode": str(executor_skill_mode),
            "skill": normalized_skill,
            "outcome_reward": (
                float(sample.reward)
                if isinstance(sample.reward, (int, float))
                else float(bool(metadata.get("won", False)))
            ),
        }
    )
    train_metadata = dict(sample.train_metadata or {})
    train_metadata.update(
        {
            "training_stage": "executor",
            "sample_role": "executor",
            "executor_variant": executor_variant,
            "executor_group_id": executor_group_id,
            "executor_group_position": int(position),
            "executor_group_pending": False,
            "executor_group_size": int(executor_group_size),
            "executor_skill_mode": str(executor_skill_mode),
        }
    )
    sample.train_metadata = train_metadata
