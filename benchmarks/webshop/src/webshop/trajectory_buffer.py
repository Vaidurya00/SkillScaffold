"""Category-partitioned canonical trajectory buffer for Delta-RL WebShop.

Each query owns exactly one canonical trajectory.  A successful no-skill scout
always wins.  When the scout fails, the shortest successful later executor
trajectory wins; if every attempt fails, the shortest failure is retained only
as a non-retrievable placeholder. Later trajectories may be no-skill during
cold start or skill-conditioned after the retrieval gate opens.

The buffer deliberately stores a neutral interaction transcript instead of the
original model prompt.  Later skill/no-skill teacher-forced forwards can thus
rebuild both prompts around the exact same target response without leaking the
skill that originally produced a with-skill trajectory.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import math
from threading import RLock
from typing import Any, Iterable

from slime.utils.types import Sample

from src.webshop.similarity_index import WebShopSimilarityIndex


_BAD_SAMPLE_STATUSES = {
    Sample.Status.TRUNCATED.value,
    Sample.Status.ABORTED.value,
    Sample.Status.FAILED.value,
}


def _metadata(sample: Sample) -> dict[str, Any]:
    return sample.metadata if isinstance(sample.metadata, dict) else {}


def _label(sample: Sample) -> dict[str, Any]:
    return sample.label if isinstance(sample.label, dict) else {}


def _category(sample: Sample) -> str:
    metadata = _metadata(sample)
    label = _label(sample)
    value = (
        metadata.get("category")
        or metadata.get("task_type")
        or label.get("category")
        or label.get("task_type")
        or "unknown"
    )
    return str(value).strip().lower() or "unknown"


def _goal_idx(sample: Sample) -> int | str | None:
    metadata = _metadata(sample)
    label = _label(sample)
    value = metadata.get("goal_idx", label.get("goal_idx"))
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def _query(sample: Sample) -> str:
    metadata = _metadata(sample)
    label = _label(sample)
    return str(
        metadata.get("task_description")
        or label.get("task_description")
        or label.get("question")
        or ""
    ).strip()


def _query_key(sample: Sample) -> str:
    goal_idx = _goal_idx(sample)
    if goal_idx is not None:
        return f"goal:{goal_idx}"
    query = _query(sample)
    if not query:
        raise ValueError("trajectory buffer record requires goal_idx or task description")
    digest = hashlib.sha256(query.encode("utf-8")).hexdigest()[:20]
    return f"query:{digest}"


def _outcome_reward(sample: Sample) -> float:
    metadata = _metadata(sample)
    value = metadata.get("outcome_reward")
    if value is None:
        value = metadata.get("won")
    if value is None and isinstance(sample.reward, (int, float)):
        value = sample.reward
    try:
        reward = float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return reward if math.isfinite(reward) else 0.0


def _is_success(sample: Sample) -> bool:
    return _outcome_reward(sample) >= 0.9


def _status_value(sample: Sample) -> str:
    status = sample.status
    return str(status.value if isinstance(status, Sample.Status) else status)


def trajectory_turn_count(sample: Sample) -> int:
    """Return the comparable environment-turn count for a trajectory."""

    metadata = _metadata(sample)
    turns = metadata.get("turns")
    if isinstance(turns, list):
        return len(turns)
    try:
        return max(0, int(metadata.get("num_steps", 0) or 0))
    except (TypeError, ValueError):
        return 0


def _executor_group_position(sample: Sample, fallback: int) -> int:
    metadata = _metadata(sample)
    value = metadata.get("executor_group_position")
    if value is None:
        value = sample.index
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(fallback)


def _selection_key(sample: Sample, fallback: int) -> tuple[int, int, int]:
    """Prefer fewer turns, then earlier group position, then input order."""

    return (
        trajectory_turn_count(sample),
        _executor_group_position(sample, fallback),
        int(fallback),
    )


def _validate_same_query(source_scout: Sample, candidates: Iterable[Sample]) -> None:
    source_key = (_category(source_scout), _query_key(source_scout))
    for candidate in candidates:
        candidate_key = (_category(candidate), _query_key(candidate))
        if candidate_key != source_key:
            raise ValueError(
                "executor-group trajectory does not match source query: "
                f"source={source_key}, candidate={candidate_key}"
            )


def select_canonical_trajectory(
    source_scout: Sample,
    executor_trajectories: Iterable[Sample],
) -> tuple[Sample, str]:
    """Select the one trajectory that represents a query in the buffer.

    Priority is fixed by the method definition:
      1. successful source scout;
      2. shortest successful later executor trajectory;
      3. shortest failure across the scout and later executor trajectories.
    """

    later_executor = list(executor_trajectories)
    _validate_same_query(source_scout, later_executor)

    if _is_success(source_scout):
        return source_scout, "successful_source_scout"

    successful_executor = [sample for sample in later_executor if _is_success(sample)]
    if successful_executor:
        indexed = list(enumerate(successful_executor, start=1))
        _, selected = min(
            indexed,
            key=lambda item: _selection_key(item[1], item[0]),
        )
        selected_variant = str(
            _metadata(selected).get("executor_variant")
            or ("with_skill" if _metadata(selected).get("skill") else "no_skill")
        )
        reason = (
            "shortest_successful_with_skill"
            if selected_variant == "with_skill"
            else "shortest_successful_no_skill_executor"
        )
        return selected, reason

    failures = [source_scout, *later_executor]
    indexed_failures = list(enumerate(failures))
    _, selected = min(
        indexed_failures,
        key=lambda item: _selection_key(item[1], item[0]),
    )
    return selected, "shortest_failed_placeholder"


def _response_token_ids(sample: Sample) -> list[int]:
    try:
        response_length = max(0, int(sample.response_length or 0))
    except (TypeError, ValueError):
        response_length = 0
    if not response_length or not isinstance(sample.tokens, list):
        return []
    return [int(token_id) for token_id in sample.tokens[-response_length:]]


def build_buffer_record(
    sample: Sample,
    *,
    selection_reason: str,
    selection_final: bool,
    update_sequence: int,
    buffer_insert_step: int = 0,
) -> dict[str, Any]:
    """Copy a Sample into a prompt-neutral, JSON-serializable record."""

    metadata = _metadata(sample)
    train_metadata = sample.train_metadata if isinstance(sample.train_metadata, dict) else {}
    success = _is_success(sample)
    status = _status_value(sample)
    retrieval_eligible = success and status not in _BAD_SAMPLE_STATUSES
    turns = metadata.get("turns") if isinstance(metadata.get("turns"), list) else []
    executor_variant = str(
        metadata.get("executor_variant")
        or train_metadata.get("executor_variant")
        or ("with_skill" if metadata.get("skill") else "no_skill")
    )
    trajectory_role = str(
        metadata.get("trajectory_role")
        or ("source_scout" if executor_variant == "no_skill" else "with_skill_executor")
    )

    provenance = {
        "executor_variant": executor_variant,
        "trajectory_role": trajectory_role,
        "executor_group_id": str(metadata.get("executor_group_id") or ""),
        "executor_group_position": _executor_group_position(sample, 0),
        "candidate_skill_index": metadata.get("candidate_skill_index"),
        "conditioning_skill": str(metadata.get("skill") or ""),
    }
    return {
        "schema_version": "canonical_webshop_trajectory_v2",
        "category": _category(sample),
        "query_key": _query_key(sample),
        "goal_idx": _goal_idx(sample),
        "query": _query(sample),
        "outcome_reward": _outcome_reward(sample),
        "won": success,
        "retrieval_eligible": bool(retrieval_eligible),
        "selection_reason": str(selection_reason),
        "selection_final": bool(selection_final),
        "update_sequence": int(update_sequence),
        "buffer_insert_step": int(buffer_insert_step),
        "sample_status": status,
        "turn_count": trajectory_turn_count(sample),
        "response": str(sample.response or ""),
        "response_token_ids": _response_token_ids(sample),
        "turns": deepcopy(turns),
        "trajectory_entropy": metadata.get("trajectory_entropy"),
        "entropy_ranked_turn_indices": deepcopy(
            metadata.get("entropy_ranked_turn_indices", [])
        ),
        "provenance": provenance,
    }


class WebShopTrajectoryBuffer:
    """Process-local, concurrency-safe canonical trajectory buffer."""

    def __init__(self) -> None:
        self._records: dict[str, dict[str, dict[str, Any]]] = {}
        self._pending_records: dict[str, dict[str, dict[str, Any]]] = {}
        self._update_sequence = 0
        self._current_step = 0
        self._ttl_steps = 25
        self._expired_count = 0
        self._expired_retrieval_eligible_count = 0
        self._expired_per_category: dict[str, int] = {}
        self._similarity_index: WebShopSimilarityIndex | None = None
        self._lock = RLock()

    def configure_similarity_index(
        self,
        cache_path: str,
        *,
        current_step: int | None = None,
        ttl_steps: int | None = None,
    ) -> dict[str, Any]:
        """Configure static cosine blocks and the rollout-step TTL clock."""

        with self._lock:
            if ttl_steps is not None:
                normalized_ttl = int(ttl_steps)
                if normalized_ttl < 0:
                    raise ValueError(
                        f"trajectory Buffer ttl_steps must be nonnegative, got {ttl_steps}"
                    )
                self._ttl_steps = normalized_ttl
            if current_step is not None:
                self._advance_to_step_locked(int(current_step))
            requested_path = WebShopSimilarityIndex.resolve_cache_path(cache_path)
            if (
                self._similarity_index is not None
                and self._similarity_index.cache_path == requested_path
            ):
                return {**self._similarity_index.stats(), **self._ttl_state_locked()}
            requested = WebShopSimilarityIndex(requested_path)
            eligible_records = [
                (str(record["category"]), int(record["goal_idx"]))
                for category_records in self._records.values()
                for record in category_records.values()
                if record.get("retrieval_eligible")
                and record.get("goal_idx") is not None
            ]
            requested.replace_eligible(eligible_records)
            self._similarity_index = requested
            return {**requested.stats(), **self._ttl_state_locked()}

    def _ttl_state_locked(self) -> dict[str, int]:
        return {
            "current_step": int(self._current_step),
            "ttl_steps": int(self._ttl_steps),
            "expired_count": int(self._expired_count),
            "expired_retrieval_eligible_count": int(
                self._expired_retrieval_eligible_count
            ),
        }

    def _record_age_locked(self, record: dict[str, Any]) -> int:
        inserted_step = int(record.get("buffer_insert_step", 0) or 0)
        return max(0, int(self._current_step) - inserted_step)

    def _record_is_expired_locked(self, record: dict[str, Any]) -> bool:
        return self._record_age_locked(record) > self._ttl_steps

    def _count_expired_locked(self, category: str, record: dict[str, Any]) -> None:
        self._expired_count += 1
        self._expired_per_category[category] = (
            self._expired_per_category.get(category, 0) + 1
        )
        if record.get("retrieval_eligible"):
            self._expired_retrieval_eligible_count += 1

    def _purge_expired_locked(self) -> int:
        expired_now = 0
        for category in list(self._records):
            category_records = self._records[category]
            for query_key, record in list(category_records.items()):
                if not self._record_is_expired_locked(record):
                    continue
                if (
                    record.get("retrieval_eligible")
                    and self._similarity_index is not None
                    and record.get("goal_idx") is not None
                ):
                    self._similarity_index.set_eligible(
                        category=category,
                        goal_idx=int(record["goal_idx"]),
                        eligible=False,
                    )
                self._count_expired_locked(category, record)
                del category_records[query_key]
                expired_now += 1
            if not category_records:
                del self._records[category]

        for category in list(self._pending_records):
            category_records = self._pending_records[category]
            for query_key, record in list(category_records.items()):
                if not self._record_is_expired_locked(record):
                    continue
                self._count_expired_locked(category, record)
                del category_records[query_key]
                expired_now += 1
            if not category_records:
                del self._pending_records[category]
        return expired_now

    def _advance_to_step_locked(self, current_step: int) -> int:
        if current_step < 0:
            raise ValueError(
                f"trajectory Buffer current_step must be nonnegative, got {current_step}"
            )
        if current_step > self._current_step:
            self._current_step = int(current_step)
        return self._purge_expired_locked()

    def advance_to_step(self, current_step: int) -> int:
        """Advance the monotonic rollout-step clock and delete age > TTL records."""

        with self._lock:
            return self._advance_to_step_locked(int(current_step))

    def clear(self) -> None:
        with self._lock:
            self._records.clear()
            self._pending_records.clear()
            self._update_sequence = 0
            self._current_step = 0
            self._expired_count = 0
            self._expired_retrieval_eligible_count = 0
            self._expired_per_category.clear()
            if self._similarity_index is not None:
                self._similarity_index.clear_eligible()

    def _store(
        self,
        sample: Sample,
        *,
        selection_reason: str,
        selection_final: bool,
    ) -> dict[str, Any]:
        with self._lock:
            self._purge_expired_locked()
            self._update_sequence += 1
            record = build_buffer_record(
                sample,
                selection_reason=selection_reason,
                selection_final=selection_final,
                update_sequence=self._update_sequence,
                buffer_insert_step=self._current_step,
            )
            category_records = self._records.setdefault(record["category"], {})
            previous = category_records.get(record["query_key"])
            category_records[record["query_key"]] = record
            if selection_final:
                pending_category = self._pending_records.get(record["category"])
                if pending_category is not None:
                    pending_category.pop(record["query_key"], None)
                    if not pending_category:
                        self._pending_records.pop(record["category"], None)
            if self._similarity_index is not None and record.get("goal_idx") is not None:
                previous_eligible = bool(
                    previous is not None and previous.get("retrieval_eligible")
                )
                current_eligible = bool(record.get("retrieval_eligible"))
                if previous_eligible != current_eligible:
                    self._similarity_index.set_eligible(
                        category=record["category"],
                        goal_idx=int(record["goal_idx"]),
                        eligible=current_eligible,
                    )
            return deepcopy(record)

    def stage_source_scout(self, source_scout: Sample) -> dict[str, Any]:
        """Insert the scout immediately.

        A successful scout is already final because it has absolute priority.
        A failed scout is kept in a separate pending map until the same-query
        executor group is available, so it cannot temporarily evict an older
        successful canonical trajectory.
        """

        success = _is_success(source_scout)
        if success:
            return self._store(
                source_scout,
                selection_reason="successful_source_scout",
                selection_final=True,
            )

        with self._lock:
            self._purge_expired_locked()
            self._update_sequence += 1
            record = build_buffer_record(
                source_scout,
                selection_reason="source_failure_pending_executor_group",
                selection_final=False,
                update_sequence=self._update_sequence,
                buffer_insert_step=self._current_step,
            )
            self._pending_records.setdefault(record["category"], {})[
                record["query_key"]
            ] = record
            return deepcopy(record)

    def finalize_executor_group(
        self,
        source_scout: Sample,
        executor_trajectories: Iterable[Sample],
    ) -> dict[str, Any]:
        selected, reason = select_canonical_trajectory(
            source_scout,
            executor_trajectories,
        )
        return self._store(
            selected,
            selection_reason=reason,
            selection_final=True,
        )

    def discard_pending_source_scout(self, source_scout: Sample) -> None:
        """Remove a failed staged scout when its executor group cannot be completed."""

        category = _category(source_scout)
        query_key = _query_key(source_scout)
        with self._lock:
            pending_category = self._pending_records.get(category)
            if pending_category is not None:
                pending_category.pop(query_key, None)
                if not pending_category:
                    self._pending_records.pop(category, None)

    def get(
        self,
        *,
        category: str,
        goal_idx: int | str | None = None,
        query: str = "",
    ) -> dict[str, Any] | None:
        normalized_category = str(category).strip().lower() or "unknown"
        if goal_idx is not None:
            try:
                normalized_goal_idx: int | str = int(goal_idx)
            except (TypeError, ValueError):
                normalized_goal_idx = str(goal_idx)
            key = f"goal:{normalized_goal_idx}"
        else:
            normalized_query = str(query).strip()
            if not normalized_query:
                raise ValueError("get requires goal_idx or query")
            digest = hashlib.sha256(normalized_query.encode("utf-8")).hexdigest()[:20]
            key = f"query:{digest}"
        with self._lock:
            self._purge_expired_locked()
            record = self._records.get(normalized_category, {}).get(key)
            return deepcopy(record) if record is not None else None

    def retrieval_candidates(
        self,
        *,
        category: str,
        exclude_goal_idx: int | str | None = None,
    ) -> list[dict[str, Any]]:
        """Return only successful canonical records from one category."""

        normalized_category = str(category).strip().lower() or "unknown"
        with self._lock:
            self._purge_expired_locked()
            records = self._records.get(normalized_category, {}).values()
            candidates = [
                deepcopy(record)
                for record in records
                if record.get("retrieval_eligible")
                and (
                    exclude_goal_idx is None
                    or str(record.get("goal_idx")) != str(exclude_goal_idx)
                )
            ]
        return sorted(candidates, key=lambda record: int(record["update_sequence"]))

    def nearest_successful_trajectory(
        self,
        *,
        category: str,
        source_goal_idx: int | str | None,
    ) -> tuple[dict[str, Any] | None, str]:
        """O(1) lookup of the nearest currently successful same-category record."""

        if source_goal_idx is None:
            return None, "missing_source_goal_idx"
        try:
            normalized_goal_idx = int(source_goal_idx)
        except (TypeError, ValueError):
            return None, "invalid_source_goal_idx"

        normalized_category = str(category).strip().lower() or "unknown"
        with self._lock:
            self._purge_expired_locked()
            if self._similarity_index is None:
                return None, "similarity_index_not_configured"
            nearest = self._similarity_index.lookup(
                category=normalized_category,
                source_goal_idx=normalized_goal_idx,
            )
            if nearest is None:
                return None, "no_successful_buffer_candidate"
            record = self._records.get(normalized_category, {}).get(
                f"goal:{nearest['goal_idx']}"
            )
            if record is None or not record.get("retrieval_eligible"):
                raise RuntimeError(
                    "similarity index and trajectory Buffer membership diverged for "
                    f"category={normalized_category}, goal_idx={nearest['goal_idx']}"
                )
            result = deepcopy(record)
            result["retrieval_similarity"] = float(nearest["similarity"])
            result["retrieval_source_candidate_count"] = int(
                nearest["candidate_count"]
            )
            return result, ""

    def snapshot(self) -> dict[str, dict[str, dict[str, Any]]]:
        with self._lock:
            self._purge_expired_locked()
            return deepcopy(self._records)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            self._purge_expired_locked()
            per_category = {}
            categories = sorted(
                set(self._records)
                | set(self._pending_records)
                | set(self._expired_per_category)
            )
            for category in categories:
                records = self._records.get(category, {})
                pending_records = self._pending_records.get(category, {})
                query_keys = set(records) | set(pending_records)
                eligible = sum(
                    bool(record.get("retrieval_eligible"))
                    for record in records.values()
                )
                pending = len(pending_records)
                category_record_ages = [
                    self._record_age_locked(record)
                    for record in [*records.values(), *pending_records.values()]
                ]
                placeholders = sum(
                    not bool(records[key].get("retrieval_eligible"))
                    if key in records
                    else True
                    for key in query_keys
                )
                per_category[category] = {
                    "query_count": len(query_keys),
                    "retrieval_eligible_count": eligible,
                    "placeholder_count": placeholders,
                    "pending_executor_group_count": pending,
                    "expired_count": self._expired_per_category.get(category, 0),
                    "record_age_mean": (
                        sum(category_record_ages) / len(category_record_ages)
                        if category_record_ages
                        else 0.0
                    ),
                    "record_age_max": max(category_record_ages, default=0),
                }
            all_records = [
                record
                for category_records in self._records.values()
                for record in category_records.values()
            ] + [
                record
                for category_records in self._pending_records.values()
                for record in category_records.values()
            ]
            all_record_ages = [
                self._record_age_locked(record) for record in all_records
            ]
            result = {
                "query_count": sum(item["query_count"] for item in per_category.values()),
                "retrieval_eligible_count": sum(
                    item["retrieval_eligible_count"] for item in per_category.values()
                ),
                "placeholder_count": sum(
                    item["placeholder_count"] for item in per_category.values()
                ),
                "pending_executor_group_count": sum(
                    item["pending_executor_group_count"]
                    for item in per_category.values()
                ),
                "update_sequence": self._update_sequence,
                **self._ttl_state_locked(),
                "record_age_mean": (
                    sum(all_record_ages) / len(all_record_ages)
                    if all_record_ages
                    else 0.0
                ),
                "record_age_max": max(all_record_ages, default=0),
                "per_category": per_category,
            }
            if self._similarity_index is not None:
                similarity_stats = self._similarity_index.stats()
                result.update(
                    {
                        "similarity_query_count": similarity_stats["query_count"],
                        "similarity_eligible_count": similarity_stats["eligible_count"],
                        "similarity_cached_lookup_count": similarity_stats[
                            "cached_lookup_count"
                        ],
                    }
                )
            return result


_TRAJECTORY_BUFFER = WebShopTrajectoryBuffer()


def get_trajectory_buffer() -> WebShopTrajectoryBuffer:
    return _TRAJECTORY_BUFFER


def attach_buffer_record_metadata(sample: Sample, record: dict[str, Any]) -> None:
    """Expose the chosen buffer state on rollout metadata for logging/debugging."""

    metadata = _metadata(sample)
    sample.metadata = metadata
    metadata.update(
        {
            "trajectory_buffer_category": record["category"],
            "trajectory_buffer_query_key": record["query_key"],
            "trajectory_buffer_selection_reason": record["selection_reason"],
            "trajectory_buffer_selection_final": record["selection_final"],
            "trajectory_buffer_retrieval_eligible": record["retrieval_eligible"],
            "trajectory_buffer_update_sequence": record["update_sequence"],
            "trajectory_buffer_insert_step": int(
                record.get("buffer_insert_step", 0) or 0
            ),
        }
    )
