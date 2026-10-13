"""Static WebShop cosine matrices plus a dynamic successful-Buffer index."""

from __future__ import annotations

from pathlib import Path
from threading import RLock
from typing import Iterable

import numpy as np


def _normalize_category(category: str) -> str:
    return str(category).strip().lower() or "unknown"


class WebShopSimilarityIndex:
    """Maintain O(1) nearest-success lookup over precomputed similarities.

    Pairwise cosine values never change. Buffer membership does, so additions
    update one matrix column against every source query in that category. A
    removal only recomputes sources whose cached winner was removed. Rollout
    lookup itself is a dictionary access and performs no embedding or scan.
    """

    @staticmethod
    def resolve_cache_path(cache_path: str | Path) -> Path:
        path = Path(cache_path).expanduser()
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[2] / path
        return path.resolve()

    def __init__(self, cache_path: str | Path) -> None:
        self.cache_path = self.resolve_cache_path(cache_path)
        if not self.cache_path.is_file():
            raise FileNotFoundError(
                f"WebShop similarity cache not found: {self.cache_path}. "
                "Run scripts/webshop/build_similarity_cache.py first."
            )

        self._goal_idxs: dict[str, np.ndarray] = {}
        self._similarities: dict[str, np.ndarray] = {}
        self._goal_to_row: dict[str, dict[int, int]] = {}
        with np.load(self.cache_path, allow_pickle=False) as cache:
            schema_version = int(cache["schema_version"][0])
            if schema_version != 1:
                raise ValueError(
                    f"unsupported WebShop similarity cache schema: {schema_version}"
                )
            category_names = [
                str(value) for value in cache["category_names"].tolist()
            ]
            for category_index, category_value in enumerate(category_names):
                category = _normalize_category(category_value)
                goal_idxs = cache[f"category_{category_index}_goal_idxs"].astype(
                    np.int64,
                    copy=True,
                )
                similarities = cache[
                    f"category_{category_index}_similarities"
                ].astype(np.float32, copy=True)
                expected_shape = (len(goal_idxs), len(goal_idxs))
                if similarities.shape != expected_shape:
                    raise ValueError(
                        f"invalid similarity block for {category}: "
                        f"{similarities.shape} != {expected_shape}"
                    )
                self._goal_idxs[category] = goal_idxs
                self._similarities[category] = similarities
                self._goal_to_row[category] = {
                    int(goal_idx): row for row, goal_idx in enumerate(goal_idxs)
                }

        self._eligible: dict[str, set[int]] = {
            category: set() for category in self._goal_idxs
        }
        self._best: dict[str, dict[int, tuple[int, float]]] = {
            category: {} for category in self._goal_idxs
        }
        self._lock = RLock()

    def categories(self) -> tuple[str, ...]:
        return tuple(sorted(self._goal_idxs))

    def category_size(self, category: str) -> int:
        return len(self._goal_idxs.get(_normalize_category(category), ()))

    def clear_eligible(self) -> None:
        with self._lock:
            for values in self._eligible.values():
                values.clear()
            for values in self._best.values():
                values.clear()

    def replace_eligible(self, records: Iterable[tuple[str, int]]) -> None:
        self.clear_eligible()
        for category, goal_idx in records:
            self.set_eligible(category=category, goal_idx=goal_idx, eligible=True)

    def _location(self, category: str, goal_idx: int) -> tuple[str, int] | None:
        normalized_category = _normalize_category(category)
        row = self._goal_to_row.get(normalized_category, {}).get(int(goal_idx))
        if row is None:
            return None
        return normalized_category, row

    def set_eligible(self, *, category: str, goal_idx: int, eligible: bool) -> None:
        """Update dynamic success membership; cost is paid only on Buffer writes."""

        location = self._location(category, int(goal_idx))
        if location is None:
            raise KeyError(
                f"goal_idx={goal_idx} category={category!r} is absent from "
                f"similarity cache {self.cache_path}"
            )
        normalized_category, target_row = location
        target_goal_idx = int(goal_idx)
        with self._lock:
            eligible_set = self._eligible[normalized_category]
            already_eligible = target_goal_idx in eligible_set
            if bool(eligible) == already_eligible:
                return
            if eligible:
                eligible_set.add(target_goal_idx)
                scores = self._similarities[normalized_category][:, target_row]
                for source_row, source_goal_value in enumerate(
                    self._goal_idxs[normalized_category]
                ):
                    source_goal_idx = int(source_goal_value)
                    if source_goal_idx == target_goal_idx:
                        continue
                    score = float(scores[source_row])
                    current = self._best[normalized_category].get(source_goal_idx)
                    if current is None or score > current[1] or (
                        score == current[1] and target_goal_idx < current[0]
                    ):
                        self._best[normalized_category][source_goal_idx] = (
                            target_goal_idx,
                            score,
                        )
                return

            eligible_set.remove(target_goal_idx)
            affected_sources = [
                source_goal_idx
                for source_goal_idx, best in self._best[normalized_category].items()
                if best[0] == target_goal_idx
            ]
            for source_goal_idx in affected_sources:
                self._recompute_source(normalized_category, source_goal_idx)

    def _recompute_source(self, category: str, source_goal_idx: int) -> None:
        eligible_goal_idxs = sorted(
            goal_idx
            for goal_idx in self._eligible[category]
            if goal_idx != source_goal_idx
        )
        if not eligible_goal_idxs:
            self._best[category].pop(source_goal_idx, None)
            return
        source_row = self._goal_to_row[category][source_goal_idx]
        target_rows = np.asarray(
            [self._goal_to_row[category][goal_idx] for goal_idx in eligible_goal_idxs],
            dtype=np.int64,
        )
        scores = self._similarities[category][source_row, target_rows]
        best_position = int(np.argmax(scores))
        self._best[category][source_goal_idx] = (
            eligible_goal_idxs[best_position],
            float(scores[best_position]),
        )

    def lookup(self, *, category: str, source_goal_idx: int) -> dict | None:
        """Return the cached nearest eligible neighbor in strict O(1) time."""

        normalized_category = _normalize_category(category)
        source_goal_idx = int(source_goal_idx)
        with self._lock:
            if self._location(normalized_category, source_goal_idx) is None:
                return None
            best = self._best.get(normalized_category, {}).get(source_goal_idx)
            if best is None:
                return None
            candidate_count = len(self._eligible[normalized_category]) - int(
                source_goal_idx in self._eligible[normalized_category]
            )
            return {
                "goal_idx": int(best[0]),
                "similarity": float(best[1]),
                "candidate_count": max(0, int(candidate_count)),
                "category": normalized_category,
            }

    def stats(self) -> dict:
        with self._lock:
            return {
                "cache_path": str(self.cache_path),
                "query_count": sum(len(values) for values in self._goal_idxs.values()),
                "eligible_count": sum(len(values) for values in self._eligible.values()),
                "cached_lookup_count": sum(len(values) for values in self._best.values()),
                "per_category": {
                    category: {
                        "query_count": len(self._goal_idxs[category]),
                        "eligible_count": len(self._eligible[category]),
                        "cached_lookup_count": len(self._best[category]),
                    }
                    for category in sorted(self._goal_idxs)
                },
            }
