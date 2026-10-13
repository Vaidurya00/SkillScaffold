from __future__ import annotations

import ast
import math
from pathlib import Path
import random
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load_functions(relative: str, names: set[str], namespace: dict) -> dict:
    source = (ROOT / relative).read_text(encoding="utf-8")
    tree = ast.parse(source)
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in names
    ]
    missing = names - {node.name for node in selected}
    if missing:
        raise AssertionError(f"missing functions in {relative}: {sorted(missing)}")
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias("annotations")], level=0), *selected],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), relative, "exec"), namespace)
    return namespace


def metadata(sample):
    if sample.metadata is None:
        sample.metadata = {}
    return sample.metadata


class FakeNumpy:
    @staticmethod
    def mean(values):
        return sum(values) / len(values)


class AlgorithmContracts(unittest.TestCase):
    def test_alfworld_signed_credit_and_positive_gate(self) -> None:
        credit = load_functions(
            "benchmarks/alfworld/src/alfworld/generate/group_delta_credit.py",
            {"aggregate_group_delta"},
            {"math": math, "np": FakeNumpy, "SCOUT_COUNT": 4},
        )
        self.assertEqual(
            credit["aggregate_group_delta"]([1, 3, -2, 2], [True, True, False, False])["diff_reward"],
            1.0,
        )
        self.assertEqual(
            credit["aggregate_group_delta"]([-1, -2, -3, -4], [False] * 4)["diff_reward"],
            2.5,
        )

        route = load_functions(
            "benchmarks/alfworld/src/alfworld/generate/group_delta_rollout.py",
            {"_select_routes"},
            {
                "math": math,
                "random": random,
                "ensure_metadata": metadata,
                "POST_SCOUT_COUNT": 4,
            },
        )
        cards = [
            SimpleNamespace(metadata={"skill": f"skill-{index}", "format_score": 1.0})
            for index in range(4)
        ]
        scores = [
            {"available": True, "diff_reward": value}
            for value in (-1.0, 0.5, 0.5, 0.0)
        ]
        routes, selection = route["_select_routes"](
            "positive_gate_trainext_raw", cards, scores
        )
        self.assertEqual(routes, ["skill-1"] * 4)
        self.assertEqual(selection["selected_skill_candidate_index"], 1)
        for score in scores:
            score["diff_reward"] = 0.0
        self.assertEqual(
            route["_select_routes"]("positive_gate_trainext_raw", cards, scores)[0],
            [None] * 4,
        )

    def test_webshop_first_success_and_stable_positive_argmax(self) -> None:
        functions = load_functions(
            "benchmarks/webshop/src/webshop/generate/rollout.py",
            {"_select_successful_scout", "_select_positive_delta_skill"},
            {"math": math, "ensure_metadata": metadata},
        )
        scouts = [SimpleNamespace(metadata={"won": value}) for value in (False, True, True)]
        self.assertEqual(functions["_select_successful_scout"](scouts), (scouts[1], 1))
        cards = [
            SimpleNamespace(index=index, metadata={"skill": f"skill-{index}", "format_score": 1.0})
            for index in range(4)
        ]
        scores = [
            {"available": True, "diff_reward": value}
            for value in (-0.2, 0.4, 0.4, float("nan"))
        ]
        self.assertEqual(
            functions["_select_positive_delta_skill"](cards, scores),
            ("skill-1", 1, 0.4),
        )

    def test_tau_fixed_slot_identity_prefers_recorded_fields(self) -> None:
        functions = load_functions(
            "benchmarks/tau/delta_card_tau/fixed_slot_rollout.py",
            {"_flatten", "_head", "_domain", "_task_id", "_slot"},
            {},
        )
        sample = SimpleNamespace(
            index=99,
            metadata={
                "fixed_query_domain": "retail",
                "fixed_query_task_id": "retail_300",
                "fixed_query_slot": 5,
            },
        )
        group = [[sample]]
        self.assertEqual(functions["_domain"](group), "retail")
        self.assertEqual(functions["_task_id"](group), "retail_300")
        self.assertEqual(functions["_slot"](group), 5)


if __name__ == "__main__":
    unittest.main()
