#!/usr/bin/env python3
"""Validate and summarize the default fixed-k tau2-bench evaluation."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


EXPECTED_TASKS = {"airline": 50, "retail": 114, "telecom": 114}


def pass_at_k(successes: int, k: int, trials: int) -> float:
    if trials - successes < k:
        return 1.0
    return 1.0 - math.comb(trials - successes, k) / math.comb(trials, k)


def result_paths(root: Path, prefix: str, domain: str) -> list[Path]:
    if domain == "airline":
        paths = [root / f"{prefix}_airline_base" / "results.json"]
    else:
        paths = sorted(root.glob(f"{prefix}_{domain}_base_shard*/results.json"))
    if not paths or not all(path.is_file() for path in paths):
        raise RuntimeError(f"missing result files for {domain}: {paths}")
    return paths


def load_domain(root: Path, prefix: str, domain: str) -> list[dict]:
    simulations: list[dict] = []
    for path in result_paths(root, prefix, domain):
        payload = json.loads(path.read_text(encoding="utf-8"))
        simulations.extend(payload.get("simulations") or [])
    return simulations


def summarize(simulations: list[dict], trials: int) -> dict:
    by_task: dict[str, list[float]] = defaultdict(list)
    terminations: dict[str, int] = defaultdict(int)
    for simulation in simulations:
        reward = float((simulation.get("reward_info") or {}).get("reward", 0.0))
        by_task[str(simulation["task_id"])].append(reward)
        terminations[str(simulation.get("termination_reason"))] += 1

    wrong_k = {task: len(values) for task, values in by_task.items() if len(values) != trials}
    if wrong_k:
        raise RuntimeError(f"expected exactly {trials} trajectories per task: {wrong_k}")

    successes = [sum(value > 0 for value in values) for values in by_task.values()]
    result = {
        "tasks": len(by_task),
        "trajectories": len(simulations),
        "successes": sum(successes),
        "pass@1": sum(successes) / (trials * len(successes)),
        "terminations": dict(sorted(terminations.items())),
    }
    if trials >= 4:
        result["pass@4"] = sum(
            pass_at_k(count, 4, trials) for count in successes
        ) / len(successes)
    if trials >= 8:
        result["pass@8"] = sum(count > 0 for count in successes) / len(successes)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulation-root", type=Path, required=True)
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-trials", type=int, default=8)
    parser.add_argument(
        "--expected-tasks-per-domain",
        type=int,
        help="Smoke-only override; the default protocol uses fixed domain sizes.",
    )
    args = parser.parse_args()

    result = {"run_prefix": args.run_prefix, "domains": {}}
    for domain, default_tasks in EXPECTED_TASKS.items():
        expected_tasks = args.expected_tasks_per_domain or default_tasks
        expected_denominator = (expected_tasks, expected_tasks * args.num_trials)
        domain_summary = summarize(
            load_domain(args.simulation_root, args.run_prefix, domain),
            args.num_trials,
        )
        denominator = (domain_summary["tasks"], domain_summary["trajectories"])
        if denominator != expected_denominator:
            raise RuntimeError(
                f"unexpected {domain} denominator: {denominator}; "
                f"expected {expected_denominator}"
            )
        result["domains"][domain] = domain_summary

    result["macro_pass@1"] = sum(
        result["domains"][domain]["pass@1"] for domain in EXPECTED_TASKS
    ) / len(EXPECTED_TASKS)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
