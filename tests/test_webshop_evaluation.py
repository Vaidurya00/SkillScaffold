from pathlib import Path
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from test_algorithm_contracts import load_functions, metadata


ROOT = Path(__file__).resolve().parents[1]


class WebShopEvaluation(unittest.IsolatedAsyncioTestCase):
    async def test_only_target_episode_runs_and_scores_its_outcome(self):
        for won in (True, False):
            with self.subTest(won=won):
                sample = SimpleNamespace(label={"goal_idx": 42}, metadata=None)
                result = SimpleNamespace(metadata={"won": won, "num_steps": 5}, reward=None)
                episode = AsyncMock(return_value=result)
                functions = load_functions(
                    "benchmarks/webshop/src/webshop/generate/eval.py",
                    {"run_eval"},
                    {"run_episode": episode, "ensure_metadata": metadata,
                     "maybe_print_random_sample": lambda *a, **kw: None},
                )
                args, params = SimpleNamespace(), {"temperature": 0.7}
                actual = await functions["run_eval"](args, sample, params)
                episode.assert_awaited_once_with(
                    args, sample=sample, sampling_params=params, skill=None, evaluation=True
                )
                self.assertIs(actual, result)
                self.assertEqual(actual.reward, 10.0 if won else 0.0)
                self.assertEqual(actual.metadata["won_no_skill"], float(won))
                self.assertEqual(actual.metadata["num_steps_no_skill"], 5)
                self.assertFalse(any("with_skill" in key for key in actual.metadata))

    async def test_environment_errors_remain_visible(self):
        episode = AsyncMock(side_effect=RuntimeError("environment unavailable"))
        functions = load_functions(
            "benchmarks/webshop/src/webshop/generate/eval.py", {"run_eval"},
            {"run_episode": episode},
        )
        with self.assertRaisesRegex(RuntimeError, "environment unavailable"):
            await functions["run_eval"](SimpleNamespace(), SimpleNamespace(), {})

    def test_metrics_use_only_target_outcomes(self):
        functions = load_functions(
            "benchmarks/webshop/src/webshop/loggers/eval_logger.py",
            {"_get_task_type", "_mean", "_meta_float", "_won", "_log_baseline_eval"},
            {"compute_statistics": lambda xs: {"mean": sum(xs) / len(xs)},
             "dict_add_prefix": lambda values, prefix: {prefix + k: v for k, v in values.items()}},
        )
        samples = [
            SimpleNamespace(label={"task_type": task}, metadata={"won": won, "num_steps": steps})
            for task, won, steps in [("a", True, 2), ("a", False, 4), ("b", True, 6)]
        ]
        metrics = {}
        functions["_log_baseline_eval"](metrics, "eval/webshop_id", samples, {})
        self.assertAlmostEqual(metrics["eval/webshop_id/win_rate_no_skill"], 2 / 3)
        self.assertEqual(metrics["eval/webshop_id/a/win_rate_no_skill"], 0.5)
        self.assertEqual(metrics["eval/webshop_id/b/win_rate_no_skill"], 1.0)
        self.assertEqual(metrics["eval/webshop_id/num_steps/mean"], 4.0)
        self.assertFalse(any("with_skill" in key for key in metrics))

    def test_eval_does_not_build_cache_start_or_wait_for_retrieval(self):
        source = (ROOT / "benchmarks/webshop/scripts/train.sh").read_text()
        cache = source.split('if [[ "${NUM_ROLLOUT}" != "0" && ! -f "${CACHE_FILE}" ]]; then', 1)[1].split("\nfi", 1)[0]
        launch = source.split('  if [[ "${NUM_ROLLOUT}" != "0" ]]; then\n', 1)[1].split("\n  fi", 1)[0]
        health = source.split('wait_for_health "${WEBSHOP_ENV_URL}/health"', 1)[1].split("\n\n", 1)[0]
        script = (
            'NUM_ROLLOUT=0; CACHE_FILE=/nonexistent/cache; WEBSHOP_ENV_URL=http://env; WEBSHOP_STARTUP_TIMEOUT=1\n'
            'bash() { echo unexpected-cache; return 1; }; nohup() { echo unexpected-launch; return 1; }; wait_for_health() { echo "$1"; }\n'
            'if [[ "${NUM_ROLLOUT}" != "0" && ! -f "${CACHE_FILE}" ]]; then' + cache + '\nfi\n'
            'if [[ "${NUM_ROLLOUT}" != "0" ]]; then\n' + launch + '\nfi\n'
            'wait_for_health "${WEBSHOP_ENV_URL}/health"' + health
        )
        result = subprocess.run(["bash", "-eu", "-c", script], capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout.strip(), "http://env/health")
