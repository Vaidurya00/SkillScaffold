from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


class DefaultConfigurationContracts(unittest.TestCase):
    def assert_contains_all(self, relative: str, fragments: list[str]) -> None:
        content = read(relative)
        for fragment in fragments:
            with self.subTest(file=relative, fragment=fragment):
                self.assertIn(fragment, content)

    def test_alfworld_defaults(self) -> None:
        self.assert_contains_all(
            "benchmarks/alfworld/scripts/train.sh",
            [
                "ALFWORLD_DELTA_CREDIT_SCOPE:-all_model_tokens",
                "ALFWORLD_GROUP_DELTA_VARIANT:-positive_gate_trainext_raw",
                '"${METHOD}" == "group_delta"',
                "NUM_ROLLOUT:-100",
                "GLOBAL_BATCH_SIZE:-128",
                "GENERATOR_ENTROPY_COEF:-",
                "ROLLOUT_MAX_CONTEXT_LEN:-32768",
                "MAX_TOKENS_PER_GPU:-8192",
            ],
        )
        self.assert_contains_all(
            "benchmarks/alfworld/scripts/eval.sh",
            [
                "N_SAMPLES_PER_EVAL_PROMPT:-8",
                "EVAL_TEMPERATURE:-1.0",
                "EVAL_TOP_P:-0.85",
                "ALFWORLD_EVAL_SPLIT:-valid_unseen",
                "expected 134 unique ALFWorld valid_unseen tasks",
            ],
        )

    def test_webshop_defaults(self) -> None:
        self.assert_contains_all(
            "benchmarks/webshop/scripts/train.sh",
            [
                "NUM_ROLLOUT:-200",
                "GLOBAL_BATCH_SIZE:-128",
                "SEED:-1291",
                "ROLLOUT_SEED:-91",
                "USE_KL_LOSS:-0",
                "EPS_CLIP:-0.2",
                "card_general_v1",
            ],
        )
        self.assert_contains_all(
            "benchmarks/webshop/scripts/eval.sh",
            [
                "N_SAMPLES_PER_EVAL_PROMPT:-8",
                "EVAL_TEMPERATURE:-0.7",
                "EVAL_TOP_P:-0.85",
                "src.webshop.generate.eval.run_eval",
            ],
        )
        self.assertIn("four_scout_direct_delta_v3", read("benchmarks/webshop/configs/server.yaml"))

    def test_scienceworld_defaults(self) -> None:
        self.assertTrue((ROOT / "benchmarks/scienceworld/train_entry.py").is_file())
        self.assert_contains_all(
            "benchmarks/scienceworld/scripts/train.sh",
            [
                "NUM_ROLLOUT:-161",
                "GLOBAL_BATCH_SIZE:-128",
                "EXECUTOR_MAX_RESPONSE_LEN",
                "ROLLOUT_MAX_RESPONSE_LEN:-12288",
                "SGLANG_MEM_FRACTION_STATIC:-0.55",
                "e7_prompts_v2.py",
            ],
        )
        self.assert_contains_all(
            "benchmarks/scienceworld/src/scienceworld/generate/group_delta_rollout.py",
            ["SCIENCEWORLD_GENERATOR_PROMPT_FILE", "e7_prompts_v2.py"],
        )
        self.assert_contains_all(
            "benchmarks/scienceworld/scripts/eval.sh",
            ["N_SAMPLES_PER_EVAL_PROMPT:-8", "EVAL_TEMPERATURE:-0.4", "EVAL_TOP_P:-1.0"],
        )
        self.assertIn('--wandb-mode "${WANDB_MODE:-online}"', read("benchmarks/scienceworld/scripts/train.sh"))

    def test_tau_defaults(self) -> None:
        self.assert_contains_all(
            "benchmarks/tau/scripts/train.sh",
            [
                "TAU_E7_FIXED_SLOT_RETRIES:-2",
                "ROLLOUT_BATCH_SIZE:-32",
                "GLOBAL_BATCH_SIZE:-32",
                "NUM_ROLLOUT:-200",
                "USE_KL_LOSS:-0",
                "fixed_slot_rollout.generate_rollout",
                "--generator-entropy-coef 0.002",
            ],
        )
        self.assert_contains_all(
            "benchmarks/tau/scripts/eval.sh",
            [
                "TAU2_EVAL_NUM_TRIALS:-8",
                "TAU2_EVAL_SEED:-300",
                "TAU2_EVAL_PYTHON",
                'export TAU2_DATA_DIR="${TAU2_BENCH_DIR}/data"',
            ],
        )
        self.assertNotIn("--default-chat-template-kwargs", read("benchmarks/tau/scripts/eval.sh"))
        self.assert_contains_all(
            "benchmarks/tau/delta_card_tau/pipeline.py",
            [
                'kwargs["enable_thinking"] = False',
                'kwargs.pop("preserve_thinking", None)',
                "**_generator_chat_template_kwargs(args)",
            ],
        )
        self.assert_contains_all(
            "benchmarks/tau/delta_card_tau/e7_pipeline.py",
            ["**_generator_chat_template_kwargs(args)"],
        )
        self.assert_contains_all(
            ".env.example",
            [
                "HF_DIR=/path/to/tau-domain-sft-checkpoint-490",
                "REF_CKPT_DIR=/path/to/tau-domain-sft-checkpoint-490-distcp",
            ],
        )

    def test_prompt_files_are_nonempty(self) -> None:
        for relative in [
            "benchmarks/alfworld/src/alfworld/group_generator_prompts.py",
            "benchmarks/webshop/src/webshop/prompts.py",
            "benchmarks/scienceworld/src/scienceworld/e7_prompts_v2.py",
            "benchmarks/tau/delta_card_tau/e7_prompts.py",
        ]:
            with self.subTest(file=relative):
                self.assertGreater((ROOT / relative).stat().st_size, 1000)

    def test_public_role_names_are_executor_and_generator(self) -> None:
        checked_suffixes = {".py", ".sh", ".yaml", ".yml", ".md"}
        legacy_terms = ("sol" + "ver", "extr" + "actor")
        violations = []
        for path in ROOT.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in checked_suffixes:
                continue
            if any(part in {".git", "__pycache__"} for part in path.parts):
                continue
            text = path.read_text(encoding="utf-8", errors="replace").lower()
            for term in legacy_terms:
                if term in text:
                    violations.append(f"{path.relative_to(ROOT)}:{term}")
        self.assertEqual(violations, [])

        self.assert_contains_all(
            "framework/slime_legacy/slime/backends/megatron_utils/actor.py",
            [
                'frozenset({"executor", "downstream_executor"})',
                'role == "generator"',
                'generator_arg_name="generator_entropy_coef"',
                'executor_arg_name="executor_entropy_coef"',
            ],
        )


if __name__ == "__main__":
    unittest.main()
