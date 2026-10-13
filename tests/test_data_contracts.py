from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class DataContracts(unittest.TestCase):
    def test_release_contains_runtime_model_and_converter_sources(self) -> None:
        required = (
            "framework/slime_legacy/scripts/models/qwen3-8B.sh",
            "framework/slime_tau/scripts/models/qwen3-8B.sh",
            "benchmarks/alfworld/src/data/alfworld/extract.py",
            "benchmarks/webshop/scripts/download_data.sh",
            "benchmarks/tau/scripts/validate_qwen3_tool_template.py",
            "benchmarks/tau/scripts/build_domain_balanced_prompt_data.py",
            "benchmarks/tau/scripts/make_qwen35_tau2_chat_template.py",
        )
        for relative in required:
            with self.subTest(file=relative):
                self.assertTrue((ROOT / relative).is_file())

    def test_webshop_split_matches_training_and_evaluation_protocol(self) -> None:
        protocol = json.loads(
            (ROOT / "benchmarks/webshop/protocol/splits.json").read_text()
        )
        self.assertEqual(protocol["meta"]["total_goals"], 12087)
        self.assertEqual(protocol["meta"]["catalogue_products"], 50000)
        expected = {"train": 3320, "val_id": 454, "val_ood": 207}
        all_indices: dict[str, set[int]] = {}
        for name, count in expected.items():
            indices = protocol[name]["goal_idxs"]
            self.assertEqual(len(indices), count)
            self.assertEqual(len(set(indices)), count)
            self.assertEqual(sum(protocol[name]["by_category"].values()), count)
            all_indices[name] = set(indices)
        self.assertFalse(all_indices["train"] & all_indices["val_id"])
        self.assertFalse(all_indices["train"] & all_indices["val_ood"])
        self.assertFalse(all_indices["val_id"] & all_indices["val_ood"])

    def test_scienceworld_protocol_is_fully_materializable(self) -> None:
        source = (
            ROOT / "benchmarks/scienceworld/scripts/prepare_data.py"
        ).read_text()
        for fragment in (
            '"train": 460',
            '"dev": 155',
            '"test_id_dev230": 230',
            '"test_ood155": 155',
            '"max_steps": 30',
        ):
            self.assertIn(fragment, source)

    def test_published_training_data_fingerprints_are_guarded(self) -> None:
        alfworld = (
            ROOT / "benchmarks/alfworld/scripts/prepare_full3553.py"
        ).read_text()
        self.assertIn(
            "9056ef9f4df1e2c3355c3f96313d5daef96fffb0b8170d085b959f84662c8279",
            alfworld,
        )
        tau = (ROOT / "scripts/audit_tau_hf_data.py").read_text()
        self.assertIn("900", tau)

    def test_alfworld_builder_materializes_every_training_input(self) -> None:
        source = (ROOT / "benchmarks/alfworld/scripts/prepare_full3553.py").read_text()
        for fragment in (
            '"train": 3553',
            '"train_seen": 2432',
            '"train_unseen": 1121',
            '"valid_seen_id_types": 103',
            '"valid_seen_heldout_types": 37',
            '"valid_unseen_id_types": 99',
            '"valid_unseen_heldout_types": 35',
        ):
            self.assertIn(fragment, source)

    def test_setup_pins_external_environment_sources(self) -> None:
        setup = (ROOT / "scripts/setup_env.sh").read_text()
        self.assertIn("703e635619901c9c84f76caff0907a37d1a262a8", setup)
        self.assertIn("5ebebbe827b455b3ed04fcb9294235c6ef4e5fd6", setup)
        self.assertIn('pip install -e "${TAU2_BENCH_DIR}" --no-deps', setup)

    def test_task_environment_versions_match_the_audited_runs(self) -> None:
        expected = {
            "environments/alfworld.yaml": (
                "alfworld==0.4.2",
                "textworld==1.7.0",
                "gymnasium==0.29.1",
            ),
            "environments/webshop.yaml": (
                "gym==0.24.0",
                "werkzeug==2.1.2",
                "pyserini==1.2.0",
            ),
            "environments/scienceworld.yaml": ("scienceworld==1.2.3",),
            "environments/tau2.yaml": (
                "litellm==1.65.0",
                "gymnasium==1.3.0",
                "openai==2.6.1",
            ),
            "environments/tau2_eval.yaml": (
                "litellm==1.65.0",
                "openai==2.6.1",
                "vllm==0.11.0",
            ),
        }
        for relative, fragments in expected.items():
            source = (ROOT / relative).read_text()
            for fragment in fragments:
                with self.subTest(file=relative, fragment=fragment):
                    self.assertIn(fragment, source)


if __name__ == "__main__":
    unittest.main()
