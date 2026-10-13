from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ReleaseWorkflowTests(unittest.TestCase):
    def test_archive_excludes_local_environment_and_configuration(self) -> None:
        spec = importlib.util.spec_from_file_location(
            "build_release", ROOT / "scripts/build_release.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            destination = Path(temporary) / "release"
            private_files = (
                ".venv/lib/site-packages/local.py",
                "venv/pyvenv.cfg",
                ".env",
                ".env.local",
                ".DS_Store",
                ".copy_quarantine_path",
                "benchmarks/alfworld/configs/alfworld.yaml",
            )
            public_files = (
                ".env.example",
                "benchmarks/alfworld/configs/alfworld.yaml.example",
                "scripts/setup_env.sh",
            )
            for relative in (*private_files, *public_files):
                path = source / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture\n")
            module.ROOT = source
            module.copy_source(destination)
            actual = {
                str(path.relative_to(destination))
                for path in destination.rglob("*")
                if path.is_file()
            }
            self.assertEqual(actual, set(public_files))

    def test_generated_env_preserves_shell_special_characters(self) -> None:
        source = (ROOT / "scripts/setup_env.sh").read_text()
        function = source.split("set_env_value() {", 1)[1].split(
            "\n}\n", 1
        )[0]
        code = function.split("<<'PY'\n", 1)[1].split("\nPY", 1)[0]
        with tempfile.TemporaryDirectory() as temporary:
            env_file = Path(temporary) / ".env"
            env_file.write_text("MEGATRON_DIR=old\nUNCHANGED=value\n")
            value = str(Path(temporary) / "space 'quote' $VALUE $(false) `false`")
            namespace = {"__name__": "__main__"}
            import sys
            from unittest.mock import patch

            with patch.object(sys, "argv", ["setup", str(env_file), "MEGATRON_DIR", value]):
                exec(compile(code, "setup-env-block", "exec"), namespace)
            result = subprocess.run(
                ["bash", "-eu", "-c", 'source "$1"; printf "%s" "$MEGATRON_DIR"', "test", str(env_file)],
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertEqual(result.stdout, value)
            self.assertIn("UNCHANGED=value", env_file.read_text())

    def test_webshop_config_uses_selected_python(self) -> None:
        source = (ROOT / "benchmarks/webshop/scripts/train.sh").read_text()
        function = next(line for line in source.splitlines() if line.startswith("_cfg()"))
        with tempfile.TemporaryDirectory() as temporary:
            executable = Path(temporary) / "selected python"
            executable.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n')
            executable.chmod(0o755)
            result = subprocess.run(
                ["bash", "-eu", "-c", function + '\n_cfg "config path.yaml" "setting"'],
                env={**os.environ, "PYTHON_BIN": str(executable)},
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertEqual(result.stdout.splitlines()[-2:], ["config path.yaml", "setting"])


if __name__ == "__main__":
    unittest.main()
