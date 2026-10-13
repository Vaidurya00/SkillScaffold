from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]


class GpuTopologyTests(unittest.TestCase):
    def test_launchers_handle_pcie_nvlink_and_unavailable_topology(self):
        fixtures = (
            ('printf "%s\\n" "GPU0 GPU1 SYS"', "0"),
            ('printf "%s\\n" "GPU0 X NV12" "GPU1 NV12 X"', "2"),
            ('return 1', "0"),
        )
        for benchmark in ("alfworld", "webshop", "scienceworld"):
            source = (ROOT / "benchmarks" / benchmark / "scripts/train.sh").read_text()
            detection = next(line for line in source.splitlines() if line.startswith("NVLINK_COUNT="))
            for output, expected in fixtures:
                with self.subTest(benchmark=benchmark, expected=expected, output=output):
                    script = 'nvidia-smi() { ' + output + '; }\n' + detection + '\nprintf "%s" "$NVLINK_COUNT"'
                    result = subprocess.run(
                        ["bash", "-euo", "pipefail", "-c", script],
                        capture_output=True, text=True, check=True,
                    )
                    self.assertEqual(result.stdout, expected)
