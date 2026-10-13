import importlib.util
from pathlib import Path
import sys
import types
from types import SimpleNamespace
from unittest import TestCase, mock


class WandbContract(TestCase):
    def test_disabled_mode_never_initializes_wandb(self):
        path = (
            Path(__file__).parents[1]
            / "framework/slime_legacy/slime/utils/wandb_utils.py"
        )
        spec = importlib.util.spec_from_file_location("release_wandb_utils", path)
        module = importlib.util.module_from_spec(spec)
        fake_wandb = types.ModuleType("wandb")
        fake_wandb.init = mock.Mock()
        with mock.patch.dict(sys.modules, {"wandb": fake_wandb}):
            spec.loader.exec_module(module)
        args = SimpleNamespace(use_wandb=True, wandb_mode="disabled")
        with mock.patch.object(module.wandb, "init") as init:
            module.init_wandb_primary(args)
        init.assert_not_called()
        self.assertFalse(args.use_wandb)
        self.assertIsNone(args.wandb_run_id)
