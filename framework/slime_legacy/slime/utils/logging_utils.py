import fcntl
import json
import logging
import os
import socket
import time

import wandb

from . import wandb_utils
from .tensorboard_utils import _TensorboardAdapter

_LOGGER_CONFIGURED = False


# ref: SGLang
def configure_logger(prefix: str = ""):
    global _LOGGER_CONFIGURED
    if _LOGGER_CONFIGURED:
        return

    _LOGGER_CONFIGURED = True

    logging.basicConfig(
        level=logging.INFO,
        format=f"[%(asctime)s{prefix}] %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def init_tracking(args, primary: bool = True, **kwargs):
    if primary:
        wandb_utils.init_wandb_primary(args, **kwargs)
    else:
        wandb_utils.init_wandb_secondary(args, **kwargs)


def finish_tracking(args):
    if not args.use_wandb:
        return
    try:
        if wandb.run is not None:
            wandb.finish()
    except Exception:
        logging.getLogger(__name__).exception("Failed to finish wandb run")


# TODO further refactor, e.g. put TensorBoard init to the "init" part
def log(args, metrics, step_key: str):
    if args.use_wandb:
        wandb.log(metrics)

    if args.use_tensorboard:
        metrics_except_step = {k: v for k, v in metrics.items() if k != step_key}
        _TensorboardAdapter(args).log(data=metrics_except_step, step=metrics[step_key])

    # Always keep a lightweight, append-only local record when requested.  This
    # covers both rollout-side metrics and optimizer-side metrics (including
    # role-specific generator/executor losses) without requiring W&B.
    metrics_path = os.environ.get("DYNAMICS_METRICS_PATH", "").strip()
    if metrics_path:
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

        def _json_default(value):
            if hasattr(value, "item"):
                return value.item()
            return str(value)

        record = {
            "time_unix": time.time(),
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "stream": step_key,
            "metrics": metrics,
        }
        with open(metrics_path, "a", encoding="utf-8") as output:
            fcntl.flock(output.fileno(), fcntl.LOCK_EX)
            output.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
            output.flush()
            fcntl.flock(output.fileno(), fcntl.LOCK_UN)
