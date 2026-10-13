"""Durable per-group evidence for the positive-Delta launch gate."""
import json
import os
from pathlib import Path


def _write(args, kind, key, value):
    root = Path(os.environ["SCIENCEWORLD_E7_AUDIT_DIR"])
    root = root / kind / str(getattr(args, "current_rollout_id", "unknown"))
    root.mkdir(parents=True, exist_ok=True)
    path = root / (str(key).replace("/", "_") + ".json")
    temp = path.with_suffix(f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str))
    temp.replace(path)


def save_group(args, source, samples):
    rows = [{"index": s.index, "group_index": s.group_index,
             "reward": s.reward, "metadata": s.metadata,
             "train_metadata": s.train_metadata,
             "prompt": s.prompt, "response": s.response} for s in samples]
    _write(args, "train", source.group_index, rows)
    executors = [s for s in samples if (s.metadata or {}).get("trajectory_role") == "executor"]
    if executors:
        m = executors[0].metadata
        print("SCIENCEWORLD_E7_GROUP " + json.dumps({k: m.get(k) for k in
            ["group_delta_scores", "generator_valid_count", "generator_positive_delta_count",
             "selected_skill_injected", "selected_skill_delta", "group_delta_loss_mask_mismatch_count"]}), flush=True)


def save_eval(args, sample):
    record_id = (sample.metadata or {}).get("record_id", "")
    _write(args, "eval", f"{record_id}__sample_{sample.index}",
           {"reward": sample.reward, "metadata": sample.metadata, "response": sample.response})
