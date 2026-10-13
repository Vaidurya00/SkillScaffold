#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
if [[ -f "${REPO_ROOT}/.env" ]]; then
  set -a
  source "${REPO_ROOT}/.env"
  set +a
fi

LOAD_CHECKPOINT="${EVAL_CHECKPOINT:-${LOAD_CHECKPOINT:-}}"
: "${LOAD_CHECKPOINT:?Set EVAL_CHECKPOINT or LOAD_CHECKPOINT to the trained Megatron checkpoint}"

export LOAD_CHECKPOINT
export NUM_ROLLOUT=0
export EVAL_ONLY_LOAD=1
export USE_WANDB="${USE_WANDB:-0}"
export EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-1.0}"
export EVAL_TOP_P="${EVAL_TOP_P:-0.85}"
export N_SAMPLES_PER_EVAL_PROMPT="${N_SAMPLES_PER_EVAL_PROMPT:-8}"
export ROLLOUT_MAX_CONTEXT_LEN="${ROLLOUT_MAX_CONTEXT_LEN:-32768}"
export MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-100}"
export EXECUTOR_MAX_RESPONSE_LEN="${EXECUTOR_MAX_RESPONSE_LEN:-1024}"
export EVAL_MAX_RESPONSE_LEN="${EVAL_MAX_RESPONSE_LEN:-1536}"
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-8192}"
export OPTIMIZER_CPU_OFFLOAD="${OPTIMIZER_CPU_OFFLOAD:-0}"
export SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.55}"
export SGLANG_DISABLE_CUDA_GRAPH="${SGLANG_DISABLE_CUDA_GRAPH:-1}"
export SGLANG_ENABLE_DETERMINISTIC_INFERENCE="${SGLANG_ENABLE_DETERMINISTIC_INFERENCE:-1}"
export RAY_SUBMISSION_ID="${RAY_SUBMISSION_ID:-skillscaffold-alfworld-eval-$(date +%Y%m%d-%H%M%S)}"
export ALFWORLD_EVAL_SPLIT="${ALFWORLD_EVAL_SPLIT:-valid_unseen}"

if [[ "${VALIDATE_EVAL_SPLITS:-1}" == "1" && -z "${ALFWORLD_EVAL_PROMPT_DATA:-}" ]]; then
  : "${ALFWORLD_PARQUET_DIR:?Set ALFWORLD_PARQUET_DIR to the evaluation parquets}"
  "${PYTHON_BIN:-python3}" - "${ALFWORLD_PARQUET_DIR}" "${ALFWORLD_EVAL_SPLIT}" <<'PY'
import sys
from pathlib import Path
import pandas as pd

root = Path(sys.argv[1])
split = sys.argv[2]
if split != "valid_unseen":
    raise SystemExit(f"the default protocol requires ALFWORLD_EVAL_SPLIT=valid_unseen, got {split}")
names = ["valid_unseen_id_types.parquet", "valid_unseen_heldout_types.parquet"]
frames = [pd.read_parquet(root / name, columns=["id"]) for name in names]
unique = pd.concat(frames, ignore_index=True)["id"].nunique()
if unique != 134:
    raise SystemExit(f"expected 134 unique ALFWorld valid_unseen tasks, found {unique}")
PY
fi

exec bash "${SCRIPT_DIR}/train.sh"
