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
export EVAL_INTERVAL=1
export WANDB_MODE="${WANDB_MODE:-disabled}"
export EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0.4}"
export EVAL_TOP_P="${EVAL_TOP_P:-1.0}"
export N_SAMPLES_PER_EVAL_PROMPT="${N_SAMPLES_PER_EVAL_PROMPT:-8}"
export MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-30}"
export EXECUTOR_MAX_RESPONSE_LEN="${EXECUTOR_MAX_RESPONSE_LEN:-384}"
export ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-12288}"
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-8192}"
export SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.55}"
export RAY_SUBMISSION_ID="${RAY_SUBMISSION_ID:-skillscaffold-scienceworld-eval-$(date +%Y%m%d-%H%M%S)}"

if [[ "${VALIDATE_EVAL_SPLITS:-1}" == "1" ]]; then
  EVAL_ID_DATA="${SCIENCEWORLD_EVAL_ID_DATA:-${EVAL_ID_DATA:-}}"
  EVAL_OOD_DATA="${SCIENCEWORLD_EVAL_OOD_DATA:-${EVAL_OOD_DATA:-}}"
  : "${EVAL_ID_DATA:?Set SCIENCEWORLD_EVAL_ID_DATA to the 230-task ScienceWorld ID parquet}"
  : "${EVAL_OOD_DATA:?Set SCIENCEWORLD_EVAL_OOD_DATA to the 155-task ScienceWorld OOD parquet}"
  export EVAL_ID_DATA EVAL_OOD_DATA
  "${PYTHON_BIN:-python3}" - "${EVAL_ID_DATA}" "${EVAL_OOD_DATA}" <<'PY'
import sys
import pyarrow.parquet as pq

expected = ((sys.argv[1], 230, "ID"), (sys.argv[2], 155, "OOD"))
for path, count, split in expected:
    actual = pq.read_metadata(path).num_rows
    if actual != count:
        raise SystemExit(f"expected {count} ScienceWorld {split} tasks, found {actual}: {path}")
PY
fi

exec bash "${SCRIPT_DIR}/train.sh"
