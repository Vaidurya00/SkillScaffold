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
export DISABLE_TRAIN_EVAL=0
export NO_LOAD_OPTIM=1
export USE_CHECKPOINT_OPT_PARAM_SCHEDULER=1
export USE_KL_LOSS=0
export EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0.7}"
export EVAL_TOP_P="${EVAL_TOP_P:-0.85}"
export N_SAMPLES_PER_EVAL_PROMPT="${N_SAMPLES_PER_EVAL_PROMPT:-8}"
export ROLLOUT_MAX_CONTEXT_LEN="${ROLLOUT_MAX_CONTEXT_LEN:-32768}"
export EVAL_CONCURRENCY="${EVAL_CONCURRENCY:-128}"
if [[ -z "${APPLY_CHAT_TEMPLATE_KWARGS:-}" ]]; then
  export APPLY_CHAT_TEMPLATE_KWARGS='{"enable_thinking":false}'
else
  export APPLY_CHAT_TEMPLATE_KWARGS
fi
export CUSTOM_GENERATE_FUNCTION_PATH="${CUSTOM_GENERATE_FUNCTION_PATH:-src.webshop.generate.eval.run_eval}"
export RAY_NO_WAIT=0
export RAY_SUBMISSION_ID="${RAY_SUBMISSION_ID:-skillscaffold-webshop-eval-$(date +%Y%m%d-%H%M%S)}"

if [[ "${VALIDATE_EVAL_SPLITS:-1}" == "1" && -z "${WEBSHOP_EVAL_PROMPT_DATA:-}" ]]; then
  ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
  DATA_DIR="${WEBSHOP_PROCESSED_DATA_DIR:-${ROOT_DIR}/data/webshop}"
  export VAL_ID_DATA="${VAL_ID_DATA:-${DATA_DIR}/val_id.parquet}"
  export VAL_OOD_DATA="${VAL_OOD_DATA:-${DATA_DIR}/val_ood.parquet}"
  "${PYTHON_BIN:-python3}" - "${VAL_ID_DATA}" "${VAL_OOD_DATA}" <<'PY'
import sys
import pyarrow.parquet as pq

expected = ((sys.argv[1], 454, "ID"), (sys.argv[2], 207, "OOD"))
for path, count, split in expected:
    actual = pq.read_metadata(path).num_rows
    if actual != count:
        raise SystemExit(f"expected {count} WebShop {split} tasks, found {actual}: {path}")
PY
fi

exec bash "${SCRIPT_DIR}/train.sh"
