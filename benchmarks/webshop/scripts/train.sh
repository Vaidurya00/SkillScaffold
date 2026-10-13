#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${ROOT_DIR}/../.." >/dev/null 2>&1 && pwd)"
if [[ -f "${REPO_ROOT}/.env" ]]; then
  set -a
  source "${REPO_ROOT}/.env"
  set +a
fi
WORK_DIR="${ROOT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${ROOT_DIR}:${SLIME_DIR:-${REPO_ROOT}/framework/slime_legacy}${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON_SITE_PACKAGES="$("${PYTHON_BIN}" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"

MEGATRON_DIR="${MEGATRON_DIR:?Set MEGATRON_DIR to your Megatron-LM checkout}"
SLIME_DIR="${SLIME_DIR:-${REPO_ROOT}/framework/slime_legacy}"
SLIME_TRAIN_SCRIPT="${SLIME_DIR}/train.py"
MODEL_SOURCE_ARGS="${MODEL_SOURCE_ARGS:-${SLIME_DIR}/scripts/models/qwen3-8B.sh}"
HF_CHECKPOINT="${HF_CHECKPOINT:?Set HF_CHECKPOINT to the Hugging Face Qwen3-8B checkpoint}"
REF_CHECKPOINT="${REF_CHECKPOINT:?Set REF_CHECKPOINT to the Megatron-format reference checkpoint}"
EXP_NAME="${EXP_NAME:-skillscaffold-webshop-qwen3-8b}"

SAVE_CHECKPOINT="${SAVE_CHECKPOINT:-${REPO_ROOT}/outputs/checkpoints/webshop/${EXP_NAME}}"
LOAD_CHECKPOINT="${LOAD_CHECKPOINT:-${REF_CHECKPOINT}}"

DATA_DIR="${WEBSHOP_PROCESSED_DATA_DIR:-${ROOT_DIR}/data/webshop}"
TRAIN_DATA="${WEBSHOP_TRAIN_DATA:-${DATA_DIR}/train.parquet}"
VAL_ID_DATA="${VAL_ID_DATA:-${DATA_DIR}/val_id.parquet}"
VAL_OOD_DATA="${VAL_OOD_DATA:-${DATA_DIR}/val_ood.parquet}"
EVAL_ID_NAME="${EVAL_ID_NAME:-webshop_id}"
EVAL_OOD_NAME="${EVAL_OOD_NAME:-webshop_ood}"
WEBSHOP_EVAL_PROMPT_DATA="${WEBSHOP_EVAL_PROMPT_DATA:-}"
ALL_DATA="${DATA_DIR}/all.parquet"
CACHE_FILE="${DATA_DIR}/cache/unified_embeddings.npz"
WEBSHOP_DATA_DIR="${WEBSHOP_DATA_DIR:?Set WEBSHOP_DATA_DIR to the prepared WebShop data directory}"
WEBSHOP_VENDOR_ROOT="${WEBSHOP_VENDOR_ROOT:?Set WEBSHOP_VENDOR_ROOT to your WebShop checkout}"
WEBSHOP_ITEMS="${WEBSHOP_DATA_DIR}/items_shuffle_human.json"
WEBSHOP_ATTRS="${WEBSHOP_DATA_DIR}/items_ins_v2_human.json"
WEBSHOP_PYTHON="${WEBSHOP_PYTHON:-python3}"

WEBSHOP_CONFIG="${WEBSHOP_CONFIG:-${ROOT_DIR}/configs/server.yaml}"
_cfg() { "${PYTHON_BIN}" -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))[sys.argv[2]])" "$1" "$2"; }
WEBSHOP_ENV_URL=$(_cfg "$WEBSHOP_CONFIG" webshop_env_url)
WEBSHOP_RETRIEVE_URL_TEST=$(_cfg "$WEBSHOP_CONFIG" webshop_retrieve_url_test)
WEBSHOP_ROUTER_HOST=$(_cfg "$WEBSHOP_CONFIG" webshop_router_host)
WEBSHOP_ROUTER_PORT=$(_cfg "$WEBSHOP_CONFIG" webshop_router_port)
WEBSHOP_WORKER_HOST=$(_cfg "$WEBSHOP_CONFIG" webshop_worker_host)
WEBSHOP_WORKER_START_PORT=$(_cfg "$WEBSHOP_CONFIG" webshop_worker_start_port)
WEBSHOP_NUM_WORKERS=$(_cfg "$WEBSHOP_CONFIG" webshop_num_workers)
WEBSHOP_RETRIEVE_TEST_PORT=$(_cfg "$WEBSHOP_CONFIG" webshop_retrieve_test_port)
WEBSHOP_PID_FILE="$(_cfg "$WEBSHOP_CONFIG" webshop_pid_file)"
WEBSHOP_RETRIEVE_PID_FILE="$(_cfg "$WEBSHOP_CONFIG" webshop_retrieve_pid_file)"
WEBSHOP_STARTUP_TIMEOUT=$(_cfg "$WEBSHOP_CONFIG" webshop_startup_timeout)
WEBSHOP_RETRIEVE_STARTUP_TIMEOUT=$(_cfg "$WEBSHOP_CONFIG" webshop_retrieve_startup_timeout)
WEBSHOP_RETRIEVE_HOST=$(_cfg "$WEBSHOP_CONFIG" webshop_retrieve_host)
WEBSHOP_RETRIEVE_QUESTION_COL=$(_cfg "$WEBSHOP_CONFIG" webshop_retrieve_question_col)
LOG_DIR="$(_cfg "$WEBSHOP_CONFIG" webshop_log_dir)"
if [[ "${WEBSHOP_PID_FILE}" != /* ]]; then WEBSHOP_PID_FILE="${ROOT_DIR}/${WEBSHOP_PID_FILE}"; fi
if [[ "${WEBSHOP_RETRIEVE_PID_FILE}" != /* ]]; then WEBSHOP_RETRIEVE_PID_FILE="${ROOT_DIR}/${WEBSHOP_RETRIEVE_PID_FILE}"; fi
if [[ "${LOG_DIR}" != /* ]]; then LOG_DIR="${ROOT_DIR}/${LOG_DIR}"; fi
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-15}"
EXECUTOR_MAX_RESPONSE_LEN="${EXECUTOR_MAX_RESPONSE_LEN:-512}"
N_EXPERIENCES="${N_EXPERIENCES:-4}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-1}"
NUM_ROLLOUT="${NUM_ROLLOUT:-200}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-16}"
ROLLOUT_MAX_CONTEXT_LEN="${ROLLOUT_MAX_CONTEXT_LEN:-16384}"
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-1536}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-16384}"
EVAL_INTERVAL="${EVAL_INTERVAL:-25}"
DISABLE_TRAIN_EVAL="${DISABLE_TRAIN_EVAL:-1}"
EVAL_CONCURRENCY="${EVAL_CONCURRENCY:-512}"
N_SAMPLES_PER_EVAL_PROMPT="${N_SAMPLES_PER_EVAL_PROMPT:-1}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0.0}"
EVAL_TOP_P="${EVAL_TOP_P:-1.0}"
APPLY_CHAT_TEMPLATE_KWARGS="${APPLY_CHAT_TEMPLATE_KWARGS:-}"
if [[ -z "${APPLY_CHAT_TEMPLATE_KWARGS}" ]]; then
  APPLY_CHAT_TEMPLATE_KWARGS='{"enable_thinking":false}'
fi
SAVE_INTERVAL="${SAVE_INTERVAL:-25}"
CUSTOM_GENERATE_FUNCTION_PATH="${CUSTOM_GENERATE_FUNCTION_PATH:-src.webshop.pipeline.webshop_pipeline}"
CUSTOM_EVAL_LOG_FUNCTION_PATH="${CUSTOM_EVAL_LOG_FUNCTION_PATH:-src.webshop.loggers.eval_logger.log_eval_by_task_type}"
SAVE_DEBUG_ROLLOUT_DATA="${SAVE_DEBUG_ROLLOUT_DATA:-}"
USE_CHECKPOINT_OPT_PARAM_SCHEDULER="${USE_CHECKPOINT_OPT_PARAM_SCHEDULER:-0}"
OVERRIDE_OPT_PARAM_SCHEDULER="${OVERRIDE_OPT_PARAM_SCHEDULER:-0}"
NO_LOAD_OPTIM="${NO_LOAD_OPTIM:-0}"
SEED="${SEED:-1291}"
ROLLOUT_SEED="${ROLLOUT_SEED:-91}"
LR="${LR:-1e-6}"
GENERATOR_LOSS_WEIGHT="${GENERATOR_LOSS_WEIGHT:-${GENERATOR_REWARD_WEIGHT:-0.1}}"
EXECUTOR_KL_LOSS_COEF="${EXECUTOR_KL_LOSS_COEF:-0.0}"
GENERATOR_KL_LOSS_COEF="${GENERATOR_KL_LOSS_COEF:-0.0}"
USE_KL_LOSS="${USE_KL_LOSS:-0}"
EPS_CLIP="${EPS_CLIP:-0.2}"
EPS_CLIP_HIGH="${EPS_CLIP_HIGH:-0.28}"



MASTER_ADDR="127.0.0.1"
NUM_GPUS="${NUM_GPUS:-8}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
RAY_DASHBOARD_ADDRESS="${RAY_DASHBOARD_ADDRESS:-http://127.0.0.1:${RAY_DASHBOARD_PORT}}"
RAY_REUSE_EXISTING="${RAY_REUSE_EXISTING:-1}"
RAY_NO_WAIT="${RAY_NO_WAIT:-0}"
RAY_SUBMISSION_ID="${RAY_SUBMISSION_ID:-}"
WEBSHOP_REUSE_SERVERS="${WEBSHOP_REUSE_SERVERS:-0}"
WEBSHOP_CLEANUP_SERVERS_ON_EXIT="${WEBSHOP_CLEANUP_SERVERS_ON_EXIT:-0}"
DEBUG_ROLLOUT_ONLY="${DEBUG_ROLLOUT_ONLY:-0}"
SGLANG_BASE_PORT="${SGLANG_BASE_PORT:-15000}"
SGLANG_ROUTER_PORT="${SGLANG_ROUTER_PORT:-}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.65}"

mkdir -p \
  "${LOG_DIR}" \
  "${DATA_DIR}/cache" \
  "$(dirname "${WEBSHOP_PID_FILE}")" \
  "$(dirname "${WEBSHOP_RETRIEVE_PID_FILE}")"

wait_for_health() {
  local url="$1"
  local timeout="${2:-200}"
  "${PYTHON_BIN}" - "$url" "$timeout" <<'PY'
import sys
import time
from urllib.request import Request, urlopen

url = sys.argv[1]
timeout = float(sys.argv[2])
deadline = time.time() + timeout
last_err = None

while time.time() < deadline:
    try:
        with urlopen(Request(url, method="GET"), timeout=5) as resp:
            if 200 <= resp.status < 300:
                sys.exit(0)
    except Exception as exc:
        last_err = exc
    time.sleep(1.0)

raise SystemExit(f"[web/train] timed out waiting for {url}: {last_err}")
PY
}

cleanup_webshop_servers() {
  [[ "${WEBSHOP_CLEANUP_SERVERS_ON_EXIT}" == "1" ]] || return 0
  "${PYTHON_BIN}" "${ROOT_DIR}/env/webshop/stop_servers.py" --pid-file "${WEBSHOP_PID_FILE}" || true
  if [[ "${NUM_ROLLOUT}" != "0" && -f "${WEBSHOP_RETRIEVE_PID_FILE}" ]]; then
    while read -r pid; do
      [[ "${pid}" =~ ^[0-9]+$ ]] || continue
      if [[ -r "/proc/${pid}/cmdline" ]] &&
         tr '\0' ' ' < "/proc/${pid}/cmdline" | grep -Fq "${ROOT_DIR}/env/webshop/retrieve_server/retrieve_server.py"; then
        kill "${pid}" 2>/dev/null || true
      fi
    done < "${WEBSHOP_RETRIEVE_PID_FILE}"
    rm -f "${WEBSHOP_RETRIEVE_PID_FILE}"
  fi
}
if [[ "${WEBSHOP_CLEANUP_SERVERS_ON_EXIT}" == "1" ]]; then
  trap cleanup_webshop_servers EXIT
fi

if [ ! -f "${MODEL_SOURCE_ARGS}" ]; then
  echo "[web/train] missing model args script: ${MODEL_SOURCE_ARGS}" >&2
  exit 1
fi

if [ ! -f "${TRAIN_DATA}" ] || [ ! -f "${VAL_ID_DATA}" ] || [ ! -f "${VAL_OOD_DATA}" ] || [ ! -f "${ALL_DATA}" ]; then
  bash "${ROOT_DIR}/scripts/prepare_data.sh"
fi

if [[ "${NUM_ROLLOUT}" != "0" && ! -f "${CACHE_FILE}" ]]; then
  bash "${ROOT_DIR}/scripts/build_cache.sh"
fi

if [[ "${WEBSHOP_REUSE_SERVERS}" != "1" ]]; then
  "${PYTHON_BIN}" "${ROOT_DIR}/env/webshop/stop_servers.py" --pid-file "${WEBSHOP_PID_FILE}" || true
  if [[ "${NUM_ROLLOUT}" != "0" && -f "${WEBSHOP_RETRIEVE_PID_FILE}" ]]; then
    while read -r pid; do
      [[ "${pid}" =~ ^[0-9]+$ ]] || continue
      if [[ -r "/proc/${pid}/cmdline" ]] &&
         tr '\0' ' ' < "/proc/${pid}/cmdline" | grep -Fq "${ROOT_DIR}/env/webshop/retrieve_server/retrieve_server.py"; then
        kill "${pid}" 2>/dev/null || true
      fi
    done < "${WEBSHOP_RETRIEVE_PID_FILE}"
    rm -f "${WEBSHOP_RETRIEVE_PID_FILE}"
  fi
fi

source "${MODEL_SOURCE_ARGS}"

NVLINK_COUNT="$( { nvidia-smi topo -m 2>/dev/null || true; } | awk '{count += gsub(/NV[0-9]+/, "")} END {print count + 0}')"
if [ "${NVLINK_COUNT}" -gt 0 ]; then HAS_NVLINK=1; else HAS_NVLINK=0; fi
echo "HAS_NVLINK: ${HAS_NVLINK} (detected ${NVLINK_COUNT} NVLink references)"

CKPT_ARGS=(
  --hf-checkpoint "${HF_CHECKPOINT}"
  --ref-load "${REF_CHECKPOINT}"
  --load "${LOAD_CHECKPOINT}"
  --save "${SAVE_CHECKPOINT}"
  --save-interval "${SAVE_INTERVAL}"
)

CUSTOM_ARGS=(
  --custom-generate-function-path "${CUSTOM_GENERATE_FUNCTION_PATH}"
  --custom-reward-post-process-path src.webshop.pipeline.webshop_reward_post_process
  --custom-eval-rollout-log-function-path "${CUSTOM_EVAL_LOG_FUNCTION_PATH}"
  --custom-rollout-log-function-path src.webshop.loggers.rollout_logger.log_rollout_data
  --custom-config-path "${WEBSHOP_CONFIG}"
)

ROLLOUT_ARGS=(
  --prompt-data "${TRAIN_DATA}"
  --input-key prompt
  --label-key metadata
  --apply-chat-template
  --rollout-shuffle

  --n-experiences "${N_EXPERIENCES}"
  --max-episode-steps "${MAX_EPISODE_STEPS}"
  --executor-max-response-len "${EXECUTOR_MAX_RESPONSE_LEN}"

  --generator-reward-weight "${GENERATOR_LOSS_WEIGHT}"
  --executor-reward-weight 1.0
  --executor-temperature 1.0
  --skill-format-penalty 1

  --num-rollout "${NUM_ROLLOUT}"
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
  --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
  --seed "${SEED}"
  --rollout-seed "${ROLLOUT_SEED}"
  --rollout-max-context-len "${ROLLOUT_MAX_CONTEXT_LEN}"
  --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
  --rollout-top-p 0.85
  --executor-top-p 0.85
  --rollout-temperature 1.0
  --global-batch-size "${GLOBAL_BATCH_SIZE}"
  --balance-data
)
if [[ -n "${APPLY_CHAT_TEMPLATE_KWARGS}" ]]; then
  ROLLOUT_ARGS+=(--apply-chat-template-kwargs "${APPLY_CHAT_TEMPLATE_KWARGS}")
fi

EVAL_ARGS=()
if [[ "${DISABLE_TRAIN_EVAL}" != "1" ]]; then
  if [[ -n "${WEBSHOP_EVAL_PROMPT_DATA}" ]]; then
    EVAL_DATA_ARGS=(probe "${WEBSHOP_EVAL_PROMPT_DATA}")
  else
    EVAL_DATA_ARGS=("${EVAL_ID_NAME}" "${VAL_ID_DATA}" "${EVAL_OOD_NAME}" "${VAL_OOD_DATA}")
  fi
  EVAL_ARGS=(
    --skip-eval-before-train
    --eval-prompt-data "${EVAL_DATA_ARGS[@]}"
    --eval-interval "${EVAL_INTERVAL}"
    --n-samples-per-eval-prompt "${N_SAMPLES_PER_EVAL_PROMPT}"
    --eval-max-response-len 4096
    --eval-temperature "${EVAL_TEMPERATURE}"
    --eval-top-p "${EVAL_TOP_P}"
    --eval-concurrency "${EVAL_CONCURRENCY}"
  )
fi

PERF_ARGS=(
  --tensor-model-parallel-size 2
  --sequence-parallel
  --pipeline-model-parallel-size 1
  --context-parallel-size 1
  --expert-model-parallel-size 1
  --expert-tensor-parallel-size 1
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --use-dynamic-batch-size
  --use-dynamic-global-batch-size
  --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
)

GRPO_ARGS=(
  --advantage-estimator grpo
  --no-normalize-advantages
  --executor-entropy-coef 0.0
  --generator-entropy-coef 0.0
  --executor-kl-loss-coef "${EXECUTOR_KL_LOSS_COEF}"
  --generator-kl-loss-coef "${GENERATOR_KL_LOSS_COEF}"
  --kl-loss-coef 0.01
  --kl-loss-type low_var_kl
  --entropy-coef 0.0
  --eps-clip "${EPS_CLIP}"
  --eps-clip-high "${EPS_CLIP_HIGH}"
)
if [[ "${USE_KL_LOSS}" == "1" ]]; then
  GRPO_ARGS+=(--use-kl-loss)
fi

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr "${LR}"
  --lr-decay-style constant
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98
  --optimizer-cpu-offload
  --overlap-cpu-optimizer-d2h-h2d
  --use-precision-aware-optimizer
)

WANDB_ARGS=(
    # --use-wandb
    --wandb-project evolving-rl
    --wandb-group $EXP_NAME
    # --wandb-key
)

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine 2
  --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}"
)
if [[ -n "${SGLANG_ROUTER_PORT}" ]]; then
  SGLANG_ARGS+=(--sglang-router-port "${SGLANG_ROUTER_PORT}")
fi
if [[ "${DEBUG_ROLLOUT_ONLY}" == "1" ]]; then
  SGLANG_ARGS+=(
    --debug-rollout-only
    --rollout-num-gpus "${NUM_GPUS}"
  )
fi

MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
)
if [[ -n "${SAVE_DEBUG_ROLLOUT_DATA}" ]]; then
  MISC_ARGS+=(--save-debug-rollout-data "${SAVE_DEBUG_ROLLOUT_DATA}")
fi
if [[ "${USE_CHECKPOINT_OPT_PARAM_SCHEDULER}" == "1" ]]; then
  MISC_ARGS+=(--use-checkpoint-opt-param-scheduler)
fi
if [[ "${OVERRIDE_OPT_PARAM_SCHEDULER}" == "1" ]]; then
  MISC_ARGS+=(--override-opt-param-scheduler)
fi
if [[ "${NO_LOAD_OPTIM}" == "1" ]]; then
  MISC_ARGS+=(--no-load-optim)
fi

export MASTER_ADDR="${MASTER_ADDR}"
# Ray workers inherit the head process environment. Clear download proxies
# before starting/reusing Ray so node-local SGLang weight updates never leak
# through an HTTP proxy (the previous .26 run failed for exactly this reason).
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export NO_PROXY="localhost,127.0.0.1,::1,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16"
export no_proxy="${NO_PROXY}"
if [[ "${RAY_REUSE_EXISTING}" == "1" ]] && ray status >/dev/null 2>&1; then
  echo "[web/train] reusing existing Ray cluster at ${RAY_DASHBOARD_ADDRESS}"
else
  ray start \
    --head \
    --node-ip-address "${MASTER_ADDR}" \
    --num-gpus "${NUM_GPUS}" \
    --disable-usage-stats \
    --dashboard-host=0.0.0.0 \
    --dashboard-port="${RAY_DASHBOARD_PORT}"
fi

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${PYTHON_SITE_PACKAGES}:${ROOT_DIR}:${SLIME_DIR}:${MEGATRON_DIR}\",
    \"PATH\": \"$(dirname "$(command -v "${PYTHON_BIN}")"):${PATH}\",
    \"VIRTUAL_ENV\": \"${VIRTUAL_ENV:-}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK\": \"1\",
    \"SLIME_SGLANG_BASE_PORT\": \"${SGLANG_BASE_PORT}\",
    \"SGLANG_PROMETHEUS_PORT\": \"${SGLANG_PROMETHEUS_PORT:-19000}\",
    \"SLIME_PLACEMENT_NODE_IP\": \"${SLIME_PLACEMENT_NODE_IP:-}\",
    \"DYNAMICS_METRICS_PATH\": \"${DYNAMICS_METRICS_PATH:-}\",
    \"DYNAMICS_DIR\": \"${DYNAMICS_DIR:-}\",
    \"DYNAMICS_RAW_ROLLOUT_INTERVAL\": \"${DYNAMICS_RAW_ROLLOUT_INTERVAL:-1}\",
    \"DYNAMICS_BUFFER_SNAPSHOT_INTERVAL\": \"${DYNAMICS_BUFFER_SNAPSHOT_INTERVAL:-25}\",
    \"DELTA_GENERATOR_PROMPT_VARIANT\": \"${DELTA_GENERATOR_PROMPT_VARIANT:-card_general_v1}\",
    \"WEBSHOP_STATIC_SKILL_BANK\": \"${WEBSHOP_STATIC_SKILL_BANK:-}\",
    \"WEBSHOP_STATIC_GENERAL_TOP_K\": \"${WEBSHOP_STATIC_GENERAL_TOP_K:-6}\",
    \"WEBSHOP_STATIC_TASK_TOP_K\": \"${WEBSHOP_STATIC_TASK_TOP_K:-6}\",
    \"WEBSHOP_STATIC_MISTAKES_TOP_K\": \"${WEBSHOP_STATIC_MISTAKES_TOP_K:-5}\",
    \"WEBSHOP_STATIC_EVAL_STRICT\": \"${WEBSHOP_STATIC_EVAL_STRICT:-0}\",
    \"WEBSHOP_STATIC_EVAL_OUTPUT_DIR\": \"${WEBSHOP_STATIC_EVAL_OUTPUT_DIR:-}\",
    \"WEBSHOP_MEMORY_BANK\": \"${WEBSHOP_MEMORY_BANK:-}\",
    \"WEBSHOP_MEMORY_EVAL_OUTPUT_DIR\": \"${WEBSHOP_MEMORY_EVAL_OUTPUT_DIR:-}\",
    \"WEBSHOP_BASE_TRIALS_JSONL\": \"${WEBSHOP_BASE_TRIALS_JSONL:-}\",
    \"WEBSHOP_STATIC_PROBE_ANCHORS\": \"${WEBSHOP_STATIC_PROBE_ANCHORS:-}\",
    \"http_proxy\": \"\",
    \"https_proxy\": \"\",
    \"HTTP_PROXY\": \"\",
    \"HTTPS_PROXY\": \"\",
    \"ALL_PROXY\": \"\",
    \"all_proxy\": \"\",
    \"NO_PROXY\": \"${NO_PROXY}\",
    \"no_proxy\": \"${NO_PROXY}\"
  }
}"


if [[ "${WEBSHOP_REUSE_SERVERS}" != "1" ]]; then
  # Put service processes in independent sessions.  `nohup` alone is not enough
  # when a no-wait launcher exits under a PTY/process-group supervisor.
  # The Ray runtime exposes Python 3.12 packages through PYTHONPATH, while the
  # pinned WebShop environment is Python 3.10.  Do not leak the actor
  # environment into the Flask/WebShop subprocesses.
  nohup setsid env -u PYTHONPATH "${WEBSHOP_PYTHON}" "${ROOT_DIR}/env/webshop/launch_servers.py" \
    --vendor-root "${WEBSHOP_VENDOR_ROOT}" \
    --file-path "${WEBSHOP_ITEMS}" \
    --attr-path "${WEBSHOP_ATTRS}" \
    --num-workers "${WEBSHOP_NUM_WORKERS}" \
    --router-port "${WEBSHOP_ROUTER_PORT}" \
    --worker-host "${WEBSHOP_WORKER_HOST}" \
    --worker-start-port "${WEBSHOP_WORKER_START_PORT}" \
    --router-host "${WEBSHOP_ROUTER_HOST}" \
    --python "${WEBSHOP_PYTHON}" \
    --log-dir "${LOG_DIR}/env" \
    --pid-file "${WEBSHOP_PID_FILE}" \
    --startup-timeout "${WEBSHOP_STARTUP_TIMEOUT}" \
    --worker-script "${ROOT_DIR}/env/webshop/worker_server.py" \
    --router-script "${ROOT_DIR}/env/webshop/router_server.py" \
    > "${LOG_DIR}/env_launcher.log" 2>&1 < /dev/null &

  if [[ "${NUM_ROLLOUT}" != "0" ]]; then
    nohup setsid "${PYTHON_BIN}" "${ROOT_DIR}/env/webshop/retrieve_server/retrieve_server.py" \
      --parquet "${ALL_DATA}" \
      --cache_file "${CACHE_FILE}" \
      --port "${WEBSHOP_RETRIEVE_TEST_PORT}" \
      --host "${WEBSHOP_RETRIEVE_HOST}" \
      --question_col "${WEBSHOP_RETRIEVE_QUESTION_COL}" > "${LOG_DIR}/retrieve_test.log" 2>&1 < /dev/null &
    WEBSHOP_RETRIEVE_TEST_PID=$!
    printf '%s\n' "${WEBSHOP_RETRIEVE_TEST_PID}" > "${WEBSHOP_RETRIEVE_PID_FILE}"
  fi
fi

wait_for_health "${WEBSHOP_ENV_URL}/health" "${WEBSHOP_STARTUP_TIMEOUT}"
if [[ "${NUM_ROLLOUT}" != "0" ]]; then
  wait_for_health "${WEBSHOP_RETRIEVE_URL_TEST}/health" "${WEBSHOP_RETRIEVE_STARTUP_TIMEOUT}"
fi


RAY_JOB_ARGS=(
  --address="${RAY_DASHBOARD_ADDRESS}"
  --runtime-env-json="${RUNTIME_ENV_JSON}"
)
if [[ "${RAY_NO_WAIT}" == "1" ]]; then
  RAY_JOB_ARGS+=(--no-wait)
fi
if [[ -n "${RAY_SUBMISSION_ID}" ]]; then
  RAY_JOB_ARGS+=(--submission-id="${RAY_SUBMISSION_ID}")
fi

ray job submit "${RAY_JOB_ARGS[@]}" \
  -- "${PYTHON_BIN}" "${SLIME_TRAIN_SCRIPT}" \
  --actor-num-gpus-per-node "${NUM_GPUS}" \
  --colocate \
  "${MODEL_ARGS[@]}" \
  "${CKPT_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${GRPO_ARGS[@]}" \
  "${WANDB_ARGS[@]}" \
  "${PERF_ARGS[@]}" \
  "${EVAL_ARGS[@]}" \
  "${SGLANG_ARGS[@]}" \
  "${CUSTOM_ARGS[@]}" \
  "${MISC_ARGS[@]}"
