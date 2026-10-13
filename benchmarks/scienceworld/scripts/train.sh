#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${ROOT_DIR}/../.." >/dev/null 2>&1 && pwd)"
if [[ -f "${REPO_ROOT}/.env" ]]; then
  set -a
  source "${REPO_ROOT}/.env"
  set +a
fi
PYTHON_BIN="${PYTHON_BIN:-python3}"
SCIENCEWORLD_PYTHON="${SCIENCEWORLD_PYTHON:-python3}"
MEGATRON_DIR="${MEGATRON_DIR:?Set MEGATRON_DIR to your Megatron-LM checkout}"
SLIME_DIR="${SLIME_DIR:-${REPO_ROOT}/framework/slime_legacy}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${ROOT_DIR}:${SLIME_DIR}:${PYTHONPATH:-}"

MODEL_SOURCE_ARGS="${MODEL_SOURCE_ARGS:-${SLIME_DIR}/scripts/models/qwen3-8B.sh}"
HF_CHECKPOINT="${HF_CHECKPOINT:?Set HF_CHECKPOINT to the Hugging Face Qwen3-8B checkpoint}"
REF_CHECKPOINT="${REF_CHECKPOINT:?Set REF_CHECKPOINT to the Megatron-format reference checkpoint}"
EXP_NAME="${EXP_NAME:-scienceworld-delta-card-qwen3-8b}"
SAVE_CHECKPOINT="${SAVE_CHECKPOINT:-${REPO_ROOT}/outputs/checkpoints/scienceworld/${EXP_NAME}}"
LOAD_CHECKPOINT="${LOAD_CHECKPOINT:-${REF_CHECKPOINT}}"
SCIENCEWORLD_E7_AUDIT_DIR="${SCIENCEWORLD_E7_AUDIT_DIR:-${REPO_ROOT}/outputs/audit/scienceworld/${EXP_NAME}}"
export SCIENCEWORLD_E7_AUDIT_DIR
TRAIN_DATA="${SCIENCEWORLD_TRAIN_DATA:-${REPO_ROOT}/data/scienceworld/train.parquet}"
EVAL_ID_DATA="${SCIENCEWORLD_EVAL_ID_DATA:-${EVAL_ID_DATA:-}}"
EVAL_OOD_DATA="${SCIENCEWORLD_EVAL_OOD_DATA:-${EVAL_OOD_DATA:-}}"
N_SAMPLES_PER_EVAL_PROMPT="${N_SAMPLES_PER_EVAL_PROMPT:-1}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0.0}"
EVAL_TOP_P="${EVAL_TOP_P:-1.0}"
SCIENCEWORLD_CONFIG="${SCIENCEWORLD_CONFIG:-${ROOT_DIR}/configs/server.yaml}"

NUM_GPUS="${NUM_GPUS:-8}"
NUM_ROLLOUT="${NUM_ROLLOUT:-161}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-16}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-30}"
EXECUTOR_MAX_RESPONSE_LEN="${EXECUTOR_MAX_RESPONSE_LEN:-384}"
ROLLOUT_MAX_CONTEXT_LEN="${ROLLOUT_MAX_CONTEXT_LEN:-32768}"
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-12288}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-8192}"
LOG_PROBS_CHUNK_SIZE="${LOG_PROBS_CHUNK_SIZE:-4096}"
SAVE_INTERVAL="${SAVE_INTERVAL:-20}"
SEED="${SEED:-1242}"
ROLLOUT_SEED="${ROLLOUT_SEED:-42}"
GENERATOR_LOSS_WEIGHT="${GENERATOR_LOSS_WEIGHT:-${GENERATOR_REWARD_WEIGHT:-0.1}}"
EPS_CLIP="${EPS_CLIP:-0.2}"
EPS_CLIP_HIGH="${EPS_CLIP_HIGH:-0.28}"
DEBUG_ROLLOUT_ONLY="${DEBUG_ROLLOUT_ONLY:-0}"
RAY_DASHBOARD_ADDRESS="${RAY_DASHBOARD_ADDRESS:?RAY_DASHBOARD_ADDRESS is required}"
RAY_SUBMISSION_ID="${RAY_SUBMISSION_ID:-}"
SGLANG_BASE_PORT="${SGLANG_BASE_PORT:-31000}"
SGLANG_ROUTER_PORT="${SGLANG_ROUTER_PORT:-31500}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.55}"
CLEAN_ORPHAN_NCCL_SHM="${CLEAN_ORPHAN_NCCL_SHM:-0}"

_cfg() { "${PYTHON_BIN}" -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))[sys.argv[2]])" "$1" "$2"; }
ENV_URL="$(_cfg "${SCIENCEWORLD_CONFIG}" scienceworld_env_url)"
ROUTER_PORT="$(_cfg "${SCIENCEWORLD_CONFIG}" scienceworld_router_port)"
ROUTER_HOST="$(_cfg "${SCIENCEWORLD_CONFIG}" scienceworld_router_host)"
WORKER_HOST="$(_cfg "${SCIENCEWORLD_CONFIG}" scienceworld_worker_host)"
WORKER_START_PORT="$(_cfg "${SCIENCEWORLD_CONFIG}" scienceworld_worker_start_port)"
NUM_WORKERS="$(_cfg "${SCIENCEWORLD_CONFIG}" scienceworld_num_workers)"
SIMPLIFICATION="$(_cfg "${SCIENCEWORLD_CONFIG}" scienceworld_simplification)"
PID_FILE="$(_cfg "${SCIENCEWORLD_CONFIG}" scienceworld_pid_file)"
LOG_DIR="$(_cfg "${SCIENCEWORLD_CONFIG}" scienceworld_log_dir)"
if [[ "${PID_FILE}" != /* ]]; then PID_FILE="${ROOT_DIR}/${PID_FILE}"; fi
if [[ "${LOG_DIR}" != /* ]]; then LOG_DIR="${ROOT_DIR}/${LOG_DIR}"; fi
STARTUP_TIMEOUT="$(_cfg "${SCIENCEWORLD_CONFIG}" scienceworld_startup_timeout)"

test -s "${HF_CHECKPOINT}/config.json"
test -s "${REF_CHECKPOINT}/latest_checkpointed_iteration.txt"
test -s "${TRAIN_DATA}"
mkdir -p "${SAVE_CHECKPOINT}" "${SCIENCEWORLD_E7_AUDIT_DIR}" "${LOG_DIR}" "$(dirname "${PID_FILE}")"

if [[ "${CLEAN_ORPHAN_NCCL_SHM}" == "1" ]]; then
  "${PYTHON_BIN}" "${ROOT_DIR}/scripts/cleanup_orphan_nccl_shm.py"
fi

wait_for_health() {
  local url="$1" timeout="$2"
  "${PYTHON_BIN}" - "$url" "$timeout" <<'PY'
import sys, time
from urllib.request import urlopen
url, timeout = sys.argv[1], float(sys.argv[2])
deadline = time.time() + timeout
last = None
while time.time() < deadline:
    try:
        with urlopen(url, timeout=5) as response:
            if 200 <= response.status < 300:
                raise SystemExit(0)
    except Exception as exc:
        last = exc
    time.sleep(1)
raise SystemExit(f"health timeout for {url}: {last}")
PY
}

cleanup_services() {
  "${PYTHON_BIN}" "${ROOT_DIR}/env/scienceworld/stop_servers.py" --pid-file "${PID_FILE}" || true
}
trap cleanup_services EXIT
cleanup_services

"${PYTHON_BIN}" - "${ROUTER_PORT}" "${WORKER_START_PORT}" "${NUM_WORKERS}" \
  "${SGLANG_BASE_PORT}" "${SGLANG_ROUTER_PORT}" <<'PY'
import socket
import sys

router_port, worker_start, num_workers, sglang_base, sglang_router = map(int, sys.argv[1:])
ports = [router_port, *range(worker_start, worker_start + num_workers), sglang_base, sglang_router]
busy = []
for port in ports:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("0.0.0.0", port))
    except OSError as exc:
        busy.append((port, str(exc)))
    finally:
        sock.close()
if busy:
    raise SystemExit(f"refusing occupied ScienceWorld/SGLang ports: {busy}")
print(f"port preflight passed: {ports}", flush=True)
PY

export JAVA_HOME="${JAVA_HOME:?Set JAVA_HOME to a Java 11 installation}"
export JAVA_TOOL_OPTIONS="${JAVA_TOOL_OPTIONS:--XX:+UseSerialGC -XX:-TieredCompilation -XX:CICompilerCount=1 -XX:ActiveProcessorCount=1 -Xss256k}"
SCIENCEWORLD_SITE_PACKAGES="${SCIENCEWORLD_SITE_PACKAGES:-$(${SCIENCEWORLD_PYTHON} -c 'import site; print(site.getsitepackages()[0])')}"

nohup setsid env \
  PYTHONPATH="${ROOT_DIR}:${SCIENCEWORLD_SITE_PACKAGES}" \
  JAVA_HOME="${JAVA_HOME}" \
  JAVA_TOOL_OPTIONS="${JAVA_TOOL_OPTIONS}" \
  PATH="${JAVA_HOME}/bin:${PATH}" \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "${SCIENCEWORLD_PYTHON}" "${ROOT_DIR}/env/scienceworld/launch_servers.py" \
  --num-workers "${NUM_WORKERS}" \
  --router-host "${ROUTER_HOST}" \
  --router-port "${ROUTER_PORT}" \
  --worker-host "${WORKER_HOST}" \
  --worker-start-port "${WORKER_START_PORT}" \
  --python "${SCIENCEWORLD_PYTHON}" \
  --worker-script "${ROOT_DIR}/env/scienceworld/worker_server.py" \
  --router-script "${ROOT_DIR}/env/scienceworld/router_server.py" \
  --simplification "${SIMPLIFICATION}" \
  --default-max-steps "${MAX_EPISODE_STEPS}" \
  --log-dir "${LOG_DIR}/env" \
  --pid-file "${PID_FILE}" \
  --startup-timeout "${STARTUP_TIMEOUT}" \
  > "${LOG_DIR}/env_launcher.log" 2>&1 < /dev/null &
wait_for_health "${ENV_URL}/health" "${STARTUP_TIMEOUT}"

source "${MODEL_SOURCE_ARGS}"
NVLINK_COUNT="$( { nvidia-smi topo -m 2>/dev/null || true; } | awk '{count += gsub(/NV[0-9]+/, "")} END {print count + 0}')"
if [[ "${NVLINK_COUNT}" -gt 0 ]]; then HAS_NVLINK=1; else HAS_NVLINK=0; fi

ROLLOUT_ARGS=(
  --prompt-data "${TRAIN_DATA}"
  --input-key prompt
  --label-key metadata
  --apply-chat-template
  --apply-chat-template-kwargs '{"enable_thinking":false}'
  --rollout-shuffle
  --n-experiences 4
  --max-episode-steps "${MAX_EPISODE_STEPS}"
  --executor-max-response-len "${EXECUTOR_MAX_RESPONSE_LEN}"
  --generator-reward-weight "${GENERATOR_LOSS_WEIGHT}"
  --executor-reward-weight 1.0
  --executor-temperature 1.0
  --executor-top-p 0.85
  --skill-format-penalty 1
  --num-rollout "${NUM_ROLLOUT}"
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
  --n-samples-per-prompt 1
  --seed "${SEED}"
  --rollout-seed "${ROLLOUT_SEED}"
  --rollout-max-context-len "${ROLLOUT_MAX_CONTEXT_LEN}"
  --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
  --rollout-top-p 0.85
  --rollout-temperature 1.0
  --global-batch-size "${GLOBAL_BATCH_SIZE}"
  --balance-data
)

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
  --log-probs-chunk-size "${LOG_PROBS_CHUNK_SIZE}"
)

GRPO_ARGS=(
  --advantage-estimator grpo
  --no-normalize-advantages
  --train-env-vars '{"NCCL_SHM_DISABLE":"1","NCCL_CUMEM_ENABLE":"1","NCCL_CUMEM_HOST_ENABLE":"1"}'
  --executor-entropy-coef 0.0
  --generator-entropy-coef 0.002
  --entropy-coef 0.0
  --eps-clip "${EPS_CLIP}"
  --eps-clip-high "${EPS_CLIP_HIGH}"
)

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr 1e-6
  --lr-decay-style constant
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98
  --use-precision-aware-optimizer
)

SGLANG_ARGS=(
  --sglang-disable-cuda-graph
  --rollout-num-gpus-per-engine 2
  --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}"
  --sglang-router-port "${SGLANG_ROUTER_PORT}"
)
if [[ "${DEBUG_ROLLOUT_ONLY}" == "1" ]]; then
  SGLANG_ARGS+=(--debug-rollout-only --rollout-num-gpus "${NUM_GPUS}")
fi

MISC_ARGS=(
  --use-wandb
  --wandb-mode "${WANDB_MODE:-online}"
  --wandb-project "${WANDB_PROJECT:-skillscaffold}"
  --wandb-group "${EXP_NAME}"
  --eval-interval "${EVAL_INTERVAL:-20}"
  --n-samples-per-eval-prompt "${N_SAMPLES_PER_EVAL_PROMPT}"
  --eval-temperature "${EVAL_TEMPERATURE}"
  --eval-top-p "${EVAL_TOP_P}"
  --eval-concurrency 16
  --eval-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
)
if [[ "${DISABLE_WANDB_RANDOM_SUFFIX:-0}" == "1" ]]; then
  MISC_ARGS+=(--disable-wandb-random-suffix)
fi
if [[ -n "${EVAL_ID_DATA}" && -n "${EVAL_OOD_DATA}" ]]; then
  test -s "${EVAL_ID_DATA}"
  test -s "${EVAL_OOD_DATA}"
  MISC_ARGS+=(--eval-prompt-data test_id "${EVAL_ID_DATA}" test_ood "${EVAL_OOD_DATA}")
elif [[ -n "${EVAL_ID_DATA}" ]]; then
  test -s "${EVAL_ID_DATA}"
  MISC_ARGS+=(--eval-prompt-data test_id "${EVAL_ID_DATA}")
elif [[ -n "${EVAL_OOD_DATA}" ]]; then
  test -s "${EVAL_OOD_DATA}"
  MISC_ARGS+=(--eval-prompt-data test_ood "${EVAL_OOD_DATA}")
else
  MISC_ARGS+=(--eval-prompt-data validation "${EVAL_DATA:?EVAL_DATA required}")
fi
if [[ "${NUM_ROLLOUT}" == "0" ]]; then
  MISC_ARGS+=(--no-load-optim --no-load-rng)
fi
if [[ -n "${SAVE_DEBUG_ROLLOUT_DATA:-}" ]]; then
  MISC_ARGS+=(--save-debug-rollout-data "${SAVE_DEBUG_ROLLOUT_DATA}")
fi

RUNTIME_ENV_JSON="$(${PYTHON_BIN} - <<PY
import json, os
env = {
  "PYTHONPATH": "${ROOT_DIR}:${SLIME_DIR}:${MEGATRON_DIR}",
  "PATH": os.environ["PATH"],
  "CUDA_DEVICE_MAX_CONNECTIONS": "1",
  "NCCL_NVLS_ENABLE": "${HAS_NVLINK}",
  "NCCL_SHM_DISABLE": "1",
  "NCCL_CUMEM_ENABLE": "1",
  "NCCL_CUMEM_HOST_ENABLE": "1",
  "SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK": "1",
  "SLIME_SGLANG_BASE_PORT": "${SGLANG_BASE_PORT}",
  "SLIME_PLACEMENT_NODE_IP": "${SLIME_PLACEMENT_NODE_IP:-}",
  "DYNAMICS_METRICS_PATH": "${DYNAMICS_METRICS_PATH:-}",
  "DYNAMICS_DIR": "${DYNAMICS_DIR:-}",
  "DYNAMICS_RAW_ROLLOUT_INTERVAL": "1",
  "ALFWORLD_GROUP_DELTA_VARIANT": "positive_gate_trainext_raw",
  "ALFWORLD_DELTA_CREDIT_SCOPE": "all_model_tokens",
  "SCIENCEWORLD_GENERATOR_PROMPT_FILE": (
      os.environ.get("SCIENCEWORLD_GENERATOR_PROMPT_FILE")
      or os.environ.get("SCIENCEWORLD_GROUP_GENERATOR_PROMPT_FILE")
      or "${ROOT_DIR}/src/scienceworld/e7_prompts_v2.py"),
}
for key in ("WANDB_API_KEY", "WANDB_ENTITY", "WANDB_MODE", "WANDB_DIR", "WANDB_INIT_TIMEOUT", "SCIENCEWORLD_E7_AUDIT_DIR", "SCIENCEWORLD_EVAL_CHECKPOINT_STEP", "http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "no_proxy", "TMPDIR"):
    if key in os.environ: env[key] = os.environ[key]
print(json.dumps({"env_vars": env}))
PY
)"

JOB_ARGS=(--address="${RAY_DASHBOARD_ADDRESS}" --runtime-env-json="${RUNTIME_ENV_JSON}")
if [[ -n "${RAY_SUBMISSION_ID}" ]]; then JOB_ARGS+=(--submission-id="${RAY_SUBMISSION_ID}"); fi

ray job submit "${JOB_ARGS[@]}" -- \
  "${PYTHON_BIN}" "${ROOT_DIR}/train_entry.py" \
  --actor-num-gpus-per-node "${NUM_GPUS}" \
  --colocate \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "${HF_CHECKPOINT}" \
  --ref-load "${REF_CHECKPOINT}" \
  --load "${LOAD_CHECKPOINT}" \
  --save "${SAVE_CHECKPOINT}" \
  --save-interval "${SAVE_INTERVAL}" \
  "${ROLLOUT_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${GRPO_ARGS[@]}" \
  "${PERF_ARGS[@]}" \
  "${SGLANG_ARGS[@]}" \
  --custom-generate-function-path src.scienceworld.pipeline.scienceworld_pipeline \
  --custom-reward-post-process-path src.scienceworld.pipeline.scienceworld_reward_post_process \
  --custom-rollout-log-function-path src.webshop.loggers.rollout_logger.log_rollout_data \
  --custom-eval-rollout-log-function-path src.scienceworld.e7_metrics.log_eval \
  --custom-config-path "${SCIENCEWORLD_CONFIG}" \
  "${MISC_ARGS[@]}"
