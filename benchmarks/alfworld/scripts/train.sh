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
ALFWORLD_PYTHON="${ALFWORLD_PYTHON:-python3}"
SLIME_DIR="${SLIME_DIR:-${REPO_ROOT}/framework/slime_legacy}"
MEGATRON_DIR="${MEGATRON_DIR:?Set MEGATRON_DIR to your Megatron-LM checkout}"
METHOD="${ALFWORLD_METHOD:?ALFWORLD_METHOD is required}"
EVAL_PROTOCOL="${ALFWORLD_EVAL_PROTOCOL:-matched_no_skill}"
CREDIT_SCOPE="${ALFWORLD_DELTA_CREDIT_SCOPE:-all_model_tokens}"
EXP_NAME="${EXP_NAME:?EXP_NAME is required}"
SERVICE_CONFIG="${ALFWORLD_SERVER_CONFIG:-${ROOT_DIR}/configs/server.yaml}"
ALF_CONFIG="${ALFWORLD_CONFIG:?ALFWORLD_CONFIG is required}"
DATA_DIR="${ALFWORLD_PARQUET_DIR:?ALFWORLD_PARQUET_DIR is required}"
DATA_ROOT="${ALFWORLD_DATA_ROOT:?ALFWORLD_DATA_ROOT is required}"
CACHE_FILE="${ALFWORLD_CACHE_FILE:?ALFWORLD_CACHE_FILE is required}"
SAVE_CHECKPOINT="${SAVE_CHECKPOINT:-${REPO_ROOT}/outputs/checkpoints/alfworld/${EXP_NAME:-skillscaffold-alfworld}}"
if [[ -n "${SAVE_DEBUG_ROLLOUT_DATA:-}" ]]; then
  DEBUG_ROLLOUT_PATH="${SAVE_DEBUG_ROLLOUT_DATA}"
else
  DEBUG_ROLLOUT_PATH="${SAVE_CHECKPOINT}/rollout/raw_{rollout_id}.pt"
fi
REF_CHECKPOINT="${REF_CHECKPOINT:?Set REF_CHECKPOINT to the Megatron-format reference checkpoint}"
LOAD_CHECKPOINT="${LOAD_CHECKPOINT:-${REF_CHECKPOINT}}"
HF_CHECKPOINT="${HF_CHECKPOINT:?Set HF_CHECKPOINT to the Hugging Face Qwen3-8B checkpoint}"
MODEL_SOURCE_ARGS="${MODEL_SOURCE_ARGS:-${SLIME_DIR}/scripts/models/qwen3-8B.sh}"
TRAIN_DATA="${ALFWORLD_TRAIN_DATA:-${DATA_DIR}/train.parquet}"
LOG_DIR="${RUN_LOG_ROOT:-${ROOT_DIR}/runs}/${EXP_NAME}"

if [[ "${VALIDATE_TRAIN_DATA:-1}" == "1" && "${NUM_ROLLOUT:-100}" != "0" ]]; then
  "${PYTHON_BIN}" - "${TRAIN_DATA}" <<'PY'
import sys
import pandas as pd

path = sys.argv[1]
frame = pd.read_parquet(path, columns=["metadata"])
if len(frame) != 3553:
    raise SystemExit(f"expected 3553 ALFWorld training rows, found {len(frame)}: {path}")
expected = {
    "pick_and_place_simple": 790,
    "pick_clean_then_place_in_recep": 650,
    "pick_cool_then_place_in_recep": 533,
    "pick_heat_then_place_in_recep": 459,
    "pick_two_obj_and_place": 813,
    "look_at_obj_in_light": 308,
}
counts = frame["metadata"].map(lambda item: item["task_type"]).value_counts().to_dict()
if counts != expected:
    raise SystemExit(f"unexpected ALFWorld task counts: {counts}; expected {expected}")
PY
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="/usr/local/lib/python3.12/dist-packages:${ROOT_DIR}:${SLIME_DIR}:${MEGATRON_DIR}"
export ALFWORLD_METHOD ALFWORLD_DATA_ROOT ALFWORLD_EVAL_PROTOCOL="${EVAL_PROTOCOL}"
export ALFWORLD_DELTA_CREDIT_SCOPE="${CREDIT_SCOPE}"
export ALFWORLD_GROUP_DELTA_VARIANT="${ALFWORLD_GROUP_DELTA_VARIANT:-positive_gate_trainext_raw}"
mkdir -p "${LOG_DIR}" "${SAVE_CHECKPOINT}"

cfg() {
  "${PYTHON_BIN}" -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))[sys.argv[2]])' "$1" "$2"
}
ENV_URL="$(cfg "${SERVICE_CONFIG}" alfworld_env_url)"
RETRIEVE_URL="$(cfg "${SERVICE_CONFIG}" alfworld_retrieve_url_train)"
RETRIEVE_TEST_URL="$(cfg "${SERVICE_CONFIG}" alfworld_retrieve_url_test)"
ROUTER_PORT="$(cfg "${SERVICE_CONFIG}" alfworld_router_port)"
WORKER_HOST="$(cfg "${SERVICE_CONFIG}" alfworld_worker_host)"
WORKER_START_PORT="$(cfg "${SERVICE_CONFIG}" alfworld_worker_start_port)"
ROUTER_HOST="$(cfg "${SERVICE_CONFIG}" alfworld_router_host)"
NUM_WORKERS="$(cfg "${SERVICE_CONFIG}" alfworld_num_workers)"
RETRIEVE_HOST="$(cfg "${SERVICE_CONFIG}" alfworld_retrieve_host)"
RETRIEVE_PORT="$(cfg "${SERVICE_CONFIG}" alfworld_retrieve_train_port)"
RETRIEVE_TEST_PORT="$(cfg "${SERVICE_CONFIG}" alfworld_retrieve_test_port)"
PID_FILE="$(cfg "${SERVICE_CONFIG}" alfworld_pid_file)"
if [[ "${PID_FILE}" != /* ]]; then PID_FILE="${ROOT_DIR}/${PID_FILE}"; fi

wait_health() {
  local url="$1" timeout="${2:-300}"
  "${PYTHON_BIN}" - "$url" "$timeout" <<'PY'
import sys,time
from urllib.request import urlopen
url,timeout=sys.argv[1],float(sys.argv[2]); deadline=time.time()+timeout; error=None
while time.time()<deadline:
    try:
        with urlopen(url,timeout=5) as response:
            if 200 <= response.status < 300: raise SystemExit(0)
    except SystemExit: raise
    except Exception as exc: error=exc
    time.sleep(1)
raise SystemExit(f"health timeout for {url}: {error}")
PY
}

ALF_LAUNCH_PID=""
RETRIEVE_PID=""
RETRIEVE_TEST_PID=""
cleanup() {
  if [[ -n "${RETRIEVE_TEST_PID}" ]]; then kill -TERM "${RETRIEVE_TEST_PID}" 2>/dev/null || true; fi
  if [[ -n "${RETRIEVE_PID}" ]]; then kill -TERM "${RETRIEVE_PID}" 2>/dev/null || true; fi
  if [[ -f "${PID_FILE}" ]]; then
    "${PYTHON_BIN}" - "${PID_FILE}" <<'PY' || true
import json, os, signal, sys
payload = json.load(open(sys.argv[1]))
pids = [payload.get("router", {}).get("pid")]
pids.extend(worker.get("pid") for worker in payload.get("workers", []))
for pid in pids:
    if isinstance(pid, int) and os.path.exists(f"/proc/{pid}"):
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
PY
  fi
  if [[ -n "${ALF_LAUNCH_PID}" ]]; then
    kill -TERM "${ALF_LAUNCH_PID}" 2>/dev/null || true
    wait "${ALF_LAUNCH_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

"${ALFWORLD_PYTHON}" "${ROOT_DIR}/env/alfworld/launch_servers.py" \
  --alf-config-path "${ALF_CONFIG}" \
  --num-workers "${NUM_WORKERS}" \
  --router-port "${ROUTER_PORT}" \
  --worker-host "${WORKER_HOST}" \
  --worker-start-port "${WORKER_START_PORT}" \
  --router-host "${ROUTER_HOST}" \
  --python "${ALFWORLD_PYTHON}" \
  --log-dir "${LOG_DIR}/env" \
  --pid-file "${PID_FILE}" \
  --startup-timeout 300 \
  --worker-script "${ROOT_DIR}/env/alfworld/worker_server.py" \
  --router-script "${ROOT_DIR}/env/alfworld/router_server.py" \
  >"${LOG_DIR}/env_launcher.log" 2>&1 &
ALF_LAUNCH_PID=$!

if [[ "${METHOD}" == "evolving" ]]; then
  "${PYTHON_BIN}" "${ROOT_DIR}/env/alfworld/retrieve_server/retrieve_server.py" \
    --parquet "${TRAIN_DATA}" \
    --cache_file "${CACHE_FILE}" \
    --port "${RETRIEVE_PORT}" \
    --host "${RETRIEVE_HOST}" \
    --question_col question >"${LOG_DIR}/retrieve_train.log" 2>&1 &
  RETRIEVE_PID=$!
fi

# Native Evolving-RL evaluation retrieves source tasks from the complete
# training corpus.  Keep this service separate from the train-time
# train_seen-only retriever, matching the original ALFWorld recipe.
if [[ "${METHOD}" == "evolving" && "${EVAL_PROTOCOL}" == "native_skill" ]]; then
  "${PYTHON_BIN}" "${ROOT_DIR}/env/alfworld/retrieve_server/retrieve_server.py" \
    --parquet "${DATA_DIR}/train.parquet" \
    --cache_file "${CACHE_FILE}" \
    --port "${RETRIEVE_TEST_PORT}" \
    --host "${RETRIEVE_HOST}" \
    --question_col question >"${LOG_DIR}/retrieve_test.log" 2>&1 &
  RETRIEVE_TEST_PID=$!
fi

wait_health "${ENV_URL}/health" 300
if [[ "${METHOD}" == "evolving" ]]; then wait_health "${RETRIEVE_URL}/health" 300; fi
if [[ "${METHOD}" == "evolving" && "${EVAL_PROTOCOL}" == "native_skill" ]]; then
  wait_health "${RETRIEVE_TEST_URL}/health" 300
fi

source "${MODEL_SOURCE_ARGS}"

ROLLOUT_ARGS=(
  --prompt-data "${TRAIN_DATA}"
  --input-key prompt --label-key metadata --apply-chat-template --rollout-shuffle
  --apply-chat-template-kwargs '{"enable_thinking":false}'
  --num-repeat 1 --n-experiences "${N_EXPERIENCES:-4}" --retrieval-topk 2 --max-episode-steps "${MAX_EPISODE_STEPS:-50}"
  --executor-max-response-len "${EXECUTOR_MAX_RESPONSE_LEN:-512}" --generator-reward-weight "${GENERATOR_LOSS_WEIGHT:-${GENERATOR_REWARD_WEIGHT:-0.1}}" --executor-reward-weight 1.0
  --executor-temperature 1.0 --skill-format-penalty 1
  --num-rollout "${NUM_ROLLOUT:-100}" --rollout-batch-size "${ROLLOUT_BATCH_SIZE:-16}" --n-samples-per-prompt 1
  --seed "${SEED:-1242}" --rollout-seed "${ROLLOUT_SEED:-42}"
  --rollout-max-context-len "${ROLLOUT_MAX_CONTEXT_LEN:-32768}" --rollout-max-response-len 1536
  --rollout-temperature 1.0 --rollout-top-p 0.85 --executor-top-p 0.85
  --global-batch-size "${GLOBAL_BATCH_SIZE:-128}" --balance-data
)

if [[ -n "${ALFWORLD_EVAL_PROMPT_DATA:-}" ]]; then
  EVAL_DATA_ARGS=(probe "${ALFWORLD_EVAL_PROMPT_DATA}")
elif [[ "${ALFWORLD_EVAL_SPLIT:-all}" == "valid_unseen" ]]; then
  EVAL_DATA_ARGS=(
    unseen_id "${DATA_DIR}/valid_unseen_id_types.parquet"
    unseen_heldout "${DATA_DIR}/valid_unseen_heldout_types.parquet"
  )
else
  EVAL_DATA_ARGS=(
    seen_id "${DATA_DIR}/valid_seen_id_types.parquet"
    seen_heldout "${DATA_DIR}/valid_seen_heldout_types.parquet"
    unseen_id "${DATA_DIR}/valid_unseen_id_types.parquet"
    unseen_heldout "${DATA_DIR}/valid_unseen_heldout_types.parquet"
  )
fi
EVAL_ARGS=(
  --skip-eval-before-train
  --eval-prompt-data "${EVAL_DATA_ARGS[@]}"
  --eval-interval 25 --n-samples-per-eval-prompt "${N_SAMPLES_PER_EVAL_PROMPT:-1}"
  --eval-max-response-len "${EVAL_MAX_RESPONSE_LEN:-1536}"
  --eval-temperature "${EVAL_TEMPERATURE:-0.0}" --eval-top-p "${EVAL_TOP_P:-1.0}"
  --eval-concurrency "${EVAL_CONCURRENCY:-128}"
)

CKPT_ARGS=(
  --hf-checkpoint "${HF_CHECKPOINT}" --ref-load "${REF_CHECKPOINT}"
  --load "${LOAD_CHECKPOINT}" --save "${SAVE_CHECKPOINT}" --save-interval "${SAVE_INTERVAL:-5}"
)
if [[ "${EVAL_ONLY_LOAD:-0}" == "1" ]]; then
  # Evaluation must load model weights only. Restoring the training optimizer
  # scheduler compares the eval-only horizon against the training token horizon.
  CKPT_ARGS+=(--no-load-optim --use-checkpoint-opt-param-scheduler)
fi

CUSTOM_ARGS=(
  --custom-generate-function-path src.alfworld.pipeline.alfworld_three_method_pipeline
  --custom-reward-post-process-path src.alfworld.pipeline.alfworld_reward_post_process
  --custom-rollout-log-function-path src.alfworld.loggers.rollout_logger.log_rollout_data
  --custom-eval-rollout-log-function-path src.alfworld.loggers.eval_logger.log_eval_by_task_type
  --custom-config-path "${SERVICE_CONFIG}"
)

PERF_ARGS=(
  --tensor-model-parallel-size "${TENSOR_MODEL_PARALLEL_SIZE:-2}" --sequence-parallel
  --pipeline-model-parallel-size 1 --context-parallel-size "${CONTEXT_PARALLEL_SIZE:-1}"
  --expert-model-parallel-size 1 --expert-tensor-parallel-size 1
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
  --use-dynamic-batch-size --use-dynamic-global-batch-size
  --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-8192}"
)
if [[ -n "${LOG_PROBS_CHUNK_SIZE:-}" ]]; then
  PERF_ARGS+=(--log-probs-chunk-size "${LOG_PROBS_CHUNK_SIZE}")
fi

OPTIMIZER_ARGS=(
  --optimizer adam --lr "${LR:-1e-6}" --lr-decay-style constant --weight-decay 0.1
  --adam-beta1 0.9 --adam-beta2 0.98 --use-precision-aware-optimizer
)
if [[ "${OPTIMIZER_CPU_OFFLOAD:-0}" == "1" ]]; then
  OPTIMIZER_ARGS+=(--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d)
fi

if [[ -n "${GRPO_PROFILE:-}" ]]; then
  GRPO_PROFILE="${GRPO_PROFILE}"
elif [[ "${METHOD}" == "group_delta" ]]; then
  GRPO_PROFILE="delta"
else
  GRPO_PROFILE="${METHOD}"
fi
case "${GRPO_PROFILE}" in
  vanilla)
    # Established plain GRPO baseline used by the prior WebShop control.
    GRPO_ARGS=(
      --advantage-estimator grpo --no-normalize-advantages --use-kl-loss
      --executor-entropy-coef 0.0 --generator-entropy-coef 0.0
      --executor-kl-loss-coef 0.01 --generator-kl-loss-coef 0.01
      --kl-loss-coef 0.01 --kl-loss-type low_var_kl --entropy-coef 0.0
      --eps-clip 0.1 --eps-clip-high 0.15
    )
    ;;
  evolving)
    # Use the role-specific regularization for this training variant.
    GRPO_ARGS=(
      --advantage-estimator grpo --no-normalize-advantages --use-kl-loss
      --executor-entropy-coef 0.0 --generator-entropy-coef -0.03
      --executor-kl-loss-coef 0.01 --generator-kl-loss-coef 0.01
      --kl-loss-coef 0.01 --kl-loss-type low_var_kl --entropy-coef 0.0
      --eps-clip 0.1 --eps-clip-high 0.15
    )
    ;;
  delta)
    # Delta-card ablation: no KL and asymmetric higher PPO clipping.
    GRPO_ARGS=(
      --advantage-estimator grpo --no-normalize-advantages
      --executor-entropy-coef 0.0 --generator-entropy-coef "${GENERATOR_ENTROPY_COEF:-0.002}"
      --executor-kl-loss-coef 0.0 --generator-kl-loss-coef 0.0
      --kl-loss-coef 0.0 --kl-loss-type low_var_kl --entropy-coef 0.0
      --eps-clip 0.2 --eps-clip-high 0.28
    )
    ;;
  *)
    echo "Unknown GRPO_PROFILE=${GRPO_PROFILE}; expected vanilla, evolving, or delta" >&2
    exit 2
    ;;
esac

WANDB_ARGS=()
if [[ "${USE_WANDB:-1}" == "1" ]]; then
  WANDB_ARGS=(--use-wandb --wandb-project "${WANDB_PROJECT:-skillscaffold}" --wandb-group "${WANDB_GROUP:-alfworld-qwen3-8b}")
fi
SGLANG_ARGS=(--rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}" --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC:-0.55}")
if [[ -n "${SGLANG_ROUTER_PORT:-}" ]]; then SGLANG_ARGS+=(--sglang-router-port "${SGLANG_ROUTER_PORT}"); fi
if [[ "${SGLANG_ENABLE_DETERMINISTIC_INFERENCE:-0}" == "1" ]]; then
  SGLANG_ARGS+=(--sglang-enable-deterministic-inference)
fi
if [[ "${SGLANG_DISABLE_CUDA_GRAPH:-0}" == "1" ]]; then
  # SGLang CUDA graphs intermittently hit an illegal-memory-access failure on
  # long Evolving-RL ALFWorld requests.  Eager execution is slower but isolates
  # that inference-only failure without changing the training algorithm.
  SGLANG_ARGS+=(--sglang-disable-cuda-graph)
fi
MISC_ARGS=(
  --attention-dropout 0.0 --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 --attention-backend flash
  --save-debug-rollout-data "${DEBUG_ROLLOUT_PATH}"
)

NVLINK_COUNT="$( { nvidia-smi topo -m 2>/dev/null || true; } | awk '{count += gsub(/NV[0-9]+/, "")} END {print count + 0}')"
if [[ "${NVLINK_COUNT}" -gt 0 ]]; then HAS_NVLINK=1; else HAS_NVLINK=0; fi
PYTHON_SITE_PACKAGES="$("${PYTHON_BIN}" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
RUNTIME_ENV_JSON="$("${PYTHON_BIN}" - <<PY
import json,os
print(json.dumps({"env_vars": {
 "PYTHONPATH": "${PYTHON_SITE_PACKAGES}:${ROOT_DIR}:${SLIME_DIR}:${MEGATRON_DIR}",
 "PATH": os.environ["PATH"], "CUDA_DEVICE_MAX_CONNECTIONS": "1",
 "NCCL_NVLS_ENABLE": "${HAS_NVLINK}", "SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK": "1",
 "SLIME_SGLANG_BASE_PORT": os.environ.get("SGLANG_BASE_PORT", "19000"),
 "SLIME_SGLANG_FLUSH_CACHE_RETRIES": os.environ.get("SLIME_SGLANG_FLUSH_CACHE_RETRIES", "60"),
 "SGLANG_PROMETHEUS_PORT": os.environ.get("SGLANG_PROMETHEUS_PORT", "19950"),
 "SLIME_PLACEMENT_NODE_IP": os.environ.get("SLIME_PLACEMENT_NODE_IP", ""),
 "ALFWORLD_METHOD": "${METHOD}", "ALFWORLD_DATA_ROOT": "${DATA_ROOT}",
 "ALFWORLD_EVAL_PROTOCOL": "${EVAL_PROTOCOL}",
 "ALFWORLD_DELTA_CREDIT_SCOPE": "${CREDIT_SCOPE}",
 "ALFWORLD_GROUP_DELTA_VARIANT": os.environ.get("ALFWORLD_GROUP_DELTA_VARIANT", "positive_gate_trainext_raw"),
 "ALFWORLD_GENERATOR_PROMPT_FILE": os.environ.get("ALFWORLD_GENERATOR_PROMPT_FILE") or os.environ.get("ALFWORLD_GROUP_GENERATOR_PROMPT_FILE") or "${ROOT_DIR}/src/alfworld/group_generator_prompts.py",
 "ALFWORLD_GROUP_DELTA_FORWARD_CONCURRENCY": os.environ.get("ALFWORLD_GROUP_DELTA_FORWARD_CONCURRENCY", "48"),
 "ALFWORLD_SELF_EXTRACT_ANSWER_COUNT": os.environ.get("ALFWORLD_SELF_EXTRACT_ANSWER_COUNT", "8"),
 "ALFWORLD_ALLOW_EVAL_TAG_SPAN_MISMATCH": os.environ.get("ALFWORLD_ALLOW_EVAL_TAG_SPAN_MISMATCH", "0"),
 "ALFWORLD_ENV_REQUEST_TIMEOUT_SECONDS": os.environ.get("ALFWORLD_ENV_REQUEST_TIMEOUT_SECONDS", "0"),
 "ALFWORLD_KL_CHECKPOINT": os.environ.get("ALFWORLD_KL_CHECKPOINT", ""),
 "ALFWORLD_KL_STAGE": os.environ.get("ALFWORLD_KL_STAGE", ""),
 "ALFWORLD_STATIC_PROBE_ANCHORS": os.environ.get("ALFWORLD_STATIC_PROBE_ANCHORS", ""),
 "WANDB_API_KEY": os.environ.get("WANDB_API_KEY", ""),
 "WANDB_MODE": os.environ.get("WANDB_MODE", "online"),
 "WANDB_ENTITY": os.environ.get("WANDB_ENTITY", ""),
 "WANDB_PROJECT": os.environ.get("WANDB_PROJECT", "evolving-rl"),
 "WANDB_INIT_TIMEOUT": os.environ.get("WANDB_INIT_TIMEOUT", "90"),
 "TMPDIR": os.environ.get("TMPDIR", "/tmp"),
 "XDG_CACHE_HOME": os.environ.get("XDG_CACHE_HOME", ""),
 "WANDB_DIR": os.environ.get("WANDB_DIR", ""),
 "http_proxy": os.environ.get("http_proxy", ""), "https_proxy": os.environ.get("https_proxy", ""),
 "HTTP_PROXY": os.environ.get("HTTP_PROXY", os.environ.get("http_proxy", "")),
 "HTTPS_PROXY": os.environ.get("HTTPS_PROXY", os.environ.get("https_proxy", "")),
 "ALL_PROXY": "", "all_proxy": "",
 "NO_PROXY": "localhost,127.0.0.1,::1,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16",
 "no_proxy": "localhost,127.0.0.1,::1,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16"
}}))
PY
)"

ray job submit \
  --address="${RAY_DASHBOARD_ADDRESS:?RAY_DASHBOARD_ADDRESS is required}" \
  --submission-id="${RAY_SUBMISSION_ID:?RAY_SUBMISSION_ID is required}" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- "${PYTHON_BIN}" "${SLIME_DIR}/train.py" \
  --actor-num-gpus-per-node 8 --colocate \
  "${MODEL_ARGS[@]}" "${CKPT_ARGS[@]}" "${ROLLOUT_ARGS[@]}" "${EVAL_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" "${GRPO_ARGS[@]}" "${WANDB_ARGS[@]}" \
  "${PERF_ARGS[@]}" "${SGLANG_ARGS[@]}" "${CUSTOM_ARGS[@]}" "${MISC_ARGS[@]}"
