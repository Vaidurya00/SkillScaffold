#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TAU_BENCHMARK_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd -- "${TAU_BENCHMARK_DIR}/../.." && pwd)"
if [[ -f "${PROJECT_ROOT}/.env" ]]; then
  set -a
  source "${PROJECT_ROOT}/.env"
  set +a
fi

: "${MEGATRON_LM_DIR:?Set MEGATRON_LM_DIR to your Megatron-LM checkout}"
: "${TAU2_BENCH_DIR:?Set TAU2_BENCH_DIR to your tau2-bench checkout}"
: "${HF_DIR:?Set HF_DIR to the Tau domain-SFT Hugging Face checkpoint}"
: "${REF_CKPT_DIR:?Set REF_CKPT_DIR to the corresponding Megatron-format checkpoint}"
: "${TAU2_USER_API_BASE:?Set TAU2_USER_API_BASE to an OpenAI-compatible /v1 endpoint}"
: "${TAU2_USER_MODEL:?Set TAU2_USER_MODEL to the served user-simulator model ID}"

export TAU2_TRAIN_POOL_ROOT="${TAU2_TRAIN_POOL_ROOT:-${PROJECT_ROOT}/data/tau/rl/runtime}"
export TAU2_TRAIN_TASKS_JSONL="${TAU2_TRAIN_TASKS_JSONL:-${TAU2_TRAIN_POOL_ROOT}/jsonl/three_domain_joint_family_conservative_round_robin.jsonl}"
if [[ ! -s "${TAU2_TRAIN_TASKS_JSONL}" ]]; then
  echo "Tau RL data not found: ${TAU2_TRAIN_TASKS_JSONL}" >&2
  echo "Download it first with scripts/download_tau_data.py and set TAU2_TRAIN_POOL_ROOT if needed." >&2
  exit 2
fi

export MODEL_FAMILY="${MODEL_FAMILY:-qwen3}" MODEL_SIZE="${MODEL_SIZE:-8B}"
export SLIME_DIR="${SLIME_DIR:-${PROJECT_ROOT}/framework/slime_tau}"
export TAU2_COMPAT_SITE="${TAU2_COMPAT_SITE:-${PROJECT_ROOT}/integrations}"
export PYTHONPATH="${TAU_BENCHMARK_DIR}:${PROJECT_ROOT}/integrations:${SLIME_DIR}:${TAU2_BENCH_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

export RUN_KIND="${RUN_KIND:-long}"
export RUN_TAG="${RUN_TAG:-skillscaffold-tau-qwen3-8b-e7}"
export LOAD_DIR="${LOAD_DIR:-${REF_CKPT_DIR}}"
export LOAD_OPTIMIZER="${LOAD_OPTIMIZER:-0}"
export SAVE_DIR="${SAVE_DIR:-${PROJECT_ROOT}/outputs/checkpoints/tau/${RUN_TAG}}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-25}"
export WANDB_PROJECT="${WANDB_PROJECT:-skillscaffold}"
export WANDB_NAME="${WANDB_NAME:-${RUN_TAG}}"
export WANDB_GROUP="${WANDB_GROUP:-${RUN_TAG}}"
export TAU2_DATA_DIR="${TAU2_DATA_DIR:-${TAU2_BENCH_DIR}/data}"
export TAU2_CUSTOM_TASKS_DIR="${TAU2_CUSTOM_TASKS_DIR:-${TAU2_TRAIN_POOL_ROOT}}"

export NUM_GPUS="${NUM_GPUS:-8}" ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-1}" ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"
export TP_SIZE="${TP_SIZE:-2}" CP_SIZE="${CP_SIZE:-1}" PP_SIZE="${PP_SIZE:-1}"
export ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-8}" ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-2}" COLOCATE="${COLOCATE:-1}"
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-6144}" LOG_PROBS_CHUNK_SIZE="${LOG_PROBS_CHUNK_SIZE:-2048}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-32}" N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-1}" GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
export NUM_ROLLOUT="${NUM_ROLLOUT:-200}"
export ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.7}" ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-1.0}" LR="${LR:-1e-6}" ENTROPY_COEF="${ENTROPY_COEF:-0.0}"
export USE_KL_LOSS="${USE_KL_LOSS:-0}" KL_LOSS_COEF="${KL_LOSS_COEF:-0.0}" USE_DYNAMIC_FILTER="${USE_DYNAMIC_FILTER:-0}" USE_ROLLOUT_ENTROPY="${USE_ROLLOUT_ENTROPY:-1}"
export TAU_E7_FIXED_SLOT_RETRIES="${TAU_E7_FIXED_SLOT_RETRIES:-2}" TAU2_E7_GENERATOR_INPUT_TOKENS="${TAU2_E7_GENERATOR_INPUT_TOKENS:-16384}"
export TAU2_PARTIAL_ACTION_WEIGHT=0 TAU2_PARTIAL_COMMUNICATE_WEIGHT=0
export TAU2_PARTIAL_ENV_ASSERTION_WEIGHT=0 TAU2_PARTIAL_DB_WEIGHT=0
export TAU2_FORMAT_RETRIES=0 TAU2_MAX_IDENTICAL_ACTIONS=0
export TAU2_MAX_EPISODE_TOKENS="${TAU2_MAX_EPISODE_TOKENS:-32768}" TAU2_MAX_STEPS="${TAU2_MAX_STEPS:-160}"
export TAU2_ENV_CONCURRENCY=64 TAU2_HTTP_MAX_CONNECTIONS=64 TAU2_HTTP_MAX_KEEPALIVE_CONNECTIONS=64
export TAU2_TRAJECTORY_DUMP_INTERVAL="${TAU2_TRAJECTORY_DUMP_INTERVAL:-1}"
export TAU2_TRAJECTORY_DUMP_FAILURES_PER_DOMAIN=8 TAU2_TRAJECTORY_DUMP_SUCCESSES_PER_DOMAIN=8
export RAY_CLUSTER_ALREADY_STARTED="${RAY_CLUSTER_ALREADY_STARTED:-1}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export SLIME_ACTOR_NODE_IP="${SLIME_ACTOR_NODE_IP:-${MASTER_ADDR}}"
export SLIME_HARD_PIN_ACTOR_NODE="${SLIME_HARD_PIN_ACTOR_NODE:-1}"
export START_WORKERS_VIA_SSH="${START_WORKERS_VIA_SSH:-0}" TAU2_CLEANUP="${TAU2_CLEANUP:-0}"
export AGENT_ENABLE_THINKING="${AGENT_ENABLE_THINKING:-0}" AGENT_PRESERVE_THINKING="${AGENT_PRESERVE_THINKING:-0}" TAU2_TOOL_PROTOCOL="${TAU2_TOOL_PROTOCOL:-qwen3_json}"
export SGLANG_MEM_FRACTION="${SGLANG_MEM_FRACTION:-0.62}"
export LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/outputs/logs/tau}"

if [[ "${RUN_KIND}" == "smoke" ]]; then
  export TAU2_ROLLOUT_SHUFFLE=0 BALANCE_DATA="${SMOKE_BALANCE_DATA:-1}"
  export ROLLOUT_BATCH_SIZE="${SMOKE_ROLLOUT_BATCH_SIZE:-4}"
  export GLOBAL_BATCH_SIZE="${SMOKE_GLOBAL_BATCH_SIZE:-${ROLLOUT_BATCH_SIZE}}"
  # A checkpoint at iteration 0 resumes with start_rollout_id=1.  Therefore
  # num_rollout=2 is the smallest setting that executes one rollout/update.
  export NUM_ROLLOUT="${SMOKE_NUM_ROLLOUT:-2}" SAVE_INTERVAL="${SMOKE_SAVE_INTERVAL:-1}"
fi

exec bash "${SCRIPT_DIR}/train_qwen3.sh" \
  --rollout-function-path delta_card_tau.fixed_slot_rollout.generate_rollout \
  --dynamic-sampling-filter-path delta_card_tau.e7_pipeline.keep_valid_tau_e7_group \
  --custom-generate-function-path delta_card_tau.e7_pipeline.tau_e7_pipeline \
  --custom-reward-post-process-path delta_card_tau.e7_pipeline.tau_e7_reward_post_process \
  --custom-config-path "${TAU_BENCHMARK_DIR}/e7_nokl.yaml" \
  --start-rollout-id "${START_ROLLOUT_ID:-0}" \
  --executor-entropy-coef 0 \
  --generator-entropy-coef 0.002
