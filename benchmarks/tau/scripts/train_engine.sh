#!/bin/bash
# GRPO on APIGen->Tau2 tasks with complete banking goals, using upstream slime.
#
# Combines:
#   - Tau2-RL-Pipeline scripts/train_grpo.sh   (tau2 rollout/reward plugins)
#   - slime scripts/models/*.sh                 (Qwen model/perf args)
#
# Prereqs (inside the training container):
#   1. pip install -e <path-to-slime>
#   2. pip install -e <path-to-tau2-bench> --no-deps
#      pip install gymnasium addict deepdiff fs langfuse plotly \
#        pydantic-argparse redis scikit-learn seaborn tenacity watchdog \
#        "litellm==1.65.0"
#   3. Convert base HF ckpt to torch_dist (cold-start GRPO ref):
#        cd <path-to-slime>
#        bash scripts/convert-qwen3.5-4B.sh
#   4. User simulator: an OpenAI-compatible endpoint supplied by the caller.
#   5. Embedding server on :8010 for banking_knowledge; shares training GPU 0
#      with low mem util (start BEFORE training so it grabs its slice first):
#        GPUS=0 EMBED_MODEL_DIR=<Qwen3-Embedding-8B> \
#          bash scripts/start_embedding_server.sh

set -euo pipefail

if [ "${TAU2_CLEANUP:-0}" = "1" ] && [ "${RAY_CLUSTER_ALREADY_STARTED:-0}" != "1" ]; then
  pkill -9 sglang || true
  ray stop --force || true
  pkill -9 ray || true
fi

set -e
if [ "${TAU2_SHELL_TRACE:-0}" = "1" ]; then
  set -x
fi

export PYTHONUNBUFFERED=1
# Keep training progress and real errors visible without flooding the driver log
# with repeated dependency deprecations and per-request server INFO messages.
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::FutureWarning,ignore::DeprecationWarning}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF-expandable_segments:True}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-0}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-}"
export SLIME_ACTOR_NODE_IP="${SLIME_ACTOR_NODE_IP:-}"
export SLIME_HARD_PIN_ACTOR_NODE="${SLIME_HARD_PIN_ACTOR_NODE:-0}"
export RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-1}"
# Ray's internal OpenTelemetry exporter is independent of experiment tracking.
# Keep the upstream default unless a model-specific launcher disables it after
# a worker-startup crash; forward the choice to all nested Ray actors below.
export RAY_enable_open_telemetry="${RAY_enable_open_telemetry:-1}"
export LITELLM_LOG="${LITELLM_LOG:-ERROR}"
export SLIME_LOG_FULL_ROLLOUT_SAMPLES="${SLIME_LOG_FULL_ROLLOUT_SAMPLES:-0}"
export SLIME_SKIP_INITIAL_UPDATE_WEIGHTS="${SLIME_SKIP_INITIAL_UPDATE_WEIGHTS:-0}"
# SGLang rejects torch 2.9.1 with cuDNN < 9.15 because of a Conv3d bug.
# Qwen3.5 is text-only and does not use Conv3d, so bypass that unrelated check
# without changing the shared Python/CUDA environment.
export SGLANG_DISABLE_CUDNN_CHECK="${SGLANG_DISABLE_CUDNN_CHECK:-1}"
# Optional shared Triton MoE tuning directory. Keep SGLang's image-local
# default unless a launcher explicitly provides a config file.
export SGLANG_MOE_CONFIG_DIR="${SGLANG_MOE_CONFIG_DIR:-/sgl-workspace/sglang/python/sglang/srt/layers/moe/fused_moe_triton}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${REPO_ROOT}/../.." &>/dev/null && pwd)"
COMPAT_SITE="${TAU2_COMPAT_SITE:-${PROJECT_ROOT}/integrations}"


# ----- wandb / experiment naming -------
# WANDB_NAME is the SINGLE experiment knob:
#   - wandb run name  (slime derives run name from --wandb-group)
#   - checkpoint dir  (SAVE_DIR)
#   - trajectory dump dir (TAU2_DUMP_SUCCESS_DIR, poolcfg scripts)
# Override per experiment: WANDB_NAME=my_exp bash scripts/...
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_PROJECT="${WANDB_PROJECT:-skillscaffold}"
export WANDB_NAME="${WANDB_NAME:-qwen35_slime_debug}"
export WANDB_GROUP="${WANDB_GROUP:-${WANDB_NAME}}"

# ---- Framework paths (this machine, not the slimerl container) ----
SLIME_DIR="${SLIME_DIR:-${PROJECT_ROOT}/framework/slime_tau}"
MEGATRON_LM_DIR="${MEGATRON_LM_DIR:?set MEGATRON_LM_DIR (Megatron-LM repo used by slime)}"
TAU2_BENCH_DIR="${TAU2_BENCH_DIR:?set TAU2_BENCH_DIR to your tau2-bench checkout}"
if ! PYTHONPATH="${SLIME_DIR}:${PYTHONPATH:-}" python -c \
  'from slime.utils.types import Sample; assert hasattr(Sample, "append_response_tokens")'; then
  echo "ERROR: SLIME_DIR=${SLIME_DIR} does not provide Sample.append_response_tokens" >&2
  echo "Set SLIME_DIR to the bundled Tau-compatible Slime directory" >&2
  exit 2
fi

# ---- Model (cold-start GRPO from the checkpoint supplied by the caller) ----
MODEL_FAMILY="${MODEL_FAMILY:-qwen3_5}"
MODEL_SIZE="${MODEL_SIZE:-4B}"
case "${MODEL_FAMILY}" in
  qwen3_5)
    HF_DIR="${HF_DIR:?set HF_DIR to the Hugging Face checkpoint}"
    REF_CKPT_DIR="${REF_CKPT_DIR:?set REF_CKPT_DIR to the Megatron-format checkpoint}"
    if [ "${MODEL_SIZE}" = "35B-A3B" ]; then
      EXPECTED_MODEL_TYPE="qwen3_5_moe"
    else
      EXPECTED_MODEL_TYPE="qwen3_5"
    fi
    export TAU2_TOOL_PROTOCOL="qwen35_xml"
    export TAU2_PREFIX_STABLE_QWEN35="${TAU2_PREFIX_STABLE_QWEN35:-1}"
    ;;
  qwen3)
    case "${MODEL_SIZE}" in
      4B)
        HF_DIR="${HF_DIR:?set HF_DIR to the Hugging Face checkpoint}"
        REF_CKPT_DIR="${REF_CKPT_DIR:?set REF_CKPT_DIR to the Megatron-format checkpoint}"
        ;;
      8B)
        RAW_QWEN3_8B_HF_DIR="${HF_DIR:?set HF_DIR to the Hugging Face checkpoint}"
        QWEN3_8B_BASE_FIXED_HF_DIR="${QWEN3_8B_BASE_FIXED_HF_DIR:-}"
        if [ -n "${QWEN3_8B_BASE_FIXED_HF_DIR}" ] && \
           [ -s "${QWEN3_8B_BASE_FIXED_HF_DIR}/tokenizer_config.json" ]; then
          HF_DIR="${QWEN3_8B_BASE_FIXED_HF_DIR}"
          echo "Using base Qwen3-8B weights with Tau2-compatible tokenizer: ${HF_DIR}"
        else
          HF_DIR="${RAW_QWEN3_8B_HF_DIR}"
        fi
        REF_CKPT_DIR="${REF_CKPT_DIR:?set REF_CKPT_DIR to the Megatron-format checkpoint}"
        ;;
      30B-A3B)
        HF_DIR="${HF_DIR:?set HF_DIR to the Hugging Face checkpoint}"
        REF_CKPT_DIR="${REF_CKPT_DIR:?set REF_CKPT_DIR to the Megatron-format checkpoint}"
        ;;
      *)
        echo "ERROR: Qwen3 MODEL_SIZE must be 4B, 8B, or 30B-A3B, got ${MODEL_SIZE}" >&2
        exit 2
        ;;
    esac
    if [ "${MODEL_SIZE}" = "30B-A3B" ]; then
      EXPECTED_MODEL_TYPE="qwen3_moe"
    else
      EXPECTED_MODEL_TYPE="qwen3"
    fi
    export TAU2_TOOL_PROTOCOL="qwen3_json"
    export TAU2_PREFIX_STABLE_QWEN35="0"
    ;;
  *)
    echo "ERROR: MODEL_FAMILY must be qwen3 or qwen3_5, got ${MODEL_FAMILY}" >&2
    exit 2
    ;;
esac

# Qwen3.5 can train either with or without reasoning. Keep false as the
# compatibility default; Qwen3's existing JSON tool protocol remains no-think.
AGENT_ENABLE_THINKING="${AGENT_ENABLE_THINKING:-0}"
AGENT_PRESERVE_THINKING="${AGENT_PRESERVE_THINKING:-0}"
case "${AGENT_PRESERVE_THINKING}" in
  0|1) ;;
  *)
    echo "ERROR: AGENT_PRESERVE_THINKING must be 0 or 1, got ${AGENT_PRESERVE_THINKING}" >&2
    exit 2
    ;;
esac
case "${AGENT_ENABLE_THINKING}" in
  0)
    if [ "${AGENT_PRESERVE_THINKING}" = "1" ]; then
      echo "ERROR: AGENT_PRESERVE_THINKING=1 requires AGENT_ENABLE_THINKING=1" >&2
      exit 2
    fi
    APPLY_CHAT_TEMPLATE_KWARGS='{"enable_thinking":false}'
    ;;
  1)
    if [ "${MODEL_FAMILY}" != "qwen3_5" ]; then
      echo "ERROR: AGENT_ENABLE_THINKING=1 is supported only for MODEL_FAMILY=qwen3_5" >&2
      exit 2
    fi
    if [ "${AGENT_PRESERVE_THINKING}" = "1" ]; then
      APPLY_CHAT_TEMPLATE_KWARGS='{"enable_thinking":true,"preserve_thinking":true}'
    else
      APPLY_CHAT_TEMPLATE_KWARGS='{"enable_thinking":true}'
    fi
    ;;
  *)
    echo "ERROR: AGENT_ENABLE_THINKING must be 0 or 1, got ${AGENT_ENABLE_THINKING}" >&2
    exit 2
    ;;
esac
export TAU2_AGENT_ENABLE_THINKING="${AGENT_ENABLE_THINKING}"
export TAU2_AGENT_PRESERVE_THINKING="${AGENT_PRESERVE_THINKING}"

ACTUAL_MODEL_TYPE="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["model_type"])' "${HF_DIR}/config.json")"
if [ "${ACTUAL_MODEL_TYPE}" != "${EXPECTED_MODEL_TYPE}" ]; then
  echo "ERROR: MODEL_FAMILY=${MODEL_FAMILY} expects model_type=${EXPECTED_MODEL_TYPE}, got ${ACTUAL_MODEL_TYPE} from ${HF_DIR}" >&2
  exit 2
fi
if [ "${MODEL_FAMILY}" = "qwen3" ]; then
  python "${REPO_ROOT}/scripts/validate_qwen3_tool_template.py" --model-dir "${HF_DIR}"
fi

LOAD_DIR="${LOAD_DIR:-${REF_CKPT_DIR}}"             # cold start: load = ref = base
SAVE_DIR="${SAVE_DIR:-${PROJECT_ROOT}/outputs/checkpoints/tau/${WANDB_NAME}}"
MEGATRON_TO_HF_MODE="${MEGATRON_TO_HF_MODE:-raw}"
if [ "${MEGATRON_TO_HF_MODE}" = "bridge" ]; then
  if [ ! -s "${LOAD_DIR}/config.json" ] || [ ! -s "${REF_CKPT_DIR}/config.json" ]; then
    echo "ERROR: bridge mode requires Hugging Face load/ref directories with config.json:" >&2
    echo "       LOAD_DIR=${LOAD_DIR}" >&2
    echo "       REF_CKPT_DIR=${REF_CKPT_DIR}" >&2
    exit 2
  fi
elif ! find "${REF_CKPT_DIR}" -maxdepth 3 -type f -name common.pt -print -quit 2>/dev/null | grep -q .; then
  echo "ERROR: distributed reference checkpoint not found under ${REF_CKPT_DIR}" >&2
  if [ "${MODEL_FAMILY}" = "qwen3" ]; then
    echo "Run: HF_CHECKPOINT=${HF_DIR} SAVE_PATH=${REF_CKPT_DIR} bash scripts/convert-qwen3-${MODEL_SIZE}.sh" >&2
  elif [ "${MODEL_SIZE}" = "35B-A3B" ]; then
    echo "Run: HF_CHECKPOINT=${HF_DIR} SAVE_PATH=${REF_CKPT_DIR} bash scripts/convert-qwen3.5-35B-A3B.sh" >&2
  fi
  exit 2
fi
if [ "${MODEL_FAMILY}" = "qwen3_5" ] && [ -z "${CHAT_TEMPLATE:-}" ]; then
  CHAT_TEMPLATE="${REPO_ROOT}/.generated/qwen35_tau2_chat_template.jinja"
  python "${REPO_ROOT}/scripts/make_qwen35_tau2_chat_template.py" \
    --source "${HF_DIR}/chat_template.jinja" \
    --output "${CHAT_TEMPLATE}"
fi
export TAU2_QWEN35_CHAT_TEMPLATE="${CHAT_TEMPLATE:-}"

# ---- Data: Qwen3/Qwen3.5 shared 1/8..7/8 pass-rate pool + banking train ----
TAU2_TRAIN_POOL_ROOT="${TAU2_TRAIN_POOL_ROOT:?set TAU2_TRAIN_POOL_ROOT to the prepared Tau training pool}"
TAU2_TRAIN_TASKS_JSONL="${TAU2_TRAIN_TASKS_JSONL:-${TAU2_TRAIN_POOL_ROOT}/jsonl/four_domain_train_pool.jsonl}"
export TAU2_CUSTOM_TASKS_DIR="${TAU2_CUSTOM_TASKS_DIR:-${TAU2_TRAIN_POOL_ROOT}}"
export TAU2_DATA_DIR="${TAU2_DATA_DIR:-${TAU2_BENCH_DIR}/data}"
if [ ! -s "${TAU2_TRAIN_TASKS_JSONL}" ]; then
  echo "ERROR: Tau2 training JSONL not found or empty: ${TAU2_TRAIN_TASKS_JSONL}" >&2
  exit 2
fi
if [ "${TAU2_DOMAIN_BALANCE:-0}" = "1" ]; then
  SOURCE_TRAIN_TASKS_JSONL="${TAU2_TRAIN_TASKS_JSONL}"
  TAU2_TRAIN_TASKS_JSONL="${TAU2_DOMAIN_BALANCED_TASKS_JSONL:-${TAU2_TRAIN_POOL_ROOT}/jsonl/three_domain_balanced_round_robin.jsonl}"
  python "${REPO_ROOT}/scripts/build_domain_balanced_prompt_data.py" \
    --input "${SOURCE_TRAIN_TASKS_JSONL}" \
    --output "${TAU2_TRAIN_TASKS_JSONL}" \
    --domains airline retail telecom \
    --seed "${ROLLOUT_SEED:-42}"
  # Preserve round-robin ordering so every rollout batch is balanced, rather
  # than merely balancing the full epoch in expectation.
  TAU2_ROLLOUT_SHUFFLE=0
fi
echo "Tau2 training pool: ${TAU2_TRAIN_TASKS_JSONL} ($(wc -l < "${TAU2_TRAIN_TASKS_JSONL}") tasks)"
echo "Tau2 custom task registry: ${TAU2_CUSTOM_TASKS_DIR}"

# Cluster layout. NUM_GPUS is GPUs per node (kept for backward
# compatibility with the existing single-node launch commands).
NUM_GPUS="${NUM_GPUS:-8}"
ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-1}"
ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-${NUM_GPUS}}"
WORKER_IPS="${WORKER_IPS:-}"
SSH_USER="${SSH_USER:-${USER:-}}"
RAY_CLUSTER_ALREADY_STARTED="${RAY_CLUSTER_ALREADY_STARTED:-0}"
START_WORKERS_VIA_SSH="${START_WORKERS_VIA_SSH:-1}"
RAY_CLUSTER_WAIT_SECONDS="${RAY_CLUSTER_WAIT_SECONDS:-300}"

# ---- User simulator (external OpenAI-compatible endpoint) ----
# litellm openai-compatible route: "openai/<served model id>"
export TAU2_USER_MODEL="${TAU2_USER_MODEL:?set TAU2_USER_MODEL to the served model identifier}"
export TAU2_USER_API_BASE="${TAU2_USER_API_BASE:?set TAU2_USER_API_BASE to an OpenAI-compatible /v1 endpoint}"
export TAU2_USER_TEMPERATURE="${TAU2_USER_TEMPERATURE:-0.0}"
export TAU2_USER_REQUEST_TIMEOUT="${TAU2_USER_REQUEST_TIMEOUT:-120}"
export TAU2_ENV_STEP_TIMEOUT_SECONDS="${TAU2_ENV_STEP_TIMEOUT_SECONDS:-300}"
export TAU2_EPISODE_TIMEOUT_SECONDS="${TAU2_EPISODE_TIMEOUT_SECONDS:-1200}"
# By default, use the same model and endpoint as the user simulator to judge
# natural-language assertions. Both variables remain independently overridable.
export TAU2_NL_ASSERTIONS_LLM="${TAU2_NL_ASSERTIONS_LLM:-${TAU2_USER_MODEL}}"
if [ -z "${TAU2_NL_ASSERTIONS_LLM_ARGS:-}" ]; then
  export TAU2_NL_ASSERTIONS_LLM_ARGS="{\"api_base\":\"${TAU2_USER_API_BASE}\",\"api_key\":\"EMPTY\",\"temperature\":${TAU2_USER_TEMPERATURE},\"timeout\":${TAU2_USER_REQUEST_TIMEOUT},\"input_cost_per_token\":0,\"output_cost_per_token\":0}"
else
  export TAU2_NL_ASSERTIONS_LLM_ARGS
fi
TAU2_NL_ASSERTIONS_LLM_ARGS_RUNTIME="${TAU2_NL_ASSERTIONS_LLM_ARGS//\\/\\\\}"
TAU2_NL_ASSERTIONS_LLM_ARGS_RUNTIME="${TAU2_NL_ASSERTIONS_LLM_ARGS_RUNTIME//\"/\\\"}"
export TAU2_DISABLE_RESPONSE_COST="${TAU2_DISABLE_RESPONSE_COST:-1}"
export TAU2_MAX_STEPS="${TAU2_MAX_STEPS:-40}"
export TAU2_BANKING_MAX_STEPS="${TAU2_BANKING_MAX_STEPS:-80}"
export TAU2_MAX_IDENTICAL_ACTIONS="${TAU2_MAX_IDENTICAL_ACTIONS:-0}"
export TAU2_MAX_EPISODE_TOKENS="${TAU2_MAX_EPISODE_TOKENS:-32768}"
export TAU2_FORMAT_RETRIES="${TAU2_FORMAT_RETRIES:-0}"
export TAU2_LOG_LEVEL="${TAU2_LOG_LEVEL:-ERROR}"
export TAU2_ENV_CONCURRENCY="${TAU2_ENV_CONCURRENCY:-32}"
export TAU2_HTTP_MAX_CONNECTIONS="${TAU2_HTTP_MAX_CONNECTIONS:-64}"
export TAU2_HTTP_MAX_KEEPALIVE_CONNECTIONS="${TAU2_HTTP_MAX_KEEPALIVE_CONNECTIONS:-64}"
export TAU2_STATIC_SKILLS_ENABLED="${TAU2_STATIC_SKILLS_ENABLED:-0}"
export TAU2_STATIC_SKILLS_JSON="${TAU2_STATIC_SKILLS_JSON:-}"
export TAU2_STATIC_SKILLS_TOP_K="${TAU2_STATIC_SKILLS_TOP_K:-6}"
export TAU2_STATIC_SKILLS_TASK_TOP_K="${TAU2_STATIC_SKILLS_TASK_TOP_K:-6}"
if [ "${TAU2_FORMAT_RETRIES}" != "0" ]; then
  echo "ERROR: native Qwen rollout requires TAU2_FORMAT_RETRIES=0; malformed output must remain an on-policy zero-reward trajectory." >&2
  exit 2
fi

# ---- banking_knowledge retrieval (Qwen3-Embedding-8B on :8010) ----
export TAU2_BANKING_RETRIEVAL_VARIANT="${TAU2_BANKING_RETRIEVAL_VARIANT:-qwen_embeddings}"
export OPENROUTER_BASE_URL="${OPENROUTER_BASE_URL:-http://127.0.0.1:8010/v1}"
export OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-EMPTY}"

# ---- Reward shaping (same defaults as train_grpo.sh) ----
export TAU2_PARTIAL_ACTION_WEIGHT="${TAU2_PARTIAL_ACTION_WEIGHT:-0.0}"
export TAU2_PARTIAL_COMMUNICATE_WEIGHT="${TAU2_PARTIAL_COMMUNICATE_WEIGHT:-0.0}"
export TAU2_PARTIAL_ENV_ASSERTION_WEIGHT="${TAU2_PARTIAL_ENV_ASSERTION_WEIGHT:-0.0}"
export TAU2_PARTIAL_DB_WEIGHT="${TAU2_PARTIAL_DB_WEIGHT:-0.0}"
export TAU2_BANKING_COMMUNICATION_BOOST="${TAU2_BANKING_COMMUNICATION_BOOST:-1}"
export TAU2_SAFETY_ENABLED="${TAU2_SAFETY_ENABLED:-1}"
export TAU2_ABORT_TRUNCATED_FRAC="${TAU2_ABORT_TRUNCATED_FRAC:-0.60}"
export TAU2_ABORT_MIN_SUCCESS_RATE="${TAU2_ABORT_MIN_SUCCESS_RATE:-0.02}"
export TAU2_ABORT_PATIENCE="${TAU2_ABORT_PATIENCE:-2}"
export TAU2_ABORT_WARMUP_ROLLOUTS="${TAU2_ABORT_WARMUP_ROLLOUTS:-3}"
export TAU2_USE_COMPRESSED_PROMPTS="${TAU2_USE_COMPRESSED_PROMPTS:-0}"

# Human-readable rollout snapshots. One JSON file is written every N rollout
# steps with independent failure/success controls for each domain.
export TAU2_TRAJECTORY_DUMP_DIR="${TAU2_TRAJECTORY_DUMP_DIR:-${PROJECT_ROOT}/outputs/trajectories/tau/${WANDB_NAME}}"
export TAU2_TRAJECTORY_DUMP_INTERVAL="${TAU2_TRAJECTORY_DUMP_INTERVAL:-10}"
export TAU2_TRAJECTORY_DUMP_FAILURES_PER_DOMAIN="${TAU2_TRAJECTORY_DUMP_FAILURES_PER_DOMAIN:-4}"
export TAU2_TRAJECTORY_DUMP_SUCCESSES_PER_DOMAIN="${TAU2_TRAJECTORY_DUMP_SUCCESSES_PER_DOMAIN:-2}"
export TAU2_TRAJECTORY_DUMP_INCLUDE_TOKENS="${TAU2_TRAJECTORY_DUMP_INCLUDE_TOKENS:-0}"

if [ "${MODEL_FAMILY}" = "qwen3_5" ]; then
  source "${SLIME_DIR}/scripts/models/qwen3.5-${MODEL_SIZE}.sh"
  QWEN_GDN_BACKEND="${QWEN_GDN_BACKEND:-fla}"
  case "${QWEN_GDN_BACKEND}" in
    fla|flashqla)
      MODEL_ARGS+=(--qwen-gdn-backend "${QWEN_GDN_BACKEND}")
      ;;
    *)
      echo "ERROR: QWEN_GDN_BACKEND must be fla or flashqla, got ${QWEN_GDN_BACKEND}" >&2
      exit 2
      ;;
  esac
  QWEN_GDN_FLASHQLA_AUTO_CP="${QWEN_GDN_FLASHQLA_AUTO_CP:-1}"
  case "${QWEN_GDN_FLASHQLA_AUTO_CP}" in
    1)
      ;;
    0)
      MODEL_ARGS+=(--no-qwen-gdn-flashqla-auto-cp)
      ;;
    *)
      echo "ERROR: QWEN_GDN_FLASHQLA_AUTO_CP must be 0 or 1, got ${QWEN_GDN_FLASHQLA_AUTO_CP}" >&2
      exit 2
      ;;
  esac
  if [ "${QWEN_GDN_BACKEND}" = "flashqla" ] && ! python3 -c 'from flash_qla import chunk_gated_delta_rule' >/dev/null 2>&1; then
    echo "ERROR: QWEN_GDN_BACKEND=flashqla but flash_qla is unavailable in ${PATH}" >&2
    exit 2
  fi
  DEFAULT_MAX_TOKENS_PER_GPU=8192
else
  QWEN3_ROTARY_BASE="$(python -c 'import json,sys; print(int(json.load(open(sys.argv[1])).get("rope_theta", 1000000)))' "${HF_DIR}/config.json")"
  QWEN3_MODEL_ARGS="${SLIME_DIR}/scripts/models/qwen3-${MODEL_SIZE}.sh"
  if [ ! -f "${QWEN3_MODEL_ARGS}" ]; then
    echo "ERROR: Qwen3 model arguments not found: ${QWEN3_MODEL_ARGS}" >&2
    exit 2
  fi
  MODEL_ARGS_ROTARY_BASE="${QWEN3_ROTARY_BASE}" source "${QWEN3_MODEL_ARGS}"
  if [ "${MODEL_SIZE}" = "8B" ]; then
    DEFAULT_MAX_TOKENS_PER_GPU=8192
  else
    DEFAULT_MAX_TOKENS_PER_GPU=12288
  fi
fi

CKPT_ARGS=(
  --hf-checkpoint "${HF_DIR}"
  --ref-load "${REF_CKPT_DIR}"
  --load "${LOAD_DIR}"
  --megatron-to-hf-mode "${MEGATRON_TO_HF_MODE}"
)
if [ "${TAU2_DISABLE_SAVE:-0}" = "0" ]; then
  CKPT_ARGS+=(--save "${SAVE_DIR}" --save-interval "${SAVE_INTERVAL:-20}")
elif [ "${TAU2_DISABLE_SAVE}" != "1" ]; then
  echo "ERROR: TAU2_DISABLE_SAVE must be 0 or 1" >&2
  exit 2
fi
if [ "${LOAD_OPTIMIZER:-0}" != "1" ]; then
  CKPT_ARGS+=(--no-load-optim)
fi
if [ "${NO_SAVE_OPTIM:-0}" = "1" ]; then
  CKPT_ARGS+=(--no-save-optim)
elif [ "${NO_SAVE_OPTIM:-0}" != "0" ]; then
  echo "ERROR: NO_SAVE_OPTIM must be 0 or 1" >&2
  exit 2
fi

ROLLOUT_ARGS=(
  --prompt-data "${TAU2_TRAIN_TASKS_JSONL}"
  --input-key text
  --metadata-key metadata
  --apply-chat-template
  # The Qwen3.5 append-only renderer validates every generation prefix against
  # this exact mode. Qwen3 keeps the historical false default.
  --apply-chat-template-kwargs "${APPLY_CHAT_TEMPLATE_KWARGS}"

  # rollout_batch_size * n_samples_per_prompt must be a multiple of global_batch_size
  --num-rollout "${NUM_ROLLOUT:-200}"
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE:-8}"
  --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT:-8}"
  # per-action generation cap (multi-turn; total context grows across turns)
  --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN:-1024}"
  --rollout-temperature "${ROLLOUT_TEMPERATURE:-0.7}"
  --rollout-top-p "${ROLLOUT_TOP_P:-1.0}"
  --rollout-top-k "${ROLLOUT_TOP_K:--1}"

  --global-batch-size "${GLOBAL_BATCH_SIZE:-64}"

  # Custom Tau2 rollout supplies an append-only mask directly. This argument
  # remains the SFT/fallback tokenizer default and is not used to rebuild RL masks.
  --loss-mask-type qwen
)
if [ -n "${START_ROLLOUT_ID:-}" ]; then
  ROLLOUT_ARGS+=(--start-rollout-id "${START_ROLLOUT_ID}")
fi
if [ "${BALANCE_DATA:-1}" = "1" ]; then
  ROLLOUT_ARGS+=(--balance-data)
elif [ "${BALANCE_DATA:-1}" = "0" ]; then
  echo "Length-based rollout data balancing disabled by BALANCE_DATA=0"
else
  echo "ERROR: BALANCE_DATA must be 0 or 1, got ${BALANCE_DATA}" >&2
  exit 2
fi
if [ "${TAU2_ROLLOUT_SHUFFLE:-1}" = "1" ]; then
  ROLLOUT_ARGS+=(--rollout-shuffle)
fi

# Retain compatibility with forks that expose a fixed scheduler length.
if grep -q -- '"--total-training-steps"' "${SLIME_DIR}/slime/utils/arguments.py"; then
  ROLLOUT_ARGS+=(--total-training-steps "${TOTAL_TRAINING_STEPS:-1000000}")
fi

# Always resample groups containing infrastructure failures or zero sampled
# policy tokens. DAPO-style zero-std filtering remains opt-in.
ROLLOUT_ARGS+=(
  --dynamic-sampling-filter-path tau2_rl_pipeline.rollout_filter.keep_valid_tau2_group
)
export TAU2_DROP_ZERO_STD_GROUPS="${USE_DYNAMIC_FILTER:-0}"

# DAPO-style dynamic sampling: drop groups whose rewards have zero std
# (all-pass or all-fail) — they contribute zero GRPO advantage and dilute the
# gradient. The rollout keeps sampling extra tasks until rollout_batch_size
# groups WITH signal are collected, so wall-clock per rollout increases
# (~1.5-2x at high success rates) in exchange for denser signal per step.
if [ "${USE_DYNAMIC_FILTER:-0}" = "1" ]; then
  echo "DAPO zero-std filtering enabled"
fi

GRPO_ARGS=(
  --advantage-estimator grpo
  # Do not turn tiny shaping differences into unit-scale advantages.
  --disable-grpo-std-normalization
  --eps-clip 0.2
  --eps-clip-high 0.28
  # 0.001 caused policy collapse here WITHOUT the dynamic filter: with most
  # groups at zero-std reward the PG signal vanishes and the entropy bonus
  # dominates, degrading outputs until every trajectory fails to parse (see
  # run 20260709_193912). Only raise this together with USE_DYNAMIC_FILTER=1,
  # which removes the zero-signal groups that let entropy dominate.
  --entropy-coef "${ENTROPY_COEF:-0.0}"
  --kl-loss-coef "${KL_LOSS_COEF:-0.01}"
  --kl-loss-type low_var_kl
)

# Keep KL enabled by default for existing launchers while allowing isolated
# no-KL ablations to disable it without maintaining a second copy of this
# large, hardware-specific launcher.
case "${USE_KL_LOSS:-1}" in
  1)
    GRPO_ARGS+=(--use-kl-loss)
    ;;
  0)
    echo "KL loss disabled by USE_KL_LOSS=0"
    ;;
  *)
    echo "ERROR: USE_KL_LOSS must be 0 or 1, got ${USE_KL_LOSS}" >&2
    exit 2
    ;;
esac

# Entropy is only a diagnostic when ENTROPY_COEF=0, and retaining it is
# expensive for long-context large-vocabulary models.  Keep it configurable.
if [ "${USE_ROLLOUT_ENTROPY:-1}" = "1" ]; then
  GRPO_ARGS+=(--use-rollout-entropy)
fi

OPTIMIZER_ARGS=(
  --optimizer adam
  # 1e-6 is the community default for 4B-scale agentic GRPO (MUA-RL, SkillRL,
  # upstream Tau2-RL-Pipeline). The old 2e-7 default predates the fixes for
  # the actual collapse causes (off-policy multi-step, std normalization,
  # every-turn tool_call parsing) and just made learning slow.
  --lr "${LR:-1e-6}"
  --lr-decay-style constant
  --weight-decay "${WEIGHT_DECAY:-0.01}"
  --adam-beta1 0.9
  --adam-beta2 "${ADAM_BETA2:-0.999}"
)

PERF_ARGS=(
  # 4B fits without TP; bump TP_SIZE for larger MODEL_SIZE
  --tensor-model-parallel-size "${TP_SIZE:-1}"
  --sequence-parallel
  --pipeline-model-parallel-size "${PP_SIZE:-1}"
  --context-parallel-size "${CP_SIZE:-1}"
  --expert-model-parallel-size "${EP_SIZE:-1}"
  --expert-tensor-parallel-size "${ETP_SIZE:-1}"
  --micro-batch-size 1
  --use-dynamic-batch-size
  # Qwen3.5's 248k vocab makes log-prob logits huge: [tokens, vocab] fp32 is
  # ~1GB per 1k tokens and calculate_log_probs_and_entropy clones it (~2-3x
  # peak). 20480 tokens OOMed (37GB alloc on top of weights+optimizer);
  # 8192 caps the logits+clone peak at ~16GB.
  --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-${DEFAULT_MAX_TOKENS_PER_GPU}}"
  # Bound the temporary logits clone/softmax working set in upstream slime.
  --log-probs-chunk-size "${LOG_PROBS_CHUNK_SIZE:-2048}"

  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
)

if { [ "${MODEL_FAMILY}" = "qwen3" ] && [ "${MODEL_SIZE}" = "30B-A3B" ]; } || \
   { [ "${MODEL_FAMILY}" = "qwen3_5" ] && [ "${MODEL_SIZE}" = "35B-A3B" ]; }; then
  # Single-node 30B-A3B runs often need CPU Adam to fit. Multi-node H800
  # runs can disable it to avoid CPU<->GPU optimizer traffic and use the
  # distributed optimizer shards instead.
  OPTIMIZER_CPU_OFFLOAD="${OPTIMIZER_CPU_OFFLOAD:-1}"
  if [ "${OPTIMIZER_CPU_OFFLOAD}" = "1" ]; then
    OPTIMIZER_ARGS+=(
      --optimizer-cpu-offload
      --overlap-cpu-optimizer-d2h-h2d
    )
  elif [ "${OPTIMIZER_CPU_OFFLOAD}" != "0" ]; then
    echo "ERROR: OPTIMIZER_CPU_OFFLOAD must be 0 or 1, got ${OPTIMIZER_CPU_OFFLOAD}" >&2
    exit 2
  fi
  if [ "${MODEL_FAMILY}" = "qwen3_5" ]; then
    OPTIMIZER_ARGS+=(--use-distributed-optimizer)
  fi
  if [ "${USE_STATELESS_ADAM:-0}" = "1" ]; then
    OPTIMIZER_ARGS+=(--use-stateless-adam)
  elif [ "${USE_STATELESS_ADAM:-0}" != "0" ]; then
    echo "ERROR: USE_STATELESS_ADAM must be 0 or 1, got ${USE_STATELESS_ADAM}" >&2
    exit 2
  fi
  OPTIMIZER_ARGS+=(--use-precision-aware-optimizer)
fi

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}"
  # Leave headroom for the colocated Qwen3-Embedding-8B server (~0.25 of one card)
  --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION:-0.60}"
  --sglang-log-level "${SGLANG_LOG_LEVEL:-warning}"
)
if [ -n "${SGLANG_ATTENTION_BACKEND:-}" ]; then
  SGLANG_ARGS+=(--sglang-attention-backend "${SGLANG_ATTENTION_BACKEND}")
fi
PLACEMENT_ARGS=()
if [ "${COLOCATE:-1}" = "1" ]; then
  PLACEMENT_ARGS+=(--colocate)
fi
ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-$((ACTOR_NUM_NODES * ACTOR_NUM_GPUS_PER_NODE))}"
if [ "${SGLANG_DISABLE_CUDA_GRAPH:-0}" = "1" ]; then
  SGLANG_ARGS+=(--sglang-disable-cuda-graph)
elif [ -n "${SGLANG_CUDA_GRAPH_MAX_BS:-}" ]; then
  SGLANG_ARGS+=(--sglang-cuda-graph-max-bs "${SGLANG_CUDA_GRAPH_MAX_BS}")
fi
if [ "${MODEL_FAMILY}" = "qwen3_5" ]; then
  SGLANG_ARGS+=(--sglang-chat-template "${CHAT_TEMPLATE}")
fi

MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
  --no-gradient-accumulation-fusion
)

# Megatron's default pre-collective NaN check aborts the whole distributed
# job.  With this opt-in switch, the optimizer performs its synchronized
# non-finite check instead and slime skips the affected update without
# applying corrupt gradients.  Keep the default off for existing launchers.
case "${SKIP_NONFINITE_GRAD_STEPS:-0}" in
  1)
    MISC_ARGS+=(--no-check-for-nan-in-loss-and-grad)
    ;;
  0)
    ;;
  *)
    echo "ERROR: SKIP_NONFINITE_GRAD_STEPS must be 0 or 1, got ${SKIP_NONFINITE_GRAD_STEPS}" >&2
    exit 2
    ;;
esac

CUSTOM_ARGS=(
  --custom-generate-function-path tau2_rl_pipeline.rollout.generate
  --custom-reward-post-process-path tau2_rl_pipeline.reward.tau2_reward_post_process
  --custom-rollout-log-function-path tau2_rl_pipeline.rollout_logging.dump_rollout_trajectories
)

if [ "${TAU2_DISABLE_WANDB:-0}" = "1" ]; then
  echo "WandB disabled by TAU2_DISABLE_WANDB=1"
  WANDB_ARGS=()
  RUNTIME_WANDB_API_KEY=""
elif [ -n "${WANDB_API_KEY:-}" ]; then
  WANDB_ARGS=(
    --use-wandb
    --wandb-project "${WANDB_PROJECT}"
    --wandb-group "${WANDB_GROUP:-grpo-qwen3.5-${MODEL_SIZE}}"
  )
  RUNTIME_WANDB_API_KEY="${WANDB_API_KEY}"
else
  echo "NOTE: WANDB_API_KEY not set, running without WandB logging"
  WANDB_ARGS=()
  RUNTIME_WANDB_API_KEY=""
fi


# Ray workers may run under a system Python that ignores the active venv, so expose the venv's
# site-packages to them via PYTHONPATH. Auto-detected from the active venv.
if [ -n "${TAU2_RAY_SITE_PACKAGES:-}" ]; then
  VENV_SITE_PACKAGES="${TAU2_RAY_SITE_PACKAGES}"
elif [ -n "${VIRTUAL_ENV:-}" ]; then
  VENV_SITE_PACKAGES="$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
else
  VENV_SITE_PACKAGES=""
fi

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
RAY_DASHBOARD_HOST="${RAY_DASHBOARD_HOST:-127.0.0.1}"
INTERNAL_NO_PROXY="${INTERNAL_NO_PROXY:-127.0.0.1,localhost,::1,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16}"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}${INTERNAL_NO_PROXY}"
export no_proxy="${NO_PROXY}"

if ! [[ "${ACTOR_NUM_NODES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: ACTOR_NUM_NODES must be a positive integer, got ${ACTOR_NUM_NODES}" >&2
  exit 2
fi
if ! [[ "${ACTOR_NUM_GPUS_PER_NODE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: ACTOR_NUM_GPUS_PER_NODE must be a positive integer, got ${ACTOR_NUM_GPUS_PER_NODE}" >&2
  exit 2
fi

# Megatron requires global_batch_size to be divisible by
# micro_batch_size * data_parallel_size.  This launcher fixes micro batch and
# pipeline parallelism at one, so fail before allocating GPUs and loading the
# checkpoint when a smoke-test override is inconsistent with the layout.
TP_SIZE_VALUE="${TP_SIZE:-1}"
CP_SIZE_VALUE="${CP_SIZE:-1}"
GLOBAL_BATCH_SIZE_VALUE="${GLOBAL_BATCH_SIZE:-64}"
ROLLOUT_BATCH_SIZE_VALUE="${ROLLOUT_BATCH_SIZE:-8}"
N_SAMPLES_PER_PROMPT_VALUE="${N_SAMPLES_PER_PROMPT:-8}"
for VALUE_NAME in TP_SIZE_VALUE CP_SIZE_VALUE GLOBAL_BATCH_SIZE_VALUE ROLLOUT_BATCH_SIZE_VALUE N_SAMPLES_PER_PROMPT_VALUE; do
  VALUE="${!VALUE_NAME}"
  if ! [[ "${VALUE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: ${VALUE_NAME} must be a positive integer, got ${VALUE}" >&2
    exit 2
  fi
done
WORLD_SIZE_VALUE=$((ACTOR_NUM_NODES * ACTOR_NUM_GPUS_PER_NODE))
MODEL_PARALLEL_SIZE_VALUE=$((TP_SIZE_VALUE * CP_SIZE_VALUE))
if [ $((WORLD_SIZE_VALUE % MODEL_PARALLEL_SIZE_VALUE)) -ne 0 ]; then
  echo "ERROR: world size ${WORLD_SIZE_VALUE} must be divisible by TP*CP=${MODEL_PARALLEL_SIZE_VALUE}" >&2
  exit 2
fi
DATA_PARALLEL_SIZE_VALUE=$((WORLD_SIZE_VALUE / MODEL_PARALLEL_SIZE_VALUE))
if [ $((GLOBAL_BATCH_SIZE_VALUE % DATA_PARALLEL_SIZE_VALUE)) -ne 0 ]; then
  echo "ERROR: GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE_VALUE} must be divisible by data parallel size ${DATA_PARALLEL_SIZE_VALUE}" >&2
  exit 2
fi
ROLLOUT_SAMPLE_COUNT_VALUE=$((ROLLOUT_BATCH_SIZE_VALUE * N_SAMPLES_PER_PROMPT_VALUE))
if [ $((ROLLOUT_SAMPLE_COUNT_VALUE % GLOBAL_BATCH_SIZE_VALUE)) -ne 0 ]; then
  echo "ERROR: ROLLOUT_BATCH_SIZE*N_SAMPLES_PER_PROMPT=${ROLLOUT_SAMPLE_COUNT_VALUE} must be divisible by GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE_VALUE}" >&2
  exit 2
fi

# Use the interface that routes to the first worker unless explicitly set.
# These values are propagated to every Ray task below.
FIRST_WORKER_IP="${WORKER_IPS//,/ }"
FIRST_WORKER_IP="${FIRST_WORKER_IP%% *}"
if [ -n "${FIRST_WORKER_IP}" ]; then
  SOCKET_IFNAME="${SOCKET_IFNAME:-$(ip route get "${FIRST_WORKER_IP}" 2>/dev/null | awk '{for (i=1;i<=NF;i++) if ($i=="dev") {print $(i+1); exit}}')}"
else
  SOCKET_IFNAME="${SOCKET_IFNAME:-lo}"
fi
export SLIME_HOST_IP="${SLIME_HOST_IP:-${MASTER_ADDR}}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${SOCKET_IFNAME}}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-${SOCKET_IFNAME}}"
export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME="${NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME:-${SOCKET_IFNAME}}"

if [ "${TAU2_LAUNCH_DRY_RUN:-0}" = "1" ]; then
  echo "Tau2 launch preflight passed"
  echo "  model_family=${MODEL_FAMILY} model_size=${MODEL_SIZE} model_type=${ACTUAL_MODEL_TYPE}"
  echo "  hf=${HF_DIR} load=${LOAD_DIR} ref=${REF_CKPT_DIR}"
  echo "  enable_thinking=${AGENT_ENABLE_THINKING} preserve_thinking=${AGENT_PRESERVE_THINKING} template_kwargs=${APPLY_CHAT_TEMPLATE_KWARGS}"
  echo "  actor_nodes=${ACTOR_NUM_NODES} gpus_per_node=${ACTOR_NUM_GPUS_PER_NODE} tp=${TP_SIZE:-1} cp=${CP_SIZE:-1} ep=${EP_SIZE:-1}"
  if [ "${MODEL_FAMILY}" = "qwen3_5" ]; then
    echo "  qwen_gdn_backend=${QWEN_GDN_BACKEND} flashqla_auto_cp=${QWEN_GDN_FLASHQLA_AUTO_CP} rollout_entropy=${USE_ROLLOUT_ENTROPY:-1}"
  fi
  echo "  rollout_gpus_per_engine=${ROLLOUT_NUM_GPUS_PER_ENGINE:-1} train_data=${TAU2_TRAIN_TASKS_JSONL}"
  exit 0
elif [ "${TAU2_LAUNCH_DRY_RUN:-0}" != "0" ]; then
  echo "ERROR: TAU2_LAUNCH_DRY_RUN must be 0 or 1" >&2
  exit 2
fi

if [ "${RAY_CLUSTER_ALREADY_STARTED}" = "1" ]; then
  echo "Reusing the existing Ray cluster at ${MASTER_ADDR}:6379"
else
  ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${ACTOR_NUM_GPUS_PER_NODE}" \
    --disable-usage-stats --dashboard-host="${RAY_DASHBOARD_HOST}" --dashboard-port=8265
fi

if [ "${ACTOR_NUM_NODES}" -gt 1 ]; then
  read -r -a WORKER_IP_ARRAY <<< "${WORKER_IPS//,/ }"
  EXPECTED_WORKERS=$((ACTOR_NUM_NODES - 1))
  if [ "${#WORKER_IP_ARRAY[@]}" -ne "${EXPECTED_WORKERS}" ]; then
    echo "ERROR: ACTOR_NUM_NODES=${ACTOR_NUM_NODES} requires ${EXPECTED_WORKERS} worker IP(s);" >&2
    echo "       got WORKER_IPS='${WORKER_IPS}'" >&2
    exit 2
  fi
  if [ "${START_WORKERS_VIA_SSH}" = "1" ] && \
     [ "${RAY_CLUSTER_ALREADY_STARTED}" != "1" ] && \
     [ -z "${SSH_USER}" ]; then
    echo "ERROR: SSH_USER is empty; set it to the login name used on worker nodes" >&2
    exit 2
  fi

  if [ "${START_WORKERS_VIA_SSH}" = "1" ] && [ "${RAY_CLUSTER_ALREADY_STARTED}" != "1" ]; then
    RAY_BIN="${RAY_BIN:-$(command -v ray)}"
    for WORKER_IP in "${WORKER_IP_ARRAY[@]}"; do
      echo "Starting Ray worker ${SSH_USER}@${WORKER_IP} -> ${MASTER_ADDR}:6379"
      ssh -o BatchMode=yes -o ConnectTimeout=10 "${SSH_USER}@${WORKER_IP}" \
        "export NO_PROXY='${NO_PROXY}' no_proxy='${NO_PROXY}' SLIME_HOST_IP='${WORKER_IP}' GLOO_SOCKET_IFNAME='${GLOO_SOCKET_IFNAME}' NCCL_SOCKET_IFNAME='${NCCL_SOCKET_IFNAME}' NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME='${NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME}'; '${RAY_BIN}' stop --force >/dev/null 2>&1 || true; '${RAY_BIN}' start --address='${MASTER_ADDR}:6379' --node-ip-address='${WORKER_IP}' --num-gpus='${ACTOR_NUM_GPUS_PER_NODE}' --disable-usage-stats"
    done
  fi

  EXPECTED_GPUS=$((ACTOR_NUM_NODES * ACTOR_NUM_GPUS_PER_NODE))
  python - "${ACTOR_NUM_NODES}" "${EXPECTED_GPUS}" "${RAY_CLUSTER_WAIT_SECONDS}" <<'PY'
import sys
import time
import ray

expected_nodes = int(sys.argv[1])
expected_gpus = int(sys.argv[2])
wait_seconds = int(sys.argv[3])
ray.init(address="auto", logging_level="ERROR")
deadline = time.monotonic() + wait_seconds
while True:
    alive_nodes = sum(node["Alive"] for node in ray.nodes())
    gpus = int(ray.cluster_resources().get("GPU", 0))
    if alive_nodes >= expected_nodes and gpus >= expected_gpus:
        print(f"Ray cluster ready: {alive_nodes} nodes, {gpus} GPUs")
        break
    if time.monotonic() >= deadline:
        raise SystemExit(
            f"ERROR: Ray cluster not ready: {alive_nodes}/{expected_nodes} nodes, "
            f"{gpus}/{expected_gpus} GPUs"
        )
    time.sleep(1)
ray.shutdown()
PY
fi

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${COMPAT_SITE}:${SLIME_DIR}:${REPO_ROOT}:${TAU2_BENCH_DIR}/src:${MEGATRON_LM_DIR}:${VENV_SITE_PACKAGES}\",
    \"LD_LIBRARY_PATH\": \"${LD_LIBRARY_PATH:-}\",
    \"http_proxy\": \"${http_proxy:-}\",
    \"https_proxy\": \"${https_proxy:-}\",
    \"HTTP_PROXY\": \"${HTTP_PROXY:-${http_proxy:-}}\",
    \"HTTPS_PROXY\": \"${HTTPS_PROXY:-${https_proxy:-}}\",
    \"NO_PROXY\": \"${NO_PROXY}\",
    \"no_proxy\": \"${no_proxy}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"MASTER_ADDR\": \"${MASTER_ADDR}\",
    \"GLOO_SOCKET_IFNAME\": \"${GLOO_SOCKET_IFNAME}\",
    \"NCCL_SOCKET_IFNAME\": \"${NCCL_SOCKET_IFNAME}\",
    \"NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME\": \"${NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME}\",
    \"SGLANG_DISABLE_CUDNN_CHECK\": \"${SGLANG_DISABLE_CUDNN_CHECK}\",
    \"SGLANG_MOE_CONFIG_DIR\": \"${SGLANG_MOE_CONFIG_DIR}\",
    \"PYTHONWARNINGS\": \"${PYTHONWARNINGS}\",
    \"TOKENIZERS_PARALLELISM\": \"${TOKENIZERS_PARALLELISM}\",
    \"TRANSFORMERS_VERBOSITY\": \"${TRANSFORMERS_VERBOSITY}\",
    \"HF_HUB_DISABLE_PROGRESS_BARS\": \"${HF_HUB_DISABLE_PROGRESS_BARS}\",
    \"PYTORCH_ALLOC_CONF\": \"${PYTORCH_ALLOC_CONF}\",
    \"NCCL_SHM_DISABLE\": \"${NCCL_SHM_DISABLE}\",
    \"NCCL_NVLS_ENABLE\": \"${NCCL_NVLS_ENABLE}\",
    \"SLIME_ACTOR_NODE_IP\": \"${SLIME_ACTOR_NODE_IP}\",
    \"SLIME_HARD_PIN_ACTOR_NODE\": \"${SLIME_HARD_PIN_ACTOR_NODE}\",
    \"RAY_DEDUP_LOGS\": \"${RAY_DEDUP_LOGS}\",
    \"RAY_enable_open_telemetry\": \"${RAY_enable_open_telemetry}\",
    \"LITELLM_LOG\": \"${LITELLM_LOG}\",
    \"SLIME_LOG_FULL_ROLLOUT_SAMPLES\": \"${SLIME_LOG_FULL_ROLLOUT_SAMPLES}\",
    \"SLIME_SKIP_INITIAL_UPDATE_WEIGHTS\": \"${SLIME_SKIP_INITIAL_UPDATE_WEIGHTS}\",
    \"TAU2_DATA_DIR\": \"${TAU2_DATA_DIR}\",
    \"TAU2_CUSTOM_TASKS_DIR\": \"${TAU2_CUSTOM_TASKS_DIR}\",
    \"TAU2_BANKING_RETRIEVAL_VARIANT\": \"${TAU2_BANKING_RETRIEVAL_VARIANT}\",
    \"OPENROUTER_BASE_URL\": \"${OPENROUTER_BASE_URL}\",
    \"OPENROUTER_API_KEY\": \"${OPENROUTER_API_KEY}\",
    \"TAU2_USER_MODEL\": \"${TAU2_USER_MODEL}\",
    \"TAU2_USER_API_BASE\": \"${TAU2_USER_API_BASE}\",
    \"TAU2_USER_TEMPERATURE\": \"${TAU2_USER_TEMPERATURE}\",
    \"TAU2_USER_REQUEST_TIMEOUT\": \"${TAU2_USER_REQUEST_TIMEOUT}\",
    \"TAU2_ENV_STEP_TIMEOUT_SECONDS\": \"${TAU2_ENV_STEP_TIMEOUT_SECONDS}\",
    \"TAU2_EPISODE_TIMEOUT_SECONDS\": \"${TAU2_EPISODE_TIMEOUT_SECONDS}\",
    \"TAU2_NL_ASSERTIONS_LLM\": \"${TAU2_NL_ASSERTIONS_LLM}\",
    \"TAU2_NL_ASSERTIONS_LLM_ARGS\": \"${TAU2_NL_ASSERTIONS_LLM_ARGS_RUNTIME}\",
    \"TAU2_DISABLE_RESPONSE_COST\": \"${TAU2_DISABLE_RESPONSE_COST}\",
    \"TAU2_MAX_STEPS\": \"${TAU2_MAX_STEPS}\",
    \"TAU2_BANKING_MAX_STEPS\": \"${TAU2_BANKING_MAX_STEPS}\",
    \"TAU2_MAX_IDENTICAL_ACTIONS\": \"${TAU2_MAX_IDENTICAL_ACTIONS}\",
    \"TAU2_MAX_EPISODE_TOKENS\": \"${TAU2_MAX_EPISODE_TOKENS}\",
    \"TAU2_FORMAT_RETRIES\": \"${TAU2_FORMAT_RETRIES}\",
    \"TAU2_LOG_LEVEL\": \"${TAU2_LOG_LEVEL}\",
    \"TAU2_ENV_CONCURRENCY\": \"${TAU2_ENV_CONCURRENCY}\",
    \"TAU2_HTTP_MAX_CONNECTIONS\": \"${TAU2_HTTP_MAX_CONNECTIONS}\",
    \"TAU2_HTTP_MAX_KEEPALIVE_CONNECTIONS\": \"${TAU2_HTTP_MAX_KEEPALIVE_CONNECTIONS}\",
    \"TAU2_STATIC_SKILLS_ENABLED\": \"${TAU2_STATIC_SKILLS_ENABLED}\",
    \"TAU2_STATIC_SKILLS_JSON\": \"${TAU2_STATIC_SKILLS_JSON}\",
    \"TAU2_STATIC_SKILLS_TOP_K\": \"${TAU2_STATIC_SKILLS_TOP_K}\",
    \"TAU2_STATIC_SKILLS_TASK_TOP_K\": \"${TAU2_STATIC_SKILLS_TASK_TOP_K}\",
    \"TAU2_DROP_ZERO_STD_GROUPS\": \"${TAU2_DROP_ZERO_STD_GROUPS}\",
    \"TAU2_TOOL_PROTOCOL\": \"${TAU2_TOOL_PROTOCOL}\",
    \"TAU2_AGENT_ENABLE_THINKING\": \"${TAU2_AGENT_ENABLE_THINKING}\",
    \"TAU2_AGENT_PRESERVE_THINKING\": \"${TAU2_AGENT_PRESERVE_THINKING}\",
    \"TAU2_PREFIX_STABLE_QWEN35\": \"${TAU2_PREFIX_STABLE_QWEN35}\",
    \"TAU2_QWEN35_CHAT_TEMPLATE\": \"${TAU2_QWEN35_CHAT_TEMPLATE}\",
    \"TAU2_PARTIAL_ACTION_WEIGHT\": \"${TAU2_PARTIAL_ACTION_WEIGHT}\",
    \"TAU2_PARTIAL_COMMUNICATE_WEIGHT\": \"${TAU2_PARTIAL_COMMUNICATE_WEIGHT}\",
    \"TAU2_PARTIAL_ENV_ASSERTION_WEIGHT\": \"${TAU2_PARTIAL_ENV_ASSERTION_WEIGHT}\",
    \"TAU2_PARTIAL_DB_WEIGHT\": \"${TAU2_PARTIAL_DB_WEIGHT}\",
    \"TAU2_BANKING_COMMUNICATION_BOOST\": \"${TAU2_BANKING_COMMUNICATION_BOOST}\",
    \"TAU2_SAFETY_ENABLED\": \"${TAU2_SAFETY_ENABLED}\",
    \"TAU2_ABORT_TRUNCATED_FRAC\": \"${TAU2_ABORT_TRUNCATED_FRAC}\",
    \"TAU2_ABORT_MIN_SUCCESS_RATE\": \"${TAU2_ABORT_MIN_SUCCESS_RATE}\",
    \"TAU2_ABORT_PATIENCE\": \"${TAU2_ABORT_PATIENCE}\",
    \"TAU2_ABORT_WARMUP_ROLLOUTS\": \"${TAU2_ABORT_WARMUP_ROLLOUTS}\",
    \"TAU2_USE_COMPRESSED_PROMPTS\": \"${TAU2_USE_COMPRESSED_PROMPTS}\",
    \"TAU2_TRAJECTORY_DUMP_DIR\": \"${TAU2_TRAJECTORY_DUMP_DIR}\",
    \"TAU2_TRAJECTORY_DUMP_INTERVAL\": \"${TAU2_TRAJECTORY_DUMP_INTERVAL}\",
    \"TAU2_TRAJECTORY_DUMP_FAILURES_PER_DOMAIN\": \"${TAU2_TRAJECTORY_DUMP_FAILURES_PER_DOMAIN}\",
    \"TAU2_TRAJECTORY_DUMP_SUCCESSES_PER_DOMAIN\": \"${TAU2_TRAJECTORY_DUMP_SUCCESSES_PER_DOMAIN}\",
    \"TAU2_TRAJECTORY_DUMP_INCLUDE_TOKENS\": \"${TAU2_TRAJECTORY_DUMP_INCLUDE_TOKENS}\",
    \"TAU_E7_SMOKE_DOMAIN\": \"${TAU_E7_SMOKE_DOMAIN:-}\",
    \"TAU_E7_FIXED_SLOT_RETRIES\": \"${TAU_E7_FIXED_SLOT_RETRIES:-2}\",
    \"TAU2_E7_GENERATOR_INPUT_TOKENS\": \"${TAU2_E7_GENERATOR_INPUT_TOKENS:-15000}\",
    \"WANDB_NAME\": \"${WANDB_NAME}\",
    \"WANDB_GROUP\": \"${WANDB_GROUP}\",
    \"WANDB_RUN_ID\": \"${WANDB_RUN_ID:-}\",
    \"WANDB_RESUME\": \"${WANDB_RESUME:-allow}\",
    \"WANDB_API_KEY\": \"${RUNTIME_WANDB_API_KEY}\",
    \"HF_TOKEN\": \"${HF_TOKEN:-}\"
  }
}"

# Persist driver output to NFS (ray's /tmp/ray copy dies with the container).
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/outputs/logs/tau}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/${WANDB_NAME}_$(date +%Y%m%d_%H%M%S).log"
echo "Driver log: ${LOG_FILE}"

ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 "${SLIME_DIR}/train.py" \
  --actor-num-nodes "${ACTOR_NUM_NODES}" \
  --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}" \
  --rollout-num-gpus "${ROLLOUT_NUM_GPUS}" \
  ${PLACEMENT_ARGS[@]+"${PLACEMENT_ARGS[@]}"} \
  ${MODEL_ARGS[@]} \
  ${CKPT_ARGS[@]} \
  ${ROLLOUT_ARGS[@]} \
  ${OPTIMIZER_ARGS[@]} \
  ${GRPO_ARGS[@]} \
  ${PERF_ARGS[@]} \
  ${SGLANG_ARGS[@]} \
  ${MISC_ARGS[@]} \
  ${CUSTOM_ARGS[@]} \
  ${WANDB_ARGS[@]+"${WANDB_ARGS[@]}"} \
  "$@" 2>&1 | tee "${LOG_FILE}"
