#!/usr/bin/env bash
# Run the default three-domain tau2-bench evaluation (8 trials per task).

set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
if [[ -f "${ROOT_DIR}/.env" ]]; then
  set -a
  source "${ROOT_DIR}/.env"
  set +a
fi

: "${TAU2_BENCH_DIR:?Set TAU2_BENCH_DIR to your tau2-bench checkout}"
: "${TAU2_EVAL_MODEL_DIR:?Set TAU2_EVAL_MODEL_DIR to the trained HF checkpoint}"
: "${TAU2_USER_API_BASE:?Set TAU2_USER_API_BASE to the OpenAI-compatible user simulator endpoint}"
: "${TAU2_USER_MODEL:?Set TAU2_USER_MODEL to the user-simulator model ID}"

TAU2_PYTHON="${TAU2_EVAL_PYTHON:-${TAU2_PYTHON:-python3}}"
VLLM_BIN="${VLLM_BIN:-$(command -v vllm || true)}"
: "${VLLM_BIN:?Set VLLM_BIN to the vllm executable}"

TAU2_EVAL_NUM_TRIALS="${TAU2_EVAL_NUM_TRIALS:-8}"
TAU2_EVAL_SMOKE="${TAU2_EVAL_SMOKE:-0}"
TAU2_EVAL_SEED="${TAU2_EVAL_SEED:-300}"
TAU2_EVAL_TAG="${TAU2_EVAL_TAG:-skillscaffold-tau-k8-seed${TAU2_EVAL_SEED}-$(date +%Y%m%d-%H%M%S)}"
TAU2_EVAL_OUTPUT_ROOT="${TAU2_EVAL_OUTPUT_ROOT:-${ROOT_DIR}/outputs/evaluation/tau/${TAU2_EVAL_TAG}}"

OTHER_GPUS="${TAU2_EVAL_OTHER_GPUS:-0,1,2}"
OTHER_PORTS="${TAU2_EVAL_OTHER_PORTS:-8000,8001,8003}"
TELECOM_GPUS="${TAU2_EVAL_TELECOM_GPUS:-3,4,5,6,7}"
TELECOM_PORTS="${TAU2_EVAL_TELECOM_PORTS:-8020,8021,8022,8023,8024}"

EVAL_CONCURRENCY="${TAU2_EVAL_CONCURRENCY:-16}"
TELECOM_CONCURRENCY="${TAU2_EVAL_TELECOM_CONCURRENCY:-24}"
AGENT_MAX_TOKENS="${TAU2_AGENT_MAX_TOKENS:-1024}"
AGENT_REQUEST_TIMEOUT="${TAU2_AGENT_REQUEST_TIMEOUT:-120}"
SIMULATION_TIMEOUT="${TAU2_SIMULATION_TIMEOUT:-900}"
SERVER_WAIT_TIMEOUT="${TAU2_SERVER_WAIT_TIMEOUT:-1200}"
MAX_MODEL_LEN="${TAU2_MAX_MODEL_LEN:-40960}"
GPU_MEMORY_UTILIZATION="${TAU2_GPU_MEMORY_UTILIZATION:-0.85}"

if [[ "${TAU2_EVAL_SMOKE}" == "1" ]]; then
  TAU2_EVAL_NUM_TRIALS="${TAU2_EVAL_SMOKE_NUM_TRIALS:-1}"
  OTHER_GPUS="${TAU2_EVAL_OTHER_GPUS:-0,1,2}"
  OTHER_PORTS="${TAU2_EVAL_OTHER_PORTS:-8000,8001,8002}"
  EVAL_CONCURRENCY="${TAU2_EVAL_SMOKE_CONCURRENCY:-1}"
elif [[ "${TAU2_EVAL_NUM_TRIALS}" != "8" ]]; then
  echo "ERROR: the default protocol requires TAU2_EVAL_NUM_TRIALS=8" >&2
  exit 2
fi

for required in config.json tokenizer_config.json model.safetensors.index.json; do
  if [[ ! -s "${TAU2_EVAL_MODEL_DIR}/${required}" ]]; then
    echo "ERROR: ${TAU2_EVAL_MODEL_DIR} is not a complete HF checkpoint (missing ${required})" >&2
    exit 2
  fi
done

for domain in airline retail telecom; do
  split_file="${TAU2_BENCH_DIR}/data/tau2/domains/${domain}/split_tasks.json"
  if [[ ! -s "${split_file}" ]]; then
    echo "ERROR: missing tau2-bench split file: ${split_file}" >&2
    exit 2
  fi
done

normalize_api_base() {
  local value="${1%/}"
  value="${value%/v1}"
  printf '%s/v1' "${value}"
}

litellm_model() {
  if [[ "$1" == */* ]]; then printf '%s' "$1"; else printf 'openai/%s' "$1"; fi
}

json_args() {
  "${TAU2_PYTHON}" - "$1" <<'PY'
import json
import sys

print(json.dumps({
    "api_base": sys.argv[1],
    "api_key": "EMPTY",
    "temperature": 0.0,
    "input_cost_per_token": 0,
    "output_cost_per_token": 0,
}))
PY
}

USER_API_BASE="$(normalize_api_base "${TAU2_USER_API_BASE}")"
USER_LLM="$(litellm_model "${TAU2_USER_MODEL}")"
USER_ARGS="$(json_args "${USER_API_BASE}")"

IFS=',' read -r -a OTHER_GPU_LIST <<<"${OTHER_GPUS}"
IFS=',' read -r -a OTHER_PORT_LIST <<<"${OTHER_PORTS}"
IFS=',' read -r -a TELECOM_GPU_LIST <<<"${TELECOM_GPUS}"
IFS=',' read -r -a TELECOM_PORT_LIST <<<"${TELECOM_PORTS}"
if ((${#OTHER_GPU_LIST[@]} != 3 || ${#OTHER_PORT_LIST[@]} != 3)); then
  echo "ERROR: TAU2_EVAL_OTHER_GPUS and TAU2_EVAL_OTHER_PORTS must contain 3 values" >&2
  exit 2
fi
if [[ "${TAU2_EVAL_SMOKE}" != "1" ]] && ((${#TELECOM_GPU_LIST[@]} != 5 || ${#TELECOM_PORT_LIST[@]} != 5)); then
  echo "ERROR: TAU2_EVAL_TELECOM_GPUS and TAU2_EVAL_TELECOM_PORTS must contain 5 values" >&2
  exit 2
fi

mkdir -p "${TAU2_EVAL_OUTPUT_ROOT}/data" "${TAU2_EVAL_OUTPUT_ROOT}/logs"
LOG_ROOT="${TAU2_EVAL_OUTPUT_ROOT}/logs"

export PYTHONPATH="${TAU2_BENCH_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
# tau2 itself appends the ``tau2/`` dataset namespace to TAU2_DATA_DIR.
export TAU2_DATA_DIR="${TAU2_BENCH_DIR}/data"
export TAU2_OUTPUT_DATA_DIR="${TAU2_EVAL_OUTPUT_ROOT}/data"
export TAU2_DISABLE_RESPONSE_COST=1
export TAU2_NL_ASSERTIONS_LLM="${USER_LLM}"
export TAU2_NL_ASSERTIONS_LLM_ARGS="${USER_ARGS}"
export NO_PROXY="127.0.0.1,localhost,10.0.0.0/8${NO_PROXY:+,${NO_PROXY}}"
export no_proxy="${NO_PROXY}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

SERVER_PIDS=()

cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  if ((${#SERVER_PIDS[@]})); then
    kill "${SERVER_PIDS[@]}" 2>/dev/null || true
    sleep 2
    kill -9 "${SERVER_PIDS[@]}" 2>/dev/null || true
  fi
  exit "${rc}"
}
trap cleanup EXIT INT TERM

port_is_open() {
  (exec 3<>"/dev/tcp/127.0.0.1/$1") >/dev/null 2>&1
}

start_server() {
  local gpu="$1"
  local port="$2"
  local log="${LOG_ROOT}/vllm_gpu${gpu}_port${port}.log"
  if port_is_open "${port}"; then
    echo "ERROR: local port ${port} is already occupied" >&2
    exit 5
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" "${VLLM_BIN}" serve "${TAU2_EVAL_MODEL_DIR}" \
    --tokenizer "${TAU2_EVAL_MODEL_DIR}" \
    --tokenizer-mode slow \
    --host 127.0.0.1 \
    --port "${port}" \
    --served-model-name local-agent \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser hermes >"${log}" 2>&1 &
  SERVER_PIDS+=("$!")
  echo "[tau/eval] vLLM gpu=${gpu} port=${port} pid=$!"
}

wait_for_server() {
  local port="$1"
  local pid="$2"
  local deadline=$((SECONDS + SERVER_WAIT_TIMEOUT))
  until curl --noproxy '*' --fail --silent --connect-timeout 2 --max-time 5 \
    "http://127.0.0.1:${port}/v1/models" >/dev/null; do
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "ERROR: vLLM on port ${port} exited during startup" >&2
      return 1
    fi
    if ((SECONDS >= deadline)); then
      echo "ERROR: vLLM on port ${port} did not start within ${SERVER_WAIT_TIMEOUT}s" >&2
      return 1
    fi
    sleep 5
  done
}

wait_for_user_model() {
  curl --noproxy '*' --fail --silent --show-error --max-time 20 \
    "${USER_API_BASE}/models" >/dev/null || {
      echo "ERROR: user simulator is unavailable: ${USER_API_BASE}" >&2
      exit 4
    }
}

task_shard() {
  local domain="$1"
  local shard_index="$2"
  local num_shards="$3"
  "${TAU2_PYTHON}" - \
    "${TAU2_BENCH_DIR}/data/tau2/domains/${domain}/split_tasks.json" \
    "${shard_index}" "${num_shards}" <<'PY'
import json
import sys

path, shard_index, num_shards = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
tasks = json.load(open(path, encoding="utf-8"))["base"]
start = len(tasks) * shard_index // num_shards
end = len(tasks) * (shard_index + 1) // num_shards
print("\n".join(map(str, tasks[start:end])))
PY
}

run_domain() {
  local domain="$1"
  local port="$2"
  local save_name="$3"
  local concurrency="$4"
  shift 4
  local agent_args
  local log="${LOG_ROOT}/${save_name}.log"
  agent_args="$("${TAU2_PYTHON}" - "${port}" "${AGENT_MAX_TOKENS}" "${AGENT_REQUEST_TIMEOUT}" <<'PY'
import json
import sys

print(json.dumps({
    "api_base": f"http://127.0.0.1:{sys.argv[1]}/v1",
    "api_key": "EMPTY",
    "temperature": 0.0,
    "max_tokens": int(sys.argv[2]),
    "timeout": int(sys.argv[3]),
    "input_cost_per_token": 0,
    "output_cost_per_token": 0,
    "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
}))
PY
)"
  (
    cd "${TAU2_BENCH_DIR}"
    "${TAU2_PYTHON}" -m tau2.cli run \
      --domain "${domain}" \
      --task-split-name base \
      --agent llm_agent \
      --agent-llm openai/local-agent \
      --agent-llm-args "${agent_args}" \
      --user-llm "${USER_LLM}" \
      --user-llm-args "${USER_ARGS}" \
      --num-trials "${TAU2_EVAL_NUM_TRIALS}" \
      --max-concurrency "${concurrency}" \
      --timeout "${SIMULATION_TIMEOUT}" \
      --max-retries 0 \
      --max-steps 200 \
      --max-errors 10 \
      --seed "${TAU2_EVAL_SEED}" \
      --save-to "${save_name}" \
      --auto-resume \
      "$@"
  ) 2>&1 | tee -a "${log}"
}

echo "[tau/eval] protocol=$([[ "${TAU2_EVAL_SMOKE}" == "1" ]] && echo smoke || echo default) domains=airline,retail,telecom trials=${TAU2_EVAL_NUM_TRIALS} seed=${TAU2_EVAL_SEED}"
if [[ "${TAU2_EVAL_SMOKE}" != "1" ]]; then
  echo "[tau/eval] expected=50+114+114 tasks, 2224 trajectories"
fi
wait_for_user_model

for i in 0 1 2; do
  start_server "${OTHER_GPU_LIST[$i]}" "${OTHER_PORT_LIST[$i]}"
done
if [[ "${TAU2_EVAL_SMOKE}" != "1" ]]; then
  for i in 0 1 2 3 4; do
    start_server "${TELECOM_GPU_LIST[$i]}" "${TELECOM_PORT_LIST[$i]}"
  done
fi

all_ports=("${OTHER_PORT_LIST[@]}")
if [[ "${TAU2_EVAL_SMOKE}" != "1" ]]; then
  all_ports+=("${TELECOM_PORT_LIST[@]}")
fi
for i in "${!all_ports[@]}"; do
  wait_for_server "${all_ports[$i]}" "${SERVER_PIDS[$i]}"
done

EVAL_PIDS=()
EVAL_NAMES=()
if [[ "${TAU2_EVAL_SMOKE}" == "1" ]]; then
  for i in 0 1 2; do
    domain=(airline retail telecom)
    mapfile -t task_ids < <(task_shard "${domain[$i]}" 0 1)
    save_name="${TAU2_EVAL_TAG}_${domain[$i]}_base"
    if [[ "${domain[$i]}" != airline ]]; then save_name+="_shard1"; fi
    run_domain "${domain[$i]}" "${OTHER_PORT_LIST[$i]}" \
      "${save_name}" \
      "${EVAL_CONCURRENCY}" --task-ids "${task_ids[0]}" &
    EVAL_PIDS+=("$!")
    EVAL_NAMES+=("${domain[$i]}")
  done
else
  run_domain airline "${OTHER_PORT_LIST[0]}" "${TAU2_EVAL_TAG}_airline_base" "${EVAL_CONCURRENCY}" &
  EVAL_PIDS+=("$!")
  EVAL_NAMES+=(airline)

  for i in 0 1; do
    mapfile -t task_ids < <(task_shard retail "${i}" 2)
    run_domain retail "${OTHER_PORT_LIST[$((i + 1))]}" \
      "${TAU2_EVAL_TAG}_retail_base_shard$((i + 1))" "${EVAL_CONCURRENCY}" \
      --task-ids "${task_ids[@]}" &
    EVAL_PIDS+=("$!")
    EVAL_NAMES+=("retail_shard$((i + 1))")
  done

  for i in 0 1 2 3 4; do
    mapfile -t task_ids < <(task_shard telecom "${i}" 5)
    run_domain telecom "${TELECOM_PORT_LIST[$i]}" \
      "${TAU2_EVAL_TAG}_telecom_base_shard$((i + 1))" "${TELECOM_CONCURRENCY}" \
      --task-ids "${task_ids[@]}" &
    EVAL_PIDS+=("$!")
    EVAL_NAMES+=("telecom_shard$((i + 1))")
  done
fi

failed=0
for i in "${!EVAL_PIDS[@]}"; do
  if wait "${EVAL_PIDS[$i]}"; then
    echo "[tau/eval] completed ${EVAL_NAMES[$i]}"
  else
    status=$?
    echo "ERROR: ${EVAL_NAMES[$i]} failed with exit ${status}" >&2
    failed=1
  fi
done
if ((failed)); then
  exit 1
fi

SUMMARY_ARGS=(
  --simulation-root "${TAU2_EVAL_OUTPUT_ROOT}/data/simulations"
  --run-prefix "${TAU2_EVAL_TAG}"
  --num-trials "${TAU2_EVAL_NUM_TRIALS}"
  --output "${TAU2_EVAL_OUTPUT_ROOT}/summary.json"
)
if [[ "${TAU2_EVAL_SMOKE}" == "1" ]]; then
  SUMMARY_ARGS+=(--expected-tasks-per-domain 1)
fi
"${TAU2_PYTHON}" "${ROOT_DIR}/benchmarks/tau/scripts/summarize_eval.py" "${SUMMARY_ARGS[@]}"

echo "[tau/eval] summary=${TAU2_EVAL_OUTPUT_ROOT}/summary.json"
