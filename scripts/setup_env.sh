#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${1:-}"
SGLANG_VERSION="${SGLANG_VERSION:-0.5.9}"
MEGATRON_COMMIT="${MEGATRON_COMMIT:-3714d81d418c9f1bca4594fc35f9e8289f652862}"
MEGATRON_DIR="${MEGATRON_DIR:-${ROOT_DIR}/third_party/Megatron-LM}"
WEBSHOP_COMMIT="${WEBSHOP_COMMIT:-703e635619901c9c84f76caff0907a37d1a262a8}"
WEBSHOP_VENDOR_ROOT="${WEBSHOP_VENDOR_ROOT:-${ROOT_DIR}/third_party/Skill0_5/agent_system/environments/env_package/webshop/webshop}"
TAU2_BENCH_COMMIT="${TAU2_BENCH_COMMIT:-5ebebbe827b455b3ed04fcb9294235c6ef4e5fd6}"
TAU2_BENCH_DIR="${TAU2_BENCH_DIR:-${ROOT_DIR}/third_party/tau2-bench}"
ALFWORLD_TASK_FILES_ROOT="${ALFWORLD_TASK_FILES_ROOT:-${ROOT_DIR}/data/alfworld/task_files}"
ALFWORLD_PARQUET_DIR="${ALFWORLD_PARQUET_DIR:-${ROOT_DIR}/benchmarks/alfworld/data}"

usage() {
  cat <<'EOF'
Usage: bash scripts/setup_env.sh <target>

Targets:
  main          Training environment for ALFWorld, WebShop, and ScienceWorld
  tau           Training and evaluation environments for tau2-bench
  alfworld      ALFWorld environment service
  webshop       WebShop environment service
  scienceworld  ScienceWorld environment service
  all           Install main and all three environment services

Environment variables:
  SKIP_GPU_BACKEND=1  Keep an existing PyTorch/SGLang installation
EOF
}

if [[ -z "${TARGET}" ]]; then
  usage
  exit 2
fi

if [[ "${TARGET}" == "-h" || "${TARGET}" == "--help" || "${TARGET}" == "help" ]]; then
  usage
  exit 0
fi

command -v conda >/dev/null 2>&1 || {
  echo "conda is required but was not found" >&2
  exit 1
}

env_exists() {
  conda env list | awk '{print $1}' | grep -Fxq "$1"
}

env_prefix() {
  conda run -n "$1" python -c 'import sys; print(sys.prefix)'
}

set_env_value() {
  local key="$1" value="$2"
  python - "${ROOT_DIR}/.env" "${key}" "${value}" <<'PY'
from pathlib import Path
import shlex
import sys

path, key, value = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
lines = path.read_text(encoding="utf-8").splitlines()
replacement = f"{key}={shlex.quote(value)}"
for index, line in enumerate(lines):
    if line.startswith(f"{key}="):
        lines[index] = replacement
        break
else:
    lines.append(replacement)
path.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
}

create_python_env() {
  local name="$1"
  if ! env_exists "${name}"; then
    conda create -n "${name}" python=3.12 -y
  fi
  conda run -n "${name}" python -m pip install -U pip setuptools wheel
}

create_yaml_env() {
  local name="$1" yaml="$2"
  if env_exists "${name}"; then
    conda env update -n "${name}" -f "${yaml}" --prune
  else
    conda env create -f "${yaml}"
  fi
}

install_gpu_backend() {
  local name="$1"
  if [[ "${SKIP_GPU_BACKEND:-0}" != "1" ]]; then
    conda run -n "${name}" python -m pip install "sglang[all]==${SGLANG_VERSION}"
  fi
  conda run -n "${name}" python -m pip install \
    flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir
}

install_megatron() {
  local name="$1"
  if [[ ! -d "${MEGATRON_DIR}/.git" ]]; then
    command -v git >/dev/null 2>&1 || {
      echo "git is required to install Megatron-LM" >&2
      exit 1
    }
    mkdir -p "$(dirname -- "${MEGATRON_DIR}")"
    git clone https://github.com/NVIDIA/Megatron-LM.git "${MEGATRON_DIR}"
  fi
  git -C "${MEGATRON_DIR}" fetch --depth 1 origin "${MEGATRON_COMMIT}"
  git -C "${MEGATRON_DIR}" checkout --detach "${MEGATRON_COMMIT}"
  git -C "${MEGATRON_DIR}" submodule update --init --recursive
  conda run -n "${name}" python -m pip install -e "${MEGATRON_DIR}"
}

install_webshop_vendor() {
  local checkout
  checkout="${ROOT_DIR}/third_party/Skill0_5"
  if [[ ! -d "${checkout}/.git" ]]; then
    git clone https://github.com/JasonZhujp/Skill0_5.git "${checkout}"
  fi
  git -C "${checkout}" fetch --depth 1 origin "${WEBSHOP_COMMIT}"
  git -C "${checkout}" checkout --detach "${WEBSHOP_COMMIT}"
  test -s "${WEBSHOP_VENDOR_ROOT}/web_agent_site/engine/engine.py"
}

install_tau2_bench() {
  local name="$1"
  if [[ ! -d "${TAU2_BENCH_DIR}/.git" ]]; then
    mkdir -p "$(dirname -- "${TAU2_BENCH_DIR}")"
    git clone https://github.com/sierra-research/tau2-bench.git "${TAU2_BENCH_DIR}"
  fi
  git -C "${TAU2_BENCH_DIR}" fetch --depth 1 origin "${TAU2_BENCH_COMMIT}"
  git -C "${TAU2_BENCH_DIR}" checkout --detach "${TAU2_BENCH_COMMIT}"
  conda run -n "${name}" python -m pip install -e "${TAU2_BENCH_DIR}" --no-deps
}

install_main() {
  local name="skillscaffold-main"
  create_python_env "${name}"
  conda run -n "${name}" python -m pip install -r "${ROOT_DIR}/requirements.txt"
  install_gpu_backend "${name}"
  conda run -n "${name}" python -m pip install \
    -r "${ROOT_DIR}/framework/slime_legacy/requirements.txt"
  install_megatron "${name}"
  echo "Installed ${name} for ALFWorld, WebShop, and ScienceWorld training."
}

install_tau() {
  local name="skillscaffold-tau"
  create_yaml_env "${name}" "${ROOT_DIR}/environments/tau2.yaml"
  conda run -n "${name}" python -m pip install -U pip setuptools wheel
  conda run -n "${name}" python -m pip install -r "${ROOT_DIR}/requirements.txt"
  install_gpu_backend "${name}"
  conda run -n "${name}" python -m pip install \
    -r "${ROOT_DIR}/framework/slime_tau/requirements.txt"
  install_megatron "${name}"
  install_tau2_bench "${name}"
  echo "Installed ${name} for tau2-bench training."

  name="skillscaffold-tau-eval"
  create_yaml_env "${name}" "${ROOT_DIR}/environments/tau2_eval.yaml"
  conda run -n "${name}" python -m pip install -U pip setuptools wheel
  conda run -n "${name}" python -m pip install -r "${ROOT_DIR}/requirements.txt"
  install_tau2_bench "${name}"
  echo "Installed ${name} for tau2-bench evaluation."
}

install_task_env() {
  local target="$1" name="skillscaffold-$1"
  create_yaml_env "${name}" "${ROOT_DIR}/environments/${target}.yaml"
  echo "Installed ${name}."
}

case "${TARGET}" in
  main) install_main ;;
  tau) install_tau ;;
  alfworld|webshop|scienceworld)
    install_main
    install_task_env "${TARGET}"
    if [[ "${TARGET}" == "webshop" ]]; then install_webshop_vendor; fi
    ;;
  all)
    install_main
    install_task_env alfworld
    install_task_env webshop
    install_task_env scienceworld
    install_webshop_vendor
    ;;
  *)
    echo "unknown target: ${TARGET}" >&2
    usage
    exit 2
    ;;
esac

if [[ ! -e "${ROOT_DIR}/.env" ]]; then
  cp "${ROOT_DIR}/.env.example" "${ROOT_DIR}/.env"
  echo "Created ${ROOT_DIR}/.env; edit its model and data paths before training."
fi

set_env_value MEGATRON_DIR "${MEGATRON_DIR}"
set_env_value MEGATRON_LM_DIR "${MEGATRON_DIR}"
set_env_value WEBSHOP_VENDOR_ROOT "${WEBSHOP_VENDOR_ROOT}"
set_env_value WEBSHOP_DATA_DIR "${WEBSHOP_VENDOR_ROOT}/data"
set_env_value WEBSHOP_SPLITS "${ROOT_DIR}/benchmarks/webshop/protocol/splits.json"

case "${TARGET}" in
  main)
    set_env_value PYTHON_BIN "$(env_prefix skillscaffold-main)/bin/python"
    ;;
  tau)
    tau_prefix="$(env_prefix skillscaffold-tau)"
    tau_eval_prefix="$(env_prefix skillscaffold-tau-eval)"
    set_env_value TAU2_PYTHON "${tau_prefix}/bin/python"
    set_env_value TAU2_EVAL_PYTHON "${tau_eval_prefix}/bin/python"
    set_env_value VLLM_BIN "${tau_eval_prefix}/bin/vllm"
    set_env_value TAU2_BENCH_DIR "${TAU2_BENCH_DIR}"
    ;;
  alfworld)
    set_env_value PYTHON_BIN "$(env_prefix skillscaffold-main)/bin/python"
    set_env_value ALFWORLD_PYTHON "$(env_prefix skillscaffold-alfworld)/bin/python"
    ;;
  webshop)
    set_env_value PYTHON_BIN "$(env_prefix skillscaffold-main)/bin/python"
    set_env_value WEBSHOP_PYTHON "$(env_prefix skillscaffold-webshop)/bin/python"
    ;;
  scienceworld)
    scienceworld_prefix="$(env_prefix skillscaffold-scienceworld)"
    set_env_value PYTHON_BIN "$(env_prefix skillscaffold-main)/bin/python"
    set_env_value SCIENCEWORLD_PYTHON "${scienceworld_prefix}/bin/python"
    set_env_value JAVA_HOME "${scienceworld_prefix}"
    ;;
  all)
    scienceworld_prefix="$(env_prefix skillscaffold-scienceworld)"
    set_env_value PYTHON_BIN "$(env_prefix skillscaffold-main)/bin/python"
    set_env_value ALFWORLD_PYTHON "$(env_prefix skillscaffold-alfworld)/bin/python"
    set_env_value WEBSHOP_PYTHON "$(env_prefix skillscaffold-webshop)/bin/python"
    set_env_value SCIENCEWORLD_PYTHON "${scienceworld_prefix}/bin/python"
    set_env_value JAVA_HOME "${scienceworld_prefix}"
    ;;
esac

if [[ "${TARGET}" == "alfworld" || "${TARGET}" == "all" ]]; then
  ALFWORLD_CONFIG="${ROOT_DIR}/benchmarks/alfworld/configs/alfworld.yaml"
  if [[ ! -e "${ALFWORLD_CONFIG}" ]]; then
    cp "${ALFWORLD_CONFIG}.example" "${ALFWORLD_CONFIG}"
  fi
  python - "${ALFWORLD_CONFIG}" "${ALFWORLD_TASK_FILES_ROOT}" <<'PY'
from pathlib import Path
import sys

path, task_files_root = Path(sys.argv[1]), Path(sys.argv[2]).resolve()
text = path.read_text(encoding="utf-8")
text = text.replace("REPLACE_WITH_ALFWORLD_ROOT/task_files", str(task_files_root))
path.write_text(text, encoding="utf-8")
PY
  set_env_value ALFWORLD_CONFIG "${ALFWORLD_CONFIG}"
  set_env_value ALFWORLD_DATA_ROOT "${ALFWORLD_TASK_FILES_ROOT}/json_2.1.1"
  set_env_value ALFWORLD_PARQUET_DIR "${ALFWORLD_PARQUET_DIR}"
  set_env_value ALFWORLD_CACHE_FILE "${ALFWORLD_PARQUET_DIR}/cache/unified_embeddings.npz"
fi
