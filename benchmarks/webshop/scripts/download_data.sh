#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${ROOT_DIR}/../.." >/dev/null 2>&1 && pwd)"
WEBSHOP_PYTHON="${WEBSHOP_PYTHON:-python3}"
WEBSHOP_VENDOR_ROOT="${WEBSHOP_VENDOR_ROOT:-${REPO_ROOT}/third_party/Skill0_5/agent_system/environments/env_package/webshop/webshop}"
WEBSHOP_DATA_DIR="${WEBSHOP_DATA_DIR:-${WEBSHOP_VENDOR_ROOT}/data}"

mkdir -p "${WEBSHOP_DATA_DIR}"

download() {
  local filename="$1" file_id="$2"
  if [[ -s "${WEBSHOP_DATA_DIR}/${filename}" ]]; then
    echo "[webshop/data] ${filename} already exists"
    return
  fi
  "${WEBSHOP_PYTHON}" -m gdown \
    "https://drive.google.com/uc?id=${file_id}" \
    -O "${WEBSHOP_DATA_DIR}/${filename}"
}

# These are the official full WebShop files used by the pinned Skill0.5
# environment. prepare_data.sh deterministically derives the audited 50K
# human-goal catalogue and the 3,320/454/207 protocol splits from them.
download items_shuffle.json 1A2whVgOO0euk5O13n2iYDM0bQRkkRduB
download items_ins_v2.json 1s2j6NgHljiZzQNL3veZaAiyW_qDEgBNi
download items_human_ins.json 14Kb5SPBk_jfdLZ_CDBNitW98QLDlKR5O

echo "[webshop/data] downloaded full catalogue to ${WEBSHOP_DATA_DIR}"
