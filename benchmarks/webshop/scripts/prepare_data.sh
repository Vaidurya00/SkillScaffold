#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${ROOT_DIR}/../.." >/dev/null 2>&1 && pwd)"
WEBSHOP_PYTHON="${WEBSHOP_PYTHON:-python3}"
WEBSHOP_VENDOR_ROOT="${WEBSHOP_VENDOR_ROOT:-${REPO_ROOT}/third_party/Skill0_5/agent_system/environments/env_package/webshop/webshop}"
WEBSHOP_DATA_DIR="${WEBSHOP_DATA_DIR:-${WEBSHOP_VENDOR_ROOT}/data}"
SPLITS="${WEBSHOP_SPLITS:-${ROOT_DIR}/protocol/splits.json}"

"${WEBSHOP_PYTHON}" "${ROOT_DIR}/env/webshop/prepare_data.py" \
  --data-dir "${WEBSHOP_DATA_DIR}" \
  --vendor-root "${WEBSHOP_VENDOR_ROOT}" \
  --splits "${SPLITS}" \
  --output-dir "${ROOT_DIR}/data/webshop"
