#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_DIR="${ROOT_DIR}/data/webshop"
MODEL_NAME="${EMBEDDING_MODEL:?Set EMBEDDING_MODEL to a sentence-transformers-compatible model}"

mkdir -p "${DATA_DIR}/cache"
"${PYTHON_BIN}" "${ROOT_DIR}/env/webshop/retrieve_server/build_unified_cache.py" \
  --parquets \
    "${DATA_DIR}/train.parquet" \
    "${DATA_DIR}/val_id.parquet" \
    "${DATA_DIR}/val_ood.parquet" \
  --output "${DATA_DIR}/cache/unified_embeddings.npz" \
  --model_name "${MODEL_NAME}" \
  --batch_size "${EMBEDDING_BATCH_SIZE:-64}"
