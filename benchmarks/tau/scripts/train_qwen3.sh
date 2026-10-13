#!/bin/bash
# Qwen3-4B / Qwen3-8B Tau2 GRPO entry point.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

export MODEL_FAMILY=qwen3
export WANDB_NAME="${WANDB_NAME:-qwen3_4b_tau2_grpo}"

exec bash "${SCRIPT_DIR}/train_engine.sh" "$@"
