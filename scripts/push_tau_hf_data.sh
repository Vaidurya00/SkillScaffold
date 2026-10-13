#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <hf-org-or-user/repo> <prepared-data-directory>" >&2
  exit 2
fi

REPO_ID="$1"
DATA_DIR="$2"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v hf >/dev/null 2>&1; then
  echo "The 'hf' CLI is missing. Install it with: python -m pip install -U huggingface_hub" >&2
  exit 2
fi

python "${SCRIPT_DIR}/audit_tau_hf_data.py" --root "${DATA_DIR}"
hf upload "${REPO_ID}" "${DATA_DIR}" . \
  --repo-type dataset \
  --commit-message "Release SkillScaffold Tau RL training data"
