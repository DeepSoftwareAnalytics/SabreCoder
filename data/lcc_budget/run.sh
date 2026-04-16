#!/usr/bin/env bash
set -euo pipefail

# One-click runner for LCC budgeted prompt buckets.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"
python "${SCRIPT_DIR}/run.py" "$@"

