#!/usr/bin/env bash
# Run C1 training with the CUDA libraries pinned inside its isolated venv.

set -euo pipefail

C1_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
C1_TRAIN_ENV="${C1_TRAIN_VENV:-$C1_REPO_ROOT/.venv-c1-train}"
C1_PYTHON="$C1_TRAIN_ENV/bin/python"

if [[ ! -x "$C1_PYTHON" ]]; then
  echo "C1 training environment is missing: $C1_TRAIN_ENV" >&2
  exit 2
fi

C1_SITE_PACKAGES="$($C1_PYTHON -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
C1_CUDA_LIBS=()
while IFS= read -r C1_LIB_DIR; do
  C1_CUDA_LIBS+=("$C1_LIB_DIR")
done < <(find "$C1_SITE_PACKAGES/nvidia" -mindepth 2 -maxdepth 2 -type d -name lib | sort)
C1_CUDA_LIBS+=("$C1_SITE_PACKAGES/torch/lib")

C1_JOINED_LIBS="$(IFS=:; echo "${C1_CUDA_LIBS[*]}")"
export LD_LIBRARY_PATH="$C1_JOINED_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONDONTWRITEBYTECODE=1

exec "$C1_PYTHON" "$@"
