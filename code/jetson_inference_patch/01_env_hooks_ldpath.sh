#!/usr/bin/env bash
set -euo pipefail

: "${CONDA_PREFIX:?Activate your conda env first}"

mkdir -p "$CONDA_PREFIX/etc/conda/activate.d" \
         "$CONDA_PREFIX/etc/conda/deactivate.d"

cat > "$CONDA_PREFIX/etc/conda/activate.d/ldpath.sh" <<'SH'
export _OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH-}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
SH

cat > "$CONDA_PREFIX/etc/conda/deactivate.d/ldpath.sh" <<'SH'
export LD_LIBRARY_PATH="${_OLD_LD_LIBRARY_PATH-}"
unset _OLD_LD_LIBRARY_PATH
SH

echo "[hooks] wrote activate/deactivate LD_LIBRARY_PATH hooks under $CONDA_PREFIX"
