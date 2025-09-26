#!/usr/bin/env bash
# safe_set_pytorch_conda.sh
# JetPack 5.1.x (Py3.8, system CUDA). Creates a conda env, installs Jetson PyTorch wheel,
# wires env to system CUDA + OpenCV, optionally installs soft-clamp hooks (activate/deactivate),
# optionally builds torchvision (hard-clamped unless --no-clamp), and verifies the stack.

set -euo pipefail

echo "=== PyTorch (Jetson) + Conda env setup (clamp-aware) ==="

# ------------------------------------------------------------------------------
# Flags / env knobs
# ------------------------------------------------------------------------------
SKIP_TV="${SKIP_TV:-0}"             # 1 to skip torchvision build
CLAMP_MODE="${CLAMP_MODE:-soft}"    # soft (default) | none
if [[ "${1:-}" == "--no-tv" ]]; then SKIP_TV=1; shift; fi
if [[ "${1:-}" == "--no-clamp" ]]; then CLAMP_MODE=none; shift; fi

# Control whether soft clamp appends old LD_LIBRARY_PATH (only when CLAMP_MODE=soft)
JETSON_APPEND_OLD_LD_DEFAULT="${JETSON_APPEND_OLD_LD:-1}"

echo "[i] CLAMP_MODE=${CLAMP_MODE}   (soft=hooks+clean build, none=no hooks, plain build)"
echo "[i] SKIP_TV=${SKIP_TV}"

# ------------------------------------------------------------------------------
# Require Miniforge/conda
# ------------------------------------------------------------------------------
if [[ ! -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]]; then
  cat <<'MSG'
Please install miniforge first.
Example :
  wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
  bash Miniforge3-Linux-aarch64.sh -b -p $HOME/miniforge3
  source "$HOME/miniforge3/etc/profile.d/conda.sh"
  conda activate
MSG
  exit 1
fi
# shellcheck disable=SC1091
source "$HOME/miniforge3/etc/profile.d/conda.sh" || { echo "Could not source conda.sh"; exit 1; }
conda --version >/dev/null || { echo "conda not found in PATH even after sourcing."; exit 1; }

# ------------------------------------------------------------------------------
# Env name
# ------------------------------------------------------------------------------
read -rp "Conda env name (e.g., jp515): " ENV_NAME
if [[ -z "${ENV_NAME}" ]]; then echo "Env name is required."; exit 1; fi
if conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  echo "Conda env '${ENV_NAME}' already exists. Choose another name."; exit 1
fi

# ------------------------------------------------------------------------------
# PyTorch version selection
# ------------------------------------------------------------------------------
echo "Choose PyTorch version:"
echo "  [1] 2.1.0  (wheel: jp/v512) -> torchvision 0.16.1"
echo "  [2] 2.0.0  (wheel: box.com ) -> torchvision 0.15.1"
echo "  [3] 1.14.0 (wheel: jp/v51  ) -> torchvision 0.14.1"
read -rp "Enter 1/2/3: " TORCH_CHOICE
case "${TORCH_CHOICE}" in
  1)
    TORCH_URL="https://developer.download.nvidia.cn/compute/redist/jp/v512/pytorch/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl"
    TV_VERSION="0.16.1"
    ;;
  2)
    TORCH_URL="https://nvidia.box.com/shared/static/i8pukc49h3lhak4kkn67tg9j4goqm0m7.whl"
    TV_VERSION="0.15.1"
    ;;
  3)
    TORCH_URL="https://developer.download.nvidia.com/compute/redist/jp/v51/pytorch/torch-1.14.0a0+44dac51c.nv23.02-cp38-cp38-linux_aarch64.whl"
    TV_VERSION="0.14.1"
    ;;
  *) echo "Invalid choice."; exit 1;;
esac

# ------------------------------------------------------------------------------
# Create and activate env (Python 3.8)
# ------------------------------------------------------------------------------
echo "[*] Creating conda env '${ENV_NAME}' (python=3.8)…"
conda create -y -n "${ENV_NAME}" python=3.8 pip
conda activate "${ENV_NAME}"

# ------------------------------------------------------------------------------
# Hard-clamp helper: with_clean_env (no-op if CLAMP_MODE=none)
# ------------------------------------------------------------------------------
with_clean_env() {
  if [[ "${CLAMP_MODE}" == "none" ]]; then
    # No hard clamp: run in current environment
    "$@"
  else
    export PYTHONPATH="$(python -c 'import site; print(":".join(site.getsitepackages()))')":${PYTHONPATH:-}
    # Hermetic command run: only whitelisted variables are kept
    env -i \
      HOME="$HOME" USER="$USER" SHELL="${SHELL:-/bin/bash}" TERM="${TERM:-xterm}" \
      CONDA_PREFIX="${CONDA_PREFIX:-}" \
      PATH="/usr/local/cuda/bin${CONDA_PREFIX:+:$CONDA_PREFIX/bin}:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
      CUDA_HOME="/usr/local/cuda" CUDA_PATH="/usr/local/cuda" \
      LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu/tegra${CONDA_PREFIX:+:$CONDA_PREFIX/lib}" \
      OpenCV_DIR="/usr/local/lib/cmake/opencv4" \
      CMAKE_PREFIX_PATH="/usr/local" PKG_CONFIG_PATH="/usr/local/lib/pkgconfig" \
      PYTHONPATH="${PYTHONPATH:-}" \
      PYTHONNOUSERSITE=1 \
      "$@"
  fi
}

# ------------------------------------------------------------------------------
# Soft clamp hooks (activate/deactivate) unless CLAMP_MODE=none
# ------------------------------------------------------------------------------
HOOK_ACT="$CONDA_PREFIX/etc/conda/activate.d/10-jetson_harden.sh"
HOOK_DEACT="$CONDA_PREFIX/etc/conda/deactivate.d/10-jetson_harden.sh"

if [[ "${CLAMP_MODE}" == "soft" ]]; then
  echo "[*] Installing soft-clamp env hooks (activate/deactivate)…"
  mkdir -p "$CONDA_PREFIX/etc/conda/activate.d" "$CONDA_PREFIX/etc/conda/deactivate.d"

  # Activate hook
  cat > "$HOOK_ACT" <<EOT
#!/usr/bin/env bash
# Soft clamp: prefer JetPack's CUDA + system libs when this env is active.

# Save originals
export __OLD_CUDA_HOME="\${CUDA_HOME-}"
export __OLD_CUDA_PATH="\${CUDA_PATH-}"
export __OLD_PATH="\$PATH"
export __OLD_LD_LIBRARY_PATH="\${LD_LIBRARY_PATH-}"
export __OLD_OPENBLAS_NUM_THREADS="\${OPENBLAS_NUM_THREADS-}"
export __OLD_PYTHONNOUSERSITE="\${PYTHONNOUSERSITE-}"
export __OLD_OpenCV_DIR="\${OpenCV_DIR-}"
export __OLD_CMAKE_PREFIX_PATH="\${CMAKE_PREFIX_PATH-}"
export __OLD_PKG_CONFIG_PATH="\${PKG_CONFIG_PATH-}"
export __OLD_PYTHONPATH="\${PYTHONPATH-}"

# Canonical Jetson paths
CUDA_ROOT="/usr/local/cuda"
JETSON_LIBS="/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu/tegra"
ENV_LIB="\${CONDA_PREFIX}/lib"
ENV_BIN="\${CONDA_PREFIX}/bin"

# CUDA anchors
export CUDA_HOME="\$CUDA_ROOT"
export CUDA_PATH="\$CUDA_ROOT"

# PATH: Construct a sane path order
# Start with system essentials to prevent commands from going missing.
SYS_PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
# Prepend CUDA and the Conda env bin for priority.
export PATH="\$CUDA_ROOT/bin:\$ENV_BIN:\$SYS_PATH"


# LD_LIBRARY_PATH clamp (append prior entries controlled by JETSON_APPEND_OLD_LD)
CLEAN_LD="\$JETSON_LIBS:\$ENV_LIB"
if [[ "\${JETSON_APPEND_OLD_LD:-$JETSON_APPEND_OLD_LD_DEFAULT}" == "1" && -n "\${LD_LIBRARY_PATH-}" ]]; then
  CLEAN_LD="\$CLEAN_LD:\$LD_LIBRARY_PATH"
fi
export LD_LIBRARY_PATH="\$CLEAN_LD"

# Tool hints
[[ -d /usr/local/lib/cmake/opencv4 ]] && export OpenCV_DIR="/usr/local/lib/cmake/opencv4"
export CMAKE_PREFIX_PATH="/usr/local:\${CMAKE_PREFIX_PATH-}"
export PKG_CONFIG_PATH="/usr/local/lib/pkgconfig:\${PKG_CONFIG_PATH-}"

# Avoid user site & stray PYTHONPATH
export PYTHONNOUSERSITE=1
unset PYTHONPATH 2>/dev/null || true

# Threading sanity (optional)
export OPENBLAS_NUM_THREADS=1

# Speed up CUDA extension builds (auto-arch)
MODEL="\$(tr -d '\0' < /proc/device-tree/model 2>/dev/null || echo "")"
if [[ "\$MODEL" == *"Xavier"* ]]; then
  export TORCH_CUDA_ARCH_LIST="7.2"
elif [[ "\$MODEL" == *"Orin"* ]]; then
  export TORCH_CUDA_ARCH_LIST="8.7"
fi
EOT
  chmod +x "$HOOK_ACT"

  # Deactivate hook
  cat > "$HOOK_DEACT" <<'EOT'
#!/usr/bin/env bash
# Restore originals saved in activate hook.

export CUDA_HOME="${__OLD_CUDA_HOME-}"
export CUDA_PATH="${__OLD_CUDA_PATH-}"
export PATH="${__OLD_PATH-}"
export LD_LIBRARY_PATH="${__OLD_LD_LIBRARY_PATH-}"
export OPENBLAS_NUM_THREADS="${__OLD_OPENBLAS_NUM_THREADS-}"
export PYTHONNOUSERSITE="${__OLD_PYTHONNOUSERSITE-}"
export OpenCV_DIR="${__OLD_OpenCV_DIR-}"
export CMAKE_PREFIX_PATH="${__OLD_CMAKE_PREFIX_PATH-}"
export PKG_CONFIG_PATH="${__OLD_PKG_CONFIG_PATH-}"
export PYTHONPATH="${__OLD_PYTHONPATH-}"

unset __OLD_CUDA_HOME __OLD_CUDA_PATH __OLD_PATH __OLD_LD_LIBRARY_PATH \
      __OLD_OPENBLAS_NUM_THREADS __OLD_PYTHONNOUSERSITE __OLD_OpenCV_DIR \
      __OLD_CMAKE_PREFIX_PATH __OLD_PKG_CONFIG_PATH __OLD_PYTHONPATH \
      TORCH_CUDA_ARCH_LIST
EOT
  chmod +x "$HOOK_DEACT"

  # Re-source to immediately apply soft clamp in current shell
  conda deactivate
  conda activate "${ENV_NAME}"

else
  # No clamp: if hooks from a previous run exist, remove them to ensure "no clamp"
  if [[ -f "$HOOK_ACT" || -f "$HOOK_DEACT" ]]; then
    echo "[i] Removing existing clamp hooks to honor --no-clamp…"
    rm -f "$HOOK_ACT" "$HOOK_DEACT" || true
  fi
fi

# ------------------------------------------------------------------------------
# nvcc sanity (non-fatal)
# ------------------------------------------------------------------------------
if ! command -v nvcc >/dev/null; then
  echo "[!] nvcc not found in PATH. Ensure JetPack CUDA is installed. Continuing…"
else
  nvcc --version | tail -n1 || true
fi

# ------------------------------------------------------------------------------
# Install PyTorch wheel (Py3.8-safe toolchain)
# ------------------------------------------------------------------------------
echo "[*] Installing PyTorch from: ${TORCH_URL}"
python -m pip install --upgrade "pip<25" "setuptools<75" "wheel<0.45"
python -m pip install "numpy<2"
python -m pip install --no-cache-dir "${TORCH_URL}"

# ------------------------------------------------------------------------------
# Expose system OpenCV (CUDA build) into this env (via .pth)
# ------------------------------------------------------------------------------
# ----- Bridge only selected system packages (cv2, jtop, smbus2) into this env -----
echo "[*] Bridging system packages (cv2, jtop, smbus2) into the env…"

# Env site-packages + bridge targets
PY_SITE=$(python - <<'PY'
import site; print(site.getsitepackages()[0])
PY
)
BRIDGE_DIR="$CONDA_PREFIX/share/jetson-python-bridge"
BRIDGE_PTH="${PY_SITE}/jetson_system_bridge.pth"
mkdir -p "$BRIDGE_DIR"

# Prefer real system python (not the conda one)
if [[ -x /usr/bin/python3 ]]; then
  SYS_PY=/usr/bin/python3
elif [[ -x /usr/local/bin/python3 ]]; then
  SYS_PY=/usr/local/bin/python3
else
  SYS_PY=$(command -v python3 || echo python3)
fi
echo "[i] System python for probing: $SYS_PY"

# Try clean env import probe with system python
readarray -t _LOCS < <( env -i \
  PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  "$SYS_PY" -s -E - <<'PY'
import importlib, pathlib, sys
mods = ("cv2","jtop","smbus2")
for m in mods:
    try:
        mod = importlib.import_module(m)
        p = pathlib.Path(getattr(mod, "__file__", "")) if hasattr(mod,"__file__") else None
        print(f"{m}|{str(p.resolve()) if p else ''}")
    except Exception:
        print(f"{m}|")
PY
)

# Fallback search paths if import probe fails (JetPack 5.x = Py3.8)
CANDS=(
  "/usr/local/lib/python3.8/dist-packages"
  "/usr/local/lib/python3/dist-packages"
  "/usr/lib/python3/dist-packages"
  "/usr/local/python"
)

: > "$BRIDGE_PTH"
FOUND_ANY=0
for line in "${_LOCS[@]}"; do
  name="${line%%|*}"
  origin="${line#*|}"

  # If probe failed, scan common system dirs
  if [[ -z "$origin" ]]; then
    for base in "${CANDS[@]}"; do
      case "$name" in
        cv2)
          if [[ -d "$base/cv2" ]]; then origin="$base/cv2/__init__.py"; break; fi
          found_so=$(ls "$base"/cv2.*.so 2>/dev/null | head -n1 || true)
          [[ -n "$found_so" ]] && origin="$found_so" && break
          ;;
        jtop|smbus2)
          if [[ -d "$base/$name" ]]; then origin="$base/$name/__init__.py"; break; fi
          ;;
      esac
    done
  fi

  if [[ -z "$origin" ]]; then
    echo "[!] System module not found: $name (skipping)"
    continue
  fi

  # Create symlinks into the bridge
  case "$name" in
    cv2)
      if [[ "$origin" == */__init__.py ]]; then
        ln -sfn "$(dirname "$origin")" "$BRIDGE_DIR/cv2"
      else
        ln -sfn "$origin" "$BRIDGE_DIR/$(basename "$origin")"
      fi
      ;;
    jtop|smbus2)
      ln -sfn "$(dirname "$origin")" "$BRIDGE_DIR/$name"
      ;;
  esac

  echo "[+] bridged $name <- $origin"
  FOUND_ANY=1
done

if [[ "$FOUND_ANY" -eq 1 ]]; then
  echo "$BRIDGE_DIR" > "$BRIDGE_PTH"
  echo "[i] Bridge path written to: $BRIDGE_PTH"
else
  echo "[!] None of (cv2, jtop, smbus2) found; no bridge created."
  echo "    If they’re installed under a different prefix, add that path manually to $BRIDGE_PTH"
fi

# ------------------------------------------------------------------------------
# Ask whether to build torchvision
# ------------------------------------------------------------------------------
if [[ "$SKIP_TV" -eq 0 ]]; then
  read -rp "Build torchvision v${TV_VERSION}? [Y/n]: " _ans || true
  if [[ "${_ans:-}" =~ ^[Nn]$ ]]; then SKIP_TV=1; fi
fi

# ------------------------------------------------------------------------------
# Build torchvision
# ------------------------------------------------------------------------------
if [[ "$SKIP_TV" -eq 0 ]]; then
  echo "[*] Building torchvision v${TV_VERSION} (${CLAMP_MODE} mode)…"

  sudo apt-get update
  sudo apt-get install -y \
  build-essential git libjpeg-dev zlib1g-dev libpython3-dev \
  libopenblas-dev libavcodec-dev libavformat-dev libswscale-dev
  # Py3.8-safe build tooling
  python -m pip install --upgrade "pip<25" "setuptools<75" "wheel<0.45" "packaging<24.2" cmake ninja

  # Device arch hint for CUDA extensions (provide anyway; harmless if CPU-only)
  MODEL=$(tr -d '\0' < /proc/device-tree/model 2>/dev/null || echo "")
  if [[ "$MODEL" == *"Xavier"* ]]; then
    export TORCH_CUDA_ARCH_LIST="7.2"
  elif [[ "$MODEL" == *"Orin"* ]]; then
    export TORCH_CUDA_ARCH_LIST="8.7"
  else
    export TORCH_CUDA_ARCH_LIST="7.2"
  fi

  # Expose CUDA for the build (even in no-clamp, we politely try to help)
  export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu/tegra:${LD_LIBRARY_PATH:-}"
  export FORCE_CUDA=1
  export BUILD_VERSION="${TV_VERSION}"

  pushd "$HOME" >/dev/null
  rm -rf torchvision
  git clone --branch "v${TV_VERSION}" https://github.com/pytorch/vision torchvision
  cd torchvision

  if [[ "${CLAMP_MODE}" == "none" ]]; then
    # Plain build in the current environment
    python -m pip install --no-build-isolation -v .
  else
    # Hermetic build
    with_clean_env python -m pip install --no-build-isolation -v .
  fi
  popd >/dev/null
else
  echo "[i] Skipping torchvision build (flag or user choice)."
fi

# ------------------------------------------------------------------------------
# Verification
# ------------------------------------------------------------------------------
echo "[*] Verifying torch/torchvision/OpenCV & CUDA access…"
python - <<'PY'
import os, subprocess
print("CUDA_HOME      :", os.environ.get("CUDA_HOME"))
try:
    nv = subprocess.check_output(["nvcc","--version"]).decode().strip().splitlines()[-1]
except Exception:
    nv = "<nvcc not found>"
print("nvcc           :", nv)

import torch
print("torch          :", torch.__version__,
      "| CUDA available:", torch.cuda.is_available(),
      "| devices:", torch.cuda.device_count())

try:
    import torchvision as tv
    print("torchvision    :", tv.__version__)
except Exception as e:
    print("torchvision    : <not installed>", type(e).__name__)

import cv2
print("cv2            :", cv2.__version__)
try:
    print("cv2 CUDA count :", cv2.cuda.getCudaEnabledDeviceCount())
except Exception as e:
    print("cv2 CUDA check :", "<failed>", type(e).__name__)
PY

echo "=== Done. Activate with:  conda activate ${ENV_NAME}"
echo "    Skip tv:         ./safe_set_pytorch_conda.sh --no-tv"
echo "    No clamps:       ./safe_set_pytorch_conda.sh --no-clamp    (no hooks; plain build env)"
echo "    Env override:    CLAMP_MODE=none SKIP_TV=1 bash safe_set_pytorch_conda.sh"
