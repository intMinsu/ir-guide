#!/usr/bin/env bash
# safe_set_pytorch_conda.sh
# JetPack 5.1.x (Py3.8, system CUDA). Creates a conda env, installs Jetson PyTorch wheel,
# wires env to system CUDA + OpenCV, optionally installs soft-clamp hooks (activate/deactivate),
# optionally builds torchvision and torchaudio (hard-clamped unless --no-clamp), and verifies the stack.
#
# Enhancements:
#  - Prints elapsed seconds for torchvision/torchaudio build steps
#  - Uses Ninja by default (CMAKE_GENERATOR=Ninja) w/ progress via NINJA_STATUS
#  - Interactive build-speed selector (safe/balanced/fast/custom/auto)
#
# Usage tips:
#   SPEED_PROFILE=safe|balanced|fast|auto|custom:N ./safe_set_pytorch_conda.sh
#   USE_NINJA=0 ./safe_set_pytorch_conda.sh          # force Unix Makefiles
#   ./safe_set_pytorch_conda.sh --no-tv --no-ta      # skip building TV/TA
#   MAX_JOBS/CMAKE_BUILD_PARALLEL_LEVEL can still override if you want.

set -euo pipefail

echo "=== PyTorch (Jetson) + Conda env setup (clamp-aware) ==="

# ------------------------------------------------------------------------------
# Flags / env knobs
# ------------------------------------------------------------------------------
SKIP_TV="${SKIP_TV:-0}"             # 1 to skip torchvision build
SKIP_TA="${SKIP_TA:-0}"             # 1 to skip torchaudio build
CLAMP_MODE="${CLAMP_MODE:-soft}"    # soft (default) | none
if [[ "${1:-}" == "--no-tv" ]]; then SKIP_TV=1; shift; fi
if [[ "${1:-}" == "--no-ta" ]]; then SKIP_TA=1; shift; fi
if [[ "${1:-}" == "--no-clamp" ]]; then CLAMP_MODE=none; shift; fi

# Control whether soft clamp appends old LD_LIBRARY_PATH (only when CLAMP_MODE=soft)
JETSON_APPEND_OLD_LD_DEFAULT="${JETSON_APPEND_OLD_LD:-1}"

# Default to Ninja unless explicitly disabled or overridden by CMAKE_GENERATOR
USE_NINJA="${USE_NINJA:-1}"
if [[ "${USE_NINJA}" == "1" && -z "${CMAKE_GENERATOR-}" ]]; then
  export CMAKE_GENERATOR="Ninja"
fi
# Ninja progress & elapsed seconds
if [[ "${CMAKE_GENERATOR-}" == "Ninja" ]]; then
  export NINJA_STATUS="${NINJA_STATUS:-[%r tasks/s | %f/%t | %es elapsed]}"
fi

echo "[i] CLAMP_MODE=${CLAMP_MODE}   (soft=hooks+clean build, none=no hooks, plain build)"
echo "[i] SKIP_TV=${SKIP_TV}"
echo "[i] SKIP_TA=${SKIP_TA}"
echo "[i] CMAKE_GENERATOR=${CMAKE_GENERATOR-<default>}"

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
# Utilities + Build-speed selector
# ------------------------------------------------------------------------------
min() { [ "$1" -le "$2" ] && echo "$1" || echo "$2"; }
detect_mem_gb() { awk '/MemTotal/{printf "%.1f", $2/1024/1024}' /proc/meminfo 2>/dev/null || echo "?"; }
detect_cpus() { nproc --all 2>/dev/null || nproc 2>/dev/null || echo 1; }

choose_build_speed() {
  local cpus; cpus="$(detect_cpus)"
  local mem;  mem="$(detect_mem_gb)"
  local choice jobs tv_video=0 use_ninja="${USE_NINJA:-1}"

  # SPEED_PROFILE env overrides the prompt.
  # Accepts: safe | balanced | fast | auto | custom:<N>
  if [[ -n "${SPEED_PROFILE:-}" ]]; then
    case "${SPEED_PROFILE}" in
      safe|SAFE)          jobs=2 ; tv_video=0 ;;
      balanced|BALANCED)  jobs=4 ; tv_video=0 ;;
      fast|FAST)
        jobs=6
        if [[ "$mem" != "?" ]]; then
          mem_int=${mem%.*}
          (( mem_int <= 8 )) && jobs=2
        fi
        ;;
      auto|AUTO)
        if [[ "$mem" != "?" ]]; then
          mem_int=${mem%.*}
          if   (( mem_int <= 5 ));  then jobs=2
          elif (( mem_int <= 12 )); then jobs=4
          else                           jobs=6
          fi
        else
          jobs=2
        fi
        ;;
      custom:*)
        jobs="${SPEED_PROFILE#custom:}"
        ;;
      *)
        echo "[!] Unknown SPEED_PROFILE='${SPEED_PROFILE}', falling back to prompt."
        ;;
    esac
  fi

  if [[ -z "${jobs:-}" ]]; then
    echo
    echo "== Build speed profiles =="
    echo "  [1] Safe      (2 job)  — 4–8GB devices; minimizes OOM risk; disables tv video"
    echo "  [2] Balanced  (4 jobs) — 8GB devices; good speed/ram balance; disables tv video"
    echo "  [3] Fast      (6 jobs) — 16GB+ devices; faster if RAM allows"
    echo "  [4] Custom    (set your own jobs)"
    echo "      Detected: ${cpus} CPU cores, ~${mem} GiB RAM"
    read -rp "Choose 1/2/3/4: " choice || choice="1"
    case "${choice}" in
      1) jobs=2 ; tv_video=0 ;;
      2) jobs=4 ; tv_video=0 ;;
      3) jobs=6 ;;
      4)
        read -rp "Jobs (1-${cpus}): " jobs
        jobs="${jobs:-1}"
        ;;
      *) jobs=1 ; tv_video=0 ;;
    esac
  fi

  # Clamp to CPU count
  [[ "${jobs}" -lt 1 ]] && jobs=1
  if [[ "${cpus}" =~ ^[0-9]+$ ]] && [[ "${jobs}" -gt "${cpus}" ]]; then jobs="${cpus}"; fi

  export MAX_JOBS="${jobs}"
  export CMAKE_BUILD_PARALLEL_LEVEL="${jobs}"
  export USE_NINJA="${use_ninja}"

  # Safe/Balanced disable torchvision video to shrink build
  if [[ "${tv_video}" == "0" ]]; then
    export TORCHVISION_USE_FFMPEG="${TORCHVISION_USE_FFMPEG:-0}"
    export TORCHVISION_USE_VIDEO_CODEC="${TORCHVISION_USE_VIDEO_CODEC:-0}"
  fi

  echo "[i] Build speed: jobs=${jobs}, generator=${CMAKE_GENERATOR-<default>}, Ninja=${USE_NINJA}, tv_video_disabled=${tv_video}"
  echo "[i] Detected resources: ${cpus} cores, ~${mem} GiB RAM"
}

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
echo "  [1] 2.1.0  (wheel: jp/v512) -> torchvision 0.16.1, torchaudio 2.1.2"
echo "  [2] 2.0.0  (wheel: box.com ) -> torchvision 0.15.1, torchaudio 2.0.2"
echo "  [3] 1.14.0 (wheel: jp/v51  ) -> torchvision 0.14.1, torchaudio 0.13.1"
read -rp "Enter 1/2/3: " TORCH_CHOICE
case "${TORCH_CHOICE}" in
  1)
    TORCH_URL="https://developer.download.nvidia.cn/compute/redist/jp/v512/pytorch/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl"
    TV_VERSION="0.16.1"
    TA_VERSION="2.1.2"
    ;;
  2)
    TORCH_URL="https://nvidia.box.com/shared/static/i8pukc49h3lhak4kkn67tg9j4goqm0m7.whl"
    TV_VERSION="0.15.1"
    TA_VERSION="2.0.2"
    ;;
  3)
    TORCH_URL="https://developer.download.nvidia.com/compute/redist/jp/v51/pytorch/torch-1.14.0a0+44dac51c.nv23.02-cp38-cp38-linux_aarch64.whl"
    TV_VERSION="0.14.1"
    TA_VERSION="0.13.1"
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
# Select build speed profile (sets MAX_JOBS/CMAKE_BUILD_PARALLEL_LEVEL/USE_NINJA and optional tv flags)
# ------------------------------------------------------------------------------
choose_build_speed

# ------------------------------------------------------------------------------
# Hard-clamp helper: with_clean_env (no-op if CLAMP_MODE=none)
# ------------------------------------------------------------------------------
with_clean_env() {
  if [[ "${CLAMP_MODE}" == "none" ]]; then
    "$@"
  else
    export PYTHONPATH="$(python -c 'import site; print(\":\".join(site.getsitepackages()))')":${PYTHONPATH:-}
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
      MAX_JOBS="${MAX_JOBS:-1}" \
      CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-1}" \
      USE_NINJA="${USE_NINJA:-1}" \
      CMAKE_GENERATOR="${CMAKE_GENERATOR:-Ninja}" \
      NINJA_STATUS="${NINJA_STATUS-}" \
      TORCHVISION_USE_FFMPEG="${TORCHVISION_USE_FFMPEG:-0}" \
      TORCHVISION_USE_VIDEO_CODEC="${TORCHVISION_USE_VIDEO_CODEC:-0}" \
      CMAKE_MAKE_PROGRAM="${CMAKE_MAKE_PROGRAM-}" \
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

CUDA_ROOT="/usr/local/cuda"
JETSON_LIBS="/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu/tegra"
ENV_LIB="\${CONDA_PREFIX}/lib"
ENV_BIN="\${CONDA_PREFIX}/bin"

export CUDA_HOME="\$CUDA_ROOT"
export CUDA_PATH="\$CUDA_ROOT"

SYS_PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export PATH="\$CUDA_ROOT/bin:\$ENV_BIN:\$SYS_PATH"

CLEAN_LD="\$JETSON_LIBS:\$ENV_LIB"
if [[ "\${JETSON_APPEND_OLD_LD:-$JETSON_APPEND_OLD_LD_DEFAULT}" == "1" && -n "\${LD_LIBRARY_PATH-}" ]]; then
  CLEAN_LD="\$CLEAN_LD:\$LD_LIBRARY_PATH"
fi
export LD_LIBRARY_PATH="\$CLEAN_LD"

[[ -d /usr/local/lib/cmake/opencv4 ]] && export OpenCV_DIR="/usr/local/lib/cmake/opencv4"
export CMAKE_PREFIX_PATH="/usr/local:\${CMAKE_PREFIX_PATH-}"
export PKG_CONFIG_PATH="/usr/local/lib/pkgconfig:\${PKG_CONFIG_PATH-}"

export PYTHONNOUSERSITE=1
unset PYTHONPATH 2>/dev/null || true

export OPENBLAS_NUM_THREADS=1

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
# Expose system TensorRT & PyCUDA into this env (via .pth)
# ------------------------------------------------------------------------------

echo "[*] Bridging system packages (cv2, jtop, smbus2, tensorrt, pycuda) into the env…"

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
import importlib, pathlib
mods = ("cv2","jtop","smbus2","tensorrt","pycuda")
for m in mods:
    try:
        mod = importlib.import_module(m)
        p = pathlib.Path(getattr(mod, "__file__", "")) if hasattr(mod,"__file__") else None
        print(f"{m}|{str(p.resolve()) if p else ''}")
    except Exception:
        print(f"{m}|")
PY
)

# Fallback search paths (adjust pythonX.Y if needed)
CANDS=(
  "/usr/lib/python3.8/dist-packages"
  "/usr/local/lib/python3.8/dist-packages"
  "/usr/lib/python3/dist-packages"
  "/usr/local/lib/python3/dist-packages"
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
        jtop|smbus2|tensorrt|pycuda)
          if [[ -d "$base/$name" ]]; then origin="$base/$name/__init__.py"; break; fi
          # special-case TRT: some distros place the .so next to __init__.py
          if [[ "$name" == "tensorrt" ]]; then
            found_trt=$(ls "$base"/tensorrt/tensorrt*.so 2>/dev/null | head -n1 || true)
            [[ -n "$found_trt" ]] && origin="$base/tensorrt/__init__.py" && break
          fi
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
    jtop|smbus2|tensorrt|pycuda)
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
  echo "[!] None of (cv2, jtop, smbus2, tensorrt, pycuda) found; no bridge created."
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
# Ask whether to build torchaudio
# ------------------------------------------------------------------------------
if [[ "$SKIP_TA" -eq 0 ]]; then
  read -rp "Build torchaudio v${TA_VERSION}? [Y/n]: " _ans || true
  if [[ "${_ans:-}" =~ ^[Nn]$ ]]; then SKIP_TA=1; fi
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
  # Tooling
  python -m pip install --upgrade "pip<25" "setuptools<75" "wheel<0.45" "packaging<24.2" cmake ninja

  # Respect chosen generator
  if [[ "${USE_NINJA}" != "1" ]]; then export CMAKE_GENERATOR="Unix Makefiles"; fi

  # Device arch hint for CUDA extensions
  MODEL=$(tr -d '\0' < /proc/device-tree/model 2>/dev/null || echo "")
  if [[ "$MODEL" == *"Xavier"* ]]; then
    export TORCH_CUDA_ARCH_LIST="7.2"
  elif [[ "$MODEL" == *"Orin"* ]]; then
    export TORCH_CUDA_ARCH_LIST="8.7"
  else
    export TORCH_CUDA_ARCH_LIST="7.2"
  fi

  # Expose CUDA for the build
  export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu/tegra:${LD_LIBRARY_PATH:-}"
  export FORCE_CUDA=1
  export BUILD_VERSION="${TV_VERSION}"

  pushd "$HOME" >/dev/null
  rm -rf torchvision
  git clone --branch "v${TV_VERSION}" https://github.com/pytorch/vision torchvision
  cd torchvision

  TV_T0=$(date +%s)
  if [[ "${CLAMP_MODE}" == "none" ]]; then
    python -m pip install --no-build-isolation -v .
  else
    with_clean_env python -m pip install --no-build-isolation -v .
  fi
  TV_T1=$(date +%s)
  echo "[i] torchvision build elapsed: $((TV_T1-TV_T0))s"

  popd >/dev/null
else
  echo "[i] Skipping torchvision build (flag or user choice)."
fi

# ------------------------------------------------------------------------------
# Build torchaudio
# ------------------------------------------------------------------------------
if [[ "$SKIP_TA" -eq 0 ]]; then
  echo "[*] Building torchaudio v${TA_VERSION} (${CLAMP_MODE} mode)…"

  sudo apt-get update
  sudo apt-get install -y \
    build-essential git cmake ninja-build \
    sox libsox-dev libsox-fmt-all \
    libsndfile1-dev libflac-dev libvorbis-dev libopus-dev libmp3lame-dev

  python -m pip install --upgrade "pip<25" "setuptools<75" "wheel<0.45" "packaging<24.2" cmake ninja

  if ! command -v ninja >/dev/null 2>&1; then
    if command -v ninja-build >/dev/null 2>&1; then
      sudo ln -sf "$(command -v ninja-build)" /usr/local/bin/ninja
    else
      python -m pip install -U ninja
    fi
  fi

  # Help CMake find it explicitly if needed
  export CMAKE_MAKE_PROGRAM="$(command -v ninja || command -v ninja-build || true)"

  if [[ "${USE_NINJA}" != "1" ]]; then export CMAKE_GENERATOR="Unix Makefiles"; fi

  pushd "$HOME" >/dev/null
  rm -rf torchaudio
  git clone --branch "v${TA_VERSION}" https://github.com/pytorch/audio torchaudio
  cd torchaudio

  TA_T0=$(date +%s)
  if [[ "${CLAMP_MODE}" == "none" ]]; then
    python -m pip install --no-build-isolation -v .
  else
    with_clean_env python -m pip install --no-build-isolation -v .
  fi
  TA_T1=$(date +%s)
  echo "[i] torchaudio build elapsed: $((TA_T1-TA_T0))s"

  popd >/dev/null
else
  echo "[i] Skipping torchaudio build (flag or user choice)."
fi

# ------------------------------------------------------------------------------
# Verification
# ------------------------------------------------------------------------------
echo "[*] Verifying torch/torchvision/torchaudio/OpenCV & CUDA access…"
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

try:
    import torchaudio as ta
    print("torchaudio     :", ta.__version__)
except Exception as e:
    print("torchaudio     : <not installed>", type(e).__name__)

import cv2
print("cv2            :", cv2.__version__)
try:
    print("cv2 CUDA count :", cv2.cuda.getCudaEnabledDeviceCount())
except Exception as e:
    print("cv2 CUDA check :", "<failed>", type(e).__name__)
PY

echo "=== Done. Activate with:  conda activate ${ENV_NAME}"
echo "    Skip tv/audio:   ./safe_set_pytorch_conda.sh --no-tv --no-ta"
echo "    No clamps:       ./safe_set_pytorch_conda.sh --no-clamp    (no hooks; plain build env)"
echo "    Speed preset:    SPEED_PROFILE=safe|balanced|fast|auto ./safe_set_pytorch_conda.sh"
echo "    Custom jobs:     SPEED_PROFILE=custom:3 ./safe_set_pytorch_conda.sh"
