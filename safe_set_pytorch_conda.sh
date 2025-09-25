#!/usr/bin/env bash
# safe_set_pytorch_conda.sh
# JP 5.1.5 (Python 3.8, CUDA from JetPack). Creates a conda env, installs a Jetson PyTorch wheel,
# wires env to system CUDA + OpenCV, optionally builds matching torchvision, and verifies everything.

set -euo pipefail

echo "=== PyTorch (Jetson) + Conda env setup ==="

# ---- optional: skip torchvision via flag/env ----
SKIP_TV="${SKIP_TV:-0}"
if [[ "${1:-}" == "--no-tv" ]]; then
  SKIP_TV=1
fi

# ----- 0) Require Miniforge/conda -----
if [[ ! -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]]; then
  cat <<'MSG'
Please install miniforge first.
Example :
>> wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
>> bash Miniforge3-Linux-aarch64.sh -b -p $HOME/miniforge3
>> source "$HOME/miniforge3/etc/profile.d/conda.sh"
>> conda activate
MSG
  exit 1
fi
# shellcheck disable=SC1091
source "$HOME/miniforge3/etc/profile.d/conda.sh" || { echo "Could not source conda.sh"; exit 1; }
conda --version >/dev/null || { echo "conda not found in PATH even after sourcing."; exit 1; }

# ----- 1) Env name -----
read -rp "Conda env name (e.g., jp515): " ENV_NAME
if [[ -z "${ENV_NAME}" ]]; then
  echo "Env name is required."; exit 1
fi
if conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  echo "Conda env '${ENV_NAME}' already exists. Choose another name."; exit 1
fi

# ----- 2) PyTorch version selection -----
echo "Choose PyTorch version:"
echo "  [1] 2.1.0  (wheel: jp/v512)"
echo "  [2] 2.0.0  (wheel: box.com link)"
echo "  [3] 1.14.0 (wheel: jp/v51)"
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
  *)
    echo "Invalid choice."; exit 1;;
esac

# ----- 3) Create and activate env (Python 3.8 for Jetson wheels) -----
echo "[*] Creating conda env '${ENV_NAME}' (python=3.8)…"
conda create -y -n "${ENV_NAME}" python=3.8 pip
conda activate "${ENV_NAME}"

# ----- 4) Make the env use system CUDA/nvcc & Jetson libs -----
echo "[*] Wiring env to system CUDA/nvcc & Jetson libs…"
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d" "$CONDA_PREFIX/etc/conda/deactivate.d"

cat > "$CONDA_PREFIX/etc/conda/activate.d/jetson_paths.sh" <<'EOT'
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
# Make L4T (Jetson) libs visible inside env
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu/tegra:${LD_LIBRARY_PATH:-}
EOT

cat > "$CONDA_PREFIX/etc/conda/deactivate.d/jetson_paths.sh" <<'EOT'
# minimal PATH cleanup on deactivate
case ":$PATH:" in
  *:/usr/local/cuda/bin:*) PATH="${PATH//\/usr\/local\/cuda\/bin:/}";;
esac
export PATH
EOT

# reload activation for current shell
conda deactivate
conda activate "${ENV_NAME}"

# quick nvcc sanity (won't fail the whole script if absent)
if ! command -v nvcc >/dev/null; then
  echo "[!] nvcc not found in PATH. Ensure JetPack CUDA is installed. Continuing…"
else
  nvcc --version | tail -n1 || true
fi

# ----- 5) Install PyTorch wheel -----
echo "[*] Installing PyTorch from: ${TORCH_URL}"
# Py3.8-safe toolchain (avoid too-new pip/setuptools/wheel)
python -m pip install --upgrade "pip<25" "setuptools<75" "wheel<0.45"
# numpy first keeps wheels happy on some combos
python -m pip install "numpy<2"
python -m pip install --no-cache-dir "${TORCH_URL}"

# ----- 6) Make env see system OpenCV-CUDA -----
echo "[*] Exposing system OpenCV (CUDA build) to the env…"
PY_SITE=$(python - <<'PY'
import site; print(site.getsitepackages()[0])
PY
)
OPENCV_PTH="${PY_SITE}/opencv_local.pth"

# Try to discover cv2 via *system* python, then fall back to known dirs
SYS_PY=$(command -v python3 || echo /usr/bin/python3)
SYS_CV2_SITE=$($SYS_PY - <<'PY'
import os
from pathlib import Path
try:
    import cv2
    p = Path(cv2.__file__).resolve()
    q = p
    while q.name not in ("site-packages","dist-packages") and q.parent != q:
        q = q.parent
    print(q if q.name in ("site-packages","dist-packages") else p.parent)
except Exception:
    print("")
PY
)

: > "$OPENCV_PTH"
FOUND_CV=0
if [[ -n "$SYS_CV2_SITE" && -d "$SYS_CV2_SITE" ]]; then
  echo "$SYS_CV2_SITE" >> "$OPENCV_PTH"
  echo "[+] Linked system OpenCV at: $SYS_CV2_SITE"
  FOUND_CV=1
else
  # Candidate locations where cv2*.so might live (add your successful path)
  CANDIDATES=(
    "/usr/local/lib/python3.8/dist-packages"
    "/usr/local/lib/python3.8/site-packages"
    "/usr/local/python"
    "/usr/lib/python3/dist-packages"
  )
  for d in "${CANDIDATES[@]}"; do
    if ls "${d}"/cv2*.so >/dev/null 2>&1 || [[ -d "${d}/cv2" ]]; then
      echo "$d" >> "$OPENCV_PTH"
      echo "[+] Linked candidate OpenCV at: $d"
      FOUND_CV=1
    fi
  done
fi

if [[ "$FOUND_CV" -eq 0 ]]; then
  echo "[!] Could not auto-locate system OpenCV; edit $OPENCV_PTH to point to the directory containing 'cv2'."
  echo "[i] Hint (outside conda): python3 -c 'import cv2, pathlib; print(pathlib.Path(cv2.__file__).resolve())'"
fi

# ----- 6.5) Ask whether to build torchvision -----
if [[ "$SKIP_TV" -eq 0 ]]; then
  read -rp "Build torchvision v${TV_VERSION}? [Y/n]: " _ans || true
  if [[ "${_ans:-}" =~ ^[Nn]$ ]]; then
    SKIP_TV=1
  fi
fi

# ----- 7) Build torchvision that matches the selected torch -----
if [[ "$SKIP_TV" -eq 0 ]]; then
  echo "[*] Building torchvision v${TV_VERSION}…"
  sudo apt-get update
  sudo apt-get install -y \
    build-essential git libjpeg-dev zlib1g-dev libpython3-dev \
    libopenblas-dev libavcodec-dev libavformat-dev libswscale-dev

  # Toolchain compatible with Python 3.8
  python -m pip install --upgrade "pip<25" "setuptools<75" "wheel<0.45" "packaging<24.2" cmake ninja

  # Pick arch list for faster build (Xavier NX = sm_72; Orin = sm_87)
  MODEL=$(tr -d '\0' < /proc/device-tree/model 2>/dev/null || echo "")
  if [[ "$MODEL" == *"Xavier"* ]]; then
    export TORCH_CUDA_ARCH_LIST="7.2"
  elif [[ "$MODEL" == *"Orin"* ]]; then
    export TORCH_CUDA_ARCH_LIST="8.7"
  else
    export TORCH_CUDA_ARCH_LIST="7.2"   # default safe-ish
  fi

  # Ensure CUDA vars for the build
  export CUDA_HOME=/usr/local/cuda
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu/tegra:${LD_LIBRARY_PATH:-}"
  export FORCE_CUDA=1
  export BUILD_VERSION="${TV_VERSION}"

  pushd "$HOME" >/dev/null
  rm -rf torchvision
  git clone --branch "v${TV_VERSION}" https://github.com/pytorch/vision torchvision
  cd torchvision

  # Use current env (no build isolation) to avoid too-new setuptools
  python -m pip install --no-build-isolation -v .
  popd >/dev/null
else
  echo "[i] Skipping torchvision build (flag or user choice)."
fi

# ----- 8) Verification -----
echo "[*] Verifying torch/torchvision/OpenCV & CUDA access…"
python - <<'PY'
import os, torch, cv2, subprocess
print("CUDA_HOME      :", os.environ.get("CUDA_HOME"))
try:
    nv = subprocess.check_output(["nvcc","--version"]).decode().strip().splitlines()[-1]
except Exception:
    nv = "<nvcc not found>"
print("nvcc           :", nv)
print("torch          :", torch.__version__, "cuda?", torch.cuda.is_available(), "devices:", torch.cuda.device_count())
try:
    import torchvision as tv
    print("torchvision    :", tv.__version__)
except Exception as e:
    print("torchvision    : <not installed>", type(e).__name__)
print("cv2            :", cv2.__version__)
try:
    print("cv2 CUDA count :", cv2.cuda.getCudaEnabledDeviceCount())
except Exception as e:
    print("cv2 CUDA check :", "<failed>", type(e).__name__)
PY

echo "=== Done. Activate with:  conda activate ${ENV_NAME}"
echo "    Re-run with: ./safe_set_pytorch_conda.sh --no-tv   # to skip torchvision"
