#!/usr/bin/env bash
# safe_set_pytorch_conda.sh
# JP 5.1.5 (Python 3.8, CUDA from JetPack). Creates a conda env, installs a Jetson PyTorch wheel,
# wires env to system CUDA + OpenCV, builds matching torchvision, and verifies everything.

set -euo pipefail

echo "=== PyTorch (Jetson) + Conda env setup ==="

# ----- 0) Require Miniforge/conda -----
if [[ ! -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]]; then
  cat <<'MSG'
Please install miniforge first.
Example :
>> wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
>> bash Miniforge3-Linux-aarch64.sh -b -p $HOME/miniforge3
>> source "$HOME/miniforge3/etc/profile.d/conda.sh
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
python -m pip install --upgrade pip wheel setuptools
# numpy first keeps wheels happy on some combos
python -m pip install "numpy==1.26.1" || python -m pip install numpy
python -m pip install --no-cache-dir "${TORCH_URL}"

# ----- 6) Make env see system OpenCV-CUDA -----
echo "[*] Exposing system OpenCV (CUDA build) to the env…"
PY_SITE=$(python - <<'PY'
import site; print(site.getsitepackages()[0])
PY
)
# Candidate locations where cv2*.so might live
CANDIDATES=(
  "/usr/local/lib/python3.8/site-packages"
  "/usr/local/python"
  "/usr/lib/python3/dist-packages"
)
OPENCV_PTH="${PY_SITE}/opencv_local.pth"
: > "$OPENCV_PTH"
FOUND_CV=0
for d in "${CANDIDATES[@]}"; do
  if ls "${d}"/cv2*.so >/dev/null 2>&1 || [[ -d "${d}/cv2" ]]; then
    echo "$d" >> "$OPENCV_PTH"
    FOUND_CV=1
  fi
done
if [[ "$FOUND_CV" -eq 0 ]]; then
  echo "[!] Could not find a system OpenCV in common locations; if you built to /usr/local, adjust $OPENCV_PTH manually."
fi

# ----- 7) Build torchvision that matches the selected torch -----
echo "[*] Building torchvision v${TV_VERSION} (this can take a while)…"
sudo apt-get update
sudo apt-get install -y \
  build-essential git libjpeg-dev zlib1g-dev libpython3-dev \
  libopenblas-dev libavcodec-dev libavformat-dev libswscale-dev

# Pick arch list for faster build (Xavier NX = sm_72; Orin = sm_87)
MODEL=$(tr -d '\0' < /proc/device-tree/model 2>/dev/null || echo "")
if [[ "$MODEL" == *"Xavier"* ]]; then
  export TORCH_CUDA_ARCH_LIST="7.2"
elif [[ "$MODEL" == *"Orin"* ]]; then
  export TORCH_CUDA_ARCH_LIST="8.7"
else
  export TORCH_CUDA_ARCH_LIST="7.2"   # default safe-ish
fi
export FORCE_CUDA=1
export BUILD_VERSION="${TV_VERSION}"

pushd "$HOME" >/dev/null
rm -rf torchvision
git clone --branch "v${TV_VERSION}" https://github.com/pytorch/vision torchvision
cd torchvision
# Prefer pip install . (builds against currently active torch)
python -m pip install --no-cache-dir -v .
popd >/dev/null

# ----- 8) Verification -----
echo "[*] Verifying torch/torchvision/OpenCV & CUDA access…"
python - <<'PY'
import subprocess, sys
import torch, torchvision, cv2
print("torch        :", torch.__version__, "CUDA available:", torch.cuda.is_available(), "device count:", torch.cuda.device_count())
try:
    print("torchvision  :", torchvision.__version__)
except Exception as e:
    print("torchvision  : import failed ->", e)
print("cv2          :", cv2.__version__, "CUDA devices:", cv2.cuda.getCudaEnabledDeviceCount())
try:
    out = subprocess.check_output(['nvcc','--version']).decode().strip().splitlines()[-1]
    print("nvcc         :", out)
except Exception as e:
    print("nvcc         : not found in PATH or failed ->", e)
PY

echo "=== Done. Activate with:  conda activate ${ENV_NAME}"