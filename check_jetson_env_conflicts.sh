#!/usr/bin/env bash
# check_jetson_env_conflicts.sh
# Checks common Jetson + Conda conflicts and prints suggested fixes.
# Run:
#   bash check_jetson_env_conflicts.sh
# Or for a specific env:
#   conda run -n <env> bash check_jetson_env_conflicts.sh

set -euo pipefail

check_jetson_env_conflicts() {
  echo "=== Jetson/Conda conflict checker ==="
  local ERR=0

  # 0) Basic env info
  echo "[i] CONDA_PREFIX: ${CONDA_PREFIX:-<none>}"
  python - <<'PY' || true
import sys
print("[i] Python      :", ".".join(map(str, sys.version_info[:3])))
PY

  # 1) Accidental CUDA/CUDNN pulled into conda (JetPack should supply CUDA)
  if command -v conda >/dev/null 2>&1; then
    if conda list 2>/dev/null | grep -Eiq '(^|\s)(cuda(|toolkit)|cudnn|cublas|cusolver|cufft|cutensor|pytorch-cuda)(\s|=)'; then
      echo "[!] Found CUDA/cuDNN-like packages in this conda env (conflicts with JetPack)."
      echo "    -> Fix: conda remove -n ${CONDA_DEFAULT_ENV:-<env>} --force cudnn cudatoolkit cublas cusolver cufft cutensor pytorch-cuda"
      ((ERR++))
    else
      echo "[OK] No conda CUDA/cuDNN packages detected."
    fi
  else
    echo "[i] conda not found on PATH; skipping conda package check."
  fi

  # 2) Accidental pip OpenCV wheels (override your CUDA OpenCV)
  if python -m pip list --format=columns 2>/dev/null | grep -Eiq '^opencv-(python|python-headless|contrib-python)\s'; then
    echo "[!] Found pip OpenCV wheels that can override your /usr/local CUDA build."
    echo "    -> Fix: pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python"
    ((ERR++))
  else
    echo "[OK] No pip OpenCV wheels detected."
  fi

  # 3) Where does cv2 come from? Is it CUDA-enabled?
  python - <<'PY'
import sys
try:
    import cv2
    src = cv2.__file__
    info = cv2.getBuildInformation() if hasattr(cv2, "getBuildInformation") else ""
    has_cuda = ("CUDA: YES" in info) or bool(getattr(cv2, "cuda", None))
    devs = cv2.cuda.getCudaEnabledDeviceCount() if getattr(cv2, "cuda", None) else -1
    print(f"[OK] cv2 path     : {src}")
    print(f"[OK] cv2 CUDA     : {'YES' if has_cuda else 'NO'}")
    print(f"[OK] cv2 CUDA devs: {devs}")
    if not has_cuda:
        print("[!] Suggestion   : Non-CUDA OpenCV in use. Ensure your env has a .pth pointing to the /usr/local .../site-packages that contains CUDA-built cv2, and remove pip wheels that shadow it.")
except Exception as e:
    print(f"[!] cv2 import failed: {e}")
    print("    -> Fix: verify your /usr/local OpenCV install and add a .pth in this env that points to its site-packages (e.g., opencv_local.pth).")
    sys.exit(1)
PY
  [[ $? -ne 0 ]] && ((ERR++))

  # 4) Torch/Torchvision compatibility (mapping-based) + CUDA availability
  python - <<'PY'
import sys
from packaging.version import Version

def compat(torch_ver, tv_ver):
    t = Version(torch_ver.split('+')[0])  # strip +nv...
    v = Version(tv_ver)
    MAP = {
      (1,0): [(0,2)], (1,1): [(0,3)], (1,2): [(0,4)], (1,3): [(0,4,2)],
      (1,4): [(0,5)], (1,5): [(0,6)], (1,6): [(0,7)], (1,7): [(0,8,1)],
      (1,8): [(0,9)], (1,9): [(0,10)], (1,10): [(0,11,1)], (1,11): [(0,12)],
      (1,12): [(0,13)], (1,13): [(0,13)], (1,14): [(0,14,1)],
      (2,0): [(0,15,1)], (2,1): [(0,16,0), (0,16,1)], (2,2): [(0,17,1)],
      (2,3): [(0,18,0)],
    }
    allowed = MAP.get((t.major, t.minor), [])
    for mv in allowed:
        if len(mv)==2 and (v.major, v.minor)==(mv[0], mv[1]): return True
        if len(mv)==3 and (v.major, v.minor, v.micro)==(mv[0], mv[1], mv[2]): return True
    return False

try:
    import torch, torchvision as tv
    print(f"[OK] torch        : {torch.__version__} | CUDA: {torch.cuda.is_available()}")
    print(f"[OK] torchvision  : {tv.__version__}")
    if not compat(torch.__version__, tv.__version__):
        print("[!] torch/torchvision not on an approved mapping (per docs).")
        print("    -> Fix: install torchvision tag matching your torch per the mapping (e.g., torch 2.1.x ↔ tv 0.16.x).")
        sys.exit(3)
except ModuleNotFoundError as e:
    if e.name == "torchvision":
        print("[i] torchvision   : <not installed> (OK if you chose to skip)")
    else:
        print(f"[!] Module missing: {e.name}")
        sys.exit(2)
except Exception as e:
    print(f"[!] torch/vision check failed: {e}")
    sys.exit(2)
PY
  case $? in
    2|3) ((ERR++));;
  esac

  # 5) CUDA_HOME & nvcc on PATH (use system JetPack)
  if [[ -z "${CUDA_HOME:-}" || ! -x "${CUDA_HOME:-}/bin/nvcc" ]]; then
    echo "[!] CUDA_HOME not set or nvcc not found."
    echo "    -> Fix: export CUDA_HOME=/usr/local/cuda && export PATH=\$CUDA_HOME/bin:\$PATH"
    ((ERR++))
  else
    echo "[OK] CUDA_HOME=${CUDA_HOME} (nvcc present)"
  fi

  # 6) LD_LIBRARY_PATH includes Jetson libs
  case ":${LD_LIBRARY_PATH:-}:" in
    *:/usr/lib/aarch64-linux-gnu/tegra:*)  echo "[OK] tegra libs visible via LD_LIBRARY_PATH";;
    *)
      echo "[!] tegra libs not in LD_LIBRARY_PATH."
      echo "    -> Fix: export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu/tegra:\${LD_LIBRARY_PATH:-}"
      ((ERR++));;
  esac

  # 7) Tooling versions safe for Python 3.8 (avoid too-new build isolation deps)
  python - <<'PY'
import sys
py38 = sys.version_info[:2]==(3,8)
def ver_or_none(pkg):
    try:
        m=__import__(pkg); return getattr(m, "__version__", None)
    except Exception:
        return None
pipv   = ver_or_none("pip") or "0"
setuv  = ver_or_none("setuptools") or "0"
def major(v):
    try: return int((v.split(".")[0]).split("+")[0])
    except: return 0
if py38:
    bad = False
    if major(pipv) >= 25:
        print(f"[!] pip={pipv} may be too new for Py3.8 build isolation.")
        print("    -> Fix: python -m pip install 'pip<25'")
        bad = True
    if major(setuv) >= 75:
        print(f"[!] setuptools={setuv} may be too new for Py3.8.")
        print("    -> Fix: python -m pip install 'setuptools<75'")
        bad = True
    if not bad:
        print(f"[OK] pip={pipv}, setuptools={setuv} look safe for Py3.8.")
else:
    print("[i] Python is not 3.8 — ensure toolchain matches your version/JetPack.")
PY
  [[ $? -ne 0 ]] && ((ERR++))

  # Summary + exit status
  if [[ $ERR -eq 0 ]]; then
    echo "=== No obvious conflicts found. ✅"
  else
    echo "=== Detected $ERR potential issue(s). See suggestions above. ⚠️"
  fi

  (( ERR > 255 )) && ERR=255
  return $ERR
}

# If executed (not sourced), run the checker
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  check_jetson_env_conflicts
  exit $?
fi
