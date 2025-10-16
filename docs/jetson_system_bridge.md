[ [English](#english) | [한국어](#한국어) ]
# Jetson System Bridge — `jetson_system_bridge.sh`
**Up one level:** [`../README.md`](../README.md)
> Location: `ir-guide/jetson_system_bridge.sh`

---

## English

### What this does (in one line)
Makes your **conda env** import **JetPack’s system Python packages** first — notably **`cv2` (CUDA OpenCV)**, **`tensorrt`**, **`pycuda`**, **`jtop`**, **`smbus2`** — **without** copying them into the env.

### Why you need it (Jetson)
- JetPack ships CUDA‑enabled **OpenCV/TensorRT** under `/usr` and `/usr/local`.  
- Pip wheels (e.g., `opencv-python(-headless)`) inside a conda env can **shadow** these, leading to **CPU‑only cv2** or ABI mismatches.  
- This bridge enforces **system packages first** by placing a **.pth** entry at **`sys.path[0]`**.

### How it works
- Probes **system Python** to locate: `cv2`, `jtop`, `smbus2`, `tensorrt`, `pycuda`.  
- Builds a small **bridge directory**:  
  `"$CONDA_PREFIX/share/jetson-python-bridge"` with **symlinks** to each system module.  
- Writes `jetson_system_bridge.pth` into the env’s **site‑packages**, containing:  
  1) a literal path line (adds the bridge to `sys.path`)  
  2) a short Python snippet that **moves the bridge path to the front** (`sys.path[0]`).  
- Idempotent: safe to run multiple times; only updates symlinks and a `.pth` file.

### When to run it
- After creating a new conda env on Jetson.  
- After installing/upgrading anything related to OpenCV/TensorRT/PyCUDA.  
- Whenever `check_jetson_env_conflicts.sh` warns about **OpenCV wheels** or **non‑CUDA cv2**.

### Usage
```bash
# (1) Activate the env
conda activate <env>

# (2) Run the bridge
bash jetson_system_bridge.sh
```

### Verify
```bash
python - <<'PY'
import os, sys
print("sys.path[0]        :", sys.path[0])
import cv2
print("cv2.__file__ (real):", os.path.realpath(getattr(cv2,"__file__","")))
info = cv2.getBuildInformation() if hasattr(cv2, "getBuildInformation") else ""
print("cv2 CUDA           :", ("CUDA: YES" in info) or bool(getattr(cv2,"cuda",None)))
try:
    import tensorrt as trt; print("tensorrt version   :", getattr(trt, "__version__", "n/a"))
except Exception as e:
    print("tensorrt import    :", f"failed ({e}) — ensure JetPack TRT Python is installed")
PY
```

Expected:
- `sys.path[0]` points to your **bridge directory**.  
- `cv2` resolves to a path under **`/usr/.../dist-packages`** and **CUDA: YES**.  
- Optional: `tensorrt` imports if installed on the system.

### Undo / disable
```bash
# Remove the .pth and bridge folder (per-env)
PY_SITE=$(python - <<'PY'
import site; print(site.getsitepackages()[0])
PY
)
rm -f "$PY_SITE/jetson_system_bridge.pth"
rm -rf "$CONDA_PREFIX/share/jetson-python-bridge"
# Optionally, reinstall wheels if you prefer pip cv2 (not recommended on Jetson)
```

### Notes & limitations
- The bridge **does not install** missing packages; it **exposes** already-installed system ones.  
- If your system uses another Python minor (e.g., 3.10), adjust the fallback search paths in the script.  
- You may still need to export **`CUDA_HOME=/usr/local/cuda`** and extend **`LD_LIBRARY_PATH`** with `.../tegra`.  
- Works best with **JetPack 5.1.x**; other releases may use different paths.  
- Pair with **`check_jetson_env_conflicts.sh`** to catch lingering issues.

### FAQ
- **Q. Will this break pip `opencv-python`?**  
  A. No. It simply **prioritizes** the system path ahead of wheels. Remove the `.pth` to restore default order.
- **Q. Multiple envs?**  
  A. Each env gets its **own** `.pth` and bridge folder.
- **Q. Can I bridge more modules?**  
  A. Yes — add your module to the probe list and create a symlink into the bridge.

---

## 한국어

### 한 줄 요약
conda 환경이 **JetPack 시스템 Python 패키지**(**CUDA OpenCV `cv2`**, **`tensorrt`**, **`pycuda`**, **`jtop`**, **`smbus2`**)를 **우선** 가져오도록 설정합니다. **복사/설치 없이** 경로만 앞세웁니다.

### 필요한 이유
- JetPack은 `/usr`, `/usr/local` 아래에 **CUDA 지원 OpenCV/TensorRT**를 제공합니다.  
- conda env 내부의 pip 휠(`opencv-python(-headless)` 등)이 이를 **가려서** **CPU‑only cv2**나 ABI 충돌이 납니다.  
- 본 브리지는 **.pth** 파일을 이용해 시스템 패키지를 **우선 경로**(sys.path[0])로 만듭니다.

### 동작 방식
- 시스템 Python으로 `cv2`, `jtop`, `smbus2`, `tensorrt`, `pycuda` 경로를 탐색합니다.  
- `"$CONDA_PREFIX/share/jetson-python-bridge"` 아래에 각 모듈로 **심볼릭 링크**를 만듭니다.  
- env의 **site‑packages**에 `jetson_system_bridge.pth`를 작성하여, 브리지 경로를 **sys.path 맨 앞**으로 이동시킵니다.  
- **멱등성**: 여러 번 실행해도 안전합니다.

### 언제 실행하나
- Jetson에서 새 conda env를 만든 직후.  
- OpenCV/TensorRT/PyCUDA 관련 패키지를 설치/업그레이드한 뒤.  
- `check_jetson_env_conflicts.sh`가 **OpenCV 휠** 또는 **비‑CUDA cv2**를 경고할 때.

### 사용법
```bash
conda activate <env>
bash jetson_system_bridge.sh
```

### 확인
```bash
python - <<'PY'
import os, sys
print("sys.path[0]        :", sys.path[0])
import cv2
print("cv2.__file__ (real):", os.path.realpath(getattr(cv2,"__file__","")))
info = cv2.getBuildInformation() if hasattr(cv2, "getBuildInformation") else ""
print("cv2 CUDA           :", ("CUDA: YES" in info) or bool(getattr(cv2,"cuda",None)))
try:
    import tensorrt as trt; print("tensorrt version   :", getattr(trt, "__version__", "n/a"))
except Exception as e:
    print("tensorrt import    :", f"failed ({e}) — JetPack TRT Python 설치 확인 필요")
PY
```

### 해제/원복
```bash
PY_SITE=$(python - <<'PY'
import site; print(site.getsitepackages()[0])
PY
)
rm -f "$PY_SITE/jetson_system_bridge.pth"
rm -rf "$CONDA_PREFIX/share/jetson-python-bridge"
```

### 메모 & 제한사항
- 브리지는 **설치**가 아니라 **경로 노출**입니다(시스템에 없는 패키지는 불가).  
- Python 마이너 버전이 다르면(예: 3.10) 스크립트의 폴백 경로를 조정하세요.  
- **`CUDA_HOME=/usr/local/cuda`**, **`LD_LIBRARY_PATH=.../tegra`** 설정이 여전히 필요할 수 있습니다.  
- **JetPack 5.1.x** 기준으로 테스트되었습니다.  
- 남은 문제는 **`check_jetson_env_conflicts.sh`**로 점검하세요.