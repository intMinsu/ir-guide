[ [English](#english) | [한국어](#한국어) ]
# jetson_inference_patch
**Up one level:** [`../README.md`](../README.md) 

This folder contains a small set of patches and helper scripts that make **NVIDIA jetson-inference** build & install cleanly **inside a Conda environment** on Jetson (JetPack 5.x / Python 3.8). It keeps all build artifacts isolated under your active Conda env and avoids polluting system paths like `/usr/lib` or `/usr/lib/python3.x/dist-packages`.

---

## English

### Why this is needed
- **jetson-inference**’s upstream CMake defaults tend to install Python modules to the **system** site-packages and link against the system `libpython`.  
- In a **Conda** workflow, Python headers, library, NumPy includes, and the target `site-packages` live under **`$CONDA_PREFIX`**, not the system. Mixing these leads to:
  - installs into `/usr/...` (requires sudo),
  - `ImportError: undefined symbol` from ABI mismatches,
  - or `ldconfig` cache warnings.
- These patches & scripts make CMake **honor the active Conda env**, so everything builds/installs into the env cleanly, with runtime rpaths pointing back to the env’s `lib/`.

### What this changes
1. **CMake (two places)**  
   - Switch Python detection to the interpreter you pass (`-DPYTHON_EXECUTABLE`) and use its **headers/library/site-packages**.  
   - Make NumPy include path pluggable and used when present.  
   - Install both `jetson_utils_python.so` and `jetson_inference_python.so` into the env’s **site-packages**.  
   - Add **RPATH** to `$CONDA_PREFIX/lib` so `LD_LIBRARY_PATH` isn’t required at runtime.
2. **Build script**  
   - Interactively asks for your **jetson-inference repo** path (or accept as CLI arg).  
   - Backs up original CMakeLists/PyNumpy files as `~.bak` and copies patched ones.  
   - Configures with your active Conda Python/NumPy, builds, and installs into the env.  
   - (By default) removes `${REPO}/build` before configuring to avoid stale caches.
3. **Optional env hook**  
   - Adds a Conda `activate.d`/`deactivate.d` script that prepends `$CONDA_PREFIX/lib` to `LD_LIBRARY_PATH`.  
   - RPATH already covers most use cases; the hook is a belt‑and‑suspenders measure (and mirrors Conda’s documented best practices).

> **Note**: We do **not** link against `npymath` (which is often absent on modern NumPy wheels).

### Prerequisites
- Jetson with **JetPack 5.x** (CUDA 11.4 / TensorRT from JetPack).  
- **Python 3.8** Conda env (e.g., `conda create -n test python=3.8 numpy cmake make`).  
- OpenCV (system `/usr/local` or elsewhere) is fine.  
- `git`, `cmake`, `make`, C++ toolchain present.

### 1) Clone jetson-inference
You can clone anywhere (example uses `$HOME`):
```bash
cd ~
git clone --recursive https://github.com/dusty-nv/jetson-inference.git
```

### 2) Run the patch tooling
Activate the **target** Conda env first:
```bash
conda activate <your-env>
```

Scripts are in this folder:
- `00_cleanup_global_jetson.sh`
- `01_env_hooks_ldpath.sh`
- `02_build_install.sh`

**Order:**
1. **Cleanup old global installs** (optional but recommended the first time)
   ```bash
   bash ./00_cleanup_global_jetson.sh
   ```
   Removes prior `jetson_*` libs/modules from system locations to prevent accidental imports from `/usr/...`.

2. **Install Conda activate hooks** (optional)
   ```bash
   bash ./01_env_hooks_ldpath.sh
   ```
   Creates `etc/conda/activate.d/` & `deactivate.d/` scripts that set/unset `LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH`.

3. **Build + install into the env**
   ```bash
   bash ./02_build_install.sh
   ```
   The script will prompt for your **absolute** `jetson-inference` path (or pass it as an argument):
   ```bash
   bash ./02_build_install.sh /absolute/path/to/jetson-inference
   ```
   It backs up and patches two CMakeLists and `PyNumpy.cpp`, deletes `${REPO}/build`, configures against your env’s Python/NumPy, builds, and installs the libraries and Python modules to your env.

### 3) Sanity checks (optional)
A few tiny tests to confirm your build is clean (run inside the env):
```bash
python - <<'PY'
import jetson_utils, jetson_inference
print("OK: imports")
PY

python - <<'PY'
import numpy as np
from jetson_utils import cudaFromNumpy, cudaToNumpy
a = (np.random.rand(8, 8, 3)*255).astype(np.uint8)
cap = cudaFromNumpy(a)
b = cudaToNumpy(cap)
print(a.shape, b.shape)
PY

# binaries present
"$CONDA_PREFIX/bin/imagenet" --help
"$CONDA_PREFIX/bin/detectnet" --help
```

### Where files go
- Libraries:  
  - `$CONDA_PREFIX/lib/libjetson-utils.so`  
  - `$CONDA_PREFIX/lib/libjetson-inference.so`
- Python modules:  
  - `$CONDA_PREFIX/lib/python3.8/site-packages/jetson_utils_python.so`  
  - `$CONDA_PREFIX/lib/python3.8/site-packages/jetson_inference_python.so`  
  - and their packages under `Jetson/*` and `jetson/*`

### Troubleshooting
- **CMake warns** about `CMP0148` (FindPythonInterp/FindPythonLibs) → harmless here.  
- **`ldconfig` permission warning** during `cmake --install` → safe to ignore (we use RPATH).  
- **`ImportError: undefined symbol`** after install → most often a mixed install. Re-run `00_cleanup_global_jetson.sh`, then `02_build_install.sh`.  
- **Headless** systems: some example scripts require display. Use `--headless` where supported.

---

## 한국어

### 왜 필요한가요?
- **jetson-inference** 기본 CMake는 Python 모듈을 종종 **시스템 경로**(예: `/usr/lib/python3.x/dist-packages`)에 설치하고, 시스템 `libpython`에 링크합니다.  
- **Conda** 환경에서는 Python 헤더/라이브러리/NumPy include/설치 경로가 **`$CONDA_PREFIX`** 아래에 있습니다. 시스템과 섞이면:
  - `/usr/...`로 설치(관리자 권한 필요),
  - ABI 불일치로 `ImportError: undefined symbol`,
  - `ldconfig` 경고 등이 발생합니다.
- 본 패치는 **활성화된 Conda 환경**을 기준으로 빌드/설치하도록 바꾸어, 모든 산출물이 해당 환경 아래에 정리되도록 합니다. 실행 시에도 env `lib/`를 찾도록 **RPATH**를 설정합니다.

### 무엇이 변경되나요?
1. **CMake (2곳)**  
   - `-DPYTHON_EXECUTABLE`로 지정한 해석기를 기준으로 헤더/라이브러리/설치 경로를 사용.  
   - NumPy include 경로를 주입하고 존재 시 사용.  
   - Python 모듈을 **Conda env의 site-packages**에 설치.  
   - 런타임 **RPATH**를 `$CONDA_PREFIX/lib`로 설정.
2. **빌드 스크립트**  
   - **jetson-inference 경로**를 인터랙티브로 받거나 인자로 전달.  
   - 원본 CMakeLists/PyNumpy를 `~.bak`으로 백업 후 패치본 복사.  
   - Conda Python/NumPy와 연동해 설정/빌드/설치.  
   - 기본적으로 `${REPO}/build`를 삭제하여 캐시로 인한 오류를 방지.
3. **(선택) 환경 훅**  
   - Conda `activate.d`/`deactivate.d` 스크립트를 만들어 `LD_LIBRARY_PATH`에 `$CONDA_PREFIX/lib`를 앞에 추가/해제.  
   - RPATH가 커버하지만 안전장치로 제공합니다.

> **참고**: 최신 NumPy에서는 `npymath` 링크가 불필요하여 링크하지 않습니다.

### 준비물
- JetPack 5.x (CUDA 11.4 / TensorRT 포함)  
- **Python 3.8** Conda 환경 (`numpy`, `cmake`, `make` 등 설치)  
- OpenCV는 시스템 `/usr/local` 설치여도 무방  
- `git`, `cmake`, `make`, C++ 툴체인

### 1) jetson-inference 클론
```bash
cd ~
git clone --recursive https://github.com/dusty-nv/jetson-inference.git
```

### 2) 패치 스크립트 실행
먼저 사용할 **Conda 환경 활성화**:
```bash
conda activate <your-env>
```

**실행 순서**
1. (권장) **이전 전역 설치 정리**
   ```bash
   bash ./00_cleanup_global_jetson.sh
   ```

2. (선택) **Conda 활성화 훅 설치**
   ```bash
   bash ./01_env_hooks_ldpath.sh
   ```

3. **빌드 + 환경에 설치**
   ```bash
   bash ./02_build_install.sh
   # 또는
   bash ./02_build_install.sh /absolute/path/to/jetson-inference
   ```

### 3) 간단 테스트 (선택)
```bash
python - <<'PY'
import jetson_utils, jetson_inference
print("OK: imports")
PY

python - <<'PY'
import numpy as np
from jetson_utils import cudaFromNumpy, cudaToNumpy
a = (np.random.rand(8,8,3)*255).astype(np.uint8)
cap = cudaFromNumpy(a)
b = cudaToNumpy(cap)
print(a.shape, b.shape)
PY
```

### 설치 위치
- 라이브러리:  
  - `$CONDA_PREFIX/lib/libjetson-utils.so`
  - `$CONDA_PREFIX/lib/libjetson-inference.so`
- Python 모듈:  
  - `$CONDA_PREFIX/lib/python3.8/site-packages/jetson_utils_python.so`
  - `$CONDA_PREFIX/lib/python3.8/site-packages/jetson_inference_python.so`
  - 및 `Jetson/*`, `jetson/*` 패키지

### 문제 해결
- **CMake `CMP0148` 경고**: 무시 가능.  
- **`ldconfig` 권한 경고**: 무시 가능(RPATH 사용).  
- **`ImportError: undefined symbol`**: 전역/환경 혼재 가능성. `00_cleanup_global_jetson.sh` 후 재빌드.  
- **헤드리스**: 일부 예제는 화면이 필요. 지원 시 `--headless` 사용.

---

**Folder layout**
```
ir-guide/code/jetson_inference_patch
├── 00_cleanup_global_jetson.sh
├── 01_env_hooks_ldpath.sh
├── 02_build_install.sh
├── patches
│   ├── PyNumpy.cpp.diff
│   ├── python
│   │   └── bindings
│   │       └── CMakeLists.txt
│   └── utils
│       └── python
│           └── bindings
│               ├── CMakeLists.txt
│               └── PyNumpy.cpp
└── README.md   ← this file
```
