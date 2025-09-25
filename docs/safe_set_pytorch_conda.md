[ [English](#english) | [한국어](#한국어) ]

# PyTorch Conda Environment (Jetson) — `safe_set_pytorch_conda.sh`

> Location: `./safe_set_pytorch_conda.sh`

---

## English

### Why this script matters (read first)
JetPack **5.1.5** on Jetson ships **system CUDA 11.4 + cuDNN + TensorRT** under `/usr` and `/usr/local`.  
Prebuilt **PyTorch for Jetson** wheels are compiled **against those system libraries** (Python 3.8).

Common pitfalls we avoid:
- ❌ Installing `cudatoolkit` inside conda → **two CUDAs** on the same device → ABI mismatches.  
- ❌ Installing `opencv` from conda/apt that shadows `/usr/local` build → CUDA in OpenCV “disappears.”  
- ❌ Grabbing upstream PyTorch (e.g., `+cu118`) → not built for Jetson → runtime errors.

This script creates a **clean conda env** that:
1. Uses **Python 3.8** (to match Jetson wheels).  
2. Installs **official Jetson PyTorch wheel** you choose (2.1.0 / 2.0.0 / 1.14.0).  
3. **Does not** install a duplicate CUDA into conda. Torch links to **system CUDA/cuDNN/TRT**.  
4. Builds **matching torchvision** from source (with CUDA).  
5. Keeps **OpenCV (CUDA)** from `/usr/local` visible inside the env.

Result: Students get a stable, reproducible env with GPU working out of the box.

---

## Quick usage
```bash
# 0) Requires Miniforge (aarch64). If missing, install and init:
# wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
# bash Miniforge3-Linux-aarch64.sh -b -p $HOME/miniforge3
# source "$HOME/miniforge3/etc/profile.d/conda.sh" && conda activate

# 1) Run the helper
./safe_set_pytorch_conda.sh

# 2) Follow prompts
#   - Enter NEW conda env name
#   - Select PyTorch version: [1] 2.1.0  [2] 2.0.0  [3] 1.14.0
#   - The script installs the wheel, builds matching torchvision, and validates CUDA.
```

### Version matrix (Torch ↔ torchvision)
- PyTorch **1.14.0** → torchvision **0.14.1**  
- PyTorch **2.0.0** → torchvision **0.15.1**  
- PyTorch **2.1.0** → torchvision **0.16.1**

> Tip: You can re-run the script with a different env name if you want multiple versions side-by-side.

---

## What the script does (high level)
- Checks **Miniforge** is present, else exits with install instructions.  
- Creates a **new** conda env (fails early if name already exists).  
- Installs **Python 3.8** and minimal essentials (pip, wheel, setuptools).  
- Installs selected **Jetson PyTorch wheel** (no cudatoolkit).  
- Builds **torchvision** from source with CUDA.  
- Ensures the env can import **system OpenCV** (`/usr/local/lib/python3.8/site-packages` if needed).  
- Validates with quick tests:
  - `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`  
  - `python -c "import cv2; print(cv2.__version__, cv2.cuda.getCudaEnabledDeviceCount())"`

### Why not venv?
`venv` doesn’t manage compiled deps well on Jetson (e.g., you still must keep system CUDA in sync).  
Conda makes it easier for students to isolate Python packages while **not** replacing system CUDA.

---

## Troubleshooting
- **`torch.cuda.is_available()` = False**  
  Ensure JetPack packages are installed: `sudo apt install -y nvidia-jetpack` and reboot.  
  Check `nvcc --version`, and that `/usr/lib/aarch64-linux-gnu` has cuDNN/TRT libraries; then `sudo ldconfig`.

- **ImportError: `libcudnn.so.8` not found**  
  JetPack runtime missing or ld path not refreshed → install `nvidia-jetpack` and run `sudo ldconfig`.

- **OpenCV has no CUDA**  
  Make sure you built OpenCV with CUDA to `/usr/local`. Remove any conda `opencv` that shadows it.

- **Wrong Torch wheel grabbed from PyPI**  
  Uninstall and install the Jetson wheel URL:
  ```bash
  pip uninstall -y torch torchvision
  pip install <the-jetson-wheel.whl>
  ```

- **Remove env / start over**
  ```bash
  conda deactivate
  conda env remove -n <envname>
  ```

---

## 한국어

### 왜 이 스크립트가 중요한가
JetPack **5.1.5**(Jetson)는 `/usr` 및 `/usr/local`에 **시스템 CUDA 11.4 / cuDNN / TensorRT**를 제공합니다.  
Jetson용 **사전 빌드 PyTorch** 휠은 **이 시스템 라이브러리**를 기준으로 컴파일되어 있습니다(파이썬 3.8).

우리가 피하고 싶은 흔한 실수:
- ❌ conda 안에 `cudatoolkit` 설치 → **CUDA가 두 개** → ABI 충돌, 런타임 오류  
- ❌ conda/apt의 `opencv`가 `/usr/local` OpenCV를 가리는 경우 → CUDA 기능 미표시  
- ❌ 일반 PyTorch(예: `+cu118`) 설치 → Jetson용이 아님 → 실행 실패

이 스크립트는 다음을 보장합니다:
1. **Python 3.8** 기반의 **깨끗한 conda 환경** 생성  
2. 선택한 **Jetson PyTorch 휠** 설치(2.1.0 / 2.0.0 / 1.14.0)  
3. conda에 **별도 CUDA 미설치** → Torch가 **시스템 CUDA/cuDNN/TRT** 사용  
4. **torchvision**을 버전에 맞춰 CUDA로 빌드  
5. `/usr/local`의 **CUDA OpenCV**를 환경에서 그대로 사용 가능

학생 입장에서는 GPU가 **바로 작동**하는 안정적인 환경을 확보합니다.

---

## 빠른 사용법
```bash
# 0) Miniforge 필요. 없으면 설치/초기화:
# wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
# bash Miniforge3-Linux-aarch64.sh -b -p $HOME/miniforge3
# source "$HOME/miniforge3/etc/profile.d/conda.sh" && conda activate

# 1) 실행
./safe_set_pytorch_conda.sh

# 2) 안내에 따라 입력
#   - 새 conda 환경 이름
#   - PyTorch 버전 선택: [1] 2.1.0  [2] 2.0.0  [3] 1.14.0
#   - 휠 설치, torchvision 빌드, CUDA 동작 확인까지 자동 처리
```

### 버전 매핑 (Torch ↔ torchvision)
- PyTorch **1.14.0** → torchvision **0.14.1**  
- PyTorch **2.0.0** → torchvision **0.15.1**  
- PyTorch **2.1.0** → torchvision **0.16.1**

---

## 동작 개요
- **Miniforge** 확인 (없으면 설치 안내 후 종료)  
- **새 conda 환경** 생성(동일 이름 존재 시 중단)  
- **Python 3.8** + 최소 패키지 설치  
- 선택한 **Jetson PyTorch 휠** 설치(※ conda CUDA 설치 안 함)  
- **torchvision** CUDA 빌드  
- 환경에서 시스템 **OpenCV(CUDA)** 사용 확인  
- **간단 테스트** 수행 (torch/cv2 + CUDA 가능 여부)

### venv를 사용하지 않는 이유
Jetson에서는 바이너리 호환성 관리가 까다롭습니다. conda는 파이썬 패키지를 격리하면서 **시스템 CUDA**를 유지하기에 학생용 실습에 안전합니다.

---

## 문제 해결
- **`torch.cuda.is_available()` 가 False**  
  `sudo apt install -y nvidia-jetpack` 후 재부팅. `nvcc --version` 확인, `sudo ldconfig` 실행.

- **`libcudnn.so.8` ImportError**  
  JetPack 런타임 미설치/링커 캐시 미갱신. `nvidia-jetpack` 설치 후 `sudo ldconfig`.

- **OpenCV에서 CUDA 미표시**  
  `/usr/local`에 CUDA 빌드 OpenCV가 있는지 확인. conda의 `opencv`가 가리지 않게 주의.

- **PyPI에서 일반 Torch를 받아버린 경우**  
  ```bash
  pip uninstall -y torch torchvision
  pip install <Jetson 휠 URL>
  ```

- **환경 삭제**
  ```bash
  conda deactivate
  conda env remove -n <envname>
  ```

---

_Back to:_ **[Root README](../README.md)** · **[Scripts](../scripts/README.md)** · **[Dev stack](./install_jetson_dev_packages.md)** 
