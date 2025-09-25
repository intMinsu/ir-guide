[ [English](#english) | [한국어](#한국어) ]

# PyTorch Conda Environment (Jetson) — `safe_set_pytorch_conda.sh`

> Location: `./safe_set_pytorch_conda.sh`

---

## English

### Why this script matters (read first)
JetPack **5.1.x** on Jetson ships **system CUDA/cuDNN/TensorRT** under `/usr` and `/usr/local`.
Prebuilt **PyTorch for Jetson** wheels are compiled **against those system libraries** (Python 3.8).

Common pitfalls we avoid:
- ❌ Installing `cudatoolkit` inside conda → **duplicate CUDA** on the same device → ABI & loader conflicts  
- ❌ Installing `opencv*` wheels in conda that shadow `/usr/local` build → CUDA in OpenCV “disappears”  
- ❌ Grabbing upstream PyTorch (e.g., `+cu118`) → not built for L4T → runtime failures

This script creates a **clean conda env** that:
1. Uses **Python 3.8** (to match Jetson wheels).  
2. Installs an **official Jetson PyTorch wheel** you choose (2.1.0 / 2.0.0 / 1.14.0).  
3. **Does not** install a duplicate CUDA into conda (Torch links to **system CUDA/cuDNN/TRT**).  
4. Optionally builds **matching torchvision** from source (with CUDA).  
5. **Selectively bridges** system Python packages into the env to avoid “flooding”: by default only **`cv2` (OpenCV, CUDA build)**, **`jtop`**, and **`smbus2`** are exposed via a tiny bridge dir.  
6. (Default) Adds **soft clamps** on activation to keep paths sane, and uses a **hard‑clamped** build shell for fragile builds (you can opt out).

Result: a stable, reproducible env with GPU working out of the box, without unintentionally importing every system package.

---

## Quick usage
```bash
# 0) Requires Miniforge (aarch64). If missing, install and init:
# wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
# bash Miniforge3-Linux-aarch64.sh -b -p $HOME/miniforge3
# source "$HOME/miniforge3/etc/profile.d/conda.sh" && conda activate

# 1) Run
./safe_set_pytorch_conda.sh

# 2) Follow prompts
#   - Enter NEW conda env name
#   - Select PyTorch version: [1] 2.1.0  [2] 2.0.0  [3] 1.14.0
#   - The script installs the wheel, bridges selected system packages (cv2/jtop/smbus2),
#     (optionally) builds torchvision, and validates CUDA.
```

### CLI flags / environment switches
- `--no-tv` or `SKIP_TV=1` — **skip** building `torchvision`  
- `--no-clamp` or `CLAMP_MODE=none` — **no soft hooks** installed and builds run in the **plain** env  
- `JETSON_APPEND_OLD_LD=0` — with soft clamps, make `LD_LIBRARY_PATH` **stricter** (don’t append prior entries)

> Default is `CLAMP_MODE=soft`: install activate/deactivate hooks (soft clamp) and build fragile packages
> (like torchvision) in a **hard-clamped** subshell to avoid leakage from your login environment.

### Version matrix (Torch ↔ torchvision)
- PyTorch **1.14.0** → torchvision **0.14.1**  
- PyTorch **2.0.0** → torchvision **0.15.1**  
- PyTorch **2.1.0** → torchvision **0.16.1**

> Tip: Run the script again with a different env name if you want multiple versions side-by-side.

---

## What the script does (high level)
- Verifies **Miniforge** is installed.  
- Creates a **new** conda env (fails fast if the name exists).  
- Installs **Python 3.8** and Py3.8-safe build tools (`pip<25`, `setuptools<75`, `wheel<0.45`).  
- Installs the selected **Jetson PyTorch wheel** (no `cudatoolkit`).  
- **Selective system package bridge**: writes a `.pth` file in the env’s `site-packages` pointing to a **bridge directory** inside the env that contains **symlinks only to the desired modules** from the system install—by default `cv2`, `jtop`, and `smbus2`.  
- **(Default)** Soft clamp hooks: on `conda activate`, set/sanitize `CUDA_HOME`, `PATH`, `LD_LIBRARY_PATH` and hints for CMake/pkg-config; restore on deactivate.  
- **(Default)** Hard clamp for fragile builds: `env -i …` wrapper so build steps don’t inherit noisy vars.  
- Validates with quick tests:
  - `import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())`  
  - `import cv2; print(cv2.__version__, cv2.cuda.getCudaEnabledDeviceCount())`

### About the selective bridge
- The bridge directory lives at:  
  `"$CONDA_PREFIX/share/jetson-python-bridge"`  
  The `.pth` file is:  
  `"$ENV_SITE_PACKAGES/jetson_system_bridge.pth"`
- Add/remove exposed modules by editing the bridge list in the script (default: `cv2`, `jtop`, `smbus2`).  
- If you ever want to expose the **whole** system `dist-packages` (not recommended), you can manually write a broad path into the `.pth` file instead of the bridge path.

### About clamps (soft vs hard)
- **Soft clamp (default):** activate/deactivate hooks keep your shell pinned to JetPack CUDA paths; reduces surprises during day‑to‑day work.  
- **Hard clamp:** only the *build command* runs in a hermetic env with whitelisted variables; ideal for extension builds that are sensitive to stray env values.  
- **No clamp:** if you pass `--no-clamp`, the script won’t install hooks and won’t use the hard‑clamped wrapper.

### Why not venv?
`venv` doesn’t manage compiled deps well on Jetson (you still must keep system CUDA in sync).  
Conda isolates Python packages while **not** replacing system CUDA.

---

## Troubleshooting

**Build isolation pulled too‑new tools on Py3.8**  
- Symptom: setuptools “requires‑python >=3.9” error; or `pip 25.x` breaks build isolation  
- Fix inside the env:  
  ```bash
  python -m pip install "pip<25" "setuptools<75" "wheel<0.45" -U
  ```

**`torch.cuda.is_available()` = False**  
- Ensure JetPack runtime is installed: `sudo apt install -y nvidia-jetpack` and reboot  
- Check `nvcc --version`; then `sudo ldconfig`

**`CUDA_HOME` not set during build**  
- The script sets it, but for manual builds:  
  ```bash
  export CUDA_HOME=/usr/local/cuda
  export PATH=$CUDA_HOME/bin:$PATH
  export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu/tegra:$LD_LIBRARY_PATH
  ```

**OpenCV has no CUDA / wrong OpenCV imported**  
- Ensure you didn’t install any pip `opencv-*` wheels inside the env (they can shadow the system build).  
- Confirm the bridge path exists and contains `cv2` symlink; `.pth` should point to the bridge dir (not to `dist-packages`).

**`jtop`/`smbus2` not found in the env**  
- Make sure those packages exist in the system Python (`/usr/local/lib/python3.8/dist-packages`) and the bridge created symlinks. Adjust the bridge list or install them in the env if you prefer per‑env copies.

**torchvision “CPU only” symptoms**  
- Ensure you built it **against the active torch** and with CUDA visible (`CUDA_HOME`, `LD_LIBRARY_PATH`).  
- If in doubt, rebuild using the script (default hard clamp).

**Accidentally installed `cudatoolkit` in conda**  
- Remove it and friends to avoid “two CUDAs”:  
  ```bash
  conda remove --force cudatoolkit cudnn cublas cusolver cufft cutensor pytorch-cuda
  ```

**Nuke and retry**  
```bash
conda deactivate
conda env remove -n <envname>
```

---

## Appendix: How the **selective bridge** works (.pth + symlinks)
Any `*.pth` in `site-packages/` is read by Python at startup; each non‑comment line is appended to `sys.path`.
Instead of pointing to the entire system `dist-packages` (which would make *all* system modules importable), this script:

1) Creates a **bridge directory** inside the env (e.g., `$CONDA_PREFIX/share/jetson-python-bridge`)  
2) Places **symlinks only for chosen modules** (default: `cv2`, `jtop`, `smbus2`) into that dir  
3) Writes a `.pth` that points to the bridge dir

Result: the env can `import cv2/jtop/smbus2` from the system install, **without** unintentionally exposing everything else.

---

## 한국어

### 왜 이 스크립트가 중요한가
JetPack **5.1.x**(Jetson)는 `/usr` 및 `/usr/local`에 **시스템 CUDA/cuDNN/TensorRT**를 제공합니다.  
Jetson용 **사전 빌드 PyTorch** 휠은 **이 시스템 라이브러리**를 기준으로 컴파일되어 있습니다(파이썬 3.8).

우리가 피하고 싶은 흔한 실수:
- ❌ conda 안에 `cudatoolkit` 설치 → **CUDA가 두 개** → ABI/로더 충돌  
- ❌ conda/pip의 `opencv*` 휠이 `/usr/local` OpenCV를 가리는 경우 → CUDA 기능 미표시  
- ❌ 일반 PyTorch(예: `+cu118`) 설치 → L4T 비호환 → 실행 실패

이 스크립트는 다음을 보장합니다:
1. **Python 3.8** 기반의 **깨끗한 conda 환경** 생성  
2. 선택한 **Jetson PyTorch 휠** 설치(2.1.0 / 2.0.0 / 1.14.0)  
3. conda에 **별도 CUDA 미설치** → Torch가 **시스템 CUDA/cuDNN/TRT** 사용  
4. 필요 시 **torchvision**을 버전에 맞춰 CUDA로 빌드  
5. **선택적 브리지**로 시스템 파이썬 패키지를 노출: 기본값으로 **`cv2`(CUDA OpenCV)**, **`jtop`**, **`smbus2`**만 환경에서 보이도록 작은 브리지 디렉터리를 사용합니다.  
6. (기본) **Soft clamp** 훅으로 활성화 시 경로를 고정, 빌드 시엔 **Hard clamp**로 깨끗한 환경 사용

결과: GPU가 바로 동작하면서도, 시스템 패키지가 **무분별하게** 환경으로 흘러들어오지 않는 재현 가능한 셋업.

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
#   - 휠 설치, 선택 모듈(cv2/jtop/smbus2) 브리지, (선택) torchvision 빌드, CUDA 확인까지 자동 처리
```

### 옵션 (플래그/환경변수)
- `--no-tv` 또는 `SKIP_TV=1` — `torchvision` 빌드 **생략**  
- `--no-clamp` 또는 `CLAMP_MODE=none` — **훅 미설치**, 빌드도 **일반 환경**에서 수행  
- `JETSON_APPEND_OLD_LD=0` — soft clamp 사용 시 `LD_LIBRARY_PATH`를 **더 엄격**하게 유지

기본값은 `CLAMP_MODE=soft` 입니다. (활성/비활성 훅 설치 + 민감한 빌드는 hard clamp로 실행)

---

## 동작 개요
- **Miniforge** 확인 (없으면 설치 안내 후 종료)  
- **새 conda 환경** 생성(동일 이름 존재 시 중단)  
- **Python 3.8** + Py3.8에 안전한 빌드 도구(`pip<25`, `setuptools<75`, `wheel<0.45`) 설치  
- 선택한 **Jetson PyTorch 휠** 설치 (`cudatoolkit` 설치 금지)  
- **선택적 시스템 패키지 브리지**: env 내부에 브리지 디렉터리를 만들고, 시스템 설치에서 **원하는 모듈만**(기본: `cv2`, `jtop`, `smbus2`) 심볼릭 링크로 노출. env의 `site-packages`에는 이 브리지 경로를 가리키는 `.pth`를 작성.  
- **(기본)** Soft clamp 훅: `CUDA_HOME`, `PATH`, `LD_LIBRARY_PATH` 등을 활성화 시 고정/정리 (비활성화 시 복원)  
- **(기본)** Hard clamp 빌드: `env -i …`로 빌드 단계만 깨끗한 환경에서 수행  
- **간단 테스트** 수행 (torch/cv2 + CUDA 가능 여부)

---

## 문제 해결

**Py3.8에서 빌드 도구가 너무 최신인 경우**  
```bash
python -m pip install "pip<25" "setuptools<75" "wheel<0.45" -U
```

**`torch.cuda.is_available()` 가 False**  
- `sudo apt install -y nvidia-jetpack` 후 재부팅  
- `nvcc --version` 확인 후 `sudo ldconfig`

**빌드 중 `CUDA_HOME` 미설정**  
```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu/tegra:$LD_LIBRARY_PATH
```

**OpenCV에서 CUDA 미표시 / 잘못된 OpenCV 선택**  
- env 내부에 pip `opencv-*` 휠이 있으면 제거 (시스템 빌드가 가려질 수 있음)  
- 브리지 디렉터리에 `cv2` 링크가 존재하는지, `.pth`가 브리지 경로를 가리키는지 확인

**`jtop`/`smbus2` 가 env에서 안 보임**  
- 시스템 파이썬에 해당 패키지가 설치되어 있는지 확인하고, 브리지 생성이 성공했는지 점검. 필요 시 브리지 목록을 수정하거나 env에 별도 설치.

**torchvision이 CPU 전용처럼 동작**  
- 활성 Torch에 맞춰 CUDA가 보이는 상태에서 빌드했는지 확인  
- 의심되면 스크립트로 재빌드 (기본 hard clamp)

**conda에 `cudatoolkit`을 실수로 설치**  
```bash
conda remove --force cudatoolkit cudnn cublas cusolver cufft cutensor pytorch-cuda
```

**환경 삭제**
```bash
conda deactivate
conda env remove -n <envname>
```

---

## 부록: **선택적 브리지** 동작 방식
`site-packages/`에 있는 `*.pth` 파일은 파이썬 시작 시 읽혀 각 줄을 `sys.path`에 추가합니다.
이 스크립트는 전체 `dist-packages`를 올리는 대신,

1) env 내부에 **브리지 디렉터리**(예: `$CONDA_PREFIX/share/jetson-python-bridge`)를 만들고  
2) 시스템 설치에서 **선택한 모듈만**(기본: `cv2`, `jtop`, `smbus2`) 심볼릭 링크로 두며  
3) `.pth`는 이 브리지 디렉터리를 가리키게 합니다.

그 결과, 필요한 모듈만 시스템 빌드로부터 보이고, 나머지는 env의 순수성/재현성을 유지합니다.

---

_Back to:_ **[Root README](../README.md)** · **[Scripts](../scripts/README.md)** · **[Dev stack](./install_jetson_dev_packages.md)**
