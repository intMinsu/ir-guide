
[ [English](#english) | [한국어](#한국어) ]

# PyTorch Conda Environment (Jetson) — `safe_set_pytorch_conda.sh`

> Location: `./safe_set_pytorch_conda.sh`

---

## English

### What’s new (this version)
- ✅ **Build‑speed selector** up front: **Safe(2)** / **Balanced(4)** / **Fast(6)** / **Custom / Auto**.  
  *Note:* Jetson Xavier (nproc=6) uses **Fast = 6 jobs** by default.
- ✅ **Ninja by default** (`CMAKE_GENERATOR=Ninja`) with live progress (`NINJA_STATUS`).  
- ✅ **Elapsed time** printed for **torchvision** and **torchaudio** builds.  
- ✅ Optional **torchaudio** build from source (version matched to your PyTorch).  
- ✅ Safer **LD_LIBRARY_PATH**/OpenCV hints in the soft‑clamp hooks.  
- ✅ Handles Ubuntu 20.04’s `ninja-build` package (symlink to `ninja` if needed).

> **Strongly recommended pre‑flight:** free memory first with `scripts/jetson_spare_mem.sh` (headless + swap).  
> Swap prevents OOM but doesn’t make builds faster — it just avoids crashes.


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
4. Optionally builds **matching torchvision** and **torchaudio** from source.  
5. **Selectively bridges** system Python packages into the env: by default only **`cv2` (OpenCV, CUDA build)**, **`jtop`**, and **`smbus2`** via a tiny bridge dir.  
6. (Default) Adds **soft clamps** on activation to keep paths sane, and uses a **hard‑clamped** build shell for fragile builds (you can opt out).

---

## Pre‑flight: free memory (recommended)
On Jetson, CPU/GPU share DRAM. Turning off the desktop and enabling a swapfile dramatically reduces OOM risk during C++/CUDA links.

```bash
# Headless (temporarily stop GUI for this boot)
scripts/jetson_spare_mem.sh headless on

# Add 8G swap (prefer NVMe path if available)
scripts/jetson_spare_mem.sh swap enable 8G /swapfile
scripts/jetson_spare_mem.sh swap tune 10
```

> Later, restore GUI with `scripts/jetson_spare_mem.sh headless off`.  
> More details: [`docs/jetson_spare_mem.md`](./jetson_spare_mem.md)

---

## Quick usage
```bash
# 0) Requires Miniforge (aarch64). If missing, install and init:
# wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
# bash Miniforge3-Linux-aarch64.sh -b -p $HOME/miniforge3
# source "$HOME/miniforge3/etc/profile.d/conda.sh" && conda activate

# (Recommended) Free memory first
scripts/jetson_spare_mem.sh headless on
scripts/jetson_spare_mem.sh swap enable 8G /swapfile

# 1) Run
./safe_set_pytorch_conda.sh

# 2) Follow prompts
#   - Enter NEW conda env name
#   - Select PyTorch version: [1] 2.1.0  [2] 2.0.0  [3] 1.14.0
#   - Select build speed profile (Safe/Balanced/Fast/Custom/Auto)
#   - The script installs the wheel, bridges cv2/jtop/smbus2,
#     (optionally) builds torchvision/torchaudio, and validates CUDA.

# (Optional) Re‑enable GUI once builds finish
scripts/jetson_spare_mem.sh headless off
```

### CLI flags / environment switches
- `--no-tv` or `SKIP_TV=1` — **skip** building `torchvision`  
- `--no-ta` or `SKIP_TA=1` — **skip** building `torchaudio`  
- `--no-clamp` or `CLAMP_MODE=none` — **no soft hooks** installed and builds run in the **plain** env  
- `SPEED_PROFILE=safe|balanced|fast|auto|custom:N` — set jobs & feature toggles in one go  
  - **Safe=2 jobs**, **Balanced=4 jobs**, **Fast=6 jobs** on Xavier (nproc=6), Auto by RAM  
- `USE_NINJA=0` — force Unix Makefiles (single‑config generators; slower but predictable)  
- `JETSON_APPEND_OLD_LD=0` — with soft clamps, make `LD_LIBRARY_PATH` **stricter** (don’t append prior entries)

> The script also prints **elapsed seconds** for tv/ta builds and shows **Ninja progress** by default.


### Version matrix (Torch ↔ torchvision ↔ torchaudio)
- PyTorch **1.14.0** → torchvision **0.14.1**, torchaudio **0.13.1**  
- PyTorch **2.0.0** → torchvision **0.15.1**, torchaudio **2.0.2**  
- PyTorch **2.1.0** → torchvision **0.16.1**, torchaudio **2.1.2**

> Tip: Run the script again with a different env name if you want multiple versions side‑by‑side.

---

## What the script does (high level)
- Verifies **Miniforge** is installed.  
- Creates a **new** conda env (fails fast if the name exists).  
- Installs **Python 3.8** and Py3.8‑safe build tools (`pip<25`, `setuptools<75`, `wheel<0.45`).  
- Installs the selected **Jetson PyTorch wheel** (no `cudatoolkit`).  
- **Selective system package bridge**: writes a `.pth` file pointing to a **bridge directory** inside the env that contains **symlinks only to the desired modules** from the system install—by default `cv2`, `jtop`, `smbus2`.  
- **Soft clamp hooks** on `conda activate` set/sanitize `CUDA_HOME`, `PATH`, `LD_LIBRARY_PATH`, and hints for CMake/pkg‑config; **restore** on deactivate.  
- **Hard‑clamped** build helper runs fragile builds in a hermetic env (`env -i` whitelist).  
- Builds optional **torchvision/torchaudio**, defaulting to **Ninja** with **NINJA_STATUS**.  
- Prints **elapsed seconds** for tv/ta; verifies torch/vision/audio + OpenCV CUDA.


### About the selective bridge
- Bridge dir: `$CONDA_PREFIX/share/jetson-python-bridge`  
- `.pth`: `site-packages/jetson_system_bridge.pth` → points to the bridge dir  
- Only selected modules (default: `cv2`, `jtop`, `smbus2`) are linked; everything else stays isolated.


### About clamps (soft vs hard)
- **Soft clamp (default):** activate/deactivate hooks keep your shell pinned to JetPack CUDA paths and known good search order (e.g., `/usr/local/lib`, CUPTI, Jetson `tegra` libs).  
- **Hard clamp:** only the *build command* runs in a hermetic env with whitelisted variables; ideal for extension builds.  
- **No clamp:** pass `--no-clamp` to skip hooks and run builds in your current shell.


### Build‑speed selector & presets
On Jetson Xavier (nproc=6), **Fast** uses **6 jobs**. Jobs are clamped to `nproc` and may auto‑reduce by RAM in `auto`.  
- **Safe (2)** — safest for 4–8 GB; disables torchvision video (FFmpeg) to shrink compile.  
- **Balanced (4)** — good speed vs RAM for 8 GB; also disables tv video.  
- **Fast (6)** — for Xavier/Orin with headroom (enable swap + headless).  
- **Custom / Auto** — set your own, or let RAM decide.

Environment enforced for all builds:
```
MAX_JOBS=<N>
CMAKE_BUILD_PARALLEL_LEVEL=<N>
CMAKE_GENERATOR=Ninja   # unless you set USE_NINJA=0
NINJA_STATUS='[%r tasks/s | %f/%t | %es elapsed]'
```

### Torchaudio “lean” vs “full”
If torchaudio is slow on low‑RAM devices, you can trim features before the TA build:
```bash
# Leanest (fastest to compile)
export BUILD_RNNT=0
export BUILD_KALDI=0
export USE_FFMPEG=0
export USE_CUDA=0
export BUILD_SOX=1
```
Rebuild with the script if later you need FFmpeg I/O or CUDA‑enhanced ops.


---

## Troubleshooting

**`E: Unable to locate package ninja` on Ubuntu 20.04**  
```bash
sudo apt-get update
sudo apt-get install -y ninja-build
sudo ln -sf "$(command -v ninja-build)" /usr/local/bin/ninja
```
*(The script attempts this workaround automatically for torchaudio.)*

**Linker OOM / “killed” while building**  
- Use the **Safe** profile (2 jobs), enable **headless** and at least **8G swap**:  
  ```bash
  scripts/jetson_spare_mem.sh headless on
  scripts/jetson_spare_mem.sh swap enable 8G /swapfile
  ```

**`torch.cuda.is_available()` = False**  
- Ensure JetPack runtime is installed: `sudo apt install -y nvidia-jetpack` and reboot.  
- `nvcc --version`; then `sudo ldconfig`.

**OpenCV has no CUDA / wrong OpenCV imported**  
- Don’t install pip `opencv-*` wheels inside the env.  
- Confirm the bridge exists and `.pth` points to it.

**Torchaudio FFmpeg not found / very slow**  
- Install `libav*` dev packages (the script covers common ones), or disable FFmpeg via `USE_FFMPEG=0` (faster compile).

**Ninja spawns too many jobs when building “manually”**  
- Upstream defaults can be aggressive; clamp with:  
  ```bash
  export MAX_JOBS=2
  export CMAKE_BUILD_PARALLEL_LEVEL=2
  ```

**Nuke and retry**  
```bash
conda deactivate
conda env remove -n <envname>
```

---

## 한국어

### 이번 버전 변경점
- 시작 시 **빌드 속도 선택**: **안전(2)** / **균형(4)** / **빠름(6)** / **사용자 지정·자동**  
  *참고:* Xavier(nproc=6)는 **빠름=6 작업** 기본.  
- 기본 **Ninja** 사용 + 진행 표시.  
- **torchvision/torchaudio** 빌드 **소요 시간** 출력.  
- **torchaudio** 소스 빌드 옵션 추가.  
- soft‑clamp 훅의 **LD_LIBRARY_PATH** 정리.  
- Ubuntu 20.04의 `ninja-build`를 자동 처리(필요 시 심볼릭 링크).

> **사전 권장:** `scripts/jetson_spare_mem.sh`로 **헤드리스 + 스왑** 먼저 적용하세요.  
> 스왑은 OOM 방지용이지 속도 향상은 아닙니다.

### 왜 필요한가
JetPack **5.1.x**는 `/usr` 및 `/usr/local`에 **시스템 CUDA/cuDNN/TensorRT**를 제공합니다.  
Jetson용 **PyTorch 휠**은 이 라이브러리에 맞춰 컴파일되어 있습니다.

피해야 할 실수:
- conda에 `cudatoolkit` 설치 → **이중 CUDA**  
- conda에서 `opencv*` 휠 설치 → `/usr/local` OpenCV(CUDA)가 가려짐  
- 일반 PyTorch(`+cu118`) 설치 → L4T 비호환

스크립트는 다음을 수행합니다:
1. **Python 3.8** 기반 새 conda 환경 생성  
2. 선택한 **Jetson PyTorch 휠** 설치  
3. conda에 CUDA 미설치(시스템 CUDA 사용)  
4. 필요 시 **torchvision/torchaudio** 소스 빌드  
5. 시스템 모듈 **선택 브리지**(`cv2`, `jtop`, `smbus2`)  
6. **Soft/Hard clamp**로 경로·빌드 환경 안정화

---

## 사전 준비: 메모리 확보(권장)
```bash
scripts/jetson_spare_mem.sh headless on
scripts/jetson_spare_mem.sh swap enable 8G /swapfile
```
작업 후 `scripts/jetson_spare_mem.sh headless off`로 GUI 복구. 자세한 내용은 [`docs/jetson_spare_mem.md`](./jetson_spare_mem.md) 참고.

---

## 빠른 사용법
```bash
./safe_set_pytorch_conda.sh
# 새 env 이름, Torch 버전, 빌드 속도 프로필 선택
```

### 옵션 / 환경변수
- `--no-tv`, `--no-ta`, `--no-clamp`  
- `SPEED_PROFILE=safe|balanced|fast|auto|custom:N` (Xavier: fast=6)  
- `USE_NINJA=0`, `JETSON_APPEND_OLD_LD=0`

### 버전 매트릭스
- Torch **1.14.0** → TV **0.14.1**, TA **0.13.1**  
- Torch **2.0.0** → TV **0.15.1**, TA **2.0.2**  
- Torch **2.1.0** → TV **0.16.1**, TA **2.1.2**

---

## 동작 개요
- Miniforge 확인 → 새 env 생성 → Py3.8 도구 고정  
- Jetson PyTorch 휠 설치 (cudatoolkit 미설치)  
- `cv2/jtop/smbus2`만 브리지 `.pth`로 노출  
- soft‑clamp 훅/ hard‑clamp 빌드 사용  
- (선택) TV/TA 빌드, 소요 시간 출력, CUDA 접근 검증

### 빌드 속도 프로필
- **안전(2)**: 4–8 GB 권장, TV 비디오 비활성화  
- **균형(4)**: 8 GB 권장, TV 비디오 비활성화  
- **빠름(6)**: Xavier/Orin 권장(스왑+헤드리스)  
- **사용자 지정 / 자동**: 직접 설정 또는 RAM 기반 자동

### Torchaudio 빠른 설정(저메모리)
```bash
export BUILD_RNNT=0 BUILD_KALDI=0 USE_FFMPEG=0 USE_CUDA=0 BUILD_SOX=1
```

---

## 문제 해결
- **`ninja` 패키지 없음** → `sudo apt install -y ninja-build && sudo ln -sf $(command -v ninja-build) /usr/local/bin/ninja`  
- **링커 OOM** → 프로필을 낮추고(안전/균형), 헤드리스+스왑 적용  
- **CUDA 비활성** → `sudo apt install -y nvidia-jetpack`, 재부팅, `sudo ldconfig`  
- **OpenCV CUDA 미표시** → pip `opencv-*` 제거, 브리지 확인

**환경 삭제**
```bash
conda deactivate
conda env remove -n <envname>
```

---

_Back to:_ **[Root README](../README.md)** · **[Scripts](../scripts/README.md)** · **[Spare Memory Guide](./jetson_spare_mem.md)**
