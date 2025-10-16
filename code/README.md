[ [English](#english) | [한국어](#한국어) ]
# Code
**Up one level:** [`../README.md`](../README.md)

---

## English

### What’s in this folder
This directory contains small, focused examples and helper scripts that accompany the guide.

- **`cuda_manipulation/`** — simple CUDA manipulation example (device/query/mem ops).  
- **`jetson_inference_patch/`** — patches and helpers to **build `jetson-inference` inside a conda env** on Jetson.  
- **`remote_streaming/`** — examples for **remote video streaming** (MediaMTX / Jetson-Inference / EfficientViT demos).  
- **`utils/`** — tiny utilities for **function signature dump** and **module path inspection** (handy for CPython bindings).

> Tip: Before running code that relies on system CUDA OpenCV / TensorRT, ensure your env is properly bridged. See  
> [`check_jetson_env_conflicts.md`](../docs/check_jetson_env_conflicts.md) and [`jetson_system_bridge.md`](../docs/jetson_system_bridge.md).

---

### 1) `cuda_manipulation/` — simple CUDA manipulation example
- **Purpose:** Minimal examples to sanity-check CUDA availability and perform basic ops.
- **Contents:** `simple_cuda_manipulation.py`, `example/` (small demo snippets).
- **Quick run:**
  ```bash
  cd cuda_manipulation
  python simple_cuda_manipulation.py --help
  # Example
  python simple_cuda_manipulation.py --list-devices
  ```
- **Notes:** Works on Jetson (JetPack 5.1.x) and x86_64 CUDA machines.

---

### 2) `jetson_inference_patch/` — build `jetson-inference` on conda
- **Problem it solves:** Building `jetson-inference` in a **conda env** can fail due to library path and CMake Python hints.
- **What it changes:** Adds environment hooks and patches so system CUDA/TRT/OpenCV are correctly discovered.
- **Layout:**  
  - `00_cleanup_global_jetson.sh` — clean up conflicting global installs.  
  - `01_env_hooks_ldpath.sh` — export **LD_LIBRARY_PATH** and related envs.  
  - `02_build_install.sh` — configure/build/install steps.  
  - `patches/` — small CMake/cpp patch set.  
  - `README.md` — step-by-step usage.
- **Minimal usage:**
  ```bash
  cd jetson_inference_patch
  bash 00_cleanup_global_jetson.sh
  bash 01_env_hooks_ldpath.sh
  bash 02_build_install.sh
  ```
- **See also:** [`check_jetson_env_conflicts.md`](../docs/check_jetson_env_conflicts.md), [`jetson_system_bridge.md`](../docs/jetson_system_bridge.md).

---

### 3) `remote_streaming/` — remote video streaming demos
- **Purpose:** End-to-end examples for **RTSP/WebRTC** style streaming and consuming on Jetson.
- **Subfolders:**  
  - `01_mediamtx/` — MediaMTX server configs & scripts.  
  - `02_jetson_inference/` — Jetson-Inference based streaming/overlay scripts.  
  - `03_efficientViT/` — Lightweight model demo for streaming pipelines.
- **Quick start:**
  ```bash
  cd remote_streaming/01_mediamtx
  # start MediaMTX server (edit config as needed)
  ./run_mediamtx.sh
  ```
  Then open the corresponding client example under `02_jetson_inference/` or `03_efficientViT/`.

---

### 4) `utils/` — tiny helpers for bindings & debugging
- **`show_func_args.py`** — print a function’s signature and default values (useful for CPython-bound methods).  
  ```bash
  python utils/show_func_args.py --module some_pkg.some_mod --func some_func
  ```
- **`show_module_dir.py`** — print module load paths (diagnose **which `cv2`/`torchvision`** is loaded).  
  ```bash
  python utils/show_module_dir.py cv2 torchvision tensorrt
  ```

---

### Prerequisites (tested)
- **Hardware:** Jetson Xavier NX (reference) / x86_64 CUDA GPU  
- **OS:** Ubuntu 20.04 (L4T r35.6.x) / JetPack **5.1.5**  
- **Python:** 3.8  
- **Key libs:** CUDA 11.4 + cuDNN 8.x (JetPack), PyTorch 2.1 / torchvision 0.16.x, CUDA OpenCV 4.10+

---

## 한국어

### 이 폴더 소개
가이드와 함께 제공되는 **작고 집중된 예제/유틸리티** 모음입니다.

- **`cuda_manipulation/`** — CUDA 장치 확인과 간단한 연산 예제.  
- **`jetson_inference_patch/`** — Jetson에서 **conda 환경으로 `jetson-inference` 빌드** 시 필요한 패치/스크립트.  
- **`remote_streaming/`** — **원격 스트리밍**(MediaMTX / Jetson-Inference / EfficientViT) 예제.  
- **`utils/`** — **함수 시그니처/모듈 경로** 확인용 초소형 도구(CPython 바인딩 디버깅에 유용).

> 참고: 시스템 CUDA OpenCV/TensorRT를 사용하려면 환경 브리지가 필요할 수 있습니다.  
> [`check_jetson_env_conflicts.md`](../docs/check_jetson_env_conflicts.md), [`jetson_system_bridge.md`](../docs/jetson_system_bridge.md)를 먼저 확인하세요.

---

### 1) `cuda_manipulation/` — 간단 CUDA 예제
- **목적:** CUDA 사용 가능 여부 및 기본 연산을 빠르게 검증.  
- **구성:** `simple_cuda_manipulation.py`, `example/`(데모).  
- **실행:**
  ```bash
  cd cuda_manipulation
  python simple_cuda_manipulation.py --help
  python simple_cuda_manipulation.py --list-devices
  ```

---

### 2) `jetson_inference_patch/` — conda에서 `jetson-inference` 빌드
- **해결 대상:** conda 환경에서의 라이브러리 경로/파이썬 힌트 문제로 인한 빌드 실패.  
- **변경 내용:** 시스템 CUDA/TRT/OpenCV를 올바르게 찾도록 env hook/패치를 적용.  
- **구성:** `00_cleanup_global_jetson.sh`, `01_env_hooks_ldpath.sh`, `02_build_install.sh`, `patches/`, `README.md`.  
- **사용 예시:**
  ```bash
  cd jetson_inference_patch
  bash 00_cleanup_global_jetson.sh
  bash 01_env_hooks_ldpath.sh
  bash 02_build_install.sh
  ```
- **관련 문서:** [`check_jetson_env_conflicts.md`](../docs/check_jetson_env_conflicts.md), [`jetson_system_bridge.md`](../docs/jetson_system_bridge.md).

---

### 3) `remote_streaming/` — 원격 스트리밍 데모
- **목적:** Jetson에서의 **RTSP/WebRTC** 스트리밍 파이프라인 예시 제공.  
- **하위 폴더:** `01_mediamtx/`, `02_jetson_inference/`, `03_efficientViT/`.  
- **빠른 시작:**
  ```bash
  cd remote_streaming/01_mediamtx
  ./run_mediamtx.sh
  ```
  이후 `02_jetson_inference/` 또는 `03_efficientViT/`의 클라이언트 예제를 실행하세요.

---

### 4) `utils/` — 바인딩/디버깅용 초소형 도구
- **`show_func_args.py`** — 함수 시그니처와 기본값 출력(CPython 바인딩 함수 확인에 유용).  
  ```bash
  python utils/show_func_args.py --module some_pkg.some_mod --func some_func
  ```
- **`show_module_dir.py`** — 모듈 로드 경로 출력(**어느 `cv2`/`torchvision`이 로드되는지** 확인).  
  ```bash
  python utils/show_module_dir.py cv2 torchvision tensorrt
  ```

---

### 사양(테스트 기준)
- **하드웨어:** Jetson Xavier NX / x86_64 CUDA GPU  
- **OS:** Ubuntu 20.04 (L4T r35.6.x) / JetPack **5.1.5**  
- **Python:** 3.8  
- **핵심 라이브러리:** CUDA 11.4 + cuDNN 8.x (JetPack), PyTorch 2.1 / torchvision 0.16.x, CUDA OpenCV 4.10+
