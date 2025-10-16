[ [English](#english) | [한국어](#한국어) ]

# Jetson + Conda Conflict Checker — `check_jetson_env_conflicts.sh`
**Up one level:** [`../README.md`](../README.md)

> Location: `ir-guide/check_jetson_env_conflicts.sh`

---

## English

### Purpose
Catch the **most common conflicts** when using a conda env with **JetPack’s system CUDA/cuDNN/TensorRT** and a
system‑built **OpenCV (CUDA)** on Jetson.

This version aligns with the latest script and adds robust detection and clearer remediation tips.

**The checker now:**  
- Detects **duplicate CUDA** inside conda (`cudatoolkit`, `cudnn`, `pytorch-cuda`, `cublas`, `cusolver`, `cufft`, `cutensor`)  
- Flags pip **OpenCV wheels** using `importlib.metadata` with a **safe fallback** to `pip list`  
- Reports **`cv2` import path**, whether **CUDA is enabled**, and **CUDA device count**  
- Verifies **PyTorch ↔ torchvision** compatibility via an **explicit mapping**  
  - Examples: **2.0.x ↔ 0.15.1**, **2.1.x ↔ 0.16.x**, **2.2.x ↔ 0.17.1**, **2.3.x ↔ 0.18.0**  
- Checks **CUDA_HOME/nvcc** presence (expects **system JetPack** CUDA at `/usr/local/cuda`)  
- Ensures **LD_LIBRARY_PATH** exposes Jetson libs (`/usr/lib/aarch64-linux-gnu/tegra`)  
- Warns if **Python 3.8** is paired with **too‑new pip/setuptools** for build isolation  
- Returns **non‑zero exit codes** per detected issue (CI‑friendly)  
- Suggests using **`jetson_system_bridge.sh`** when `cv2` import fails or wheel shadowing is detected

### Quick usage
```bash
# Run directly
bash check_jetson_env_conflicts.sh

# Or target a specific env
conda run -n <env> bash check_jetson_env_conflicts.sh
```

- Returns **0** if no problems found  
- Returns **non‑zero** if issues are detected (CI‑friendly)

### Typical output (abridged)
```
=== Jetson/Conda conflict checker ===
[i] CONDA_PREFIX: /home/user/miniforge3/envs/jp515
[i] Python      : 3.8.20
[OK] No conda CUDA/cuDNN packages detected.
[!] Found pip OpenCV wheels: opencv-python-headless
    -> Fix: pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python
[OK] cv2 path     : /usr/local/lib/python3.8/dist-packages/cv2/...
[OK] cv2 CUDA     : YES
[OK] cv2 CUDA devs: 1
[OK] torch        : 2.1.0+nv... | CUDA: True
[OK] torchvision  : 0.16.1
[OK] CUDA_HOME=/usr/local/cuda (nvcc present)
[!] tegra libs not in LD_LIBRARY_PATH.
    -> Fix: export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu/tegra:${LD_LIBRARY_PATH:-}
[OK] pip=24.2, setuptools=68.2 look safe for Py3.8.
=== Detected 2 potential issue(s). ⚠️
```

### Quick fixes the script suggests
- Remove duplicate CUDA in conda:  
  `conda remove --force cudatoolkit cudnn cublas cusolver cufft cutensor pytorch-cuda`
- Uninstall pip OpenCV wheels (prevent shadowing system CUDA OpenCV):  
  `pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python`
- Export CUDA anchors (if missing):  
  `export CUDA_HOME=/usr/local/cuda; export PATH=$CUDA_HOME/bin:$PATH`
- Expose Jetson libs to loader:  
  `export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu/tegra:$LD_LIBRARY_PATH`
- Py3.8 toolchain pins:  
  `python -m pip install "pip<25" "setuptools<75" -U`
- If `cv2` import fails or is non‑CUDA:  
  Run **`jetson_system_bridge.sh`** to prioritize system Python packages (cv2, tensorrt, pycuda, …)

### Mapping used for torch/torchvision
The checker accepts the following pairs (examples, not exhaustive):
- **1.13.x ↔ 0.14.1**  
- **2.0.x  ↔ 0.15.1**  
- **2.1.x  ↔ 0.16.0 / 0.16.1**  
- **2.2.x  ↔ 0.17.1**  
- **2.3.x  ↔ 0.18.0**

### Integration tips
- Run this checker **right after** creating an env or **after** installing PyTorch/vision/OpenCV.  
- In your setup flow, run `jetson_system_bridge.sh` first, then run this checker.  
- Use the **exit status** to gate downstream steps in CI/student lab machines.

### Troubleshooting
- **`cv2` loads from your conda site‑packages and reports `CUDA: NO`** → Remove pip wheels and run `jetson_system_bridge.sh`.  
- **`torch/vision` mismatch** → Reinstall a matching torchvision (or torch) build per the mapping above.  
- **`nvcc` not found** → Set `CUDA_HOME=/usr/local/cuda` and update `PATH`.  
- **TRT/PyCUDA import fails** → Ensure JetPack TRT/python bindings are installed and `LD_LIBRARY_PATH` exposes `/usr/lib/aarch64-linux-gnu/tegra`.

---

## 한국어

### 목적
JetPack의 **시스템 CUDA/cuDNN/TensorRT** 및 **CUDA 빌드 OpenCV**와 conda 환경을 함께 사용할 때 발생하는
**대표적 충돌**을 빠르게 찾습니다. 최신 스크립트 기준으로 **감지 로직 강화**와 **해결 가이드**가 업데이트되었습니다.

**주요 점검 항목**  
- conda 내부 **중복 CUDA**(`cudatoolkit`, `cudnn`, `pytorch-cuda`, `cublas`, `cusolver`, `cufft`, `cutensor`) 탐지  
- pip **OpenCV 휠**을 `importlib.metadata`로 우선 탐지(필요 시 `pip list` **폴백**)  
- **`cv2` import 경로**, **CUDA 활성화 여부**, **CUDA 디바이스 수** 보고  
- **PyTorch ↔ torchvision** **버전 매핑** 확인  
  - 예: **2.0.x ↔ 0.15.1**, **2.1.x ↔ 0.16.x**, **2.2.x ↔ 0.17.1**, **2.3.x ↔ 0.18.0**  
- **CUDA_HOME/nvcc** 확인(기본: `/usr/local/cuda`)  
- **LD_LIBRARY_PATH**에 Jetson 라이브러리(`.../aarch64-linux-gnu/tegra`) 포함 여부 검사  
- **Python 3.8**에서 **pip/setuptools 과신버전** 경고  
- 문제 시 **비‑0 종료코드**로 반환 (CI에 유용)  
- `cv2` 실패/비‑CUDA인 경우 **`jetson_system_bridge.sh`** 사용을 제안

### 사용법
```bash
bash check_jetson_env_conflicts.sh
# 또는 특정 env에 대해:
conda run -n <env> bash check_jetson_env_conflicts.sh
```

- 문제가 없으면 **0** 반환  
- 문제가 있으면 **0 이외의 값** 반환 (CI용으로 적합)

### 스크립트가 제안하는 빠른 해결책
- conda 내부 중복 CUDA 제거:  
  `conda remove --force cudatoolkit cudnn cublas cusolver cufft cutensor pytorch-cuda`
- pip OpenCV 휠 제거(시스템 CUDA OpenCV 가리기 방지):  
  `pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python`
- CUDA 앵커 설정:  
  `export CUDA_HOME=/usr/local/cuda; export PATH=$CUDA_HOME/bin:$PATH`
- Jetson 라이브러리 경로 노출:  
  `export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu/tegra:$LD_LIBRARY_PATH`
- Python 3.8 도구 체인 고정:  
  `python -m pip install "pip<25" "setuptools<75" -U`
- `cv2` import 실패/비‑CUDA일 때:  
  **`jetson_system_bridge.sh`**로 시스템 패키지(cv2, tensorrt, pycuda 등)를 우선하도록 설정

### torch/torchvision 허용 매핑(예시)
- **1.13.x ↔ 0.14.1**  
- **2.0.x  ↔ 0.15.1**  
- **2.1.x  ↔ 0.16.0 / 0.16.1**  
- **2.2.x  ↔ 0.17.1**  
- **2.3.x  ↔ 0.18.0**

### 통합 팁
- env 생성 직후, 또는 PyTorch/vision/OpenCV 설치 후 **바로 실행**하세요.  
- 보통 **`jetson_system_bridge.sh` 실행 → 본 체크 실행** 순서를 권장합니다.  
- **종료코드**를 이용해 이후 단계 진행 여부를 제어(CI/교육환경)할 수 있습니다.