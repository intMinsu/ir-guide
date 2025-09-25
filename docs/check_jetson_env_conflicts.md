[ [English](#english) | [한국어](#한국어) ]

# Jetson + Conda Conflict Checker — `check_jetson_env_conflicts.sh`

> Location: `./check_jetson_env_conflicts.sh`

---

## English

### Purpose
Catch the **most common conflicts** when using a conda env with **JetPack’s system CUDA/cuDNN/TRT** and a
system‑built **OpenCV (CUDA)** on Jetson.

The checker:
- Scans for **duplicate CUDA** inside conda (`cudatoolkit`, `cudnn`, `pytorch-cuda`, etc.)  
- Flags pip **OpenCV wheels** inside the env that can shadow your `/usr/local` CUDA build  
- Reports where `cv2` is imported from and whether **CUDA** is enabled  
- Verifies **PyTorch ↔ torchvision** compatibility via **mapping** (e.g., 2.1.x ↔ 0.16.x)  
- Checks **CUDA_HOME/nvcc** and **LD_LIBRARY_PATH** for Jetson libs (`…/tegra`)  
- Warns if **pip/setuptools** look too new for **Python 3.8** build isolation

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
[i] CONDA_PREFIX: /home/user/miniforge3/envs/jp515
[i] Python      : 3.8.20
[OK] No conda CUDA/cuDNN packages detected.
[!] Found pip OpenCV wheels that can override your /usr/local CUDA build.
    -> Fix: pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python
[OK] cv2 path     : /usr/local/lib/python3.8/dist-packages/cv2/...
[OK] cv2 CUDA     : YES
[OK] torch        : 2.1.0a0+... | CUDA: True
[OK] torchvision  : 0.16.1
[OK] CUDA_HOME=/usr/local/cuda (nvcc present)
[!] tegra libs not in LD_LIBRARY_PATH.
    -> Fix: export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu/tegra:${LD_LIBRARY_PATH:-}
=== Detected 2 potential issue(s). ⚠️
```

### Quick fixes the script suggests
- Remove duplicate CUDA in conda:  
  `conda remove --force cudatoolkit cudnn cublas cusolver cufft cutensor pytorch-cuda`
- Uninstall pip OpenCV wheels:  
  `pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python`
- Export CUDA anchors (if missing):  
  `export CUDA_HOME=/usr/local/cuda; export PATH=$CUDA_HOME/bin:$PATH`
- Expose Jetson libs to loader:  
  `export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu/tegra:$LD_LIBRARY_PATH`
- Downgrade build tools on Py3.8:  
  `python -m pip install "pip<25" "setuptools<75" -U`

### Integrating with the setup script
- You can run this checker **after** `safe_set_pytorch_conda.sh`, or call it in CI to validate student machines.  
- Exit code is non‑zero on problems, so you can gate subsequent steps.

---

## 한국어

### 목적
JetPack의 **시스템 CUDA/cuDNN/TRT** 및 **CUDA 빌드 OpenCV**와 conda 환경을 함께 사용할 때 발생하는
**대표적인 충돌**을 빠르게 찾아냅니다.

점검 항목:
- conda 내부 **중복 CUDA** (`cudatoolkit`, `cudnn`, `pytorch-cuda` 등) 존재 여부  
- env 내부 pip **OpenCV 휠**이 `/usr/local` OpenCV를 가리는지  
- `cv2`의 실제 import 경로 및 **CUDA 활성화** 여부  
- **PyTorch ↔ torchvision** 버전 **매핑** 검사 (예: 2.1.x ↔ 0.16.x)  
- **CUDA_HOME/nvcc** 및 **LD_LIBRARY_PATH** 설정 확인 (`…/tegra` 포함 여부)  
- **Python 3.8**에서 **pip/setuptools**가 너무 최신인지 경고

### 사용법
```bash
bash check_jetson_env_conflicts.sh
# 또는 특정 env에 대해:
conda run -n <env> bash check_jetson_env_conflicts.sh
```

- 문제가 없으면 **0** 반환  
- 문제가 있으면 **0 이외의 값** 반환 (CI에 적합)

### 스크립트가 제안하는 빠른 해결책
- conda 내부 중복 CUDA 제거:  
  `conda remove --force cudatoolkit cudnn cublas cusolver cufft cutensor pytorch-cuda`
- pip OpenCV 휠 제거:  
  `pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python`
- CUDA 앵커 설정:  
  `export CUDA_HOME=/usr/local/cuda; export PATH=$CUDA_HOME/bin:$PATH`
- Jetson 라이브러리 경로 노출:  
  `export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu/tegra:$LD_LIBRARY_PATH`
- Py3.8에서 빌드 도구 다운그레이드:  
  `python -m pip install "pip<25" "setuptools<75" -U`

### 설정 스크립트와 함께 쓰기
- `safe_set_pytorch_conda.sh` 실행 후에 검사하거나, CI에서 학생 PC를 자동 검증할 수 있습니다.  
- 종료코드가 0이 아니면 이후 단계 진행을 중단하도록 구성할 수 있습니다.

---

_Back to:_ **[Root README](../README.md)** · **[Scripts](../scripts/README.md)** · **[Setup guide](./safe_set_pytorch_conda.md)**
