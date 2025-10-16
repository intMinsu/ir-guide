[ [English](#english) | [한국어](#한국어) ]

# Jetson Dev Stack Installer — `install_jetson_dev_packages.sh`
**Up one level:** [`../README.md`](../README.md)
> Location: `ir-guide/scripts/install_jetson_dev_packages.sh`

---

## English

### Why this script matters
For classes and labs, provisioning multiple Jetson Xavier NX boards is time‑consuming and error‑prone.  
This installer standardizes the **developer stack** on JetPack **5.1.5** (L4T r35.6.x), avoiding common pitfalls:
- Installs via **NVIDIA meta‑packages** (`nvidia-jetpack*`) so CUDA/cuDNN/TensorRT versions **match** L4T.
- Avoids duplicate CUDA in userspace (no `cudatoolkit` from conda/apt).
- Builds **OpenCV with CUDA** once, in `/usr/local`, visible to all users/environments.
- Optional **DeepStream** and **container runtime** for downstream projects.

### What it installs (by menu item)
1. **Max performance** — picks strongest `nvpmodel` mode, runs `jetson_clocks` (best for building).  
2. **GitHub CLI (gh)** — adds repo keyring + installs `gh`.  
3. **jtop** — installs jetson-stats (`jtop`) for monitoring.  
4. **Jetson runtime** — `nvidia-jetpack` (or `nvidia-jetpack-runtime` via `JETPACK_FLAVOR`), plus `nvidia-container-toolkit`.  
5. **OpenCV (CUDA/FFmpeg/GStreamer)** — builds OpenCV (default `4.10.0`) with CUDA and media I/O.  
6. **DeepStream** — optional install (default `6.3`) from NVIDIA repo.

> The script is **idempotent**: re-running a step won’t break the system (it may just say “already installed”).

### Prerequisites
- Run **on the Jetson device** (Ubuntu 20.04, JP 5.1.5).  
- Internet connection and `sudo` privileges.  
- For step **5 (OpenCV build)**: recommend **MaxN** power + swap (or zram enabled).

---

## Quick usage

### Interactive (recommended for students)
```bash
cd scripts
./install_jetson_dev_packages.sh          # choose 1..6 (default: 1,2,3,4,5)
```

### Non‑interactive (CI / batch)
```bash
# Example: max perf + JetPack runtime + OpenCV
./install_jetson_dev_packages.sh 1,4,5

# Full dev stack (1..6)
./install_jetson_dev_packages.sh 1,2,3,4,5,6
```

### Environment variables
```bash
# Choose runtime vs dev for JetPack (affects step 4)
JETPACK_FLAVOR=dev|runtime          # default: dev (full nvidia-jetpack)

# OpenCV build options (step 5)
OPENCV_VER=4.10.0                   # version tag
JOBS=4                              # make -j
# Optional extra CMake flags:
OPENCV_CMAKE_EXTRA="-D WITH_OPENMP=ON"

# DeepStream version (step 6)
DS_VER=6.3
```

### Verification
```bash
# CUDA / cuDNN / TensorRT present via JetPack
nvcc --version
ldconfig -p | egrep 'cudnn|nvinfer'

# Container runtime
nvidia-ctk --version || nvidia-container-runtime --version
docker info | grep -i nvidia || true

# OpenCV build has CUDA
python3 - <<'PY'
import cv2, sys
print("OpenCV:", cv2.__version__)
print("CUDA devices:", cv2.cuda.getCudaEnabledDeviceCount())
PY

# jtop
jtop --version || echo "run: sudo -E jtop"
```

---

## Troubleshooting

- **`E: Unable to locate package cuda-toolkit`**  
  On JP 5.x, packages are versioned (e.g., `cuda-toolkit-11-4`). Use `nvidia-jetpack` (or `nvidia-jetpack-runtime`) instead.

- **`E: Unable to locate package nvidia-jetpack`**  
  After flashing, run `sudo apt update`. Ensure NVIDIA L4T apt sources exist under `/etc/apt/sources.list.d/` and match r35.6.x.

- **OpenCV build takes too long / OOM**  
  Enable **Max performance** (step 1), reduce `JOBS` (e.g., `JOBS=2`), or add swap. Close browsers / heavy apps.

- **DeepStream package missing**  
  Make sure you’re on JetPack 5.x with NVIDIA repo set; run `sudo apt update`. Some locales require license acknowledgement.

- **Docker not picking up NVIDIA runtime**  
  Run `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`.

- **Re-run a specific step**  
  Re-invoke the script with just that number: e.g., `./install_jetson_dev_packages.sh 5` (OpenCV).

---

## Uninstall notes (advanced)
- JetPack meta: `sudo apt remove --purge 'nvidia-jetpack*'` (not recommended on active systems).  
- OpenCV (from source): remove `/usr/local/{lib,include}/opencv*` and `/usr/local/bin/opencv_*` (careful!).  
- DeepStream: `sudo apt remove --purge 'deepstream*'`.

---

## 한국어

### 스크립트의 목적
여러 대의 Jetson을 빠르게 **표준 환경**으로 맞추기 위한 설치기입니다.  
JetPack **5.1.5**(L4T r35.6.x)에 맞춰 CUDA/cuDNN/TensorRT를 **공식 메타 패키지**로 설치하고,  
**CUDA OpenCV**를 `/usr/local`에 빌드하여 모든 사용자/환경에서 활용 가능하도록 합니다.

### 메뉴 구성 (설치 항목)
1. **최대 성능 모드** — 가장 강한 `nvpmodel` + `jetson_clocks` 적용  
2. **GitHub CLI (gh)** 설치  
3. **jtop** 설치(모니터링)  
4. **Jetson 런타임** — `nvidia-jetpack`(또는 `nvidia-jetpack-runtime`) + 컨테이너 런타임  
5. **OpenCV (CUDA/FFmpeg/GStreamer)** 빌드 (`/usr/local` 설치)  
6. **DeepStream** 설치(선택)

### 사전 준비
- **Jetson 기기**에서 실행 (Ubuntu 20.04, JP 5.1.5)  
- 인터넷 연결, `sudo` 권한  
- (5) OpenCV 빌드 시 **최대 성능** 권장 + 스왑(zram) 사용 권장

---

## 사용법

### 인터랙티브 모드
```bash
cd scripts
./install_jetson_dev_packages.sh          # 1..6 선택 (기본: 1,2,3,4,5)
```

### 비대화식 (배치/자동화)
```bash
./install_jetson_dev_packages.sh 1,4,5
./install_jetson_dev_packages.sh 1,2,3,4,5,6
```

### 환경변수
```bash
JETPACK_FLAVOR=dev|runtime          # 4번 단계에 영향, 기본: dev
OPENCV_VER=4.10.0                   # OpenCV 버전
JOBS=4                              # make -j 병렬 빌드 수
OPENCV_CMAKE_EXTRA="-D WITH_OPENMP=ON"   # 선택 CMake 옵션
DS_VER=6.3                          # DeepStream 버전
```

### 동작 확인
```bash
nvcc --version
ldconfig -p | egrep 'cudnn|nvinfer'

python3 - <<'PY'
import cv2, sys
print("OpenCV:", cv2.__version__)
print("CUDA devices:", cv2.cuda.getCudaEnabledDeviceCount())
PY

jtop --version || echo "sudo -E jtop 로 실행하세요"
```

---

## 문제 해결
- **`cuda-toolkit` 패키지 없음** → JP 5.x는 버전 명시형(`cuda-toolkit-11-4`). `nvidia-jetpack` 사용 권장.  
- **`nvidia-jetpack` 미발견** → `sudo apt update`; L4T 저장소(r35.6.x) 확인.  
- **OpenCV 빌드 느림/메모리 부족** → 1번 수행 후 `JOBS` 낮추기, 스왑 사용.  
- **DeepStream 패키지 없음** → NVIDIA 저장소/지역 설정 확인, `sudo apt update`.  
- **Docker가 NVIDIA 런타임 인식 못함** → `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`.

---

## 제거(주의)
- JetPack 메타: `sudo apt remove --purge 'nvidia-jetpack*'` (활성 시스템에서는 비권장)  
- OpenCV(소스 설치): `/usr/local`의 관련 파일 수동 삭제(주의)  
- DeepStream: `sudo apt remove --purge 'deepstream*'`
