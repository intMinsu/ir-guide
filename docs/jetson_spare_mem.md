[ [English](#english) | [한국어](#한국어) ]
# `jetson_spare_mem.sh` — Free RAM Quickly for Builds & Inference
**Up one level:** [`../README.md`](../README.md)
> Location: `ir-guide/scripts/jetson_spare_mem.sh`
---

## English

### What is this?
`jetson_spare_mem.sh` is a small helper to **free memory on Jetsons** (Nano/TX2/Xavier/Orin) and **run low‑RAM builds/inference safely**. Jetsons use **unified memory**—CPU and GPU share DRAM—so turning off the desktop and managing swap/parallelism directly benefits CUDA workloads.

**Key features**
- One‑command **headless toggle** (stop/start GUI to free hundreds of MB).
- **Swapfile management** (enable/disable/tune) as a safety net during large links.
- **Build caps**: print or wrap commands with safe env vars (`MAX_JOBS`, `CMAKE_BUILD_PARALLEL_LEVEL`, etc.).
- Simple **status** view (RAM, swap, boot target).

> **Swap ≠ speed‑up.** It prevents OOM kills; if you actually *use* swap heavily, builds get slower. Aim to keep swap activity near zero while using conservative parallelism.

---

### Prerequisites
- **Hardware:** Jetson Nano/TX2/Xavier/Orin (4–32 GB)
- **OS:** Ubuntu 20.04/22.04 (L4T r35.x/r36.x), systemd available
- **Privileges:** `sudo` for headless/swap operations

---

### Quick Start
```bash
cd scripts
chmod +x jetson_spare_mem.sh

# 1) Free RAM right now (temporary, for this boot)
./jetson_spare_mem.sh headless on

# 2) Add 8G swap (prefer NVMe path if you have it)
./jetson_spare_mem.sh swap enable 8G /swapfile
./jetson_spare_mem.sh swap tune 10

# 3) Build with conservative caps (4 GB profile)
./jetson_spare_mem.sh wrap 4g -- pip install --no-build-isolation -v .

# 4) Bring GUI back when done
./jetson_spare_mem.sh headless off
```

If you prefer to export env vars into your current shell:
```bash
eval "$(./jetson_spare_mem.sh caps print 8g)"
# then build normally
pip install --no-build-isolation -v .
```

---

### Commands
| Command | Description |
|---|---|
| `status` | Show RAM/swap and active/boot targets. |
| `headless on` | Stop desktop (non‑graphical mode) for this boot. |
| `headless off` | Start desktop for this boot. |
| `headless persist-on` | Set default boot to console (survives reboot). |
| `headless persist-off` | Set default boot to GUI (survives reboot). |
| `swap enable [SIZE] [PATH]` | Create & enable a swapfile (default `8G /swapfile`). |
| `swap disable [PATH]` | Disable & remove a swapfile (default `/swapfile`). |
| `swap tune [SWAPPINESS]` | Set `vm.swappiness` (default recommended `10`). |
| `swap show` | Same as `status` (quick view). |
| `caps print [4g|8g|16g|auto]` | Print recommended env caps; use with `eval`. |
| `wrap [4g|8g|16g|auto] -- CMD ...` | Run `CMD` with env caps applied in a clean shell. |

---

### Recommended caps by RAM
> These are printed by `caps print` and used by `wrap`.

**4 GB** (Nano / some TX2):  
```
export MAX_JOBS=1
export CMAKE_BUILD_PARALLEL_LEVEL=1
export USE_NINJA=0
export CFLAGS="-O2 -g0"; export CXXFLAGS="-O2 -g0"; export LDFLAGS="-Wl,--no-keep-memory"
export TORCHVISION_USE_FFMPEG=0
export TORCHVISION_USE_VIDEO_CODEC=0
# export TORCH_CUDA_ARCH_LIST=5.3   # Nano (use 6.2 for TX2)
```

**8 GB** (Xavier NX 8G):  
```
export MAX_JOBS=2
export CMAKE_BUILD_PARALLEL_LEVEL=2
export USE_NINJA=1
export CFLAGS="-O2 -g0"; export CXXFLAGS="-O2 -g0"; export LDFLAGS="-Wl,--no-keep-memory"
# export TORCH_CUDA_ARCH_LIST=7.2   # Xavier
```

**16 GB+** (Orin 16G/32G):  
```
export MAX_JOBS=2-4
export CMAKE_BUILD_PARALLEL_LEVEL=2-4
export USE_NINJA=1
export CFLAGS="-O2 -g0"; export CXXFLAGS="-O2 -g0"; export LDFLAGS="-Wl,--no-keep-memory"
# export TORCH_CUDA_ARCH_LIST=8.7   # Orin
```

---

### Typical workflows

**A) Build heavy C++/CUDA (e.g., torchvision)**
```bash
./jetson_spare_mem.sh headless on
./jetson_spare_mem.sh swap enable 8G /swapfile
./jetson_spare_mem.sh wrap 8g -- python -m pip install --no-build-isolation -v .
./jetson_spare_mem.sh headless off
```

**B) Model inference with limited RAM**
- Use smaller batches; reuse tensors/buffers.
- Prefer FP16 on Xavier/Orin; Nano/TX2 may benefit in memory only.
- Avoid many dataloader workers (`num_workers=0–2`, `pin_memory=False`).

---

### Monitoring & troubleshooting
```bash
tegrastats                # overall Jetson health
free -h                   # memory snapshot
vmstat 1                  # si/so columns (swap in/out) — keep near 0
top                       # %wa (I/O wait) — high means swap thrashing
./jetson_spare_mem.sh status
dmesg | egrep -i 'killed process|out of memory' | tail -n5
```
**If you see swap thrashing or OOM kills:** lower parallelism (`MAX_JOBS=1`) and keep GUI off during builds.

---

### Cleanup / revert
```bash
# Remove swapfile
./jetson_spare_mem.sh swap disable /swapfile

# Restore GUI default on boot
./jetson_spare_mem.sh headless persist-off
```

---

## 한국어

### 개요
`jetson_spare_mem.sh`는 Jetson(Nano/TX2/Xavier/Orin)에서 **메모리를 즉시 확보**하고, **저메모리 환경에서 안전하게 빌드/추론**할 수 있도록 돕는 스크립트입니다. Jetson은 **통합 메모리**(CPU/GPU가 DRAM 공유)를 사용하므로, 데스크톱을 끄고 스왑/병렬도를 관리하면 CUDA 작업에도 직접적인 이점이 있습니다.

**핵심 기능**
- 명령 한 줄로 **헤드리스 전환**(GUI 중지/재개 → 수백 MB 확보)
- **스왑파일 관리**(생성/삭제/튜닝) – 대규모 링크 시 OOM 방지용 안전망
- **빌드 캡**: 안전한 환경 변수(`MAX_JOBS`, `CMAKE_BUILD_PARALLEL_LEVEL` 등) 출력/적용
- 간단한 **상태 조회**(RAM/스왑/부팅 타깃)

> **스왑 = 속도 향상 아님.** OOM을 막아줄 뿐, 스왑을 많이 사용하면 빌드는 느려집니다. 스왑 트래픽을 0에 가깝게 유지하고 보수적인 병렬도를 쓰세요.

---

### 요구 사항
- **하드웨어:** Jetson Nano/TX2/Xavier/Orin (4–32 GB)
- **OS:** Ubuntu 20.04/22.04 (L4T r35.x/r36.x), systemd 사용
- **권한:** `sudo` (헤드리스/스왑 작업에 필요)

---

### 빠른 시작
```bash
cd scripts
chmod +x jetson_spare_mem.sh

# 1) 즉시 메모리 확보 (이번 부팅 동안만)
./jetson_spare_mem.sh headless on

# 2) 8G 스왑 추가 (가능하면 NVMe 경로 권장)
./jetson_spare_mem.sh swap enable 8G /swapfile
./jetson_spare_mem.sh swap tune 10

# 3) 보수적 캡으로 빌드 (4 GB 프로필)
./jetson_spare_mem.sh wrap 4g -- pip install --no-build-isolation -v .

# 4) 작업 후 GUI 복구
./jetson_spare_mem.sh headless off
```

셸에 직접 환경 변수 적용을 원하면:
```bash
eval "$(./jetson_spare_mem.sh caps print 8g)"
pip install --no-build-isolation -v .
```

---

### 명령어
| 명령 | 설명 |
|---|---|
| `status` | RAM/스왑, 현재/기본 타깃 조회 |
| `headless on` | 이번 부팅 동안 데스크톱 중지(비그래픽 모드) |
| `headless off` | 이번 부팅 동안 데스크톱 시작 |
| `headless persist-on` | 부팅 기본값을 콘솔로 설정(재부팅 후에도 유지) |
| `headless persist-off` | 부팅 기본값을 GUI로 설정(재부팅 후에도 유지) |
| `swap enable [SIZE] [PATH]` | 스왑파일 생성/활성화 (기본 `8G /swapfile`) |
| `swap disable [PATH]` | 스왑파일 비활성/삭제 (기본 `/swapfile`) |
| `swap tune [SWAPPINESS]` | `vm.swappiness` 설정 (권장 기본 `10`) |
| `swap show` | `status`와 동일(빠른 확인) |
| `caps print [4g|8g|16g|auto]` | 권장 환경 변수 출력 (`eval`과 함께 사용) |
| `wrap [4g|8g|16g|auto] -- CMD ...` | 깨끗한 셸에서 `CMD`를 캡 적용하여 실행 |

---

### RAM 용량별 권장 캡
> 아래 값은 `caps print`로 출력되며, `wrap` 명령 내부에서도 사용됩니다.

**4 GB** (Nano / 일부 TX2):  
```
export MAX_JOBS=1
export CMAKE_BUILD_PARALLEL_LEVEL=1
export USE_NINJA=0
export CFLAGS="-O2 -g0"; export CXXFLAGS="-O2 -g0"; export LDFLAGS="-Wl,--no-keep-memory"
export TORCHVISION_USE_FFMPEG=0
export TORCHVISION_USE_VIDEO_CODEC=0
# export TORCH_CUDA_ARCH_LIST=5.3   # Nano (TX2는 6.2 권장)
```

**8 GB** (Xavier NX 8G):  
```
export MAX_JOBS=2
export CMAKE_BUILD_PARALLEL_LEVEL=2
export USE_NINJA=1
export CFLAGS="-O2 -g0"; export CXXFLAGS="-O2 -g0"; export LDFLAGS="-Wl,--no-keep-memory"
# export TORCH_CUDA_ARCH_LIST=7.2   # Xavier
```

**16 GB 이상** (Orin 16G/32G):  
```
export MAX_JOBS=2-4
export CMAKE_BUILD_PARALLEL_LEVEL=2-4
export USE_NINJA=1
export CFLAGS="-O2 -g0"; export CXXFLAGS="-O2 -g0"; export LDFLAGS="-Wl,--no-keep-memory"
# export TORCH_CUDA_ARCH_LIST=8.7   # Orin
```

---

### 자주 쓰는 워크플로우

**A) 대규모 C++/CUDA 빌드 (예: torchvision)**
```bash
./jetson_spare_mem.sh headless on
./jetson_spare_mem.sh swap enable 8G /swapfile
./jetson_spare_mem.sh wrap 8g -- python -m pip install --no-build-isolation -v .
./jetson_spare_mem.sh headless off
```

**B) 제한된 메모리에서 모델 추론**
- 배치를 작게; 텐서/버퍼 재사용.
- Xavier/Orin은 FP16 권장; Nano/TX2는 메모리 절약 위주.
- 데이터로더 워커 수 과도하게 늘리지 않기 (`num_workers=0–2`, `pin_memory=False`).

---

### 모니터링 & 트러블슈팅
```bash
tegrastats
free -h
vmstat 1          # si/so(스왑 입/출) — 0에 가깝게 유지
top               # %wa(I/O wait) — 높으면 스왑 병목
./jetson_spare_mem.sh status
dmesg | egrep -i 'killed process|out of memory' | tail -n5
```
**스왑이 계속 쓰이거나 OOM이 나면:** 병렬도를 낮추고(`MAX_JOBS=1`) 빌드 동안 GUI를 끄세요.

---

### 정리 / 되돌리기
```bash
# 스왑파일 제거
./jetson_spare_mem.sh swap disable /swapfile

# 부팅 기본값을 GUI로 복원
./jetson_spare_mem.sh headless persist-off
```