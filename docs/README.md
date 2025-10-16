[ [English](#english) | [한국어](#한국어) ]

# Intelligent Robot Guide — Docs (Quick Index)
**Up one level:** [`../README.md`](../README.md)

---

## English

Short pointers to each doc:

- **install_jetson_dev_packages.md** — One‑shot installer for common Jetson dev deps (CUDA/OpenCV/TRT tools) with safe defaults and selectable steps.
- **safe_set_pytorch_conda.md** — Creates a PyTorch conda env aligned with JetPack; avoids duplicate CUDA; optional `--no-tv` for quicker setup.
- **check_jetson_env_conflicts.md** — Sanity‑check your env: detects pip OpenCV wheels, CUDA_HOME/nvcc, Torch↔Vision mapping, and `tegra` libs on `LD_LIBRARY_PATH`.
- **jetson_system_bridge.md** — Bridges system **cv2/TRT/PyCUDA/jtop** into the conda env via `.pth`, ensuring CUDA‑enabled system packages take precedence.
- **jetson_spare_mem.md** — Swap/headless tweaks and memory housekeeping for Jetson before heavy builds or training (free a few precious GBs).

---

## 한국어

각 문서의 간단 안내:

- **install_jetson_dev_packages.md** — Jetson 개발 필수 패키지 일괄 설치 스크립트(안전한 기본값, 단계 선택 가능).
- **safe_set_pytorch_conda.md** — JetPack에 맞춘 PyTorch conda 환경 생성; 중복 CUDA 방지; `--no-tv` 옵션으로 빠른 구성.
- **check_jetson_env_conflicts.md** — 환경 점검: pip OpenCV 휠, CUDA_HOME/nvcc, Torch↔Vision 매핑, `tegra` 라이브러리 노출 여부 확인.
- **jetson_system_bridge.md** — 시스템 **cv2/TRT/PyCUDA/jtop**을 `.pth`로 conda 환경에 우선 연결(CUDA 지원 시스템 패키지를 먼저 사용).
- **jetson_spare_mem.md** — 대규모 빌드/학습 전을 위한 스왑/헤드리스 설정과 메모리 정리(몇 GB 여유 확보용).