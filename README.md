[ [English](#english) | [한국어](#한국어) ]

# Intelligent Robot Guide
Author: Minsu Kwon (M.S in Dept of AI, University of Seoul)  
![Banner](assets/logo.jpg)

---

## English

### Useful docs
- 👉 **Install dev packages safely, easily:** [`install_jetson_dev_packages.md`](docs/install_jetson_dev_packages.md)
- 👉 **Safely setup conda venv for jetson:** [`safe_set_pytorch_conda.md`](docs/safe_set_pytorch_conda.md)
- 👉 **Code examples:** [`code/README.md`](code/README.md)

### Prerequisites
- **Hardware**: Jetson Xavier NX  
- **OS**: Ubuntu 20.04 (L4T r35.6.x) / JetPack **5.1.5**  
- **Python**: 3.8  
- **Key libs**:
  - CUDA 11.4 (JetPack)
  - cuDNN 8.x (JetPack)
  - PyTorch 2.1.0 (Jetson wheel)
  - Torchvision 0.16.1 (from source)
  - OpenCV 4.10.0 (CUDA, installed to `/usr/local`)
  - (Optional) DeepStream 6.3

### Quick Start
1) **(Only for UOS class students!)** Wi-Fi (AX210) patch + static IPv4
   ```bash
   cd scripts
   ./enable_ax210.sh
   ./set_wifi_ipv4.sh                 # prompts SSID/PSK, sets static IPv4
   ```
---
2) Install jetson dev stack (see [`install_jetson_dev_packages.md`](docs/install_jetson_dev_packages.md) for detail)
   ```bash
   ./install_jetson_dev_packages.sh   # default 1,2,3,4,5
   ```
3) Safely create PyTorch conda venv for project. **(Recommended!)**
   ```bash
   ./safe_set_pytorch_conda.sh        # choose torch 2.1 / 2.0 / 1.14
   ```

---

## 한국어

### 참고 문서
- 👉 **개발 패키지 안전·간편 설치:** [`install_jetson_dev_packages.md`](docs/install_jetson_dev_packages.md)
- 👉 **Jetson용 conda 환경 안전 설정:** [`safe_set_pytorch_conda.md`](docs/safe_set_pytorch_conda.md)
- 👉 **코드 예제:** [`code/README.md`](code/README.md)

### 요구 사항
- **하드웨어**: Jetson Xavier NX  
- **OS**: Ubuntu 20.04 (L4T r35.6.x) / JetPack **5.1.5**  
- **Python**: 3.8  
- **핵심 라이브러리**:
  - CUDA 11.4 (JetPack 기본 제공)
  - cuDNN 8.x (JetPack)
  - PyTorch 2.1.0 (Jetson 전용 휠)
  - Torchvision 0.16.1 (소스 빌드)
  - OpenCV 4.10.0 (CUDA 지원, `/usr/local` 설치)
  - (선택) DeepStream 6.3

### 빠른 시작
1) **(서울시립대 수업 전용)** AX210 Wi‑Fi 패치 + 고정 IPv4
   ```bash
   cd scripts
   ./enable_ax210.sh
   ./set_wifi_ipv4.sh                 # SSID/PSK 입력받아 고정 IP 설정
   ```
---
2) Jetson 개발 스택 설치 (자세한 내용: [`install_jetson_dev_packages.md`](docs/install_jetson_dev_packages.md))
   ```bash
   ./install_jetson_dev_packages.sh   # 기본 1,2,3,4,5 실행
   ```
3) 프로젝트용 PyTorch conda 환경 생성 **(권장)**  
   ```bash
   ./safe_set_pytorch_conda.sh        # torch 2.1 / 2.0 / 1.14 선택
   ```

---

_Back to:_ **[Scripts](scripts/README.md)** · **[Backups](scripts/backups/README.md)** · **[Top](#intelligent-robot-guide)**
