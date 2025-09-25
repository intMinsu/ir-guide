[ [English](#english) | [한국어](#한국어) ]

# Scripts

**Up one level:** [`../README.md`](../README.md) · **Backups:** [`backups/README.md`](backups/README.md)  

---

## English

### 1) `enable_ax210.sh`
**(Only for UOS class students!)** Patch Intel AX210, set hostname. This script will disable Bluetooth functionality (L4T-side bug).  
```bash
./enable_ax210.sh <hostname>           # one can specify custom hostname
KEEP_BT=1 ./enable_ax210.sh            # keep Bluetooth enabled (not recommended; buggy)
```

### 2) `set_wifi_ipv4.sh`
**(Only for UOS class students!)** Connect to Wi‑Fi and set static IPv4.  
- Type PSK when prompted.  
- If hostname is `deviceNN`, uses `192.168.1.2NN`.  
- Otherwise prompts for the last octet (0–255).
```bash
./set_wifi_ipv4.sh                      # interactive SSID/PSK
# non-interactive
SSID='rcv-robot-5G' PSK='your-psk' DIGIT=42 ./set_wifi_ipv4.sh
```

### 3) `install_jetson_dev_packages.sh`
Menu installer (defaults `1,2,3,4,5`). See detailed doc: [`install_jetson_dev_packages.md`](../docs/install_jetson_dev_packages.md)
1. Max performance (nvpmodel + jetson_clocks)  
2. GitHub CLI (gh)  
3. jtop  
4. JetPack meta (runtime/dev) + container runtime  
5. OpenCV (CUDA + FFmpeg + GStreamer)  
6. DeepStream (optional)
```bash
./install_jetson_dev_packages.sh        # interactive menu
./install_jetson_dev_packages.sh 1,4,5  # non-interactive
# Env:
JETPACK_FLAVOR=dev|runtime
OPENCV_VER=4.10.0
JOBS=4
DS_VER=6.3
```

---

## 한국어

### 1) `enable_ax210.sh`
**(서울시립대 수업 전용)** Intel AX210 패치 및 호스트명 설정 스크립트입니다.  
L4T 버그로 인해 기본적으로 **블루투스가 비활성화**됩니다.
```bash
./enable_ax210.sh <hostname>           # 사용자 지정 호스트명 지정 가능
KEEP_BT=1 ./enable_ax210.sh            # 블루투스 유지(권장하지 않음, 버그 있음)
```

### 2) `set_wifi_ipv4.sh`
**(서울시립대 수업 전용)** Wi‑Fi 연결 및 고정 IPv4 설정.  
- 안내에 따라 **PSK(비밀번호)** 를 입력합니다.  
- 호스트명이 `deviceNN` 형태면 IP를 `192.168.1.2NN`으로 자동 설정합니다.  
- 그렇지 않으면 마지막 옥텟(0–255)을 입력받아 설정합니다.
```bash
./set_wifi_ipv4.sh
# 비대화식 예시
SSID='rcv-robot-5G' PSK='비밀번호' DIGIT=42 ./set_wifi_ipv4.sh
```

### 3) `install_jetson_dev_packages.sh`
메뉴 기반 설치기(기본 `1,2,3,4,5`). 자세한 문서: [`install_jetson_dev_packages.md`](../docs/install_jetson_dev_packages.md)  
1. 최대 성능(nvpmodel + jetson_clocks)  
2. GitHub CLI (gh)  
3. jtop  
4. JetPack 메타(runtime/dev) + 컨테이너 런타임  
5. OpenCV (CUDA + FFmpeg + GStreamer)  
6. DeepStream (선택)
```bash
./install_jetson_dev_packages.sh        # 대화식 메뉴
./install_jetson_dev_packages.sh 1,4,5  # 비대화식 사용
# 환경변수:
JETPACK_FLAVOR=dev|runtime
OPENCV_VER=4.10.0
JOBS=4
DS_VER=6.3
```

---

_Back to:_ **[Root README](../README.md)** · **[Backups](backups/README.md)** · **[Top](#scripts)**
