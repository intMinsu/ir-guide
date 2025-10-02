# Remote Camera Streaming Examples
[ [English](#english) | [한국어](#한국어) ]

This folder contains two complementary paths for **remote video streaming on NVIDIA Jetson**:

- **01_mediamtx/opencv_videocapture_rtsp.py** — OpenCV (GStreamer backend) publisher → **MediaMTX** RTSP server
- **02_jetson_inference/simple_videocapture_rtsp.py** — **jetson-utils** zero-copy path with **built-in RTSP server**

They’re designed for classroom labs and come with **low‑latency defaults**, an **E2E latency overlay**, and a **bandwidth sanity overlay** for quick diagnostics.

---

## English

### Overview

| Path | Where the RTSP server runs | Pros | Considerations |
|---|---|---|---|
| **OpenCV → MediaMTX** (`01_mediamtx/opencv_videocapture_rtsp.py`) | **MediaMTX** (separate process) | Works with OpenCV code you already know; flexible routing to multiple clients | OpenCV’s BGR path breaks NVMM zero‑copy; ensure MediaMTX is running |
| **jetson-utils RTSP** (`02_jetson_inference/simple_videocapture_rtsp.py`) | **Inside jetson-utils** (`videoOutput("rtsp://...")`) | Fast to set up; stays in CUDA/NVMM; great for demos | Can conflict with MediaMTX on port 8554; fewer knobs than MediaMTX |

**Recommended lab order:** start with jetson-utils (quick win), then move to the OpenCV→MediaMTX pipeline to learn the GStreamer/encoder pieces.

---

## 1) `code/01_mediamtx/opencv_videocapture_rtsp.py`

### What it does
- Captures from a USB camera via **GStreamer** (`v4l2src`), converts to **BGR** for OpenCV.
- Publishes frames via **appsrc → NVENC (H.264/H.265) → rtspclientsink → MediaMTX**.
- Adds **low‑latency settings** (is‑live timestamps, leaky queue, config‑interval, publisher latency=0).
- Optional **overlays**:
  - **E2E Latency**: embeds Jetson TX time (ISO + epoch_ms) on each frame
  - **Bandwidth**: shows encoder target bitrate & NIC tx/rx throughput (Mbps)
- Optional **OpenCV tweaks**: disable OpenCL, lower thread count, shrink capture buffer.

### Prerequisites
- **Camera**: e.g., Logitech C920 (prefer **MJPG**).
- **MediaMTX** server running on Jetson:  
  ```bash
  ~/mediamtx/mediamtx     # or use your service/Docker method
  ```
- **VLC** (or any RTSP player) on the client PC.
- **Same network** between Jetson and client (e.g., `wlan0`).

### Generate camera mode table (once per camera)
Create `camera_formats.json` so the script can validate `(format, WxH, fps)`:
```bash
cd code/01_mediamtx/utils
python v4l2_extformat_parse.py --device /dev/video0 --out ../camera_formats.json
```

### Run
```bash
cd code/01_mediamtx
python opencv_videocapture_rtsp.py   --device /dev/video0 --src_format MJPG --w 1920 --h 1080 --fps 30/1   --codec h265 --bitrate 8000000   --host <JETSON_IP> --port 8554 --name mystream   --opencv_tweaks 1 --overlay_latency 1 --overlay_bandwidth 1 --netdev wlan0
```

Open the stream on the client (VLC → **Media → Open Network Stream**):
```
rtsp://<JETSON_IP>:8554/mystream
```

### Useful flags
- `--src_format {MJPG,YUY2}` (C920 → **MJPG** recommended)
- `--codec {h264,h265}` (H.265 saves bandwidth; H.264 is widely compatible)
- `--bitrate 4000000` (720p@30), `8000000` (1080p@30) as starting points
- `--overlay_latency 1` to show TX timestamps; compare with client clock
- `--overlay_bandwidth 1 --netdev wlan0` to show NIC throughput (Mbps)

### Try a simple processing edit
Uncomment the grayscale example in the loop:
```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img  = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
```

### Troubleshooting
- **Failed to open output** → Make sure **MediaMTX** is running and URL matches `--host --port --name`.
- **High or growing latency** → Keep the overlays on, verify NIC throughput, try lower bitrate/resolution, and ensure `rtspclientsink latency=0` is in the pipeline (already set by default).
- **No picture / color off** → Confirm `--src_format` matches what your camera actually outputs in `camera_formats.json` (use the parser above). Prefer **MJPG** for C920.
- **Port conflicts** → If also running jetson-utils RTSP, change one of the RTSP ports (e.g., 8555).

---

## 2) `code/02_jetson_inference/simple_videocapture_rtsp.py`

### What it does
- Uses **jetson-utils** to capture from `v4l2:///dev/video0` into a **cudaImage** (zero-copy).
- Streams with **videoOutput("rtsp://...")**, which **runs its own RTSP server**.
- Minimal, **fast** pipeline; great for first demos and CUDA-based image ops.

### Prerequisites
- **jetson-inference / jetson-utils** installed & working (see the course build guide).
- **Do not** run MediaMTX on the **same port** (default 8554).

### Run
```bash
cd code/02_jetson_inference
python simple_videocapture_rtsp.py   --input v4l2:///dev/video0   --output rtsp://localhost:8554/mystream   --width 1920 --height 1080 --bitrate 8000000 --codec h265
```

Open the stream on the client:
```
rtsp://<JETSON_IP>:8554/mystream
```

### Notes
- The input options set `codec: mjpeg` for V4L2 — good default for C920.
- If you see “address already in use”, either stop MediaMTX or change the RTSP port in `--output` (e.g., `rtsp://localhost:8555/mystream`).

### Troubleshooting
- **No module named jetson_utils** → Rebuild/activate your jetson-inference environment.
- **Black screen in client** → Check that Jetson and client are on the same LAN; lower bitrate; try H.264.
- **Port in use** → Change `--output` port or stop MediaMTX.

---

## Low‑latency tips (both paths)

- Publisher side is already set to **is-live timestamps + leaky queues**.
- On the client player (VLC): set **Network Caching** to **100–300 ms** (Tools → Preferences → Input/Codecs).
- Keep **GOP** moderate (we use `iframeinterval=60` by default) and start with CBR (`control-rate=1`) at a sane bitrate.
- Prefer **wired** (eth0) for demos; Wi‑Fi (wlan0) works but is more jittery.
- On Jetson, avoid extra background CPU/GPU tasks during the lab.

---

## 한국어

### 개요

| 경로 | RTSP 서버 실행 위치 | 장점 | 유의사항 |
|---|---|---|---|
| **OpenCV → MediaMTX** (`01_mediamtx/opencv_videocapture_rtsp.py`) | **MediaMTX**(별도 프로세스) | 기존 OpenCV 코드와 쉽게 연동, 다수 클라이언트 라우팅 유연 | OpenCV BGR 경로는 NVMM 제로‑카피가 끊김, MediaMTX 실행 필수 |
| **jetson-utils RTSP** (`02_jetson_inference/simple_videocapture_rtsp.py`) | **jetson-utils 내부** | 설정이 간단하고 빠름, CUDA/NVMM 유지 | 기본 포트(8554)가 MediaMTX와 충돌 가능, 세부 옵션은 적음 |

**권장 실습 순서:** jetson-utils로 빠르게 성공 경험 → OpenCV→MediaMTX로 파이프라인 구성 학습.

---

## 1) `code/01_mediamtx/opencv_videocapture_rtsp.py`

### 동작
- **GStreamer**(`v4l2src`)로 USB 카메라 캡처 → OpenCV(BGR) 처리.
- **appsrc → NVENC(H.264/H.265) → rtspclientsink → MediaMTX** 로 송출.
- **저지연 옵션**(is-live 타임스탬프, leaky queue, config-interval, publisher latency=0) 내장.
- **오버레이**: TX 시간(ISO + epoch_ms) / NIC 트래픽(Mbps) / 목표 비트레이트.
- **OpenCV 최적화 옵션**: OpenCL 비활성화, 스레드 수 축소, 캡처 버퍼 최소화.

### 준비
- **카메라**: Logitech C920 권장(포맷은 **MJPG** 추천)
- **MediaMTX** 실행:
  ```bash
  ~/mediamtx/mediamtx
  ```
- **VLC**(클라이언트), **동일 네트워크** 연결.

### 카메라 모드 테이블 생성(1회)
```bash
cd code/01_mediamtx/utils
python v4l2_extformat_parse.py --device /dev/video0 --out ../camera_formats.json
```

### 실행
```bash
cd code/01_mediamtx
python opencv_videocapture_rtsp.py   --device /dev/video0 --src_format MJPG --w 1920 --h 1080 --fps 30/1   --codec h265 --bitrate 8000000   --host <JETSON_IP> --port 8554 --name mystream   --opencv_tweaks 1 --overlay_latency 1 --overlay_bandwidth 1 --netdev wlan0
```

클라이언트(VLC)에서 아래 주소로 열기:
```
rtsp://<JETSON_IP>:8554/mystream
```

### 문제 해결
- **출력 열기 실패** → MediaMTX 실행/URL 확인.
- **지연 증가** → 오버레이 켜고 NIC Mbps 확인, 해상도/비트레이트 낮추기.
- **색상/영상 이상** → `camera_formats.json`로 확인, C920은 **MJPG** 권장.
- **포트 충돌** → jetson-utils RTSP 사용 시 포트 변경(예: 8555).

---

## 2) `code/02_jetson_inference/simple_videocapture_rtsp.py`

### 동작
- **jetson-utils** 로 `v4l2:///dev/video0` 캡처(제로‑카피 **cudaImage**).
- **videoOutput("rtsp://...")** 가 **내장 RTSP 서버**를 실행하여 송출.
- 빠르게 데모하기에 적합.

### 실행
```bash
cd code/02_jetson_inference
python simple_videocapture_rtsp.py   --input v4l2:///dev/video0   --output rtsp://localhost:8554/mystream   --width 1920 --height 1080 --bitrate 8000000 --codec h265
```

클라이언트에서 열기:
```
rtsp://<JETSON_IP>:8554/mystream
```

### 참고/문제 해결
- **jetson_utils 모듈 없음** → jetson-inference 빌드/환경 점검.
- **포트 사용 중** → MediaMTX 중지 또는 출력 포트 변경.
- **화면 검정** → 같은 네트워크인지 확인, 비트레이트 하향, H.264 시도.

---

## Low‑latency 팁
- 퍼블리셔 쪽은 이미 **is-live + leaky queue**를 적용.
- VLC에서 **네트워크 캐싱**을 **100–300 ms**로 조정.
- **GOP(iframeinterval)** 은 과도하게 길지 않게(기본 60), **CBR**로 시작.
- 데모 시에는 **유선(eth0)** 권장, Wi‑Fi는 지터가 큼.
- Jetson에서는 불필요한 백그라운드 작업을 줄이세요.

---

_Back to:_ **[Top-level README](../README.md)** · **[Scripts](../scripts/README.md)** · **[Docs](../docs/)**