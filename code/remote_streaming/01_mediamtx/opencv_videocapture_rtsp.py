# -*- coding: utf-8 -*-
"""
OpenCV (GStreamer backend) → MediaMTX RTSP publisher with teaching overlays.

Adds three student examples:
  1) OpenCV tweaks (--opencv_tweaks on/off):
     - disable OpenCL, set limited threads, try CAP_PROP_BUFFERSIZE
  2) E2E latency overlay (--overlay_latency):
     - draws Jetson TX timestamp on each frame (ISO + epoch ms)
     - students compare against their viewer's local clock to estimate E2E lag
  3) Bandwidth sanity (--overlay_bandwidth, --netdev):
     - shows encoder target bitrate and live NIC throughput (tx/rx Mbps)
     - (NIC stats are system-wide; assume minimal other traffic during lab)
---    
OpenCV와 GStreamer를 사용해 RTSP 스트림을 발행하고 교육용 오버레이를 추가합니다.

이 스크립트는 학생들의 학습을 돕기 위한 세 가지 오버레이 예제를 제공합니다.

주요 기능:
    1) OpenCV 성능 조정 (`--opencv_tweaks` 켜기/끄기):
        - OpenCL 비활성화, 스레드 수 제한, CAP_PROP_BUFFERSIZE 설정을 시도합니다.

    2) 종단간(E2E) 지연 시간 오버레이 (`--overlay_latency`):
        - 각 프레임에 Jetson의 타임스탬프(ISO + epoch 밀리초)를 표시합니다.
        - 학생들은 뷰어의 로컬 시계와 비교하여 종단간(E2E) 지연을 추정할 수 있습니다.

    3) 대역폭 상태 확인 (`--overlay_bandwidth`, `--netdev`):
        - 인코더 목표 비트레이트와 실시간 네트워크 인터페이스(NIC) 처리량(tx/rx Mbps)을 표시합니다.
        - (NIC 통계는 시스템 전체에 해당하므로, 실습 중 다른 트래픽은 최소라고 가정합니다.)
---
Run example(실행 예시):
    python opencv_videocapture_rtsp.py \
        --device /dev/video0 --src_format MJPG --w 1920 --h 1080 --fps 30/1 \
        --codec h265 --bitrate 8000000 --host <JETSON_IP> --port 8554 --name mystream \
        --opencv_tweaks 1 --overlay_latency 1 --overlay_bandwidth 1 --netdev wlan0
"""
import sys, cv2, argparse, signal, time, os
from datetime import datetime
from pathlib import Path

from utils.camera import val_camera_configuration
from utils.gst import build_usb_caps, build_writer_caps, parse_fps, get_rtsp_url

# ---------- Utility: simple text overlay box ----------

def _draw_boxed_text(
    img,
    lines,
    org=(10, 24),
    font_scale=0.6,
    color=(255, 255, 255),
    bg=(0, 0, 0),
    thickness=1,
    line_gap=6,
):
    """
    Draw multi-line text with a background rectangle for readability.
    """
    if isinstance(lines, str):
        lines = [lines]

    font = cv2.FONT_HERSHEY_SIMPLEX
    # Measure box
    widths, heights = [], []
    for ln in lines:
        (w, h), _ = cv2.getTextSize(ln, font, font_scale, thickness)
        widths.append(w); heights.append(h)

    box_w = max(widths) + 12
    box_h = sum(heights) + (len(lines) - 1) * line_gap + 12

    x, y = org
    x2, y2 = x + box_w, y + box_h
    cv2.rectangle(img, (x, y - 18), (x2, y2 - 6), bg, -1)

    y_text = y
    for ln, h in zip(lines, heights):
        cv2.putText(img, ln, (x + 6, y_text), font, font_scale, color, thickness, cv2.LINE_AA)
        y_text += h + line_gap

# ---------- Utility: NIC throughput meter ----------

class NetDevMeter:
    """
    Reads /sys/class/net/<dev>/statistics/{tx_bytes,rx_bytes} to estimate Mbps.
    Use when MediaMTX sends to remote client(s) from the same Jetson NIC.
    """
    def __init__(self, dev: str):
        self.dev = dev
        self.tx_path = self.rx_path = None
        self._ok = False
        self._prev_t = None
        self._prev_tx = None
        self._prev_rx = None

        if dev:
            base = Path(f"/sys/class/net/{dev}/statistics")
            txp = base / "tx_bytes"
            rxp = base / "rx_bytes"
            if txp.exists() and rxp.exists():
                self.tx_path = txp
                self.rx_path = rxp
                self._ok = True

    def read_mbps(self):
        """
        Returns (tx_Mbps, rx_Mbps) as floats, or (None, None) if unavailable.
        """
        if not self._ok:
            return (None, None)
        try:
            now = time.time()
            tx = int(self.tx_path.read_text().strip())
            rx = int(self.rx_path.read_text().strip())
        except Exception:
            return (None, None)

        if self._prev_t is None:
            self._prev_t = now
            self._prev_tx = tx
            self._prev_rx = rx
            return (0.0, 0.0)

        dt = max(1e-6, now - self._prev_t)
        dtx = max(0, tx - self._prev_tx)
        drx = max(0, rx - self._prev_rx)

        self._prev_t = now
        self._prev_tx = tx
        self._prev_rx = rx

        tx_mbps = (dtx * 8.0) / dt / 1e6
        rx_mbps = (drx * 8.0) / dt / 1e6
        return (tx_mbps, rx_mbps)

# ---------- Optional OpenCV tweaks ----------

def apply_opencv_tweaks(enable: bool, cap: cv2.VideoCapture):
    """
    Lightweight hints that often reduce latency/overhead in OpenCV.
    """
    if not enable:
        return
    # Fewer threads reduces scheduling overhead on small SoCs
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass
    # Explicitly disable OpenCL (can cause additional copies)
    try:
        cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass
    # Keep the capture queue small to avoid additional buffering
    if cap is not None:
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

# ---------- Formatting helpers ----------

def _fmt_bps(bps: float) -> str:
    if bps is None:
        return "-"
    if bps >= 1e9:
        return f"{bps/1e9:.2f} Gbps"
    if bps >= 1e6:
        return f"{bps/1e6:.2f} Mbps"
    if bps >= 1e3:
        return f"{bps/1e3:.2f} Kbps"
    return f"{bps:.0f} bps"

def _fmt_mbps(mbps: float) -> str:
    return "-" if mbps is None else f"{mbps:.2f} Mbps"

# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    # Source & writer args
    ap.add_argument("--device", default="/dev/video0")
    ap.add_argument("--src_format", default="MJPG", choices=["MJPG", "YUY2"])
    ap.add_argument("--w", type=int, default=1920)
    ap.add_argument("--h", type=int, default=1080)
    ap.add_argument("--fps", default="30/1")
    ap.add_argument("--flip", type=int, default=0)
    ap.add_argument("--codec", default="h265", choices=["h264", "h265"])
    ap.add_argument("--bitrate", type=int, default=8_000_000)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8554)
    ap.add_argument("--name", default="mystream")

    # Example toggles
    ap.add_argument("--opencv_tweaks", type=int, default=1)          # 1 = on
    ap.add_argument("--overlay_latency", type=int, default=1)
    ap.add_argument("--overlay_bandwidth", type=int, default=1)
    ap.add_argument("--netdev", type=str, default="")                # e.g., wlan0 / eth0

    args = ap.parse_args()
    target_fps = parse_fps(args.fps)

    # Validate camera mode table (students generated camera_formats.json earlier)
    if not val_camera_configuration(args.src_format, args.w, args.h, args.fps):
        raise ValueError("Unsupported camera format. Check camera_formats.json")

    # Build pipelines
    gst_cap = build_usb_caps(
        device=args.device,
        src_format=args.src_format,
        w=args.w,
        h=args.h,
        fps=args.fps,
        flip=args.flip,
    )
    gst_writer = build_writer_caps(
        codec=args.codec,
        flip=args.flip,
        bitrate=args.bitrate,
        host=args.host,
        port=args.port,
        name=args.name,
        w=args.w,
        h=args.h,
        fps=args.fps,
    )

    # Open capture & writer
    cap = cv2.VideoCapture(gst_cap, cv2.CAP_GSTREAMER)
    apply_opencv_tweaks(bool(args.opencv_tweaks), cap)
    print(f"[SRC] {args.device} opened, {args.w}x{args.h} @ {args.fps} ({args.src_format})")

    out = cv2.VideoWriter(
        gst_writer,
        cv2.CAP_GSTREAMER,
        0,
        target_fps,
        (args.w, args.h),
        True,
    )
    if not out.isOpened():
        print("Failed to open output. Is MediaMTX running? RTSP URL:", get_rtsp_url(args.host, args.port, args.name))
        sys.exit(1)

    # Bandwidth meter (system-wide NIC stats)
    meter = NetDevMeter(args.netdev if args.netdev else None)

    stop = False
    def _sigint(_a, _b):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGINT, _sigint)

    # Stats
    t0 = last_stat_t = time.time()
    frames = 0
    ema_fps = None   # exponential moving average for smoother readout

    try:
        if cap.isOpened():
            while not stop:
                ok, img = cap.read()
                if not ok:
                    break

                now = time.time()
                frames += 1

                # --- Example 2: E2E latency overlay (TX timestamp) ---
                if args.overlay_latency:
                    # ISO time and epoch ms at TX side
                    iso = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    epoch_ms = int(now * 1000)
                    _draw_boxed_text(
                        img,
                        [f"TX: {iso}", f"epoch_ms: {epoch_ms}"],
                        org=(10, 28),
                        font_scale=0.6,
                        bg=(0, 0, 0),
                    )

                # --- Example 3: Bandwidth sanity overlay ---
                lines_bw = []
                if args.overlay_bandwidth:
                    lines_bw.append(f"Enc target: {_fmt_bps(args.bitrate)} ({args.codec.upper()})")
                    if meter and meter._ok:
                        tx_mbps, rx_mbps = meter.read_mbps()
                        lines_bw.append(f"{args.netdev} tx: {_fmt_mbps(tx_mbps)}  rx: {_fmt_mbps(rx_mbps)}")
                    else:
                        lines_bw.append("NIC: - (set --netdev wlan0/eth0)")

                    # Show effective FPS (EMA) as part of bandwidth sanity
                    dt = now - t0
                    inst_fps = (frames / dt) if dt > 0 else 0.0
                    alpha = 0.2
                    ema_fps = inst_fps if ema_fps is None else alpha * inst_fps + (1 - alpha) * ema_fps
                    lines_bw.append(f"FPS: {ema_fps:.1f} (target {target_fps:.1f})")

                    _draw_boxed_text(
                        img,
                        lines_bw,
                        org=(10, 92),
                        font_scale=0.6,
                        bg=(0, 0, 0),
                    )

                # Write to RTSP pipeline
                out.write(img)

                # Throttle prints to ~1 Hz
                if now - last_stat_t >= 1.0:
                    if ema_fps is None:
                        dt = now - t0
                        eff = (frames / dt) if dt > 0 else 0.0
                    else:
                        eff = ema_fps
                    msg = f"[{datetime.now()}] {frames} frames, ~{eff:.1f} fps → {get_rtsp_url(args.host, args.port, args.name)}"
                    if meter and meter._ok:
                        tx_mbps, rx_mbps = meter.read_mbps()
                        msg += f" | {args.netdev} tx { _fmt_mbps(tx_mbps) } rx { _fmt_mbps(rx_mbps) }"
                    print(msg)
                    last_stat_t = now
        else:
            print("Pipeline open failed")
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print("successfully exit")

if __name__ == "__main__":
    main()
