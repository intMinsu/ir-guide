# code/remote_streaming/02_jetson_inference/simple_videocapture.py
"""
Zero-copy video capture → output sink with optional measurement and viewer-gated rendering.

Supported outputs (videoOutput URI):
  1) Display (local monitor):     display://0
  2) RTSP streaming (built-in):   rtsp://localhost:8554/mystream
  3) WebRTC streaming (builtin):  webrtc://@:8554/mystream
  4) Files/directories:           file://out.mp4, file://frame_%06i.jpg, file://out.jpg

Notes
- WebRTC is typically lower-latency and applies backpressure (often fewer idle-memory issues).
- For RTSP, `--render-when-viewed` renders only when a viewer is actually connected.
  This mitigates VRAM growth when no client is consuming frames.

Examples
  # Local display preview
  python simple_videocapture.py --input v4l2:///dev/video0 \
  --output display://0 \
  --measure

  # RTSP (built-in server), render only when viewed
  python simple_videocapture.py --input v4l2:///dev/video0 \
  --output rtsp://<JETSON-IP>:8554/mystream \
  --codec h264 --latency 10 \
  --render-when-viewed \
  --measure

  # WebRTC (open https://<JETSON-IP>:8554 in browser)
  python simple_videocapture.py --input v4l2:///dev/video0 \
  --output webrtc://<JETSON-IP>:8554/mystream \
  --codec h264 --latency 10 \
  --measure
"""
import argparse
import time
from jetson_utils import videoSource, videoOutput, cudaFont

from utils.stagetimer import StageTimer
from utils.overlay_helper import overlay_text
from utils.rtsp_utils import RTSPGate, is_rtsp_uri

def main():
    ap = argparse.ArgumentParser(description="Capture → output (display/RTSP/WebRTC/file) with zero-copy")
    ap.add_argument("--input",  type=str, default="v4l2:///dev/video0", help="input URI")
    ap.add_argument("--output", type=str, default="display://0", help="output URI (display://, rtsp://, webrtc://, file://...)")
    ap.add_argument("--width",  type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--bitrate",type=int, default=8_000_000)
    ap.add_argument("--codec",  type=str, default="h264",
                    choices=["h264","h265","vp8","vp9","mpeg2","mpeg4","mjpeg"])
    ap.add_argument("--latency", type=int, default=10, help="network buffering latency in ms (RTSP/WebRTC). Default is 10")
    ap.add_argument("--measure", action="store_true", help="print per-stage timings and display on status bar")

    # Simple RTSP gating (only applies to rtsp:// outputs)
    ap.add_argument("--render-when-viewed", action="store_true",
                    help="(RTSP only) call Render() only when a viewer is connected")
    ap.add_argument("--startup-grace", type=float, default=5.0,
                    help="(RTSP only) seconds to always allow rendering after start")
    ap.add_argument("--idle-timeout", type=float, default=None,
                    help="(RTSP only) auto-exit if no viewer for this many seconds (None=disabled)")
    args = ap.parse_args()

    videosource_dict = {
        "width":   args.width,
        "height":  args.height,
        "codec":   "mjpeg",   # Logitech C920 works well with MJPEG
        "encoder": "v4l2",
    }
    videooutput_dict = {
        "codec":   args.codec,
        "encoder": "v4l2",
        "bitrate": args.bitrate,
    }
    if args.latency is not None:
        videooutput_dict["latency"] = args.latency

    print(f"[videoSource] {args.input}  → {videosource_dict}")
    print(f"[videoOutput] {args.output} → {videooutput_dict}")

    inp = videoSource(args.input, options=videosource_dict)
    out = videoOutput(args.output, options=videooutput_dict)

    # Instantiate RTSP gate ONCE
    gate = RTSPGate(args.output, idle_timeout=args.idle_timeout) \
           if args.render_when_viewed and is_rtsp_uri(args.output) else None

    # FPS smoothing + measurement
    warm = 10
    frame_idx = 0
    t_prev = time.time()
    smoothed_fps = 0.0
    timer = StageTimer(enabled=args.measure, warmup=warm, log_interval=2.0)

    font = cudaFont()

    while True:
        timer.next_frame()
        with timer.span("tot"):
            with timer.span("cap"):
                img = inp.Capture(format="rgb8", timeout=1000)
            if img is None:
                continue

            overlay = f"{smoothed_fps:4.1f} FPS"
            overlay_text(font, img, overlay, x=10, y=10)

            # Update connection state (no-ops if gate is None)
            if gate:
                gate.update()
                should_render = gate.should_render()
            else:
                should_render = True

            if should_render:
                with timer.span("rend"):
                    out.Render(img)

            # Optional: terminate cleanly after prolonged inactivity
            if gate and gate.idle_timed_out():
                print("[rtsp] idle timeout — no viewer detected, exiting.")
                break

        # FPS smoothing + status text
        frame_idx += 1
        now = time.time()
        if frame_idx > warm:
            dt = now - t_prev
            smoothed_fps = 0.9 * smoothed_fps + 0.1 * (1.0 / max(1e-6, dt))
        t_prev = now

        extra = timer.status_suffix(keys=("cap", "rend", "tot"))
        proto = args.output.split("://", 1)[0].upper()
        view_tag = " | VIEW:" + ("ON" if (gate and gate.should_render()) else "OFF") if gate else ""
        out.SetStatus(f"Capture → {proto} | ~{smoothed_fps:4.1f} FPS{extra}{view_tag}")

        timer.end_frame()
        timer.log_this_interval(smoothed_fps, "io-only")

        if not inp.IsStreaming() or not out.IsStreaming():
            break


if __name__ == "__main__":
    main()
