# code/remote_streaming/02_jetson_inference/simple_mobilenet.py
"""
DetectNet (MobileNet) → display/RTSP/WebRTC with zero-copy and optional RTSP viewer gating.

Supported outputs (videoOutput URI):
  1) Display (local monitor):     display://0
  2) RTSP streaming (built-in):   rtsp://localhost:8554/mystream
  3) WebRTC streaming (builtin):  webrtc://@:8554/mystream

See pre-trained detection models available in jetson-inference here:
https://github.com/dusty-nv/jetson-inference/blob/master/docs/detectnet-console-2.md#pre-trained-detection-models-available
  1) ssd-mobilenet-v2
  2) ssd-inception-v2
  3) peoplenet
  4) peoplenet-pruned
  5) dashcamnet
  6) trafficcamnet
  7) facedetect

Examples
  # Local monitor preview
  python simple_mobilenet.py --input v4l2:///dev/video0 \
  --output display://0 \
  --network ssd-mobilenet-v2 \
  --measure

  # RTSP stream (built-in server), render only when viewed
  python simple_mobilenet.py --input v4l2:///dev/video0 \
  --output rtsp://localhost:8554/mystream \
  --codec h264 --latency 30 \
  --network ssd-mobilenet-v2 \
  --render-when-viewed \
  --measure

  # WebRTC stream (open https://<JETSON-IP>:8554 in browser)
  python simple_mobilenet.py --input v4l2:///dev/video0 \
  --output webrtc://<JETSON-IP>:8554/mystream \
  --network ssd-mobilenet-v2 \
  --codec h264 --latency 30 \
  --measure
"""
import os
import argparse
import time
from datetime import datetime

from jetson_inference import detectNet
from jetson_utils import (
    videoSource,
    videoOutput,
    saveImage,
    cudaAllocMapped,
    cudaCrop,
    cudaDeviceSynchronize,
    cudaFont,
)

from utils.stagetimer import StageTimer
from utils.overlay_helper import overlay_text
from utils.rtsp_utils import RTSPGate, is_rtsp_uri


def main():
    ap = argparse.ArgumentParser(
        description="Object detection (DetectNet/MobileNet) → display/RTSP/WebRTC with jetson-utils."
    )
    # IO
    ap.add_argument("--input",  type=str, default="v4l2:///dev/video0", help="input URI")
    ap.add_argument("--output", type=str, default="display://0", help="output URI (display://, rtsp://, webrtc://)")
    ap.add_argument("--width",  type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--bitrate",type=int, default=8_000_000)
    ap.add_argument("--codec",  type=str, default="h264", choices=["h264", "h265"])
    ap.add_argument("--latency", type=int, default=30, help="network buffering latency in ms (RTSP/WebRTC). Default is 30")

    # DetectNet options
    ap.add_argument("--network",   type=str, default="ssd-mobilenet-v2", help="pre-trained model to load")
    ap.add_argument("--overlay",   type=str, default="box,labels,conf",
                    help="detection overlay flags: 'box', 'labels', 'conf', 'none' (comma-separated)")
    ap.add_argument("--threshold", type=float, default=0.5, help="minimum detection threshold")
    ap.add_argument("--alpha",     type=int,   default=120, help="overlay alpha (transparency)")

    # Snapshots (optional)
    ap.add_argument("--save-snapshot",      dest="save_snapshot", action="store_true", help="save crops of detections")
    ap.add_argument("--dont-save-snapshot", dest="save_snapshot", action="store_false")
    ap.set_defaults(save_snapshot=False)
    ap.add_argument("--snapshot-dir", type=str, default="image/detection", help="directory to save detection crops")
    ap.add_argument("--timestamp",   type=str, default="%Y%m%d-%H%M%S",    help="strftime format for snapshot names")

    # Measurement & gating
    ap.add_argument("--measure", action="store_true", help="print per-stage timings and show on status bar")
    ap.add_argument("--render-when-viewed", action="store_true",
                    help="(RTSP only) call Render() only when a viewer is connected")
    ap.add_argument("--startup-grace", type=float, default=5.0,
                    help="(RTSP only) seconds to always allow rendering after start")
    ap.add_argument("--idle-timeout", type=float, default=None,
                    help="(RTSP only) auto-exit if no viewer for this many seconds (None=disabled)")

    args = ap.parse_args()

    videosource_dict = {"width": args.width, "height": args.height, "codec": "mjpeg", "encoder": "v4l2"}
    videooutput_dict = {"codec": args.codec, "encoder": "v4l2", "bitrate": args.bitrate}
    if args.latency is not None:
        videooutput_dict["latency"] = args.latency

    print(f"[videoSource] {args.input}  → {videosource_dict}")
    print(f"[videoOutput] {args.output} → {videooutput_dict}")
    print(f"[detectNet] network={args.network}, threshold={args.threshold}, overlay='{args.overlay}', alpha={args.alpha}")

    if args.save_snapshot and not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir, exist_ok=True)

    net = detectNet(args.network, argv=[], threshold=args.threshold)
    try:
        net.SetOverlayAlpha(args.alpha)
    except Exception:
        pass

    inp = videoSource(args.input, options=videosource_dict)
    out = videoOutput(args.output, options=videooutput_dict)

    gate = RTSPGate(args.output, startup_grace=args.startup_grace, idle_timeout=args.idle_timeout) \
           if args.render_when_viewed and is_rtsp_uri(args.output) else None

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
                img = inp.Capture()  # zero-copy GPU buffer
            if img is None:
                continue

            with timer.span("infer"):
                detections = net.Detect(img, overlay=args.overlay)

            with timer.span("post"):
                if args.save_snapshot and len(detections) > 0:
                    ts = datetime.now().strftime(args.timestamp)
                    for i, det in enumerate(detections):
                        roi = (int(det.Left), int(det.Top), int(det.Right), int(det.Bottom))
                        snap = cudaAllocMapped(width=roi[2]-roi[0], height=roi[3]-roi[1], format=img.format)
                        cudaCrop(img, snap, roi)
                        cudaDeviceSynchronize()
                        saveImage(os.path.join(args.snapshot_dir, f"{ts}-{i}.jpg"), snap, 100)
                        del snap

                overlay = f"{smoothed_fps:4.1f} FPS | dets {len(detections)}"
                overlay_text(font, img, overlay, x=10, y=10)

            if gate:
                gate.update()
                should_render = gate.should_render()
            else:
                should_render = True

            if should_render:
                with timer.span("rend"):
                    out.Render(img)

            if gate and gate.idle_timed_out():
                print("[rtsp] idle timeout — no viewer detected, exiting.")
                break

        frame_idx += 1
        now = time.time()
        if frame_idx > warm:
            dt = now - t_prev
            smoothed_fps = 0.9 * smoothed_fps + 0.1 * (1.0 / max(1e-6, dt))
        t_prev = now

        extra = timer.status_suffix(keys=("cap", "infer", "post", "rend", "tot"))
        view_tag = " | VIEW:" + ("ON" if (gate and gate.should_render()) else "OFF") if gate else ""
        out.SetStatus(f"{args.network} | Net {net.GetNetworkFPS():.0f} FPS | ~{smoothed_fps:4.1f} FPS{extra}{view_tag}")

        timer.end_frame()
        timer.log_this_interval(smoothed_fps, "mobilenet")

        # jetson-inference profiler (net.PrintProfilerTimes) This reports GPU kernel time only for the model and its internal pre/post/visualize steps
        if args.measure and (frame_idx % 600 == 0):
            net.PrintProfilerTimes()

        if not inp.IsStreaming() or not out.IsStreaming():
            break


if __name__ == "__main__":
    main()
