"""
jetson-utils zero-copy path (built-in RTSP server if 'rtsp://...'):
- Input:  videoSource('v4l2:///dev/video0', options=...)
- Output: videoOutput('rtsp://localhost:8554/mystream', options=...)

Note:
  • If MediaMTX already uses :8554, stop it or change the output port here.
---
jetson-utils의 제로 카피(zero-copy) 경로를 사용하여 RTSP 스트림을 발행합니다.

'rtsp://...' 형식의 출력을 지정하면 내장된 RTSP 서버가 활성화됩니다.

- 입력: videoSource('v4l2:///dev/video0', options=...)
- 출력: videoOutput('rtsp://localhost:8554/mystream', options=...)

주의:
    • 만약 MediaMTX가 이미 8554 포트를 사용 중이라면, 해당 서비스를 중지하거나
      스크립트의 출력 포트를 다른 번호로 변경해야 합니다.

Example code(실행 예시):
    python simple_videocapture_rtsp.py \
        --input v4l2:///dev/video0 \
        --output rtsp://localhost:8554/mystream \
        --width 1920 --height 1080 --bitrate 8000000 --codec h265

"""
import argparse
from jetson_utils import videoSource, videoOutput

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  type=str, default="v4l2:///dev/video0", help="input URI")
    ap.add_argument("--output", type=str, default="rtsp://localhost:8554/mystream", help="output URI")
    ap.add_argument("--width",  type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--bitrate",type=int, default=8_000_000)
    ap.add_argument("--codec",  type=str, default="h265", choices=["h264","h265"])
    args = ap.parse_args()

    videosource_dict = {
        "width": args.width,
        "height": args.height,
        "codec": "mjpeg",   # Logitech C920 works well with MJPEG
        "encoder": "v4l2",
    }
    videooutput_dict = {
        "codec": args.codec,
        "encoder": "v4l2",
        "bitrate": args.bitrate,
    }

    print(f"[videoSource] {args.input}  → {videosource_dict}")
    print(f"[videoOutput] {args.output} → {videooutput_dict}")

    inp = videoSource(args.input, options=videosource_dict)
    out = videoOutput(args.output, options=videooutput_dict)

    while True:
        img = inp.Capture(format='rgb8', timeout=1000)
        if img is None:
            continue

        # Example: overlay text (uncomment if needed)
        # from jetson_utils import cudaFont
        # font = cudaFont()
        # font.OverlayText(img, 10, 10, "Jetson RTSP", 255, 255, 255, 255)

        out.Render(img)

        if not inp.IsStreaming() or not out.IsStreaming():
            break

if __name__ == "__main__":
    main()
