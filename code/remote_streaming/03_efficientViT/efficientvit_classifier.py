# ir-guide/code/remote_streaming/03_efficientViT/efficientvit_classifier.py
"""
EfficientViT (classification) — live camera → top-k label overlay → display/RTSP/WebRTC

- IO flags:
    --input / --output / --width / --height / --bitrate / --codec / --latency
- Auto-downloads checkpoints to ./checkpoints/ (for torch runtime)
- Measurement:
    --measure            print & status-bar per-stage timings
- Model flags:
    --arch l1/l2/l3      EfficientViT architecture (default: l1)
    --resolution 224/256/288/320/384 (default: 224; l1 only supports 224)
    --labels             path to labels.txt (default: None, uses ImageNet-1k)
    --checkpoint-dir     directory to store checkpoints (default: ./checkpoints)
    --checkpoint         path to checkpoint (default: None, auto-downloads)
    --checkpoint-url     URL to checkpoint (default: None, uses built-in URL)
- Inference flags:
    --topk               number of top-k predictions to overlay (default: 3)
    --device            'cuda' (default) or 'cpu'
    --fp16              use FP16 (default: False, uses FP32)

Export ONNX / build TensorRT with trtexec (via efficientvit_tensorrt)
-------------------------------------
- Runtimes:
    --runtime torch      (default; uses EfficientViT PyTorch model)
    --runtime tensorrt   (loads a .engine; use --engine to point to it)

RTSP caution (VLC / first-frame freeze)
---------------------------------------
If your RTSP client (e.g., VLC) connects but shows a stuck frame or fails to start:
- Prefer `--codec h264` and add a small jitter buffer like `--latency 100` (100–200 ms often helps).
- On the client, you can try TCP and network buffering:
    VLC:  `--rtsp-tcp --network-caching=200`
This helps ensure SPS/PPS/IDR arrive intact at startup and prevents stalls.

--------------------------------------    
Examples1 (FP16 native PyTorch runtime)
--------------------------------------
  # Local monitor (PyTorch, FP16) L2 @ 384x384
  python efficientvit_classifier.py --input v4l2:///dev/video0 \
  --output display://0 \
  --arch l2 --resolution 384 \
  --fp16 --measure
  
  # RTSP (PyTorch, FP16) L2 @ 384x384
  python efficientvit_classifier.py --input v4l2:///dev/video0 \
  --output rtsp://<JETSON-IP>:8554/efficientvit \
  --arch l2 --resolution 384 \
  --codec h264 --latency 30 \
  --render-when-viewed --fp16 --measure

  # WebRTC (open https://<JETSON-IP>:8554) L2 @ 384x384
  python efficientvit_classifier.py --input v4l2:///dev/video0 \
  --output webrtc://<JETSON-IP>:8554/efficientvit --codec h264 --latency 30 \
  --arch l2 --resolution 384 \
  --fp16 --measure

--------------------------------------
Examples2 (FP16 TensorRT runtime)
--------------------------------------
  First, build the engine (if you don't have one):
    # Export ONNX & build TensorRT (no run) we now assume L2 @ 320x320.
    python efficientvit_classifier.py --arch l2 --resolution 320 \
    --export-onnx assets/export/efficientvit_l2_r320.onnx \
    --build-engine assets/export/efficientvit_l2_r320_fp16.engine --no-run

  Second, run with TensorRT:
    # Local monitor (TensorRT, FP16)
    python efficientvit_classifier.py --runtime tensorrt \
        --engine assets/export/efficientvit_l2_r320_fp16.engine --output display://0 \
        --arch l2 --resolution 320 --topk 3 --fp16 --measure 
    
    # WebRTC (TensorRT, FP16)
    python efficientvit_classifier.py --runtime tensorrt \
        --engine assets/export/efficientvit_l2_r320_fp16.engine \
        --output webrtc://<JETSON-IP>:8554/efficientvit --codec h
        --latency 30 --arch l2 --resolution 320 --topk 3 --fp16 --measure

"""

import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F

from jetson_utils import videoSource, videoOutput, cudaFont
from efficientvit.cls_model_zoo import create_efficientvit_cls_model

from common_utils.overlay_helper import overlay_text
from common_utils.stagetimer import StageTimer
from common_utils.rtsp_utils import RTSPGate, is_rtsp_uri

from core_utils.efficientvit_helpers import (
    download_if_missing,
    preprocess_cuda_rgb8_to_tensor,
)
from cls_utils.clsnet_utils import load_labels
from core_utils.tensorrt_utils import export_onnx, build_trt_engine_with_trtexec
from cls_utils.clsnet_tensorrt import TRTClassifier

CLS_CKPT_URLS = {
    ("l1", 224): "https://huggingface.co/han-cai/efficientvit-cls/resolve/main/efficientvit_l1_r224.pt",
    ("l2", 224): "https://huggingface.co/han-cai/efficientvit-cls/resolve/main/efficientvit_l2_r224.pt",
    ("l2", 256): "https://huggingface.co/han-cai/efficientvit-cls/resolve/main/efficientvit_l2_r256.pt",
    ("l2", 288): "https://huggingface.co/han-cai/efficientvit-cls/resolve/main/efficientvit_l2_r288.pt",
    ("l2", 320): "https://huggingface.co/han-cai/efficientvit-cls/resolve/main/efficientvit_l2_r320.pt",
    ("l2", 384): "https://huggingface.co/han-cai/efficientvit-cls/resolve/main/efficientvit_l2_r384.pt",
    ("l3", 224): "https://huggingface.co/han-cai/efficientvit-cls/resolve/main/efficientvit_l3_r224.pt",
    ("l3", 256): "https://huggingface.co/han-cai/efficientvit-cls/resolve/main/efficientvit_l3_r256.pt",
    ("l3", 288): "https://huggingface.co/han-cai/efficientvit-cls/resolve/main/efficientvit_l3_r288.pt",
    ("l3", 320): "https://huggingface.co/han-cai/efficientvit-cls/resolve/main/efficientvit_l3_r320.pt",
    ("l3", 384): "https://huggingface.co/han-cai/efficientvit-cls/resolve/main/efficientvit_l3_r384.pt",
}
def _default_ckpt_url(arch: str, res: int) -> str:
    if (arch, res) in CLS_CKPT_URLS:
        return CLS_CKPT_URLS[(arch, res)]
    raise ValueError(f"[warn] no default checkpoint known for arch={arch} res={res}. Provide --checkpoint or --checkpoint-url.")


def parse_args():
    ap = argparse.ArgumentParser(description="EfficientViT classification on Jetson (display/RTSP/WebRTC)")
    # IO
    ap.add_argument("--input",  type=str, default="v4l2:///dev/video0")
    ap.add_argument("--output", type=str, default="display://0")
    ap.add_argument("--width",  type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--bitrate",type=int, default=8_000_000)
    ap.add_argument("--codec",  type=str, default="h264", choices=["h264","h265"])
    ap.add_argument("--latency", type=int, default=100)

    # model / labels
    ap.add_argument("--arch", type=str, default="l1", choices=["l1","l2","l3"])
    ap.add_argument("--resolution", type=int, default=224)
    ap.add_argument("--labels", type=str, default=None)
    ap.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--checkpoint-url", type=str, default=None)

    # export/build
    ap.add_argument("--export-onnx", type=str, default=None)
    ap.add_argument("--onnx-opset", type=int, default=13)
    ap.add_argument("--build-engine", type=str, default=None)
    ap.add_argument("--trtexec-extra", type=str, nargs="*", default=None)
    ap.add_argument("--no-run", action="store_true")

    # runtime/perf
    ap.add_argument("--runtime", type=str, default="torch", choices=["torch","tensorrt"])
    ap.add_argument("--engine", type=str, default=None)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--measure", action="store_true")

    # RTSP gating
    ap.add_argument("--render-when-viewed", action="store_true")
    ap.add_argument("--idle-timeout", type=float, default=300.0)

    return ap.parse_args()

def load_cls_model(args) -> torch.nn.Module:
    arch = args.arch.lower()
    res  = int(args.resolution)

    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        url = args.checkpoint_url or _default_ckpt_url(arch, res)
        ckpt_name = f"efficientvit_{arch}_r{res}.pt"
        ckpt_path = os.path.join(args.checkpoint_dir, ckpt_name)
        ckpt_path = download_if_missing(ckpt_path, url)

    model_name_res  = f"efficientvit-{arch}-r{res}"
    model_name_arch = f"efficientvit-{arch}"
    try:
        model = create_efficientvit_cls_model(name=model_name_res,  pretrained=True, weight_url=ckpt_path)
    except ValueError:
        model = create_efficientvit_cls_model(name=model_name_arch, pretrained=True, weight_url=ckpt_path)

    model.eval()
    if args.device.startswith("cuda") and torch.cuda.is_available():
        model = model.to("cuda")
        if args.fp16:
            print("[info] using FP16")
    else:
        print("[warn] CUDA not available; running on CPU.")
    return model

def maybe_export_and_build(args, model):
    if args.export_onnx:
        export_onnx(model, args.export_onnx, input_hw=(args.resolution, args.resolution), opset=args.onnx_opset)
    if args.build_engine and args.export_onnx:
        build_trt_engine_with_trtexec(args.export_onnx, args.build_engine, fp16=True, extra=args.trtexec_extra)
    if args.no_run:
        print("[done] export/build requested with --no-run, exiting.")
        sys.exit(0)

def softmax_np(x: np.ndarray, axis: int = 1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x, dtype=np.float32)
    return e / np.sum(e, axis=axis, keepdims=True)

def topk_np(probs: np.ndarray, k: int):
    k = max(1, k)
    part = np.argpartition(-probs[0], kth=range(k))[:k]
    order = np.argsort(-probs[0, part])
    idx = part[order]
    return probs[:, idx], idx.reshape(1, -1)

def main():
    args = parse_args()
    labels = load_labels(args.labels, num_classes=1000)

    trt_classifier = None
    model = None
    if args.runtime == "torch":
        model = load_cls_model(args)
    else:
        engine_path = args.engine or args.build_engine or f"assets/export/efficientvit_{args.arch}_r{args.resolution}_fp16.engine"
        print(f"[runtime] TensorRT engine: {engine_path}")
        trt_classifier = TRTClassifier(engine_path)

    if args.runtime == "torch":
        maybe_export_and_build(args, model)
    else:
        if args.export_onnx or args.build_engine:
            tmp_model = load_cls_model(args)
            maybe_export_and_build(args, tmp_model)

    videosource_dict = {"width": args.width, "height": args.height, "codec": "mjpeg", "encoder":"v4l2"}
    videooutput_dict = {"codec": args.codec, "encoder": "v4l2", "bitrate": args.bitrate}
    if args.latency is not None:
        videooutput_dict["latency"] = args.latency

    print(f"[videoSource] {args.input}  → {videosource_dict}")
    print(f"[videoOutput] {args.output} → {videooutput_dict}")
    print(f"[model] EfficientViT-{args.arch.upper()} @ {args.resolution} | runtime={args.runtime} (fp16={args.fp16})")

    inp = videoSource(args.input, options=videosource_dict)
    out = videoOutput(args.output, options=videooutput_dict)
    font = cudaFont()

    gate = RTSPGate(args.output, idle_timeout=args.idle_timeout) \
           if args.render_when_viewed and is_rtsp_uri(args.output) else None

    res_hw = (args.resolution, args.resolution)
    device = "cuda" if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"

    frame_idx, t_prev, smoothed_fps = 0, time.time(), 0.0
    timer = StageTimer(enabled=args.measure, log_interval=2.0)

    while True:
        timer.next_frame()
        with timer.span("tot"):
            with timer.span("cap"):
                img = inp.Capture(format="rgb8", timeout=1000)
            if img is None:
                continue

            with timer.span("prep"):
                t_tensor = preprocess_cuda_rgb8_to_tensor(img, res_hw, device=device)

            with timer.span("infer"):
                if args.runtime == "torch":
                    with torch.inference_mode():
                        if args.fp16 and next(model.parameters()).is_cuda:
                            with torch.cuda.amp.autocast(dtype=torch.float16):
                                logits = model(t_tensor)
                        else:
                            logits = model(t_tensor)
                        probs_t = F.softmax(logits.float(), dim=1)
                        topk_prob_t, topk_idx_t = probs_t.topk(k=max(1, args.topk), dim=1)
                        topk_prob = topk_prob_t.detach().cpu().numpy()
                        topk_idx = topk_idx_t.detach().cpu().numpy()
                else:
                    t_np = t_tensor.float().cpu().numpy()
                    logits_np = trt_classifier.infer(t_np)
                    probs_np = softmax_np(logits_np, axis=1)
                    topk_prob, topk_idx = topk_np(probs_np, k=max(1, args.topk))

            # overlay predicted labels with class id
            txts = []
            for rank in range(topk_idx.shape[1]):
                cls_id = int(topk_idx[0, rank]); p = float(topk_prob[0, rank])
                label = labels[cls_id] if cls_id < len(labels) else f"class_{cls_id}"
                txts.append(f"{rank+1}:[{cls_id}] {label} ({p*100:.1f}%)")
            overlay_text(font, img, " | ".join(txts), x=10, y=10)

            if gate:
                gate.update()
                # print(f"[rtsp] connected={gate._connected_now} last_seen={gate._last_seen_connected}")
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
        if frame_idx > 10:
            dt = now - t_prev
            smoothed_fps = 0.9 * smoothed_fps + 0.1 * (1.0 / max(1e-6, dt))
        t_prev = now

        extra = timer.status_suffix(keys=("cap", "prep", "infer", "tot"))
        view_tag = " | VIEW:" + ("ON" if (gate and gate.should_render()) else "OFF") if gate else ""
        out.SetStatus(f"EfficientViT-{args.arch.upper()} | ~{smoothed_fps:4.1f} FPS ({args.runtime}{' FP16-AMP' if args.fp16 and args.runtime=='torch' else ''}){extra}{view_tag}")

        timer.end_frame()
        timer.log_this_interval(smoothed_fps, f"{args.runtime}{' FP16' if args.fp16 else ''}")

        if not inp.IsStreaming() or not out.IsStreaming():
            break

if __name__ == "__main__":
    main()
