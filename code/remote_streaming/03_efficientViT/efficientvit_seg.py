# efficientvit_seg.py
"""
EfficientViT (semantic segmentation) — live camera → overlay/mask/composite via SegmentationBuffers + cudaOverlay

What it does
------------
- Captures frames (camera/file), runs EfficientViT segmentation on GPU.
- Colors the class mask on GPU and either:
  • alpha-blends it over the input (overlay), or
  • outputs the mask alone, or
  • shows overlay|mask side-by-side (composite) with cudaOverlay.
- Everything stays on the GPU (zero-copy), no CPU/Numpy post-processing.

RTSP caution (VLC / first-frame freeze)
---------------------------------------
If your RTSP client (e.g., VLC) connects but shows a stuck frame or fails to start:
- Prefer `--codec h264` and add a small jitter buffer like `--latency 100` (100–200 ms often helps).
- On the client, you can try TCP and network buffering:
    VLC:  `--rtsp-tcp --network-caching=200`
This helps ensure SPS/PPS/IDR arrive intact at startup and prevents stalls.


--------------------------------------    
Examples (FP16 native PyTorch runtime)
--------------------------------------
# Local monitor (PyTorch, FP16) Cityscapes, L2 @ 1024x2048
python efficientvit_seg.py --input v4l2:///dev/video0 \
  --output display://0 \
  --arch l2 --resolution 1024 2048 \
  --dataset cityscapes \
  --fp16 --measure

# RTSP (PyTorch, FP16) ADE20K, L2 @ 512x512
python efficientvit_seg.py --input v4l2:///dev/video0 \
--output rtsp://<JETSON-IP>:8554/seg \
--arch l2 --resolution 512 512 \
--dataset ade20k \
--codec h264 --latency 100 \
--render-when-viewed --fp16 --measure

# WebRTC (PyTorch, FP16) ADE20K, L2 @ 512x512
python efficientvit_seg.py --input v4l2:///dev/video0 \
--output webrtc://<JETSON-IP>:8554/seg \
--arch l2 --resolution 512 512 \
--dataset ade20k \
--codec h264 --latency 100 \
--fp16 --measure

"""
import os
import argparse
import time
import torch
import torch.nn.functional as F

from jetson_utils import videoSource, videoOutput, cudaFont, cudaOverlay

from efficientvit.seg_model_zoo import create_efficientvit_seg_model

from common_utils.stagetimer import StageTimer
from common_utils.overlay_helper import overlay_text
from common_utils.rtsp_utils import RTSPGate, is_rtsp_uri

from core_utils.efficientvit_helpers import (
    download_if_missing,
    preprocess_cuda_rgb8_to_tensor,
)

from seg_utils.segnet_utils import (
    SegmentationBuffers,
    generate_palette,
    alpha_blend_inplace,
)

# Known checkpoints
SEG_CKPT_URLS = {
    ("l1", "cityscapes"): "https://huggingface.co/han-cai/efficientvit-seg/resolve/main/efficientvit_seg_l1_cityscapes.pt",
    ("l1", "ade20k"):     "https://huggingface.co/han-cai/efficientvit-seg/resolve/main/efficientvit_seg_l1_ade20k.pt",
    ("l2", "cityscapes"): "https://huggingface.co/han-cai/efficientvit-seg/resolve/main/efficientvit_seg_l2_cityscapes.pt",
    ("l2", "ade20k"):     "https://huggingface.co/han-cai/efficientvit-seg/resolve/main/efficientvit_seg_l2_ade20k.pt",
}


def parse_args():
    ap = argparse.ArgumentParser(description="EfficientViT semantic segmentation (Torch)")
    # IO
    ap.add_argument("--input",  type=str, default="v4l2:///dev/video0")
    ap.add_argument("--output", type=str, default="display://0")
    ap.add_argument("--width",  type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--bitrate",type=int, default=8_000_000)
    ap.add_argument("--codec",  type=str, default="h264", choices=["h264","h265"])
    ap.add_argument("--latency", type=int, default=100)

    # model
    ap.add_argument("--arch", type=str, default="l1", choices=["l1","l2"])
    ap.add_argument("--dataset", type=str, default="ade20k", choices=["ade20k","cityscapes"])
    ap.add_argument("--resolution", type=int, nargs=2, default=[512,512], metavar=("H","W"))
    ap.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--checkpoint-url", type=str, default=None)

    # perf/ui
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--alpha", type=float, default=0.5, help="mask overlay alpha [0,1]")
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--visualize", type=str, default="overlay",
                    choices=["overlay","mask","overlay,mask"],
                    help="what to render: overlay and/or mask (composite side-by-side when both)")

    # RTSP gating
    ap.add_argument("--render-when-viewed", action="store_true")
    ap.add_argument("--idle-timeout", type=float, default=300.0)

    return ap.parse_args()


def load_seg_model(args):
    arch = args.arch.lower()
    dataset = args.dataset.lower()

    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        url = args.checkpoint_url or SEG_CKPT_URLS.get((arch, dataset))
        if not url:
            raise ValueError(f"No default seg checkpoint URL for ({arch}, {dataset}) — pass --checkpoint or --checkpoint-url")
        ckpt_name = f"efficientvit_seg_{arch}_{dataset}.pt"
        ckpt_path = os.path.join(args.checkpoint_dir, ckpt_name)
        ckpt_path = download_if_missing(ckpt_path, url)

    model_name = f"efficientvit-seg-{arch}-{dataset}"
    model = create_efficientvit_seg_model(name=model_name, pretrained=True, weight_url=ckpt_path)

    model.eval()
    if args.device.startswith("cuda") and torch.cuda.is_available():
        model = model.to("cuda")
        if args.fp16:
            print("[info] FP16 autocast (AMP) enabled; model weights remain FP32")
    else:
        print("[warn] CUDA not available; running on CPU.")
    return model


def main():
    args = parse_args()

    H_in, W_in = int(args.resolution[0]), int(args.resolution[1])
    model = load_seg_model(args)

    videosource_dict = {"width": args.width, "height": args.height, "codec": "mjpeg", "encoder":"v4l2"}
    videooutput_dict = {"codec": args.codec, "encoder": "v4l2", "bitrate": args.bitrate}
    if args.latency is not None:
        videooutput_dict["latency"] = args.latency

    print(f"[videoSource] {args.input}  → {videosource_dict}")
    print(f"[videoOutput] {args.output} → {videooutput_dict}")
    print(f"[model] EfficientViT-SEG ({args.dataset}) @ {H_in}x{W_in} | fp16={args.fp16}")

    inp  = videoSource(args.input, options=videosource_dict)
    vout = videoOutput(args.output, options=videooutput_dict)
    font = cudaFont()

    gate = RTSPGate(args.output, idle_timeout=args.idle_timeout) \
           if args.render_when_viewed and is_rtsp_uri(args.output) else None

    device = "cuda" if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    timer  = StageTimer(enabled=args.measure, log_interval=2.0)

    # buffer manager (overlay/mask/composite)
    buffers = SegmentationBuffers(visualize=args.visualize)

    # palette cache (GPU)
    palette_t = None
    C_cached  = None

    frame_idx, t_prev, smoothed_fps = 0, time.time(), 0.0

    while True:
        timer.next_frame()
        with timer.span("tot"):
            with timer.span("cap"):
                img = inp.Capture(format="rgb8", timeout=1000)
            if img is None:
                continue

            h0, w0 = int(img.height), int(img.width)
            buffers.alloc((h0, w0), img.format)

            with timer.span("prep"):
                t_in = preprocess_cuda_rgb8_to_tensor(img, (H_in, W_in), device=device)

            with timer.span("infer"):
                with torch.inference_mode():
                    if args.fp16 and device == "cuda":
                        with torch.cuda.amp.autocast(dtype=torch.float16):
                            logits = model(t_in) # (1, C, H_in, W_in)
                    else:
                        logits = model(t_in) # (1, C, H_in, W_in)
                              
                    if isinstance(logits, (list, tuple)):
                        logits = logits[0]
                    logits = logits.float()
                    C = logits.shape[1]

            with timer.span("post"):
                # argmax + upsample on GPU
                mask_small = logits.argmax(dim=1).squeeze(0)     # (H_in, W_in) long (CUDA)
                mask_up = F.interpolate(
                    mask_small[None, None].float(), size=(h0, w0), mode="nearest"
                ).squeeze(0).squeeze(0).to(dtype=torch.int64)    # (H0, W0) long (CUDA)

                # palette (cache on GPU)
                if palette_t is None or C_cached != C:
                    palette_t = generate_palette(C)  # Cx3 uint8
                    palette_t = palette_t.to(dtype=torch.uint8)
                    C_cached   = C

                # colorize on GPU -> (H0, W0, 3) uint8 CUDA
                color_mask_t = palette_t[mask_up]                # gather

                # render targets
                if buffers.use_overlay:
                    # overlay ← alpha(frame, color_mask) (GPU only)
                    alpha_blend_inplace(buffers.overlay, img, color_mask_t, args.alpha)

                if buffers.use_mask:
                    # mask ← color_mask (GPU copy)
                    torch.as_tensor(buffers.mask, device='cuda').copy_(color_mask_t)

                if buffers.use_composite:
                    # left: overlay, right: color mask
                    cudaOverlay(buffers.overlay, buffers.composite, 0, 0)
                    cudaOverlay(buffers.mask,    buffers.composite, buffers.overlay.width, 0)

                # status text (drawn on whichever buffer is routed to output)
                overlay_text(font, buffers.output, f"{args.dataset} {C}c | alpha={args.alpha:.2f}", x=10, y=10)

            if gate:
                gate.update()
                should_render = gate.should_render()
            else:
                should_render = True

            # Fence PyTorch work before handing buffer to jetson-utils (stream 0)
            torch.cuda.synchronize()

            if should_render:
                with timer.span("rend"):
                    vout.Render(buffers.output)

            if gate and gate.idle_timed_out():
                print("[rtsp] idle timeout — no viewer detected, exiting.")
                break

        # FPS smoothing & status
        frame_idx += 1
        now = time.time()
        if frame_idx > 10:
            dt = now - t_prev
            smoothed_fps = 0.9 * smoothed_fps + 0.1 * (1.0 / max(1e-6, dt))
        t_prev = now

        extra = timer.status_suffix(keys=("cap","prep","infer","post","rend","tot"))
        view_tag = " | VIEW:" + ("ON" if (gate and gate.should_render()) else "OFF") if gate else ""
        vout.SetStatus(f"EfficientViT-SEG {args.dataset} | ~{smoothed_fps:4.1f} FPS{extra}{view_tag}")

        timer.end_frame()
        timer.log_this_interval(smoothed_fps, f"seg-{args.dataset}")

        if not inp.IsStreaming() or not vout.IsStreaming():
            break


if __name__ == "__main__":
    main()
