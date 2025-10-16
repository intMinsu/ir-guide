"""
EfficientViT-SAM (box-prompted) — DetectNet boxes + SAM masks (GPU-first), optional Gradio UI.

New (Gradio controls):
  • DetectNet overlay: 'box,labels,conf' or 'none'
  • DetectNet threshold (live)
  • Show & change DetectNet model at runtime (dropdown or custom)

- GPU-only post: no NumPy in the main loop (mask colorize/blend/composite on CUDA)
- Uses EfficientViTSamPredictor.set_image_cuda(...) and predict_boxes_cuda(...) if available.
  (Falls back to CPU-only SAM calls automatically if your predictor lacks the CUDA box API.)
- Optional Gradio UI:
    * WebRTC viewer embed (via gradio_utils.build_webrtc_embed)
    * Manual boxes: add (x1,y1,x2,y2) / random / undo / clear
    * Toggle: use detector on/off
    * Toggle: draw boxes on output
    * DetectNet overlay/threshold/model controls

Examples
--------
# Display (default detectNet + SAM L1 + FP16)
python efficientvit_sam_boundingbox.py --input /dev/video0 \
--output display://0 \
--sam-arch l1 \
--detect-network ssd-mobilenet-v2 \
--fp16 --measure

# WebRTC + Gradio interactive boxes (default detectNet + SAM L1 + FP16, recommended)
python efficientvit_sam_boundingbox.py --input /dev/video0 \
--output webrtc://<JETSON-IP>:8554/sam_boxes --codec h264 --latency 100 \
--sam-arch l1 \
--detect-network ssd-mobilenet-v2 \
--use-gradio --gradio-port 8050 \
--fp16 --measure
"""
from __future__ import annotations

import os
import argparse
import time
import threading
import random
from typing import Iterable, Tuple, List, Optional

import torch
import numpy as np  # only used in *fallback* path when CUDA box API is missing

import contextlib
from torch.cuda.amp import autocast

from jetson_inference import detectNet
from jetson_utils import (
    videoSource, videoOutput, cudaFont, cudaOverlay, cudaToNumpy,
    cudaDrawRect, cudaDrawLine, cudaDeviceSynchronize
)

from efficientvit.sam_model_zoo import create_efficientvit_sam_model
from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor

from common_utils.stagetimer import StageTimer
from common_utils.overlay_helper import overlay_text
from common_utils.rtsp_utils import RTSPGate, is_rtsp_uri
from core_utils.efficientvit_helpers import download_if_missing
from core_utils.preprocess import preprocess_cuda_rgb8_to_tensor

from seg_utils.segnet_utils import (
    SegmentationBuffers,
    alpha_blend_inplace,
    draw_boxes_inplace
)

from seg_utils.sam_utils import SAMBoxBuffers

# Reusable WebRTC embed
from gradio_utils import parse_webrtc_output, build_webrtc_embed


SAM_CKPT_URLS = {
    "l0": "https://huggingface.co/mit-han-lab/efficientvit-sam/resolve/main/efficientvit_sam_l0.pt",
    "l1": "https://huggingface.co/mit-han-lab/efficientvit-sam/resolve/main/efficientvit_sam_l1.pt",
}

# ------------------------------ DetectNet holder ------------------------------
class DetectNetHolder:
    """Thread-safe wrapper to hot-swap detectNet, adjust threshold, etc."""
    def __init__(self, net: detectNet, name_hint: str = ""):
        self._lock = threading.Lock()
        self.net = net
        try:
            self.name = net.GetNetworkName()
        except Exception:
            self.name = name_hint or "detectNet"

    def set_threshold(self, thr: float):
        with self._lock:
            try:
                self.net.SetThreshold(float(thr))
            except Exception:
                pass

    def swap_network(self, network: str, threshold: float, overlay_alpha: Optional[int] = None):
        """Create a new detectNet and swap it in."""
        new_net = detectNet(network, argv=[], threshold=threshold)
        if overlay_alpha is not None:
            try:
                new_net.SetOverlayAlpha(int(overlay_alpha))
            except Exception:
                pass
        with self._lock:
            old = self.net
            self.net = new_net
            try:
                self.name = new_net.GetNetworkName()
            except Exception:
                self.name = network
        # let GC clean up old

    def get_for_infer(self) -> detectNet:
        """Return current net without holding the lock during inference."""
        with self._lock:
            return self.net


# ------------------------------ args / model ------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="EfficientViT-SAM (box prompts) + DetectNet (Jetson, GPU-first)")
    # IO
    ap.add_argument("--input", type=str, default="v4l2:///dev/video0")
    ap.add_argument("--output", type=str, default="display://0")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--bitrate", type=int, default=8_000_000)
    ap.add_argument("--codec", type=str, default="h264", choices=["h264", "h265"])
    ap.add_argument("--latency", type=int, default=100)

    # fp16
    ap.add_argument("--fp16", action="store_true", help="use FP16 autocast for SAM inference")

    # detector
    ap.add_argument("--detect-network", type=str, default="ssd-mobilenet-v2")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--overlay-alpha", type=int, default=160)

    # SAM
    ap.add_argument("--sam-arch", type=str, default="l1", choices=["l1", "l2"])
    ap.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    ap.add_argument("--sam-checkpoint", type=str, default=None)
    ap.add_argument("--sam-checkpoint-url", type=str, default=None)

    # perf/ui
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--visualize", type=str, default="overlay", choices=["overlay", "mask", "overlay,mask"])
    ap.add_argument("--measure", action="store_true")

    # RTSP gating
    ap.add_argument("--render-when-viewed", action="store_true")
    ap.add_argument("--idle-timeout", type=float, default=300.0)

    # Gradio hybrid UI
    ap.add_argument("--use-gradio", action="store_true")
    ap.add_argument("--gradio-port", type=int, default=8051)

    # Interactive capacity
    ap.add_argument("--max-boxes", type=int, default=100)

    return ap.parse_args()


def load_sam(args):
    arch = args.sam_arch.lower()
    if args.sam_checkpoint:
        ckpt_path = args.sam_checkpoint
    else:
        url = args.sam_checkpoint_url or SAM_CKPT_URLS.get(arch)
        if not url:
            raise ValueError(
                f"No default SAM checkpoint URL for arch={arch} — pass --sam-checkpoint or --sam-checkpoint-url"
            )
        name = f"efficientvit_sam_{arch}.pt"
        ckpt_path = os.path.join(args.checkpoint_dir, name)
        ckpt_path = download_if_missing(ckpt_path, url)

    model = create_efficientvit_sam_model(
        name=f"efficientvit-sam-{arch}", pretrained=True, weight_url=ckpt_path
    ).eval().cuda()
    return EfficientViTSamPredictor(model)


# ------------------------------ Gradio UI ------------------------------
def start_gradio_ui(box_buf: SAMBoxBuffers,
                    net_holder: DetectNetHolder,
                    output_url: str, width: int, height: int,
                    port: int, ui_state: dict):
    import gradio as gr

    host_hint, webrtc_port, stream_name = parse_webrtc_output(output_url)
    webrtc_html = build_webrtc_embed(host_hint, webrtc_port, stream_name)

    # ----- helper getters/setters -----
    def _rows():
        return [[i, int(x1), int(y1), int(x2), int(y2)]
                for i, (x1, y1, x2, y2) in enumerate(box_buf.get_boxes())]

    def _status_text():
        return f"Model: **{net_holder.name}** | Threshold: **{ui_state.get('threshold', 0.5):.2f}** | Overlay: **{ui_state.get('detect_overlay','none')}**"

    # ----- manual boxes -----
    def _add_box(x1, y1, x2, y2):
        # clamp & order
        try:
            xi1 = max(0, min(width - 1, int(x1)))
            yi1 = max(0, min(height - 1, int(y1)))
            xi2 = max(0, min(width - 1, int(x2)))
            yi2 = max(0, min(height - 1, int(y2)))
        except Exception:
            return _rows()
        if xi2 < xi1: xi1, xi2 = xi2, xi1
        if yi2 < yi1: yi1, yi2 = yi2, yi1
        box_buf.add_box(xi1, yi1, xi2, yi2)
        return _rows()

    def _add_random():
        w = random.randint(max(8, width // 16), max(10, width // 4))
        h = random.randint(max(8, height // 16), max(10, height // 4))
        x1 = random.randint(0, max(0, width - w))
        y1 = random.randint(0, max(0, height - h))
        x2 = x1 + w
        y2 = y1 + h
        box_buf.add_box(x1, y1, x2, y2)
        return _rows()

    def _undo():
        box_buf.remove_last()
        return _rows()

    def _clear():
        box_buf.clear_boxes()
        return _rows()

    def _toggle_draw(val):
        ui_state["draw_boxes"] = bool(val)
        return f"Draw boxes: **{'ON' if ui_state['draw_boxes'] else 'OFF'}**"

    def _toggle_detector(val):
        ui_state["use_detector"] = bool(val)
        return f"Detector: **{'ON' if ui_state['use_detector'] else 'OFF'}**"

    # ----- DetectNet controls -----
    def _set_overlay(choice):
        ui_state["detect_overlay"] = choice
        return _status_text()

    def _set_threshold(val):
        thr = float(val)
        ui_state["threshold"] = thr
        net_holder.set_threshold(thr)
        return _status_text()

    DETECTNET_CHOICES = [
        "ssd-mobilenet-v2",
        "ssd-inception-v2",
        "pednet",
        "multiped-500",
        "facenet",
        "dashcam",
    ]

    def _apply_model(sel, custom):
        target = (custom or "").strip() or sel
        if not target:
            return _status_text()
        try:
            net_holder.swap_network(target, threshold=ui_state.get("threshold", 0.5),
                                    overlay_alpha=ui_state.get("overlay_alpha", 160))
            ui_state["network"] = target
        except Exception as e:
            # Keep status but append brief error
            return _status_text() + f"  \n⚠️ Failed to load **{target}**: `{e}`"
        return _status_text()

    # ----- UI -----
    with gr.Blocks(theme=gr.themes.Soft(primary_hue="lime")) as demo:
        gr.Markdown("### EfficientViT-SAM — Stream + Boxes (Detector + Manual)")
        gr.HTML(webrtc_html)

        with gr.Row():
            # Left: manual boxes & toggles
            with gr.Column():
                with gr.Row():
                    x1_in = gr.Number(label="x1", value=100, precision=0)
                    y1_in = gr.Number(label="y1", value=100, precision=0)
                    x2_in = gr.Number(label="x2", value=200, precision=0)
                    y2_in = gr.Number(label="y2", value=200, precision=0)
                with gr.Row():
                    add_btn   = gr.Button("Add box (x1,y1,x2,y2)", variant="primary")
                    rand_btn  = gr.Button("Add random box")
                with gr.Row():
                    undo_btn  = gr.Button("Undo")
                    clear_btn = gr.Button("Clear all", variant="stop")

                draw_chk   = gr.Checkbox(label="Draw boxes on stream", value=ui_state.get("draw_boxes", False))
                draw_stat  = gr.Markdown(_toggle_draw(ui_state.get("draw_boxes", False)))
                det_chk    = gr.Checkbox(label="Use detector", value=ui_state.get("use_detector", True))
                det_stat   = gr.Markdown(_toggle_detector(ui_state.get("use_detector", True)))

            with gr.Column(scale=0):
                tbl = gr.Dataframe(
                    headers=["idx", "x1", "y1", "x2", "y2"],
                    datatype=["number"] * 5,
                    value=_rows(),
                    interactive=False,
                    row_count=(0, "dynamic"),
                    col_count=5,
                    label="Manual boxes"
                )

        gr.Markdown("---")
        gr.Markdown("### DetectNet Controls")

        with gr.Row():
            overlay_sel = gr.Radio(
                choices=["box,labels,conf", "none"],
                value=ui_state.get("detect_overlay", "none"),
                label="Show DetectNet bounding box overlay",
            )
            thr_slider  = gr.Slider(0.05, 1.0, step=0.01,
                                    value=ui_state.get("threshold", 0.5),
                                    label="Detection threshold")

        with gr.Row():
            model_drop  = gr.Dropdown(DETECTNET_CHOICES,
                                      value=ui_state.get("network", "ssd-mobilenet-v2"),
                                      label="Select model")
            custom_txt  = gr.Textbox(label="...or custom model name/path", placeholder="e.g., ssd-mobilenet-v2")
            apply_btn   = gr.Button("Apply model", variant="primary")

        status_md = gr.Markdown(_status_text())

        # wiring
        add_btn.click(_add_box, inputs=[x1_in, y1_in, x2_in, y2_in], outputs=[tbl])
        rand_btn.click(_add_random, outputs=[tbl])
        undo_btn.click(_undo, outputs=[tbl])
        clear_btn.click(_clear, outputs=[tbl])

        draw_chk.change(_toggle_draw, inputs=[draw_chk], outputs=[draw_stat])
        det_chk.change(_toggle_detector, inputs=[det_chk], outputs=[det_stat])

        overlay_sel.change(_set_overlay, inputs=[overlay_sel], outputs=[status_md])
        thr_slider.release(_set_threshold, inputs=[thr_slider], outputs=[status_md])  # apply on release
        apply_btn.click(_apply_model, inputs=[model_drop, custom_txt], outputs=[status_md])

    threading.Thread(
        target=lambda: demo.launch(server_name="0.0.0.0", server_port=port, show_api=False),
        daemon=True,
    ).start()


# ------------------------------ main ------------------------------
def main():
    args = parse_args()

    videosource_dict = {"width": args.width, "height": args.height, "codec": "mjpeg", "encoder": "v4l2"}
    videooutput_dict = {"codec": args.codec, "encoder": "v4l2", "bitrate": args.bitrate}
    if args.latency is not None:
        videooutput_dict["latency"] = args.latency

    print(f"[videoSource] {args.input}  → {videosource_dict}")
    print(f"[videoOutput] {args.output} → {videooutput_dict}")

    inp = videoSource(args.input, options=videosource_dict)
    out = videoOutput(args.output, options=videooutput_dict)
    font = cudaFont()

    gate = RTSPGate(args.output, idle_timeout=args.idle_timeout) if args.render_when_viewed and is_rtsp_uri(
        args.output
    ) else None

    # detector + SAM
    net = detectNet(args.detect_network, argv=[], threshold=args.threshold)
    try:
        net.SetOverlayAlpha(args.overlay_alpha)
    except Exception:
        pass
    net_holder = DetectNetHolder(net, name_hint=args.detect_network)

    predictor = load_sam(args)

    timer = StageTimer(enabled=args.measure, log_interval=2.0)
    buffers = SegmentationBuffers(visualize=args.visualize)
    box_buf = SAMBoxBuffers(max_boxes=args.max_boxes, device="cuda")

    # Gradio UI state (shared toggles)
    ui_state = {
        "use_detector": True,
        "draw_boxes": False,
        "detect_overlay": "none",        # start clean; you can switch on in UI
        "threshold": float(args.threshold),
        "network": args.detect_network,
        "overlay_alpha": int(args.overlay_alpha),
    }
    if args.use_gradio:
        start_gradio_ui(box_buf, net_holder, args.output, args.width, args.height, args.gradio_port, ui_state)

    frame_idx, t_prev, smoothed_fps = 0, time.time(), 0.0

    # Check if CUDA box API exists
    has_cuda_boxes = hasattr(predictor, "predict_boxes_cuda")

    while True:
        timer.next_frame()
        with timer.span("tot"):
            with timer.span("cap"):
                img = inp.Capture(format="rgb8", timeout=1000)
            if img is None:
                continue

            h0, w0 = int(img.height), int(img.width)
            buffers.alloc((h0, w0), img.format)

            # 1) Build combined box list (detector + manual)
            with timer.span("infer"):
                cudaDeviceSynchronize()

                det_boxes: List[Tuple[int, int, int, int]] = []
                man_boxes = box_buf.get_boxes() if args.use_gradio else []

                net_local = net_holder.get_for_infer()
                if ui_state.get("use_detector", True):
                    overlay_choice = ui_state.get("detect_overlay", "none")
                    dets = net_local.Detect(img, overlay=overlay_choice)
                    for d in dets:
                        det_boxes.append((int(d.Left), int(d.Top), int(d.Right), int(d.Bottom)))

                all_boxes = det_boxes + man_boxes

                img_t = torch.as_tensor(img, device="cuda")  # (H,W,3) uint8
                predictor.set_image_cuda(img_t, image_format="RGB")

                accum_mask_t = None
                if len(all_boxes) > 0:
                    if has_cuda_boxes:
                        n = box_buf.load_boxes(all_boxes)
                        if n > 0:
                            boxes_t = box_buf.tensors(n)  # (N,4) float32 cuda

                            amp_ctx = autocast(dtype=torch.float16, enabled=(args.fp16 and torch.cuda.is_available()))
                            with torch.inference_mode(), amp_ctx:
                                masks, iou = predictor.predict_boxes_cuda(boxes_t, multimask_output=True)

                            masks_t = masks.to(torch.uint8) if masks.dtype != torch.uint8 else masks
                            accum_mask_t = masks_t.max(dim=0).values  # (H,W) 0/1
                    else:
                        # CPU fallback unchanged
                        rgb_np = cudaToNumpy(img)
                        predictor.set_image(rgb_np)
                        for (x1, y1, x2, y2) in all_boxes:
                            box_np = np.array([[x1, y1, x2, y2]], dtype=np.float32)
                            m, scores, _ = predictor.predict(
                                point_coords=None, point_labels=None, box=box_np, multimask_output=True
                            )
                            best = m[int(np.argmax(scores))]
                            m_t = torch.as_tensor(best, device="cuda", dtype=torch.uint8)
                            accum_mask_t = m_t if accum_mask_t is None else torch.maximum(accum_mask_t, m_t)
            
            # 2) Post (GPU colorize/blend/composite + optional box drawing)
            with timer.span("post"):
                if accum_mask_t is not None:
                    mask_long = accum_mask_t.to(torch.int64)
                    palette = torch.zeros((2, 3), device="cuda", dtype=torch.uint8)
                    palette[1] = torch.tensor([0, 255, 0], device="cuda", dtype=torch.uint8)  # green
                    color_mask_t = palette[mask_long]  # (H,W,3) u8

                    if buffers.use_overlay:
                        alpha_blend_inplace(buffers.overlay, img, color_mask_t, args.alpha)
                    if buffers.use_mask:
                        torch.as_tensor(buffers.mask, device="cuda").copy_(color_mask_t)
                else:
                    # pass-through if no mask this frame
                    if buffers.use_overlay:
                        torch.as_tensor(buffers.overlay, device="cuda").copy_(torch.as_tensor(img, device="cuda"))
                    if buffers.use_mask:
                        torch.as_tensor(buffers.mask, device="cuda").copy_(torch.as_tensor(img, device="cuda"))

                if buffers.use_composite:
                    cudaOverlay(buffers.overlay, buffers.composite, 0, 0)
                    cudaOverlay(buffers.mask, buffers.composite, buffers.overlay.width, 0)

                # Optional: draw extra boxes (our own overlay) if requested
                if ui_state.get("draw_boxes", False):
                    target = buffers.output
                    if det_boxes:
                        draw_boxes_inplace(target, det_boxes, color=(255, 255, 0, 255), thickness=2)
                    if man_boxes:
                        draw_boxes_inplace(target, man_boxes, color=(0, 200, 255, 255), thickness=2)

                # Status
                overlay_text(
                    font,
                    buffers.output,
                    f"SAM(box) det={len(det_boxes)} man={len(man_boxes)} | alpha={args.alpha:.2f}",
                    x=10,
                    y=10,
                )

            # 3) Render / stream
            if gate:
                gate.update()
                should_render = gate.should_render()
            else:
                should_render = True

            torch.cuda.synchronize()
            cudaDeviceSynchronize()

            if should_render:
                with timer.span("rend"):
                    out.Render(buffers.output)

            if gate and gate.idle_timed_out():
                print("[rtsp] idle timeout — no viewer detected, exiting.")
                break

        # FPS/status
        frame_idx += 1
        now = time.time()
        if frame_idx > 10:
            dt = now - t_prev
            smoothed_fps = 0.9 * smoothed_fps + 0.1 * (1.0 / max(1e-6, dt))
        t_prev = now

        extra = timer.status_suffix(keys=("cap", "post", "rend", "tot"))
        view_tag = " | VIEW:" + ("ON" if (gate and gate.should_render()) else "OFF") if gate else ""
        out.SetStatus(
            f"SAM(box)+{net_holder.name}{' FP16-AMP' if args.fp16 else ''} | thr {ui_state.get('threshold',0.5):.2f} "
            f"| overlay {ui_state.get('detect_overlay','none')} | ~{smoothed_fps:4.1f} FPS{extra}{view_tag}"
        )
        timer.end_frame()
        timer.log_this_interval(smoothed_fps, "sam-box")

        if not inp.IsStreaming() or not out.IsStreaming():
            break


if __name__ == "__main__":
    main()
