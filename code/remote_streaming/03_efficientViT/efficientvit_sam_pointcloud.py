# efficientvit_sam_pointcloud.py
"""
EfficientViT-SAM (point-prompted) — user points → instance mask → overlay/mask/composite (GPU)

- GPU-only post (no NumPy): uses SegmentationBuffers for overlay/mask/composite.
- Optional Gradio UI (hybrid): WebRTC video stays on jetson-utils, Gradio collects clicks.
- Minimal WebRTC viewer/embed logic moved to gradio_utils.

Examples
--------
# Display with two fixed points (SAM L1+FP16)
python efficientvit_sam_pointcloud.py --input /dev/video0 \
--output display://0 \
--sam-arch l1 \
--points "200,150; 400,240" --highlight-points \
--fp16 --measure

# WebRTC + Gradio interactive points (SAM L1+FP16, recommended)
python efficientvit_sam_pointcloud.py --input /dev/video0 \
--output webrtc://<Jetson IP>:8554/sam_points --codec h264 --latency 100 \
--sam-arch l1 \
--use-gradio --gradio-port 8050 \
--fp16 --measure
"""
import os, argparse, time, threading, json, torch
import contextlib
from torch.cuda.amp import autocast

from jetson_utils import videoSource, videoOutput, cudaFont, cudaOverlay, cudaDeviceSynchronize
from efficientvit.sam_model_zoo import create_efficientvit_sam_model
from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor
from common_utils.stagetimer import StageTimer
from common_utils.overlay_helper import overlay_text
from common_utils.rtsp_utils import RTSPGate, is_rtsp_uri
from core_utils.efficientvit_helpers import download_if_missing

from seg_utils.segnet_utils import (
    SegmentationBuffers,
    alpha_blend_inplace,
    draw_points_inplace_labeled,   # NEW
)
from seg_utils.sam_utils import SAMPointBuffers

from gradio_utils import parse_webrtc_output, build_webrtc_embed

SAM_CKPT_URLS = {
    "l0": "https://huggingface.co/mit-han-lab/efficientvit-sam/resolve/main/efficientvit_sam_l0.pt",
    "l1": "https://huggingface.co/mit-han-lab/efficientvit-sam/resolve/main/efficientvit_sam_l1.pt",
}

def parse_points(s: str):
    """
    Accepts 'x,y' (FG by default) or 'x,y,label' where label in {0,1}.
    Example: '200,150; 400,240,0' → [(200,150,1),(400,240,0)]
    """
    pts = []
    for tok in s.split(";"):
        tok = tok.strip()
        if not tok:
            continue
        try:
            parts = [t.strip() for t in tok.split(",")]
            if len(parts) >= 3:
                x_str, y_str, l_str = parts[:3]
                pts.append((float(x_str), float(y_str), 1 if int(l_str) != 0 else 0))
            else:
                x_str, y_str = parts[:2]
                pts.append((float(x_str), float(y_str), 1))  # default FG
        except Exception:
            print(f"[warn] couldn't parse point '{tok}', expected 'x,y' or 'x,y,label'")
    return pts

def parse_color(s: str):
    try:
        r, g, b = [int(t.strip()) for t in s.split(",")]
        return (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)), 255)
    except Exception:
        return (255, 255, 0, 255)

def parse_args():
    ap = argparse.ArgumentParser(description="EfficientViT-SAM point prompts (Jetson, GPU-first)")
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

    # SAM
    ap.add_argument("--sam-arch", type=str, default="l1", choices=["l0", "l1"])
    ap.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    ap.add_argument("--sam-checkpoint", type=str, default=None)
    ap.add_argument("--sam-checkpoint-url", type=str, default=None)

    # prompts
    ap.add_argument("--points", type=str, default="", help='points as "x,y[,[0|1]]; ..." (1=FG blue, 0=BG red)')
    ap.add_argument("--highlight-points", action="store_true", help="draw points on the output")
    ap.add_argument("--point-radius", type=int, default=4, help="radius (px) for point highlight")
    ap.add_argument("--point-color", type=str, default="255,255,0", help="(unused when labeled draw is active)")
    ap.add_argument("--max-points", type=int, default=100, help="fixed GPU capacity for point prompts")

    # perf/ui
    ap.add_argument("--alpha", type=float, default=0.5, help="mask overlay alpha [0,1]")
    ap.add_argument("--visualize", type=str, default="overlay", choices=["overlay", "mask", "overlay,mask"])
    ap.add_argument("--measure", action="store_true")

    # RTSP gating
    ap.add_argument("--render-when-viewed", action="store_true")
    ap.add_argument("--idle-timeout", type=float, default=300.0)

    # Gradio hybrid UI
    ap.add_argument("--use-gradio", action="store_true", help="launch Gradio UI for interactive point clicks")
    ap.add_argument("--gradio-port", type=int, default=8050, help="HTTP port for Gradio UI")
    return ap.parse_args()

def load_sam(args):
    arch = args.sam_arch.lower()
    if args.sam_checkpoint:
        ckpt_path = args.sam_checkpoint
    else:
        url = args.sam_checkpoint_url or SAM_CKPT_URLS.get(arch)
        if not url:
            raise ValueError(f"No default SAM checkpoint URL for arch={arch} — pass --sam-checkpoint or --sam-checkpoint-url")
        name = f"efficientvit_sam_{arch}.pt"
        ckpt_path = os.path.join(args.checkpoint_dir, name)
        ckpt_path = download_if_missing(ckpt_path, url)
    model = create_efficientvit_sam_model(name=f"efficientvit-sam-{arch}", pretrained=True, weight_url=ckpt_path).eval().cuda()
    return EfficientViTSamPredictor(model)

def start_gradio_ui(sam_buf, output_url: str, width: int, height: int, port: int, ui_state: dict):
    import gradio as gr, random
    host_hint, webrtc_port, stream_name = parse_webrtc_output(output_url)
    webrtc_html = build_webrtc_embed(host_hint, webrtc_port, stream_name)

    def _rows():
        return [[i, int(x), int(y), int(l)] for i, (x, y, l) in enumerate(sam_buf.get_points_labeled())]

    def _add_point(x, y, label):
        try:
            xi = 0 if x is None else int(x)
            yi = 0 if y is None else int(y)
            li = 1 if int(label) != 0 else 0
        except Exception:
            return _rows()
        xi = min(max(0, xi), width - 1)
        yi = min(max(0, yi), height - 1)
        sam_buf.add_point(xi, yi, li)
        return _rows()

    def _add_random(label=1):
        xi = random.randint(0, max(0, width - 1))
        yi = random.randint(0, max(0, height - 1))
        sam_buf.add_point(xi, yi, 1 if int(label) != 0 else 0)
        return _rows()

    def _undo():
        sam_buf.remove_last()
        return _rows()

    def _clear():
        sam_buf.clear_points()
        return _rows()

    def _on_toggle_draw(val):
        ui_state["highlight"] = bool(val)
        return f"Draw points: **{'ON' if ui_state['highlight'] else 'OFF'}**"

    with gr.Blocks(theme=gr.themes.Soft(primary_hue="lime")) as demo:
        gr.Markdown("### EfficientViT-SAM — Stream + Manual Points (FG/BG)")
        gr.HTML(webrtc_html)

        with gr.Row():
            with gr.Column():
                gr.Markdown(f"Add points within **{width}×{height}**. 1=FG(blue), 0=BG(red)")
                with gr.Row():
                    x_in = gr.Number(label="x", value=100, precision=0)
                    y_in = gr.Number(label="y", value=100, precision=0)
                with gr.Row():
                    add_fg_btn = gr.Button("Add FG (x,y,1)", variant="primary")
                    add_bg_btn = gr.Button("Add BG (x,y,0)")
                with gr.Row():
                    add_rand_fg_btn = gr.Button("Add random FG")
                    add_rand_bg_btn = gr.Button("Add random BG")
                with gr.Row():
                    clear_btn = gr.Button("Clear all", variant="stop")
                    undo_btn  = gr.Button("Undo")
                draw_toggle = gr.Checkbox(label="Draw points on stream", value=ui_state.get("highlight", False))
                draw_status = gr.Markdown(_on_toggle_draw(ui_state.get("highlight", False)))

            with gr.Column(scale=0):
                pts_tbl = gr.Dataframe(
                    headers=["idx", "x", "y", "label"],
                    datatype=["number", "number", "number", "number"],
                    value=_rows(),
                    interactive=False,
                    row_count=(0, "dynamic"),
                    col_count=4,
                    label="Active points (1=FG blue, 0=BG red)"
                )

        # wiring
        add_fg_btn.click(lambda x,y: _add_point(x,y,1), inputs=[x_in, y_in], outputs=[pts_tbl])
        add_bg_btn.click(lambda x,y: _add_point(x,y,0), inputs=[x_in, y_in], outputs=[pts_tbl])
        add_rand_fg_btn.click(lambda: _add_random(1), outputs=[pts_tbl])
        add_rand_bg_btn.click(lambda: _add_random(0), outputs=[pts_tbl])
        clear_btn.click(_clear, outputs=[pts_tbl])
        undo_btn.click(_undo, outputs=[pts_tbl])
        draw_toggle.change(_on_toggle_draw, inputs=[draw_toggle], outputs=[draw_status])

    threading.Thread(target=lambda: demo.launch(server_name="0.0.0.0", server_port=port, show_api=False),
                     daemon=True).start()

def main():
    args = parse_args()

    cli_points_labeled = parse_points(args.points)  # [(x,y,label)]
    # kept for legacy arg, but unused in labeled draw:
    _ = parse_color(args.point_color)

    videosource_dict = {"width": args.width, "height": args.height, "codec": "mjpeg", "encoder": "v4l2"}
    videooutput_dict = {"codec": args.codec, "encoder": "v4l2", "bitrate": args.bitrate}
    if args.latency is not None:
        videooutput_dict["latency"] = args.latency

    print(f"[videoSource] {args.input}  → {videosource_dict}")
    print(f"[videoOutput] {args.output} → {videooutput_dict}")

    inp = videoSource(args.input, options=videosource_dict)
    out = videoOutput(args.output, options=videooutput_dict)
    font = cudaFont()

    gate = RTSPGate(args.output, idle_timeout=args.idle_timeout) if args.render_when_viewed and is_rtsp_uri(args.output) else None

    predictor = load_sam(args)
    timer = StageTimer(enabled=args.measure, log_interval=2.0)

    seg_buffers = SegmentationBuffers(visualize=args.visualize)
    sam_buf = SAMPointBuffers(max_points=args.max_points, device="cuda")

    ui_state = {"highlight": args.highlight_points}
    if args.use_gradio:
        start_gradio_ui(sam_buf, args.output, args.width, args.height, args.gradio_port, ui_state=ui_state)

    frame_idx, t_prev, smoothed_fps = 0, time.time(), 0.0
    mask_long_t = None

    while True:
        timer.next_frame()
        with timer.span("tot"):
            with timer.span("cap"):
                img = inp.Capture(format="rgb8", timeout=1000)
            if img is None:
                continue

            h0, w0 = int(img.height), int(img.width)
            seg_buffers.alloc((h0, w0), img.format)

            # Build prompt lists (coords + labels)
            if args.use_gradio:
                pts_lab = [(x,y,l) for (x,y,l) in sam_buf.get_points_labeled()
                           if 0 <= x < w0 and 0 <= y < h0]
            else:
                pts_lab = [(x,y,l) for (x,y,l) in cli_points_labeled
                           if 0 <= x < w0 and 0 <= y < h0]

            pts_xy  = [(x,y) for (x,y,_) in pts_lab]
            labels  = [l for (_,_,l) in pts_lab]

            with timer.span("infer"):
                cudaDeviceSynchronize()
                img_t = torch.as_tensor(img, device="cuda")  # (H,W,3) u8
                # It’s safe to call set_image_cuda() outside AMP; it consumes uint8
                predictor.set_image_cuda(img_t, image_format="RGB")

                mask_long_t = None
                if pts_lab:
                    n_pts = sam_buf.stage_from_manager() if args.use_gradio else sam_buf.load_points(pts_lab)
                    if n_pts > 0:
                        pts_t, lbl_t = sam_buf.tensors(n_pts)

                        amp_ctx = autocast(dtype=torch.float16, enabled=(args.fp16 and torch.cuda.is_available()))
                        with torch.inference_mode(), amp_ctx:
                            masks, iou, _ = predictor.predict_points_cuda(pts_t, lbl_t, multimask_output=True)

                        m = masks[0, torch.argmax(iou[0]).item()]      # (H,W) bool
                        mask_long_t = m.to(torch.int64)

            with timer.span("post"):
                color_mask_t = None
                if mask_long_t is not None:
                    palette = torch.zeros((2, 3), device="cuda", dtype=torch.uint8)
                    palette[1] = torch.tensor([0, 255, 0], device="cuda", dtype=torch.uint8)  # green
                    color_mask_t = palette[mask_long_t]

                if seg_buffers.use_overlay:
                    if color_mask_t is not None and args.alpha > 0.0:
                        alpha_blend_inplace(seg_buffers.overlay, img, color_mask_t, args.alpha)
                    else:
                        torch.as_tensor(seg_buffers.overlay, device="cuda").copy_(torch.as_tensor(img, device="cuda"))

                if seg_buffers.use_mask:
                    if color_mask_t is not None:
                        torch.as_tensor(seg_buffers.mask, device="cuda").copy_(color_mask_t)
                    else:
                        torch.as_tensor(seg_buffers.mask, device="cuda").copy_(torch.as_tensor(img, device="cuda"))

                if seg_buffers.use_composite:
                    cudaOverlay(seg_buffers.overlay, seg_buffers.composite, 0, 0)
                    cudaOverlay(seg_buffers.mask,    seg_buffers.composite, seg_buffers.overlay.width, 0)

                if (ui_state["highlight"] if args.use_gradio else args.highlight_points) and pts_lab:
                    draw_points_inplace_labeled(
                        seg_buffers.output, pts_xy, labels,
                        radius=args.point_radius,
                        color_pos=(0, 0, 255, 255),    # FG = blue
                        color_neg=(255, 0, 0, 255),    # BG = red
                    )

                overlay_text(font, seg_buffers.output,
                             f"SAM(points) {len(pts_lab)} (FG/BG) | alpha={args.alpha:.2f}",
                             x=10, y=10)

            if gate:
                gate.update()
                should_render = gate.should_render()
            else:
                should_render = True

            torch.cuda.synchronize()
            cudaDeviceSynchronize()

            if should_render:
                with timer.span("rend"):
                    out.Render(seg_buffers.output)

            if gate and gate.idle_timed_out():
                print("[rtsp] idle timeout — no viewer detected, exiting.")
                break

        frame_idx += 1
        now = time.time()
        if frame_idx > 10:
            dt = now - t_prev
            smoothed_fps = 0.9 * smoothed_fps + 0.1 * (1.0 / max(1e-6, dt))
        t_prev = now

        extra = timer.status_suffix(keys=("cap", "post", "rend", "tot"))
        view_tag = " | VIEW:" + ("ON" if (gate and gate.should_render()) else "OFF") if gate else ""
        out.SetStatus(
            f"SAM(points){' FP16-AMP' if args.fp16 else ''} | ~{smoothed_fps:4.1f} FPS{extra}{view_tag}"
        )

        timer.end_frame()
        timer.log_this_interval(smoothed_fps, "sam-points")

        if not inp.IsStreaming() or not out.IsStreaming():
            break

if __name__ == "__main__":
    main()
