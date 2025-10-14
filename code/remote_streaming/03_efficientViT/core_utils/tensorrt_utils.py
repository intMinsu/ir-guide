# TensorRT utilities isolated for reuse:
# - export_onnx()
# - build_trt_engine_with_trtexec()

import os
import shutil
import subprocess
from typing import Optional, List, Tuple

import numpy as np

try:
    import torch
except Exception:
    torch = None

from .efficientvit_helpers import ensure_dir, is_jetson

# -------------------------
# ONNX export
# -------------------------
def export_onnx(model, onnx_path: str, input_hw: Tuple[int, int] = (224, 224), opset: int = 13) -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required for ONNX export.")
    model.eval()
    dummy = torch.randn(1, 3, input_hw[0], input_hw[1], device="cpu")
    print(f"[export] ONNX -> {onnx_path} (opset {opset})")
    ensure_dir(os.path.dirname(onnx_path))
    torch.onnx.export(
        model.cpu(),
        dummy,
        onnx_path,
        input_names=["input_0"],
        output_names=["logits"],
        opset_version=opset,
        do_constant_folding=True,
        dynamic_axes=None,  # static shapes
    )
    print("[export] done")

# -------------------------
# TensorRT build (Jetson-friendly)
# -------------------------
def _find_trtexec() -> str:
    default = "/usr/src/tensorrt/bin/trtexec"
    if os.path.exists(default):
        return default
    return "trtexec"

def build_trt_engine_with_trtexec(
    onnx_path: str,
    engine_path: str,
    fp16: bool = True,
    extra: Optional[List[str]] = None,
):
    trtexec = _find_trtexec()
    if not shutil.which(trtexec) and not os.path.exists(trtexec):
        raise FileNotFoundError(f"trtexec not found at '{trtexec}' (is TensorRT installed?)")

    ensure_dir(os.path.dirname(engine_path))
    timing_cache = os.path.splitext(engine_path)[0] + ".timing"

    cmd = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--memPoolSize=workspace:512MiB",
        "--minTiming=1",
        "--avgTiming=1",
        "--heuristic",
        f"--timingCacheFile={timing_cache}",
        "--buildOnly",
    ]
    if fp16:
        cmd.append("--fp16")

    if is_jetson() and fp16:
        cmd += ["--inputIOFormats=fp16:chw", "--outputIOFormats=fp16:chw"]

    if extra:
        cmd += list(extra)

    print("[trtexec] " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"[trtexec] engine saved -> {engine_path}")