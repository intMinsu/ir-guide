# efficientvit/apps/utils/export.py
from __future__ import annotations
import io
import os
from typing import Any

import onnx
import torch
import torch.nn as nn

# --- make onnxsim optional ---
try:
    from onnxsim import simplify as _simplify_func
except Exception:
    _simplify_func = None

__all__ = ["export_onnx"]


def export_onnx(
    model: nn.Module,
    export_path: str,
    sample_inputs: Any,
    simplify: bool = True,
    opset: int = 11,
) -> None:
    """
    Export a model to ONNX (optionally simplify if onnxsim is available).

    Args:
        model: torch.nn.Module
        export_path: output .onnx path
        sample_inputs: example inputs (tensor or tuple/dict of tensors)
        simplify: try to run onnx-simplifier (skipped if onnxsim isn't installed)
        opset: ONNX opset version (default 11)
    """
    model.eval()

    buffer = io.BytesIO()
    with torch.no_grad():
        # export to in-memory buffer
        torch.onnx.export(model, sample_inputs, buffer, opset_version=opset)
        buffer.seek(0, 0)

        # optional simplification
        if simplify:
            if _simplify_func is None:
                print("[efficientvit] onnxsim not available; skipping simplification")
                simplify = False

        if simplify:
            onnx_model = onnx.load_model(buffer)
            try:
                onnx_model, success = _simplify_func(onnx_model)
                if not success:
                    print("[efficientvit] onnxsim reported unsuccessful simplification; keeping original graph")
                else:
                    new_buffer = io.BytesIO()
                    onnx.save(onnx_model, new_buffer)
                    buffer = new_buffer
            except Exception as e:
                print(f"[efficientvit] onnxsim failed ({e}); keeping original graph")
            finally:
                buffer.seek(0, 0)

    # write out
    if buffer.getbuffer().nbytes > 0:
        save_dir = os.path.dirname(export_path)
        os.makedirs(save_dir, exist_ok=True)
        buffer.seek(0, 0)
        with open(export_path, "wb") as f:
            f.write(buffer.read())