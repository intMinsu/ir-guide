from typing import Tuple, Optional, Literal
import numpy as np

import torch

# jetson-utils
from jetson_utils import (
    cudaAllocMapped,
    cudaMalloc,
    cudaResize,
    cudaToNumpy,
    cudaDeviceSynchronize,
)

"""
GPU-friendly preprocessing for jetson_utils.cudaImage → Torch tensor.

Entrypoint:
    preprocess_cuda_rgb8_to_tensor(img_cuda, out_hw, device="cuda", ...)

Two zero-copy–aware paths are selected **at runtime** (or forced by `resize_dst`):

1) **Mapped-staging** (host-mapped pinned memory, ZeroCopy)
   Triggered when `img_cuda.mapped == True` or `resize_dst="mapped"`.
   `cudaResize()` → mapped pinned host buffer (from `cudaAllocMapped`) → zero-copy NumPy view
   → a single async H2D upload → CHW/dtype/normalize on GPU.

2) **Pure-device** (true device memory)
   Triggered when `img_cuda.mapped == False` or `resize_dst="device"`.
   `cudaResize()` → device buffer (from `cudaMalloc`) → wrap with `torch.as_tensor(..., 'cuda')`
   → **Safer single-copy pattern**: after `permute`, use `to(dtype, copy=True, memory_format=torch.contiguous_format)`
   to both **break aliasing** and **materialize contiguous CHW** in one pass.

Normalization presets (`norm`) supported:
- "imagenet" (default): mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]
- "clip":     mean=[0.48145466,0.4578275,0.40821073], std=[0.26862954,0.26130258,0.27577711]
- "unit":     mean=[0,0,0], std=[1,1,1]              (i.e., 0–1 scaling only)
- "zero_centered": mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5] (maps 0–1 → [-1,1])
- "none":     **no normalization** and **no 1/255 scaling**. Only resize and return raw pixel values:
              if dtype is float → values in [0,255]; if dtype is uint8 → [0,255].

Note: ImageNet stats are common for classification/backbones, but not universal.
Models like CLIP or some segmentation/detection pipelines expect different stats.
Pick the `norm` preset that matches your model’s training.
"""

# ----------------------------- module-level caches -----------------------------

# Resize-buffer caches (avoid heap churn). Separate caches for mapped/device.
#   keys: (format: str, H: int, W: int)
_RESIZE_CACHE_MAPPED = {}
_RESIZE_CACHE_DEVICE = {}

# Mean/Std caches for GPU normalization: key = (device: str, dtype: torch.dtype, norm: str)
_MEANSTD_GPU = {}

# Normalization presets (CPU layout = CHW Numpy)
_NORM_REGISTRY_CPU = {
    "imagenet": (
        np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None],
        np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None],
    ),
    "clip": (
        np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)[:, None, None],
        np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)[:, None, None],
    ),
    "unit": (
        np.array([0.0, 0.0, 0.0], dtype=np.float32)[:, None, None],
        np.array([1.0, 1.0, 1.0], dtype=np.float32)[:, None, None],
    ),
    "zero_centered": (
        np.array([0.5, 0.5, 0.5], dtype=np.float32)[:, None, None],
        np.array([0.5, 0.5, 0.5], dtype=np.float32)[:, None, None],
    ),
}


def _get_or_alloc_resize_buf(
    dst: Literal["mapped", "device"],
    fmt: str,
    H: int,
    W: int,
    reuse: bool,
):
    """Return a cached or freshly-allocated cudaImage for resize output."""
    if dst == "mapped":
        cache = _RESIZE_CACHE_MAPPED
        alloc_fn = cudaAllocMapped
    else:
        cache = _RESIZE_CACHE_DEVICE
        alloc_fn = cudaMalloc

    key = (fmt, H, W)
    if reuse:
        img = cache.get(key)
        if img is None:
            img = alloc_fn(width=W, height=H, format=fmt)
            cache[key] = img
        return img
    else:
        return alloc_fn(width=W, height=H, format=fmt)


def _get_meanstd_gpu(device: str, dtype: "torch.dtype", norm: str):
    """
    Get (mean, std) tensors for the given `norm` preset on the specified device/dtype with caching.

    Presets:
      "imagenet" (default), "clip", "unit", "zero_centered"
    """
    norm = norm.lower()
    if norm not in _NORM_REGISTRY_CPU:
        raise ValueError(f"Unknown norm preset '{norm}'. Choose from {list(_NORM_REGISTRY_CPU.keys())}.")

    key = (device, dtype, norm)
    mean_t, std_t = _MEANSTD_GPU.get(key, (None, None))
    if mean_t is None:
        mean_cpu, std_cpu = _NORM_REGISTRY_CPU[norm]
        mean_t = torch.tensor(mean_cpu.squeeze(), device=device, dtype=dtype).view(3, 1, 1)
        std_t = torch.tensor(std_cpu.squeeze(), device=device, dtype=dtype).view(3, 1, 1)
        _MEANSTD_GPU[key] = (mean_t, std_t)
    return mean_t, std_t


def _get_meanstd_cpu(norm: str):
    """Return (mean,std) NumPy arrays (CHW) for the given preset."""
    norm = norm.lower()
    if norm not in _NORM_REGISTRY_CPU:
        raise ValueError(f"Unknown norm preset '{norm}'. Choose from {list(_NORM_REGISTRY_CPU.keys())}.")
    return _NORM_REGISTRY_CPU[norm]


def preprocess_cuda_rgb8_to_tensor(
    img_cuda,
    out_hw: Tuple[int, int],
    device: str = "cuda",
    *,
    normalize_on: Literal["gpu", "cpu", "none"] = "gpu",
    dtype: Optional["torch.dtype"] = None,  # None -> torch.float32
    reuse_resize_buf: bool = True,
    resize_dst: Literal["auto", "mapped", "device"] = "auto",
    norm: Literal["imagenet", "clip", "unit", "zero_centered"] = "imagenet",
):
    """
    Resize a jetson_utils `cudaImage` (format='rgb8') and return a Torch tensor `(1,3,H,W)`.

    The function **auto-selects** a safe & fast path based on `img_cuda.mapped`
    (or you can override with `resize_dst`), and applies the requested normalization preset.

    Parameters
    ----------
    img_cuda : jetson_utils.cudaImage
        Source CUDA image. Expected `format='rgb8'`. `img_cuda.mapped` indicates mapped pinned host.
    out_hw : (int, int)
        `(H, W)` of the resized output.
    device : {"cuda","cpu"}, default "cuda"
        Destination device for the returned tensor.
    normalize_on : {"gpu","cpu","none"}, default "gpu"
        - "gpu"/"cpu": perform CHW + convert + (1/255) + mean/std on that side.
        - "none": **no normalization and no 1/255 scaling** — only resize & return raw values.
                  If `dtype` is float, values will be in **[0,255]**; if `uint8`, still **[0,255]**.
        If `device="cpu"` and normalize_on=="gpu", it is forced to "cpu".
    dtype : torch.dtype or None, default None
        Output dtype. Typical choices: `torch.float16`, `torch.float32` (default).
    reuse_resize_buf : bool, default True
        Reuse internal resize buffers keyed by `(format,H,W)` to reduce alloc churn.
    resize_dst : {"auto","mapped","device"}, default "auto"
        - "auto"   : choose "mapped" if `img_cuda.mapped == True`, else "device"
        - "mapped" : force mapped-staging path (`cudaAllocMapped`)
        - "device" : force pure device path (`cudaMalloc`)
    norm : {"imagenet","clip","unit","zero_centered"}, default "imagenet"
        Normalization preset. Ignored if `normalize_on="none"`.

    Returns
    -------
    torch.Tensor
        Tensor on `device` with shape `(1, 3, H, W)` and requested `dtype`.

    Implementation notes
    --------------------
    • **Safer single-copy pattern** is used where applicable:
        after `permute`, use `to(dtype, copy=True, memory_format=torch.contiguous_format)`
        to both **break aliasing** and ensure **contiguous CHW** in one pass.
    • On Jetson, unified DRAM doesn’t make mapped host memory == device memory for kernel perf.
      Use mapped for **staging/I/O**, and device memory for **hot compute**.
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for preprocessing tensors.")

    if device not in ("cuda", "cpu"):
        raise ValueError(f"Unsupported device '{device}'. Use 'cuda' or 'cpu'.")

    H, W = out_hw
    fmt = getattr(img_cuda, "format", "rgb8")
    if fmt != "rgb8":
        raise ValueError(f"Expected img_cuda.format == 'rgb8', got '{fmt}'")

    if dtype is None:
        dtype = torch.float32

    # If destination is CPU and caller asked for GPU norm, force CPU norm.
    if device == "cpu" and normalize_on == "gpu":
        normalize_on = "cpu"

    # Decide the resize destination kind based on actual memory we received.
    src_is_mapped = bool(getattr(img_cuda, "mapped", False))
    if resize_dst == "auto":
        dst_kind: Literal["mapped", "device"] = "mapped" if src_is_mapped else "device"
    else:
        dst_kind = "mapped" if resize_dst == "mapped" else "device"

    # For debugging!
    # print(f"[preprocess] img_cuda.mapped={src_is_mapped}, resize_dst='{resize_dst}' → dst_kind='{dst_kind}'")

    # ------------------------------ resize on GPU ------------------------------
    resized = _get_or_alloc_resize_buf(dst_kind, fmt, H, W, reuse_resize_buf)
    cudaResize(img_cuda, resized)
    cudaDeviceSynchronize()  # replace with stream/event sync in stream-aware code

    # -------------------------- branch by memory kind --------------------------
    if dst_kind == "mapped":
        # ----- MAPPED-STAGING PATH -----
        np_img = cudaToNumpy(resized)  # (H, W, 3) uint8, zero-copy CPU view

        if normalize_on == "gpu":
            # H2D upload (async if pinned) → HWC→CHW → single-copy cast+contiguous → normalize
            t_cpu = torch.from_numpy(np_img)  # CPU uint8 view (no copy)
            t = t_cpu.to(device, non_blocking=True)  # H2D copy
            del t_cpu, np_img

            t = t.permute(2, 0, 1)  # CHW view (non-contiguous)

            if dtype in (torch.float16, torch.float32):
                # Single pass: decouple/contiguous + cast
                t = t.to(dtype, copy=True, memory_format=torch.contiguous_format).div_(255.0)
                mean_t, std_t = _get_meanstd_gpu(device, dtype, norm)
                t = (t - mean_t) / std_t
            else:
                # Staying uint8 — still need to materialize contiguous & decouple
                t = t.contiguous().to(dtype)

            return t.unsqueeze(0)

        elif normalize_on == "cpu":
            # CPU normalization
            chw = np_img.astype(np.float32).transpose(2, 0, 1) / 255.0
            del np_img
            mean, std = _get_meanstd_cpu(norm)
            chw = (chw - mean) / std
            t = torch.from_numpy(chw).unsqueeze(0)  # (1,3,H,W) on CPU
            if device != "cpu":
                t = t.to(device, non_blocking=True)
            if dtype is not torch.float32:
                t = t.to(dtype)
            return t

        else:  # normalize_on == "none"
            # Only resize → CHW → cast/contiguous; no 1/255 and no mean/std
            t_cpu = torch.from_numpy(np_img)  # HWC uint8 on CPU
            if device != "cpu":
                t = t_cpu.to(device, non_blocking=True)
                del t_cpu
            else:
                t = t_cpu
            # CHW view then materialize contiguous in requested dtype
            t = t.permute(2, 0, 1)
            if dtype in (torch.float16, torch.float32):
                t = t.to(dtype, copy=True, memory_format=torch.contiguous_format)
            else:
                t = t.contiguous().to(dtype)
            return t.unsqueeze(0)

    else:
        # ----- PURE-DEVICE PATH -----
        t = torch.as_tensor(resized, device="cuda")  # (H, W, 3) uint8 on GPU (aliases producer)

        if normalize_on == "cpu":
            # Not recommended (D2H), retained for parity
            t_cpu = t.to("cpu")  # D2H
            t_cpu = t_cpu.permute(2, 0, 1).contiguous().to(torch.float32).div_(255.0)
            mean, std = _get_meanstd_cpu(norm)
            chw = (t_cpu.numpy() - mean) / std
            t = torch.from_numpy(chw).to(device, non_blocking=True)
            if dtype is not torch.float32:
                t = t.to(dtype)
            return t.unsqueeze(0)

        elif normalize_on == "gpu":
            # GPU path: Safer single-copy pattern after permute
            t = t.permute(2, 0, 1)  # CHW view (non-contiguous, still aliasing)

            if dtype in (torch.float16, torch.float32):
                # One copy does: decouple from resize buffer + make contiguous + cast
                t = t.to(dtype, copy=True, memory_format=torch.contiguous_format).div_(255.0)
                mean_t, std_t = _get_meanstd_gpu(device, dtype, norm)
                t = (t - mean_t) / std_t
            else:
                # Staying uint8 — explicitly decouple & make contiguous
                t = t.contiguous().to(dtype)

            return t.unsqueeze(0)

        else:  # normalize_on == "none"
            # Only resize → CHW → cast/contiguous; no 1/255 and no mean/std
            t = t.permute(2, 0, 1)
            if dtype in (torch.float16, torch.float32):
                t = t.to(dtype, copy=True, memory_format=torch.contiguous_format)
            else:
                t = t.contiguous().to(dtype)
            return t.unsqueeze(0)
