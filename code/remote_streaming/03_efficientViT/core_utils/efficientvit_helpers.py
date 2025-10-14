# Common helpers shared by EfficientViT demos (no model/task-specific code).
# - ensure_dir(), is_jetson()
# - download_if_missing()
# - cuda_rgb8_to_numpy() zero-copy view
# - preprocess_cuda_rgb8_to_tensor() with cached resize buffers + optional GPU normalization

import os
import time
import urllib.request
from typing import Tuple, Optional

import numpy as np

try:
    import torch
except Exception:
    torch = None  # allow import on machines without torch (e.g., export-only envs)

from jetson_utils import (
    cudaAllocMapped,
    cudaResize,
    cudaToNumpy,
    cudaDeviceSynchronize,
)

# -------------------------
# Filesystem / platform
# -------------------------
def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)

def is_jetson() -> bool:
    try:
        if os.path.isfile("/etc/nv_tegra_release"):
            return True
        model_path = "/proc/device-tree/model"
        if os.path.exists(model_path):
            return b"tegra" in open(model_path, "rb").read().lower()
    except Exception:
        pass
    return False

# -------------------------
# Downloads
# -------------------------
def _progress_hook(prefix: str):
    start = time.time()
    def _hook(blocks, block_size, total_size):
        downloaded = blocks * block_size
        if total_size > 0:
            pct = min(100.0, downloaded * 100.0 / total_size)
            elapsed = max(1e-6, time.time() - start)
            rate = downloaded / elapsed / 1e6
            print(f"\r[{prefix}] {pct:5.1f}%  {downloaded/1e6:7.2f}/{total_size/1e6:7.2f} MB ({rate:5.2f} MB/s)", end="")
    return _hook

def download_if_missing(dst_path: str, url: str) -> str:
    """Download url → dst_path if file doesn't exist (atomic temp -> rename)."""
    ensure_dir(os.path.dirname(dst_path))
    if os.path.exists(dst_path) and os.path.getsize(dst_path) > 0:
        print(f"[checkpoint] found: {dst_path}")
        return dst_path
    tmp = dst_path + ".part"
    print(f"[checkpoint] downloading -> {dst_path}")
    urllib.request.urlretrieve(url, tmp, _progress_hook("download"))
    print()
    os.replace(tmp, dst_path)
    return dst_path

# -------------------------
# CUDA image helpers
# -------------------------
def cuda_rgb8_to_numpy(img, copy: bool = False) -> np.ndarray:
    """
    Get an HWC RGB uint8 NumPy view over a jetson_utils CUDA image
    that was allocated with mapped (pinned) memory.
    Set copy=True to make a distinct CPU copy.
    """
    arr = cudaToNumpy(img)  # zero-copy view into mapped host memory
    return arr.copy() if copy else arr

# -------------------------
# Preprocess: CUDA → Torch (CPU/GPU normalization, zero-copy, cached buffers)
# -------------------------
_RESIZE_CACHE = {}          # key: (format, H, W) -> mapped buffer
_MEANSTD_GPU = {}           # key: (device, dtype) -> (mean_t, std_t)
_MEANSTD_CPU = (
    np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3,1,1),
    np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3,1,1),
)

def clear_preprocess_cache():
    """Free cached mapped buffers and GPU tensors (call on shutdown or size change)."""
    for buf in list(_RESIZE_CACHE.values()):
        try: del buf
        except Exception: pass
    _RESIZE_CACHE.clear()
    _MEANSTD_GPU.clear()

def preprocess_cuda_rgb8_to_tensor(
    img_cuda,
    out_hw: Tuple[int, int],
    device: str,
    *,
    normalize_on: str = "gpu",          # "cpu" | "gpu"
    dtype = None,                        # None -> torch.float32
    reuse_resize_buf: bool = True,      # reuse mapped buffer to avoid RAM growth
):
    """
    Resize a CUDA image on-GPU to (H,W), then return Torch tensor (1,3,H,W).

    Zero-copy path:
      • `cudaAllocMapped()` creates page-locked host memory mapped into GPU.
      • `cudaResize()` writes the resized frame on GPU into that mapped buffer.
      • `cudaToNumpy()` returns a NumPy view over the same mapped buffer (no copy).
      • `torch.from_numpy()` wraps that view as a CPU tensor (still no copy).
      • Only `.to('cuda')` actually uploads data to GPU.

    Args:
        img_cuda: jetson_utils CUDA image (expects format='rgb8').
        out_hw:   (H, W) output size.
        device:   "cuda" or "cpu".
        normalize_on: "cpu" or "gpu" (forced to "cpu" when device="cpu").
        dtype:    output dtype (default torch.float32; torch.float16 supported).
        reuse_resize_buf: cache mapped resize buffers per (format,H,W).

    Returns:
        Tensor on `device` (1,3,H,W) with requested dtype.
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for preprocessing tensors.")

    H, W = out_hw
    fmt = getattr(img_cuda, "format", "rgb8")
    if dtype is None:
        dtype = torch.float32

    # 1) Resize on GPU into cached mapped buffer
    if reuse_resize_buf:
        key = (fmt, H, W)
        resized = _RESIZE_CACHE.get(key)
        if resized is None:
            resized = cudaAllocMapped(width=W, height=H, format=fmt)
            _RESIZE_CACHE[key] = resized
    else:
        resized = cudaAllocMapped(width=W, height=H, format=fmt)

    cudaResize(img_cuda, resized)
    cudaDeviceSynchronize()

    # 2) Zero-copy CPU view (H,W,3) uint8
    np_img = cudaToNumpy(resized)

    # If output device is CPU, normalize on CPU.
    if device == "cpu":
        normalize_on = "cpu"

    if normalize_on.lower() == "gpu":
        # GPU path: upload uint8, then CHW/scale/normalize on GPU
        t_cpu = torch.from_numpy(np_img)                      # CPU uint8 view
        t = t_cpu.to(device, non_blocking=True)               # H2D copy (uint8)
        del t_cpu, np_img                                     # drop CPU views ASAP

        t = t.permute(2, 0, 1).contiguous()                   # CHW on GPU

        if dtype in (torch.float16, torch.float32):
            t = t.to(dtype).div_(255.0)
            ms_key = (device, dtype)
            mean_t, std_t = _MEANSTD_GPU.get(ms_key, (None, None))
            if mean_t is None:
                mean_t = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=dtype).view(3,1,1)
                std_t  = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=dtype).view(3,1,1)
                _MEANSTD_GPU[ms_key] = (mean_t, std_t)
            t = (t - mean_t) / std_t
        else:
            t = t.to(dtype)

        return t.unsqueeze(0)

    else:
        # CPU path
        chw = np_img.astype(np.float32).transpose(2, 0, 1) / 255.0
        del np_img
        mean, std = _MEANSTD_CPU
        chw = (chw - mean) / std
        t = torch.from_numpy(chw).unsqueeze(0)
        t = t.to(device, non_blocking=True)
        if dtype is not torch.float32:
            t = t.to(dtype)
        return t
