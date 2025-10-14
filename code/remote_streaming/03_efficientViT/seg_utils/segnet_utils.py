# segnet_utils.py (GPU-only)
# Utilities for segmentation overlays using mapped CUDA buffers.
# - SegmentationBuffers: pre-allocate/reuse mapped CUDA buffers (overlay/mask/composite)
# - generate_palette(): deterministic GPU palette (Cx3 uint8 CUDA)
# - colorize_mask(): GPU colorization (mask HxW long CUDA -> HxWx3 uint8 CUDA)
# - alpha_blend_inplace(): GPU alpha blend of color mask over frame into a CUDA buffer
# - draw_points_inplace(): GPU points via cudaDrawCircle
# - draw_boxes_inplace(): GPU rectangles/lines via cudaDrawRect/cudaDrawLine
# Notes:
# * No NumPy, no cudaToNumpy — everything stays on the GPU.
# * cudaImage <-> torch.Tensor is zero-copy via __cuda_array_interface__.
# * All functions expect CUDA tensors / cudaImage buffers.

from typing import Iterable, Tuple

import torch
from jetson_utils import (
    cudaAllocMapped,
    cudaOverlay,
    cudaDrawCircle,  # GPU drawing primitive
    cudaDrawLine,
    cudaDrawRect,
)

class SegmentationBuffers:
    """
    Manage mapped CUDA buffers for overlay/mask/composite, reusing allocations
    when the image size is unchanged. All buffers are cudaImage objects.
    """
    def __init__(self, visualize="overlay"):
        self.overlay = None
        self.mask = None
        self.composite = None

        self.use_overlay = "overlay" in visualize
        self.use_mask = "mask" in visualize
        self.use_composite = self.use_overlay and self.use_mask

        if not self.use_overlay and not self.use_mask:
            raise ValueError("visualize must include 'overlay' and/or 'mask'")

    @property
    def output(self):
        if self.use_overlay and self.use_mask:
            return self.composite
        elif self.use_overlay:
            return self.overlay
        else:
            return self.mask

    def alloc(self, shape_hw: Tuple[int, int], fmt: str):
        h, w = int(shape_hw[0]), int(shape_hw[1])
        if self.overlay is not None and self.overlay.height == h and self.overlay.width == w:
            return  # already matching

        if self.use_overlay:
            self.overlay = cudaAllocMapped(width=w, height=h, format=fmt)

        if self.use_mask:
            self.mask = cudaAllocMapped(width=w, height=h, format=fmt)

        if self.use_composite:
            self.composite = cudaAllocMapped(width=self.overlay.width + self.mask.width,
                                             height=self.overlay.height, format=fmt)

def _to_cuda_tensor(img_or_tensor):
    """
    Convert a jetson_utils cudaImage or an existing tensor into a CUDA uint8 tensor (H,W,3).
    This is zero-copy for cudaImage via __cuda_array_interface__.
    """
    if isinstance(img_or_tensor, torch.Tensor):
        t = img_or_tensor
        if t.device.type != "cuda":
            raise ValueError("expected CUDA tensor")
        if t.dtype != torch.uint8:
            t = t.to(torch.uint8)
        return t
    # cudaImage -> torch tensor (zero-copy)
    return torch.as_tensor(img_or_tensor, device='cuda')

def generate_palette(num_classes: int) -> torch.Tensor:
    """
    Deterministic GPU palette (Cx3) uint8 using a VOC-style bit pattern.
    Entirely on CUDA, no CPU work or NumPy.

    Returns
    -------
    torch.Tensor (uint8, CUDA): shape (C,3)
    """
    C = int(num_classes)
    if C <= 0:
        raise ValueError("num_classes must be > 0")

    # VOC-like palette generation in vectorized form on GPU
    # See: bitwise pattern spreading bits of class index across RGB channels.
    idx = torch.arange(C, device='cuda', dtype=torch.int64)  # (C,)
    palette = torch.zeros((C, 3), device='cuda', dtype=torch.uint8)

    # Work on a copy so shifting doesn't mutate idx we still need
    c = idx.clone()
    for j in range(8):
        # Extract 3 bits for RGB channels
        r_bit = (c >> 0) & 1
        g_bit = (c >> 1) & 1
        b_bit = (c >> 2) & 1

        # Set bit (7-j) in each channel
        shift = torch.tensor(7 - j, device='cuda', dtype=torch.int64)
        palette[:, 0] |= (r_bit.to(torch.uint8) << shift)  # R
        palette[:, 1] |= (g_bit.to(torch.uint8) << shift)  # G
        palette[:, 2] |= (b_bit.to(torch.uint8) << shift)  # B

        c = c >> 3

    # Optional: force background to black
    if C > 0:
        palette[0] = torch.tensor([0, 0, 0], device='cuda', dtype=torch.uint8)

    return palette

def colorize_mask(mask_hw_long_cuda: torch.Tensor,
                  palette_c3_u8_cuda: torch.Tensor) -> torch.Tensor:
    """
    GPU colorization: palette index -> RGB color.

    Parameters
    ----------
    mask_hw_long_cuda : torch.LongTensor (CUDA), shape (H,W)
        Each element is a class index in [0, C-1].
    palette_c3_u8_cuda : torch.ByteTensor (CUDA), shape (C,3)

    Returns
    -------
    torch.ByteTensor (CUDA), shape (H,W,3)
    """
    if mask_hw_long_cuda.device.type != "cuda" or palette_c3_u8_cuda.device.type != "cuda":
        raise ValueError("mask and palette must be CUDA tensors")
    if mask_hw_long_cuda.dtype != torch.int64:
        raise ValueError("mask must be torch.int64 (long)")
    if palette_c3_u8_cuda.dtype != torch.uint8 or palette_c3_u8_cuda.dim() != 2 or palette_c3_u8_cuda.size(1) != 3:
        raise ValueError("palette must be (C,3) uint8 CUDA")

    # palette[mask] -> (H,W,3) uint8 on GPU
    return palette_c3_u8_cuda[mask_hw_long_cuda]

def alpha_blend_inplace(dst_img, src_img, color_mask_t: torch.Tensor, alpha: float) -> None:
    """
    In-place alpha blend onto dst_img (cudaImage) on the GPU:
      dst = src*(1-a) + color_mask*a
    When alpha==0, still copies src -> dst so the frame updates every time.
    All ops are GPU-only, zero-copy via __cuda_array_interface__.
    """
    # map cudaImage -> CUDA tensors (zero-copy)
    dst_t = torch.as_tensor(dst_img, device='cuda')      # (H,W,3) uint8
    src_t = torch.as_tensor(src_img, device='cuda')      # (H,W,3) uint8

    # always copy the live frame into dst first (critical when alpha==0)
    dst_t.copy_(src_t)

    a = float(max(0.0, min(1.0, alpha)))
    if a <= 0.0:
        return

    # blend into dst (compute into temp then copy back — still all on GPU)
    blended = (
        src_t.to(torch.float32).mul(1.0 - a)
        .add_(color_mask_t.to(torch.float32), alpha=a)
        .clamp_(0, 255).to(torch.uint8)
    )
    dst_t.copy_(blended)
    
def compose_side_by_side(left_img, right_img, out_img, x_offset_right: int = None):
    """
    GPU composition with cudaOverlay (side-by-side).

    Parameters
    ----------
    left_img : cudaImage
    right_img : cudaImage
    out_img : cudaImage
    x_offset_right : int (optional)
        X position where the right image starts in the output. If None,
        uses left_img.width (but caller should typically pass that).
    """
    xo = left_img.width if x_offset_right is None else int(x_offset_right)
    cudaOverlay(left_img,  out_img, 0, 0)
    cudaOverlay(right_img, out_img, xo, 0)

def draw_points_inplace(img, pts: Iterable[Tuple[int, int]], radius: int = 4, color=(0, 255, 0, 255)):
    """
    Draw small filled circles at (x,y) positions into a cudaImage using GPU.

    Parameters
    ----------
    img : cudaImage
    pts : iterable of (x, y)
    radius : int
    color : tuple(int,int,int,int) RGBA
    """
    r = max(1, int(radius))
    rgba = tuple(int(c) for c in color)
    for (x, y) in pts:
        cudaDrawCircle(img, (int(x), int(y)), r, rgba)

def draw_points_inplace_labeled(img,
                                pts: Iterable[Tuple[int,int]],
                                labels: Iterable[int],
                                radius: int = 4,
                                color_pos=(0, 0, 255, 255),  # blue = foreground (1)
                                color_neg=(255, 0, 0, 255)   # red  = background (0)
                                ):
    r = max(1, int(radius))
    rgba_p = tuple(int(c) for c in color_pos)
    rgba_n = tuple(int(c) for c in color_neg)
    for (x, y), l in zip(pts, labels):
        rgba = rgba_p if int(l) != 0 else rgba_n
        cudaDrawCircle(img, (int(x), int(y)), r, rgba)

def draw_boxes_inplace(img, boxes: Iterable[Tuple[int,int,int,int]],
                       color=(0, 200, 255, 255), thickness=2, clamp=True):
    rgba = tuple(int(c) for c in color)
    H, W = int(img.height), int(img.width)
    t = max(1, int(thickness))
    for b in boxes:
        x1, y1, x2, y2 = map(int, b)
        if x1 > x2: x1, x2 = x2, x1
        if y1 > y2: y1, y2 = y2, y1
        if clamp:
            x1 = max(0, min(W - 1, x1)); x2 = max(0, min(W - 1, x2))
            y1 = max(0, min(H - 1, y1)); y2 = max(0, min(H - 1, y2))
        if t == 1:
            cudaDrawRect(img, (x1, y1, x2, y2), rgba)
        else:
            cudaDrawLine(img, (x1, y1), (x2, y1), rgba, t)
            cudaDrawLine(img, (x2, y1), (x2, y2), rgba, t)
            cudaDrawLine(img, (x2, y2), (x1, y2), rgba, t)
            cudaDrawLine(img, (x1, y2), (x1, y1), rgba, t)