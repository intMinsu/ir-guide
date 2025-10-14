# seg_utils/sam_utils.py
"""
Prompt buffers for EfficientViT-SAM (GPU-first).

- SAMBuffers: fixed-capacity (points + labels)
- BoxBuffers: fixed-capacity (x1,y1,x2,y2)
Both avoid per-frame CUDA allocs by staging pinned host -> device once per update.
Include thread-safe managers for interactive UIs (Gradio, etc.).
"""

from __future__ import annotations
from typing import Iterable, Tuple, Optional, List
import threading
import torch

# ----------------------------- Points -----------------------------
class _PointsManager:
    """Thread-safe list of (x,y,label) with capacity limit (label: 1=FG, 0=BG)."""
    def __init__(self, capacity: int):
        self._lock = threading.Lock()
        self._pts: List[Tuple[int, int, int]] = []  # (x,y,label)
        self._cap = int(capacity)

    def add(self, x: float, y: float, label: int = 1) -> List[Tuple[int, int, int]]:
        xi, yi = int(x), int(y)
        li = 1 if int(label) != 0 else 0
        with self._lock:
            if len(self._pts) < self._cap:
                self._pts.append((xi, yi, li))
            return list(self._pts)

    def clear(self) -> List[Tuple[int, int, int]]:
        with self._lock:
            self._pts.clear()
            return []

    def set_points(self, pts: Iterable[Tuple[float, float] | Tuple[float, float, int]]):
        buf: List[Tuple[int,int,int]] = []
        for p in pts:
            if len(p) >= 3:
                x, y, l = p[:3]
                buf.append((int(x), int(y), 1 if int(l) != 0 else 0))
            else:
                x, y = p[:2]
                buf.append((int(x), int(y), 1))
        with self._lock:
            self._pts = buf[: self._cap]
            return list(self._pts)

    def get(self) -> List[Tuple[int, int, int]]:
        with self._lock:
            return list(self._pts)

    def count(self) -> int:
        with self._lock:
            return len(self._pts)

    def remove_last(self):
        with self._lock:
            if self._pts:
                self._pts.pop()
            return list(self._pts)

    def remove_index(self, i: int):
        with self._lock:
            if 0 <= i < len(self._pts):
                self._pts.pop(i)
            return list(self._pts)

    def remove_nearest(self, x: float, y: float, radius: int = 10):
        xi, yi = int(x), int(y)
        with self._lock:
            if not self._pts:
                return []
            best = None
            for idx, (px, py, _) in enumerate(self._pts):
                d2 = (px - xi) * (px - xi) + (py - yi) * (py - yi)
                if best is None or d2 < best[0]:
                    best = (d2, idx)
            if best and best[0] <= radius * radius:
                self._pts.pop(best[1])
            return list(self._pts)
        

class SAMPointBuffers:
    """
    Fixed-capacity buffers for EfficientViT-SAM point prompts (points + labels).
      - device (CUDA): pts_d [max,2] float32, lbl_d [max] int64
      - host (pinned): pts_h [max,2] float32, lbl_h [max] int64
    Labels: 1=foreground (blue), 0=background (red).
    """
    def __init__(self, max_points: int = 100, device: str = "cuda"):
        self.max_points = int(max_points)
        self.device = torch.device(device)

        self.pts_d = torch.empty((self.max_points, 2), device=self.device, dtype=torch.float32)
        self.lbl_d = torch.empty((self.max_points,),   device=self.device, dtype=torch.int64)

        self.pts_h = torch.empty((self.max_points, 2), dtype=torch.float32, pin_memory=True)
        self.lbl_h = torch.empty((self.max_points,),   dtype=torch.int64,   pin_memory=True)

        self._count = 0
        self.manager = _PointsManager(capacity=self.max_points)

    # ---- stage from raw arrays ----
    def load_points(self,
                    pts_in: Iterable[Tuple[float, float] | Tuple[float, float, int]],
                    labels_in: Optional[Iterable[int]] = None) -> int:
        pts_list = list(pts_in) if not isinstance(pts_in, list) else pts_in
        n = min(len(pts_list), self.max_points)
        if n <= 0:
            self._count = 0
            return 0

        # coords
        coords = []
        labels = []
        if labels_in is None:
            for p in pts_list[:n]:
                if len(p) >= 3:
                    x, y, l = p[:3]
                    coords.append((float(x), float(y)))
                    labels.append(1 if int(l) != 0 else 0)
                else:
                    x, y = p[:2]
                    coords.append((float(x), float(y)))
                    labels.append(1)
        else:
            labels = [1 if int(l) != 0 else 0 for l in list(labels_in)[:n]]
            for p in pts_list[:n]:
                x, y = p[:2]
                coords.append((float(x), float(y)))

        self.pts_h[:n].copy_(torch.tensor(coords, dtype=torch.float32))
        self.lbl_h[:n].copy_(torch.tensor(labels, dtype=torch.int64))

        self.pts_d[:n].copy_(self.pts_h[:n], non_blocking=True)
        self.lbl_d[:n].copy_(self.lbl_h[:n], non_blocking=True)

        self._count = n
        if len(pts_list) > self.max_points:
            print(f"[sam_buffers] warning: clipped {len(pts_list)} → {n} points (capacity={self.max_points})")
        return n

    def tensors(self, n: Optional[int] = None):
        if n is None:
            n = self._count
        if n <= 0:
            raise ValueError("No points loaded - call load_points() or stage_from_manager() first.")
        return self.pts_d[:n], self.lbl_d[:n]

    @property
    def count(self) -> int:
        return self._count

    def clear(self) -> None:
        self._count = 0

    # ---- interactive helpers ----
    def add_point(self, x: float, y: float, label: int = 1) -> int:
        self.manager.add(x, y, label)
        return self.manager.count()

    def clear_points(self) -> None:
        self.manager.clear()

    def set_points(self, pts: Iterable[Tuple[float, float] | Tuple[float, float, int]]) -> int:
        self.manager.set_points(pts)
        return self.manager.count()

    def get_points_labeled(self) -> list[Tuple[int, int, int]]:
        return self.manager.get()

    def get_points(self) -> list[Tuple[int, int]]:
        # coords only (legacy)
        return [(x, y) for (x, y, _) in self.manager.get()]

    def stage_from_manager(self) -> int:
        return self.load_points(self.manager.get())

    def remove_last(self): 
        self.manager.remove_last()

    def remove_index(self, i: int): 
        self.manager.remove_index(i)

    def remove_nearest(self, x: float, y: float, radius: int = 10): 
        self.manager.remove_nearest(x, y, radius)

# ----------------------------- Boxes -----------------------------
class _BoxManager:
    """Thread-safe list of (x1,y1,x2,y2) boxes for interactive UIs."""
    def __init__(self, capacity: int):
        self._lock = threading.Lock()
        self._cap = int(capacity)
        self._boxes: List[Tuple[int, int, int, int]] = []

    def add(self, x1: float, y1: float, x2: float, y2: float):
        b = (int(x1), int(y1), int(x2), int(y2))
        if b[2] < b[0]: b = (b[2], b[1], b[0], b[3])
        if b[3] < b[1]: b = (b[0], b[3], b[2], b[1])
        with self._lock:
            if len(self._boxes) < self._cap:
                self._boxes.append(b)
            return list(self._boxes)

    def clear(self):
        with self._lock:
            self._boxes.clear()
            return []

    def remove_last(self):
        with self._lock:
            if self._boxes:
                self._boxes.pop()
            return list(self._boxes)

    def get(self) -> List[Tuple[int, int, int, int]]:
        with self._lock:
            return list(self._boxes)


class SAMBoxBuffers:
    """
    Fixed-capacity CUDA storage for box prompts (N,4) float32 and an interactive manager.
    """
    def __init__(self, max_boxes: int = 100, device: str = "cuda"):
        self.max_boxes = int(max_boxes)
        self.device = torch.device(device)
        self.boxes_d = torch.empty((self.max_boxes, 4), device=self.device, dtype=torch.float32)
        self.boxes_h = torch.empty((self.max_boxes, 4), dtype=torch.float32, pin_memory=True)
        self._count = 0
        self.manager = _BoxManager(capacity=self.max_boxes)

    def load_boxes(self, boxes_in: Iterable[Tuple[float, float, float, float]]) -> int:
        arr = list(boxes_in) if not isinstance(boxes_in, list) else boxes_in
        n = min(len(arr), self.max_boxes)
        if n <= 0:
            self._count = 0
            return 0
        self.boxes_h[:n].copy_(torch.tensor(arr[:n], dtype=torch.float32))
        self.boxes_d[:n].copy_(self.boxes_h[:n], non_blocking=True)
        self._count = n
        if len(arr) > self.max_boxes:
            print(f"[box_buffers] warning: clipped {len(arr)} → {n} boxes (capacity={self.max_boxes})")
        return n

    def tensors(self, n: Optional[int] = None):
        if n is None:
            n = self._count
        if n <= 0:
            raise ValueError("No boxes loaded")
        return self.boxes_d[:n]

    # Interactive helpers
    def add_box(self, x1, y1, x2, y2):
        return self.manager.add(x1, y1, x2, y2)

    def clear_boxes(self):
        self.manager.clear()

    def remove_last(self):
        self.manager.remove_last()

    def get_boxes(self) -> List[Tuple[int, int, int, int]]:
        return self.manager.get()

    def stage_from_manager(self) -> int:
        return self.load_boxes(self.manager.get())
