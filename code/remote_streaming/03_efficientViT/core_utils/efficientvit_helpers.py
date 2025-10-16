# Common helpers shared by EfficientViT demos (no model/task-specific code).
# - ensure_dir(), is_jetson()
# - download_if_missing()

import os
import time
import urllib.request
from typing import Tuple, Optional


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