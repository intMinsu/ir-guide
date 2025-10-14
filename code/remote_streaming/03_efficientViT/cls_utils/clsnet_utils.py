# Classification-specific utilities
# - load_labels() with auto ImageNet-1k fallback (Keras-vis JSON)
#
# Supports multiple JSON layouts:
#   1) {"0": ["n01440764","tench"], "1": ["n01443537","goldfish"], ...}  (dict)
#   2) [["n01440764","tench"], ["n01443537","goldfish"], ...]            (list)
#   3) [{"id":"n01440764","label":"tench"}, ...]                         (list of dicts)
#
# If parsing fails, it falls back to class_000.. labels.

import json
import os
import urllib.request
from typing import List, Optional

_IMAGENET_JSON_URL = "https://raw.githubusercontent.com/raghakot/keras-vis/master/resources/imagenet_class_index.json"


def _download_imagenet1k_json(dst_json: str) -> str:
    os.makedirs(os.path.dirname(dst_json), exist_ok=True)
    print(f"[labels] downloading ImageNet-1k JSON -> {dst_json}")
    tmp = dst_json + ".part"
    urllib.request.urlretrieve(_IMAGENET_JSON_URL, tmp)
    os.replace(tmp, dst_json)
    print("[labels] Using default ImageNet-1k labels.")
    return dst_json


def _parse_imagenet_json(data) -> List[str]:
    """
    Accepts dict or list variants and returns a list[str] of human-readable labels.
    """
    labels: List[str] = []

    if isinstance(data, dict):
        # Expected: {"0": ["n01440764","tench"], ...}
        # Sort by numeric key to ensure correct order.
        try:
            n = len(data)
            for i in range(n):
                pair = data[str(i)]
                if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                    readable = str(pair[1]).replace("_", " ").strip()
                    labels.append(readable)
                else:
                    # Unexpected structure, just stringify
                    labels.append(str(pair))
        except Exception:
            # Fallback: iterate keys in numeric order anyway
            for k in sorted(data.keys(), key=lambda x: int(x) if str(x).isdigit() else x):
                v = data[k]
                if isinstance(v, (list, tuple)) and len(v) >= 2:
                    labels.append(str(v[1]).replace("_", " ").strip())
                elif isinstance(v, dict):
                    labels.append(str(v.get("label") or v.get("name") or v.get("id") or k))
                else:
                    labels.append(str(v))

    elif isinstance(data, list):
        # Could be:
        #   [["n01440764","tench"], ...]  OR  [{"id":"...","label":"..."}, ...]
        for item in data:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                labels.append(str(item[1]).replace("_", " ").strip())
            elif isinstance(item, dict):
                readable = item.get("label") or item.get("name") or item.get("synset") or item.get("id")
                labels.append(str(readable).replace("_", " ").strip() if readable is not None else "unknown")
            else:
                labels.append(str(item))
    else:
        raise ValueError(f"Unexpected JSON structure: {type(data).__name__}")

    return labels


def _read_imagenet_json(json_path: str) -> List[str]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return _parse_imagenet_json(data)


def load_labels(labels_path: Optional[str], num_classes: int = 1000) -> List[str]:
    """
    Load labels from:
      1) explicit file (.txt or .json) if provided
      2) fallback to assets/labels/imagenet1k.json (auto-download)
      3) otherwise generate class_XXX
    """
    # 1) Explicit user-provided file
    if labels_path and os.path.exists(labels_path):
        if labels_path.lower().endswith(".json"):
            try:
                labels = _read_imagenet_json(labels_path)
                # Normalize length
                if len(labels) < num_classes:
                    labels += [f"class_{i:03d}" for i in range(len(labels), num_classes)]
                return labels[:num_classes]
            except Exception as e:
                print(f"[labels] failed to read JSON '{labels_path}': {e}")
        else:
            try:
                with open(labels_path, "r", encoding="utf-8") as f:
                    labels = [ln.strip() for ln in f if ln.strip()]
                if len(labels) < num_classes:
                    labels += [f"class_{i:03d}" for i in range(len(labels), num_classes)]
                return labels[:num_classes]
            except Exception as e:
                print(f"[labels] failed to read TXT '{labels_path}': {e}")

    # 2) Default fallback (assets/labels/imagenet1k.json)
    assets_json = os.path.join(os.path.dirname(__file__), "..", "assets", "labels", "imagenet1k.json")
    assets_json = os.path.normpath(assets_json)

    if not os.path.exists(assets_json):
        _download_imagenet1k_json(assets_json)

    try:
        labels = _read_imagenet_json(assets_json)
        if len(labels) < num_classes:
            labels += [f"class_{i:03d}" for i in range(len(labels), num_classes)]
        return labels[:num_classes]
    except Exception as e:
        print(f"[labels] failed to parse default JSON: {e}")

    # 3) Final fallback
    return [f"class_{i:03d}" for i in range(num_classes)]
