#!/usr/bin/env python3
# copy_patch_efficientvit.py
"""
Copy patched EfficientViT files into an installed package.

Sources are expected under:
  <this_script_dir>/efficientvit/...

Patched targets:
  • efficientvit/apps/utils/export.py         (onnxsim optional)
  • efficientvit/models/nn/norm.py            (triton optional)
  • efficientvit/models/nn/triton_rms_norm.py (triton optional)
  • efficientvit/models/efficientvit/sam.py   (CUDA predict_* helpers)

Usage:
  python copy_patch_efficientvit.py --root /path/to/site-packages
  python copy_patch_efficientvit.py --root /path/to/site-packages/efficientvit
  python copy_patch_efficientvit.py --root ... --backup
  python copy_patch_efficientvit.py --root ... --dry-run
"""

import argparse
import hashlib
import os
import shutil
import sys
from pathlib import Path
from typing import Tuple

PATCH_RELATIVE_FILES = [
    ("apps/utils/export.py",                "apps/utils/export.py"),
    ("models/nn/norm.py",                   "models/nn/norm.py"),
    ("models/nn/triton_rms_norm.py",        "models/nn/triton_rms_norm.py"),
    ("models/efficientvit/sam.py",          "models/efficientvit/sam.py"),
]

def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def resolve_target_root(root_arg: str) -> Path:
    """
    Accepts either:
      • .../efficientvit  (package dir containing __init__.py)
      • .../site-packages (parent that contains efficientvit/)
    Returns the path to .../efficientvit
    """
    p = Path(root_arg).expanduser().resolve()
    if not p.exists():
        raise SystemExit(f"[!] --root not found: {p}")

    if p.name == "efficientvit" and (p / "__init__.py").exists():
        return p

    cand = p / "efficientvit"
    if cand.is_dir() and (cand / "__init__.py").exists():
        return cand

    # last chance: search shallow
    maybe = list(p.glob("**/efficientvit/__init__.py"))
    if maybe:
        return Path(maybe[0]).parent

    raise SystemExit(f"[!] '{p}' doesn't look like 'efficientvit' or a parent of it.")

def copy_one(src: Path, dst: Path, backup: bool, dry: bool) -> Tuple[str, bool]:
    """
    Copy src -> dst. Returns (status, changed?)
    Status values:
      'missing-src', 'identical', 'copied', 'updated'
    """
    if not src.exists():
        return ("missing-src", False)

    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        try:
            if sha256(src) == sha256(dst):
                return ("identical", False)
        except Exception:
            # if hashing fails, treat as different
            pass

        if dry:
            return ("updated", True)

        if backup:
            bak = dst.with_suffix(dst.suffix + ".bak")
            if not bak.exists():
                shutil.copy2(dst, bak)

        shutil.copy2(src, dst)
        return ("updated", True)
    else:
        if dry:
            return ("copied", True)
        if backup:
            # no existing file => no backup needed
            pass
        shutil.copy2(src, dst)
        return ("copied", True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="Path to installed efficientvit package OR its parent (site-packages).")
    ap.add_argument("--backup", action="store_true",
                    help="Write <target>.bak before overwriting (default: off).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show what would change, do not write.")
    args = ap.parse_args()

    script_dir = Path(__file__).parent.resolve()
    patch_root = script_dir / "efficientvit"
    if not patch_root.exists():
        raise SystemExit(f"[!] Patch sources not found: {patch_root}")

    target_root = resolve_target_root(args.root)
    print(f"[info] patch source root : {patch_root}")
    print(f"[info] target pkg root   : {target_root}")
    print(f"[info] backup            : {'ON' if args.backup else 'OFF'}")
    print(f"[info] dry-run           : {'ON' if args.dry_run else 'OFF'}")

    changed_total = 0
    missing_total = 0

    for rel_src, rel_dst in PATCH_RELATIVE_FILES:
        src = (patch_root / rel_src).resolve()
        dst = (target_root / rel_dst).resolve()

        status, changed = copy_one(src, dst, args.backup, args.dry_run)
        if status == "missing-src":
            print(f"[skip] missing source: {src}")
            missing_total += 1
            continue

        if status == "identical":
            print(f"[ok]   identical: {rel_dst}")
        elif status == "copied":
            print(f"[copy] {rel_dst}  (new)")
            changed_total += 1
        elif status == "updated":
            print(f"[upd ] {rel_dst}  (replaced)")
            changed_total += 1
        else:
            print(f"[?]    {rel_dst}: {status}")

    if args.dry_run:
        print(f"[done] dry-run complete. Would change: {changed_total}, missing sources: {missing_total}")
    else:
        print(f"[done] applied patches. Changed: {changed_total}, missing sources: {missing_total}")

if __name__ == "__main__":
    main()
