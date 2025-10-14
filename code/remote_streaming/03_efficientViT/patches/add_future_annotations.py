#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Add: from __future__ import annotations  to every *.py under efficientvit (for Python 3.8 compat).

Usage:
  # auto-locate installed package
  python add_future_annotations.py

  # specify root explicitly (either .../efficientvit or the site-packages that contains it)
  python add_future_annotations.py --root /path/to/site-packages/efficientvit
  python add_future_annotations.py --root /path/to/site-packages
"""

import sys, pathlib, re, argparse

def find_pkg_root_auto(pkg_name: str) -> pathlib.Path:
    for p in map(pathlib.Path, sys.path):
        cand = p / pkg_name
        if cand.is_dir():
            return cand
    raise SystemExit(f"[!] couldn't find '{pkg_name}' on sys.path (try --root)")

def resolve_root_arg(root_arg: str, pkg_name: str) -> pathlib.Path:
    p = pathlib.Path(root_arg).expanduser().resolve()
    if not p.exists():
        raise SystemExit(f"[!] --root path not found: {p}")
    # case 1: they gave .../efficientvit
    if (p / "__init__.py").exists() and p.name == pkg_name:
        return p
    # case 2: they gave site-packages; look for efficientvit inside
    if (p / pkg_name / "__init__.py").exists():
        return p / pkg_name
    # case 3: they gave a random dir that still contains efficientvit somewhere
    cand = p / pkg_name
    if cand.is_dir() and (cand / "__init__.py").exists():
        return cand
    raise SystemExit(f"[!] '{p}' doesn't look like '{pkg_name}' or a parent of it")

def insert_future(src: str) -> str:
    if "from __future__ import annotations" in src:
        return src
    lines = src.splitlines(True)  # keep line endings
    i = 0
    # shebang
    if i < len(lines) and lines[i].startswith("#!"):
        i += 1
    # encoding/comment lines
    while i < len(lines) and (lines[i].lstrip().startswith("#") and "coding" in lines[i]):
        i += 1
    # module docstring
    if i < len(lines) and re.match(r'^\s*[ruRU]{0,2}("""|\'\'\')', lines[i]):
        q = '"""' if '"""' in lines[i] else "'''"
        if lines[i].count(q) == 1:
            i += 1
            while i < len(lines) and q not in lines[i]:
                i += 1
            i = min(i + 1, len(lines))
        else:
            i += 1
    lines.insert(i, "from __future__ import annotations\n")
    return "".join(lines)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=None,
                    help="Path to package root (…/efficientvit) or parent (…/site-packages)")
    args = ap.parse_args()

    PKG = "efficientvit"
    pkg_root = resolve_root_arg(args.root, PKG) if args.root else find_pkg_root_auto(PKG)

    if not (pkg_root / "__init__.py").exists():
        raise SystemExit(f"[!] '{pkg_root}' is not a package (missing __init__.py)")

    print(f"[info] scanning: {pkg_root}")
    changed = 0
    for f in sorted(pkg_root.rglob("*.py")):
        s = f.read_text(encoding="utf-8")
        ns = insert_future(s)
        if ns != s:
            bak = f.with_suffix(f.suffix + ".bak")
            if not bak.exists():
                bak.write_text(s, encoding="utf-8")
            f.write_text(ns, encoding="utf-8")
            print(f"[patch] +future: {f}")
            changed += 1

    if changed == 0:
        print("[info] nothing to patch (already present)")

if __name__ == "__main__":
    main()
