#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Patch PEP-604 union types (A | B) in efficientvit to Python 3.8-friendly typing.Union[A, B].

Usage:
  python pep604_patch_efficientvit.py            # patch in-place
  python pep604_patch_efficientvit.py --dry-run  # show what would change
  python pep604_patch_efficientvit.py --root /path/to/site-packages/efficientvit

Notes:
- We only rewrite *type annotations* (def signatures, return types, variable annotations).
- We do not touch code expressions where '|' is used bitwise.
- Backups are written as <file>.bak
"""

import argparse
import os
import re
import sys
import pathlib
from typing import List, Tuple

# ------------------------------
# Utilities
# ------------------------------

def find_pkg_root(pkg_name: str) -> pathlib.Path:
    for p in map(pathlib.Path, sys.path):
        cand = p / pkg_name
        if cand.exists() and cand.is_dir():
            return cand
    raise FileNotFoundError(f"Couldn't locate '{pkg_name}' on sys.path. Use --root to specify it manually.")

def needs_union_import(src: str) -> bool:
    # Already has 'from typing import Union' or 'import typing' (latter ok; we still add explicit import if missing)
    return "from typing import Union" not in src

def insert_union_import(src: str) -> str:
    lines = src.splitlines()
    insert_idx = 0
    # Skip shebang, encoding, and module docstring opening
    if lines and lines[0].startswith("#!"):
        insert_idx = 1
    # Insert after initial comments/encoding lines
    while insert_idx < len(lines) and (lines[insert_idx].startswith("#") or "coding" in lines[insert_idx]):
        insert_idx += 1
    # If there's a module docstring at top, insert after it
    if insert_idx < len(lines) and re.match(r'^\s*(?:[ruRU]{0,2}["\'])', lines[insert_idx] or ""):
        # find docstring end
        delim = lines[insert_idx].strip()[0]
        q = '"""' if '"""' in lines[insert_idx] else "'''"
        if q in lines[insert_idx]:
            if lines[insert_idx].count(q) == 1:
                j = insert_idx + 1
                while j < len(lines) and q not in lines[j]:
                    j += 1
                insert_idx = min(j + 1, len(lines))
            else:
                insert_idx += 1
    lines.insert(insert_idx, "from typing import Union")
    return "\n".join(lines) + ("\n" if not src.endswith("\n") else "")

def split_union_tokens(s: str) -> List[str]:
    """Split a type annotation by top-level '|' while respecting brackets/quotes."""
    out, buf = [], []
    depth_paren = depth_brack = depth_brace = 0
    in_squote = in_dquote = False
    i = 0
    while i < len(s):
        ch = s[i]
        # string toggles
        if ch == "'" and not in_dquote:
            in_squote = not in_squote
        elif ch == '"' and not in_squote:
            in_dquote = not in_dquote
        elif not in_squote and not in_dquote:
            if ch == '(':
                depth_paren += 1
            elif ch == ')':
                depth_paren = max(0, depth_paren - 1)
            elif ch == '[':
                depth_brack += 1
            elif ch == ']':
                depth_brack = max(0, depth_brack - 1)
            elif ch == '{':
                depth_brace += 1
            elif ch == '}':
                depth_brace = max(0, depth_brace - 1)
            elif ch == '|' and depth_paren == depth_brack == depth_brace == 0:
                out.append("".join(buf).strip())
                buf = []
                i += 1
                continue
        buf.append(ch)
        i += 1
    if buf:
        out.append("".join(buf).strip())
    return out

def unionize(s: str) -> str:
    # Only convert if there is a top-level '|'
    if '|' not in s:
        return s
    parts = split_union_tokens(s)
    if len(parts) <= 1:
        return s
    return f"Union[{', '.join(parts)}]"

# Process a def signature (possibly multi-line) and return the patched signature
def patch_def_signature(sig: str) -> str:
    # Work on a copy
    s = sig

    # Patch parameter annotations: occurrences of ":" <annot> [',' or ')' or '=']
    def repl_param(m):
        prefix, annot, suffix = m.group(1), m.group(2), m.group(3)
        return f"{prefix}{unionize(annot)}{suffix}"

    # This regex scans only in the param list (no strings). It's heuristic but works well on clean code.
    param_pattern = re.compile(r"(:\s*)([^,)=]+(?:\|[^,)=]+)+)(\s*[,)=])")
    s = param_pattern.sub(repl_param, s)

    # Patch return annotation: '-> <annot> :'
    def repl_ret(m):
        prefix, annot, suffix = m.group(1), m.group(2), m.group(3)
        return f"{prefix}{unionize(annot)}{suffix}"

    ret_pattern = re.compile(r"(->\s*)([^:]+?\|[^:]+?)(\s*:)")
    s = ret_pattern.sub(repl_ret, s)

    return s

def patch_variable_annotation(line: str) -> str:
    """
    Patch a single-line variable annotation: name: A|B [= ...]
    Avoid touching lines that look like 'for x: y in ...' or 'class X:' or 'def ...'
    """
    if line.lstrip().startswith(("def ", "class ", "@", "for ", "while ", "if ", "elif ", "else:", "try:", "except", "with ")):
        return line

    # simple variable annotation pattern
    m = re.match(r"^(\s*[A-Za-z_][A-Za-z0-9_]*\s*:\s*)([^=#\n]+?\|[^=#\n]+?)(\s*(?:=|#|$).*)", line)
    if not m:
        return line
    prefix, annot, suffix = m.groups()
    return f"{prefix}{unionize(annot.strip())}{suffix}"

def patch_file(src: str) -> Tuple[str, bool, bool]:
    """
    Return (new_source, changed, added_import)
    """
    changed = False
    added_import = False
    lines = src.splitlines(keepends=True)
    out: List[str] = []

    i = 0
    while i < len(lines):
        line = lines[i]

        # Collect def signature (may be multi-line) up to the line ending with ':' and balanced parens
        if line.lstrip().startswith("def "):
            sig_lines = [line]
            # track parentheses until header end
            paren = line.count("(") - line.count(")")
            while not line.rstrip().endswith(":") or paren > 0:
                i += 1
                if i >= len(lines):
                    break
                line = lines[i]
                sig_lines.append(line)
                paren += line.count("(") - line.count(")")
            sig = "".join(sig_lines)
            patched_sig = patch_def_signature(sig)
            if patched_sig != sig:
                out.append(patched_sig)
                changed = True
            else:
                out.append(sig)
            i += 1
            continue

        # Variable annotation on one line
        patched_line = patch_variable_annotation(line)
        if patched_line != line:
            changed = True
        out.append(patched_line)
        i += 1

    new_src = "".join(out)

    # If we introduced Union[...], ensure 'from typing import Union' exists
    if "Union[" in new_src and needs_union_import(src):
        new_src = insert_union_import(new_src)
        changed = True
        added_import = True

    return new_src, changed, added_import

# ------------------------------
# CLI
# ------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=None,
                    help="Path to efficientvit package root (folder that contains __init__.py). "
                         "If omitted, auto-locate from sys.path.")
    ap.add_argument("--dry-run", action="store_true", help="Don't write files; just print what would change.")
    args = ap.parse_args()

    pkg_root = pathlib.Path(args.root) if args.root else find_pkg_root("efficientvit")
    if not (pkg_root / "__init__.py").exists():
        print(f"[!] '{pkg_root}' doesn't look like a package (missing __init__.py).")
        sys.exit(2)

    print(f"[info] scanning: {pkg_root}")
    py_files = sorted(p for p in pkg_root.rglob("*.py"))

    total_changed = 0
    for f in py_files:
        src = f.read_text(encoding="utf-8")
        new_src, changed, added_import = patch_file(src)
        if changed:
            total_changed += 1
            print(f"[patch] {f} {'(+ import Union)' if added_import else ''}")
            if not args.dry_run:
                backup = f.with_suffix(f.suffix + ".bak")
                if not backup.exists():
                    backup.write_text(src, encoding="utf-8")
                f.write_text(new_src, encoding="utf-8")

    if total_changed == 0:
        print("[info] no PEP-604 unions found in annotations (or nothing needed patching).")
    else:
        print(f"[done] patched {total_changed} file(s). Backups: '*.py.bak'")

if __name__ == "__main__":
    main()
