#!/usr/bin/env python3
# pep604_patch.py
"""
Rewrite PEP-604 union annotations (A | B) to Python 3.8-friendly typing.Union[A, B].

What it does
------------
- Rewrites *type annotations* in:
  • function signatures (params + returns)
  • single-line variable annotations
- Adds `from typing import Union` if needed.
- Leaves bitwise `|` operations in code alone.

Targets
-------
You can target by:
  1) --root <path to a package folder that contains __init__.py>, OR
  2) --root <site-packages> plus --package <name>, OR
  3) --package <name> (auto-locate on sys.path)

Usage
-----
  # Auto-locate a package on sys.path
  python pep604_patch.py --package efficientvit

  # Explicit package dir
  python pep604_patch.py --root /path/to/site-packages/efficientvit

  # Parent site-packages + package name
  python pep604_patch.py --root /path/to/site-packages --package efficientvit

  # Dry-run + backups
  python pep604_patch.py --package efficientvit --dry-run
  python pep604_patch.py --package efficientvit --backup
"""

import argparse
import re
import sys
from pathlib import Path
from typing import List, Tuple, Optional

# ------------------------------
# Utilities
# ------------------------------

def find_pkg_root_by_name(pkg_name: str) -> Path:
    for p in map(Path, sys.path):
        cand = p / pkg_name
        if cand.exists() and cand.is_dir() and (cand / "__init__.py").exists():
            return cand
    raise FileNotFoundError(
        "Couldn't locate package '{}' on sys.path. "
        "Use --root to specify a path or ensure the package is importable."
        .format(pkg_name)
    )

def resolve_target_root(root_arg: Optional[str], package: Optional[str]) -> Path:
    """
    Resolve to the concrete package directory (that contains __init__.py).
    Accepts:
      • root == package dir directly
      • root == parent like site-packages (+ --package required)
      • package name only (auto-locate on sys.path)
    """
    if root_arg:
        p = Path(root_arg).expanduser().resolve()
        if not p.exists():
            raise SystemExit("[!] --root not found: {}".format(p))

        # case 1: they gave the package dir directly
        if (p / "__init__.py").exists():
            return p

        # case 2: they gave parent dir, need package name
        if package:
            cand = p / package
            if cand.is_dir() and (cand / "__init__.py").exists():
                return cand
            hits = list(p.glob("**/{}/__init__.py".format(package)))
            if hits:
                return hits[0].parent
            raise SystemExit("[!] '{}' not found under '{}'".format(package, p))
        else:
            raise SystemExit(
                "[!] '{}' isn't a package (missing __init__.py). "
                "Provide --package <name> or point --root to the package directory."
                .format(p)
            )
    else:
        if not package:
            package = "efficientvit"
        return find_pkg_root_by_name(package)

def needs_union_import(src: str) -> bool:
    return "from typing import Union" not in src

def insert_union_import(src: str) -> str:
    lines = src.splitlines()
    insert_idx = 0
    # shebang
    if lines and lines[0].startswith("#!"):
        insert_idx = 1
    # encoding/comment lines
    while insert_idx < len(lines) and (lines[insert_idx].startswith("#") or "coding" in lines[insert_idx]):
        insert_idx += 1
    # module docstring
    if insert_idx < len(lines) and re.match(r'^\s*(?:[ruRU]{0,2}["\'])', lines[insert_idx] or ""):
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
    if '|' not in s:
        return s
    parts = split_union_tokens(s)
    if len(parts) <= 1:
        return s
    return "Union[{}]".format(", ".join(parts))

def patch_def_signature(sig: str) -> str:
    s = sig

    def repl_param(m):
        prefix, annot, suffix = m.group(1), m.group(2), m.group(3)
        return "{}{}{}".format(prefix, unionize(annot), suffix)

    # heuristic that stays within params list
    param_pattern = re.compile(r"(:\s*)([^,)=]+(?:\|[^,)=]+)+)(\s*[,)=])")
    s = param_pattern.sub(repl_param, s)

    def repl_ret(m):
        prefix, annot, suffix = m.group(1), m.group(2), m.group(3)
        return "{}{}{}".format(prefix, unionize(annot), suffix)

    ret_pattern = re.compile(r"(->\s*)([^:]+?\|[^:]+?)(\s*:)")
    s = ret_pattern.sub(repl_ret, s)
    return s

def patch_variable_annotation(line: str) -> str:
    if line.lstrip().startswith(("def ", "class ", "@", "for ", "while ", "if ", "elif ", "else:", "try:", "except", "with ")):
        return line
    m = re.match(r"^(\s*[A-Za-z_][A-Za-z0-9_]*\s*:\s*)([^=#\n]+?\|[^=#\n]+?)(\s*(?:=|#|$).*)", line)
    if not m:
        return line
    prefix, annot, suffix = m.groups()
    return "{}{}{}".format(prefix, unionize(annot.strip()), suffix)

def patch_file(src: str) -> Tuple[str, bool, bool]:
    """
    Return (new_source, changed, added_import)
    """
    changed = False
    added_import = False
    lines = src.splitlines(keepends=True)
    out = []  # type: List[str]

    i = 0
    while i < len(lines):
        line = lines[i]
        if line.lstrip().startswith("def "):
            sig_lines = [line]
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

        patched_line = patch_variable_annotation(line)
        if patched_line != line:
            changed = True
        out.append(patched_line)
        i += 1

    new_src = "".join(out)

    # Ensure 'from typing import Union'
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
                    help="Package directory (contains __init__.py) or parent like site-packages.")
    ap.add_argument("--package", type=str, default=None,
                    help="Package name (used if --root is a parent, or to auto-locate on sys.path).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show what would change; do not write files.")
    ap.add_argument("--backup", action="store_true",
                    help="Write <file>.bak before patching (default: off).")
    args = ap.parse_args()

    pkg_root = resolve_target_root(args.root, args.package)
    if not (pkg_root / "__init__.py").exists():
        print("[!] '{}' doesn't look like a package (missing __init__.py).".format(pkg_root))
        sys.exit(2)

    print("[info] scanning: {}".format(pkg_root))
    py_files = sorted(p for p in pkg_root.rglob("*.py"))

    total_changed = 0
    for f in py_files:
        src = f.read_text(encoding="utf-8")
        new_src, changed, added_import = patch_file(src)
        if changed:
            total_changed += 1
            print("[patch] {} {}".format(f, "(+ import Union)" if added_import else ""))
            if not args.dry_run:
                if args.backup:
                    bak = f.with_suffix(f.suffix + ".bak")
                    if not bak.exists():
                        bak.write_text(src, encoding="utf-8")
                f.write_text(new_src, encoding="utf-8")

    if total_changed == 0:
        print("[info] no PEP-604 unions found in annotations (or nothing needed patching).")
    else:
        if args.backup:
            print("[done] patched {} file(s). Backups: '*.py.bak'".format(total_changed))
        else:
            print("[done] patched {} file(s).".format(total_changed))

if __name__ == "__main__":
    main()
