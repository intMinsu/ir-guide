#!/usr/bin/env bash
# jetson_system_bridge.sh
# ------------------------------------------------------------------------------
# Expose system cv2 / jtop / smbus2 / tensorrt / pycuda into this env (via .pth)
# and prioritize them over pip wheels by ensuring the bridge path is at sys.path[0].
# ------------------------------------------------------------------------------

set -euo pipefail

echo "[*] Bridging system packages (cv2, jtop, smbus2, tensorrt, pycuda) into the env…"

PY_SITE=$(python - <<'PY'
import site; print(site.getsitepackages()[0])
PY
)
BRIDGE_DIR="$CONDA_PREFIX/share/jetson-python-bridge"
PTH_FILE="${PY_SITE}/jetson_system_bridge.pth"
mkdir -p "$BRIDGE_DIR"

# Prefer real system python (not the conda one)
if [[ -x /usr/bin/python3 ]]; then
  SYS_PY=/usr/bin/python3
elif [[ -x /usr/local/bin/python3 ]]; then
  SYS_PY=/usr/local/bin/python3
else
  SYS_PY=$(command -v python3 || echo python3)
fi
echo "[i] System python for probing: $SYS_PY"

# Try clean env import probe with system python
readarray -t _LOCS < <( env -i \
  PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  "$SYS_PY" -s -E - <<'PY'
import importlib, pathlib
mods = ("cv2","jtop","smbus2","tensorrt","pycuda")
for m in mods:
    try:
        mod = importlib.import_module(m)
        p = pathlib.Path(getattr(mod, "__file__", "")) if hasattr(mod,"__file__") else None
        print(f"{m}|{str(p.resolve()) if p else ''}")
    except Exception:
        print(f"{m}|")
PY
)

# Fallback search paths (adjust pythonX.Y if needed)
CANDS=(
  "/usr/lib/python3.8/dist-packages"
  "/usr/local/lib/python3.8/dist-packages"
  "/usr/lib/python3/dist-packages"
  "/usr/local/lib/python3/dist-packages"
)

FOUND_ANY=0
for line in "${_LOCS[@]}"; do
  name="${line%%|*}"
  origin="${line#*|}"

  # If probe failed, scan common system dirs
  if [[ -z "$origin" ]]; then
    for base in "${CANDS[@]}"; do
      case "$name" in
        cv2)
          if [[ -d "$base/cv2" ]]; then origin="$base/cv2/__init__.py"; break; fi
          found_so=$(ls "$base"/cv2.*.so 2>/dev/null | head -n1 || true)
          [[ -n "$found_so" ]] && origin="$found_so" && break
          ;;
        jtop|smbus2|tensorrt|pycuda)
          if [[ -d "$base/$name" ]]; then origin="$base/$name/__init__.py"; break; fi
          if [[ "$name" == "tensorrt" ]]; then
            found_trt=$(ls "$base"/tensorrt/tensorrt*.so 2>/dev/null | head -n1 || true)
            [[ -n "$found_trt" ]] && origin="$base/tensorrt/__init__.py" && break
          fi
          ;;
      esac
    done
  fi

  if [[ -z "$origin" ]]; then
    echo "[!] System module not found: $name (skipping)"
    continue
  fi

  mkdir -p "$BRIDGE_DIR"

  # Create symlinks into the bridge (use realpaths so cv2.__file__ realpath is truthful)
  case "$name" in
    cv2)
      if [[ "$origin" == */__init__.py ]]; then
        ln -sfn "$(realpath -m "$(dirname "$origin")")" "$BRIDGE_DIR/cv2"
      else
        ln -sfn "$(realpath -m "$origin")" "$BRIDGE_DIR/$(basename "$origin")"
      fi
      ;;
    jtop|smbus2|tensorrt|pycuda)
      ln -sfn "$(realpath -m "$(dirname "$origin")")" "$BRIDGE_DIR/$name"
      ;;
  esac

  echo "[+] bridged $name <- $origin"
  FOUND_ANY=1
done

if [[ "$FOUND_ANY" -eq 1 ]]; then
  # Write a .pth with:
  #  1) a literal path line (appends to sys.path)
  #  2) a single executed line that moves it to the front (must be one line)
  cat > "$PTH_FILE" <<EOF
$BRIDGE_DIR
import sys; p=r"$BRIDGE_DIR"; sys.path.insert(0, sys.path.pop(sys.path.index(p))) if p in sys.path else sys.path.insert(0,p)
EOF
  echo "[i] Bridge path written to: $PTH_FILE"

  # Quick sanity check
  python - <<'PY' || true
import os, sys, importlib
print("[i] sys.path[0]       :", sys.path[0])
try:
    import cv2
    print("[i] cv2 path (real)   :", os.path.realpath(getattr(cv2, "__file__", "")))
    info = cv2.getBuildInformation() if hasattr(cv2, "getBuildInformation") else ""
    has_cuda = ("CUDA: YES" in info) or bool(getattr(cv2, "cuda", None))
    print("[i] cv2 CUDA          :", "YES" if has_cuda else "NO")
except Exception as e:
    print("[!] cv2 import failed :", e)
PY

else
  echo "[!] None of (cv2, jtop, smbus2, tensorrt, pycuda) found; no bridge created."
  echo "    If they’re installed under a different prefix, add that path manually to $PTH_FILE"
fi
