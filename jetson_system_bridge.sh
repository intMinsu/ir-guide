# ------------------------------------------------------------------------------
# Expose system TensorRT & PyCUDA into this env (via .pth)
# ------------------------------------------------------------------------------

echo "[*] Bridging system packages (cv2, jtop, smbus2, tensorrt, pycuda) into the env…"

PY_SITE=$(python - <<'PY'
import site; print(site.getsitepackages()[0])
PY
)
BRIDGE_DIR="$CONDA_PREFIX/share/jetson-python-bridge"
BRIDGE_PTH="${PY_SITE}/jetson_system_bridge.pth"
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

: > "$BRIDGE_PTH"
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
          # special-case TRT: some distros place the .so next to __init__.py
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

  # Create symlinks into the bridge
  case "$name" in
    cv2)
      if [[ "$origin" == */__init__.py ]]; then
        ln -sfn "$(dirname "$origin")" "$BRIDGE_DIR/cv2"
      else
        ln -sfn "$origin" "$BRIDGE_DIR/$(basename "$origin")"
      fi
      ;;
    jtop|smbus2|tensorrt|pycuda)
      ln -sfn "$(dirname "$origin")" "$BRIDGE_DIR/$name"
      ;;
  esac

  echo "[+] bridged $name <- $origin"
  FOUND_ANY=1
done

if [[ "$FOUND_ANY" -eq 1 ]]; then
  echo "$BRIDGE_DIR" > "$BRIDGE_PTH"
  echo "[i] Bridge path written to: $BRIDGE_PTH"
else
  echo "[!] None of (cv2, jtop, smbus2, tensorrt, pycuda) found; no bridge created."
  echo "    If they’re installed under a different prefix, add that path manually to $BRIDGE_PTH"
fi