#!/usr/bin/env bash
set -euo pipefail

echo "[cleanup] moving stray global jetson-inference files aside (if any)"

sudo mkdir -p /usr/local/_jetson_bak

# binaries
for b in imagenet detectnet posenet segnet actionnet backgroundnet camera-capture; do
  if [ -f "/usr/local/bin/$b" ]; then
    sudo mv "/usr/local/bin/$b" "/usr/local/_jetson_bak/$b.$(date +%s).bak"
    echo "  moved /usr/local/bin/$b"
  fi
done

# python site libs (if there are old .bak, just leave them)
for f in /usr/lib/python3.8/dist-packages/jetson_*_python.so; do
  [ -e "$f" ] || continue
  base="$(basename "$f")"
  sudo mv "$f" "/usr/local/_jetson_bak/$base.$(date +%s).bak"
  echo "  moved $f"
done

echo "[cleanup] done"