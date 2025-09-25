#!/usr/bin/env bash
# make_jetson_backup.sh — JetPack 5.1.x official backup wrapper
# Run on the HOST PC inside the Linux_for_Tegra directory you flashed from.

set -euo pipefail

BOARD="${BOARD:-jetson-xavier-nx-devkit-emmc}"   # change if you use a different board name
EXT_DEV="${EXT_DEV:-}"                           # e.g. nvme0n1 (optional; use if rootfs is on NVMe)

# sanity: must be in Linux_for_Tegra
[[ -f "./flash.sh" && -d "./tools/backup_restore" ]] || {
  echo "Run this from your Linux_for_Tegra directory (where flash.sh lives)."; exit 1; }

echo "[*] Ensure exactly ONE Jetson is in force-recovery and connected via USB."
read -rp "Press Enter to continue…"

CMD=(sudo ./tools/backup_restore/l4t_backup_restore.sh -b)
[[ -n "$EXT_DEV" ]] && CMD+=(-e "$EXT_DEV")
CMD+=("$BOARD")

echo "[*] Running: ${CMD[*]}"
"${CMD[@]}"

# show where the image landed
IMG_DIR=./tools/backup_restore/images
echo "[✓] Backup complete. Images are in: $IMG_DIR"
ls -lh --time-style=long-iso "$IMG_DIR" || true
