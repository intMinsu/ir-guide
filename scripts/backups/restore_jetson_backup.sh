#!/usr/bin/env bash
# restore_jetson_backup.sh — JetPack 5.1.x official restore wrapper

set -euo pipefail

BOARD="${BOARD:-jetson-xavier-nx-devkit-emmc}"   # same board name you use for flashing
EXT_DEV="${EXT_DEV:-nvme0n1}"                    # target disk (nvme0n1 for NVMe-root setups)

[[ -f "./flash.sh" && -d "./tools/backup_restore" ]] || {
  echo "Run this from your Linux_for_Tegra directory."; exit 1; }

IMG_DIR=./tools/backup_restore/images
[[ -d "$IMG_DIR" ]] || { echo "No images folder found at $IMG_DIR"; exit 1; }

echo "[*] Ensure ONE Jetson (same SKU) is in force-recovery and USB-connected."
read -rp "Press Enter to restore the latest backup in $IMG_DIR to $EXT_DEV …"

# l4t_backup_restore.sh picks the backup from images/, latest by default
sudo ./tools/backup_restore/l4t_backup_restore.sh -e "$EXT_DEV" -r "$BOARD"

echo "[✓] Restore finished. Power-cycle the Jetson and boot."
echo "    After first boot, verify:"
echo "      findmnt /           # should show /dev/nvme0n1p? if using NVMe"
echo "      mount | grep efi    # ESP should be mounted from NVMe"
