#!/usr/bin/env bash
# jetson_backup.sh  — JP 5.x official backup/restore wrapper for Jetson (NVMe/eMMC)
# Run this on the HOST PC, inside your Linux_for_Tegra/ directory.

set -euo pipefail

# -------- defaults you can override via env or flags --------
BOARD_DEFAULT="${BOARD:-jetson-xavier-nx-devkit-emmc}"   # set to your board name if different
IMAGES_DIR_DEFAULT="./tools/backup_restore/images"
EXT_DEV_DEFAULT="${EXT_DEV:-}"                            # e.g., nvme0n1 (leave empty for eMMC-only targets)

# -------- pretty logging --------
c_cyan='\033[1;36m'; c_yel='\033[1;33m'; c_grn='\033[1;32m'; c_red='\033[1;31m'; c_off='\033[0m'
log()  { printf "\n${c_cyan}[*]${c_off} %s\n" "$*"; }
ok()   { printf "${c_grn}[✓]${c_off} %s\n" "$*\n"; }
warn() { printf "${c_yel}[!]${c_off} %s\n" "$*\n"; }
die()  { printf "${c_red}[x] %s${c_off}\n" "$*" >&2; exit 1; }

# -------- usage --------
usage() {
  cat <<EOF
Usage:
  $(basename "$0") backup   [--board NAME] [--ext-dev DISK] [--images-dir DIR] [--no-confirm]
  $(basename "$0") restore  [--board NAME] [--ext-dev DISK] [--images-dir DIR] [--latest] [--no-confirm]
  $(basename "$0") list     [--images-dir DIR]

Notes:
  • Run this from your Linux_for_Tegra/ directory (where flash.sh lives).
  • For NVMe-root systems, pass --ext-dev nvme0n1 to back up / restore the external disk.
  • The restore flow uses the *latest* backup in images/ by default (see "list" to review).
  • --board defaults to: ${BOARD_DEFAULT}
  • --images-dir defaults to: ${IMAGES_DIR_DEFAULT}
Examples:
  # Backup an NVMe-root Xavier NX:
  ./jetson_backup.sh backup --ext-dev nvme0n1

  # Restore that image to another unit's NVMe:
  ./jetson_backup.sh restore --ext-dev nvme0n1

  # Just show available backups:
  ./jetson_backup.sh list
EOF
}

# -------- arg parsing --------
SUBCMD="${1:-}"
shift || true

BOARD="$BOARD_DEFAULT"
IMAGES_DIR="$IMAGES_DIR_DEFAULT"
EXT_DEV="$EXT_DEV_DEFAULT"
NO_CONFIRM=0
USE_LATEST=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --board)       BOARD="${2:-}"; shift 2 ;;
    --ext-dev)     EXT_DEV="${2:-}"; shift 2 ;;
    --images-dir)  IMAGES_DIR="${2:-}"; shift 2 ;;
    --no-confirm)  NO_CONFIRM=1; shift ;;
    --latest)      USE_LATEST=1; shift ;;
    -h|--help)     usage; exit 0 ;;
    *)             die "Unknown option: $1 (see --help)";;
  esac
done

# -------- sanity checks --------
[[ -f "./flash.sh" && -x "./flash.sh" ]] || die "Run this from Linux_for_Tegra/ (flash.sh not found)."
[[ -x "./tools/backup_restore/l4t_backup_restore.sh" ]] || die "backup/restore script not found under tools/backup_restore/."

mkdir -p "$IMAGES_DIR"

# Optional: quick USB check (single device in recovery)
usb_hint() {
  if command -v lsusb >/dev/null; then
    COUNT=$(lsusb -d 0955: | wc -l | tr -d ' ')
    if [[ "$COUNT" -eq 0 ]]; then
      warn "No NVIDIA USB device detected. Make sure exactly ONE Jetson is in Force-Recovery and connected."
    elif [[ "$COUNT" -gt 1 ]]; then
      warn "Multiple NVIDIA USB devices detected. Keep exactly ONE Jetson connected in Force-Recovery."
    else
      ok "One NVIDIA USB device detected (looks good)."
    fi
  fi
}

confirm() {
  [[ "$NO_CONFIRM" -eq 1 ]] && return 0
  read -rp "Proceed? [Y/n]: " yn; yn=${yn:-Y}
  [[ "$yn" =~ ^[Yy]$ ]] || die "Aborted."
}

list_images() {
  log "Available backups in: $IMAGES_DIR"
  shopt -s nullglob
  mapfile -t files < <(ls -1t "$IMAGES_DIR")
  if [[ ${#files[@]} -eq 0 ]]; then
    echo "(none)"
    return 1
  fi
  idx=1
  for f in "${files[@]}"; do
    sz=$(du -h "$IMAGES_DIR/$f" | awk '{print $1}')
    ts=$(stat -c '%y' "$IMAGES_DIR/$f" 2>/dev/null | cut -d'.' -f1)
    printf "  %2d) %s  (%s, %s)\n" "$idx" "$f" "$sz" "$ts"
    ((idx++))
  done
  echo "  Latest is item 1."
  return 0
}

do_backup() {
  log "Backup selected"
  echo "  Board       : $BOARD"
  echo "  Images dir  : $IMAGES_DIR"
  if [[ -n "$EXT_DEV" ]]; then
    echo "  External dev: $EXT_DEV (e.g., nvme0n1)"
  else
    echo "  External dev: (none) — backing up internal root only"
  fi
  usb_hint
  echo
  echo "Ensure EXACTLY ONE Jetson is in Force-Recovery and connected via USB."
  confirm

  CMD=(sudo ./tools/backup_restore/l4t_backup_restore.sh -b)
  [[ -n "$EXT_DEV" ]] && CMD+=(-e "$EXT_DEV")
  CMD+=("$BOARD")

  log "Running: ${CMD[*]}"
  "${CMD[@]}"

  ok "Backup complete."
  list_images || true
}

do_restore() {
  log "Restore selected"
  echo "  Board       : $BOARD"
  echo "  Images dir  : $IMAGES_DIR"
  if [[ -n "$EXT_DEV" ]]; then
    echo "  Target dev  : $EXT_DEV (e.g., nvme0n1)"
  else
    echo "  Target dev  : (none) — restoring to internal storage only"
  fi
  usb_hint
  echo
  echo "Ensure EXACTLY ONE Jetson is in Force-Recovery and connected via USB."
  echo

  if [[ "$USE_LATEST" -eq 1 ]]; then
    echo "Will use the latest backup found in $IMAGES_DIR."
  else
    if list_images; then
      echo
      read -rp "Use the LATEST (item 1)? [Y/n]: " yn; yn=${yn:-Y}
      if [[ ! "$yn" =~ ^[Yy]$ ]]; then
        warn "This wrapper uses the latest image by default. If you need a specific one, move others out of $IMAGES_DIR so it becomes the latest."
      fi
    else
      die "No backup images found to restore."
    fi
  fi
  confirm

  CMD=(sudo ./tools/backup_restore/l4t_backup_restore.sh)
  [[ -n "$EXT_DEV" ]] && CMD+=(-e "$EXT_DEV")
  CMD+=(-r "$BOARD")

  log "Running: ${CMD[*]}"
  "${CMD[@]}"

  ok "Restore finished. Power-cycle the Jetson and boot."
  echo "  After first boot, verify on the device:"
  echo "    findmnt /           # expect /dev/nvme0n1p? if restoring to NVMe"
  echo "    mount | grep efi    # ESP should be mounted"
}

# -------- subcommand dispatch --------
case "$SUBCMD" in
  backup)  do_backup ;;
  restore) do_restore ;;
  list)    list_images || true ;;
  ""|-h|--help) usage ;;
  *) die "Unknown subcommand: $SUBCMD (use backup|restore|list or --help)";;
esac
