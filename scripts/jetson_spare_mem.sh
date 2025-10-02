#!/usr/bin/env bash
# jetson_spare_mem.sh — helper for freeing RAM and running low‑RAM builds on Jetsons
#
# Usage:
#   chmod +x jetson_spare_mem.sh
#   ./jetson_spare_mem.sh <command> [args]
#
# Commands (overview):
#   status                            Show RAM/swap/headless status
#   headless on|off                   Temporarily disable/enable GUI for this boot
#   headless persist-on|persist-off   Set default boot target to console or GUI
#   swap enable [SIZE] [PATH]         Create & enable swapfile (default 8G /swapfile)
#   swap disable [PATH]               Turn off & remove swapfile (default /swapfile)
#   swap tune [SWAPPINESS]            Set vm.swappiness (default 10)
#   swap show                         Show /proc/swaps and swappiness
#   caps print [4g|8g|16g|auto]       Print recommended env exports
#   wrap [4g|8g|16g|auto] -- CMD...   Run CMD with recommended env caps
#
# Notes:
# - Jetsons use unified memory; turning off the desktop frees DRAM used by Xorg/Wayland/compositor.
# - Swap is a *safety net* to avoid OOM kills; heavy swapping hurts performance on eMMC/SD.
# - "caps" = environment variables that limit build parallelism and memory usage.
#
set -euo pipefail

# ---------- util ----------
err() { echo -e "\e[31m[ERROR]\e[0m $*" >&2; }
inf() { echo -e "\e[36m[INFO]\e[0m  $*"; }
ok()  { echo -e "\e[32m[OK]\e[0m    $*"; }

have_cmd() { command -v "$1" &>/dev/null; }

detect_ram_gb() {
  local kb
  kb=$(awk '/MemTotal:/ {print $2}' /proc/meminfo)
  # Round to nearest 4/8/16 bucket
  local gb=$(( (kb + 524288) / 1048576 ))
  if (( gb <= 6 )); then echo 4; elif (( gb <= 12 )); then echo 8; else echo 16; fi
}

as_size() {
  # normalize SIZE like 8G/12G/4096M to dd-friendly count
  local s=${1:-8G}
  echo "$s" | tr '[:lower:]' '[:upper:]'
}

need_root_msg() {
  echo "This action requires root. Re-run with sudo, e.g.:"; echo "  sudo $0 $*"; }

# ---------- status ----------
cmd_status() {
  echo "==== STATUS ===="
  free -h || true
  echo
  echo "-- swap --"
  cat /proc/swaps || true
  echo "swappiness:" $(sysctl -n vm.swappiness 2>/dev/null || echo "<n/a>")
  echo
  echo "-- target (boot default) --"
  systemctl get-default || true
  echo "-- graphical target active? --"
  if systemctl is-active --quiet graphical.target; then echo "graphical.target: active"; else echo "graphical.target: inactive"; fi
}

# ---------- headless ----------
cmd_headless() {
  local mode=${1:-}
  case "$mode" in
    on)
      inf "Switching to non-graphical (multi-user) target for this boot…"
      sudo systemctl isolate multi-user.target
      ok  "Desktop stopped. Use: sudo systemctl isolate graphical.target to restore."
      ;;
    off)
      inf "Switching to graphical target for this boot…"
      sudo systemctl isolate graphical.target
      ok  "Desktop started."
      ;;
    persist-on)
      inf "Setting default boot target to multi-user (console)…"
      sudo systemctl set-default multi-user.target
      ok  "Default is now console. Revert with: sudo systemctl set-default graphical.target"
      ;;
    persist-off)
      inf "Setting default boot target to graphical (desktop)…"
      sudo systemctl set-default graphical.target
      ok  "Default is now graphical."
      ;;
    *)
      err "Usage: $0 headless on|off|persist-on|persist-off"; return 1 ;;
  esac
}

# ---------- swap ----------
cmd_swap() {
  local sub=${1:-}
  shift || true
  case "$sub" in
    enable)
      local size=$(as_size "${1:-8G}")
      local path=${2:-/swapfile}
      inf "Creating $size swapfile at $path…"
      if [[ -e "$path" ]]; then err "$path already exists"; return 1; fi
      sudo fallocate -l "$size" "$path"
      sudo chmod 600 "$path"
      sudo mkswap "$path"
      sudo swapon "$path"
      if ! grep -q "^$path" /etc/fstab; then echo "$path swap swap defaults 0 0" | sudo tee -a /etc/fstab >/dev/null; fi
      ok  "Swap enabled."
      ;;
    disable)
      local path=${1:-/swapfile}
      inf "Disabling and removing $path…"
      if swapoff "$path" 2>/dev/null; then sudo swapoff "$path" || true; fi
      sudo sed -i "\|^$path |d" /etc/fstab || true
      sudo sed -i "\|^$path swap swap defaults 0 0$|d" /etc/fstab || true
      sudo rm -f "$path"
      ok  "Swap removed."
      ;;
    tune)
      local sw=${1:-10}
      inf "Setting vm.swappiness=$sw (lower = less eager to swap)…"
      echo "vm.swappiness=$sw" | sudo tee /etc/sysctl.d/99-swap.conf >/dev/null
      sudo sysctl -p /etc/sysctl.d/99-swap.conf
      ;;
    show)
      cmd_status
      ;;
    *)
      err "Usage: $0 swap enable [SIZE] [PATH] | disable [PATH] | tune [SWAPPINESS] | show"; return 1 ;;
  esac
}

# ---------- caps / wrap ----------
print_caps() {
  local bucket=${1:-auto}
  if [[ "$bucket" == auto ]]; then
    local ram=$(detect_ram_gb); bucket="${ram}g"
  fi
  case "$bucket" in
    4g)
      cat <<'EOF'
export MAX_JOBS=1
export CMAKE_BUILD_PARALLEL_LEVEL=1
export USE_NINJA=0
export CFLAGS="-O2 -g0"; export CXXFLAGS="-O2 -g0"; export LDFLAGS="-Wl,--no-keep-memory"
# Optional TorchVision trims
export TORCHVISION_USE_FFMPEG=0
export TORCHVISION_USE_VIDEO_CODEC=0
# Set only your board's SM (Nano 5.3 | TX2 6.2)
# export TORCH_CUDA_ARCH_LIST=5.3
EOF
      ;;
    8g)
      cat <<'EOF'
export MAX_JOBS=2
export CMAKE_BUILD_PARALLEL_LEVEL=2
export USE_NINJA=1
export CFLAGS="-O2 -g0"; export CXXFLAGS="-O2 -g0"; export LDFLAGS="-Wl,--no-keep-memory"
# Optional TorchVision trims
# export TORCHVISION_USE_FFMPEG=0
# export TORCHVISION_USE_VIDEO_CODEC=0
# Set only your board's SM (Xavier 7.2)
# export TORCH_CUDA_ARCH_LIST=7.2
EOF
      ;;
    16g)
      cat <<'EOF'
export MAX_JOBS=3
export CMAKE_BUILD_PARALLEL_LEVEL=3
export USE_NINJA=1
export CFLAGS="-O2 -g0"; export CXXFLAGS="-O2 -g0"; export LDFLAGS="-Wl,--no-keep-memory"
# Set only your board's SM (Orin 8.7)
# export TORCH_CUDA_ARCH_LIST=8.7
EOF
      ;;
    *) err "Unknown bucket: $bucket (use 4g|8g|16g|auto)"; return 1 ;;
  esac
}

cmd_caps() {
  local sub=${1:-}
  case "$sub" in
    print)
      print_caps "${2:-auto}"
      ;;
    *) err "Usage: $0 caps print [4g|8g|16g|auto]"; return 1 ;;
  esac
}

cmd_wrap() {
  local bucket=${1:-auto}
  shift || true
  if [[ "${1:-}" == "--" ]]; then shift; fi
  if [[ $# -eq 0 ]]; then err "Usage: $0 wrap [4g|8g|16g|auto] -- <command> [args]"; return 1; fi
  # shellcheck disable=SC2046
  env -i HOME="$HOME" USER="$USER" SHELL="${SHELL:-/bin/bash}" PATH="$PATH" \
    bash -lc "$(print_caps "$bucket") ; exec \"$*\""
}

# ---------- main ----------
main() {
  local cmd=${1:-}
  shift || true
  case "$cmd" in
    status)   cmd_status ;; 
    headless) cmd_headless "$@" ;;
    swap)     cmd_swap "$@" ;;
    caps)     cmd_caps "$@" ;;
    wrap)     cmd_wrap "$@" ;;
    ''|-h|--help|help)
      cat <<'HELP'
jetson_spare_mem.sh — free RAM & run low‑RAM builds

Commands:
  status                          Show RAM, swaps, and GUI target
  headless on|off                 Temporarily disable/enable GUI for this boot
  headless persist-on|persist-off Set default boot target
  swap enable [SIZE] [PATH]       Create & enable swapfile (default 8G /swapfile)
  swap disable [PATH]             Disable & remove swapfile
  swap tune [SWAPPINESS]          Set vm.swappiness (e.g., 10)
  swap show                       Show swaps & swappiness
  caps print [4g|8g|16g|auto]     Print env caps you can `eval` in your shell
  wrap [4g|8g|16g|auto] -- CMD    Run CMD with env caps applied (hermetic)

Examples:
  # Free RAM now (console mode) and later restore GUI
  ./jetson_spare_mem.sh headless on
  # build …
  ./jetson_spare_mem.sh headless off

  # Add 8G swap (NVMe path recommended if available)
  ./jetson_spare_mem.sh swap enable 8G /swapfile
  ./jetson_spare_mem.sh swap tune 10

  # Use safe build caps for 8G board
  eval "$(./jetson_spare_mem.sh caps print 8g)"; python setup.py bdist_wheel

  # Or wrap a command hermetically with caps
  ./jetson_spare_mem.sh wrap 4g -- pip install --no-build-isolation -v .
HELP
      ;;
    *) err "Unknown command: $cmd (run with --help)"; return 1 ;;
  esac
}

main "$@"
