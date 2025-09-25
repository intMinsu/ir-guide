#!/usr/bin/env bash
set -euo pipefail
# Usage: ./enable_ax210.sh [hostname]
# Env: KEEP_BT=1  (skip disabling Bluetooth)

HOSTNAME_ARG="${1:-$(hostname)}"
KEEP_BT="${KEEP_BT:-0}"

echo "[*] Preparing system (NetworkManager, firmware)…"
sudo apt-get update
sudo apt-get install -y unzip wget network-manager linux-firmware
sudo systemctl enable --now NetworkManager

if [[ "$HOSTNAME_ARG" != "$(hostname)" ]]; then
  echo "[*] Setting hostname -> $HOSTNAME_ARG"
  sudo hostnamectl set-hostname "$HOSTNAME_ARG"
fi

echo "[*] Applying AX210 (third-party) patch…"
pushd /tmp >/dev/null
URL="https://github.com/mistelektronik/forecr_blog_files/raw/master/Intel/AX210_driver.zip"
wget -O AX210_driver.zip "$URL"
rm -rf AX210_driver && unzip -q AX210_driver.zip -d AX210_driver
cd AX210_driver
sudo bash ./apply_patch.sh || true
if [[ "$KEEP_BT" != "1" ]]; then
  sudo bash ./disable_bt.sh || true
else
  echo "[i] KEEP_BT=1 set — skipping Bluetooth disable"
fi
# Reload Wi-Fi driver
sudo modprobe -r iwlwifi || true
sudo modprobe iwlwifi || true
popd >/dev/null

# Try to spot the Wi-Fi iface so user knows it's alive
IFACE="$(nmcli -t -f DEVICE,TYPE dev status | awk -F: '$2=="wifi"{print $1;exit}')"
[[ -n "$IFACE" ]] && echo "[✓] Wireless interface detected: $IFACE" || echo "[!] No Wi-Fi iface yet; reboot may be required"

echo "[i] Done. You can now run set_wifi_ipv4.sh to connect & set static IP."
