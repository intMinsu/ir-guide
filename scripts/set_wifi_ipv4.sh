#!/usr/bin/env bash
set -euo pipefail
# Prompts for SSID/PSK; sets static IPv4 based on hostname "deviceNN" or asks for 0–255
# Non-interactive: SSID=... PSK=... DIGIT=42 ./set_wifi_ipv4.sh

# ----- inputs (no secrets stored) -----
read -rp "Wi-Fi SSID [rcv-robot-5G]: " SSID_IN
SSID="${SSID_IN:-rcv-robot-5G}"

if [[ -n "${PSK:-}" ]]; then
  PSK_INPUT="$PSK"
else
  read -srp "Wi-Fi password (PSK): " PSK_INPUT; echo
fi

HOSTNAME_ARG="$(hostname)"

# ----- find Wi-Fi interface -----
IFACE="$(nmcli -t -f DEVICE,TYPE dev status | awk -F: '$2=="wifi"{print $1;exit}')"
if [[ -z "$IFACE" ]]; then
  echo "[!] No Wi-Fi interface found. Try: dmesg | grep -iE 'iwl|ax210|wifi|wlan'" >&2
  exit 1
fi
nmcli radio wifi on || true
nmcli dev set "$IFACE" managed yes || true

# ----- create/refresh connection -----
if ! nmcli -t -f NAME connection show | grep -Fxq "$SSID"; then
  nmcli connection add type wifi ifname "$IFACE" con-name "$SSID" ssid "$SSID"
fi
nmcli connection modify "$SSID" wifi-sec.key-mgmt wpa-psk wifi-sec.psk "$PSK_INPUT"
nmcli connection modify "$SSID" connection.autoconnect yes

# ----- decide static IPv4 -----
IP=""
if [[ "$HOSTNAME_ARG" =~ ^device([0-9]{1,3})$ ]]; then
  NUM="${BASH_REMATCH[1]}"
  if (( NUM >=0 && NUM <= 99 )); then
    IP="192.168.1.2${NUM}"   # deviceNN -> 192.168.1.2NN
  else
    IP="192.168.1.${NUM}"    # deviceXYZ -> 192.168.1.XYZ
  fi
else
  DIG="${DIGIT:-}"
  while :; do
    if [[ -z "$DIG" ]]; then
      read -rp "Static IPv4 last octet (0–255): " DIG
    fi
    if [[ "$DIG" =~ ^[0-9]{1,3}$ ]] && (( DIG >=0 && DIG <=255 )); then
      IP="192.168.1.${DIG}"
      break
    fi
    echo "Invalid value. Enter an integer 0–255."
    DIG=""
  done
fi

# ----- apply static IPv4 -----
nmcli connection modify "$SSID" \
  ipv4.addresses "${IP}/24" \
  ipv4.gateway "192.168.1.1" \
  ipv4.dns "8.8.8.8 8.8.4.4" \
  ipv4.method manual \
  ipv6.method ignore

nmcli connection up "$SSID" ifname "$IFACE" || true

echo "✓ Connected (requested) on ${IFACE}. Static IP = ${IP}"
echo "Check: ip a show ${IFACE} ; ping -c1 192.168.1.1"
