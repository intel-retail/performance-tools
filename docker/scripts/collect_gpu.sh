#!/bin/bash
#
# RESULTS_DIR defaults to /tmp/results, matching the previous hardcoded path
RESULTS_DIR="${RESULTS_DIR:-/tmp/results}"
mkdir -p "${RESULTS_DIR}"

# 0 (default) reproduces the original behavior exactly: qmassa runs once,
# unbounded, for the life of the script -- fine for a benchmark run that
# lasts minutes. Set > 0 to make qmassa exit and restart every N seconds,
# deleting its own output file first, so a 24/7 live dashboard doesn't
# eventually OOM on an ever-growing JSON file (qmassa never rotates it).
QMASSA_CYCLE_SECONDS="${QMASSA_CYCLE_SECONDS:-0}"

# Get all lines containing pci: and both device= and card=
mapfile -t pci_devices < <(
  for card in /dev/dri/card*; do
    pci_id=$(udevadm info --query=all --name=$card 2>/dev/null | grep -w DEVPATH | cut -d= -f2 | grep -oE '[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9]'  | tail -1 )
    pci_info=$(lspci -s $pci_id -nn | head -n1)
    if echo "$pci_info" | grep -iq "VGA compatible controller"; then
      vendor_device=$(echo "$pci_info" | grep -oP '\[\K[0-9a-f]{4}:[0-9a-f]{4}(?=\])')
      vendor=${vendor_device%%:*}
      device=${vendor_device##*:}
      driver=$(lspci -k -s $pci_id | grep "Kernel driver in use:" | awk '{print $5}')
      card_num=${card##*card}
      echo "pci:$pci_id,vendor=$vendor,device=$device,card=$card_num,driver=$driver"
    fi
  done
)

if [ ${#pci_devices[@]} -eq 0 ]; then
    echo "No valid PCI GPU devices with both device ID and card number found."
    exit 1
fi

for device_line in "${pci_devices[@]}"; do

    driver=$(echo $device_line | grep -oP 'driver=\K\S+')
    if [[ "$driver" != "i915" && "$driver" != "xe" && "$driver" != "amdgpu" ]]; then
      echo "Skipping device with driver $driver. Only i915, xe, and amdgpu are supported."
      exit 1
    fi
    # Extract the full pci string (starting from "pci:")
    pci_info="${device_line#pci:}"

    # Extract device ID and card number
    device_id=$(echo "$device_line" | grep -oP 'device=\K[^,]+')
    card_num=$(echo "$device_line" | grep -oP '(?<=card=)[^,]+')

    driver="${device_line#*,driver=}"

    if [[ -n "$device_id" && -n "$card_num" ]]; then
        echo "Valid device found: $pci_info | Device ID: $device_id | Card Number: $card_num"

        output_file="${RESULTS_DIR}/qmassa${card_num}-${device_id}-${driver}-tool-generated.json"
        touch "$output_file"
        chown 1000:1000 "$output_file"

        echo "Starting igt capture to $output_file"
        if [ "$QMASSA_CYCLE_SECONDS" -gt 0 ] 2>/dev/null; then
            # Bounded-cycle mode: run for QMASSA_CYCLE_SECONDS, then drop the
            # file and start fresh, so a long-running live dashboard reader
            # never sees an unbounded (multi-GB) JSON document.
            while true; do
                timeout "${QMASSA_CYCLE_SECONDS}s" "$HOME/.cargo/bin/qmassa" \
                    -d $pci_info -g -x -t "$output_file" 2>> "${RESULTS_DIR}/qmassa_error.log"
                rm -f "$output_file"
                touch "$output_file"
                chown 1000:1000 "$output_file"
            done
        else
            # Legacy/default behavior: single unbounded invocation, matching
            # every existing Docker-mode benchmarking consumer exactly.
            $HOME/.cargo/bin/qmassa -d $pci_info -g -x -t "$output_file" 2>> "${RESULTS_DIR}/qmassa_error.log"
        fi
    else
        echo "Skipping $card: Incomplete pci info"
    fi
done

# Continuous logging
while true; do
    echo "Capturing igt metrics..."
    sleep 15
done
