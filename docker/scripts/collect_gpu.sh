#!/bin/bash

# Get all lines containing pci: and both device= and card=
mapfile -t pci_devices < <(
  for card in /dev/dri/card*; do
    pci_id=$(udevadm info --query=all --name=$card 2>/dev/null | grep PCI_SLOT_NAME | cut -d= -f2)
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
    device_id="${device_line#*,device=}"
    card_num="${device_line#*,card=}"

    driver="${device_line#*,driver=}"

    if [[ -n "$device_id" && -n "$card_num" ]]; then
        echo "Valid device found: $pci_info | Device ID: $device_id | Card Number: $card_num"

        output_file="/tmp/results/qmassa${card_num}-${device_id}-tool-generated.json"
        touch $output_file
        chown 1000:1000 $output_file

        echo "Starting igt capture to $output_file"
        $HOME/.cargo/bin/qmassa -d $pci_info -g -x -t "$output_file" &
    else
        echo "Skipping $card: Incomplete pci info"
    fi
done

# Continuous logging
while true; do
    echo "Capturing igt metrics..."
    sleep 15
done