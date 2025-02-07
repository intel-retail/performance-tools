#!/usr/bin/env bash
#
# Copyright (C) 2024 Intel Corporation.
#
# SPDX-License-Identifier: Apache-2.0
#

# shellcheck disable=SC2086 # Intended work splitting
devices=$(intel_gpu_top -L  | grep card | awk '{print $1,$5;}' | sed ":a;N;s/\n/@/g")
echo "devices found $devices"

if [ -z "$devices" ]
then
    echo "No valid GPU devices found"
	exit 1
fi

IFS='@'
for device in $devices
do
    # shellcheck disable=SC2086 # Intended work splitting
    # the real device card number should be from card= instead of the device index string itself
    dev=$(echo $device | awk '{print $NF}' | sed -E 's/.*?card=//')
    echo "$dev"
    # shellcheck disable=SC2086 # Intended work splitting
    deviceId=$(echo $device | awk '{print $2}' | sed -E 's/.*?device=//' | cut -f1 -d",")
    echo "$deviceId"
    # shellcheck disable=SC2086 # Intended work splitting
    deviceNum=$(echo $dev | sed -E 's/.*?card//')
    echo "device number: $deviceNum"
    touch /tmp/results/igt$deviceNum-$deviceId.csv
    chown 1000:1000 /tmp/results/igt$deviceNum-$deviceId.csv
    # shellcheck disable=SC2086 # Intended work splitting
    intel_gpu_top -d pci:card=$deviceNum -c -o /tmp/results/igt$deviceNum-$deviceId.csv &
    echo "Starting igt capture for $device in igt$deviceNum-$deviceId.csv"
done

while true
do
	echo "Capturing igt metrics"
	sleep 15
done