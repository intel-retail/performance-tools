#!/usr/bin/env bash
#
# Copyright (C) 2024 Intel Corporation.
#
# SPDX-License-Identifier: Apache-2.0
#

echo "Starting platform data collection"

echo "Starting sar collection"
touch /tmp/results/cpu_usage.log
chown 1000:1000 /tmp/results/cpu_usage.log
sar 1 >& /tmp/results/cpu_usage.log &

echo "Starting free collection"
touch /tmp/results/memory_usage.log
chown 1000:1000 /tmp/results/memory_usage.log
free -s 1 >& /tmp/results/memory_usage.log &

echo "Starting iotop collection"
touch /tmp/results/disk_bandwidth.log
chown 1000:1000 /tmp/results/disk_bandwidth.log
iotop -o -P -b >& /tmp/results/disk_bandwidth.log &

is_xeon=`lscpu | grep -i xeon | wc -l`

if [ "$is_xeon"  == "1"  ]
  then
    echo "Starting pcm-memory collection"
    touch /tmp/results/pcm-memory.csv
    chown 1000:1000 /tmp/results/pcm-memory.csv
    /opt/intel/pcm-bin/bin/pcm-memory 1 -silent -nc -csv=/tmp/results/pcm-memory.csv &

    echo "Starting pcm-power collection"
    touch /tmp/results/pcm-power.log
    chown 1000:1000 /tmp/results/pcm-power.log
    /opt/intel/pcm-bin/bin/pcm-power >& /tmp/results/pcm-power.log &
  fi

echo "Starting general pcm collection"
touch /tmp/results/pcm.csv
chown 1000:1000 /tmp/results/pcm.csv
/opt/intel/pcm-bin/bin/pcm 1 -silent -nc -nsys -csv=/tmp/results/pcm.csv &

while true
do
	echo "Capturing platform data"
	sleep 15
done