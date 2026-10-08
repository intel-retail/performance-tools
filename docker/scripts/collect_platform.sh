#!/usr/bin/env bash
#
# Copyright (C) 2024 Intel Corporation.
#
# SPDX-License-Identifier: Apache-2.0
#
# RESULTS_DIR defaults to /tmp/results, matching the previous hardcoded
# path exactly -- callers that don't set it (e.g. docker-compose.yaml) see
# no behavior change.
RESULTS_DIR="${RESULTS_DIR:-/tmp/results}"
mkdir -p "${RESULTS_DIR}"

echo "Starting platform data collection"

echo "Starting sar collection"
touch "${RESULTS_DIR}/cpu_usage.log"
chown 1000:1000 "${RESULTS_DIR}/cpu_usage.log"
sar 1 >& "${RESULTS_DIR}/cpu_usage.log" &

echo "Starting free collection"
touch "${RESULTS_DIR}/memory_usage.log"
chown 1000:1000 "${RESULTS_DIR}/memory_usage.log"
free -s 1 >& "${RESULTS_DIR}/memory_usage.log" &

echo "Starting iotop collection"
touch "${RESULTS_DIR}/disk_bandwidth.log"
chown 1000:1000 "${RESULTS_DIR}/disk_bandwidth.log"
iotop -o -P -b >& "${RESULTS_DIR}/disk_bandwidth.log" &

is_xeon=`lscpu | grep -i xeon | wc -l`

if [ "$is_xeon"  == "1"  ]
  then
    echo "Starting pcm-memory collection"
    touch "${RESULTS_DIR}/pcm-memory.csv"
    chown 1000:1000 "${RESULTS_DIR}/pcm-memory.csv"
    /opt/intel/pcm-bin/bin/pcm-memory 1 -silent -nc -csv="${RESULTS_DIR}/pcm-memory.csv" &

    echo "Starting pcm-power collection"
    touch "${RESULTS_DIR}/pcm-power.log"
    chown 1000:1000 "${RESULTS_DIR}/pcm-power.log"
    /opt/intel/pcm-bin/bin/pcm-power >& "${RESULTS_DIR}/pcm-power.log" &
  fi

echo "Starting general pcm collection"
touch "${RESULTS_DIR}/pcm.csv"
chown 1000:1000 "${RESULTS_DIR}/pcm.csv"
/opt/intel/pcm-bin/bin/pcm 1 -silent -r -nc -nsys -csv="${RESULTS_DIR}/pcm.csv" &

while true
do
	echo "Capturing platform data"
	sleep 15
done
