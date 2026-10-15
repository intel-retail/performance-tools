'''
* Copyright (C) 2023 Intel Corporation.
*
* SPDX-License-Identifier: Apache-2.0
'''

import pathlib
import datetime
import argparse
from abc import ABC, abstractmethod
from statistics import mean, stdev, median
import os
import re
import fnmatch
import numpy as np
import pandas as pd
from collections import defaultdict
from natsort import natsorted
from operator import add
import json
import csv

# constants
AVG_CPU_USAGE_CONSTANT = "CPU Utilization %"
AVG_NPU_USAGE_CONSTANT = "NPU Utilization %"
AVG_GPU_USAGE_CONSTANT = "GPU Utilization %"
AVG_GPU_MEM_USAGE_CONSTANT = "Memory Utilization %"
AVG_GPU_COMPUTE_USAGE_CONSTANT = "Compute Utilization %"
AVG_GPU_VDBOX_USAGE_CONSTANT = "Utilization %"

AVG_DISK_READ_BANDWIDTH_CONSTANT = "Disk Read MB/s"
AVG_DISK_WRITE_BANDWIDTH_CONSTANT = "Disk Write MB/s"
AVG_MEM_USAGE_CONSTANT = "Memory Utilization %"
AVG_POWER_USAGE_CONSTANT = "Power Draw W"
AVG_MEM_BANDWIDTH_CONSTANT = "Memory Bandwidth Usage MB/s"
AVG_FPS_CONSTANT = "FPS"
TEXT_COUNT_CONSTANT = "Total Text count"
BARCODE_COUNT_CONSTANT = "Total Barcode count"
PIPELINE_LATENCY_CONSTANT = "Latency"

def get_vlm_application_latency(log_file_path):
    """
    Extract timing data from VLM application metrics log files.
    
    Args:
        log_file_path (str): Path to the log file
        
    Returns:
        dict: Statistics for each application-id combination
    """
    timing_data = defaultdict(list)
    with open(log_file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if 'application=' in line and 'timestamp_ms=' in line:
                pattern = r'(\w+)=((?:[^\s=]+(?:\s(?!\w+=)[^\s=]+)*)?)'
                matches = re.findall(pattern, line)
                data = dict(matches)
                if data:
                    app_name = data.get('application')
                    id_value = data.get('id')
                    event = data.get('event')
                    timestamp_ms = data.get('timestamp_ms')
            
                    if app_name and id_value and event in ['start', 'end'] and timestamp_ms:
                        timing_data[f"{app_name}_{id_value}"].append({
                            'event': event,
                            'timestamp_ms': int(timestamp_ms)
                        })
    
    # Calculate durations for each application
    app_durations = defaultdict(list)
    apps_to_analyze = timing_data.keys()
    
    for app_id_combo in apps_to_analyze:
        events = sorted(timing_data[app_id_combo], key=lambda x: x['timestamp_ms'])
        start_times = []  # Use a stack to track multiple start times
        
        for event in events:
            if event['event'] == 'start':
                start_times.append(event['timestamp_ms'])
            elif event['event'] == 'end' and start_times:
                # Pair with the most recent start time (LIFO)
                start_time = start_times.pop()
                duration = event['timestamp_ms'] - start_time
                app_durations[app_id_combo].append(duration)

    durations = dict(app_durations)

    # Generate statistics
    statistics = {}
    for app_id_combo, duration_list in durations.items():
        count = 0
        if duration_list:
            count = len(duration_list)
            statistics[f"Based on {count} total calls, the average {app_id_combo} in (ms)"] = (sum(duration_list)/count)
        else:
            statistics[f"Based on {count} total calls, the average {app_id_combo} in (ms)"] = 0
    
    return statistics


def get_vlm_application_latency_stream_density(results_dir, last_n_pairs=20,
                                               since_ms=None):
    """Extract per-application average latency from the most recent
    vlm_application_metrics file in *results_dir*, using only the last
    *last_n_pairs* completed start/end pairs that begin at or after
    *since_ms* (epoch milliseconds).  Pass ``since_ms=None`` to use all pairs.

    Args:
        results_dir: Directory to search for vlm_application_metrics*.txt.
        last_n_pairs: Number of most-recent completed pairs to use per app.
        since_ms: If set, ignore events with timestamp_ms < since_ms.

    Returns:
        dict mapping ``app_id`` → ``avg_latency_ms`` (float), or empty dict
        if no metrics files are found.
    """
    metrics_files = sorted(
        fnmatch.filter(
            [str(p) for p in pathlib.Path(results_dir).rglob("vlm_application_metrics*.txt")],
            "*vlm_application_metrics*",
        ),
        key=os.path.getmtime,
    )
    if not metrics_files:
        return {}

    log_file_path = metrics_files[-1]

    timing_data = defaultdict(list)
    with open(log_file_path, "r") as fh:
        for line in fh:
            line = line.strip()
            if "application=" not in line or "timestamp_ms=" not in line:
                continue
            pattern = r'(\w+)=((?:[^\s=]+(?:\s(?!\w+=)[^\s=]+)*)?)'
            matches = re.findall(pattern, line)
            data = dict(matches)
            if not data:
                continue
            app_name = data.get("application", "")
            id_value = data.get("id", "")
            event = data.get("event", "")
            timestamp_ms = data.get("timestamp_ms", "")
            if app_name and id_value and event in ("start", "end") and timestamp_ms:
                ts = int(timestamp_ms)
                if since_ms is not None and ts < since_ms:
                    continue
                timing_data[f"{app_name}_{id_value}"].append(
                    {"event": event, "timestamp_ms": ts}
                )

    result = {}
    for app_id, events in timing_data.items():
        events.sort(key=lambda x: x["timestamp_ms"])
        durations = []
        start_stack = []
        for ev in events:
            if ev["event"] == "start":
                start_stack.append(ev["timestamp_ms"])
            elif ev["event"] == "end" and start_stack:
                durations.append(ev["timestamp_ms"] - start_stack.pop())
        if durations:
            tail = durations[-last_n_pairs:]
            result[app_id] = sum(tail) / len(tail)

    return result

class KPIExtractor(ABC):
    @abstractmethod
    def extract_data(self, log_file_path):
        pass

    @abstractmethod
    def return_blank(self):
        pass

class CPUUsageExtractor(KPIExtractor):
    _SAR_CPU_USAGE_PATTERN = "(\\d\\d:\\d\\d:\\d\\d)\\s+(\\w+)\\s+(\\d+.\\d+)\\s+(\\d+.\\d+)\\s+(\\d+.\\d+)\\s+(\\d+.\\d+)\\s+(\\d+.\\d+)\\s+(\\d+.\\d+)"
    _IDLE_CPU_PERCENT_GROUP = 8

    #overriding abstract method
    def extract_data(self, log_file_path):
        if os.path.getsize(log_file_path) == 0:
            return {AVG_CPU_USAGE_CONSTANT: "NA"}

        print("parsing CPU usages")
        sar_cpu_usage_row_p = re.compile(self._SAR_CPU_USAGE_PATTERN)
        cpu_usages = []
        with open(log_file_path) as f:
            for line in f:
                sar_cpu_usage_row_m= sar_cpu_usage_row_p.match(line)
                if sar_cpu_usage_row_m:
                    idle_cpu_percentage = float(sar_cpu_usage_row_m.group(self._IDLE_CPU_PERCENT_GROUP))
                    cpu_usages.append(float(100) - idle_cpu_percentage)
        if len(cpu_usages) > 0:
            return {AVG_CPU_USAGE_CONSTANT: round(mean(cpu_usages), 2)}
        else:
            return {AVG_CPU_USAGE_CONSTANT: "NA"}

    def return_blank(self):
        return {AVG_CPU_USAGE_CONSTANT: "NA"}

class NPUUsageExtractor(KPIExtractor):
    #overriding abstract method
    def extract_data(self, log_file_path):
        print("parsing NPU csv")
        npu_df = pd.read_csv(log_file_path)
        if len(npu_df['percent_usage']) > 0:
            return {AVG_NPU_USAGE_CONSTANT: mean(npu_df['percent_usage'])}
        else:
            return {AVG_NPU_USAGE_CONSTANT: "NA"}

    def return_blank(self):
        return {AVG_NPU_USAGE_CONSTANT: "NA"}


class VLMAppMetricsExtractor(KPIExtractor):
    def extract_data(self, log_file_path):
        print("parsing application performance metrics")
        return get_vlm_application_latency(log_file_path)

    def return_blank(self):
        return {"Application Performance Metrics": "NA"}

class VLMPerformanceMetricsExtractor(KPIExtractor):
    #overriding abstract method
    def extract_data(self, log_file_path):       
        print("parsing VLM latency")
        app_id_performance_data = defaultdict(list)
        
        with open(log_file_path, "r") as f:
            for line in f:
                line = line.strip()
                if 'Generate_Duration_Mean=' in line and 'application=' in line and 'id=' in line:
                    # Parse the line to extract key-value pairs
                    pattern = r'(\w+)=((?:[^\s=]+(?:\s(?!\w+=)[^\s=]+)*)?)'
                    matches = re.findall(pattern, line)
                    data = dict(matches)
                    
                    if data:
                        app_name = data.get('application')
                        id_value = data.get('id')
                        duration_str = data.get('Generate_Duration_Mean')
                        
                        if app_name and id_value and duration_str:
                            try:
                                duration = float(duration_str)
                                app_id_combo = f"{app_name}_{id_value}"
                                app_id_performance_data[app_id_combo].append(duration)
                            except ValueError:
                                continue
        
        # Calculate statistics for each application-id combination
        performance_stats = {}
        for app_id_combo, durations in app_id_performance_data.items():
            if durations:
                call_count = len(durations)
                avg_duration = sum(durations) / call_count
                performance_stats[f"VLM_TOTAL_CALLS_{app_id_combo}"] = call_count
                performance_stats[f"VLM_AVERAGE_CALL_DURATION_{app_id_combo}"] = avg_duration
        
        # If no application-id based data found, fall back to the original approach
        if not performance_stats:
            total_duration = 0.0
            count = 0
            with open(log_file_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    for part in parts:
                        if part.startswith("Generate_Duration_Mean="):
                            duration_str = part.split("=")[1]
                            try:
                                duration = float(duration_str)
                                total_duration += duration
                                count += 1
                            except ValueError:
                                continue
            
            performance_stats['VLM_TOTAL_CALLS'] = count
            performance_stats['VLM_AVERAGE_CALL_DURATION'] = total_duration / count if count > 0 else 0
    
        return performance_stats
    
    def return_blank(self):
        return {"VLM_TOTAL_CALLS": "NA", 'VLM_AVERAGE_CALL_DURATION': "NA"}

class GPUUsageExtractor(KPIExtractor):
    #overriding abstract method
    def extract_data(self, log_file_path):
        print("parsing GPU usages")
        device = re.findall(r'\d+', os.path.basename(log_file_path))
        desc_map = {
            'CCS %': 'Compute[CCS] Utilization %',
            'RCS %': 'Render/3D[RCS] Utilization %',
            'VCS %': 'Video[VCS] Utilization %',
            'VECS %': 'VideoEnhance[VECS] Utilization %',
            'Power W pkg': 'GPU Power (W)',
            'RC6 %': 'GPU Idle Time (RC6 %)'
        }

        device_prefix = f"GPU_{device[0]}"
        gpu_device_usage = {
            f"{device_prefix} {desc}": 'NA' for desc in desc_map.values()
        }

        if os.path.isfile(log_file_path) and os.path.getsize(log_file_path) > 0:
            try:
                with open(log_file_path, 'r') as f:
                    data = json.load(f)

                if isinstance(data, list) and data:
                    sample_count = len(data)
                    sums = {metric: 0.0 for metric in desc_map.keys()}

                    for entry in data:
                        for metric in desc_map.keys():
                            try:
                                value = float(str(entry.get(metric, '0')).replace('%', '').strip())
                                sums[metric] += value
                            except ValueError:
                                continue

                    for metric, total in sums.items():
                        desc = f"{device_prefix} {desc_map[metric]}"
                        gpu_device_usage[desc] = round(total / sample_count, 2)
            except (json.JSONDecodeError, ValueError):
                pass

        return gpu_device_usage

    def return_blank(self):
        return {AVG_GPU_USAGE_CONSTANT: "NA"}

class QMASSAGPUUsageExtractor(KPIExtractor):
    #overriding abstract method
    def extract_data(self, log_file_path):
        print("parsing GPU usages")
        device = re.findall(r'\d+', os.path.basename(log_file_path))
        desc_map = {
            'CCS %': 'Compute[CCS] Utilization %',
            'RCS %': 'Render/3D[RCS] Utilization %',
            'VCS %': 'Video[VCS] Utilization %',
            'VECS %': 'VideoEnhance[VECS] Utilization %',            
            'BCS %': 'Blitter Copy Engine %',
            'Power Usage (W)': 'GPU Power (W)'
       }

        device_prefix = f"GPU_{device[0]}"
        gpu_device_usage = {
            f"{device_prefix} {desc}": 'NA' for desc in desc_map.values()
        }

        if os.path.isfile(log_file_path) and os.path.getsize(log_file_path) > 0:
            try:
                with open(log_file_path, 'r') as f:
                    data = json.load(f)

                if isinstance(data, list) and data:
                    sample_count = len(data)
                    sums = {metric: 0.0 for metric in desc_map.keys()}

                    for entry in data:
                        for metric in desc_map.keys():
                            try:
                                value = float(str(entry.get(metric, '0')).replace('%', '').strip())
                                sums[metric] += value
                            except ValueError:
                                continue

                    for metric, total in sums.items():
                        desc = f"{device_prefix} {desc_map[metric]}"
                        gpu_device_usage[desc] = round(total / sample_count, 2)
            except (json.JSONDecodeError, ValueError):
                pass

        return gpu_device_usage

    def return_blank(self):
        return {AVG_GPU_USAGE_CONSTANT: "NA"}

class XPUMUsageExtractor(KPIExtractor):
    #overriding abstract method
    def extract_data(self, log_file_path):
        print("parsing GPU usages")
        #print("log file path: {}".format(log_file_path))
        gpu_device_usage = {}
        gpu_device_mem_usage = {}
        gpu_device_compute_usage = {}
        gpu_encode_usage = {}
        gpu_decode_usage = {}
        device = re.findall(r'\d+', os.path.basename(log_file_path))
        #print("Device: {}".format(device))
        device_usage_key = "GPU_{} {}".format(device[0], AVG_GPU_USAGE_CONSTANT)
        device_mem_usage_key = "GPU_{} {}".format(device[0], AVG_GPU_MEM_USAGE_CONSTANT)
        device_compute_usage_key = "GPU_{} {}".format(device[0], AVG_GPU_COMPUTE_USAGE_CONSTANT)
        device_vdbox0_usage_key = "GPU_{} VDBOX0 {}".format(device[0], AVG_GPU_VDBOX_USAGE_CONSTANT)
        device_vdbox1_usage_key = "GPU_{} VDBOX1 {}".format(device[0], AVG_GPU_VDBOX_USAGE_CONSTANT)
        #print("{}".format(device_usage_key))
        with open(log_file_path) as f:
            gpu_samples = []
            mem_samples = []
            compute_samples = []
            encode0_samples = []
            encode1_samples = []
            decode0_samples = []
            decode1_samples = []
            data = json.load(f)
            for entry in data:
                try:
                  gpu_samples.append(float(entry[" GPU Utilization (%)"]))
                  mem_samples.append(float(entry[" GPU Memory Utilization (%)"]))
                  compute_samples.append(float(entry[" Compute Engine 0 (%)"]))
                  encode0_samples.append(float(entry[" Encoder Engine 0 (%)"]))
                  encode1_samples.append(float(entry[" Encoder Engine 1 (%)"]))
                  decode0_samples.append(float(entry[" Decoder Engine 0 (%)"]))
                  decode1_samples.append(float(entry[" Decoder Engine 1 (%)"]))
                except Exception:
                  # there might be some anomaly in xpu manager outputs when collecting metrics, eg. emptry strings
                  # here we ignore that formatting issue
                  pass # nosec
            entries = len(gpu_samples)
            #for index in range(entries):
               # print("sample: {}".format(gpu_samples[index]))

        if len(gpu_samples) > 0:
            gpu_device_usage[device_usage_key] = mean(gpu_samples)
            gpu_device_usage[device_mem_usage_key] = mean(mem_samples)
            gpu_device_usage[device_compute_usage_key] = mean(compute_samples)
            gpu_device_usage[device_vdbox0_usage_key] = (mean(encode0_samples) + mean(decode0_samples))
            gpu_device_usage[device_vdbox1_usage_key] = (mean(encode1_samples) + mean(decode1_samples))
        if gpu_device_usage:
            return gpu_device_usage
        else:
            return {AVG_GPU_USAGE_CONSTANT: "NA"}

    def return_blank(self):
        return {AVG_GPU_USAGE_CONSTANT: "NA"}


class MetaExtractor(KPIExtractor):
    _TEXT_PATTERN = "Total Text count:"
    _BARCODE_PATTERN = "Total Barcode count:"
    #overriding abstract method
    def extract_data(self, log_file_path):
        if os.path.getsize(log_file_path) == 0:
            return {TEXT_COUNT_CONSTANT: "NA", BARCODE_COUNT_CONSTANT: "NA"}

        print("parsing text and barcode data")
        #text_count = 0
        #barcode_count = 0
        with open(log_file_path) as f:
            for line in f:
                if self._TEXT_PATTERN in line:
                    print("got text pattern")
                    text_count = line.split(":", 1)
                    text_count = int(text_count[1])
                    #print("text count: {}".format(text_count))
                elif self._BARCODE_PATTERN in line:
                    print("got barcode pattern")
                    barcode_count = line.split(":", 1)
                    barcode_count = int(barcode_count[1])
                    #print("barcode count: {}".format(barcode_count))

        if 'text_count' in locals() and 'barcode_count' in locals():
            return {TEXT_COUNT_CONSTANT: text_count, BARCODE_COUNT_CONSTANT: barcode_count}
        else:
            return {TEXT_COUNT_CONSTANT: "NA", BARCODE_COUNT_CONSTANT: "NA"}

    def return_blank(self):
        return {TEXT_COUNT_CONSTANT: "NA", BARCODE_COUNT_CONSTANT: "NA"}

class MemUsageExtractor(KPIExtractor):
    _MEM_USAGE_PATTERN = "Mem:\\s+(\\d+)\\s+(\\d+)\\s+(\\d+)\\s+(\\d+)\\s+(\\d+)\\s+(\\d+)"
    _MEM_TOTAL_GROUP = 1
    _MEM_USED_GROUP = 2

    #overriding abstract method
    def extract_data(self, log_file_path):
        if os.path.getsize(log_file_path) == 0:
            return {AVG_MEM_USAGE_CONSTANT: "NA"}

        print("parsing memory usage")
        mem_usages = []
        mem_usage_p = re.compile(self._MEM_USAGE_PATTERN)
        with open(log_file_path) as f:
            for line in f:
                mem_usage_m = mem_usage_p.match(line)
                if mem_usage_m:
                    mem_usage = float(mem_usage_m.group(self._MEM_USED_GROUP)) / float(mem_usage_m.group(self._MEM_TOTAL_GROUP))
                    mem_usages.append(mem_usage)

        if len(mem_usages) > 0:
            return {AVG_MEM_USAGE_CONSTANT: round(mean(mem_usages) * 100, 2)}
        else:
            return {AVG_MEM_USAGE_CONSTANT: "NA"}

    def return_blank(self):
        return {AVG_MEM_USAGE_CONSTANT: "NA"}

class PowerUsageExtractor(KPIExtractor):
    #overriding abstract method
    _POWER_USAGE_PATTERN = "(\\w+);.Consumed.energy.units:.(\\d+).+Joules: (\\d+.\\d+).+Watts:.(\\d+.\\d+).+TjMax:.(\\d+)"
    _SOCKET_ID_GROUP = 1
    _POWER_USAGE_GROUP = 4
    def extract_data(self, log_file_path):
        if os.path.getsize(log_file_path) == 0:
            return {AVG_POWER_USAGE_CONSTANT: "NA"}

        power_dict = defaultdict(list)
        power_usage_p = re.compile(self._POWER_USAGE_PATTERN)
        print("parsing power usage")
        with open(log_file_path) as f:
            for line in f:
                power_usage_m = power_usage_p.match(line)
                if power_usage_m:
                    socket_id = power_usage_m.group(self._SOCKET_ID_GROUP)
                    power_usage = float(power_usage_m.group(self._POWER_USAGE_GROUP))
                    power_dict[socket_id].append(power_usage)

        power_kpi_dict = {}
        for socket_id, power_usages in power_dict.items():
            socket_key = "{} {}".format(socket_id, AVG_POWER_USAGE_CONSTANT)
            power_kpi_dict[socket_key] = round(mean(power_usages), 2)

        if power_kpi_dict:
            return power_kpi_dict
        else:
            return {AVG_POWER_USAGE_CONSTANT: "NA"}

    def return_blank(self):
        return {AVG_POWER_USAGE_CONSTANT: "NA"}

class DiskBandwidthExtractor(KPIExtractor):
    _DISK_BANDWIDTH_PATTERN = "Total DISK READ:.+\\s(\\d+.\\d+).(B\\/s|K\\/s).+\\s(\\d+.\\d+).(B\\/s|K\\/s)"
    _READ_BYTES_PER_SECOND_GROUP = 1
    _READ_BYTES_UNITS_GROUP = 2
    _WRITE_BYTES_PER_SECOND_GROUP = 3
    _WRITE_BYTES_UNITS_GROUP = 4

    #overriding abstract method
    def extract_data(self, log_file_path):
        print("parsing disk bandwidth")
        disk_read_bytes_per_second = []
        disk_write_bytes_per_second = []
        disk_bandwidth_p = re.compile(self._DISK_BANDWIDTH_PATTERN)
        with open(log_file_path) as f:
            for line in f:
                disk_bandwidth_m = disk_bandwidth_p.match(line)
                if disk_bandwidth_m:
                    # we want the data in Bytes first before finally converting to MegaBytes
                    unit_multiplier = 1 if 'B' in disk_bandwidth_m.group(self._READ_BYTES_UNITS_GROUP) else 1000
                    read_bytes_per_second = float(disk_bandwidth_m.group(self._READ_BYTES_PER_SECOND_GROUP)) * unit_multiplier

                    unit_multiplier = 1 if 'B' in disk_bandwidth_m.group(self._WRITE_BYTES_PER_SECOND_GROUP) else 1000
                    write_bytes_per_second = float(disk_bandwidth_m.group(self._WRITE_BYTES_PER_SECOND_GROUP)) * unit_multiplier

                    disk_read_bytes_per_second.append(read_bytes_per_second)
                    disk_write_bytes_per_second.append(write_bytes_per_second)

        megabytes_to_bytes = 1000000
        if len(disk_read_bytes_per_second) > 0 and len(disk_write_bytes_per_second) > 0:
            return {AVG_DISK_READ_BANDWIDTH_CONSTANT: round(mean(disk_read_bytes_per_second) / megabytes_to_bytes, 2), 
                    AVG_DISK_WRITE_BANDWIDTH_CONSTANT: round(mean(disk_write_bytes_per_second) / megabytes_to_bytes, 2)}
        else:
            {AVG_DISK_READ_BANDWIDTH_CONSTANT: "NA", AVG_DISK_WRITE_BANDWIDTH_CONSTANT: "NA"}

    def return_blank(self):
        return {AVG_DISK_READ_BANDWIDTH_CONSTANT: "NA", AVG_DISK_WRITE_BANDWIDTH_CONSTANT: "NA"}

class MemBandwidthExtractor(KPIExtractor):
    #overriding abstract method
    def extract_data(self, log_file_path):
        if os.path.getsize(log_file_path) == 0:
            return {AVG_MEM_BANDWIDTH_CONSTANT: "NA"}

        print("parsing memory bandwidth")
        socket_memory_bandwidth = {}
        df = pd.read_csv(log_file_path, header=1, on_bad_lines='skip')
        socket_count = 0
        for column in df.columns:
            if 'Memory (MB/s)' in column:
                socket_key = "S{} {}".format(socket_count, AVG_MEM_BANDWIDTH_CONSTANT)
                mem_bandwidth = df[column].tolist()
                socket_memory_bandwidth[socket_key] = round(mean([ x for x in mem_bandwidth if pd.isna(x) == False ]), 2)
                socket_count = socket_count + 1

        if socket_memory_bandwidth:
            return socket_memory_bandwidth
        else:
            return {AVG_MEM_BANDWIDTH_CONSTANT: "NA"}

    def return_blank(self):
        return {AVG_MEM_BANDWIDTH_CONSTANT: "NA"}


class PIPELINEFPSExtractor(KPIExtractor):
    """Time-weighted FPS over the whole pipeline_stream log, matching the
    stream_density rate (sum(fps * seconds) / sum(seconds))."""

    #overriding abstract method
    def extract_data(self, log_file_path):
        print("parsing fps")
        weighted_frames = 0.0
        total_seconds = 0.0
        camera_fps = {}
        cam = re.findall(r'\d+', os.path.basename(log_file_path))
        camera_key = "Camera_{} {}".format(cam[0], AVG_FPS_CONSTANT)
        with open(log_file_path) as f:
            for row in csv.reader(f):
                if len(row) < 2:
                    continue
                try:
                    fps = float(row[0])
                    seconds = float(row[1])
                except ValueError:
                    continue
                if not (np.isfinite(fps) and np.isfinite(seconds)) \
                        or fps < 0 or seconds <= 0:
                    continue
                weighted_frames += fps * seconds
                total_seconds += seconds

        if total_seconds > 0:
            camera_fps[camera_key] = round(weighted_frames / total_seconds, 2)
        else:
            camera_fps[camera_key] = "NA"

        return camera_fps

    def return_blank(self):
        return {AVG_FPS_CONSTANT: "NA"}

class FPSExtractor(KPIExtractor):
    _FPS_KEYWORD = "avg_fps"

    #overriding abstract method
    def extract_data(self, log_file_path):
        print("parsing fps")
        average_fps_list = []
        camera_fps = {}
        cam = re.findall(r'\d+', os.path.basename(log_file_path))
        camera_key = "Camera_{} {}".format(cam[0], AVG_FPS_CONSTANT)
        with open(log_file_path) as f:
            for line in f:
                if self._FPS_KEYWORD in line:
                    average_fps_list.append(float((line.split(":"))[1].replace(",", "")))

        if len(average_fps_list) > 0:
            camera_fps[camera_key] = round(mean(average_fps_list), 2)

        if camera_fps:
            return camera_fps
        else:
            return {AVG_FPS_CONSTANT: "NA"}

    def return_blank(self):
        return {AVG_FPS_CONSTANT: "NA"}

class PipelineLatencyExtractor(KPIExtractor):
    """Collect whole-log per-frame latency samples from a gst-launch log.

    The samples are stored in ``self.record`` in the same shape as a
    stream-density latency record, so both are summarised by
    build_window_latency_metrics().
    """

    #overriding abstract method
    def extract_data(self, log_file_path):
        print("parsing latency")
        # {"source -> sink": {frame_number: frame_latency_value}}
        samples_by_channel = {}
        with open(log_file_path) as f:
            for line in f:
                if "latency_tracer_pipeline" not in line:
                    continue
                match = re.search(r'frame_latency=\(double\)([0-9]*\.?[0-9]+)', line)
                source_name_match = re.search(r'source_name=\(string\)([^,\s]+)', line)
                sink_name_match = re.search(r'sink_name=\(string\)([^,\s]+)', line)
                frame_number_match = re.search(r'frame_num=\(uint\)(\d+)', line)
                if not (match and source_name_match and sink_name_match
                        and frame_number_match):
                    continue
                channel = (f"{source_name_match.group(1)} -> "
                           f"{sink_name_match.group(1)}")
                samples_by_channel.setdefault(channel, {})[
                    int(frame_number_match.group(1))] = float(match.group(1))

        container = re.sub(r'^gst-launch_?', '',
                           os.path.splitext(os.path.basename(log_file_path))[0])
        self.record = {
            "container_name": container or "unknown",
            "samples_by_stream": {
                channel: list(frames.values())
                for channel, frames in samples_by_channel.items() if frames
            },
        }
        return {}

    def return_blank(self):
        return {"LATENCY": "NA"}

class StreamDensityExtractor(KPIExtractor):
    """Extract stream density benchmark results from JSON files."""
    def extract_data(self, log_file_path):
        try:
            with open(log_file_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}

        results = {}
        results["Stream Density Target Latency (ms)"] = data.get("target_latency_ms", "NA")
        results["Stream Density Max Scenes"] = data.get("max_scenes", "NA")
        results["Stream Density Met Target"] = data.get("met_target", "NA")

        if data.get("best_iteration"):
            best = data["best_iteration"]
            results["Stream Density Best Latency (ms)"] = best.get("latency_ms", "NA")
            results["Stream Density Best Scenes"] = best.get("num_scenes", "NA")

        iterations = data.get("iterations", [])
        for it in iterations:
            n = it.get("num_scenes", "?")
            prefix = f"Stream Density {n} Scene(s)"
            results[f"{prefix} Latency (ms)"] = it.get("latency_ms", "NA")
            results[f"{prefix} Throughput Ratio"] = it.get("throughput_ratio", "NA")
            results[f"{prefix} Actual Samples"] = it.get("actual_samples", "NA")
            results[f"{prefix} Expected Samples"] = it.get("expected_samples", "NA")
            results[f"{prefix} Samples/Scene"] = it.get("samples_per_scene", "NA")
            results[f"{prefix} Passed"] = it.get("passed", "NA")
            results[f"{prefix} Memory %"] = it.get("memory_percent", "NA")
            results[f"{prefix} CPU %"] = it.get("cpu_percent", "NA")

        return results

    def return_blank(self):
        return {"Stream Density Max Scenes": "-"}
        
class PCMExtractor(KPIExtractor):
    #overriding abstract method
    def extract_data(self, log_file_path):
        if os.path.getsize(log_file_path) == 0:
            return {AVG_POWER_USAGE_CONSTANT: "NA", AVG_MEM_BANDWIDTH_CONSTANT: "NA"}

        socket_memory_and_power = {}
        print("parsing memory bandwidth")
        df = pd.read_csv(log_file_path, header=1, on_bad_lines='skip')
        socket_count = 0
        for column in df.columns:
            if 'READ' in column:
                mem_read = df[column].tolist()
            elif 'WRITE' in column:
                mem_write = df[column].tolist()
                mem_bandwidth = list(map(add, mem_read, mem_write))
                numeric_values = []
                for x in mem_bandwidth:
                    if pd.isna(x) == False:
                        try:
                            numeric_values.append(float(x))
                        except (ValueError, TypeError):
                            # Skip values that can't be converted to float
                            pass
                if numeric_values:  # Make sure we have values to calculate mean
                    socket_key = "S{} {}".format(socket_count, AVG_MEM_BANDWIDTH_CONSTANT)
                    socket_memory_and_power[socket_key] = round(1000 * mean(numeric_values), 2)
                socket_count = socket_count + 1

        print("parsing power usage")
        df = pd.read_csv(log_file_path, on_bad_lines='skip')
        socket_power_usage = {}
        socket_count = 0
        for column in df.columns:
            if 'Proc Energy (Joules)' in column:
                power_usage = df[column].tolist()
                del power_usage[0]
                numeric_values = []
                for x in power_usage:
                    if pd.isna(x) == False:
                        try:
                            numeric_values.append(float(x))
                        except (ValueError, TypeError):
                            # Skip values that can't be converted to float
                            pass
                if numeric_values:  # Make sure we have values to calculate mean
                    socket_key = "S{} {}".format(socket_count, AVG_POWER_USAGE_CONSTANT)
                    socket_memory_and_power[socket_key] = round(mean(numeric_values), 2)
                socket_count = socket_count + 1

        if socket_memory_and_power:
            return socket_memory_and_power
        else:
            return {AVG_POWER_USAGE_CONSTANT: "NA", AVG_MEM_BANDWIDTH_CONSTANT: "NA"}

    def return_blank(self):
        return {AVG_POWER_USAGE_CONSTANT: "-", AVG_MEM_BANDWIDTH_CONSTANT: "-"}

PIPELINE_FPS_LOG_PATTERN = r"^pipeline.*\.log$"

KPIExtractor_OPTION = {"meta_summary.txt":MetaExtractor,
                       "camera":FPSExtractor,
                       PIPELINE_FPS_LOG_PATTERN:PIPELINEFPSExtractor,
                       "gst-launch":PipelineLatencyExtractor,
                       "cpu_usage.log":CPUUsageExtractor,
                       "npu_usage.csv":NPUUsageExtractor,
                       "memory_usage.log":MemUsageExtractor, 
                       "memory_bandwidth.csv":MemBandwidthExtractor,
                       "disk_bandwidth.log":DiskBandwidthExtractor,
                       "power_usage.log":PowerUsageExtractor,
                       "pcm.csv":PCMExtractor,
                       r"(?:^xpum).*\.json$": XPUMUsageExtractor,
                       r"^qmassa.*parsed.*\.json$": QMASSAGPUUsageExtractor,
                       r"^vlm_application_metrics.*\.txt$": VLMAppMetricsExtractor,
                       r"^vlm_performance_metrics.*\.txt$": VLMPerformanceMetricsExtractor,
                       r"^(?:swlp|poi)_stream_density.*\.json$": StreamDensityExtractor}

def add_parser():
    parser = argparse.ArgumentParser(description='Consolidate data')
    parser.add_argument('--root_directory', nargs=1, help='Root directory that consists all log directory that store log file', required=True)
    parser.add_argument('--output', nargs=1, help='Output file to store consolidate data', required=True)
    return parser


# Calculated for internal use but not written to the summary CSV.
CSV_EXCLUDED_METRICS = {"Per-stream Latency Sample Count"}


def load_window_latency_records(root_directory):
    """Load latency samples recorded for final confirming stream-density windows."""
    records = []
    sample_filename = "stream_density_latency_samples.jsonl"
    for sample_path in pathlib.Path(root_directory).rglob(sample_filename):
        with open(sample_path, "r", encoding="utf-8") as sample_file:
            for line_number, line in enumerate(sample_file, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    print(
                        f"WARN: Invalid latency sample record at "
                        f"{sample_path}:{line_number}: {error}")
                    continue
                if not isinstance(record, dict) or not isinstance(
                        record.get("samples_by_stream"), dict):
                    print(
                        f"WARN: Ignoring malformed latency sample record at "
                        f"{sample_path}:{line_number}")
                    continue
                records.append(record)
    return records


def build_window_fps_metrics(records):
    """Build Camera_N FPS rows from final-window time-weighted stream FPS."""
    containers = {record.get("container_name", "unknown") for record in records}
    metric_values = {}
    for record in records:
        container = record.get("container_name", "unknown")
        for stream_name, raw_fps in record.get("stream_fps", {}).items():
            try:
                fps = float(raw_fps)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(fps):
                continue
            stream_index = re.search(r"\d+", str(stream_name))
            camera_id = stream_index.group(0) if stream_index else stream_name
            key = f"Camera_{camera_id} {AVG_FPS_CONSTANT}"
            if len(containers) > 1:
                key = f"{key} ({container})"
            metric_values[key] = round(fps, 2)
    return metric_values


def build_window_latency_metrics(records):
    """Build per-stream and pooled-system stats with linear percentiles.

    Records come from the final confirming stream-density window, or from
    whole gst-launch logs when no window was saved. Each stream's samples
    are merged into one set before its statistics are calculated.
    """
    per_stream_samples = defaultdict(list)
    all_samples = []

    for record in records:
        container = record.get("container_name", "unknown")
        for stream_name, raw_values in record["samples_by_stream"].items():
            if not isinstance(raw_values, list):
                continue
            values = []
            for raw_value in raw_values:
                try:
                    value = float(raw_value)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(value) and value >= 0:
                    values.append(value)
            if not values:
                continue
            stream_key = f"{container} | {stream_name}"
            per_stream_samples[stream_key].extend(values)
            all_samples.extend(values)

    if not all_samples:
        return {}

    def percentile(values, p):
        return float(np.percentile(values, p, method="linear"))

    metric_values = {
        "Per-stream Latency p50 (ms)": {},
        "Per-stream Latency p95 (ms)": {},
        "Per-stream Latency p99 (ms)": {},
        "Per-stream Latency Average (ms)": {},
        "Per-stream Latency Stddev (ms)": {},
        "Per-stream Latency Sample Count": {},
    }
    for stream_key, values in sorted(per_stream_samples.items()):
        metric_values["Per-stream Latency p50 (ms)"][stream_key] = round(
            percentile(values, 50), 3)
        metric_values["Per-stream Latency p95 (ms)"][stream_key] = round(
            percentile(values, 95), 3)
        metric_values["Per-stream Latency p99 (ms)"][stream_key] = round(
            percentile(values, 99), 3)
        metric_values["Per-stream Latency Average (ms)"][stream_key] = round(
            mean(values), 3)
        metric_values["Per-stream Latency Stddev (ms)"][stream_key] = round(
            stdev(values) if len(values) > 1 else 0.0, 3)
        metric_values["Per-stream Latency Sample Count"][stream_key] = len(
            values)

    metric_values["System Latency p50 (ms)"] = round(
        percentile(all_samples, 50), 3)
    metric_values["System Latency p95 (ms)"] = round(
        percentile(all_samples, 95), 3)
    metric_values["System Latency p99 (ms)"] = round(
        percentile(all_samples, 99), 3)
    metric_values["System Latency Sample Count"] = len(all_samples)
    return metric_values


if __name__ == '__main__':
    parser = add_parser()
    args = vars(parser.parse_args())

    root_directory = args['root_directory'][0]
    output = args['output'][0]

    n = 0
    df = pd.DataFrame()
    full_kpi_dict = {}
    # Final confirming stream-density window; empty when nothing was saved.
    window_records = load_window_latency_records(root_directory)
    whole_log_latency_records = []

    for kpiExtractor in KPIExtractor_OPTION:
        # When the final window was saved, it replaces the whole-log FPS
        # and latency so metrics.csv matches stream_density.log.
        if window_records and kpiExtractor in (
                "gst-launch", PIPELINE_FPS_LOG_PATTERN):
            continue
        fileFound = False
        for dirpath, dirname, filename in os.walk(root_directory):
            for file in filename:
                if re.search(kpiExtractor, file):
                    fileFound = True
                    extractor = KPIExtractor_OPTION.get(kpiExtractor)()
                    kpi_dict = extractor.extract_data(
                        os.path.join(dirpath, file))
                    if isinstance(extractor, PipelineLatencyExtractor):
                        whole_log_latency_records.append(extractor.record)
                    if kpi_dict:
                        full_kpi_dict.update(kpi_dict)

    if window_records:
        full_kpi_dict.update(build_window_fps_metrics(window_records))
    full_kpi_dict.update(build_window_latency_metrics(
        window_records or whole_log_latency_records))
    # Write out summary csv file from dictionary
    with open(output, 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in full_kpi_dict.items():
            if key in CSV_EXCLUDED_METRICS:
                continue
            if isinstance(value, dict):
                # Format dictionary values without curly braces
                formatted_value = ', '.join([f'{k}: {v}' for k, v in value.items()])
                writer.writerow([key, formatted_value])
            else:
                writer.writerow([key, value])
