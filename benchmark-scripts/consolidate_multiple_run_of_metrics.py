'''
* Copyright (C) 2023 Intel Corporation.
*
* SPDX-License-Identifier: Apache-2.0
'''

import pathlib
import datetime
import argparse
from abc import ABC, abstractmethod
from statistics import mean
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
LAST_MODIFIED_LOG = "Last log update"
TEXT_COUNT_CONSTANT = "Total Text count"
BARCODE_COUNT_CONSTANT = "Total Barcode count"

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
            return {AVG_CPU_USAGE_CONSTANT: mean(cpu_usages)}
        else:
            return {AVG_CPU_USAGE_CONSTANT: "NA"}

    def return_blank(self):
        return {AVG_CPU_USAGE_CONSTANT: "NA"}
        

class GPUUsageExtractor(KPIExtractor):
    _USAGE_PATTERN = "Render/3D/0"
    #overriding abstract method
    def extract_data(self, log_file_path):
        print("parsing GPU usages")
        device = re.findall(r'\d+', os.path.basename(log_file_path))
        gpu_device_usage = {}
        device_vdbox0_usage_key = "GPU_{} VDBOX0 {}".format(device[0], AVG_GPU_VDBOX_USAGE_CONSTANT)
        with open(log_file_path, 'r') as f:
            vdbox0_samples = []
            data = json.load(f)
        for entry in data:
            # extract gpu render device usage from RCS field
            vdbox0_samples.append(float(entry["RCS %"]))

        if len(vdbox0_samples) > 0:
            gpu_device_usage[device_vdbox0_usage_key] = mean(vdbox0_samples)

        if gpu_device_usage:
            return gpu_device_usage
        else:
            return {AVG_GPU_USAGE_CONSTANT: "NA"}

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
            return {AVG_MEM_USAGE_CONSTANT: mean(mem_usages) * 100}
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
            power_kpi_dict[socket_key] = mean(power_usages)

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
            return {AVG_DISK_READ_BANDWIDTH_CONSTANT: mean(disk_read_bytes_per_second) / megabytes_to_bytes, 
                    AVG_DISK_WRITE_BANDWIDTH_CONSTANT: mean(disk_write_bytes_per_second) / megabytes_to_bytes}
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
                socket_memory_bandwidth[socket_key] = mean([ x for x in mem_bandwidth if pd.isna(x) == False ])
                socket_count = socket_count + 1

        if socket_memory_bandwidth:
            return socket_memory_bandwidth
        else:
            return {AVG_MEM_BANDWIDTH_CONSTANT: "NA"}

    def return_blank(self):
        return {AVG_MEM_BANDWIDTH_CONSTANT: "NA"}


class PIPELINEFPSExtractor(KPIExtractor):
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
              average_fps_list.append(float(line))

        if len(average_fps_list) > 0:
            camera_fps[camera_key] = mean(average_fps_list)
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
            camera_fps[camera_key] = mean(average_fps_list)

        if camera_fps:
            return camera_fps
        else:
            return {AVG_FPS_CONSTANT: "NA"}

    def return_blank(self):
        return {AVG_FPS_CONSTANT: "NA"}

class PIPELINLastModifiedExtractor(KPIExtractor):
    #overriding abstract method
    def extract_data(self, log_file_path):
        print("parsing last modified log time")
        average_fps_list = []
        last_modified = {}
        cam = re.findall(r'\d+', os.path.basename(log_file_path))
        print(log_file_path)
        print(cam)
        camera_key = "Camera_{} {}".format(cam[0], LAST_MODIFIED_LOG)
        
        #get the last file modified time
        print(log_file_path)
        unix_date = os.path.getmtime(log_file_path) if os.path.exists(log_file_path) else None
        print(unix_date)
        #convert unix time to human readable date time
        formatted_date = datetime.datetime.fromtimestamp(unix_date) if not (unix_date is None) else None
        print(formatted_date)
        #convert date format to string
        last_modified[camera_key] = formatted_date.strftime('%m/%d/%Y %H:%M:%f') if not (formatted_date is None) else {LAST_MODIFIED_LOG: "NA"}
        print(last_modified[camera_key])

        return last_modified

    def return_blank(self):
        return {LAST_MODIFIED_LOG: "NA"}

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
                socket_key = "S{} {}".format(socket_count, AVG_MEM_BANDWIDTH_CONSTANT)
                socket_memory_and_power[socket_key] = 1000 * mean([ x for x in mem_bandwidth if pd.isna(x) == False ])
                socket_count = socket_count + 1

        print("parsing power usage")
        df = pd.read_csv(log_file_path, on_bad_lines='skip')
        socket_power_usage = {}
        socket_count = 0
        for column in df.columns:
            if 'Proc Energy (Joules)' in column:
                power_usage = df[column].tolist()
                del power_usage[0]
                socket_key = "S{} {}".format(socket_count, AVG_POWER_USAGE_CONSTANT)
                socket_memory_and_power[socket_key] = mean([ float(x) for x in power_usage if pd.isna(x) == False ])
                socket_count = socket_count + 1

        if socket_memory_and_power:
            return socket_memory_and_power
        else:
            return {AVG_POWER_USAGE_CONSTANT: "NA", AVG_MEM_BANDWIDTH_CONSTANT: "NA"}

    def return_blank(self):
        return {AVG_POWER_USAGE_CONSTANT: "-", AVG_MEM_BANDWIDTH_CONSTANT: "-"}

KPIExtractor_OPTION = {"meta_summary.txt":MetaExtractor,
                       "camera":FPSExtractor,
                       "pipeline":PIPELINEFPSExtractor,
                       r"(?:^r).*\.jsonl$": PIPELINLastModifiedExtractor,
                       "cpu_usage.log":CPUUsageExtractor, 
                       "memory_usage.log":MemUsageExtractor, 
                       "memory_bandwidth.csv":MemBandwidthExtractor,
                       "disk_bandwidth.log":DiskBandwidthExtractor,
                       "power_usage.log":PowerUsageExtractor,
                       "pcm.csv":PCMExtractor,
                       r"(?:^xpum).*\.json$": XPUMUsageExtractor,
                       r"(?:^igt).*\\.json": GPUUsageExtractor, }

def add_parser():
    parser = argparse.ArgumentParser(description='Consolidate data')
    parser.add_argument('--root_directory', nargs=1, help='Root directory that consists all log directory that store log file', required=True)
    parser.add_argument('--output', nargs=1, help='Output file to store consolidate data', required=True)
    return parser

if __name__ == '__main__':
    parser = add_parser()
    args = vars(parser.parse_args())

    root_directory = args['root_directory'][0]
    output = args['output'][0]

    n = 0
    df = pd.DataFrame()
    full_kpi_dict = {}
    for kpiExtractor in KPIExtractor_OPTION:
        fileFound = False
        for dirpath, dirname, filename in os.walk(root_directory):
            for file in filename:
                if re.search(kpiExtractor, file):
                    fileFound = True
                    extractor = KPIExtractor_OPTION.get(kpiExtractor)()
                    kpi_dict = extractor.extract_data(
                        os.path.join(root_directory, file))
                    if kpi_dict:
                        full_kpi_dict.update(kpi_dict)

    # Write out summary csv file from dictionary
    with open(output, 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in full_kpi_dict.items():
            writer.writerow([key, value])
