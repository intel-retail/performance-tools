'''
* Copyright (C) 2024 Intel Corporation.
*
* SPDX-License-Identifier: Apache-2.0
'''

import os
import subprocess
import time
import benchmark
import glob
import sys
import re
import statistics
import psutil
import json
import math
import csv

# Constants:
TARGET_FPS_KEY = "TARGET_FPS"
CONTAINER_NAME_KEY = "CONTAINER_NAME"
PIPELINE_INCR_KEY = "PIPELINE_INC"
INIT_DURATION_KEY = "INIT_DURATION"
RESULTS_DIR_KEY = "RESULTS_DIR"
DEFAULT_TARGET_FPS = 15
MAX_GUESS_INCREMENTS = 5
CONSECUTIVE_FAIL_WINDOWS_KEY = "CONSECUTIVE_FAIL_WINDOWS"
CONSECUTIVE_PASS_WINDOWS_KEY = "CONSECUTIVE_PASS_WINDOWS"
DEFAULT_CONSECUTIVE_FAIL_WINDOWS = 2
DEFAULT_CONSECUTIVE_PASS_WINDOWS = 2
PASS_TOLERANCE_RATIO_KEY = "PASS_TOLERANCE_RATIO"
DEFAULT_PASS_TOLERANCE_RATIO = 0.95
MEASUREMENT_WINDOW_SECONDS_KEY = "MEASUREMENT_WINDOW_SECONDS"
DEFAULT_MEASUREMENT_WINDOW_SECONDS = 100
MIN_SAMPLES_PER_STREAM = 100
MEASUREMENT_SAMPLE_POLL_SECONDS = 1
CAMERA_STREAM_KEY = "CAMERA_STREAM"


class ArgumentError(Exception):
    pass

def build_per_stream_target_fps(stream_fps_dict, default_target_fps, env_vars=None):
    """Resolve per-stream FPS targets.

    Precedence: TARGET_FPS from env_vars (applies to every camera) > per-camera
    targetFps > per-camera fps > default_target_fps fallback.
    """
    env_vars = env_vars if env_vars is not None else os.environ
    # Get camera config path
    camera_stream = env_vars.get(CAMERA_STREAM_KEY, "camera_to_workload.json")
    config_path = camera_stream if os.path.isabs(camera_stream) else os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "configs", camera_stream)
    )

    # Build unified stream index -> target FPS mapping (cached by config path + mtime)
    cache = getattr(build_per_stream_target_fps, "_camera_config_cache", {})
    
    # Create cache key that includes file modification time to detect file changes
    file_mtime = os.path.getmtime(config_path) if os.path.isfile(config_path) else 0
    cache_key = (config_path, file_mtime)
    
    cached = cache.get(cache_key)
    if cached is not None:
        stream_idx_to_target_fps, total_cameras = cached
    else:
        stream_idx_to_target_fps = {}
        total_cameras = 0
        if os.path.isfile(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    cameras = json.load(f).get("lane_config", {}).get("cameras", [])
                total_cameras = len(cameras)
                for idx, cam in enumerate(cameras):
                    if isinstance(cam, dict):
                        resolved_fps = None
                        try:
                            target_fps_value = float(cam.get("targetFps", 0))
                            if target_fps_value > 0:
                                resolved_fps = target_fps_value
                        except (TypeError, ValueError):
                            pass
                        if resolved_fps is None:
                            try:
                                fps_value = float(cam.get("fps", 0))
                                if fps_value > 0:
                                    resolved_fps = fps_value
                            except (TypeError, ValueError):
                                pass
                        if resolved_fps is not None:
                            stream_idx_to_target_fps[idx] = resolved_fps
                cache[cache_key] = (dict(stream_idx_to_target_fps), int(total_cameras))
                build_per_stream_target_fps._camera_config_cache = cache
            except (IOError, ValueError) as e:
                print(f"WARN: Failed to load camera config {config_path}: {e}")
                cache[cache_key] = ({}, 0)
                build_per_stream_target_fps._camera_config_cache = cache
        else:
            print(
                f"WARN: camera configuration not found at {config_path}. Using default target FPS for all streams."
            )
            cache[cache_key] = ({}, 0)
            build_per_stream_target_fps._camera_config_cache = cache
            
    # Compile regex once (if needed for stream name parsing)
    stream_pattern = re.compile(r"pipeline_stream(\d+)")
    per_stream_targets = {}

    # TARGET_FPS from env_vars, when set, overrides every camera's config value.
    forced_target_fps = None
    env_target_fps = str(env_vars.get(TARGET_FPS_KEY, "")).strip()
    if env_target_fps:
        try:
            candidate = float(env_target_fps)
            if candidate > 0:
                forced_target_fps = candidate
        except ValueError:
            pass

    for stream_name in stream_fps_dict:
        if forced_target_fps is not None:
            per_stream_targets[stream_name] = forced_target_fps
            continue
        match = stream_pattern.search(stream_name)
        if match and total_cameras > 0:
            camera_idx = int(match.group(1)) % total_cameras
            target_fps = stream_idx_to_target_fps.get(camera_idx, default_target_fps)
        else:
            target_fps = default_target_fps
        per_stream_targets[stream_name] = float(target_fps)
    
    if os.getenv("STREAM_DENSITY_DEBUG", "0") == "1":
        print(f"INFO: per-stream target FPS map: {per_stream_targets}")
    return per_stream_targets


def describe_target_fps_configuration(default_target_fps, env_vars=None):
    """Describe the configured target FPS values and their sources."""
    env_vars = env_vars if env_vars is not None else os.environ
    env_target_fps = str(env_vars.get(TARGET_FPS_KEY, "")).strip()
    if env_target_fps:
        try:
            forced_target_fps = float(env_target_fps)
            if forced_target_fps > 0:
                return f"TARGET_FPS environment override: {forced_target_fps:.2f} FPS for every stream"
        except ValueError:
            pass

    camera_stream = env_vars.get(CAMERA_STREAM_KEY, "camera_to_workload.json")
    config_path = camera_stream if os.path.isabs(camera_stream) else os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "configs", camera_stream)
    )
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cameras = json.load(f).get("lane_config", {}).get("cameras", [])
    except (IOError, ValueError):
        return f"default target FPS: {default_target_fps:.2f} FPS"

    descriptions = []
    for index, camera in enumerate(cameras):
        if not isinstance(camera, dict):
            continue
        camera_id = camera.get("camera_id", f"camera{index}")
        if camera.get("targetFps", 0):
            try:
                target_fps = float(camera["targetFps"])
                if target_fps > 0:
                    descriptions.append(f"{camera_id}={target_fps:.2f} FPS from targetFps")
                    continue
            except (TypeError, ValueError):
                pass
        if camera.get("fps", 0):
            try:
                target_fps = float(camera["fps"])
                if target_fps > 0:
                    descriptions.append(f"{camera_id}={target_fps:.2f} FPS from fps")
                    continue
            except (TypeError, ValueError):
                pass
        descriptions.append(f"{camera_id}={default_target_fps:.2f} FPS from default")

    if descriptions:
        return "per-camera target FPS configuration: " + ", ".join(descriptions)
    return f"default target FPS: {default_target_fps:.2f} FPS"

def build_per_stream_camera_meta(stream_fps_dict, env_vars=None):
    """Build per-stream camera_id and workload labels from camera config."""
    env_vars = env_vars if env_vars is not None else os.environ
    camera_stream = env_vars.get(CAMERA_STREAM_KEY, "camera_to_workload.json")
    config_path = camera_stream if os.path.isabs(camera_stream) else os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "configs", camera_stream)
    )

    cache = getattr(build_per_stream_camera_meta, "_camera_meta_cache", {})
    file_mtime = os.path.getmtime(config_path) if os.path.isfile(config_path) else 0
    cache_key = (config_path, file_mtime)

    cached = cache.get(cache_key)
    if cached is not None:
        idx_to_meta, total_cameras = cached
    else:
        idx_to_meta = {}
        total_cameras = 0
        if os.path.isfile(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    cameras = json.load(f).get("lane_config", {}).get("cameras", [])
                total_cameras = len(cameras)
                for idx, cam in enumerate(cameras):
                    if isinstance(cam, dict):
                        camera_id = cam.get("camera_id", f"camera{idx}")
                        workloads = cam.get("workloads", [])
                        workload = workloads[0] if isinstance(workloads, list) and workloads else "-"
                        idx_to_meta[idx] = {"camera": str(camera_id), "workload": str(workload)}
                cache[cache_key] = (dict(idx_to_meta), int(total_cameras))
                build_per_stream_camera_meta._camera_meta_cache = cache
            except (IOError, ValueError) as e:
                print(f"WARN: Failed to load camera config {config_path}: {e}")
                cache[cache_key] = ({}, 0)
                build_per_stream_camera_meta._camera_meta_cache = cache
        else:
            cache[cache_key] = ({}, 0)
            build_per_stream_camera_meta._camera_meta_cache = cache

    stream_pattern = re.compile(r"pipeline_stream(\d+)")
    per_stream_meta = {}
    for stream_name in stream_fps_dict:
        match = stream_pattern.search(stream_name)
        if match and total_cameras > 0:
            camera_idx = int(match.group(1)) % total_cameras
            meta = idx_to_meta.get(camera_idx, {"camera": stream_name, "workload": "-"})
        else:
            meta = {"camera": stream_name, "workload": "-"}
        per_stream_meta[stream_name] = meta
    return per_stream_meta

def get_mean_target_fps(stream_target_fps, fallback_target_fps):
    target_fps_values = [float(v) for v in stream_target_fps.values() if float(v) > 0]
    return statistics.mean(target_fps_values) if target_fps_values else float(fallback_target_fps)

def measure_pipeline_memory(env_vars, compose_files, results_dir, container_name):
    
    if env_vars.get("OOM_PROTECTION", "1")== "0":
        print("OOM protection is disabled. Skipping memory measurement.")
        return 0

    """
    Measures the memory usage (in MB) of a single pipeline instance.
    Returns the memory usage in MB.
    """

    # Clean up any previous logs and containers
    clean_up_pipeline_logs(results_dir)
    benchmark.docker_compose_containers(
        "down", compose_files=compose_files,
        compose_post_args="-t 30 --volumes --remove-orphans", env_vars=env_vars)
    time.sleep(5)

    # Measure available memory before starting the pipeline
    before_mem = psutil.virtual_memory().available

    # Start a single pipeline
    env_vars["PIPELINE_COUNT"] = "1"
    benchmark.docker_compose_containers(
        "up", compose_files=compose_files, compose_post_args="-d", env_vars=env_vars)
    print("Waiting for pipeline to stabilize...")
    time.sleep(10)  # Let it stabilize

    # Measure available memory after starting the pipeline
    after_mem = psutil.virtual_memory().available

    # Stop the pipeline
    benchmark.docker_compose_containers(
        "down", compose_files=compose_files,
        compose_post_args="-t 30 --volumes --remove-orphans", env_vars=env_vars)
    time.sleep(5)

    # Calculate usage in MB
    usage_bytes = before_mem - after_mem
    usage_mb = usage_bytes // (1024 * 1024)
    print(f"Measured memory usage for one pipeline: {usage_mb} MB")
    return usage_mb

def check_can_add_pipelines(increment, per_pipeline_mb, safety_buffer_mb=3072, env_vars=None):
    
    if env_vars and env_vars.get("OOM_PROTECTION", "1") == "0":
        print("OOM protection is disabled. Skipping memory check.")
        return True

    """
    Checks if the system has enough available memory to add 'increment' more pipelines.
    Args:
        increment: Number of new pipelines to add.
        per_pipeline_mb: Memory required per pipeline (in MB).
        safety_buffer_mb: Safety buffer in MB (default 3072 MB).
    Returns:
        True if enough memory is available, False otherwise.
    """

    available_mb = psutil.virtual_memory().available // (1024 * 1024)
    needed_mb = increment * per_pipeline_mb
    if available_mb < (needed_mb + safety_buffer_mb):
        print(
            f"Insufficient memory: {available_mb}MB available, "
            f"need {needed_mb}MB + {safety_buffer_mb}MB buffer"
        )
        return False
    
    if monitor_memory_pressure():
        print(
            f"Memory pressure detected. Not safe to add {increment} more pipelines."
        )
        return False
    
    print(
        f"Memory check passed: {available_mb}MB available for {increment} new pipelines"
    )
    return True

def monitor_memory_pressure(env_vars=None):

    if env_vars and env_vars.get("OOM_PROTECTION", "1") == "0":
        print("OOM protection is disabled. Skipping memory pressure monitoring.")
        return False

    """
    Monitors the system for memory pressure signals during execution.
    
    Checks for:
    1. High swap activity (thrashing indicates pressure)
    2. Memory pressure stall information (kernel 4.20+ with PSI enabled)
    
    Returns:
        True if memory pressure detected, False otherwise.
    """

    try:
        # Check swap activity (thrashing indicates pressure)
        vmstat_output = subprocess.run(['vmstat', '1', '2'], 
                                     capture_output=True, text=True, timeout=5)
        if vmstat_output.returncode == 0:
            lines = vmstat_output.stdout.strip().split('\n')
            if len(lines) >= 2:
                # Get the last line (most recent data)
                last_line = lines[-1].split()
                if len(last_line) >= 8:
                    try:
                        swap_in = int(last_line[6])   # si: swap pages read from disk
                        swap_out = int(last_line[7])  # so: swap pages written to disk
                        
                        if swap_in > 1000 or swap_out > 1000:
                            print(f"High swap activity detected: in={swap_in} out={swap_out}")
                            return True
                    except (ValueError, IndexError):
                        print("WARN: Could not parse vmstat output for swap activity")
        
        # Check if Memory Pressure Stall Information is available (kernel 4.20+)
        psi_memory_path = "/proc/pressure/memory"
        if os.path.exists(psi_memory_path):
            try:
                with open(psi_memory_path, 'r') as f:
                    content = f.read()
                    
                # Look for the "some" line which shows percentage of time processes are stalled
                for line in content.split('\n'):
                    if line.startswith('some'):
                        # Parse: some avg10=2.04 avg60=1.23 avg300=0.85 total=12345678
                        parts = line.split()
                        for part in parts:
                            if part.startswith('avg10='):
                                stall_avg10 = float(part.split('=')[1])
                                if stall_avg10 > 10.0:
                                    print(f"Memory pressure detected: {stall_avg10}% stall time")
                                    return True
                                break
            except (IOError, ValueError) as e:
                print(f"WARN: Could not read memory pressure info: {e}")
        
        return False
        
    except subprocess.TimeoutExpired:
        print("WARN: vmstat command timed out")
        return False
    except Exception as e:
        print(f"WARN: Error monitoring memory pressure: {e}")
        return False


def is_env_non_empty(env_vars, key):
    '''
    checks if the environment variable dict env_vars is not empty
    and the env key exists and the value of that is not empty
    Args:
        env_vars: dict of current environment variables
        key: the env key to the env dict
    Returns:
        boolean to indicate if the env with key is empty or not
    '''
    if not env_vars:
        return False
    if key in env_vars:
        if env_vars[key]:
            return True
        else:
            return False
    else:
        return False


def clean_up_pipeline_logs(results_dir):
    '''
    cleans up the pipeline log files under results_dir
    Args:
        results_dir: directory holding the benchmark results
    '''
    print('Cleaning logs')
    matching_files = glob.glob(os.path.join(results_dir, 'pipeline*_*.log')) \
        + glob.glob(os.path.join(results_dir, 'gst*_*.log')) \
        + glob.glob(os.path.join(results_dir, 'rs*_*.jsonl')) \
        + glob.glob(os.path.join(results_dir, 'qmassa*-*.json'))
    if len(matching_files) > 0:
        for log_file in matching_files:
            os.remove(log_file)
    else:
        print('INFO: no match files to clean up')


def check_non_empty_result_logs(num_pipelines, results_dir,
                                container_name, max_retries=5):
    '''
    checks the current non-empty pipeline log files with some
    retries upto max_retires if file not exists or empty
    Args:
        num_pipelines: number of currently running pipelines
        container_name: the name of the container to match in log files,
                        expected to be part of the filename pattern
                        after the underscore (_)
        results_dir: directory holding the benchmark results
        max_retries: maximum number of retires, default 5 retires
    '''
    retry = 0
    while True:
        if retry >= max_retries:
            raise ValueError(
                f"""ERROR: cannot find all pipeline log files
                    after max retries: {max_retries},
                    pipelines may have been failed...""")
        print("INFO: checking presence of all pipeline log files... " +
              "retry: {}".format(retry))
        matching_files = glob.glob(os.path.join(
            results_dir, f'pipeline*_{container_name}*.log'))
        matching_files = filter_logs_for_current_timestamp(matching_files)
        if len(matching_files) >= num_pipelines and all([
              os.path.isfile(file) and os.path.getsize(file) > 0
              for file in matching_files]):
            print(
                f'found all non-empty log files for container name '
                f'{container_name}')
            break
        else:
            # some log files still empty or not found, retry it
            print('still having some missing or empty log files')
            retry += 1
            time.sleep(1)


def get_latest_pipeline_logs(num_pipelines, pipeline_log_files):
    '''
    obtains a list of the latest pipeline log files based on
    the timestamps of the files and only returns num_pipelines
    files if number of pipeline log files is more than num_pipelines
    Args:
        num_pipelines: number of currently running pipelines
        pipeline_log_files: all matching pipeline log files
    Return:
        latest_files: number of num_pipelines files based on
        the timestamps of files if number of pipeline log files
        is more than num_pipelines; otherwise whatever the number
        of the matching files will be returned
    '''
    timestamp_files = [
        (file, os.path.getmtime(file)) for file in pipeline_log_files]
    # sort timestamp_file by time in descending order
    sorted_timestamp = sorted(
        timestamp_files, key=lambda x: x[1], reverse=True)
    latest_files = [
        file for file, mtime in sorted_timestamp[:num_pipelines]]
    return latest_files


def extract_fps_samples(file_path, start_offset=0, end_offset=None):
    """Extract (FPS, seconds) samples within an optional byte range."""
    with open(file_path, "r") as file:
        if start_offset:
            file.seek(start_offset)
        if end_offset is not None:
            lines = file.read(max(0, end_offset - start_offset)).splitlines()
        else:
            lines = file.readlines()

    samples = []
    for row in csv.reader(lines):
        if not row or row == ['fps', 'duration_seconds']:
            continue
        try:
            if len(row) != 2:
                continue
            fps = float(row[0])
            seconds = float(row[1])
            if not math.isfinite(fps) or not math.isfinite(seconds) or fps < 0 or seconds <= 0:
                continue
            samples.append((fps, seconds))
        except ValueError:
            print(f"DEBUG: Skipping invalid FPS row {row} in {file_path}")

    return samples


def extract_numeric_fps(file_path, start_offset=0, end_offset=None):
    """Extract FPS values from FPS/duration rows."""
    return [fps for fps, _ in extract_fps_samples(file_path, start_offset, end_offset)]


def filter_logs_for_current_timestamp(log_files):
    '''
    keep only log files for the current benchmark run when TIMESTAMP
    is present in the environment.
    Args:
        log_files: candidate log files from glob matching
    Returns:
        timestamp-filtered files if possible, otherwise original list
    '''
    if not log_files:
        return log_files

    current_timestamp = os.getenv("TIMESTAMP", "").strip()
    if current_timestamp:
        filtered_files = [
            file for file in log_files
            if current_timestamp in os.path.basename(file)
        ]
        if filtered_files:
            print(
                f"DEBUG: TIMESTAMP={current_timestamp}, "
                f"filtered {len(filtered_files)}/{len(log_files)} files"
            )
            return filtered_files
        print(
            f"WARN: TIMESTAMP={current_timestamp} did not match any files; "
            f"attempting filename-based run detection"
        )

    # Fallback for cases where TIMESTAMP env is not propagated:
    # infer the latest run token from filenames (14-20 contiguous digits).
    token_pattern = re.compile(r'(?<!\d)(\d{14,20})(?!\d)')
    token_to_latest_mtime = {}
    for file in log_files:
        basename = os.path.basename(file)
        matches = token_pattern.findall(basename)
        if not matches:
            continue
        mtime = os.path.getmtime(file)
        for token in matches:
            if token not in token_to_latest_mtime or mtime > token_to_latest_mtime[token]:
                token_to_latest_mtime[token] = mtime

    if not token_to_latest_mtime:
        print("WARN: No run token detected in filenames; using all matching files")
        return log_files

    inferred_token = max(token_to_latest_mtime.items(), key=lambda item: item[1])[0]
    filtered_files = [
        file for file in log_files
        if inferred_token in os.path.basename(file)
    ]

    if filtered_files:
        print(
            f"DEBUG: Inferred run token={inferred_token}, "
            f"filtered {len(filtered_files)}/{len(log_files)} files"
        )
        return filtered_files

    print(
        f"WARN: Inferred run token={inferred_token} produced no matches; "
        f"using all matching files"
    )
    return log_files

def calculate_pipeline_latency(num_pipelines, results_dir, container_name):
    total_pipeline_latency = 0.0
    total_pipeline_latency_per_stream = 0.0
    container_timestamp = os.getenv("TIMESTAMP")
    matching_files = glob.glob(os.path.join(
        results_dir, f'gst-launch*_{container_name}*.log'))
    matching_files = filter_logs_for_current_timestamp(matching_files)
    print(f"DEBUG: {container_name} {container_timestamp}  num. of gst launch matching_files = {len(matching_files)}")
    latest_latency_logs = get_latest_pipeline_logs(
        num_pipelines, matching_files)
    
    pipeline_count = 0
    for latency_file in latest_latency_logs:
        pipeline_latency = 0.0
        try:
            with open(latency_file) as f:
                last_latency_line = None
                chunk_size = 8192  # Read in 8KB chunks
                buffer = ""
                
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    
                    buffer += chunk
                    lines = buffer.split('\n')
                    buffer = lines[-1]  # Keep incomplete line in buffer
                    
                    # Process complete lines
                    for line in lines[:-1]:
                        if "latency_tracer_pipeline" in line:
                            last_latency_line = line
                
                # Process any remaining content in buffer
                if buffer and "latency_tracer_pipeline" in buffer:
                    last_latency_line = buffer
                
                if last_latency_line:
                    match = re.search(r'avg=\(double\)([0-9]*\.?[0-9]+)', last_latency_line)
                    if match:
                        pipeline_latency = float(match.group(1))
            
            if pipeline_latency > 0:
                total_pipeline_latency += pipeline_latency
                pipeline_count += 1
                print(f"DEBUG: Added latency {pipeline_latency} from {latency_file}")
                
        except (IOError, ValueError) as e:
            print(f"WARN: Error processing {latency_file}: {e}")
            continue
    
    if pipeline_count > 0:
        total_pipeline_latency_per_stream = total_pipeline_latency / pipeline_count
    
    print(f"DEBUG: Total latency: {total_pipeline_latency}, Per stream: {total_pipeline_latency_per_stream}")
    return total_pipeline_latency, total_pipeline_latency_per_stream
    
def validate_and_setup_env(env_vars, target_fps_list):
    '''
    Validates and sets up the environment variables needed for
    running stream density.
    Args:
        env_vars: dict of current environment variables
        target_fps_list: list of target FPS values for stream density
    '''
    if not is_env_non_empty(env_vars, RESULTS_DIR_KEY):
        raise ArgumentError('ERROR: missing ' +
                            RESULTS_DIR_KEY + 'in env')

    # Set default values if missing
    if not target_fps_list:
        target_fps_list.append(DEFAULT_TARGET_FPS)
    elif any(float(fps) <= 0.0 for fps in target_fps_list):
        raise ArgumentError(
            'ERROR: stream density target fps ' +
            'should be greater than 0')

    if is_env_non_empty(env_vars, PIPELINE_INCR_KEY) and int(
            env_vars[PIPELINE_INCR_KEY]) <= 0:
        raise ArgumentError(
            'ERROR: stream density increments ' +
            'should be greater than 0')

    if is_env_non_empty(env_vars, CONSECUTIVE_FAIL_WINDOWS_KEY) and int(
            env_vars[CONSECUTIVE_FAIL_WINDOWS_KEY]) <= 0:
        raise ArgumentError(
            'ERROR: consecutive fail intervals should be greater than 0')

    if is_env_non_empty(env_vars, CONSECUTIVE_PASS_WINDOWS_KEY) and int(
            env_vars[CONSECUTIVE_PASS_WINDOWS_KEY]) <= 0:
        raise ArgumentError(
            'ERROR: run acceptance criterion (consecutive passes) should be greater than 0')

    if is_env_non_empty(env_vars, MEASUREMENT_WINDOW_SECONDS_KEY) and int(
            env_vars[MEASUREMENT_WINDOW_SECONDS_KEY]) <= 0:
        raise ArgumentError(
            'ERROR: measurement interval seconds should be greater than 0')

    if is_env_non_empty(env_vars, PASS_TOLERANCE_RATIO_KEY):
        pass_tolerance_ratio = float(env_vars[PASS_TOLERANCE_RATIO_KEY])
        if pass_tolerance_ratio <= 0.0 or pass_tolerance_ratio > 1.0:
            raise ArgumentError(
                'ERROR: pass tolerance ratio should be in (0, 1]')

    if not is_env_non_empty(env_vars, INIT_DURATION_KEY):
        env_vars[INIT_DURATION_KEY] = "10"


def describe_run_acceptance_criterion(consecutive_pass_windows):
    passes = "two" if consecutive_pass_windows == 2 else str(consecutive_pass_windows)
    return f"run acceptance criterion ({passes} consecutive passes)"


def count_valid_streams(stream_fps_dict):
    """Return the number of concurrent streams with valid FPS data."""
    # pipeline.sh is generated with every lane's sources, so indices already span all lanes.
    return sum(
        1 for measured in stream_fps_dict.values() if float(measured) > 0
    )


def print_stream_density_report(num_pipelines, stream_fps_dict,
                                stream_target_fps, pass_thresholds,
                                measurement_window_seconds,
                                consecutive_pass_windows,
                                consecutive_fail_windows,
                                pass_tolerance_ratio, stream_meta=None,
                                init_duration_seconds=None,
                                stream_samples=None,
                                actual_measurement_seconds=None,
                                sample_counts=None):
    """Print the per-stream stream density result summary."""
    stream_meta = stream_meta or {}
    stream_samples = stream_samples or {}
    binding_name = None
    binding_headroom = None
    for name in stream_fps_dict:
        pass_mark = pass_thresholds.get(name, 0.0)
        measured = stream_fps_dict.get(name, 0.0)
        headroom = ((measured - pass_mark) / pass_mark
                    if pass_mark > 0 else float('inf'))
        if binding_headroom is None or headroom < binding_headroom:
            binding_headroom = headroom
            binding_name = name

    overall_pass = bool(stream_fps_dict) and all(
        stream_fps_dict[name] >= pass_thresholds.get(name, 0.0)
        for name in stream_fps_dict
    )

    print("")
    print("Stream density result")
    print("-" * 47)
    camera_stream_count = count_valid_streams(stream_fps_dict)
    acceptance_criterion = describe_run_acceptance_criterion(consecutive_pass_windows)
    print(f"Use case density (lanes) {num_pipelines}")
    print(f"Stream density (streams) {camera_stream_count}")
    if init_duration_seconds is not None:
        print(
            f"Warmup period            {init_duration_seconds} s "
            f"(INIT_DURATION before measuring)")
    print(
        f"Measurement interval     {measurement_window_seconds} s minimum; "
        f"ramp-up is immediate, confirmation uses the {acceptance_criterion}")
    if actual_measurement_seconds is not None:
        print(
            f"Actual measurement time  {actual_measurement_seconds:.1f} s "
            f"(minimum {measurement_window_seconds} s)")
    if sample_counts:
        print(
            f"Samples per pipeline log {min(sample_counts.values())}-"
            f"{max(sample_counts.values())} across {len(sample_counts)} logs "
            f"(minimum {MIN_SAMPLES_PER_STREAM})")
    print(
        "                         (each interval is measured on its own; the "
        "count is not accumulated)")
    print(
        f"                          passing intervals increase the count during ramp-up, "
        f"and the final density is accepted only after the {acceptance_criterion} is met")
    print(
        f"Throughput threshold     target x {pass_tolerance_ratio:g} "
        f"(pass tolerance ratio)")
    print("")
    stream_w = max([len('stream')] + [len(n) for n in stream_fps_dict]) + 2
    camera_w = max([len('camera')] + [len(str(stream_meta.get(n, {}).get('camera', n)))
                                      for n in stream_fps_dict]) + 2
    workload_w = max([len('workload')] + [len(str(stream_meta.get(n, {}).get('workload', '-')))
                                          for n in stream_fps_dict]) + 2
    print(
        f"{'stream':<{stream_w}}{'camera':<{camera_w}}{'workload':<{workload_w}}{'target':>8}"
        f"{'throughput threshold':>22}{'measured (avg)':>16}{'p10':>9}{'p90':>9}"
        f"{'seconds below throughput threshold':>36}{'result':>8}")
    for name in sorted(
            stream_fps_dict,
            key=lambda n: int(re.search(r'(\d+)', n).group(1)) if re.search(r'(\d+)', n) else 0):
        meta = stream_meta.get(name, {})
        camera = meta.get("camera", name)
        workload = meta.get("workload", "-")
        target = stream_target_fps.get(name, 0.0)
        pass_mark = pass_thresholds.get(name, 0.0)
        measured = stream_fps_dict.get(name, 0.0)
        result = "pass" if measured >= pass_mark else "fail"
        samples = stream_samples.get(name, [])
        if samples:
            sample_pairs = [
                (float(sample[0]), float(sample[1]))
                for sample in samples
            ]
            fps_values = [fps for fps, _ in sample_pairs]
            sorted_samples = sorted(fps_values)
            p10 = sorted_samples[math.ceil(0.1 * len(sorted_samples)) - 1]
            p90 = sorted_samples[math.ceil(0.9 * len(sorted_samples)) - 1]
            total_sample_duration = sum(seconds for _, seconds in sample_pairs)
            below_duration = sum(
                seconds for fps, seconds in sample_pairs if fps < pass_mark)
            below_text = (
                f"{below_duration:.2f} s "
                f"({below_duration / total_sample_duration * 100:.0f}%)")
        else:
            p10 = p90 = 0.0
            below_text = "-"
        print(
            f"{name:<{stream_w}}{camera:<{camera_w}}{workload:<{workload_w}}{target:>8.2f}"
            f"{pass_mark:>22.2f}{measured:>16.6f}{p10:>9.2f}{p90:>9.2f}"
            f"{below_text:>36}{result:>8}")
    print("")
    print(
        f"Result: {'Pass' if overall_pass else 'Fail'} - "
        f"stream density {camera_stream_count} streams")
    if binding_name is not None:
        bt = stream_target_fps.get(binding_name, 0.0)
        bp = pass_thresholds.get(binding_name, 0.0)
        bm = stream_fps_dict.get(binding_name, 0.0)
        binding_camera = stream_meta.get(binding_name, {}).get("camera", binding_name)
        binding_workload = stream_meta.get(binding_name, {}).get("workload", "")
        target_values = [float(v) for v in stream_target_fps.values()]
        uniform_targets = (max(target_values) - min(target_values) < 1e-9) if target_values else True
        if uniform_targets:
            targets_clause = (
                f"All streams met the {bp:.2f} FPS throughput threshold."
                if overall_pass else f"Not all streams met the {bp:.2f} FPS throughput threshold.")
            stream_descriptor = "lowest-throughput stream"
        else:
            targets_clause = (
                "All streams met their individual throughput thresholds."
                if overall_pass else "Not all streams met their individual throughput thresholds.")
            stream_descriptor = "lowest-headroom stream"
        workload_clause = f" ({binding_workload})" if binding_workload else ""
        absolute_headroom = bm - bp
        relative_headroom = binding_headroom * 100
        settle_clause = (
            f"after a {init_duration_seconds} second warmup period (INIT_DURATION), "
            if init_duration_seconds is not None else "")
        print(
            f"The run was evaluated {settle_clause}using the {acceptance_criterion} "
            f"over consecutive measurement intervals. "
            f"{targets_clause} The {stream_descriptor} was {binding_name} "
            f"({binding_camera}{workload_clause}), targeting {bt:.2f} FPS and "
            f"measuring {bm:.6f} FPS against its {bp:.2f} FPS throughput threshold, leaving "
            f"{absolute_headroom:.2f} FPS "
            f"({relative_headroom:.1f}% relative headroom).")


def run_pipeline_iterations( 
        env_vars, compose_files, results_dir,
        container_name, target_fps, explicit_target_fps=True):
    '''
    runs an iteration of stream density benchmarking for
    a given container name and target FPS.
    Args:
        env_vars: Environment variables for docker compose.
        compose_files: Docker compose files.
        results_dir: Directory for storing results.
        container_name: Name of the container to run.
        target_fps: Target FPS to achieve.
    Returns:
        num_pipelines: Number of pipelines used.
        meet_target_fps: Whether the target FPS was achieved.
    '''
    INIT_DURATION = int(env_vars[INIT_DURATION_KEY])
    num_pipelines = 1
    in_decrement = False
    increments = 1
    meet_target_fps = False
    consecutive_fail_windows = int(
        env_vars.get(
            CONSECUTIVE_FAIL_WINDOWS_KEY,
            DEFAULT_CONSECUTIVE_FAIL_WINDOWS,
        )
    )
    consecutive_pass_windows = int(
        env_vars.get(
            CONSECUTIVE_PASS_WINDOWS_KEY,
            DEFAULT_CONSECUTIVE_PASS_WINDOWS,
        )
    )
    pass_tolerance_ratio = float(
        env_vars.get(
            PASS_TOLERANCE_RATIO_KEY,
            DEFAULT_PASS_TOLERANCE_RATIO,
        )
    )
    measurement_window_seconds = int(
        env_vars.get(
            MEASUREMENT_WINDOW_SECONDS_KEY,
            DEFAULT_MEASUREMENT_WINDOW_SECONDS,
        )
    )
    fail_window_count = 0
    pass_window_count = 0
    streams_sustained = 0

    # Measure memory usage of a single pipeline
    per_pipeline_memory_mb = measure_pipeline_memory(
        env_vars.copy(), compose_files, results_dir, container_name
    )
    print(f"TEST: per_pipeline_memory_mb = {per_pipeline_memory_mb} MB")

    # clean up any residual pipeline log files before starts:
    clean_up_pipeline_logs(results_dir)
    target_fps_label = describe_target_fps_configuration(target_fps, env_vars=env_vars)
    print(
        f"INFO: Stream density {target_fps_label} "
        f"with container_name {container_name} "
        f"and INIT_DURATION set for {INIT_DURATION} seconds; "
        f"measurement interval {measurement_window_seconds} seconds; "
        f"{describe_run_acceptance_criterion(consecutive_pass_windows)}; "
        f"{consecutive_fail_windows} consecutive failing intervals to decrement")

    while not meet_target_fps:
        # --- Memory check before scaling up ---
        pipelines_to_add = increments if increments > 0 else 0
        
        if pipelines_to_add > 0 and not check_can_add_pipelines(pipelines_to_add, per_pipeline_memory_mb, env_vars=env_vars):
            print(
                f"Aborting: Cannot add {pipelines_to_add} more pipelines due to memory constraints or pressure. "
                f"Current successful count: {num_pipelines - increments if num_pipelines > increments else num_pipelines}"
            )
            num_pipelines = num_pipelines - increments
            if num_pipelines < 1:
                num_pipelines = 1
            return num_pipelines, False, streams_sustained

        # Bring down previous iteration's containers before starting new ones
        print("Stopping previous containers before scaling...")
        benchmark.docker_compose_containers(
            "down", compose_files=compose_files,
            compose_post_args="-t 30 --volumes --remove-orphans", env_vars=env_vars)
        time.sleep(15)  # Allow kernel to reclaim cgroup memory and shared memory

        env_vars["PIPELINE_COUNT"] = str(num_pipelines)
        print(f"Starting num. of pipelines: {num_pipelines}")
        benchmark.docker_compose_containers(
            "up", compose_files=compose_files,
            compose_post_args="-d", env_vars=env_vars)
        print(f"waiting for warmup period ({INIT_DURATION} s)...")
        time.sleep(INIT_DURATION)
        
        # note: before reading the pipeline log files
        # we want to give pipelines some time as the log files
        # producing could be lagging behind...
        try:
            check_non_empty_result_logs(
                num_pipelines, results_dir, container_name, 50)
        except ValueError as e:
            print(f"ERROR: {e}")
            # since we are not able to get all non-empty log
            # the best we can do is to use the previous num_pipelines
            # before this current num_pipelines
            num_pipelines = num_pipelines - increments
            if num_pipelines < 1:
                num_pipelines = 1
            return num_pipelines, False, streams_sustained
        # once we have all non-empty pipeline log files
        # capture where each stream log ends so only samples produced
        # during the measurement interval feed the decision
        (window_start_offsets, window_end_offsets,
         actual_measurement_seconds, sample_counts,
         enough_samples) = collect_measurement_window(
            num_pipelines, results_dir, container_name,
            measurement_window_seconds)
        if not enough_samples:
            print(
                "INCONCLUSIVE: Measurement interval did not collect "
                f"{MIN_SAMPLES_PER_STREAM} samples from every pipeline log "
                f"after {actual_measurement_seconds:.1f} s "
                f"(limit {measurement_window_seconds * 2} s).")
            for log_file, count in sorted(sample_counts.items()):
                print(f"  {os.path.basename(log_file)}: {count} samples")
            return num_pipelines, False, streams_sustained
        # --- Calculate FPS and latency metrics ---
        total_fps, min_stream_fps, stream_fps_dict, stream_samples_dict = calculate_multi_stream_fps(
            num_pipelines, results_dir, container_name, env_vars,
            start_offsets=window_start_offsets,
            end_offsets=window_end_offsets)
        streams_sustained = count_valid_streams(stream_fps_dict)

        print('container name:', container_name)
        print('Total FPS:', total_fps)
        print('stream_fps_dict:', stream_fps_dict)
        avg_fps_per_stream = statistics.mean(stream_fps_dict.values()) if stream_fps_dict else 0.0
        print(f"Averaged FPS per stream (time-weighted): {avg_fps_per_stream} "
              f"for {num_pipelines} pipeline(s)")
        
        total_pipeline_latency, total_pipeline_latency_per_stream = calculate_pipeline_latency(
            num_pipelines, results_dir, container_name)
        print(f"Total Pipeline Latency: {total_pipeline_latency} "
        f"for {num_pipelines} pipeline(s)")
        print(f"Total Pipeline Latency per stream: "
        f"{total_pipeline_latency_per_stream} "
        f"for {num_pipelines} pipeline(s)")

        # --- Decide scaling logic (per-stream thresholds) ---
        stream_target_fps = build_per_stream_target_fps(
            stream_fps_dict, target_fps, env_vars=env_vars
        )

        pass_thresholds = {
            name: stream_target_fps[name] * pass_tolerance_ratio
            for name in stream_fps_dict
        }

        passing_streams = {
            name: fps for name, fps in stream_fps_dict.items()
            if fps >= pass_thresholds[name]
        }
        all_streams_meet_target = bool(stream_fps_dict) and len(passing_streams) == len(stream_fps_dict)
        if os.getenv("STREAM_DENSITY_DEBUG", "0") == "1":
            print('pass_thresholds:', pass_thresholds)
            print('passing_streams:', passing_streams)
        if os.getenv("STREAM_DENSITY_DEBUG", "0") == "1":
            print("INFO: All streams meet target" if all_streams_meet_target else "INFO: Not all streams meet target")

        if not in_decrement:
            if all_streams_meet_target:
                fail_window_count = 0
                pass_window_count = 0
                if is_env_non_empty(env_vars, PIPELINE_INCR_KEY):
                    increments = int(env_vars[PIPELINE_INCR_KEY])
                else:
                    per_stream_values = list(stream_fps_dict.values())
                    robust_per_stream = statistics.median(per_stream_values)
                    conservative_per_stream = min(min_stream_fps, robust_per_stream)
                    average_target_fps = get_mean_target_fps(stream_target_fps, target_fps)
                    if os.getenv("STREAM_DENSITY_DEBUG", "0") == "1":
                        print('mean target fps:', average_target_fps)
                    increments = int(conservative_per_stream / average_target_fps)
                    min_increment = 1
                    max_increment = MAX_GUESS_INCREMENTS
                    if increments <= 1:
                        increments = max_increment
                    else:
                        increments = min(increments, max_increment)
                print(
                    f"✅ All streams meet throughput thresholds {pass_thresholds} "
                    f"for target FPS map {stream_target_fps}. "
                    f"Incrementing pipeline no. by {increments}"
                )
            else:
                pass_window_count = 0
                fail_window_count += 1
                if fail_window_count >= consecutive_fail_windows:
                    increments = -1
                    in_decrement = True
                    fail_window_count = 0
                    print(
                        f"⚠️ throughput thresholds not met in streams: observed={stream_fps_dict}, "
                        f"throughput thresholds={pass_thresholds}. "
                        f"Observed {consecutive_fail_windows} consecutive "
                        f"below throughput threshold intervals; starting to decrement pipelines by 1..."
                    )
                else:
                    increments = 0
                    print(
                        f"INFO: Streams are below throughput threshold ({pass_thresholds}). "
                        f"Fail interval {fail_window_count}/{consecutive_fail_windows}; "
                        f"holding pipeline count at {num_pipelines} for confirmation."
                    )
        else:
            # --- In decrement phase ---
            if all_streams_meet_target:
                pass_window_count += 1
                fail_window_count = 0
                if pass_window_count >= consecutive_pass_windows:
                    print(
                        f"✅ Found maximum number of pipelines to reach "
                        f"target FPS map {stream_target_fps}"
                    )
                    meet_target_fps = True
                    print(
                        f"🎯 Max stream density achieved for target FPS map "
                        f"{stream_target_fps} is {num_pipelines}"
                    )
                    increments = 0
                    stream_camera_meta = build_per_stream_camera_meta(
                        stream_fps_dict, env_vars=env_vars)
                    print_stream_density_report(
                        num_pipelines, stream_fps_dict, stream_target_fps,
                        pass_thresholds, measurement_window_seconds,
                        consecutive_pass_windows, consecutive_fail_windows,
                        pass_tolerance_ratio, stream_camera_meta,
                        INIT_DURATION, stream_samples_dict,
                        actual_measurement_seconds, sample_counts)
                else:
                    increments = 0
                    print(
                        f"✅ Target FPS met again. Pass interval "
                        f"{pass_window_count}/{consecutive_pass_windows}; "
                        f"holding pipeline count at {num_pipelines} for confirmation."
                    )
            elif num_pipelines <= 1:
                pass_window_count = 0
                fail_window_count = 0
                per_stream_fps = [
                    float(value) for value in stream_fps_dict.values() if float(value) > 0
                ]
                avg_stream_fps = (
                    statistics.mean(per_stream_fps)
                    if per_stream_fps else 0.0
                )
                print(
                    f"already reached num. pipeline 1, and the average fps per stream is "
                    f"{avg_stream_fps:.2f} but target FPS map is {stream_target_fps}"
                )
                stream_camera_meta = build_per_stream_camera_meta(
                    stream_fps_dict, env_vars=env_vars)
                print_stream_density_report(
                    num_pipelines, stream_fps_dict, stream_target_fps,
                    pass_thresholds, measurement_window_seconds,
                    consecutive_pass_windows, consecutive_fail_windows,
                    pass_tolerance_ratio, stream_camera_meta,
                    INIT_DURATION, stream_samples_dict,
                    actual_measurement_seconds, sample_counts)
                meet_target_fps = False
                break
            else:
                pass_window_count = 0
                fail_window_count += 1
                if fail_window_count >= consecutive_fail_windows:
                    increments = -1
                    fail_window_count = 0
                    print(
                        f"decrementing number of pipelines {num_pipelines} by 1 "
                        f"because streams stayed below throughput thresholds "
                        f"({pass_thresholds}) for {consecutive_fail_windows} intervals."
                    )
                else:
                    increments = 0
                    print(
                        f"INFO: Below throughput thresholds ({pass_thresholds}). "
                        f"Fail interval {fail_window_count}/{consecutive_fail_windows}; "
                        f"holding pipeline count at {num_pipelines} for confirmation."
                    )
                        
        # --- Update pipeline count ---
        num_pipelines += increments
        if num_pipelines <= 0:
            num_pipelines = 1
            print(f"already reached min. pipeline number, stopping...")
            break

        
        
    # end of while
    #print(
    #    f"pipeline iterations done for "
    #    f"container_name: {container_name} "
    #    f"with input target_fps = {target_fps}"
    #)

    return num_pipelines, meet_target_fps, streams_sustained


def run_stream_density(env_vars, compose_files, target_fps_list,
                       container_names_list, explicit_target_fps=True):
    '''
    runs stream density using docker compose for the specified target FPS
    values and the corresponding container names
    with optional stream density pipeline increment numbers
    Args:
        env_vars: the dict of current environment variables
        compose_files: the list of compose files to run pipelines
        target_fps_list: list of target FPS values for stream density
        container_names_list: list of container names for
                              the corresponding target FPS
    Returns:
        results as a list of tuples (target_fps, container_name,
                                     num_pipelines, meet_target_fps) where
        target_fps: the desire frames per second to maintain for pipeline
        container_name: the corresponding container name for the pipeline
        num_pipelines: maximum number of pipelines to achieve TARGET_FPS
        meet_target_fps: boolean to indicate whether the returned
        number_pipelines can achieve the TARGET_FPS goal or not
    '''
    results = []
    validate_and_setup_env(env_vars, target_fps_list)
    results_dir = env_vars[RESULTS_DIR_KEY]
    log_file_path = os.path.join(results_dir, 'stream_density.log')
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    try:
        with open(log_file_path, 'a') as logger:
            logger.reconfigure(line_buffering=True, write_through=True)
            sys.stdout = logger
            sys.stderr = logger

            # loop through the target_fps list and find out the stream density:
            for target_fps, container_name in zip(
                target_fps_list, container_names_list
            ):
                print(
                    f"DEBUG: in for-loop, Starting stream density"
                    f"container_name={container_name}")
                if explicit_target_fps:
                    env_vars[TARGET_FPS_KEY] = str(target_fps)
                env_vars[CONTAINER_NAME_KEY] = container_name
                # stream density main logic:
                try:
                    num_pipelines, meet_target_fps, streams_sustained = run_pipeline_iterations(
                        env_vars, compose_files, results_dir,
                        container_name, target_fps, explicit_target_fps
                    )
                    results.append(
                        (
                            target_fps,
                            container_name,
                            num_pipelines,
                            meet_target_fps,
                            streams_sustained
                        )
                    )
                finally:
                    # better to compose-down before the next iteration
                    benchmark.docker_compose_containers(
                        "down",
                        compose_files=compose_files,
                        compose_post_args="-t 30 --volumes --remove-orphans",
                        env_vars=env_vars
                    )
                    # give time for processes and kernel cgroup memory to clean up:
                    time.sleep(15)

            # end of for-loop
            print("stream_density done!")
    except Exception as ex:
        # Restore default streams before logging the exception, otherwise
        # writing to redirected logger can fail if the file handle is closed.
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        print(f'ERROR: found exception: {ex}')
        raise
    finally:
        # reset sys stdout and err back to it's own
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr

    return results


def snapshot_stream_log_offsets(results_dir, container_name):
    """Record current end-of-file byte offsets for the run's stream logs."""
    offsets = {}
    stream_count = get_pipeline_stream_count()
    for idx in range(stream_count):
        pattern = os.path.join(
            results_dir, f'pipeline_stream{idx}_*_{container_name}.log')
        matching = filter_logs_for_current_timestamp(glob.glob(pattern))
        for log_file in matching:
            try:
                offsets[log_file] = os.path.getsize(log_file)
            except OSError:
                offsets[log_file] = 0
    return offsets


def count_measurement_samples(num_pipelines, results_dir, container_name,
                              start_offsets, end_offsets):
    """Count valid samples in each expected pipeline log for this interval."""
    sample_counts = {}
    stream_count = get_pipeline_stream_count()
    expected_log_count = stream_count
    for stream_index in range(stream_count):
        pattern = os.path.join(
            results_dir,
            f'pipeline_stream{stream_index}_*_{container_name}.log')
        matching = filter_logs_for_current_timestamp(glob.glob(pattern))
        pipeline_logs = get_latest_pipeline_stream_logs(
            1, matching)
        for pipeline_file in pipeline_logs:
            try:
                start_offset = start_offsets.get(pipeline_file, 0)
                end_offset = end_offsets.get(pipeline_file)
                if end_offset is None:
                    sample_counts[pipeline_file] = 0
                    continue
                samples = extract_numeric_fps(
                    pipeline_file,
                    start_offset=start_offset,
                    end_offset=end_offset)
                sample_counts[pipeline_file] = len(samples)
            except (IOError, OSError) as error:
                print(f"WARN: Could not count samples in {pipeline_file}: {error}")
                sample_counts[pipeline_file] = 0
    return sample_counts, expected_log_count


def collect_measurement_window(num_pipelines, results_dir, container_name,
                               minimum_duration_seconds,
                               minimum_samples=MIN_SAMPLES_PER_STREAM):
    """Collect one interval, extending it only as needed to meet the sample floor."""
    start_offsets = snapshot_stream_log_offsets(results_dir, container_name)
    start_time = time.monotonic()
    max_extension_seconds = minimum_duration_seconds
    deadline = start_time + minimum_duration_seconds + max_extension_seconds
    print(
        f"INFO: SWEEPING-START (Measurement Interval Start): "
        f"{time.strftime('%Y-%m-%dT%H:%M:%S')}")

    while True:
        elapsed = time.monotonic() - start_time
        if elapsed < minimum_duration_seconds:
            time.sleep(min(MEASUREMENT_SAMPLE_POLL_SECONDS,
                           minimum_duration_seconds - elapsed))
            continue

        end_offsets = snapshot_stream_log_offsets(results_dir, container_name)
        sample_counts, expected_log_count = count_measurement_samples(
            num_pipelines, results_dir, container_name,
            start_offsets, end_offsets)
        elapsed = time.monotonic() - start_time
        enough_samples = (
            expected_log_count > 0
            and len(sample_counts) == expected_log_count
            and all(count >= minimum_samples
                    for count in sample_counts.values())
        )
        if enough_samples or elapsed >= deadline - start_time:
            print(
                f"INFO: SWEEPING-STOP (Measurement Interval Stop): "
                f"{time.strftime('%Y-%m-%dT%H:%M:%S')}")
            return (start_offsets, end_offsets, elapsed,
                    sample_counts, enough_samples)

        print(
            f"INFO: Measurement interval has {len(sample_counts)}/"
            f"{expected_log_count} pipeline logs; minimum samples so far: "
            f"{min(sample_counts.values()) if sample_counts else 0}/"
            f"{minimum_samples}. Extending interval.")
        remaining = max(0.0, deadline - time.monotonic())
        if remaining:
            time.sleep(min(MEASUREMENT_SAMPLE_POLL_SECONDS, remaining))


def calculate_multi_stream_fps(num_pipelines, results_dir, container_name, env_vars=None, start_offsets=None, end_offsets=None):
    """
    Calculate per-stream FPS from the selected measurement-log interval.

    Each stream index is handled independently.
    Returns:
        total_fps: sum of per-stream time-weighted FPS
        min_stream_fps: lowest per-stream time-weighted FPS
        stream_fps_dict: stream -> time-weighted FPS used for pass/fail
        stream_samples_dict: stream -> FPS samples used for p10/p90 reporting
    """

    stream_count = get_pipeline_stream_count()

    # --- Initialize accumulators ---
    total_fps = 0.0
    stream_fps_dict = {}
    stream_samples_dict = {}

    # --- Loop over all streams ---
    for idx in range(stream_count):
        pattern = os.path.join(results_dir, f'pipeline_stream{idx}_*_{container_name}.log')
        matching = glob.glob(pattern)
        matching = filter_logs_for_current_timestamp(matching)
    
        if not matching:
            print(f"[WARN] No log file found for stream {idx} (container: {container_name}). Skipping...")
            stream_fps_dict[f'pipeline_stream{idx}'] = 0.0
            continue

        print(f"DEBUG: idx={idx}, match_count={len(matching)}, pattern={pattern}")
        
        latest_pipeline_logs = get_latest_pipeline_stream_logs(
            1, [path for path in matching if start_offsets is None or path in start_offsets])
        
        if not latest_pipeline_logs:
            print(f"WARN: No log file for stream index {idx}")
            stream_fps_dict[f'pipeline_stream{idx}'] = 0.0
            continue

        stream_rate = None
        selected_samples = []

        for pipeline_file in latest_pipeline_logs:
            print(f"DEBUG: Processing file: {pipeline_file}")
            try:
                samples = extract_fps_samples(
                    pipeline_file,
                    start_offset=(start_offsets or {}).get(pipeline_file, 0),
                    end_offset=(end_offsets or {}).get(pipeline_file))

                if not samples:
                    print(f"WARN: No valid FPS entries for {pipeline_file}")
                    continue

                duration = sum(seconds for _, seconds in samples)
                weighted_frames = sum(fps * seconds for fps, seconds in samples)
                rate = round(weighted_frames / duration, 6)
                print(
                    f"RATE-CALC {os.path.basename(pipeline_file)}: "
                    f"weighted_frames_estimate={weighted_frames:.6f} "
                    f"total_duration_seconds={duration:.6f} "
                    f"samples={len(samples)} measured_fps={rate:.6f}")
                if stream_rate is None or rate < stream_rate:
                    stream_rate = rate
                    selected_samples = samples

                if os.getenv("STREAM_DENSITY_DEBUG", "0") == "1":
                    print(f"INFO: Time-weighted FPS for {pipeline_file}: {rate} (samples={len(samples)}, duration={duration})")
                else:
                    print(f"INFO: Time-weighted FPS for {pipeline_file}: {rate}")

            except (IOError, OSError) as e:
                print(f"WARN: Read error on {pipeline_file}: {e}")
        
        if stream_rate is not None:
            total_fps += stream_rate
            stream_fps_dict[f'pipeline_stream{idx}'] = stream_rate
            stream_samples_dict[f'pipeline_stream{idx}'] = selected_samples
        else:
            stream_fps_dict[f'pipeline_stream{idx}'] = 0.0
            print(f"WARN: No valid FPS data for stream index {idx}")

    min_stream_fps = min(stream_fps_dict.values()) if stream_fps_dict else 0.0

    return total_fps, min_stream_fps, stream_fps_dict, stream_samples_dict


def get_pipeline_stream_count(base_dir=None):
    """
    Detects the number of video streams defined in the pipeline.sh script
   by counting occurrences of 'filesrc' or 'rtspsrc' elements.

    Args:
        base_dir (str, optional): Base directory to locate the pipeline script.
                                  Defaults to the directory of the current file.

    Returns:
        int: Number of detected streams (0 if not found or error).
    """
    try:
        # Default base_dir to the current script location if not provided
        if base_dir is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))

        # Construct pipeline.sh path (adjust relative path as needed)
        pipeline_script_path = os.path.join(base_dir, '..', '..', 'src', 'pipelines', 'pipeline.sh')
        pipeline_script_path = os.path.normpath(pipeline_script_path)

        if not os.path.isfile(pipeline_script_path):
            print(f"WARN: Pipeline script not found at {pipeline_script_path}")
            return 0

        # Read and search for 'filesrc' or 'rtspsrc' occurrences
        with open(pipeline_script_path, 'r') as f:
            content = f.read()

        matches = re.findall(r'\b(filesrc|rtspsrc)\b', content)
        if matches:
            detected_streams = len(matches)
            print(f"DEBUG: Detected {detected_streams} stream(s) from {pipeline_script_path}")
            return detected_streams
        else:
            print(f"DEBUG: No 'filesrc' or 'rtspsrc' tokens found in {pipeline_script_path}")
            return 0

    except Exception as e:
        print(f"WARN: Failed to parse pipeline script: {e}")
        return 0

def get_latest_pipeline_stream_logs(num_pipelines, pipeline_log_files):
    '''
    obtains a list of the latest pipeline log files based on
    the timestamps of the files and only returns num_pipelines
    files if number of pipeline log files is more than num_pipelines
    Args:
        num_pipelines: number of currently running pipelines
        pipeline_log_files: all matching pipeline log files
    Return:
        latest_files: number of num_pipelines files based on
        the timestamps of files if number of pipeline log files
        is more than num_pipelines; otherwise whatever the number
        of the matching files will be returned
    '''
    timestamp_files = [
        (file, os.path.getmtime(file)) for file in pipeline_log_files]
    # sort timestamp_file by time in descending order
    sorted_timestamp = sorted(
        timestamp_files, key=lambda x: x[1], reverse=True)
    latest_files = [
        file for file, mtime in sorted_timestamp[:num_pipelines]]
    return latest_files
