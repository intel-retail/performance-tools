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
import psutil

# Constants:
TARGET_FPS_KEY = "TARGET_FPS"
CONTAINER_NAME_KEY = "CONTAINER_NAME"
PIPELINE_INCR_KEY = "PIPELINE_INC"
INIT_DURATION_KEY = "INIT_DURATION"
RESULTS_DIR_KEY = "RESULTS_DIR"
DEFAULT_TARGET_FPS = 14.95
MAX_GUESS_INCREMENTS = 5


class ArgumentError(Exception):
    pass

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
        "down", compose_files=compose_files, env_vars=env_vars)
    time.sleep(3)

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
        "down", compose_files=compose_files, env_vars=env_vars)
    time.sleep(3)

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

def calculate_pipeline_latency(num_pipelines, results_dir, container_name):
    total_pipeline_latency = 0.0
    total_pipeline_latency_per_stream = 0.0
    matching_files = glob.glob(os.path.join(
        results_dir, f'gst-launch*_{container_name}*.log'))
    print(f"DEBUG: num. of gst launch matching_files = {len(matching_files)}")
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
    
def calculate_total_fps(num_pipelines, results_dir, container_name):
    '''
    calculates averaged fps from the current running num_pipelines
    Args:
        num_pipelines: number of currently running pipelines
        results_dir: directory holding the benchmark results
        container_name: the name of the container to match in log files,
                        expected to be part of the filename pattern
                        after the underscore (_)
    Returns:
        total_fps: accumulative total fps from all pipelines
        total_fps_per_stream: the averaged fps for pipelines
    '''
    total_fps = 0
    total_fps_per_stream = 0
    stream_fps_dict = {} 
    matching_files = glob.glob(os.path.join(
        results_dir, f'pipeline*_{container_name}*.log'))
    print(f"DEBUG: num. of matching_files = {len(matching_files)}")
    latest_pipeline_logs = get_latest_pipeline_logs(
        num_pipelines, matching_files)
    for pipeline_file in latest_pipeline_logs:
        print(f"DEBUG: in for loop pipeline_file:{pipeline_file}")
        with open(pipeline_file, "r") as file:
            stream_fps_list = [
                fps for fps in
                file.readlines()[-20:] if 'na' not in fps]
        if not stream_fps_list:
            print(f"WARN: No FPS returned from {pipeline_file}")
            continue
        stream_fps_sum = sum(float(fps) for fps in stream_fps_list)
        stream_fps_count = len(stream_fps_list)
        stream_fps_avg = stream_fps_sum / stream_fps_count
        total_fps += stream_fps_avg
        total_fps_per_stream = total_fps / num_pipelines
        print(
            f"INFO: Averaged FPS for pipeline file "
            f"{pipeline_file}: {stream_fps_avg}")
            
    stream_fps_dict["pipeline_stream"] = round(total_fps_per_stream, 2)         
    return total_fps, total_fps_per_stream, stream_fps_dict


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

    if not is_env_non_empty(env_vars, INIT_DURATION_KEY):
        env_vars[INIT_DURATION_KEY] = "120"


def run_pipeline_iterations( 
        env_vars, compose_files, results_dir,
        container_name, target_fps):
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

    # Measure memory usage of a single pipeline
    per_pipeline_memory_mb = measure_pipeline_memory(
        env_vars.copy(), compose_files, results_dir, container_name
    )
    print(f"TEST: per_pipeline_memory_mb = {per_pipeline_memory_mb} MB")

    # clean up any residual pipeline log files before starts:
    clean_up_pipeline_logs(results_dir)
    print(
        f"INFO: Stream density TARGET_FPS set for {target_fps} "
        f"with container_name {container_name} "
        f"and INIT_DURATION set for {INIT_DURATION} seconds")

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
            return num_pipelines, False

        env_vars["PIPELINE_COUNT"] = str(num_pipelines)
        print(f"Starting num. of pipelines: {num_pipelines}")
        benchmark.docker_compose_containers(
            "up", compose_files=compose_files,
            compose_post_args="-d", env_vars=env_vars)
        print("waiting for pipelines to settle...")
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
            return num_pipelines, False
        # once we have all non-empty pipeline log files
        # we then can calculate the average fps
        print(f"INFO: ########## MULTI_STREAM_MODE VALUE==== {env_vars.get('MULTI_STREAM_MODE', 0)}")
        # --- Calculate FPS and latency metrics ---
        if int(env_vars.get("MULTI_STREAM_MODE", 0)) == 0:
            # Multi-stream mode: use LP variant
            total_fps, total_fps_per_stream, stream_fps_dict = calculate_total_fps(
                num_pipelines, results_dir, container_name)
        else:
            # Single-stream mode: standard calculation
            total_fps, total_fps_per_stream, stream_fps_dict = calculate_multi_stream_fps(
                num_pipelines, results_dir, container_name)

       
        print('container name:', container_name)
        print('Total FPS:', total_fps)
        print('stream_fps_dict:', stream_fps_dict)
        print(f"Total averaged FPS per stream: {total_fps_per_stream} "
              f"for {num_pipelines} pipeline(s)")
        
        total_pipeline_latency, total_pipeline_latency_per_stream = calculate_pipeline_latency(
            num_pipelines, results_dir, container_name)
        print(f"Total Pipeline Latency: {total_pipeline_latency} "
        f"for {num_pipelines} pipeline(s)")
        print(f"Total Pipeline Latency per stream: "
        f"{total_pipeline_latency_per_stream} "
        f"for {num_pipelines} pipeline(s)")

        # --- Decide scaling logic ---
        if not in_decrement:
            # Check if all streams meet or exceed target FPS
            all_streams_meet_target = all(fps >= target_fps for fps in stream_fps_dict.values())
            if all_streams_meet_target:
                if is_env_non_empty(env_vars, PIPELINE_INCR_KEY):
                    increments = int(env_vars[PIPELINE_INCR_KEY])
                else:
                    increments = int(total_fps_per_stream / target_fps)
                    if increments == 1:
                        increments = MAX_GUESS_INCREMENTS
                print(f"✅ All streams meet target FPS ({target_fps}). Incrementing pipeline no. by {increments}")
            else:
                # Some streams below target
                below_streams = {k: v for k, v in stream_fps_dict.items() if v < target_fps}
                increments = -1
                in_decrement = True
                print(
                    f"⚠️ Below target FPS ({target_fps}) in streams: {below_streams}. "
                    f"Starting to decrement pipelines by 1..."
                )
        else:
            # --- In decrement phase ---
            all_streams_meet_target = all(fps >= target_fps for fps in stream_fps_dict.values())

            if all_streams_meet_target:
                print(
                    f"✅ Found maximum number of pipelines to reach "
                    f"target FPS {target_fps}")
                meet_target_fps = True
                print(
                    f"🎯 Max stream density achieved for target FPS "
                    f"{target_fps} is {num_pipelines}")
                increments = 0
            elif num_pipelines <= 1:
                print(
                    f"already reached num. pipeline 1, and "
                    f"the fps per stream is {total_fps_per_stream} "
                    f"but target FPS is {target_fps}")
                meet_target_fps = False
                break
            else:
                print(
                    f"decrementing number of pipelines "
                    f"{num_pipelines} by 1")

        # --- Update pipeline count ---
        num_pipelines += increments
        if num_pipelines <= 0:
            num_pipelines = 1
            print(f"already reached min. pipeline number, stopping...")
            break

        
        
    # end of while
    print(
        f"pipeline iterations done for "
        f"container_name: {container_name} "
        f"with input target_fps = {target_fps}"
    )

    return num_pipelines, meet_target_fps


def run_stream_density(env_vars, compose_files, target_fps_list,
                       container_names_list):
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
            sys.stdout = logger
            sys.stderr = logger

            # loop through the target_fps list and find out the stream density:
            for target_fps, container_name in zip(
                target_fps_list, container_names_list
            ):
                print(
                    f"DEBUG: in for-loop, target_fps={target_fps} "
                    f"container_name={container_name}")
                env_vars[TARGET_FPS_KEY] = str(target_fps)
                env_vars[CONTAINER_NAME_KEY] = container_name
                # stream density main logic:
                try:
                    num_pipelines, meet_target_fps = run_pipeline_iterations(
                        env_vars, compose_files, results_dir,
                        container_name, target_fps
                    )
                    results.append(
                        (
                            target_fps,
                            container_name,
                            num_pipelines,
                            meet_target_fps
                        )
                    )
                finally:
                    # better to compose-down before the next iteration
                    benchmark.docker_compose_containers(
                        "down",
                        compose_files=compose_files,
                        env_vars=env_vars
                    )
                    # give some time for processes to clean up:
                    time.sleep(10)

            # end of for-loop
            print("stream_density done!")
    except Exception as ex:
        print(f'ERROR: found exception: {ex}')
        raise
    finally:
        # reset sys stdout and err back to it's own
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr

    return results


def calculate_multi_stream_fps(num_pipelines, results_dir, container_name):
    """
    Calculates averaged FPS per stream for all matching log files named pipeline_stream<idx>*.log.
    Each stream index is handled independently.
    Returns:
        total_fps: sum of per-stream averaged FPS
        total_fps_per_stream: average FPS across all streams
        stream_fps_dict: filename -> averaged FPS
    """

    stream_count = get_pipeline_stream_count()

    # --- Initialize accumulators ---
    total_fps = 0.0
    stream_fps_dict = {}
    time.sleep(100)  # Ensure logs are fully written
    
    # --- Loop over all streams ---
    for idx in range(stream_count):
        pattern = os.path.join(results_dir, f'pipeline_stream{idx}_*_{container_name}.log')
        matching = glob.glob(pattern)
    
        if not matching:
            print(f"[WARN] No log file found for stream {idx} (container: {container_name}). Skipping...")
            continue

        print(f"DEBUG: idx={idx}, match_count={len(matching)}, pattern={pattern}")
        
        latest_pipeline_logs = get_latest_pipeline_stream_logs(num_pipelines, matching)
        
        if not latest_pipeline_logs:
            print(f"WARN: No log file for stream index {idx}")
            stream_fps_dict[f'pipeline_stream{idx}'] = 0.0
            continue

        # --- Process all matching log files for this stream ---
        stream_avg_sum = 0.0

        for pipeline_file in latest_pipeline_logs:
            print(f"DEBUG: Processing file: {pipeline_file}")
            try:
                with open(pipeline_file, 'r') as f:
                    tail_lines = f.readlines()[-20:]
                fps_lines = [l.strip() for l in tail_lines if l.strip() and 'na' not in l.lower()]
                numeric_fps = []
                for v in fps_lines:
                    try:
                        numeric_fps.append(float(v))
                    except ValueError:
                        print(f"DEBUG: Skipping non-numeric line '{v}' in {pipeline_file}")

                if not numeric_fps:
                    print(f"WARN: No numeric FPS entries for {pipeline_file}")
                    continue
                file_name = os.path.basename(pipeline_file)
                stream_fps_avg = sum(numeric_fps) / len(numeric_fps)
                stream_avg_sum += stream_fps_avg

                print(f"INFO: Averaged FPS for {pipeline_file}: {stream_fps_avg}")

            except (IOError, OSError) as e:
                print(f"WARN: Read error on {pipeline_file}: {e}")
        
        # --- Compute average across all files for this stream index ---
        if num_pipelines > 0:
            final_stream_avg = stream_avg_sum / num_pipelines
            stream_fps_dict[f'pipeline_stream{idx}'] = round(final_stream_avg, 2)
            total_fps += final_stream_avg
        else:
            stream_fps_dict[f'pipeline_stream{idx}'] = 0.0
            print(f"WARN: No valid FPS data for stream index {idx}")

    # --- Compute total and per-stream averages ---
    total_fps_per_stream = total_fps / stream_count if stream_count > 0 else 0.0

    return total_fps, total_fps_per_stream, stream_fps_dict


def get_pipeline_stream_count(base_dir=None):
    """
    Detects the number of video streams defined in the pipeline.sh script
    by counting occurrences of 'filesrc' elements.

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

        # Read and search for 'filesrc' occurrences
        with open(pipeline_script_path, 'r') as f:
            content = f.read()

        matches = re.findall(r'\bfilesrc\b', content)
        if matches:
            detected_streams = len(matches)
            print(f"DEBUG: Detected {detected_streams} stream(s) from {pipeline_script_path}")
            return detected_streams
        else:
            print(f"DEBUG: No 'filesrc' tokens found in {pipeline_script_path}")
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
        timestamp_files, key=lambda x: x[1], reverse=False)
    latest_files = [
        file for file, mtime in sorted_timestamp[:num_pipelines]]
    return latest_files