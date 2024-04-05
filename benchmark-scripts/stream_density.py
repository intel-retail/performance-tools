'''
* Copyright (C) 2024 Intel Corporation.
*
* SPDX-License-Identifier: Apache-2.0
'''

import os
import time
import benchmark
import glob
import sys


# Constants:
TARGET_FPS_KEY = "TARGET_FPS"
PIPELINE_INCR_KEY = "PIPELINE_INC"
INIT_DURATION_KEY = "INIT_DURATION"
RESULTS_DIR_KEY = "RESULTS_DIR"


class ArgumentError(Exception):
    pass


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
    matching_files = glob.glob(os.path.join(results_dir, 'pipeline*_*.log'))
    if len(matching_files) > 0:
        for log_file in matching_files:
            os.remove(log_file)
    else:
        print('INFO: no match files to clean up')


def check_non_empty_result_logs(num_pipelines, results_dir, max_retries=5):
    '''
    checks the current non-empty pipeline log files with some
    retries upto max_retires if file not exists or empty
    Args:
        num_pipelines: number of currently running pipelines
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
            results_dir, 'pipeline*_*.log'))
        if len(matching_files) >= num_pipelines and all([
              os.path.isfile(file) and os.path.getsize(file) > 0
              for file in matching_files]):
            print('found all non-empty log files')
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


def calculate_total_fps(num_pipelines, results_dir):
    '''
    calculates averaged fps from the current running num_pipelines
    Args:
        num_pipelines: number of currently running pipelines
        results_dir: directory holding the benchmark results
    Returns:
        total_fps: accumulative total fps from all pipelines
        total_fps_per_stream: the averaged fps for pipelines
    '''
    total_fps = 0
    total_fps_per_stream = 0
    matching_files = glob.glob(os.path.join(
        results_dir, 'pipeline*_*.log'))
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
    return total_fps, total_fps_per_stream


def run_stream_density(env_vars, compose_files):
    '''
    runs stream density using docker compose for the specified target FPS
    with optional stream density pipeline increment numbers
    Args:
        env_vars: the dict of current environment variables
        compose_files: the list of compose files to run pipelines
    Returns:
        num_pipelines: maximum number of pipelines to achieve TARGET_FPS
        meet_target_fps: boolean to indicate whether the returned
        number_pipelines can achieve the TARGET_FPS goal or not
    '''
    if not is_env_non_empty(env_vars, RESULTS_DIR_KEY):
        raise ArgumentError('ERROR: missing ' +
                            RESULTS_DIR_KEY + 'in env')

    # use some default values is missing
    if not is_env_non_empty(env_vars, TARGET_FPS_KEY):
        env_vars[TARGET_FPS_KEY] = "15.0"
    elif float(env_vars[TARGET_FPS_KEY]) <= 0.0:
        raise ArgumentError(
            'ERROR: stream density target fps should be greater than 0')

    if is_env_non_empty(env_vars, PIPELINE_INCR_KEY) and int(
            env_vars[PIPELINE_INCR_KEY]) <= 0:
        raise ArgumentError(
            'ERROR: stream density increments should be greater than 0')

    if not is_env_non_empty(env_vars, INIT_DURATION_KEY):
        env_vars[INIT_DURATION_KEY] = "120"

    # set up:
    TARGET_FPS = float(env_vars[TARGET_FPS_KEY])
    INIT_DURATION = int(env_vars[INIT_DURATION_KEY])
    MAX_GUESS_INCREMENTS = 5
    num_pipelines = 1
    results_dir = env_vars[RESULTS_DIR_KEY]
    log_file_path = os.path.join(results_dir, 'stream_density.log')
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    try:
        logger = open(log_file_path, 'w')
        sys.stdout = logger
        sys.stderr = logger

        # clean up any residual pipeline log files before starts:
        clean_up_pipeline_logs(results_dir)

        print(f"INFO: Stream density TARGET_FPS set for {TARGET_FPS} "
              f"and INIT_DURATION set for {INIT_DURATION}")

        # stream density main logic:
        in_decrement = False
        increments = 1
        meet_target_fps = False
        while not meet_target_fps:
            total_fps_per_stream = 0.0
            total_fps = 0.0
            env_vars["PIPELINE_COUNT"] = str(num_pipelines)
            print("Starting num. of pipelines: %d" % num_pipelines)
            benchmark.docker_compose_containers(
                "up", compose_files=compose_files,
                compose_post_args="-d", env_vars=env_vars)
            print("waiting for pipelines to settle...")
            time.sleep(INIT_DURATION)
            # note: before reading the pipeline log files
            # we want to give pipelines some time as the log files
            # producing could be lagging behind...
            check_non_empty_result_logs(num_pipelines, results_dir, 50)
            # once we have all non-empty pipeline log files
            # we then can calculate the average fps
            total_fps, total_fps_per_stream = calculate_total_fps(
                num_pipelines, results_dir)
            print('Total FPS:', total_fps)
            print(
                f"Total averaged FPS per stream: {total_fps_per_stream} "
                f"for {num_pipelines} pipeline(s)")
            if not in_decrement:
                if total_fps_per_stream >= TARGET_FPS:
                    # if the increments hint from $PIPELINE_INC is not empty
                    # we will use it as the increments
                    # otherwise, we will try to adjust increments dynamically
                    # based on the rate of {total_fps_per_stream}
                    # and $TARGET_FPS
                    if is_env_non_empty(env_vars, PIPELINE_INCR_KEY):
                        increments = int(env_vars[PIPELINE_INCR_KEY])
                    else:
                        increments = int(
                            total_fps_per_stream / TARGET_FPS)
                        if increments == 1:
                            increments = MAX_GUESS_INCREMENTS
                    print(f"incrementing pipeline no. by {increments}")
                else:
                    # below TARGET_FPS, start decrementing
                    increments = -1
                    in_decrement = True
                    print(f"Below target fps {TARGET_FPS}, "
                          f"starting to decrement pipelines by 1...")
            else:
                if total_fps_per_stream >= TARGET_FPS:
                    print(f"found maximum number of pipelines to reach "
                          f"target fps {TARGET_FPS}")
                    meet_target_fps = True
                    print(f"Max stream density achieved for target FPS "
                          f"{TARGET_FPS} is {num_pipelines}")
                    increments = 0
                    print("Finished stream density benchmarking")
                else:
                    if num_pipelines <= 1:
                        print(f"already reached num pipeline 1, and the "
                              f"fps per stream is {total_fps_per_stream} "
                              f"but target FPS is {TARGET_FPS}")
                        meet_target_fps = False
                        break
                    else:
                        print(f"decrementing number of pipelines "
                              f"{num_pipelines} by 1")
            # end of if not in_decrement:
            num_pipelines += increments
        # end of while
        print("stream_density done!")
    except Exception as ex:
        print(f'ERROR: found exception: {ex}')
        raise
    finally:
        benchmark.docker_compose_containers(
            "down", compose_files=compose_files,
            env_vars=env_vars)
        # reset sys stdout and err back to it's own
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
    return num_pipelines, meet_target_fps
