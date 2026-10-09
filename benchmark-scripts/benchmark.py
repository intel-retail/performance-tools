'''
* Copyright (C) 2025 Intel Corporation.
*
* SPDX-License-Identifier: Apache-2.0
'''

import argparse
import os
import shlex
import subprocess
import time
import traceback
import csv
import json
import stream_density
from device_validation import validate_target_device, resolve_target_device_default


def parse_args(print=False):
    '''
    parses the input arguments for the command line

    Args:
        print: boolean on whether to print the help or not

    Returns:
        None: if print is True
        parser_object: if the input arguments are parsed
    '''
    parser = argparse.ArgumentParser(
        prog='benchmark',
        description='runs benchmarking using docker compose')
    parser.add_argument('--pipelines', type=int, default=1,
                        help='number of pipelines')
    # allowed multiple inputs for target_fps: e.g.: --target_fps 14.95 8.5
    parser.add_argument('--target_fps', type=float, nargs='*', default=None,
                        help='stream density target FPS; ' +
                        'can take multiple values for multiple different ' +
                        'pipelines with 1-to-1 mapping with pipeline ' +
                        'container name via --container_names')
    # when multiple target_fps is specified, the 1-to-1 mapping between
    # target_fps and container_names are used; examples are given below
    # --target_fps 14.95 8.5 14.95 \
    # --container_names container1 container2 container3
    parser.add_argument('--container_names', type=str, nargs='*',
                        default=None, help='stream density target ' +
                        'container names; used together with --target_fps ' +
                        'to have 1-to-1 mapping with the pipeline')
    parser.add_argument('--benchmark_type', default='default', 
                        choices=['default', 'reg'],
                        help='benchmark type: default uses docker-compose.yaml, reg uses docker-compose-reg.yaml')
    parser.add_argument('--density_increment', type=int, default=None,
                        help='pipeline increment number for ' +
                             'stream density. If not specified, then ' +
                             ' it will be dynamically adjusted.')
    parser.add_argument('--results_dir',
                        default=os.path.join(os.curdir, 'results'),
                        help='full path to the desired directory for logs ' +
                             'and results')
     # BEGIN fix: safely read MULTI_STREAM_MODE from environment
    multi_stream_raw = os.getenv("MULTI_STREAM_MODE", "0")
    try:
        multi_stream_mode = int(multi_stream_raw)
    except ValueError:
        multi_stream_mode = 0
    # END fix
    if multi_stream_mode == 0:
        parser.add_argument('--duration', type=int, default=30,
                            help='time in seconds, not needed when ' +
                                 '--target_fps is specified')
    else:
        parser.add_argument('--duration', type=int, default=150,
                            help='time in seconds, not needed when ' +
                                 '--target_fps is specified')  

    parser.add_argument('--init_duration', type=int, default=10,
                        help='initial time in seconds before ' +
                             'starting metric data collection')
    parser.add_argument('--measurement_window_seconds', type=int, default=100,
                        help='duration in seconds to collect FPS samples ' +
                             'after INIT_DURATION for the stream density ' +
                             'pass/fail decision; minimum 100, because ' +
                             'each pipeline log needs 100 samples at about ' +
                             'one per second and collection extends to at ' +
                             'most twice this duration')
    # TODO: change target_device to an env variable in docker compose
    try:
        default_target_device = resolve_target_device_default('CPU')
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    parser.add_argument('--target_device', default=default_target_device,
                        type=validate_target_device,
                        help='desired running platform [CPU|GPU|GPU.<index>|NPU]')
    parser.add_argument('--compose_file', default=None, action='append',
                        help='path to docker compose files. ' +
                             'can be used multiple times')
    parser.add_argument('--retail_use_case_root',
                        default=os.path.join(
                            os.curdir, '..', '..'),
                        help='full path to the retail-use-cases repo root')
    parser.add_argument('--docker_log', default=None, 
                        help='docker container name to get logs of and save to file')
    parser.add_argument('--parser_script', 
                        default=os.path.join(os.path.curdir, 'parse_qmassa_metrics_to_json.py'), 
                        help='full path to the parsing script to obtain FPS')
    parser.add_argument('--parser_args', default='-k device -k qmassa', 
                        help='arguments to pass to the parser script, ' + 
                        'pass args with spaces in quotes: "args with spaces"')
    if print:
        parser.print_help()
        return
    args = parser.parse_args()
    if args.density_increment and not args.target_fps and not args.container_names:
        parser.error(
            '--density_increment needs --target_fps or --container_names '
            'to be specified')
    if args.compose_file is None:
        parser.error(
            '--compose_file is empty, please provide compose files')
    return args


def docker_compose_containers(command, compose_files=[], compose_pre_args="",
                              compose_post_args="",
                              env_vars=os.environ.copy()):
    '''
    helper function to bring up or down containers using the provided params

    Args:
        command: valid docker compose command like "up" or "down"
        compose_files: list of docker compose files
        compose_pre_args: string of arguments called before the command
        compose_post_args: string of arguments called after the command
        env_vars: environment variables to use in the shell for calling compose

    Returns:
        stdout: console output from Popen when running the command
        stderr: console error from Popen when running the command
        returncode: Popen return code
    '''
    try:
        files = " -f ".join(compose_files)
        compose_string = ("docker compose %s -f %s %s %s" %
                          (compose_pre_args, files, command,
                           compose_post_args))
        compose_args = shlex.split(compose_string)

        p = subprocess.Popen(compose_args,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             env=env_vars)  # nosec B404, B603
        stdout, stderr = p.communicate()

        if p.returncode and stderr:
            print("Error bringing %s the compose files: %s" %
                  (command, stderr))
        return stdout.strip(), stderr, p.returncode
    except subprocess.CalledProcessError:
        print("Exception bringing %s the compose files: %s" %
              (command, traceback.format_exc()))



def print_stream_density_result(status, num_pipelines, streams_sustained,
                                results_dir):
    if status == stream_density.STATUS_PASS:
        print("Result: Pass")
        print(
            f"Stream density completed: use case density {num_pipelines} lanes, "
            f"stream density {streams_sustained} streams. "
            f"For the detailed report, see {results_dir}/stream_density.log")
    elif status == stream_density.STATUS_INCONCLUSIVE:
        print("Result: Inconclusive")
        print(
            f"Measurement intervals did not collect "
            f"{stream_density.MIN_SAMPLES_PER_STREAM} samples from every "
            f"pipeline log, so they were not used for a pass/fail decision.")
        if num_pipelines:
            print(
                f"Last passing density: use case density {num_pipelines} lanes, "
                f"stream density {streams_sustained} streams, not confirmed by "
                f"the run acceptance criterion.")
        else:
            print("No measurement interval had passed.")
        print(
            f"Per-log sample counts are in {results_dir}/stream_density.log")
    else:
        print("Result: Fail")
        print(
            "Stream density did not meet the target. Check "
            f"{results_dir}/stream_density.log and "
            f"{results_dir}/gst-launch*.log for further analysis.")


def main():
    '''
    runs benchmarking using docker compose for the specified pipeline
    '''
    my_args = parse_args()

    target_fps_list = my_args.target_fps if my_args.target_fps else []
    container_names_list = (
        my_args.container_names
        if my_args.container_names else []
    )

    # Distinguishes a user-supplied target from the synthesized fallback so
    # the fallback can never be treated as an explicit TARGET_FPS override.
    explicit_target_fps = bool(target_fps_list)

    # With no explicit target FPS, container names alone drive stream
    # density; per-camera targetFps/fps (or env TARGET_FPS) resolves the
    # actual per-stream targets, with DEFAULT_TARGET_FPS as final fallback.
    if not target_fps_list and container_names_list:
        target_fps_list = [
            stream_density.DEFAULT_TARGET_FPS
        ] * len(container_names_list)

    if (len(target_fps_list) > 1
            and len(target_fps_list) != len(container_names_list)):
        raise ValueError(
            "For stream density, the number of target FPS "
            "values must match the number of "
            "container names provided."
        )

    results_dir = os.path.abspath(my_args.results_dir)
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    print("Starting workload(s)")

    # start the docker containers
    # pass in necessary variables using env vars
    compose_files = []
    for file in my_args.compose_file:
        compose_files.append(os.path.abspath(file))

    # Determine which benchmark compose file to use
    if my_args.benchmark_type == 'reg':
        benchmark_compose = os.path.abspath(os.path.join(
            os.curdir, '..', 'docker', 'docker-compose-reg.yaml'))
    else:
        benchmark_compose = os.path.abspath(os.path.join(
            os.curdir, '..', 'docker', 'docker-compose.yaml'))
    
    # Replace the hardcoded compose file addition
    compose_files.append(benchmark_compose)

    env_vars = os.environ.copy()
    env_vars["log_dir"] = results_dir
    env_vars["RESULTS_DIR"] = results_dir
    env_vars["DEVICE"] = my_args.target_device
    retail_use_case_root = os.path.abspath(my_args.retail_use_case_root)
    env_vars["RETAIL_USE_CASE_ROOT"] = retail_use_case_root
    env_vars["VLM_WORKLOAD_ENABLED"] = str(os.getenv("LP_VLM_WORKLOAD_ENABLED"))
    env_vars["MEASUREMENT_WINDOW_SECONDS"] = str(my_args.measurement_window_seconds)
    env_vars["INIT_DURATION"] = str(my_args.init_duration)
    if my_args.density_increment:
        env_vars["PIPELINE_INC"] = str(my_args.density_increment)
    if len(target_fps_list) > 1 and container_names_list:
        # stream density for multiple target FPS values and containers
        print('starting stream density for multiple running pipelines...')
        results = stream_density.run_stream_density(env_vars, compose_files,
                                                    target_fps_list,
                                                    container_names_list,
                                                    explicit_target_fps)
        for result in results:
            target_fps, container_name, num_pipelines, status, streams_sustained = result
            print_stream_density_result(
                status, num_pipelines, streams_sustained, results_dir)
    elif len(target_fps_list) == 1:
        # single target_fps stream density mode:
        print('starting stream density...')
        if explicit_target_fps:
            env_vars["TARGET_FPS"] = str(target_fps_list[0])
        if my_args.density_increment:
            env_vars["PIPELINE_INC"] = str(my_args.density_increment)
        # use a default name since there is no
        # --container_names provided in this case
        container_name = (container_names_list[0]
                          if container_names_list else "default_container")
        results = stream_density.run_stream_density(
            env_vars, compose_files, [target_fps_list[0]], [container_name],
            explicit_target_fps)
        target_fps, container_name, num_pipelines, status, streams_sustained = results[0]
        print_stream_density_result(
            status, num_pipelines, streams_sustained, results_dir)
    else:
        # regular --pipelines mode:
        stream_density.clean_up_pipeline_logs(results_dir)
        if my_args.pipelines > 0:
            env_vars["PIPELINE_COUNT"] = str(my_args.pipelines)
        docker_compose_containers("up", compose_files=compose_files,
                                  compose_post_args="-d",
                                  env_vars=env_vars)
        print("Waiting for %ds init duration to complete" % my_args.init_duration)
        time.sleep(my_args.init_duration)

        # use duration to sleep
        print(
            "Waiting for %ds for workload to finish"
            % my_args.duration)
        time.sleep(my_args.duration)
        
        # grab the container logs if necessary
        if my_args.docker_log:
            try:
                docker_log = ("docker logs %s" % my_args.docker_log)
                docker_log_args = shlex.split(docker_log)
                log_file = os.path.join(my_args.results_dir, "%s.log" % my_args.docker_log)    
                print("writing docker log to %s" % log_file)
                with open(log_file, 'wb') as f:
                    subprocess.run(docker_log_args,
                                   stdout=f,
                                   stderr=subprocess.STDOUT,
                                   check=True, env=env_vars)  # nosec B404, B603
            
            except subprocess.CalledProcessError:
                print("Exception getting the docker log %s: %s" %
                    (my_args.docker_log, traceback.format_exc()))        
        
        # stop all containers and camera-simulator
        docker_compose_containers("down", compose_files=compose_files,
                                  env_vars=env_vars)

    # collect metrics using copy-platform-metrics
    print("workloads finished...")
    # TODO: implement results handling based on what pipeline is run
    try:
        parser_string = ("python3 %s -d %s %s" % (my_args.parser_script, results_dir, my_args.parser_args))
        # print("======DEBUG======: %s" % parser_string)
        parser_args = shlex.split(parser_string)

        subprocess.run(parser_args,
                       check=True, env=env_vars)  # nosec B404, B603
    except subprocess.CalledProcessError:
        print("Exception calling %s\n parser %s: %s" %
              (parser_string, my_args.parser_script, traceback.format_exc()))

if __name__ == '__main__':
    main()
