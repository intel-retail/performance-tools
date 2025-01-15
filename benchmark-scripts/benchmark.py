'''
* Copyright (C) 2024 Intel Corporation.
*
* SPDX-License-Identifier: Apache-2.0
'''

import argparse
import os
import shlex
import subprocess  # nosec B404
import time
import traceback
import csv
import json
import stream_density


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
    parser.add_argument('--density_increment', type=int, default=None,
                        help='pipeline increment number for ' +
                             'stream density. If not specified, then ' +
                             ' it will be dynamically adjusted.')
    parser.add_argument('--results_dir',
                        default=os.path.join(os.curdir, 'results'),
                        help='full path to the desired directory for logs ' +
                             'and results')
    parser.add_argument('--duration', type=int, default=30,
                        help='time in seconds, not needed when ' +
                             '--target_fps is specified')
    parser.add_argument('--init_duration', type=int, default=20,
                        help='initial time in seconds before ' +
                             'starting metric data collection')
    # TODO: change target_device to an env variable in docker compose
    parser.add_argument('--target_device', default='CPU',
                        help='desired running platform [cpu|core|xeon|dgpu.x]')
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
                        default=os.path.join(os.path.curdir, 'parse_csv_to_json.py'), 
                        help='full path to the parsing script')
    parser.add_argument('--parser_args', default='-k device -k igt', 
                        help='arguments to pass to the parser script, ' + 
                        'pass args with spaces in quotes: "args with spaces"')
    if print:
        parser.print_help()
        return
    args = parser.parse_args()
    if args.density_increment and not args.target_fps:
        parser.error(
            '--density_increment needs to have --target_fps be specified')
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

    # add the benchmark docker compose file
    compose_files.append(os.path.abspath(os.path.join(
        os.curdir, '..', 'docker', 'docker-compose.yaml')))
    env_vars = os.environ.copy()
    env_vars["log_dir"] = results_dir
    env_vars["RESULTS_DIR"] = results_dir
    env_vars["DEVICE"] = my_args.target_device
    retail_use_case_root = os.path.abspath(my_args.retail_use_case_root)
    env_vars["RETAIL_USE_CASE_ROOT"] = retail_use_case_root
    if my_args.density_increment:
        env_vars["PIPELINE_INC"] = str(my_args.density_increment)
    if len(target_fps_list) > 1 and container_names_list:
        # stream density for multiple target FPS values and containers
        print('starting stream density for multiple running pipelines...')
        results = stream_density.run_stream_density(env_vars, compose_files,
                                                    target_fps_list,
                                                    container_names_list)
        for result in results:
            target_fps, container_name, num_pipelines, met_fps = result
            print(
                f"Completed stream density for target FPS: {target_fps} in "
                f"container: {container_name}. "
                f"Max pipelines: {num_pipelines}, "
                f"Met target FPS? {met_fps}")
    elif len(target_fps_list) == 1:
        # single target_fps stream density mode:
        print('starting stream density...')
        env_vars["TARGET_FPS"] = str(target_fps_list[0])
        if my_args.density_increment:
            env_vars["PIPELINE_INC"] = str(my_args.density_increment)
        env_vars["INIT_DURATION"] = str(my_args.init_duration)
        # use a default name since there is no
        # --container_names provided in this case
        container_name = (container_names_list[0]
                          if container_names_list else "default_container")
        results = stream_density.run_stream_density(
            env_vars, compose_files, [target_fps_list[0]], [container_name])
        target_fps, container_name, num_pipelines, met_fps = results[0]
        print(
            f"Max number of pipelines in stream density found for target "
            f"FPS = {target_fps} is {num_pipelines}. "
            f"Met target FPS? {met_fps}")
    else:
        # regular --pipelines mode:
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
        parser_string = ("python %s -d %s %s" % (my_args.parser_script, results_dir, my_args.parser_args))
        print("======DEBUG======: %s" % parser_string)
        parser_args = shlex.split(parser_string)

        subprocess.run(parser_args,
                       stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE,
                       check=True, env=env_vars)  # nosec B404, B603
    except subprocess.CalledProcessError:
        print("Exception starting the parser %s: %s" %
              (my_args.parser_script, traceback.format_exc()))

if __name__ == '__main__':
    main()
