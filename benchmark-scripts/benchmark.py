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
    parser.add_argument('--target_fps', type=float, default=None,
                        help='stream density target FPS')
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


def convert_csv_results_to_json(results_dir, log_name):
    '''
    convert the csv output to json format for readability

    Args:
        results_dir: directory containing the benchmark results
        log_name: first portion of the log filename to search for
    '''
    for entry in os.scandir(results_dir):
        if entry.name.startswith(log_name) and entry.is_file():
            print(entry.path)
            csv_file = open(entry.path)
            json_file = json.dumps([dict(r) for r in csv.DictReader(csv_file)])
            device_name = entry.name.split('.')
            json_result_path = os.path.join(
                results_dir, device_name[0]+".json")
            with open(json_result_path, "w") as outfile:
                outfile.write(json_file)
            outfile.close()
            csv_file.close()


def main():
    '''
    runs benchmarking using docker compose for the specified pipeline
    '''
    my_args = parse_args()

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
    if my_args.target_fps:
        # stream density mode:
        print('starting stream density...')
        env_vars["TARGET_FPS"] = str(my_args.target_fps)
        if my_args.density_increment:
            env_vars["PIPELINE_INC"] = str(my_args.density_increment)
        env_vars["INIT_DURATION"] = str(my_args.init_duration)
        max_num_pipelines, met_fps = stream_density.run_stream_density(
            env_vars, compose_files)
        input_target_fps = env_vars["TARGET_FPS"]
        print(
            f"Max number of pipelines in stream density found for "
            f"target FPS = {input_target_fps} is "
            f"{max_num_pipelines}. met target fps? {met_fps}")
    else:
        # regular --pipelines mode:
        if my_args.pipelines > 0:
            env_vars["PIPELINE_COUNT"] = str(my_args.pipelines)
        docker_compose_containers("up", compose_files=compose_files,
                                  compose_post_args="-d",
                                  env_vars=env_vars)
        print("Waiting for init duration to complete...")
        time.sleep(my_args.init_duration)

        # use duration to sleep
        print(
            "Waiting for %d seconds for workload to finish"
            % my_args.duration)
        time.sleep(my_args.duration)
        # stop all containers and camera-simulator
        docker_compose_containers("down", compose_files=compose_files,
                                  env_vars=env_vars)

    # collect metrics using copy-platform-metrics
    print("workloads finished...")
    # TODO: implement results handling based on what pipeline is run
    # convert xpum results to json
    convert_csv_results_to_json(results_dir, 'device')
    # convert igt results to json
    convert_csv_results_to_json(results_dir, 'igt')

if __name__ == '__main__':
    main()
