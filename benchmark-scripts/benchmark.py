'''
* Copyright (C) 2024 Intel Corporation.
*
* SPDX-License-Identifier: Apache-2.0
'''

import argparse
import os
import shlex
import subprocess
import time
import traceback


def parse_args(print=False):
    parser = argparse.ArgumentParser(
        prog='benchmark',
        description='runs benchmarking processes',
        epilog='Note:\n' +
               '\n\t1. dgpu.x should be replaced with targeted GPUs such as dgpu (for all GPUs), dgpu.0, dgpu.1, etc' +
               '\n\t2. filesrc will utilize videos stored in the sample-media folder' +
               '\n\t3. Set environment variable STREAM_DENSITY_MODE=1 for starting single container ' +
               'stream density testing' +
               '\n\t4. Set environment variable RENDER_MODE=1 for displaying pipeline and overlay CV metadata' +
               '\n\t5. Stream density can take two parameters: first one TARGET_FPS is for target fps, a float type ' +
               'value, and the second one PIPELINE_INCREMENT is increment integer of pipelines and is optional (in ' +
               'which case the increments will be dynamically adjusted internally)\n')
    parser.add_argument('--pipelines', type=int, default=1, help='number of pipelines')
    parser.add_argument('--target_fps', type=int, default=None, help='stream density target FPS')
    # TODO: add variable for stream density increment when implemnting stream_density
    parser.add_argument('--results_dir', default=os.path.join(os.curdir, 'results'),
                        help='full path to the desired directory for logs and results')
    parser.add_argument('--duration', type=int, default=30,
                        help='time in seconds, not needed when --stream_density is specified')
    parser.add_argument('--init_duration', type=int, default=5, help='time in seconds')
    # TODO: change target_device to an env variable in docker compose, then set it when calling compose
    parser.add_argument('--target_device', default='CPU',
                        help='desired running platform [cpu|core|xeon|dgpu.x]')
    parser.add_argument('--compose_file', default=None, action='append',
                        help='path to docker compose files "/path/to/pipeline-compose.yml". can be used multiple times')
    parser.add_argument('--retail_use_case_root', default=os.path.join(os.curdir, '..', '..', 'retail-use-cases'),
                        help='full path to the retail-use-cases repo root')
    if print:
        parser.print_help()
        return
    return parser.parse_args()


def docker_compose_containers(command, compose_files=[], compose_pre_args="", compose_post_args="",
                              env_vars=os.environ.copy()):
    try:
        files = " -f ".join(compose_files)
        compose_string = "docker compose %s -f %s %s %s" % (compose_pre_args, files, command, compose_post_args)
        compose_args = shlex.split(compose_string)

        p = subprocess.Popen(compose_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env_vars)
        stdout, stderr = p.communicate()

        if p.returncode and stderr:
            print("Error bringing %s the compose files: %s" % (command, stderr))
        return stdout.strip(), stderr, p.returncode
    except:
        print("Exception bringing %s the compose files: %s" % (command, traceback.format_exc()))


def main():
    my_args = parse_args()

    results_dir = os.path.abspath(my_args.results_dir)
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    print("Starting workload(s)")

    # set replica count to my_args.pipelines and use env variable for replica in docker compose file
    # pass in all variables needed from the input args
    # start the docker containers
    compose_files = []
    for file in my_args.compose_file:
        compose_files.append(os.path.abspath(file))
    compose_files.append(os.path.abspath(os.path.join(os.curdir, '..', 'docker', 'docker-compose.yaml')))
    env_vars = os.environ.copy()
    env_vars["log_dir"] = results_dir
    env_vars["RESULTS_DIR"] = results_dir
    env_vars["DEVICE"] = my_args.target_device
    retail_use_case_root = os.path.abspath(my_args.retail_use_case_root)
    env_vars["RETAIL_USE_CASE_ROOT"] = retail_use_case_root
    if my_args.pipelines > 0:
        env_vars["PIPELINE_COUNT"] = str(my_args.pipelines)

    docker_compose_containers("up", compose_files=compose_files, compose_post_args="-d", env_vars=env_vars)
    print("Waiting for init duration to complete...")
    time.sleep(my_args.init_duration)

    # use duration to sleep
    print("Waiting for %d seconds for workload to finish" % my_args.duration)
    time.sleep(my_args.duration)
    # stop all containers and camera-simulator
    docker_compose_containers("down", compose_files=compose_files, env_vars=env_vars)

    # collect metrics using copy-platform-metrics
    print("workloads finished...")
    # TODO: implement results handling based on what pipeline is run


if __name__ == '__main__':
    main()
