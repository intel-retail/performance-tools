'''
* Copyright (C) 2025 Intel Corporation.
*
* SPDX-License-Identifier: Apache-2.0
'''

import argparse
import csv
import json
import os

def parse_args():

    parser = argparse.ArgumentParser(
        prog='parse_docker_log', 
        description='parses docker output to json')
    parser.add_argument('--directory', '-d', 
                        default=os.path.join(os.curdir, 'results'),
                        help='full path to the directory with the results')
    parser.add_argument('--keyword', '-k', default=['device'], action='append',
                        help='keyword that results file(s) start with, ' +
                        'can be used multiple times')
    return parser.parse_args()


def parse_fps_from_log(results_dir, log_name):
    '''
    parses the log output for FPS information

    Args:
        results_dir: directory containing the benchmark results
        log_name: first portion of the log filename to search for
    '''
    for entry in os.scandir(results_dir):
        if entry.name.startswith(log_name) and entry.is_file():
            print(entry.path)
            fps_info = dict()
            count = 0
            sum_list = list()
            with open(entry, "r") as f:
                for line in f:
                    words = line.split()
                    if "FPS" in line:
                        count += 1
                        fps_indices = [i for i, x in enumerate(words) if x == "FPS"]
                        for fps in fps_indices:
                            #fps_info[words[fps - 1]] = float(words[fps + 1])
                            if "sum_%s" % words[fps - 1] not in fps_info:
                                fps_info["sum_%s" % words[fps - 1]] = 0
                                sum_list.append("sum_%s" % words[fps - 1])
                            fps_info["sum_%s" % words[fps - 1]] += float(words[fps + 1])
            total_fps = 0.0
            for sum in sum_list:
                name = sum.split('sum_')[-1]
                fps_info["avg_%s" % name] = fps_info[sum]/count
                total_fps += fps_info[sum]
                fps_info.pop(sum)
            fps_info["avg_fps"] = total_fps/(count * len(sum_list))
            # TODO: how do we want to save the FPS information?
            outfile = os.path.join(os.path.split(entry.path)[0], "%s.json" % entry.name.split(".")[0])
            with open(outfile, "w") as output:
                json.dump(fps_info, output)

def main():
    my_args = parse_args()
    for k in my_args.keyword:
        parse_fps_from_log(my_args.directory, k)


if __name__ == '__main__':
    main()