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
        prog='parse_csv_to_json', 
        description='parses csv output to json')
    parser.add_argument('--directory', '-d', 
                        default=os.path.join(os.curdir, 'results'),
                        help='full path to the directory with the results')
    parser.add_argument('--keyword', '-k', default=['device'], action='append',
                        help='keyword that results file(s) start with, ' +
                        'can be used multiple times')
    return parser.parse_args()


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
    my_args = parse_args()
    for k in my_args.keyword:
        convert_csv_results_to_json(my_args.directory, k)


if __name__ == '__main__':
    main()