import json
import argparse
import os


def parse_args():

    parser = argparse.ArgumentParser(
        prog='parse_qmassa_metrics_to_json', 
        description='parses qmassa metrics output to json')
    parser.add_argument('--directory', '-d', 
                        default=os.path.join(os.curdir, 'results'),
                        help='full path to the directory with the results')
    parser.add_argument('--keyword', '-k', default=['device'], action='append',
                        help='keyword that results file(s) start with, ' +
                        'can be used multiple times')
    return parser.parse_args()
    
def parse_qmassa_files(results_dir,log_name):
    for entry in os.scandir(results_dir):
        if entry.name.startswith(log_name) and entry.name.endswith('tool-generated.json') and entry.is_file():
            # Split the filename by '-'
            file_name_split = entry.name.split('-')
            driver_name = file_name_split[2]
            if driver_name == "xe":
                parse_qmassa_metrics_driver_xe(results_dir, entry)
            elif driver_name == "i915":
                parse_qmassa_metrics_driver_i915(results_dir, entry)
            else:
                print(f"Unsupported driver: {driver_name}")

def parse_qmassa_metrics_driver_i915(results_dir, file_obj):
    
    metrics_list = []
    with open(file_obj.path, 'r') as file:
        data = json.load(file)            
    states = data.get('states')
    for state in states:
        devs_state = state.get('devs_state')
        for dev in devs_state:
            metrics = {}
            eng_usage_list = dev.get('dev_stats').get('eng_usage')
            power_usage_last_record = (dev.get('dev_stats').get('power')[-1])
            metrics["Power Usage (W)"] = str(round(power_usage_last_record.get('pkg_cur_power'),6))
            metrics["CCS %"] = str(round(eng_usage_list.get('compute')[-1],6))
            metrics["VECS %"] = str(round(eng_usage_list.get('video-enhance')[-1],6))
            metrics["RCS %"] = str(round(eng_usage_list.get('render')[-1],6))
            metrics["BCS %"] = str(round(eng_usage_list.get('copy')[-1],6))
            metrics["VCS %"] = str(round(eng_usage_list.get('video')[-1],6))
            metrics_list.append(metrics)
    # Write the output to a JSON file
    device_name = file_obj.name.replace('tool-generated','parsed')
    json_result_path = os.path.join(
        results_dir, device_name)
    with open(json_result_path, 'w') as outfile:
        json.dump(metrics_list, outfile, indent=4)

def parse_qmassa_metrics_driver_xe(results_dir, file_obj):
    
    metrics_list = []
    with open(file_obj.path, 'r') as file:
        data = json.load(file)            
    states = data.get('states')
    for state in states:
        devs_state = state.get('devs_state')
        for dev in devs_state:
            metrics = {}
            eng_usage_list = dev.get('dev_stats').get('eng_usage')
            power_usage_last_record = (dev.get('dev_stats').get('power')[-1])
            metrics["Power Usage (W)"] = str(round(power_usage_last_record.get('pkg_cur_power'),6))
            metrics["CCS %"] = str(round(eng_usage_list.get('ccs')[-1],6))
            metrics["VECS %"] = str(round(eng_usage_list.get('vecs')[-1],6))
            metrics["RCS %"] = str(round(eng_usage_list.get('rcs')[-1],6))
            metrics["BCS %"] = str(round(eng_usage_list.get('bcs')[-1],6))
            metrics["VCS %"] = str(round(eng_usage_list.get('vcs')[-1],6))
            metrics_list.append(metrics)
    # Write the output to a JSON file
    device_name = file_obj.name.replace('tool-generated','parsed')
    json_result_path = os.path.join(
        results_dir, device_name)
    with open(json_result_path, 'w') as outfile:
        json.dump(metrics_list, outfile, indent=4)

def main():
    my_args = parse_args()

    if not os.path.isdir(my_args.directory):
        print(f"Error: Directory '{my_args.directory}' does not exist.")
        return

    found_file = False
    for k in my_args.keyword:
        # Check if any file matches the pattern before processing
        matching_files = [entry for entry in os.scandir(my_args.directory)
                          if entry.name.startswith(k) and entry.name.endswith('tool-generated.json') and entry.is_file()]
        if not matching_files:
            print(f"Warning: No files starting with '{k}' and ending with 'tool-generated.json' found in '{my_args.directory}'.")
        else:
            found_file = True
            parse_qmassa_files(my_args.directory, k)
    if not found_file:
        print("No matching files found for any keyword. Exiting.")

if __name__ == '__main__':
    main()
