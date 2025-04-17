'''
* Copyright (C) 2025 Intel Corporation.
*
* SPDX-License-Identifier: Apache-2.0
'''

import datetime
import os
import time

# use lspci | grep -i npu to get the right npu path for the system
NPU_PATH = os.getenv("NPU_PATH", "/sys/devices/pci0000:00/0000:00:0b.0/power/runtime_active_time")
NPU_LOG = os.getenv("NPU_LOG", "npu_log.csv")

def read_npu_runtime(path):
    try:
        with open(path, "r") as f:
            return int(f.read().strip())
    except Exception as e:
        print(f"Error reading NPU runtime: {e}")
        return None
    
def main():
    print(f"Logging NPU usage to {NPU_LOG} (Ctrl+C to stop)...")
    with open(NPU_LOG, "w") as f:
        f.write("timestamp,percent_usage\n")

        prev_runtime = read_npu_runtime(NPU_PATH)
        prev_time = datetime.datetime.now()

        if prev_runtime is None:
            print("Failed to read initial NPU Runtime. Exiting...")
            return
        
        while True:
            try: 
                time.sleep(1)

                current_runtime = read_npu_runtime(NPU_PATH)
                current_time = datetime.datetime.now()

                if current_runtime is None:
                    continue

                delta_runtime = current_runtime - prev_runtime
                # grab the change in time in milliseconds
                delta_time = (current_time - prev_time).total_seconds() * 1000 

                percent_usage = 0.0
                if delta_time > 0:
                    percent_usage = delta_runtime/delta_time * 100
                
                f.write("%s,%.2f\n" % (current_time.isoformat(), percent_usage))

                prev_runtime = current_runtime
                prev_time = current_time
            except KeyboardInterrupt:
                print("\nStopped logging.")

if __name__ == "__main__":
    main()