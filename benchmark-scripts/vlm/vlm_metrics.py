import datetime
import os
import sys
import uuid

_unique_file_name = None

def get_unique_file_name():
    global _unique_file_name
    if _unique_file_name is None:
        # First call: create a new unique name
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
        unique_id = uuid.uuid4().hex[:6]
        _unique_file_name = f"vlm_performance_metrics_{timestamp}_{unique_id}.txt"
    else:
        # Subsequent calls reuse it
        print(f"Reusing existing file: {_unique_file_name}")
    return _unique_file_name
    
def log_vlm_metrics(vlm_result):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") 
    filename = get_unique_file_name()
    results_dir = os.getenv("CONTAINER_RESULTS_PATH")
    os.makedirs(results_dir, exist_ok=True)  # <--- Ensure directory exists
    filepath = os.path.join(results_dir, filename)
    print("The file path is: ", filepath)
    with open(filepath, "a") as f:
         if hasattr(vlm_result, 'perf_metrics'):
             # Log format: key=value pairs separated by spaces
             metrics = [
                 f'Timestamp="{timestamp}"',
                 f'Load_Time={vlm_result.perf_metrics.get_load_time()}',
                 f'Generated_Tokens={vlm_result.perf_metrics.get_num_generated_tokens()}',
                 f'Input_Tokens={vlm_result.perf_metrics.get_num_input_tokens()}',
                 f'TTFT_Mean={vlm_result.perf_metrics.get_ttft().mean}',
                 f'TPOT_Mean={vlm_result.perf_metrics.get_tpot().mean}',
                 f'Throughput_Mean={vlm_result.perf_metrics.get_throughput().mean}',
                 f'Generate_Duration_Mean={vlm_result.perf_metrics.get_generate_duration().mean}',
                 f'Tokenization_Duration_Mean={vlm_result.perf_metrics.get_tokenization_duration().mean}',
                 f'Detokenization_Duration_Mean={vlm_result.perf_metrics.get_detokenization_duration().mean}',
                 f'Grammar_Compile_Max={vlm_result.perf_metrics.get_grammar_compile_time().max}',
                 f'Grammar_Compile_Min={vlm_result.perf_metrics.get_grammar_compile_time().min}',
                 f'Grammar_Compile_Std={vlm_result.perf_metrics.get_grammar_compile_time().std}',
                 f'Grammar_Compile_Mean={vlm_result.perf_metrics.get_grammar_compile_time().mean}'
             ]
            f.write(" ".join(metrics) + "\n")
            f.close()

def get_vlm_call_average_duration():
    filename = "vlm_performance_metrics.txt"
    results_dir = os.getenv("RESULTS_DIR")
    filepath = os.path.join(results_dir, filename)
    total_duration = 0.0
    count = 0
    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split()
            for part in parts:
                if part.startswith("Generate_Duration_Mean="):
                    duration_str = part.split("=")[1]
                    try:
                        duration = float(duration_str)
                        total_duration += duration
                        count += 1
                    except ValueError:
                        continue
    if count == 0:
        return 0, 0.0
    
    average_duration = total_duration / count
    return count, average_duration
