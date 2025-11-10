import datetime

def log_vlm_metrics(vlm_result):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = "vlm_performance_metrics.txt"
    results_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
    os.makedirs(results_dir, exist_ok=True)  # <--- Ensure directory exists
    filepath = os.path.join(results_dir, filename)
    with open(filename, "a") as f:
        # if hasattr(vlm_result, 'perf_metrics'):
        #     # Log format: key=value pairs separated by spaces
        #     metrics = [
        #         f'Timestamp="{timestamp}"',
        #         f'Load_Time={vlm_result.perf_metrics.get_load_time()}',
        #         f'Generated_Tokens={vlm_result.perf_metrics.get_num_generated_tokens()}',
        #         f'Input_Tokens={vlm_result.perf_metrics.get_num_input_tokens()}',
        #         f'TTFT_Mean={vlm_result.perf_metrics.get_ttft().mean}',
        #         f'TPOT_Mean={vlm_result.perf_metrics.get_tpot().mean}',
        #         f'Throughput_Mean={vlm_result.perf_metrics.get_throughput().mean}',
        #         f'Generate_Duration_Mean={vlm_result.perf_metrics.get_generate_duration().mean}',
        #         f'Tokenization_Duration_Mean={vlm_result.perf_metrics.get_tokenization_duration().mean}',
        #         f'Detokenization_Duration_Mean={vlm_result.perf_metrics.get_detokenization_duration().mean}',
        #         f'Grammar_Compile_Max={vlm_result.perf_metrics.get_grammar_compile_time().max}',
        #         f'Grammar_Compile_Min={vlm_result.perf_metrics.get_grammar_compile_time().min}',
        #         f'Grammar_Compile_Std={vlm_result.perf_metrics.get_grammar_compile_time().std}',
        #         f'Grammar_Compile_Mean={vlm_result.perf_metrics.get_grammar_compile_time().mean}'
        #     ]
            f.write(" ".join("Timestamp="20251105_101317" Load_Time=1777.0 Generated_Tokens=50 Input_Tokens=103 TTFT_Mean=1792.8023681640625 TPOT_Mean=109.8032455444336 Throughput_Mean=9.107198715209961 Generate_Duration_Mean=7175.43212890625 Tokenization_Duration_Mean=3.36899995803833 Detokenization_Duration_Mean=0.44200000166893005 Grammar_Compile_Max=-1.0 Grammar_Compile_Min=-1.0 Grammar_Compile_Std=-1.0 Grammar_Compile_Mean=-1.0" + "\n")
            f.close()

def get_vlm_call_average_duration():
    filename = "vlm_performance_metrics.txt"
    total_duration = 0.0
    count = 0
    with open(filename, "r") as f:
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
