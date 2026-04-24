"""
Order Accuracy Benchmark Script

Orchestrates benchmark execution for the Order Accuracy pipeline.
Integrates with performance-tools for metrics collection.

Usage:
    python benchmark_order_accuracy.py --compose_file ../../docker-compose.yaml --workers 2 --duration 300

Note: For stream density testing, use the application-specific scripts directly:
    - Take-Away: stream_density_latency_oa.py (RTSP/workers based)
    - Dine-In: stream_density_oa_dine_in.py (concurrent images based)
"""

import argparse
import contextlib
import io
import os
import sys
import time
import subprocess
import shlex
import json
import csv
from pathlib import Path
from typing import List, Dict, Optional

# Import from performance-tools benchmark scripts
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stream_density


class OrderAccuracyBenchmark:
    """
    Benchmark orchestrator for Order Accuracy pipeline.
    
    Runs N station workers for specified duration and collects metrics.
    Order Accuracy uses WORKERS to scale, not traditional pipelines.
    Each worker processes one RTSP station stream.
    
    For stream density testing, use application-specific scripts:
        - Take-Away: stream_density_latency_oa.py
        - Dine-In: stream_density_oa_dine_in.py
    """
    
    # Default configuration
    DEFAULT_INIT_DURATION = 120  # seconds
    DEFAULT_DURATION = 300  # seconds
    DEFAULT_WORKERS = 1
    
    def __init__(
        self,
        compose_files: List[str],
        results_dir: str,
        target_device: str = "GPU"
    ):
        self.compose_files = compose_files
        self.results_dir = os.path.abspath(results_dir)
        self.target_device = target_device
        self.env_vars = os.environ.copy()
        
        # Ensure results directory exists
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Configure environment
        self.env_vars["RESULTS_DIR"] = self.results_dir
        self.env_vars["log_dir"] = self.results_dir
        self.env_vars["DEVICE"] = target_device
        self.env_vars["TARGET_DEVICE"] = target_device
        self.env_vars["VLM_DEVICE"] = target_device
        self.env_vars["OPENVINO_DEVICE"] = target_device
        
    def docker_compose_cmd(
        self,
        action: str,
        compose_post_args: str = ""
    ) -> int:
        """
        Execute docker compose command.
        
        Args:
            action: Docker compose action (up, down, logs)
            compose_post_args: Additional arguments after action
            
        Returns:
            Exit code from docker compose
        """
        compose_file_args = " ".join(
            f"-f {shlex.quote(f)}" for f in self.compose_files
        )
        
        cmd = f"docker compose {compose_file_args} {action} {compose_post_args}"
        
        print(f"Executing: {cmd}")
        
        result = subprocess.run(
            cmd,
            shell=True,
            env=self.env_vars,
            capture_output=False
        )
        
        return result.returncode
    
    def run_fixed_workers(
        self,
        workers: int,
        init_duration: int,
        duration: int,
        profile: str = "parallel",
        iterations: int = 0,
        skip_export: bool = False
    ) -> Dict:
        """
        Run benchmark with fixed number of station workers.
        
        Args:
            workers: Number of concurrent station workers
            init_duration: Warmup duration in seconds
            duration: Benchmark duration in seconds (ignored if iterations > 0)
            profile: Docker compose profile to use (parallel for take-away, benchmark for dine-in)
            iterations: Number of iterations per worker (0 = use duration-based)
            
        Returns:
            Dictionary with benchmark results
        """
        is_iteration_mode = iterations > 0
        
        print(f"\n{'='*60}")
        print(f"Order Accuracy Benchmark - Fixed Workers Mode")
        print(f"Workers: {workers}")
        print(f"Profile: {profile}")
        if is_iteration_mode:
            print(f"Iterations: {iterations} per worker")
        else:
            print(f"Init Duration: {init_duration}s")
            print(f"Benchmark Duration: {duration}s")
        print(f"{'='*60}\n")
        
        # Clean previous logs
        self._clean_pipeline_logs()
        
        # Set workers count for order-accuracy
        self.env_vars["WORKERS"] = str(workers)
        self.env_vars["VLM_WORKERS"] = str(workers)
        self.env_vars["SERVICE_MODE"] = "parallel"
        
        # Set iterations for dine-in mode
        if is_iteration_mode:
            self.env_vars["ITERATIONS"] = str(iterations)
        
        # Start containers with specified profile
        print("Starting containers...")
        self.docker_compose_cmd(f"--profile {profile} up", "-d")
        
        # Wait for initialization
        print(f"Waiting {init_duration}s for initialization...")
        time.sleep(init_duration)
        
        if is_iteration_mode:
            # Wait for workers to complete iterations
            print(f"Waiting for workers to complete {iterations} iterations...")
            self._wait_for_workers_completion(workers, profile, timeout=duration if duration > 0 else 3600)
        else:
            # Run benchmark for fixed duration
            print(f"Running benchmark for {duration}s...")
            time.sleep(duration)
        
        # Collect metrics
        results = self._collect_metrics(workers)
        
        # Collect VLM metrics from vlm_metrics_logger
        results["vlm_metrics"] = self._collect_vlm_logger_metrics()
        
        # Derive vlm_inference from vlm_logger data when direct log parsing yields nothing
        # (order-accuracy does not emit the GStreamer vlm*.log files that _collect_vlm_metrics parses)
        if results["vlm_inference"].get("inference_count", 0) == 0:
            vm = results.get("vlm_metrics", {})
            if vm.get("total_transactions", 0) > 0:
                results["vlm_inference"] = {
                    "inference_count": vm["total_transactions"],
                    "avg_inference_ms": vm["avg_latency_ms"],
                    "tokens_per_second": vm["avg_tps"]
                }
        
        # Collect dine-in results if in iteration mode
        if is_iteration_mode:
            results["worker_results"] = self._collect_worker_results()
        
        # Compute FPS for image-based workflows (no GStreamer pipeline logs exist)
        if results["fps"].get("total", 0) == 0:
            if is_iteration_mode:
                wr = results.get("worker_results", {})
                if wr.get("successful", 0) > 0 and wr.get("avg_latency_ms", 0) > 0:
                    # Each worker processes one image at a time; throughput = workers / avg_latency_s
                    fps = workers * 1000.0 / wr["avg_latency_ms"]
                    results["fps"] = {
                        "total": round(fps, 2),
                        "per_stream": round(fps / workers, 2),
                        "per_pipeline": {"pipeline_stream": round(fps / workers, 2)}
                    }
            else:
                vm = results.get("vlm_metrics", {})
                if vm.get("total_transactions", 0) > 0 and duration > 0:
                    # Duration-based mode: use VLM transaction count over benchmark duration
                    fps = vm["total_transactions"] / duration
                    results["fps"] = {
                        "total": round(fps, 2),
                        "per_stream": round(fps / workers, 2),
                        "per_pipeline": {"pipeline_stream": round(fps / workers, 2)}
                    }
        
        # Derive latency from vlm_metrics when GStreamer pipeline logs are absent
        if results.get("latency", {}).get("total_ms", 0) == 0:
            vm = results.get("vlm_metrics", {})
            if vm.get("avg_latency_ms", 0) > 0:
                results["latency"] = {
                    "total_ms": vm["avg_latency_ms"],
                    "per_stream_ms": vm["avg_latency_ms"]
                }
        
        # Stop containers
        print("Stopping containers...")
        self.docker_compose_cmd(f"--profile {profile} down")
        
        # Export results
        if not skip_export:
            self._export_results(results, "fixed_workers")
        
        return results
    
    def _wait_for_workers_completion(self, workers: int, profile: str, timeout: int = 3600):
        """Wait for all worker containers to complete."""
        import subprocess
        
        start_time = time.time()
        check_interval = 10  # seconds
        
        while time.time() - start_time < timeout:
            # Check if any worker containers are still running
            cmd = f"docker compose {' '.join(f'-f {f}' for f in self.compose_files)} --profile {profile} ps --format json"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=self.env_vars)
            
            if result.returncode != 0:
                print(f"Warning: Could not check container status: {result.stderr}")
                time.sleep(check_interval)
                continue
            
            # Parse JSON output (docker compose ps outputs one JSON per line)
            running_workers = 0
            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue
                try:
                    container = json.loads(line)
                    name = container.get('Name', container.get('name', ''))
                    state = container.get('State', container.get('state', ''))
                    if 'worker' in name.lower() and state == 'running':
                        running_workers += 1
                except json.JSONDecodeError:
                    continue
            
            if running_workers == 0:
                print("All workers completed.")
                return
            
            print(f"  {running_workers} workers still running...")
            time.sleep(check_interval)
        
        print(f"Warning: Timeout reached after {timeout}s. Workers may not have completed.")
    
    def _collect_worker_results(self) -> Dict:
        """Collect results from dine-in worker output files."""
        import glob
        
        worker_results = {
            "total_iterations": 0,
            "successful": 0,
            "failed": 0,
            "avg_latency_ms": 0.0,
            "results_files": [],
            "details": []
        }
        
        # Look for worker result files (only worker_*.json pattern to avoid duplicates)
        seen_files = set()
        for f in glob.glob(os.path.join(self.results_dir, "worker_*.json")):
            if f in seen_files:
                continue
            seen_files.add(f)
            try:
                with open(f, 'r') as fp:
                    data = json.load(fp)
                    worker_results["results_files"].append(f)
                    
                    # Worker results have stats at root level, not in "stats" key
                    worker_results["total_iterations"] += data.get("total_iterations", 0)
                    worker_results["successful"] += data.get("successful_iterations", 0)
                    worker_results["failed"] += data.get("failed_iterations", 0)
                    
                    # Collect detailed results
                    if "results" in data:
                        for result in data["results"]:
                            worker_results["details"].append({
                                "worker_id": result.get("worker_id"),
                                "order_id": result.get("order_id"),
                                "success": result.get("success"),
                                "order_complete": result.get("order_complete"),
                                "accuracy_score": result.get("accuracy_score"),
                                "items_detected": result.get("items_detected"),
                                "items_expected": result.get("items_expected"),
                                "missing_items": result.get("missing_items", 0),
                                "extra_items": result.get("extra_items", 0),
                                "missing_items_list": result.get("missing_items_list", []),
                                "extra_items_list": result.get("extra_items_list", []),
                                "total_latency_ms": result.get("total_latency_ms"),
                                "tps": result.get("tps")
                            })
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not read results from {f}: {e}")
        
        # Calculate average latency from root level avg_latency_ms
        if worker_results["successful"] > 0:
            total_latency = 0
            count = 0
            for f in worker_results["results_files"]:
                try:
                    with open(f, 'r') as fp:
                        data = json.load(fp)
                        if "avg_latency_ms" in data:
                            total_latency += data["avg_latency_ms"]
                            count += 1
                except:
                    pass
            if count > 0:
                worker_results["avg_latency_ms"] = total_latency / count
        
        return worker_results
    
    def _clean_pipeline_logs(self):
        """Remove previous pipeline log and results files to prevent stale data."""
        import glob
        
        patterns = [
            "pipeline*.log",
            "gst*.log",
            "vlm*.log",
            "latency*.json",
            # Worker result files from previous runs — must be cleaned or _collect_worker_results
            # will aggregate both old and new data, producing inflated/incorrect metrics.
            "worker_*.json",
            "vlm_application_metrics_*.txt",
            "vlm_performance_metrics_*.txt",
        ]
        
        for pattern in patterns:
            for f in glob.glob(os.path.join(self.results_dir, pattern)):
                try:
                    os.remove(f)
                except OSError:
                    pass
        
        # Also clean VLM metrics from storage/results (compose-adjacent path)
        compose_dir = os.path.dirname(self.compose_files[0]) if self.compose_files else "."
        storage_results_dir = os.path.join(compose_dir, "storage", "results")
        for pattern in ["vlm_application_metrics_*.txt", "vlm_performance_metrics_*.txt"]:
            for f in glob.glob(os.path.join(storage_results_dir, pattern)):
                try:
                    os.remove(f)
                except OSError:
                    pass
    
    def _collect_metrics(self, num_pipelines: int) -> Dict:
        """
        Collect metrics from pipeline logs.
        
        Args:
            num_pipelines: Number of pipelines run
            
        Returns:
            Dictionary with collected metrics
        """
        metrics = {
            "num_pipelines": num_pipelines,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "fps": {},
            "latency": {},
            "vlm_inference": {}
        }
        
        # Calculate FPS from logs
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                total_fps, fps_per_stream, fps_dict = stream_density.calculate_total_fps(
                    num_pipelines,
                    self.results_dir,
                    "order-accuracy"
                )
            metrics["fps"] = {
                "total": total_fps,
                "per_stream": fps_per_stream,
                "per_pipeline": fps_dict
            }
        except Exception as e:
            print(f"Warning: Could not calculate FPS: {e}")
        
        # Calculate latency from logs
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                total_latency, latency_per_stream = stream_density.calculate_pipeline_latency(
                    num_pipelines,
                    self.results_dir,
                    "order-accuracy"
                )
            metrics["latency"] = {
                "total_ms": total_latency,
                "per_stream_ms": latency_per_stream
            }
        except Exception as e:
            print(f"Warning: Could not calculate latency: {e}")
        
        # Collect VLM-specific metrics
        metrics["vlm_inference"] = self._collect_vlm_metrics()
        
        return metrics
    
    def _collect_vlm_metrics(self) -> Dict:
        """
        Collect VLM inference metrics from OVMS logs.
        
        Returns:
            Dictionary with VLM metrics
        """
        vlm_metrics = {
            "inference_count": 0,
            "avg_inference_ms": 0.0,
            "tokens_per_second": 0.0
        }
        
        # Parse VLM log files
        vlm_log_pattern = os.path.join(self.results_dir, "vlm*.log")
        import glob
        
        inference_times = []
        
        for log_file in glob.glob(vlm_log_pattern):
            try:
                with open(log_file, 'r') as f:
                    for line in f:
                        if "inference_time" in line.lower():
                            # Extract inference time from log line
                            # Format: "inference_time=XXXms" or similar
                            import re
                            match = re.search(r'inference_time[=:]?\s*(\d+\.?\d*)', line)
                            if match:
                                inference_times.append(float(match.group(1)))
            except (IOError, OSError):
                continue
        
        if inference_times:
            vlm_metrics["inference_count"] = len(inference_times)
            vlm_metrics["avg_inference_ms"] = sum(inference_times) / len(inference_times)
        
        return vlm_metrics
    
    def _collect_vlm_logger_metrics(self) -> Dict:
        """
        Collect metrics from vlm_metrics_logger output files.
        
        Parses vlm_application_metrics_*.txt for:
          - Start/end timestamps per transaction (unique_id = station_id_order_id)
          - OVMS call latencies
          - Tokens per second
        
        Returns:
            Dictionary with aggregated VLM metrics
        """
        import glob
        import re
        
        metrics = {
            "transactions": [],
            "total_transactions": 0,
            "avg_latency_ms": 0.0,
            "avg_tps": 0.0,
            "p95_latency_ms": 0.0
        }
        
        # Find vlm_application_metrics files in storage/results (relative to compose file)
        # The vlm_metrics_logger writes to CONTAINER_RESULTS_PATH which maps to storage/results
        compose_dir = os.path.dirname(self.compose_files[0]) if self.compose_files else "."
        storage_results_dir = os.path.join(compose_dir, "storage", "results")
        
        # Try storage/results first, fallback to results_dir
        pattern = os.path.join(storage_results_dir, "vlm_application_metrics_*.txt")
        log_files = glob.glob(pattern)
        
        if not log_files:
            # Fallback to results_dir
            pattern = os.path.join(self.results_dir, "vlm_application_metrics_*.txt")
            log_files = glob.glob(pattern)
        
        if not log_files:
            print(f"No vlm_metrics_logger files found in {storage_results_dir} or {self.results_dir}")
            return metrics
        
        start_times = {}  # unique_id -> timestamp_ms
        end_times = {}    # unique_id -> timestamp_ms
        tps_values = []   # tokens per second
        
        for log_file in log_files:
            try:
                with open(log_file, 'r') as f:
                    for line in f:
                        # Parse: id=<unique_id> event=start timestamp_ms=1234567890
                        # Use \S+ to capture IDs that may contain dashes (e.g., dine_in_MCD-1001)
                        id_match = re.search(r'id=(\S+)', line)
                        event_match = re.search(r'event=(\w+)', line)
                        ts_match = re.search(r'timestamp_ms=(\d+)', line)
                        
                        if id_match and event_match and ts_match:
                            unique_id = id_match.group(1)
                            event = event_match.group(1)
                            timestamp = int(ts_match.group(1))
                            
                            if event == "start":
                                start_times[unique_id] = timestamp
                            elif event == "end":
                                end_times[unique_id] = timestamp
                        
                        # Parse TPS from ovms_vlm_request event.
                        # Dine-in emits tps=<val>; take-away emits throughput_mean=<val> (tokens/sec)
                        if "ovms_vlm_request" in line:
                            tps_match = re.search(r'(?:tps|throughput_mean)=([\d.]+)', line)
                            if tps_match:
                                tps_values.append(float(tps_match.group(1)))
            except (IOError, OSError) as e:
                print(f"Warning: Could not read {log_file}: {e}")
        
        # Calculate latencies
        latencies = []
        for unique_id in start_times:
            if unique_id in end_times:
                latency_ms = end_times[unique_id] - start_times[unique_id]
                latencies.append(latency_ms)
                metrics["transactions"].append({
                    "id": unique_id,
                    "latency_ms": latency_ms
                })
        
        metrics["total_transactions"] = len(latencies)
        
        if latencies:
            metrics["avg_latency_ms"] = sum(latencies) / len(latencies)
            sorted_latencies = sorted(latencies)
            p95_idx = int(len(sorted_latencies) * 0.95)
            metrics["p95_latency_ms"] = sorted_latencies[p95_idx] if p95_idx < len(sorted_latencies) else sorted_latencies[-1]
        
        if tps_values:
            metrics["avg_tps"] = sum(tps_values) / len(tps_values)
        
        return metrics

    def _export_results(self, results: Dict, prefix: str):
        """
        Export benchmark results to JSON and CSV.
        
        Args:
            results: Results dictionary
            prefix: Filename prefix
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Export JSON
        json_path = os.path.join(
            self.results_dir,
            f"{prefix}_results_{timestamp}.json"
        )
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results exported to: {json_path}")
        
        # Export CSV summary
        csv_path = os.path.join(
            self.results_dir,
            f"{prefix}_summary_{timestamp}.csv"
        )
        self._write_csv_summary(results, csv_path)
        print(f"Summary exported to: {csv_path}")
    
    def _write_csv_summary(self, results: Dict, csv_path: str):
        """Write results summary to CSV."""
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "Value"])
            
            def flatten_dict(d, prefix=""):
                for k, v in d.items():
                    key = f"{prefix}{k}" if prefix else k
                    if isinstance(v, dict):
                        yield from flatten_dict(v, f"{key}_")
                    else:
                        yield (key, v)
            
            for metric, value in flatten_dict(results):
                writer.writerow([metric, value])


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Order Accuracy Benchmark Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Take-Away: Run with 2 workers for 5 minutes (RTSP streams)
  python benchmark_order_accuracy.py --compose_file ../../docker-compose.yaml --workers 2 --duration 300
  
  # Dine-In: Run with 2 workers, 10 iterations each (image-based)
  python benchmark_order_accuracy.py --compose_file ../../docker-compose.yml --workers 2 --iterations 10 --profile benchmark
  
  # Dine-In: Quick test with 1 iteration
  python benchmark_order_accuracy.py --compose_file ../../docker-compose.yml --workers 1 --iterations 1 --profile benchmark --skip_perf_tools

For stream density testing, use application-specific scripts:
  # Take-Away (RTSP/workers based)
  python stream_density_latency_oa.py --compose_files ../../docker-compose.yaml
  
  # Dine-In (concurrent images based)
  python stream_density_oa_dine_in.py --compose_file ../../docker-compose.yml
        """
    )
    
    parser.add_argument(
        '--compose_file',
        type=str,
        nargs='+',
        required=True,
        help='Docker compose file(s) to use'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='Number of station workers to run'
    )
    
    parser.add_argument(
        '--profile',
        type=str,
        default='parallel',
        help='Docker compose profile to use (parallel for take-away, benchmark for dine-in)'
    )
    
    parser.add_argument(
        '--iterations',
        type=int,
        default=0,
        help='Number of iterations per worker (dine-in mode). 0 = use duration-based.'
    )
    
    parser.add_argument(
        '--init_duration',
        type=int,
        default=120,
        help='Initialization duration in seconds'
    )
    
    parser.add_argument(
        '--duration',
        type=int,
        default=300,
        help='Benchmark duration in seconds (ignored if --iterations > 0)'
    )
    
    parser.add_argument(
        '--results_dir',
        type=str,
        default=os.path.join(os.curdir, 'results'),
        help='Directory for results output'
    )
    
    parser.add_argument(
        '--target_device',
        type=str,
        default='GPU',
        choices=['CPU', 'GPU', 'NPU'],
        help='Target inference device'
    )
    
    parser.add_argument(
        '--skip_perf_tools',
        action='store_true',
        help='Skip adding performance-tools docker-compose.yaml (avoids benchmark container build)'
    )
    
    parser.add_argument(
        '--skip_export',
        action='store_true',
        help='Skip exporting fixed_workers results JSON/CSV (useful when metrics come from app-level reports)'
    )
    parser.add_argument('--parser_script', 
                        default=os.path.join(os.path.curdir, 'parse_qmassa_metrics_to_json.py'), 
                        help='full path to the parsing script to obtain FPS')
    parser.add_argument('--parser_args', default='-k device -k qmassa', 
                        help='arguments to pass to the parser script, ' + 
                        'pass args with spaces in quotes: "args with spaces"')
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Resolve compose files
    compose_files = [os.path.abspath(f) for f in args.compose_file]
    env_vars = os.environ.copy()
    # Validate compose files exist
    for cf in compose_files:
        if not os.path.exists(cf):
            print(f"Error: Compose file not found: {cf}")
            sys.exit(1)
    
    # Add benchmark compose file from performance-tools (unless skipped)
    if not args.skip_perf_tools:
        benchmark_compose = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', 'docker', 'docker-compose.yaml')
        )
        if os.path.exists(benchmark_compose):
            compose_files.append(benchmark_compose)
    
    # Create benchmark instance
    benchmark = OrderAccuracyBenchmark(
        compose_files=compose_files,
        results_dir=args.results_dir,
        target_device=args.target_device
    )
    
    # Run fixed workers benchmark
    print(f"Running benchmark with {args.workers} worker(s)...")
    results = benchmark.run_fixed_workers(
        workers=args.workers,
        init_duration=args.init_duration,
        duration=args.duration,
        profile=args.profile,
        iterations=args.iterations,
        skip_export=args.skip_export
    )
    if not args.skip_export:
        print(f"\nBenchmark complete. Results: {results}")
    else:
        print("\nBenchmark complete.")

    try:
        # Only pass -k <keyword> for keywords that have matching *tool-generated.json files
        # in results_dir; avoids spurious "No files found" warnings for absent hardware counters.
        all_keywords = []
        parts = shlex.split(args.parser_args)
        i = 0
        while i < len(parts):
            if parts[i] == '-k' and i + 1 < len(parts):
                all_keywords.append(parts[i + 1])
                i += 2
            else:
                i += 1

        active_keywords = []
        if os.path.isdir(args.results_dir):
            for entry in os.scandir(args.results_dir):
                if entry.is_file() and entry.name.endswith('tool-generated.json'):
                    for kw in all_keywords:
                        if entry.name.startswith(kw) and kw not in active_keywords:
                            active_keywords.append(kw)

        effective_parser_args = (
            ' '.join(f'-k {kw}' for kw in active_keywords)
            if active_keywords else args.parser_args  # fall back so script can diagnose
        )

        parser_string = ("python3 %s -d %s %s" % (args.parser_script, args.results_dir, effective_parser_args))
        cmd_args = shlex.split(parser_string)

        subprocess.run(cmd_args,
                       check=True, env=env_vars)  # nosec B404, B603
    except subprocess.CalledProcessError:
        print("Exception calling %s\n parser %s: %s" %
              (parser_string, args.parser_script, traceback.format_exc()))

if __name__ == '__main__':
    main()
