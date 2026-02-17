"""
Order Accuracy Benchmark Script

Orchestrates benchmark execution for the Order Accuracy pipeline.
Integrates with performance-tools for metrics collection and stream density analysis.

Usage:
    python benchmark_order_accuracy.py --compose_file ../../docker-compose.yaml --pipelines 1
    python benchmark_order_accuracy.py --compose_file ../../docker-compose.yaml --target_fps 14.95
"""

import argparse
import os
import sys
import time
import subprocess
import shlex
import json
import csv
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# Import from performance-tools benchmark scripts
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stream_density
from stream_density_latency_oa import OrderAccuracyStreamDensity


class OrderAccuracyBenchmark:
    """
    Benchmark orchestrator for Order Accuracy pipeline.
    
    Supports two modes:
    1. Fixed workers: Runs N station workers for specified duration
    2. Stream density: Finds maximum workers that maintain target throughput
    
    Order Accuracy uses WORKERS to scale, not traditional pipelines.
    Each worker processes one RTSP station stream.
    """
    
    # Default configuration
    DEFAULT_INIT_DURATION = 120  # seconds
    DEFAULT_DURATION = 300  # seconds
    DEFAULT_TARGET_FPS = 14.95
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
        duration: int
    ) -> Dict:
        """
        Run benchmark with fixed number of station workers.
        
        Args:
            workers: Number of concurrent station workers (RTSP streams)
            init_duration: Warmup duration in seconds
            duration: Benchmark duration in seconds
            
        Returns:
            Dictionary with benchmark results
        """
        print(f"\n{'='*60}")
        print(f"Order Accuracy Benchmark - Fixed Workers Mode")
        print(f"Workers: {workers}")
        print(f"Init Duration: {init_duration}s")
        print(f"Benchmark Duration: {duration}s")
        print(f"{'='*60}\n")
        
        # Clean previous logs
        self._clean_pipeline_logs()
        
        # Set workers count for order-accuracy
        self.env_vars["WORKERS"] = str(workers)
        self.env_vars["VLM_WORKERS"] = str(workers)
        self.env_vars["SERVICE_MODE"] = "parallel"
        
        # Start containers with parallel profile
        print("Starting containers...")
        self.docker_compose_cmd("--profile parallel up", "-d")
        
        # Wait for initialization
        print(f"Waiting {init_duration}s for initialization...")
        time.sleep(init_duration)
        
        # Run benchmark
        print(f"Running benchmark for {duration}s...")
        time.sleep(duration)
        
        # Collect metrics
        results = self._collect_metrics(workers)
        
        # Collect VLM metrics from vlm_metrics_logger
        results["vlm_metrics"] = self._collect_vlm_logger_metrics()
        
        # Stop containers
        print("Stopping containers...")
        self.docker_compose_cmd("--profile parallel down")
        
        # Export results
        self._export_results(results, "fixed_workers")
        
        return results
    
    def run_stream_density(
        self,
        target_fps: float,
        init_duration: int,
        density_increment: Optional[int] = None,
        container_name: str = "order-accuracy-vlm"
    ) -> Tuple[int, bool, Dict]:
        """
        Run stream density benchmark to find maximum pipelines.
        
        Args:
            target_fps: Target FPS to maintain per stream
            init_duration: Warmup duration in seconds
            density_increment: Pipeline count increment (auto if None)
            container_name: Container to monitor
            
        Returns:
            Tuple of (max_pipelines, met_target, metrics_dict)
        """
        print(f"\n{'='*60}")
        print(f"Order Accuracy Benchmark - Stream Density Mode")
        print(f"Target FPS: {target_fps}")
        print(f"Init Duration: {init_duration}s")
        print(f"Container: {container_name}")
        print(f"{'='*60}\n")
        
        # Configure stream density
        self.env_vars["TARGET_FPS"] = str(target_fps)
        self.env_vars["INIT_DURATION"] = str(init_duration)
        self.env_vars["CONTAINER_NAME"] = container_name
        
        if density_increment:
            self.env_vars["PIPELINE_INC"] = str(density_increment)
        
        # Use Order Accuracy specific stream density
        oa_density = OrderAccuracyStreamDensity(
            self.env_vars,
            self.compose_files,
            self.results_dir
        )
        
        max_pipelines, met_target = oa_density.run_iterations(
            target_fps=target_fps,
            container_name=container_name
        )
        
        # Collect final metrics
        metrics = oa_density.get_final_metrics()
        
        # Export results
        self._export_results({
            "mode": "stream_density",
            "target_fps": target_fps,
            "max_pipelines": max_pipelines,
            "met_target": met_target,
            "metrics": metrics
        }, "stream_density")
        
        return max_pipelines, met_target, metrics
    
    def _clean_pipeline_logs(self):
        """Remove previous pipeline log files."""
        import glob
        
        patterns = [
            "pipeline*.log",
            "gst*.log",
            "vlm*.log",
            "latency*.json"
        ]
        
        for pattern in patterns:
            for f in glob.glob(os.path.join(self.results_dir, pattern)):
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
                        # Parse: id=station_1_7 event=start timestamp_ms=1234567890
                        id_match = re.search(r'id=([\w_]+)', line)
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
                        
                        # Parse TPS from ovms_metrics event
                        if "ovms_metrics" in line:
                            tps_match = re.search(r'tps=([\d.]+)', line)
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
  # Run with 2 workers for 5 minutes
  python benchmark_order_accuracy.py --compose_file ../../docker-compose.yaml --workers 2 --duration 300
  
  # Run stream density to find max workers at target throughput
  python benchmark_order_accuracy.py --compose_file ../../docker-compose.yaml --target_fps 15.0
  
  # Run stream density with custom increment
  python benchmark_order_accuracy.py --compose_file ../../docker-compose.yaml --target_fps 14.95 --density_increment 2
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
        default=0,
        help='Number of station workers to run (0 = use stream density mode)'
    )
    
    parser.add_argument(
        '--target_fps',
        type=float,
        nargs='*',
        default=None,
        help='Target FPS for stream density mode'
    )
    
    parser.add_argument(
        '--container_names',
        type=str,
        nargs='*',
        default=None,
        help='Container names for stream density (1:1 with target_fps)'
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
        help='Benchmark duration in seconds'
    )
    
    parser.add_argument(
        '--density_increment',
        type=int,
        default=None,
        help='Worker increment for stream density'
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
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Resolve compose files
    compose_files = [os.path.abspath(f) for f in args.compose_file]
    
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
    
    # Determine mode
    target_fps_list = args.target_fps if args.target_fps else []
    container_names = args.container_names if args.container_names else []
    
    if args.workers > 0:
        # Fixed workers mode
        print("Running in fixed workers mode...")
        results = benchmark.run_fixed_workers(
            workers=args.workers,
            init_duration=args.init_duration,
            duration=args.duration
        )
        print(f"\nBenchmark complete. Results: {results}")
        
    elif target_fps_list:
        # Stream density mode
        print("Running in stream density mode...")
        
        if len(target_fps_list) > 1 and len(target_fps_list) != len(container_names):
            print("Error: Number of target_fps values must match container_names")
            sys.exit(1)
        
        for i, target_fps in enumerate(target_fps_list):
            container_name = container_names[i] if container_names else "oa_service"
            
            max_workers, met_target, metrics = benchmark.run_stream_density(
                target_fps=target_fps,
                init_duration=args.init_duration,
                density_increment=args.density_increment,
                container_name=container_name
            )
            
            print(f"\n{'='*60}")
            print(f"Stream Density Results:")
            print(f"  Target FPS: {target_fps}")
            print(f"  Max Workers: {max_workers}")
            print(f"  Met Target: {met_target}")
            print(f"{'='*60}")
    else:
        # Default: single worker
        print("No mode specified. Running with 1 worker...")
        results = benchmark.run_fixed_workers(
            workers=1,
            init_duration=args.init_duration,
            duration=args.duration
        )
        print(f"\nBenchmark complete. Results: {results}")


if __name__ == '__main__':
    main()