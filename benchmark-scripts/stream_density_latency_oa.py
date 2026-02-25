"""
Order Accuracy Stream Density - Latency-Based Scaling

Implements stream density testing for Order Accuracy using LATENCY as the 
scaling metric. Increases workers until target latency is exceeded.

Flow:
  1. Start with 1 worker
  2. Wait for minimum transactions to complete
  3. Collect VLM latency from vlm_metrics_logger
  4. If latency <= target: increase workers, repeat
  5. If latency > target: STOP, report max workers = previous iteration

No fixed duration - runs until enough transactions to measure latency,
then scales and repeats.
"""

import os
import sys
import time
import glob
import re
import json
import psutil
import shlex
import subprocess
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class IterationResult:
    """Results from a single density iteration."""
    workers: int
    avg_latency_ms: float
    p95_latency_ms: float
    avg_tps: float
    total_transactions: int
    passed: bool
    memory_percent: float
    timestamp: str


@dataclass
class DensityResult:
    """Final stream density results."""
    mode: str = "latency_density"
    target_latency_ms: float = 15000.0
    max_workers: int = 0
    met_target: bool = False
    iterations: List[IterationResult] = field(default_factory=list)
    best_iteration: Optional[IterationResult] = None


def get_env_int(name: str, default: int) -> int:
    """Get integer from environment variable with default."""
    value = os.environ.get(name)
    if value is not None:
        try:
            return int(value)
        except ValueError:
            print(f"WARNING: Invalid {name}={value}, using default {default}")
    return default


def get_env_float(name: str, default: float) -> float:
    """Get float from environment variable with default."""
    value = os.environ.get(name)
    if value is not None:
        try:
            return float(value)
        except ValueError:
            print(f"WARNING: Invalid {name}={value}, using default {default}")
    return default


class LatencyBasedStreamDensity:
    """
    Stream density tester using latency-based scaling.
    
    Iteratively increases worker count until VLM latency exceeds target.
    Polls for completed transactions instead of waiting fixed duration.
    
    Environment Variables (CLI args override these):
        TARGET_LATENCY_MS: Target latency threshold in ms (default: 15000)
        LATENCY_METRIC: Which metric to use: 'avg' or 'p95' (default: avg)
        WORKER_INCREMENT: Workers to add per iteration (default: 1)
        INIT_DURATION: Init wait time in seconds (default: 120)
        MIN_TRANSACTIONS: Min transactions before measuring (default: 3)
        MAX_ITERATIONS: Maximum scaling iterations (default: 50)
        MAX_WAIT_SEC: Max wait time per iteration (default: 600)
    """
    
    # Configuration constants (can be overridden by env vars)
    DEFAULT_TARGET_LATENCY_MS = 15000  # 15 seconds
    MAX_ITERATIONS = get_env_int("MAX_ITERATIONS", 50)
    MEMORY_SAFETY_BUFFER_MB = 2048
    MIN_TRANSACTIONS = 3  # Minimum transactions to measure latency
    POLL_INTERVAL_SEC = 10  # How often to check for transactions
    MAX_WAIT_SEC = get_env_int("MAX_WAIT_SEC", 600)  # Maximum wait time per iteration
    
    def __init__(
        self,
        compose_files: List[str],
        results_dir: str,
        target_latency_ms: float = DEFAULT_TARGET_LATENCY_MS,
        latency_metric: str = "avg",  # "avg" or "p95"
        worker_increment: int = 1,
        init_duration: int = 120,
        min_transactions: int = MIN_TRANSACTIONS
    ):
        self.compose_files = compose_files
        self.results_dir = results_dir
        self.target_latency_ms = target_latency_ms
        self.latency_metric = latency_metric
        self.worker_increment = worker_increment
        self.init_duration = init_duration
        self.min_transactions = min_transactions
        
        self.env_vars = os.environ.copy()
        self.iterations: List[IterationResult] = []
        
        # Ensure results directory exists
        os.makedirs(results_dir, exist_ok=True)
        
    def run(self) -> DensityResult:
        """
        Run latency-based stream density test.
        
        Keeps increasing workers until latency exceeds target.
        No fixed duration - polls for transactions.
        
        Returns:
            DensityResult with max workers and iteration history
        """
        print("=" * 70)
        print("Order Accuracy Stream Density - Latency Mode")
        print("=" * 70)
        print(f"Target Latency: {self.target_latency_ms}ms ({self.target_latency_ms/1000:.1f}s)")
        print(f"Latency Metric: {self.latency_metric}")
        print(f"Worker Increment: {self.worker_increment}")
        print(f"Init Duration: {self.init_duration}s")
        print(f"Min Transactions: {self.min_transactions} per worker (scales with worker count)")
        print("=" * 70)
        
        workers = 1
        best_result: Optional[IterationResult] = None
        
        for iteration in range(1, self.MAX_ITERATIONS + 1):
            print(f"\n{'='*60}")
            print(f"Iteration {iteration}: Testing {workers} worker(s)")
            print(f"{'='*60}")
            
            # Check memory before scaling
            if not self._check_memory_available(workers):
                print(f"Memory limit reached. Stopping at {workers - self.worker_increment} workers.")
                break
            
            # Clean previous metrics
            self._clean_metrics_files()
            
            # Run benchmark iteration
            result = self._run_iteration(workers)
            
            # Check if we got enough transactions (scaled by worker count)
            required_transactions = self.min_transactions * workers
            if result.total_transactions < required_transactions:
                print(f"WARNING: Only got {result.total_transactions} transactions (need {required_transactions} = {self.min_transactions} × {workers} workers)")
                print("Latency measurement may be unreliable.")
            
            self.iterations.append(result)
            
            # Print results
            print(f"\nResults for {workers} worker(s):")
            print(f"  Transactions: {result.total_transactions}")
            print(f"  Avg Latency: {result.avg_latency_ms:.0f}ms ({result.avg_latency_ms/1000:.1f}s)")
            print(f"  P95 Latency: {result.p95_latency_ms:.0f}ms ({result.p95_latency_ms/1000:.1f}s)")
            print(f"  Avg TPS: {result.avg_tps:.2f}")
            print(f"  Memory: {result.memory_percent:.1f}%")
            
            # Check against target
            current_latency = (
                result.avg_latency_ms if self.latency_metric == "avg"
                else result.p95_latency_ms
            )
            
            if current_latency <= self.target_latency_ms and current_latency > 0:
                print(f"  ✓ PASSED (latency {current_latency/1000:.1f}s <= {self.target_latency_ms/1000:.1f}s)")
                best_result = result
                workers += self.worker_increment
            elif current_latency == 0:
                print(f"  ⚠ NO DATA - No latency measurements collected")
                print(f"  Trying next iteration anyway...")
                workers += self.worker_increment
            elif current_latency < 0:
                print(f"  ⚠ CORRUPTED METRICS - Negative latency {current_latency:.0f}ms detected")
                print(f"  This usually means a video-loop ID collision in the metrics file.")
                print(f"  Discarding this iteration result; trying next...")
                self.iterations.pop()  # Remove the corrupt entry already appended above
                workers += self.worker_increment
            else:
                print(f"  ✗ FAILED (latency {current_latency/1000:.1f}s > {self.target_latency_ms/1000:.1f}s)")
                break
        
        # Final cleanup
        self._docker_compose("--profile parallel down")
        
        # Build result
        density_result = DensityResult(
            mode="latency_density",
            target_latency_ms=self.target_latency_ms,
            max_workers=best_result.workers if best_result else 0,
            met_target=best_result is not None,
            iterations=self.iterations,
            best_iteration=best_result
        )
        
        # Export results
        self._export_results(density_result)
        
        # Print summary
        self._print_summary(density_result)
        
        return density_result
    
    def _run_iteration(self, workers: int) -> IterationResult:
        """
        Run a single benchmark iteration with specified workers.
        
        Polls for transactions instead of waiting fixed duration.
        """
        # Set environment for workers
        self.env_vars["WORKERS"] = str(workers)
        self.env_vars["VLM_WORKERS"] = str(workers)
        self.env_vars["SERVICE_MODE"] = "parallel"
        
        # Stop any existing containers
        print("Stopping existing containers...")
        self._docker_compose("--profile parallel down")
        time.sleep(5)
        
        # Start containers
        print("Starting containers...")
        self._docker_compose("--profile parallel up -d")
        
        # Wait for initialization
        print(f"Waiting {self.init_duration}s for initialization...")
        time.sleep(self.init_duration)
        
        # Poll for transactions until we have enough (scaled by worker count)
        required_transactions = self.min_transactions * workers
        print(f"Waiting for {required_transactions} transactions ({self.min_transactions} per worker × {workers} worker(s))...")
        start_time = time.time()
        transactions = 0
        
        while transactions < required_transactions:
            elapsed = time.time() - start_time
            
            if elapsed > self.MAX_WAIT_SEC:
                print(f"Timeout after {self.MAX_WAIT_SEC}s. Got {transactions} transactions.")
                break
            
            # Check current transaction count
            metrics = self._collect_vlm_logger_metrics()
            transactions = metrics.get("total_transactions", 0)
            
            if transactions < required_transactions:
                remaining = required_transactions - transactions
                print(f"  Transactions: {transactions}/{required_transactions} "
                      f"(waiting for {remaining} more, {elapsed:.0f}s elapsed)")
                time.sleep(self.POLL_INTERVAL_SEC)
        
        # Collect final metrics
        vlm_metrics = self._collect_vlm_logger_metrics()
        memory_percent = psutil.virtual_memory().percent
        
        # Stop containers
        print("Stopping containers...")
        self._docker_compose("--profile parallel down")
        time.sleep(3)
        
        return IterationResult(
            workers=workers,
            avg_latency_ms=vlm_metrics.get("avg_latency_ms", 0.0),
            p95_latency_ms=vlm_metrics.get("p95_latency_ms", 0.0),
            avg_tps=vlm_metrics.get("avg_tps", 0.0),
            total_transactions=vlm_metrics.get("total_transactions", 0),
            passed=True,  # Will be updated after comparison
            memory_percent=memory_percent,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
    
    def _collect_vlm_logger_metrics(self) -> Dict:
        """
        Collect metrics from vlm_metrics_logger output files.
        
        Returns:
            Dictionary with VLM latency metrics
        """
        metrics = {
            "transactions": [],
            "total_transactions": 0,
            "avg_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "avg_tps": 0.0
        }
        
        # Find vlm_application_metrics files
        pattern = os.path.join(self.results_dir, "vlm_application_metrics_*.txt")
        log_files = glob.glob(pattern)
        
        if not log_files:
            return metrics
        
        start_times = {}
        end_times = {}
        tps_values = []
        
        for log_file in log_files:
            try:
                with open(log_file, 'r') as f:
                    for line in f:
                        # Parse unique_id
                        id_match = re.search(r'id=([\w_]+)', line)
                        event_match = re.search(r'event=(\w+)', line)
                        ts_match = re.search(r'timestamp_ms=(\d+)', line)
                        
                        if id_match and event_match and ts_match:
                            unique_id = id_match.group(1)
                            event = event_match.group(1)
                            timestamp = int(ts_match.group(1))
                            
                            if event == "start":
                                # If this ID already has a completed start+end pair (video loop
                                # reuse), save it under a unique key before overwriting so the
                                # latency isn't corrupted by end(loop N) - start(loop N+1).
                                if unique_id in start_times and unique_id in end_times:
                                    saved_key = f"{unique_id}_run{len([k for k in start_times if k.startswith(unique_id)])}"
                                    start_times[saved_key] = start_times[unique_id]
                                    end_times[saved_key] = end_times[unique_id]
                                    del end_times[unique_id]
                                start_times[unique_id] = timestamp
                            elif event == "end":
                                end_times[unique_id] = timestamp
                        
                        # Parse TPS from custom events
                        if "ovms_metrics" in line:
                            tps_match = re.search(r'tps=([\d.]+)', line)
                            if tps_match:
                                tps_values.append(float(tps_match.group(1)))
            except (IOError, OSError) as e:
                continue
        
        # Calculate latencies from start/end pairs
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
            # P95 latency
            sorted_latencies = sorted(latencies)
            p95_idx = int(len(sorted_latencies) * 0.95)
            metrics["p95_latency_ms"] = sorted_latencies[min(p95_idx, len(sorted_latencies) - 1)]
        
        if tps_values:
            metrics["avg_tps"] = sum(tps_values) / len(tps_values)
        
        return metrics
    
    def _docker_compose(self, action: str):
        """Execute docker compose command."""
        compose_args = " ".join(f"-f {shlex.quote(f)}" for f in self.compose_files)
        cmd = f"docker compose {compose_args} {action}"
        
        print(f"Executing: {cmd}")
        
        result = subprocess.run(
            cmd,
            shell=True,
            env=self.env_vars,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0 and "down" not in action:
            print(f"Warning: docker compose failed: {result.stderr}")
    
    def _clean_metrics_files(self):
        """Clean up previous metrics files."""
        patterns = [
            "vlm_application_metrics_*.txt",
            "vlm_performance_metrics_*.txt"
        ]
        
        for pattern in patterns:
            for f in glob.glob(os.path.join(self.results_dir, pattern)):
                try:
                    os.remove(f)
                except OSError:
                    pass
    
    def _check_memory_available(self, workers: int) -> bool:
        """Check if memory is available for workers."""
        mem = psutil.virtual_memory()
        
        # Check memory pressure
        if mem.percent > 85:
            print(f"High memory pressure: {mem.percent}%")
            return False
        
        available_mb = mem.available / (1024 * 1024)
        if available_mb < self.MEMORY_SAFETY_BUFFER_MB:
            print(f"Insufficient memory: {available_mb:.0f}MB available")
            return False
        
        return True
    
    def _export_results(self, result: DensityResult):
        """Export results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Convert to dict for JSON serialization
        result_dict = {
            "mode": result.mode,
            "target_latency_ms": result.target_latency_ms,
            "target_latency_sec": result.target_latency_ms / 1000,
            "max_workers": result.max_workers,
            "met_target": result.met_target,
            "iterations": [
                {
                    "workers": it.workers,
                    "avg_latency_ms": it.avg_latency_ms,
                    "avg_latency_sec": it.avg_latency_ms / 1000,
                    "p95_latency_ms": it.p95_latency_ms,
                    "p95_latency_sec": it.p95_latency_ms / 1000,
                    "avg_tps": it.avg_tps,
                    "total_transactions": it.total_transactions,
                    "passed": it.passed,
                    "memory_percent": it.memory_percent,
                    "timestamp": it.timestamp
                }
                for it in result.iterations
            ],
            "best_iteration": {
                "workers": result.best_iteration.workers,
                "avg_latency_ms": result.best_iteration.avg_latency_ms,
                "avg_latency_sec": result.best_iteration.avg_latency_ms / 1000,
                "p95_latency_ms": result.best_iteration.p95_latency_ms,
                "avg_tps": result.best_iteration.avg_tps,
                "total_transactions": result.best_iteration.total_transactions
            } if result.best_iteration else None
        }
        
        # Export JSON
        json_path = os.path.join(
            self.results_dir,
            f"latency_density_results_{timestamp}.json"
        )
        with open(json_path, 'w') as f:
            json.dump(result_dict, f, indent=2)
        print(f"\nResults exported to: {json_path}")
        
        # Export CSV summary
        csv_path = os.path.join(
            self.results_dir,
            f"latency_density_summary_{timestamp}.csv"
        )
        with open(csv_path, 'w') as f:
            f.write("workers,avg_latency_ms,avg_latency_sec,p95_latency_ms,avg_tps,transactions,passed,memory_percent\n")
            for it in result.iterations:
                f.write(f"{it.workers},{it.avg_latency_ms:.0f},{it.avg_latency_ms/1000:.1f},"
                       f"{it.p95_latency_ms:.0f},{it.avg_tps:.2f},{it.total_transactions},"
                       f"{it.passed},{it.memory_percent:.1f}\n")
        print(f"Summary exported to: {csv_path}")
    
    def _print_summary(self, result: DensityResult):
        """Print final summary."""
        print("\n" + "=" * 70)
        print("STREAM DENSITY RESULTS - LATENCY MODE")
        print("=" * 70)
        print(f"Target Latency: {result.target_latency_ms}ms ({result.target_latency_ms/1000:.1f}s)")
        print(f"Max Workers: {result.max_workers}")
        print(f"Met Target: {'Yes' if result.met_target else 'No'}")
        
        if result.best_iteration:
            print(f"\nBest Iteration (max workers maintaining target latency):")
            print(f"  Workers: {result.best_iteration.workers}")
            print(f"  Avg Latency: {result.best_iteration.avg_latency_ms:.0f}ms ({result.best_iteration.avg_latency_ms/1000:.1f}s)")
            print(f"  P95 Latency: {result.best_iteration.p95_latency_ms:.0f}ms ({result.best_iteration.p95_latency_ms/1000:.1f}s)")
            print(f"  Avg TPS: {result.best_iteration.avg_tps:.2f}")
            print(f"  Transactions: {result.best_iteration.total_transactions}")
        
        print("\nIteration History:")
        print("-" * 70)
        print(f"{'Workers':<10}{'Avg Latency':<20}{'P95 Latency':<20}{'TPS':<10}{'Status':<10}")
        print("-" * 70)
        for it in result.iterations:
            latency_str = f"{it.avg_latency_ms/1000:.1f}s"
            p95_str = f"{it.p95_latency_ms/1000:.1f}s"
            status = "✓ PASS" if it.avg_latency_ms <= result.target_latency_ms and it.avg_latency_ms > 0 else "✗ FAIL"
            print(f"{it.workers:<10}{latency_str:<20}{p95_str:<20}{it.avg_tps:<10.2f}{status:<10}")
        print("=" * 70)


def main():
    """Main entry point."""
    import argparse
    
    # Get defaults from environment variables
    env_target_latency = get_env_float("TARGET_LATENCY_MS", 15000)
    env_latency_metric = os.environ.get("LATENCY_METRIC", "avg")
    env_worker_increment = get_env_int("WORKER_INCREMENT", 1)
    env_init_duration = get_env_int("INIT_DURATION", 120)
    env_min_transactions = get_env_int("MIN_TRANSACTIONS", 3)
    env_results_dir = os.environ.get("RESULTS_DIR", "./results")
    
    parser = argparse.ArgumentParser(
        description="Order Accuracy Stream Density Test - Latency Based\n\n"
                    "Increases workers until target latency is exceeded.\n"
                    "No fixed duration - polls for transactions.\n\n"
                    "Environment Variables (CLI args override):\n"
                    "  TARGET_LATENCY_MS   - Target latency in ms (default: 15000)\n"
                    "  LATENCY_METRIC      - 'avg' or 'p95' (default: avg)\n"
                    "  WORKER_INCREMENT    - Workers per iteration (default: 1)\n"
                    "  INIT_DURATION       - Init wait seconds (default: 120)\n"
                    "  MIN_TRANSACTIONS    - Min transactions (default: 3)\n"
                    "  MAX_ITERATIONS      - Max iterations (default: 50)\n"
                    "  MAX_WAIT_SEC        - Max wait per iteration (default: 600)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--compose_file",
        type=str,
        nargs="+",
        required=True,
        help="Docker compose file(s)"
    )
    parser.add_argument(
        "--target_latency_ms",
        type=float,
        default=env_target_latency,
        help=f"Target latency threshold in milliseconds (default: {env_target_latency}, env: TARGET_LATENCY_MS)"
    )
    parser.add_argument(
        "--latency_metric",
        type=str,
        choices=["avg", "p95"],
        default=env_latency_metric,
        help=f"Which latency metric to use: avg or p95 (default: {env_latency_metric}, env: LATENCY_METRIC)"
    )
    parser.add_argument(
        "--worker_increment",
        type=int,
        default=env_worker_increment,
        help=f"Number of workers to add each iteration (default: {env_worker_increment}, env: WORKER_INCREMENT)"
    )
    parser.add_argument(
        "--init_duration",
        type=int,
        default=env_init_duration,
        help=f"Initialization duration in seconds (default: {env_init_duration}, env: INIT_DURATION)"
    )
    parser.add_argument(
        "--min_transactions",
        type=int,
        default=env_min_transactions,
        help=f"Minimum transactions per iteration before measuring latency (default: {env_min_transactions}, env: MIN_TRANSACTIONS)"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=env_results_dir,
        help=f"Directory for results output (default: {env_results_dir}, env: RESULTS_DIR)"
    )
    
    args = parser.parse_args()
    
    # Log configuration source
    print("Configuration:")
    print(f"  TARGET_LATENCY_MS: {args.target_latency_ms}")
    print(f"  LATENCY_METRIC: {args.latency_metric}")
    print(f"  WORKER_INCREMENT: {args.worker_increment}")
    print(f"  INIT_DURATION: {args.init_duration}")
    print(f"  MIN_TRANSACTIONS: {args.min_transactions}")
    print(f"  RESULTS_DIR: {args.results_dir}")
    print()
    
    # Resolve compose files
    compose_files = [os.path.abspath(f) for f in args.compose_file]
    
    # Validate compose files exist
    for cf in compose_files:
        if not os.path.exists(cf):
            print(f"Error: Compose file not found: {cf}")
            sys.exit(1)
    
    # Create and run density test
    tester = LatencyBasedStreamDensity(
        compose_files=compose_files,
        results_dir=args.results_dir,
        target_latency_ms=args.target_latency_ms,
        latency_metric=args.latency_metric,
        worker_increment=args.worker_increment,
        init_duration=args.init_duration,
        min_transactions=args.min_transactions
    )
    
    result = tester.run()
    
    # Exit code based on result
    sys.exit(0 if result.met_target else 1)


if __name__ == "__main__":
    main()
