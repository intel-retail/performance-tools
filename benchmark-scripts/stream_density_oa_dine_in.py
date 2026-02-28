#!/usr/bin/env python3
# Copyright © 2025 Intel Corporation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Dine-In Order Accuracy Stream Density - Image-Based Latency Scaling

Implements stream density testing for Dine-In application using LATENCY as the 
scaling metric. Increases concurrent image validations until target latency is exceeded.

Flow:
  1. Start dine-in services
  2. Send N concurrent image validation requests (density = N images)
  3. Collect VLM latency from API responses and vlm_metrics_logger
  4. If latency <= target: increase density by +1 image, repeat
  5. If latency > target: STOP, report max density = previous iteration

Stream Density Definition:
  In dine-in context, "stream density" refers to the number of concurrent
  image validation requests the system can handle while maintaining target latency.
  Each +1 density = +1 concurrent image being processed through VLM.

Usage:
  python stream_density_oa_dine_in.py \\
      --compose_file /path/to/dine-in/docker-compose.yml \\
      --target_latency_ms 15000 \\
      --results_dir ./results
"""

import os
import sys
import time
import glob
import re
import json
import shlex
import subprocess
import asyncio
import psutil
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import aiohttp for async HTTP requests
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    logger.error("aiohttp not installed. Install with: pip install aiohttp")


# =============================================================================
# Environment Variable Helpers
# =============================================================================

def get_env_int(name: str, default: int) -> int:
    """Get integer from environment variable with default."""
    value = os.environ.get(name)
    if value is not None:
        try:
            return int(value)
        except ValueError:
            logger.warning(f"Invalid {name}={value}, using default {default}")
    return default


def get_env_float(name: str, default: float) -> float:
    """Get float from environment variable with default."""
    value = os.environ.get(name)
    if value is not None:
        try:
            return float(value)
        except ValueError:
            logger.warning(f"Invalid {name}={value}, using default {default}")
    return default


def get_env_str(name: str, default: str) -> str:
    """Get string from environment variable with default."""
    return os.environ.get(name, default)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class DineInIterationResult:
    """Results from a single dine-in density iteration."""
    density: int  # Number of concurrent images
    avg_latency_ms: float
    p95_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    avg_tps: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    passed: bool
    memory_percent: float
    cpu_percent: float
    timestamp: str


@dataclass
class DineInDensityResult:
    """Final stream density results for dine-in."""
    mode: str = "dinein_image_density"
    target_latency_ms: float = 15000.0
    max_density: int = 0  # Maximum concurrent images maintaining target latency
    met_target: bool = False
    iterations: List[DineInIterationResult] = field(default_factory=list)
    best_iteration: Optional[DineInIterationResult] = None
    total_images_processed: int = 0


# =============================================================================
# Image Validator - Handles Async HTTP Requests
# =============================================================================

class DineInImageValidator:
    """
    Async image validator for dine-in API.
    
    Handles concurrent image validation requests to measure stream density.
    """
    
    def __init__(
        self,
        api_endpoint: str,
        images_dir: str,
        orders_file: str,
        timeout: int = 300
    ):
        self.api_endpoint = api_endpoint
        self.validate_endpoint = f"{api_endpoint}/api/validate"
        self.health_endpoint = f"{api_endpoint}/health"
        self.images_dir = Path(images_dir)
        self.orders_file = Path(orders_file)
        self.timeout = timeout
        
        # Load test data
        self.test_data: List[Tuple[Path, Dict]] = []
        self._load_test_data()
        
    def _load_test_data(self):
        """Load images and corresponding orders for testing."""
        if not self.orders_file.exists():
            logger.error(f"Orders file not found: {self.orders_file}")
            return
        
        with open(self.orders_file, 'r') as f:
            orders_data = json.load(f)
        
        orders_by_id = {
            order['image_id']: order 
            for order in orders_data.get('orders', [])
        }
        
        # Find all images and their orders
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            for img_path in sorted(self.images_dir.glob(ext)):
                image_id = img_path.stem
                if image_id in orders_by_id:
                    order = orders_by_id[image_id]
                    # Convert to API format
                    order_manifest = {
                        "order_id": order.get('order_id', image_id),
                        "items": [
                            {"name": item.get('item'), "quantity": item.get('quantity', 1)}
                            for item in order.get('items_ordered', [])
                        ]
                    }
                    self.test_data.append((img_path, order_manifest))
        
        logger.info(f"Loaded {len(self.test_data)} test image/order pairs")
    
    async def health_check(self, session: aiohttp.ClientSession) -> bool:
        """Check if dine-in API is healthy."""
        try:
            async with session.get(self.health_endpoint, timeout=10) as resp:
                return resp.status == 200
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    async def validate_image(
        self,
        session: aiohttp.ClientSession,
        image_path: Path,
        order_manifest: Dict,
        request_id: str
    ) -> Dict:
        """
        Send a single image validation request.
        
        Returns:
            Dict with latency_ms, success, error, and response data
        """
        start_time = time.time()
        result = {
            "request_id": request_id,
            "image_id": image_path.stem,
            "latency_ms": 0,
            "success": False,
            "error": None,
            "response": None
        }
        
        try:
            # Prepare multipart form data
            data = aiohttp.FormData()
            data.add_field('order', json.dumps(order_manifest))
            data.add_field(
                'image',
                open(image_path, 'rb'),
                filename=image_path.name,
                content_type='image/jpeg'
            )
            
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            async with session.post(
                self.validate_endpoint,
                data=data,
                timeout=timeout
            ) as resp:
                result["latency_ms"] = (time.time() - start_time) * 1000
                
                if resp.status == 200:
                    result["success"] = True
                    response_data = await resp.json()
                    result["response"] = response_data
                    # Log validation outcome
                    accuracy = response_data.get("accuracy_score", 0)
                    complete = response_data.get("order_complete", False)
                    matched = response_data.get("matched_items", [])
                    missing = response_data.get("missing_items", [])
                    extra = response_data.get("extra_items", [])
                    mismatches = response_data.get("quantity_mismatches", [])
                    status_str = "COMPLETE" if complete else "INCOMPLETE"
                    logger.info(
                        f"[VALIDATE] {request_id} | {status_str} | "
                        f"accuracy={accuracy:.2f} | latency={result['latency_ms']:.0f}ms | "
                        f"matched={len(matched)} missing={len(missing)} "
                        f"extra={len(extra)} mismatches={len(mismatches)}"
                    )
                    if missing:
                        missing_names = [item.get('name', item) for item in missing]
                        logger.info(f"[VALIDATE] {request_id} | MISSING: {missing_names}")
                    if extra:
                        extra_names = [item.get('name', item) for item in extra]
                        logger.info(f"[VALIDATE] {request_id} | EXTRA: {extra_names}")
                    if mismatches:
                        logger.info(f"[VALIDATE] {request_id} | QUANTITY MISMATCHES: {mismatches}")
                else:
                    result["error"] = f"HTTP {resp.status}: {await resp.text()}"
                    
        except asyncio.TimeoutError:
            result["latency_ms"] = self.timeout * 1000
            result["error"] = f"Timeout after {self.timeout}s"
        except Exception as e:
            result["latency_ms"] = (time.time() - start_time) * 1000
            result["error"] = str(e)
        
        return result
    
    async def run_concurrent_validations(
        self,
        density: int,
        iteration: int
    ) -> List[Dict]:
        """
        Run N concurrent image validations.
        
        Args:
            density: Number of concurrent requests
            iteration: Iteration number for request ID generation
            
        Returns:
            List of validation results
        """
        if not self.test_data:
            logger.error("No test data loaded")
            return []
        
        # Select images for this density level (cycle through available images)
        requests = []
        for i in range(density):
            img_path, order = self.test_data[i % len(self.test_data)]
            request_id = f"iter{iteration}_img{i}_{img_path.stem}"
            requests.append((img_path, order, request_id))
        
        logger.info(f"Starting {density} concurrent validations")
        
        # Create connector with proper limits
        connector = aiohttp.TCPConnector(
            limit=density + 10,
            limit_per_host=density + 10,
            force_close=True
        )
        
        results = []
        async with aiohttp.ClientSession(connector=connector) as session:
            # Check health first
            if not await self.health_check(session):
                logger.error("API health check failed")
                return []
            
            # Launch all requests concurrently
            tasks = [
                self.validate_image(session, img_path, order, req_id)
                for img_path, order, req_id in requests
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions
            valid_results = []
            for r in results:
                if isinstance(r, Exception):
                    valid_results.append({
                        "success": False,
                        "error": str(r),
                        "latency_ms": 0
                    })
                else:
                    valid_results.append(r)
            
            return valid_results


# =============================================================================
# Main Stream Density Tester
# =============================================================================

class DineInStreamDensity:
    """
    Dine-In Stream Density Tester using latency-based scaling.
    
    Iteratively increases concurrent image validations until VLM latency 
    exceeds the target threshold.
    
    Key Concepts:
    - Density = Number of concurrent image validation requests
    - Each image goes through VLM for item detection
    - Target is to find max density while maintaining target latency
    """
    
    # Configuration constants
    DEFAULT_TARGET_LATENCY_MS = 15000  # 15 seconds
    DEFAULT_MAX_ITERATIONS = 50
    MEMORY_SAFETY_THRESHOLD_PERCENT = 90
    MIN_REQUESTS_PER_ITERATION = 3
    
    def __init__(
        self,
        compose_file: str,
        results_dir: str,
        api_endpoint: str = "http://localhost:8083",
        images_dir: Optional[str] = None,
        orders_file: Optional[str] = None,
        target_latency_ms: float = DEFAULT_TARGET_LATENCY_MS,
        latency_metric: str = "avg",  # "avg", "p95", "max"
        density_increment: int = 1,
        init_duration: int = 60,
        min_requests: int = MIN_REQUESTS_PER_ITERATION,
        request_timeout: int = 300,
        single_run: bool = False,
        concurrent_images: int = 1,
        max_iterations: int = DEFAULT_MAX_ITERATIONS
    ):
        self.compose_file = compose_file
        self.results_dir = Path(results_dir)
        self.api_endpoint = api_endpoint
        self.target_latency_ms = target_latency_ms
        self.latency_metric = latency_metric
        self.density_increment = density_increment
        self.init_duration = init_duration
        self.min_requests = min_requests
        self.request_timeout = request_timeout
        self.single_run = single_run
        self.concurrent_images = concurrent_images
        self.max_iterations = max_iterations
        
        # Resolve paths relative to compose file
        compose_dir = Path(compose_file).parent
        self.images_dir = Path(images_dir) if images_dir else compose_dir / "images"
        self.orders_file = Path(orders_file) if orders_file else compose_dir / "configs" / "orders.json"
        
        self.env_vars = os.environ.copy()
        self.iterations: List[DineInIterationResult] = []
        
        # Ensure results directory exists
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"DineInStreamDensity initialized:")
        logger.info(f"  Compose: {self.compose_file}")
        logger.info(f"  API: {self.api_endpoint}")
        logger.info(f"  Images: {self.images_dir}")
        logger.info(f"  Orders: {self.orders_file}")
        logger.info(f"  Target Latency: {self.target_latency_ms}ms")
        logger.info(f"  Single Run Mode: {self.single_run}")
        if self.single_run:
            logger.info(f"  Concurrent Images: {self.concurrent_images}")
    
    def run(self) -> DineInDensityResult:
        """
        Run latency-based stream density test for dine-in.
        
        Keeps increasing concurrent image validations until latency exceeds target.
        
        Returns:
            DineInDensityResult with max density and iteration history
        """
        self._print_header()
        
        # In single_run mode, just run once with specified concurrent_images
        density = self.concurrent_images if self.single_run else 1
        max_iter = 1 if self.single_run else self.max_iterations
        best_result: Optional[DineInIterationResult] = None
        total_images = 0
        
        for iteration in range(1, max_iter + 1):
            print(f"\n{'='*70}")
            print(f"Iteration {iteration}: Testing density={density} concurrent images")
            print(f"{'='*70}")
            
            # Memory safety check
            if not self._check_memory_safe():
                print(f"Memory threshold exceeded. Stopping at density={density - self.density_increment}")
                break
            
            # Clean previous metrics only on first iteration
            # (subsequent iterations append to metrics files)
            if iteration == 1:
                self._clean_metrics_files()
            
            # Start services if not running (first iteration)
            if iteration == 1:
                self._start_services()
                self._wait_for_ready()
            
            # Run benchmark iteration
            result = self._run_iteration(density, iteration)
            
            if result.total_requests < self.min_requests:
                print(f"WARNING: Only completed {result.total_requests} requests (need {self.min_requests})")
            
            self.iterations.append(result)
            total_images += result.successful_requests
            
            # Print results
            self._print_iteration_results(result)
            
            # Check against target
            current_latency = self._get_latency_metric(result)
            
            if current_latency <= self.target_latency_ms and current_latency > 0:
                print(f"  ✓ PASSED (latency {current_latency/1000:.2f}s <= {self.target_latency_ms/1000:.2f}s)")
                result.passed = True
                best_result = result
                density += self.density_increment
            elif current_latency == 0:
                print(f"  ⚠ NO DATA - No successful validations")
                print(f"  Stopping due to validation failures")
                break
            else:
                print(f"  ✗ FAILED (latency {current_latency/1000:.2f}s > {self.target_latency_ms/1000:.2f}s)")
                result.passed = False
                break
        
        # Stop services
        self._stop_services()
        
        # Build final result
        density_result = DineInDensityResult(
            mode="dinein_image_density",
            target_latency_ms=self.target_latency_ms,
            max_density=best_result.density if best_result else 0,
            met_target=best_result is not None,
            iterations=self.iterations,
            best_iteration=best_result,
            total_images_processed=total_images
        )
        
        # Export results
        self._export_results(density_result)
        
        # Print summary
        self._print_summary(density_result)
        
        return density_result
    
    def _run_iteration(self, density: int, iteration: int) -> DineInIterationResult:
        """
        Run a single density iteration with N concurrent images.
        
        Args:
            density: Number of concurrent image validations
            iteration: Iteration number
            
        Returns:
            DineInIterationResult with metrics
        """
        # Create validator
        validator = DineInImageValidator(
            api_endpoint=self.api_endpoint,
            images_dir=str(self.images_dir),
            orders_file=str(self.orders_file),
            timeout=self.request_timeout
        )
        
        # Run concurrent validations
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            results = loop.run_until_complete(
                validator.run_concurrent_validations(density, iteration)
            )
        finally:
            loop.close()
        
        # Collect latencies from successful requests
        latencies = [r["latency_ms"] for r in results if r.get("success")]
        successful = len(latencies)
        failed = len(results) - successful

        # Log per-request validation summary
        for r in results:
            if r.get("success") and r.get("response"):
                resp = r["response"]
                complete = resp.get("order_complete", False)
                accuracy = resp.get("accuracy_score", 0)
                matched = len(resp.get("matched_items", []))
                missing = len(resp.get("missing_items", []))
                extra = len(resp.get("extra_items", []))
                print(
                    f"  [{r.get('image_id', '?')}] "
                    f"{'✓ COMPLETE' if complete else '✗ INCOMPLETE'} | "
                    f"accuracy={accuracy:.2f} | matched={matched} missing={missing} extra={extra} | "
                    f"latency={r['latency_ms']:.0f}ms"
                )
            elif not r.get("success"):
                print(f"  [{r.get('image_id', '?')}] ✗ FAILED - {r.get('error', 'unknown error')}")
        
        # Print detailed validation results
        self._print_detailed_validation_results(results)
        
        # Calculate metrics from HTTP response latencies
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        sorted_latencies = sorted(latencies) if latencies else [0]
        p95_idx = int(len(sorted_latencies) * 0.95) if sorted_latencies else 0
        p95_latency = sorted_latencies[min(p95_idx, len(sorted_latencies) - 1)]
        min_latency = min(sorted_latencies) if sorted_latencies else 0
        max_latency = max(sorted_latencies) if sorted_latencies else 0
        
        # TPS calculated from successful requests / total time
        total_time_sec = max_latency / 1000 if max_latency > 0 else 1
        avg_tps = successful / total_time_sec if successful > 0 else 0.0
        
        # System metrics
        memory_percent = psutil.virtual_memory().percent
        cpu_percent = psutil.cpu_percent(interval=0.5)
        
        iteration_result = DineInIterationResult(
            density=density,
            avg_latency_ms=avg_latency,
            p95_latency_ms=p95_latency,
            min_latency_ms=min_latency,
            max_latency_ms=max_latency,
            avg_tps=avg_tps,
            total_requests=len(results),
            successful_requests=successful,
            failed_requests=failed,
            passed=False,  # Will be set after comparison
            memory_percent=memory_percent,
            cpu_percent=cpu_percent,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        # Attach raw results for detailed validation export
        iteration_result._raw_results = results
        return iteration_result
    
    def _get_latency_metric(self, result: DineInIterationResult) -> float:
        """Get the configured latency metric value."""
        if self.latency_metric == "avg":
            return result.avg_latency_ms
        elif self.latency_metric == "p95":
            return result.p95_latency_ms
        elif self.latency_metric == "max":
            return result.max_latency_ms
        return result.avg_latency_ms
    
    def _print_detailed_validation_results(self, results: List[Dict]):
        """Print detailed validation results with item breakdowns."""
        print("\n" + "=" * 70)
        print("VALIDATION DETAILS")
        print("=" * 70)
        
        for r in results:
            image_id = r.get("image_id", "unknown")
            latency_ms = r.get("latency_ms", 0)
            
            if not r.get("success"):
                print(f"\n🖼️  {image_id} - ❌ Request Failed")
                print(f"   Error: {r.get('error', 'unknown error')}")
                print(f"   Latency: {latency_ms:.0f}ms")
                continue
            
            resp = r.get("response", {})
            order_complete = resp.get("order_complete", False)
            accuracy = resp.get("accuracy_score", 0.0)
            matched_items = resp.get("matched_items", [])
            missing_items = resp.get("missing_items", [])
            extra_items = resp.get("extra_items", [])
            qty_mismatches = resp.get("quantity_mismatches", [])
            
            complete_icon = "✅" if order_complete else "❌"
            complete_label = "Order Complete" if order_complete else "Order Incomplete"
            
            print(f"\n🖼️  {image_id}")
            print(f"   {complete_icon} {complete_label}")
            print(f"   Accuracy: {accuracy * 100:.0f}%")
            print(f"   Latency: {latency_ms:.0f}ms")
            
            if matched_items:
                print(f"   ✔️  Matched Items:")
                for item in matched_items:
                    name = item.get("detected_name") or item.get("expected_name", "?")
                    qty = item.get("quantity", 1)
                    sim = item.get("similarity", 0)
                    print(f"      • {name} (×{qty}) - {sim * 100:.0f}% match")
            
            if missing_items:
                print(f"   ⚠️  Missing Items:")
                for item in missing_items:
                    name = item.get("name", "?")
                    qty = item.get("quantity", 1)
                    print(f"      • {name} (×{qty})")
            
            if extra_items:
                print(f"   ➕ Extra Items Detected:")
                for item in extra_items:
                    name = item.get("name", "?")
                    qty = item.get("quantity", 1)
                    print(f"      • {name} (×{qty})")
            
            if qty_mismatches:
                print(f"   🔢 Quantity Mismatches:")
                for item in qty_mismatches:
                    name = item.get("item", "?")
                    exp = item.get("expected_quantity", "?")
                    got = item.get("detected_quantity", "?")
                    print(f"      • {name}: expected ×{exp}, detected ×{got}")
        
        print("\n" + "=" * 70)
    
    def _start_services(self):
        """Start dine-in services via docker compose."""
        print("Starting dine-in services...")
        self._docker_compose("up -d")
        time.sleep(5)
    
    def _stop_services(self):
        """Stop dine-in services via docker compose."""
        print("Stopping dine-in services...")
        self._docker_compose("down")
        time.sleep(3)
    
    def _wait_for_ready(self):
        """Wait for services to be ready."""
        print(f"Waiting {self.init_duration}s for services to initialize...")
        
        # Poll for health
        import urllib.request
        import urllib.error
        
        start = time.time()
        ready = False
        
        while time.time() - start < self.init_duration:
            try:
                req = urllib.request.Request(f"{self.api_endpoint}/health")
                with urllib.request.urlopen(req, timeout=5) as resp:
                    if resp.status == 200:
                        ready = True
                        break
            except Exception:
                pass
            time.sleep(5)
        
        if ready:
            print("Services ready!")
            # Additional warmup time
            remaining = self.init_duration - (time.time() - start)
            if remaining > 0:
                print(f"Additional warmup: {remaining:.0f}s")
                time.sleep(remaining)
        else:
            print("WARNING: Services may not be fully ready")
    
    def _docker_compose(self, action: str):
        """Execute docker compose command."""
        cmd = f"docker compose -f {shlex.quote(self.compose_file)} {action}"
        logger.debug(f"Executing: {cmd}")
        
        result = subprocess.run(
            cmd,
            shell=True,
            env=self.env_vars,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0 and "down" not in action:
            logger.warning(f"docker compose warning: {result.stderr}")
    
    def _clean_metrics_files(self):
        """Clean up previous metrics files."""
        patterns = [
            "vlm_application_metrics_*.txt",
            "vlm_performance_metrics_*.txt"
        ]
        
        search_dirs = [
            self.results_dir,
            Path("/tmp"),
            Path(self.compose_file).parent / "results"
        ]
        
        for dir_path in search_dirs:
            for pattern in patterns:
                for f in glob.glob(str(dir_path / pattern)):
                    try:
                        os.remove(f)
                        logger.debug(f"Removed: {f}")
                    except OSError:
                        pass
    
    def _check_memory_safe(self) -> bool:
        """Check if memory usage is within safe threshold."""
        mem = psutil.virtual_memory()
        if mem.percent > self.MEMORY_SAFETY_THRESHOLD_PERCENT:
            logger.warning(f"Memory usage {mem.percent}% exceeds threshold {self.MEMORY_SAFETY_THRESHOLD_PERCENT}%")
            return False
        return True
    
    def _print_header(self):
        """Print test header."""
        print("=" * 70)
        print("Dine-In Order Accuracy Stream Density - Image-Based Latency Mode")
        print("=" * 70)
        print(f"Target Latency: {self.target_latency_ms}ms ({self.target_latency_ms/1000:.1f}s)")
        print(f"Latency Metric: {self.latency_metric}")
        print(f"Density Increment: +{self.density_increment} image(s) per iteration")
        print(f"Init Duration: {self.init_duration}s")
        print(f"Min Requests: {self.min_requests}")
        print(f"Request Timeout: {self.request_timeout}s")
        print(f"Images Directory: {self.images_dir}")
        print(f"Results Directory: {self.results_dir}")
        print("=" * 70)
    
    def _print_iteration_results(self, result: DineInIterationResult):
        """Print results for a single iteration."""
        print(f"\nResults for density={result.density} concurrent images:")
        print(f"  Requests: {result.successful_requests}/{result.total_requests} successful")
        print(f"  Avg Latency: {result.avg_latency_ms:.0f}ms ({result.avg_latency_ms/1000:.2f}s)")
        print(f"  P95 Latency: {result.p95_latency_ms:.0f}ms ({result.p95_latency_ms/1000:.2f}s)")
        print(f"  Min/Max: {result.min_latency_ms:.0f}ms / {result.max_latency_ms:.0f}ms")
        print(f"  Avg TPS: {result.avg_tps:.2f}")
        print(f"  Memory: {result.memory_percent:.1f}%  CPU: {result.cpu_percent:.1f}%")
    
    def _export_results(self, result: DineInDensityResult):
        """Export results to JSON and CSV files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Build JSON result
        result_dict = {
            "mode": result.mode,
            "target_latency_ms": result.target_latency_ms,
            "target_latency_sec": result.target_latency_ms / 1000,
            "max_density": result.max_density,
            "met_target": result.met_target,
            "total_images_processed": result.total_images_processed,
            "iterations": [
                {
                    "density": it.density,
                    "avg_latency_ms": round(it.avg_latency_ms, 2),
                    "avg_latency_sec": round(it.avg_latency_ms / 1000, 2),
                    "p95_latency_ms": round(it.p95_latency_ms, 2),
                    "p95_latency_sec": round(it.p95_latency_ms / 1000, 2),
                    "min_latency_ms": round(it.min_latency_ms, 2),
                    "max_latency_ms": round(it.max_latency_ms, 2),
                    "avg_tps": round(it.avg_tps, 2),
                    "total_requests": it.total_requests,
                    "successful_requests": it.successful_requests,
                    "failed_requests": it.failed_requests,
                    "passed": it.passed,
                    "memory_percent": round(it.memory_percent, 1),
                    "cpu_percent": round(it.cpu_percent, 1),
                    "timestamp": it.timestamp
                }
                for it in result.iterations
            ],
            "best_iteration": {
                "density": result.best_iteration.density,
                "avg_latency_ms": round(result.best_iteration.avg_latency_ms, 2),
                "avg_latency_sec": round(result.best_iteration.avg_latency_ms / 1000, 2),
                "p95_latency_ms": round(result.best_iteration.p95_latency_ms, 2),
                "avg_tps": round(result.best_iteration.avg_tps, 2),
                "successful_requests": result.best_iteration.successful_requests
            } if result.best_iteration else None
        }
        
        # Collect all per-request validation details across all iterations
        all_validations = []
        for it in result.iterations:
            if hasattr(it, '_raw_results'):
                for r in it._raw_results:
                    if r.get("success") and r.get("response"):
                        resp = r["response"]
                        all_validations.append({
                            "density": it.density,
                            "request_id": r.get("request_id"),
                            "image_id": r.get("image_id"),
                            "latency_ms": round(r["latency_ms"], 2),
                            "order_complete": resp.get("order_complete"),
                            "accuracy_score": resp.get("accuracy_score"),
                            "matched_items": resp.get("matched_items", []),
                            "missing_items": resp.get("missing_items", []),
                            "extra_items": resp.get("extra_items", []),
                            "quantity_mismatches": resp.get("quantity_mismatches", []),
                            "metrics": resp.get("metrics")
                        })
        if all_validations:
            result_dict["validation_details"] = all_validations

        # Export JSON
        json_path = self.results_dir / f"dinein_density_results_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(result_dict, f, indent=2)
        print(f"\nResults exported to: {json_path}")
        
        # Export CSV summary
        csv_path = self.results_dir / f"dinein_density_summary_{timestamp}.csv"
        with open(csv_path, 'w') as f:
            f.write("density,avg_latency_ms,avg_latency_sec,p95_latency_ms,avg_tps,")
            f.write("successful,failed,passed,memory_pct,cpu_pct\n")
            for it in result.iterations:
                f.write(f"{it.density},{it.avg_latency_ms:.0f},{it.avg_latency_ms/1000:.2f},")
                f.write(f"{it.p95_latency_ms:.0f},{it.avg_tps:.2f},")
                f.write(f"{it.successful_requests},{it.failed_requests},{it.passed},")
                f.write(f"{it.memory_percent:.1f},{it.cpu_percent:.1f}\n")
        print(f"Summary exported to: {csv_path}")

        # Export per-request validation details CSV
        val_csv_path = self.results_dir / f"dinein_validation_details_{timestamp}.csv"
        with open(val_csv_path, 'w') as f:
            f.write("density,request_id,image_id,latency_ms,order_complete,accuracy_score,"
                    "matched,missing,extra,mismatches\n")
            for it in result.iterations:
                raw = getattr(it, '_raw_results', [])
                for r in raw:
                    if r.get("success") and r.get("response"):
                        resp = r["response"]
                        f.write(
                            f"{it.density},{r.get('request_id','')},{r.get('image_id','')},"
                            f"{r['latency_ms']:.0f},{resp.get('order_complete','')},"
                            f"{resp.get('accuracy_score', 0):.2f},"
                            f"{len(resp.get('matched_items',[]))},"
                            f"{len(resp.get('missing_items',[]))},"
                            f"{len(resp.get('extra_items',[]))},"
                            f"{len(resp.get('quantity_mismatches',[]))}\n"
                        )
        print(f"Validation details exported to: {val_csv_path}")

        # Export Markdown report
        self._export_markdown_report(result, timestamp)

    def _export_markdown_report(self, result: DineInDensityResult, timestamp: str):
        """Export a human-readable Markdown validation report per iteration and image."""
        md_path = self.results_dir / f"dinein_validation_report_{timestamp}.md"

        with open(md_path, 'w') as f:
            f.write(f"# Dine-In Stream Density — Validation Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
            f.write(f"**Target Latency:** {result.target_latency_ms:.0f} ms "
                    f"({result.target_latency_ms / 1000:.1f} s)  \n")
            f.write(f"**Max Density Achieved:** {result.max_density} concurrent image(s)  \n")
            f.write(f"**Met Target:** {'✅ Yes' if result.met_target else '❌ No'}  \n")
            f.write(f"**Total Images Processed:** {result.total_images_processed}  \n\n")
            f.write("---\n\n")

            for it in result.iterations:
                status_icon = "✅" if it.passed else "❌"
                f.write(f"## {status_icon} Iteration — Density {it.density} "
                        f"({'PASS' if it.passed else 'FAIL'})\n\n")

                f.write(f"| Metric | Value |\n")
                f.write(f"|--------|-------|\n")
                f.write(f"| Concurrent Images | {it.density} |\n")
                f.write(f"| Avg Latency | {it.avg_latency_ms:.0f} ms "
                        f"({it.avg_latency_ms / 1000:.2f} s) |\n")
                f.write(f"| P95 Latency | {it.p95_latency_ms:.0f} ms "
                        f"({it.p95_latency_ms / 1000:.2f} s) |\n")
                f.write(f"| Min / Max Latency | {it.min_latency_ms:.0f} ms / "
                        f"{it.max_latency_ms:.0f} ms |\n")
                f.write(f"| Avg TPS | {it.avg_tps:.2f} |\n")
                f.write(f"| Requests | {it.successful_requests} succeeded / "
                        f"{it.failed_requests} failed |\n")
                f.write(f"| Memory | {it.memory_percent:.1f}% |\n")
                f.write(f"| CPU | {it.cpu_percent:.1f}% |\n")
                f.write(f"| Timestamp | {it.timestamp} |\n\n")

                # Per-image validation results
                raw = getattr(it, '_raw_results', [])
                if not raw:
                    f.write("_No per-image results available for this iteration._\n\n")
                    f.write("---\n\n")
                    continue

                for r in raw:
                    image_id = r.get("image_id", "unknown")
                    latency_ms = r.get("latency_ms", 0)

                    if not r.get("success"):
                        f.write(f"### 🖼️ `{image_id}` — ❌ Request Failed\n\n")
                        f.write(f"> **Error:** {r.get('error', 'unknown error')}  \n")
                        f.write(f"> **Latency:** {latency_ms:.0f} ms\n\n")
                        continue

                    resp = r.get("response", {})
                    order_complete = resp.get("order_complete", False)
                    accuracy = resp.get("accuracy_score", 0.0)
                    matched_items = resp.get("matched_items", [])
                    missing_items = resp.get("missing_items", [])
                    extra_items = resp.get("extra_items", [])
                    qty_mismatches = resp.get("quantity_mismatches", [])
                    metrics = resp.get("metrics") or {}

                    complete_icon = "✅" if order_complete else "❌"
                    complete_label = "Order Complete" if order_complete else "Order Incomplete"

                    f.write(f"### 🖼️ `{image_id}`\n\n")
                    f.write(f"**✅ Validation Result**\n\n")
                    f.write(f"{complete_icon} **{complete_label}**  \n")
                    f.write(f"**Accuracy:** {accuracy * 100:.0f}%  \n")
                    f.write(f"**Latency:** {latency_ms:.0f} ms  \n\n")

                    if matched_items:
                        f.write(f"**✔️ Matched Items**\n\n")
                        for item in matched_items:
                            name = item.get("detected_name") or item.get("expected_name", "?")
                            qty = item.get("quantity", 1)
                            sim = item.get("similarity", 0)
                            f.write(f"- {name} (×{qty}) _{sim * 100:.0f}% match_\n")
                        f.write("\n")

                    if missing_items:
                        f.write(f"**⚠️ Missing Items**\n\n")
                        for item in missing_items:
                            name = item.get("name", "?")
                            qty = item.get("quantity", 1)
                            f.write(f"- {name} (×{qty})\n")
                        f.write("\n")

                    if extra_items:
                        f.write(f"**➕ Extra Items Detected**\n\n")
                        for item in extra_items:
                            name = item.get("name", "?")
                            qty = item.get("quantity", 1)
                            f.write(f"- {name} (×{qty})\n")
                        f.write("\n")

                    if qty_mismatches:
                        f.write(f"**🔢 Quantity Mismatches**\n\n")
                        for item in qty_mismatches:
                            name = item.get("item", "?")
                            exp = item.get("expected_quantity", "?")
                            got = item.get("detected_quantity", "?")
                            f.write(f"- {name}: expected ×{exp}, detected ×{got}\n")
                        f.write("\n")

                    if metrics:
                        f.write(f"**📊 Performance Metrics**\n\n")
                        f.write(f"| Metric | Value |\n")
                        f.write(f"|--------|-------|\n")
                        for k, v in metrics.items():
                            f.write(f"| {k} | {v} |\n")
                        f.write("\n")

                f.write("---\n\n")

            # Final summary section
            f.write("## 📋 Summary Table\n\n")
            f.write("| Density | Avg Latency | P95 Latency | TPS | Succeeded | Failed | Status |\n")
            f.write("|---------|-------------|-------------|-----|-----------|--------|--------|\n")
            for it in result.iterations:
                status = "✅ PASS" if it.passed else "❌ FAIL"
                f.write(
                    f"| {it.density} "
                    f"| {it.avg_latency_ms:.0f} ms ({it.avg_latency_ms / 1000:.2f}s) "
                    f"| {it.p95_latency_ms:.0f} ms ({it.p95_latency_ms / 1000:.2f}s) "
                    f"| {it.avg_tps:.2f} "
                    f"| {it.successful_requests} "
                    f"| {it.failed_requests} "
                    f"| {status} |\n"
                )

        print(f"Markdown report exported to: {md_path}")

    def _print_summary(self, result: DineInDensityResult):
        """Print final summary."""
        print("\n" + "=" * 70)
        print("DINE-IN STREAM DENSITY RESULTS - IMAGE-BASED LATENCY MODE")
        print("=" * 70)
        print(f"Target Latency: {result.target_latency_ms}ms ({result.target_latency_ms/1000:.1f}s)")
        print(f"Max Density: {result.max_density} concurrent images")
        print(f"Met Target: {'Yes' if result.met_target else 'No'}")
        print(f"Total Images Processed: {result.total_images_processed}")
        
        if result.best_iteration:
            print(f"\nBest Iteration (max density maintaining target latency):")
            print(f"  Density: {result.best_iteration.density} concurrent images")
            print(f"  Avg Latency: {result.best_iteration.avg_latency_ms:.0f}ms ({result.best_iteration.avg_latency_ms/1000:.2f}s)")
            print(f"  P95 Latency: {result.best_iteration.p95_latency_ms:.0f}ms ({result.best_iteration.p95_latency_ms/1000:.2f}s)")
            print(f"  Avg TPS: {result.best_iteration.avg_tps:.2f}")
            print(f"  Successful Requests: {result.best_iteration.successful_requests}")
        
        print("\nIteration History:")
        print("-" * 70)
        print(f"{'Density':<10}{'Avg Latency':<15}{'P95 Latency':<15}{'TPS':<10}{'Success':<10}{'Status':<10}")
        print("-" * 70)
        for it in result.iterations:
            latency_str = f"{it.avg_latency_ms/1000:.2f}s"
            p95_str = f"{it.p95_latency_ms/1000:.2f}s"
            success_str = f"{it.successful_requests}/{it.total_requests}"
            status = "✓ PASS" if it.passed else "✗ FAIL"
            print(f"{it.density:<10}{latency_str:<15}{p95_str:<15}{it.avg_tps:<10.2f}{success_str:<10}{status:<10}")
        print("=" * 70)


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point."""
    import argparse
    
    if not AIOHTTP_AVAILABLE:
        print("Error: aiohttp is required. Install with: pip install aiohttp")
        sys.exit(1)
    
    # Get defaults from environment variables
    env_target_latency = get_env_float("TARGET_LATENCY_MS", 15000)
    env_latency_metric = get_env_str("LATENCY_METRIC", "avg")
    env_density_increment = get_env_int("DENSITY_INCREMENT", 1)
    env_init_duration = get_env_int("INIT_DURATION", 60)
    env_min_requests = get_env_int("MIN_REQUESTS", 3)
    env_request_timeout = get_env_int("REQUEST_TIMEOUT", 300)
    env_api_endpoint = get_env_str("API_ENDPOINT", "http://localhost:8083")
    env_results_dir = get_env_str("RESULTS_DIR", "./results")
    
    parser = argparse.ArgumentParser(
        description="Dine-In Order Accuracy Stream Density Test - Image-Based Latency Scaling\n\n"
                    "Increases concurrent image validations until target latency is exceeded.\n"
                    "Density = Number of concurrent images being processed through VLM.\n\n"
                    "Environment Variables (CLI args override):\n"
                    "  TARGET_LATENCY_MS   - Target latency in ms (default: 15000)\n"
                    "  LATENCY_METRIC      - 'avg', 'p95', or 'max' (default: avg)\n"
                    "  DENSITY_INCREMENT   - Images per iteration (default: 1)\n"
                    "  INIT_DURATION       - Init wait seconds (default: 60)\n"
                    "  MIN_REQUESTS        - Min requests per iteration (default: 3)\n"
                    "  REQUEST_TIMEOUT     - Request timeout seconds (default: 300)\n"
                    "  API_ENDPOINT        - Dine-in API URL (default: http://localhost:8083)\n"
                    "  RESULTS_DIR         - Results output directory (default: ./results)\n\n"
                    "Example:\n"
                    "  python stream_density_oa_dine_in.py \\\n"
                    "      --compose_file /path/to/dine-in/docker-compose.yml \\\n"
                    "      --target_latency_ms 15000 \\\n"
                    "      --results_dir ./results",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--compose_file",
        type=str,
        required=True,
        help="Path to dine-in docker-compose.yml file"
    )
    parser.add_argument(
        "--target_latency_ms",
        type=float,
        default=env_target_latency,
        help=f"Target latency threshold in ms (default: {env_target_latency}, env: TARGET_LATENCY_MS)"
    )
    parser.add_argument(
        "--latency_metric",
        type=str,
        choices=["avg", "p95", "max"],
        default=env_latency_metric,
        help=f"Which latency metric to use (default: {env_latency_metric}, env: LATENCY_METRIC)"
    )
    parser.add_argument(
        "--density_increment",
        type=int,
        default=env_density_increment,
        help=f"Concurrent images to add per iteration (default: {env_density_increment}, env: DENSITY_INCREMENT)"
    )
    parser.add_argument(
        "--init_duration",
        type=int,
        default=env_init_duration,
        help=f"Service init duration in seconds (default: {env_init_duration}, env: INIT_DURATION)"
    )
    parser.add_argument(
        "--min_requests",
        type=int,
        default=env_min_requests,
        help=f"Min successful requests per iteration (default: {env_min_requests}, env: MIN_REQUESTS)"
    )
    parser.add_argument(
        "--request_timeout",
        type=int,
        default=env_request_timeout,
        help=f"Request timeout in seconds (default: {env_request_timeout}, env: REQUEST_TIMEOUT)"
    )
    parser.add_argument(
        "--api_endpoint",
        type=str,
        default=env_api_endpoint,
        help=f"Dine-in API endpoint (default: {env_api_endpoint}, env: API_ENDPOINT)"
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default=None,
        help="Path to images directory (default: compose_file_dir/images)"
    )
    parser.add_argument(
        "--orders_file",
        type=str,
        default=None,
        help="Path to orders.json file (default: compose_file_dir/configs/orders.json)"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=env_results_dir,
        help=f"Directory for results output (default: {env_results_dir}, env: RESULTS_DIR)"
    )
    parser.add_argument(
        "--single_run",
        action="store_true",
        help="Run a single benchmark iteration without density scaling (simple benchmark mode)"
    )
    parser.add_argument(
        "--concurrent_images",
        type=int,
        default=1,
        help="Number of concurrent images for single_run mode (default: 1)"
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=50,
        help="Maximum iterations for density scaling (default: 50)"
    )
    
    args = parser.parse_args()
    
    # Log configuration source
    print("Configuration:")
    print(f"  TARGET_LATENCY_MS: {args.target_latency_ms}")
    print(f"  LATENCY_METRIC: {args.latency_metric}")
    print(f"  DENSITY_INCREMENT: {args.density_increment}")
    print(f"  INIT_DURATION: {args.init_duration}")
    print(f"  MIN_REQUESTS: {args.min_requests}")
    print(f"  REQUEST_TIMEOUT: {args.request_timeout}")
    print(f"  API_ENDPOINT: {args.api_endpoint}")
    print(f"  RESULTS_DIR: {args.results_dir}")
    print(f"  SINGLE_RUN: {args.single_run}")
    if args.single_run:
        print(f"  CONCURRENT_IMAGES: {args.concurrent_images}")
    print()
    
    # Validate compose file exists
    compose_file = os.path.abspath(args.compose_file)
    if not os.path.exists(compose_file):
        print(f"Error: Compose file not found: {compose_file}")
        sys.exit(1)
    
    # Create and run density test
    tester = DineInStreamDensity(
        compose_file=compose_file,
        results_dir=args.results_dir,
        api_endpoint=args.api_endpoint,
        images_dir=args.images_dir,
        orders_file=args.orders_file,
        target_latency_ms=args.target_latency_ms,
        latency_metric=args.latency_metric,
        density_increment=args.density_increment,
        init_duration=args.init_duration,
        min_requests=args.min_requests,
        request_timeout=args.request_timeout,
        single_run=args.single_run,
        concurrent_images=args.concurrent_images,
        max_iterations=args.max_iterations
    )
    
    result = tester.run()
    
    # Exit code based on result
    sys.exit(0 if result.met_target else 1)


if __name__ == "__main__":
    main()
