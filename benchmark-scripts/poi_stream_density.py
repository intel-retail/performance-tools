#!/usr/bin/env python3
"""
Stream Density Benchmark - Performance Tools

This file contains ALL benchmark logic including:
- Orchestration
- Latency calculation from metrics
- Pass/fail decisions
- Scaling decisions

POI only provides raw data via files and APIs.
"""

import argparse
import calendar
import csv
import json
import logging
import glob
import os
import re
import subprocess
import sys
import time
import psutil
from datetime import datetime, UTC
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark run"""
    target_latency_ms: float = 2000
    latency_metric: str = "avg"  # avg or max
    scene_increment: int = 1
    init_duration: int = 45
    stabilise_duration: int = 30
    max_iterations: int = 50
    max_alert_wait: int = 180
    benchmark_duration: int = 120
    single_run: bool = False
    single_run_scenes: int = 1
    results_dir: str = "./results"


@dataclass
class IterationMetrics:
    """Metrics for one iteration"""
    num_scenes: int
    new_component: Optional[str]
    latency_ms: float
    passed: bool
    memory_percent: float
    cpu_percent: float
    timestamp: str
    detections: int
    alerts: int
    raw_metrics: Dict[str, Any]  # Raw metrics for debugging


class StreamDensityBenchmark:
    """
    Orchestrates the benchmark - contains ALL decision logic.
    """
    
    def __init__(
        self,
        config: BenchmarkConfig,
        poi_scripts_dir: str,
        app_dir: str,
        resource_config: str = ""
    ):
        self.config = config
        self.poi_scripts_dir = Path(poi_scripts_dir)
        self.app_dir = Path(app_dir)
        self.resource_config = resource_config
        
        # POI script paths (only for actions, NOT for metrics)
        self.scale_script = self.poi_scripts_dir / "benchmark_scale.py"
        self.alert_script = self.poi_scripts_dir / "benchmark_alert.py"
        self.reset_script = self.poi_scripts_dir / "benchmark_reset.py"
        
        # Validate POI scripts exist
        for script in [self.scale_script, self.alert_script, self.reset_script]:
            if not script.exists():
                raise FileNotFoundError(f"POI must provide: {script}")
        
        os.makedirs(config.results_dir, exist_ok=True)
    
    def run(self) -> Dict[str, Any]:
        """Execute benchmark - makes all scaling decisions"""
        self._print_header()
        
        results = {
            "config": asdict(self.config),
            "iterations": [],
            "max_scenes": 0,
            "met_target": False,
            "best_iteration": None,
            "timestamp": datetime.now().isoformat()
        }
        
        num_scenes = self.config.single_run_scenes if self.config.single_run else 1
        best_metrics = None
        max_iterations = 1 if self.config.single_run else self.config.max_iterations
        
        for iteration in range(1, max_iterations + 1):
            # Get new component name for this iteration
            new_component = self._get_new_component_name(num_scenes)
            
            print(f"\n{'='*70}")
            print(f"Iteration {iteration}: Testing {num_scenes} scene(s)")
            if new_component:
                print(f"  New camera: {new_component}")
            print(f"{'='*70}")
            
            # Check memory
            if not self._memory_safe():
                logger.warning("Memory threshold exceeded – stopping")
                break
            
            # Reset POI state
            self._reset_poi_state()
            iteration_start = datetime.now(UTC)
            
            # Scale POI every iteration in stream-density mode.
            # In single-run mode, skip if services are already healthy.
            should_scale = True
            if self.config.single_run and self._is_poi_healthy():
                should_scale = False
                logger.info("Services already healthy - skipping scale in single-run mode")
            if should_scale:
                logger.info("Scaling POI to %d scene(s)...", num_scenes)
                self._scale_poi(num_scenes)
            
            # Wait for alert (tell POI to wait)
            if self.config.single_run:
                got_alert = self._call_alert_script(
                    new_component,
                    self.config.benchmark_duration,
                    iteration_start,
                )
            else:
                got_alert = self._wait_for_alert_with_retry(
                    new_component, iteration_start
                )
                if not new_component and got_alert:
                    # Keep baseline behavior aligned with legacy implementation.
                    time.sleep(self.config.stabilise_duration)
            
            # ================================================================
            # METRICS COLLECTION & LATENCY CALCULATION (IN PERFORMANCE TOOLS)
            # ================================================================
            
            # Read raw metrics files written by POI
            raw_metrics = self._collect_raw_metrics(iteration_start)

            # Save alert thumbnails for this iteration (best-effort).
            _save_alert_thumbnails(self.config.results_dir, iteration=iteration, since=iteration_start)
            
            # Calculate latency from raw metrics (PURE PERFORMANCE TOOLS LOGIC)
            latency_ms = self._calculate_latency_from_metrics(
                raw_metrics, 
                component_filter=new_component,
                metric_type=self.config.latency_metric
            )
            
            # Extract detections and alerts from raw metrics
            detections = self._extract_detection_count(raw_metrics)
            alerts = self._extract_alert_count(raw_metrics, new_component)
            
            # Create metrics object
            metrics = IterationMetrics(
                num_scenes=num_scenes,
                new_component=new_component or "baseline",
                latency_ms=latency_ms,
                passed=False,  # Will set below
                memory_percent=psutil.virtual_memory().percent,
                cpu_percent=psutil.cpu_percent(interval=1),
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                detections=detections,
                alerts=alerts,
                raw_metrics=raw_metrics
            )
            
            # DECISION: Determine pass/fail based on calculated latency
            metrics.passed = self._determine_pass_fail(
                metrics, got_alert, new_component
            )
            
            # Store iteration
            self._print_iteration(metrics)
            results["iterations"].append(asdict(metrics))
            
            # Track best iteration
            if metrics.passed and (
                not best_metrics or metrics.num_scenes > best_metrics.num_scenes
            ):
                best_metrics = metrics
            
            # DECISION: Continue or stop based on pass/fail
            if not metrics.passed:
                reason = "no alert" if (new_component and not got_alert) else "latency exceeded"
                logger.info("Iteration %d failed (%s) – stopping", iteration, reason)
                break
            
            # DECISION: Scale up for next iteration
            num_scenes += self.config.scene_increment
        
        # Finalize results
        if best_metrics:
            results["max_scenes"] = best_metrics.num_scenes
            results["met_target"] = True
            results["best_iteration"] = asdict(best_metrics)
        
        self._export_results(results)
        self._print_summary(results)
        
        return results
    
    # ========================================================================
    # LATENCY CALCULATION LOGIC (Moved from POI to Performance Tools)
    # ========================================================================
    
    def _collect_raw_metrics(self, since: datetime) -> Dict[str, Any]:
        """
        Collect raw metrics from POI's metrics files and API.
        
        POI writes:
        - /tmp/vlm_application_metrics_*.txt (VLM metrics)
        - Provides /api/v1/alerts endpoint
        
        Performance Tools reads these directly.
        """
        raw_metrics = {}
        
        # Read VLM metrics files (written by POI's alert_service.py)
        vlm_data = self._read_vlm_metrics_files(since)
        if vlm_data:
            raw_metrics['vlm'] = vlm_data
        
        # Read alerts API (provided by POI)
        alerts_data = self._read_alerts_api(since)
        if alerts_data:
            raw_metrics['alerts'] = alerts_data
        
        return raw_metrics
    
    def _read_vlm_metrics_files(self, since: datetime) -> Dict[str, Any]:
        """
        Parse VLM application metrics files.

        Uses a pure-Python implementation so no pandas/numpy is required.
        The files contain start/end event lines like:
          application=Person-of-Interest id=Camera_01-3 event=start timestamp_ms=1234567890
          application=Person-of-Interest id=Camera_01-3 event=end   timestamp_ms=1234568000
        """
        metrics = {}
        since_ms = int(calendar.timegm(since.timetuple()) * 1000)

        for d in (self.config.results_dir, "/tmp"):
            if not os.path.isdir(d):
                continue
            try:
                stats = _parse_vlm_metrics_dir(d, last_n_pairs=20, since_ms=since_ms)
                for app_name, avg_ms in stats.items():
                    metrics[f"vlm_{app_name}_avg_ms"] = avg_ms
            except Exception as e:
                logger.debug("Failed to parse VLM metrics from %s: %s", d, e)

        if not metrics:
            logger.warning("No VLM metrics parsed from results_dir or /tmp")
        return metrics
    
    def _read_alerts_api(self, since: datetime) -> Dict[str, Any]:
        """
        Read alerts from POI's API and calculate latencies.
        
        Alert format:
        {
            "timestamp": "2024-01-01T12:00:00.123Z",  # Frame capture time
            "dispatched_at": "2024-01-01T12:00:00.456Z",  # Alert dispatch time
            "match": {"camera_id": "Camera_01-3"}
        }
        
        Latency = dispatched_at - timestamp
        """
        import urllib.request
        from datetime import timezone
        
        try:
            req = urllib.request.Request("http://localhost:8000/api/v1/alerts?limit=200")
            with urllib.request.urlopen(req, timeout=10) as resp:
                alerts = json.loads(resp.read().decode())
        except Exception as e:
            logger.warning("Failed to fetch alerts: %s", e)
            return {}
        
        if not isinstance(alerts, list):
            return {}
        
        # Group latencies by camera
        camera_latencies = {}
        all_latencies = []
        
        since_aware = since.replace(tzinfo=timezone.utc) if since.tzinfo else since
        
        for alert in alerts:
            # Get camera ID
            camera_id = alert.get("match", {}).get("camera_id") or alert.get("camera_id", "unknown")
            
            # Get timestamps
            frame_ts = alert.get("timestamp", "")
            dispatched_ts = alert.get("dispatched_at", "")
            
            if not frame_ts or not dispatched_ts:
                continue
            
            try:
                # Parse timestamps
                start = datetime.fromisoformat(frame_ts.replace('Z', '+00:00'))
                end = datetime.fromisoformat(dispatched_ts.replace('Z', '+00:00'))
                
                # Make timezone aware
                if start.tzinfo is None:
                    start = start.replace(tzinfo=timezone.utc)
                if end.tzinfo is None:
                    end = end.replace(tzinfo=timezone.utc)
                
                # Filter by since time
                if end < since_aware:
                    continue
                
                # Calculate latency in milliseconds
                latency_ms = (end - start).total_seconds() * 1000
                
                if latency_ms >= 0:
                    all_latencies.append(latency_ms)
                    if camera_id not in camera_latencies:
                        camera_latencies[camera_id] = []
                    camera_latencies[camera_id].append(latency_ms)
                    
            except (ValueError, TypeError) as e:
                logger.debug("Failed to parse alert timestamps: %s", e)
                continue
        
        result = {
            "total_alerts": len(all_latencies),
            "alerts_by_camera": {k: len(v) for k, v in camera_latencies.items()},
        }
        
        if all_latencies:
            result["overall_avg_ms"] = sum(all_latencies) / len(all_latencies)
            result["overall_max_ms"] = max(all_latencies)
            result["overall_min_ms"] = min(all_latencies)
        
        # Per-camera metrics
        for camera_id, latencies in camera_latencies.items():
            result[f"camera_{camera_id}_avg_ms"] = sum(latencies) / len(latencies)
            result[f"camera_{camera_id}_max_ms"] = max(latencies)
            result[f"camera_{camera_id}_alert_count"] = len(latencies)
        
        return result
    
    def _calculate_latency_from_metrics(
        self, 
        raw_metrics: Dict[str, Any], 
        component_filter: Optional[str] = None,
        metric_type: str = "avg"
    ) -> float:
        """
        Calculate representative latency from raw metrics.
        
        Priority:
        1. VLM metrics for specific camera (if component_filter provided)
        2. VLM aggregate metrics
        3. Alerts API metrics
        """
        vlm_data = raw_metrics.get('vlm', {})
        alerts_data = raw_metrics.get('alerts', {})
        
        # If filtering to specific camera, look for camera-specific VLM metrics
        if component_filter:
            # Look for vlm_Person-of-Interest_{camera}_avg_ms
            for key, value in vlm_data.items():
                if component_filter in key and 'avg_ms' in key and value > 0:
                    logger.info("Using VLM metrics for %s: %.0fms", component_filter, value)
                    return value
            
            # Fallback to alerts API for specific camera
            camera_key = f"camera_{component_filter}_{metric_type}_ms"
            if camera_key in alerts_data and alerts_data[camera_key] > 0:
                logger.info("Using alerts API for %s: %.0fms", component_filter, alerts_data[camera_key])
                return alerts_data[camera_key]
        
        # No filter or no per-camera metrics - use aggregate
        if 'vlm_Person-of-Interest_avg_ms' in vlm_data and vlm_data['vlm_Person-of-Interest_avg_ms'] > 0:
            logger.info("Using aggregate VLM: %.0fms", vlm_data['vlm_Person-of-Interest_avg_ms'])
            return vlm_data['vlm_Person-of-Interest_avg_ms']
        
        if metric_type == 'avg' and alerts_data.get('overall_avg_ms', 0) > 0:
            return alerts_data['overall_avg_ms']
        elif metric_type == 'max' and alerts_data.get('overall_max_ms', 0) > 0:
            return alerts_data['overall_max_ms']
        
        logger.warning("No valid latency metrics found")
        return 0.0
    
    def _extract_detection_count(self, raw_metrics: Dict[str, Any]) -> int:
        """Extract detection count from raw metrics"""
        # Try to get from alerts API
        alerts_data = raw_metrics.get('alerts', {})
        return alerts_data.get('total_alerts', 0)
    
    def _extract_alert_count(self, raw_metrics: Dict[str, Any], component_filter: Optional[str]) -> int:
        """Extract alert count from raw metrics"""
        alerts_data = raw_metrics.get('alerts', {})
        
        if component_filter:
            return alerts_data.get(f"camera_{component_filter}_alert_count", 0)
        
        return alerts_data.get('total_alerts', 0)
    
    # ========================================================================
    # POI Action Delegation (only for actions, NOT for metrics)
    # ========================================================================
    
    def _scale_poi(self, num_scenes: int) -> None:
        """Tell POI to scale to N scenes"""
        result = subprocess.run(
            [
                sys.executable, str(self.scale_script),
                "--app_dir", str(self.app_dir),
                "--num_scenes", str(num_scenes),
                "--resource_config", self.resource_config
            ],
            capture_output=True, text=True
        )
        
        if result.returncode != 0:
            logger.error("Scale script failed: %s", result.stderr)
            raise RuntimeError(f"Scale failed: {result.stderr}")
        
        logger.info("POI scaled to %d scenes", num_scenes)
    
    def _get_new_component_name(self, num_scenes: int) -> Optional[str]:
        """Ask POI what component was added"""
        result = subprocess.run(
            [
                sys.executable, str(self.scale_script),
                "--app_dir", str(self.app_dir),
                "--get_new_component", str(num_scenes)
            ],
            capture_output=True, text=True
        )
        
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
        return None
    
    def _wait_for_alert_with_retry(
        self, component_name: Optional[str], iteration_start: datetime
    ) -> bool:
        """Tell POI to wait for alert"""
        if not component_name:
            return self._call_alert_script(None, self.config.init_duration, iteration_start)
        
        total_waited = 0
        while total_waited < self.config.max_alert_wait:
            window = min(self.config.init_duration, 
                        self.config.max_alert_wait - total_waited)
            
            got_alert = self._call_alert_script(component_name, window, iteration_start)
            
            if got_alert:
                logger.info("Alert received from %s", component_name)
                time.sleep(self.config.stabilise_duration)
                return True
            
            total_waited += window
            logger.info("No alert yet (%ds/%ds) - retrying", total_waited, self.config.max_alert_wait)
        
        return False
    
    def _call_alert_script(
        self, component_name: Optional[str], timeout: int, since: datetime
    ) -> bool:
        """Call POI's alert waiting script"""
        cmd = [
            sys.executable, str(self.alert_script),
            "--app_dir", str(self.app_dir),
            "--timeout", str(timeout),
            "--since", since.isoformat()
        ]
        if component_name:
            cmd.extend(["--component", component_name])
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0
    
    def _reset_poi_state(self) -> None:
        """Tell POI to reset state"""
        subprocess.run(
            [sys.executable, str(self.reset_script), "--app_dir", str(self.app_dir)],
            capture_output=True
        )
    
    def _is_poi_healthy(self) -> bool:
        """Check if POI services are already running"""
        result = subprocess.run(
            [sys.executable, str(self.scale_script), "--app_dir", str(self.app_dir), "--check_healthy"],
            capture_output=True, text=True
        )
        return result.returncode == 0
    
    # ========================================================================
    # Decision Logic (in performance-tools)
    # ========================================================================
    
    def _determine_pass_fail(
        self, metrics: IterationMetrics, got_alert: bool, new_component: Optional[str]
    ) -> bool:
        """DECISION: Did this iteration meet the latency target?"""
        if new_component:
            if not got_alert:
                print(f"  ✗ FAILED - No alert from {new_component}")
                return False
            elif metrics.latency_ms <= 0:
                print(f"  ✓ PASSED - Alert received")
                return True
            elif metrics.latency_ms <= self.config.target_latency_ms:
                print(f"  ✓ PASSED - {metrics.latency_ms:.0f}ms ≤ {self.config.target_latency_ms:.0f}ms")
                return True
            else:
                print(f"  ✗ FAILED - {metrics.latency_ms:.0f}ms > {self.config.target_latency_ms:.0f}ms")
                return False
        else:
            if got_alert and metrics.latency_ms <= 0:
                print("  ✓ PASSED - Baseline alert received")
                return True
            if metrics.latency_ms > 0 and metrics.latency_ms <= self.config.target_latency_ms:
                print(f"  ✓ PASSED - {metrics.latency_ms:.0f}ms ≤ {self.config.target_latency_ms:.0f}ms")
                return True
            print("  ✗ FAILED - Baseline did not meet alert/latency criteria")
            return False
    
    def _memory_safe(self) -> bool:
        """DECISION: Is memory usage safe?"""
        mem = psutil.virtual_memory()
        if mem.percent > 90:
            logger.warning("Memory at %.1f%% (threshold 90%%)", mem.percent)
            return False
        return True
    
    # ========================================================================
    # Output Methods
    # ========================================================================
    
    def _print_header(self):
        print("=" * 70)
        print("Stream Density Benchmark - Performance Tools")
        print("(Performance Tools handles ALL metrics & decisions)")
        print("=" * 70)
        print(f"  Target Latency:    {self.config.target_latency_ms:.0f}ms")
        print(f"  Latency Metric:    {self.config.latency_metric}")
        print(f"  Scene Increment:   +{self.config.scene_increment}")
        print(f"  Results Dir:       {self.config.results_dir}")
        print("=" * 70)
    
    def _print_iteration(self, metrics: IterationMetrics):
        print(f"\n  Scenes:      {metrics.num_scenes}")
        if metrics.new_component != "baseline":
            print(f"  New Camera:  {metrics.new_component}")
        print(f"  Latency:     {metrics.latency_ms:.0f}ms")
        print(f"  Detections:  {metrics.detections}")
        print(f"  Alerts:      {metrics.alerts}")
        print(f"  Memory:      {metrics.memory_percent:.1f}%")
        print(f"  CPU:         {metrics.cpu_percent:.1f}%")
    
    def _print_summary(self, results: Dict):
        print("\n" + "=" * 70)
        print("BENCHMARK RESULTS")
        print("=" * 70)
        print(f"  Target Latency:  {self.config.target_latency_ms:.0f}ms")
        print(f"  Max Scenes:      {results['max_scenes']}")
        print(f"  Met Target:      {'Yes' if results['met_target'] else 'No'}")
        print("=" * 70)

        iterations = results.get("iterations", [])
        if not iterations:
            return

        print()
        print(
            f"{'Scenes':<8}"
            f"{'Camera':<16}"
            f"{'Latency':<12}"
            f"{'Detections':<12}"
            f"{'Alerts':<8}"
            f"{'Mem %':<8}"
            f"{'CPU %':<8}"
            f"{'Status':<8}"
        )
        print("-" * 80)
        for it in iterations:
            status = "PASS" if it.get("passed") else "FAIL"
            camera = it.get("new_component") or "baseline"
            print(
                f"{it.get('num_scenes', 0):<8}"
                f"{camera:<16}"
                f"{it.get('latency_ms', 0):<12.0f}"
                f"{it.get('detections', 0):<12}"
                f"{it.get('alerts', 0):<8}"
                f"{it.get('memory_percent', 0):<8.1f}"
                f"{it.get('cpu_percent', 0):<8.1f}"
                f"{status:<8}"
            )
        print("=" * 80)
    
    def _export_results(self, results: Dict):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        json_path = os.path.join(self.config.results_dir, f"stream_density_{timestamp}.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nJSON results: {json_path}")
        
        csv_path = os.path.join(self.config.results_dir, f"stream_density_{timestamp}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["scenes", "component", "latency_ms", "detections", "alerts", "passed"])
            for it in results['iterations']:
                writer.writerow([
                    it['num_scenes'], it['new_component'],
                    f"{it['latency_ms']:.0f}", it['detections'],
                    it['alerts'], it['passed']
                ])
        print(f"CSV results: {csv_path}")


def _save_alert_thumbnails(results_dir: str, iteration: int = 1,
                           since: Optional[datetime] = None) -> int:
    """Fetch alert thumbnails from POI API and save under results_dir.

    Returns the number of thumbnails saved.
    """
    import urllib.error
    import urllib.request
    from datetime import datetime as _dt, timezone as _tz

    thumbs_dir = os.path.join(results_dir, f"thumbnails_iter{iteration}")
    try:
        os.makedirs(thumbs_dir, exist_ok=True)
    except PermissionError:
        logger.warning(
            "Cannot create thumbnails directory %s (permission denied)",
            thumbs_dir,
        )
        return 0

    try:
        req = urllib.request.Request("http://localhost:8000/api/v1/alerts")
        with urllib.request.urlopen(req, timeout=10) as resp:
            alerts = json.loads(resp.read().decode())
    except Exception as e:
        logger.warning("Failed to fetch alerts for thumbnails: %s", e)
        return 0

    if not isinstance(alerts, list) or not alerts:
        return 0

    saved = 0
    for i, alert in enumerate(alerts):
        if since is not None:
            dispatched_str = alert.get("dispatched_at", "")
            if dispatched_str:
                try:
                    dispatched = _dt.fromisoformat(dispatched_str.replace("Z", "+00:00"))
                    since_aware = since.astimezone(_tz.utc) if since.tzinfo else since.replace(tzinfo=_tz.utc)
                    dispatched_aware = dispatched.astimezone(_tz.utc) if dispatched.tzinfo else dispatched.replace(tzinfo=_tz.utc)
                    if dispatched_aware < since_aware:
                        continue
                except (ValueError, TypeError):
                    pass

        match_data = alert.get("match", {})
        thumb_url = match_data.get("thumbnail_path") or alert.get("thumbnail_path") or ""
        object_id = alert.get("object_id", "")
        poi_id = alert.get("poi_id", "unknown")
        camera_id = match_data.get("camera_id") or alert.get("camera_id", "unknown")

        if thumb_url.startswith("/"):
            thumb_url = f"http://localhost:8000{thumb_url}"
        elif not thumb_url and object_id:
            thumb_url = f"http://localhost:8000/api/v1/thumbnail/{object_id}"
        else:
            continue

        try:
            req = urllib.request.Request(thumb_url)
            with urllib.request.urlopen(req, timeout=10) as resp:
                img_data = resp.read()

            safe_cam = re.sub(r"[^a-zA-Z0-9_-]", "_", camera_id)
            safe_poi = re.sub(r"[^a-zA-Z0-9_-]", "_", poi_id)
            out_name = f"alert_{i:03d}_{safe_poi}_{safe_cam}.jpg"
            out_path = os.path.join(thumbs_dir, out_name)
            with open(out_path, "wb") as fh:
                fh.write(img_data)
            saved += 1
        except urllib.error.HTTPError as e:
            logger.debug("Thumbnail HTTP %d for %s", e.code, thumb_url)
        except Exception as e:
            logger.debug("Thumbnail save failed for %s: %s", thumb_url, e)

    if saved:
        logger.info("Saved %d alert thumbnails to %s", saved, thumbs_dir)
    return saved
    

# ---------------------------------------------------------------------------
# Pure-Python VLM metrics parser (no pandas/numpy dependency)
# ---------------------------------------------------------------------------

def _parse_vlm_metrics_dir(results_dir: str, last_n_pairs: int = 20,
                            since_ms: Optional[int] = None) -> Dict[str, float]:
    """Return per-app average latency (ms) from the most recent
    vlm_application_metrics*.txt file in *results_dir*.

    Parses lines of the form:
      application=<name> id=<id> event=start|end timestamp_ms=<epoch_ms>
    Pairs start/end events (LIFO) and returns the average of the last
    *last_n_pairs* completed durations per app_id key.
    """
    import re as _re
    from collections import defaultdict
    from pathlib import Path as _Path

    files = sorted(
        _Path(results_dir).rglob("vlm_application_metrics*.txt"),
        key=lambda p: p.stat().st_mtime,
    )
    if not files:
        return {}

    timing: dict = defaultdict(list)
    kv_pat = _re.compile(r'(\w+)=([^\s]+)')
    with open(files[-1]) as fh:
        for line in fh:
            if "application=" not in line or "timestamp_ms=" not in line:
                continue
            data = dict(kv_pat.findall(line))
            app = data.get("application", "")
            id_ = data.get("id", "")
            event = data.get("event", "")
            ts_str = data.get("timestamp_ms", "")
            if not (app and id_ and event in ("start", "end") and ts_str):
                continue
            ts = int(ts_str)
            if since_ms is not None and ts < since_ms:
                continue
            timing[f"{app}_{id_}"].append({"event": event, "timestamp_ms": ts})

    result: Dict[str, float] = {}
    for app_id, events in timing.items():
        events.sort(key=lambda x: x["timestamp_ms"])
        durations = []
        stack: list = []
        for ev in events:
            if ev["event"] == "start":
                stack.append(ev["timestamp_ms"])
            elif ev["event"] == "end" and stack:
                durations.append(ev["timestamp_ms"] - stack.pop())
        if durations:
            tail = durations[-last_n_pairs:]
            result[app_id] = sum(tail) / len(tail)
    return result


# ---------------------------------------------------------------------------
# poi_scaling helper loader (loads from POI benchmark/ folder)
# ---------------------------------------------------------------------------

def _load_poi_scaling(app_dir: str):
    """Import poi_scaling from person-of-interest/benchmark/."""
    import importlib.util
    scaling_path = os.path.abspath(
        os.path.join(app_dir, "benchmark", "poi_scaling.py")
    )
    if not os.path.exists(scaling_path):
        raise FileNotFoundError(f"poi_scaling.py not found at {scaling_path}")
    spec = importlib.util.spec_from_file_location("poi_scaling", scaling_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load poi_scaling from {scaling_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    if len(sys.argv) > 1 and sys.argv[1] in {"generate", "clean", "down"}:
        _run_subcommand()
        return

    parser = argparse.ArgumentParser(description="Stream Density Benchmark")
    parser.add_argument("--poi_scripts_dir", required=True)
    parser.add_argument("--app_dir", required=True)
    parser.add_argument("--target_latency_ms", type=float, default=2000)
    parser.add_argument("--latency_metric", choices=["avg", "max"], default="avg")
    parser.add_argument("--scene_increment", type=int, default=1)
    parser.add_argument("--init_duration", type=int, default=45)
    parser.add_argument("--stabilise_duration", type=int, default=30)
    parser.add_argument("--max_iterations", type=int, default=50)
    parser.add_argument("--max_alert_wait", type=int, default=180)
    parser.add_argument("--benchmark_duration", type=int, default=120)
    parser.add_argument("--single_run", action="store_true")
    parser.add_argument("--scenes", type=int, default=1)
    parser.add_argument("--results_dir", default="./results")
    parser.add_argument("--resource_config", default="")

    args = parser.parse_args()

    config = BenchmarkConfig(
        target_latency_ms=args.target_latency_ms,
        latency_metric=args.latency_metric,
        scene_increment=args.scene_increment,
        init_duration=args.init_duration,
        stabilise_duration=args.stabilise_duration,
        max_iterations=args.max_iterations,
        max_alert_wait=args.max_alert_wait,
        benchmark_duration=args.benchmark_duration,
        single_run=args.single_run,
        single_run_scenes=args.scenes,
        results_dir=args.results_dir,
    )

    benchmark = StreamDensityBenchmark(
        config=config,
        poi_scripts_dir=args.poi_scripts_dir,
        app_dir=args.app_dir,
        resource_config=args.resource_config,
    )

    results = benchmark.run()
    sys.exit(0 if results["met_target"] else 1)


def _run_subcommand() -> None:
    """Handle generate / clean / down subcommands via poi_scaling.py."""
    import shutil
    sub = sys.argv[1]
    parser = argparse.ArgumentParser(description=f"{sub} helper")
    parser.add_argument("command")
    parser.add_argument("app_dir")
    parser.add_argument("--scenes", type=int, default=1)
    parser.add_argument("--resource_config", default="")
    args = parser.parse_args()

    ps = _load_poi_scaling(args.app_dir)

    if sub == "generate":
        ps.set_stream_density(args.app_dir, args.scenes)
        ps.generate_dlstreamer_config(args.app_dir, args.scenes)
        ps.generate_cameras_override(args.app_dir, args.scenes)
        ps.reinit_env(args.app_dir, resource_config=args.resource_config)
        print(f"Generated overrides for {args.scenes} scene(s).")

    elif sub == "clean":
        bak = ps.zone_config_path(args.app_dir).with_suffix(".json.bak")
        if bak.exists():
            shutil.copy2(bak, ps.zone_config_path(args.app_dir))
            bak.unlink()
        else:
            ps.set_stream_density(args.app_dir, 1)
        ps.generate_dlstreamer_config(args.app_dir, 1)
        ps.clean_cameras_override(args.app_dir)
        ps.reinit_env(args.app_dir, resource_config=args.resource_config)
        print("Cleaned stream-density overrides.")

    elif sub == "down":
        ps.docker_compose(args.app_dir, "down -t 30 --volumes --remove-orphans")
        ps.clean_cameras_override(args.app_dir)
        print("Brought down stream-density services.")


if __name__ == "__main__":
    main()