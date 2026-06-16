#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation.
# SPDX-License-Identifier: Apache-2.0

"""
Suspicious Activity Detection (SAD) Stream Density Benchmark

Iteratively increases the number of camera/scene pipelines until end-to-end
latency exceeds a configurable threshold.

Scaling is achieved by:
  1. Updating ``stream_density`` in ``zone_config.json``
  2. Re-running ``init.sh`` to regenerate ``.env`` + DLStreamer pipeline config
  3. Generating ``docker-compose.cameras.yaml`` with additional RTSP camera
     streams (``lp-cams-N``) for each new camera
  4. Restarting ``scene-import`` (imports cloned scenes into SceneScape),
     ``lp-cams-*``, ``lp-video`` (DLStreamer), and ``swlp-service``
  5. SceneScape core services (web, controller, broker, etc.), ovms-vlm,
     behavioral-analysis, seaweedfs, and alert-service stay running

Sub-commands
------------
run       Full automated stream-density loop.
generate  Write a ``docker-compose.cameras.yaml`` override for *N* scenes.
clean     Revert overrides and set ``stream_density`` back to 1.
down      Tear down all services.

Environment variables
---------------------
TARGET_LATENCY_MS       Latency threshold in ms  (default: 30000)
LATENCY_METRIC          Which metric to compare: avg | max  (default: avg)
SCENE_INCREMENT         Scenes to add per iteration  (default: 1)
INIT_DURATION           Warm-up seconds after restart  (default: 90)
STABILISE_DURATION      Extra wait for pipeline to stabilise  (default: 30)
RESULTS_DIR             Where to write results  (default: ./results)
MAX_ITERATIONS          Safety cap on iterations  (default: 50)
"""

import argparse
import csv
import glob
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Dict, List, Optional

import psutil

# Import the latency extractor from the sibling module
from consolidate_multiple_run_of_metrics import get_vlm_application_latency_stream_density

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is not None:
        try:
            return int(v)
        except ValueError:
            logger.warning("Invalid %s=%s, using default %d", name, v, default)
    return default


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    if v is not None:
        try:
            return float(v)
        except ValueError:
            logger.warning("Invalid %s=%s, using default %.1f", name, v, default)
    return default


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class IterationResult:
    """Metrics captured during one iteration."""
    num_scenes: int
    latency_ms: float  # chosen metric (avg or max)
    latency_details: Dict[str, float] = field(default_factory=dict)
    passed: bool = False
    memory_percent: float = 0.0
    cpu_percent: float = 0.0
    timestamp: str = ""
    # Throughput metrics
    actual_samples: int = 0
    expected_samples: int = 0
    throughput_ratio: float = 1.0  # actual / expected (1.0 = healthy)
    samples_per_scene: float = 0.0


@dataclass
class StreamDensityResult:
    """Aggregate result of the full stream-density run."""
    target_latency_ms: float = 30000.0
    max_scenes: int = 0
    met_target: bool = False
    iterations: List[IterationResult] = field(default_factory=list)
    best_iteration: Optional[IterationResult] = None


# ---------------------------------------------------------------------------
# Helpers – zone_config.json manipulation
# ---------------------------------------------------------------------------

def _zone_config_path(app_dir: str) -> Path:
    return Path(app_dir) / "configs" / "zone_config.json"


def _read_zone_config(app_dir: str) -> dict:
    p = _zone_config_path(app_dir)
    with open(p) as f:
        return json.load(f)


def _write_zone_config(app_dir: str, cfg: dict) -> None:
    p = _zone_config_path(app_dir)
    # Keep a backup before first write
    bak = p.with_suffix(".json.bak")
    if not bak.exists():
        shutil.copy2(p, bak)
    with open(p, "w") as f:
        json.dump(cfg, f, indent=2)
    logger.info("Updated %s  (stream_density=%s)", p, cfg.get("stream_density"))


def _set_stream_density(app_dir: str, density: int) -> None:
    cfg = _read_zone_config(app_dir)
    cfg["stream_density"] = density
    _write_zone_config(app_dir, cfg)


# ---------------------------------------------------------------------------
# Helpers – docker compose
# ---------------------------------------------------------------------------

def _compose_cmd(app_dir: str) -> str:
    """Build the base ``docker compose`` invocation matching the main Makefile."""
    scenescape_dir = str(Path(app_dir) / ".." / "scenescape")
    scenescape_compose = os.path.join(scenescape_dir, "docker-compose.yaml")
    lp_compose = os.path.join(app_dir, "docker", "docker-compose.yaml")
    env_file = os.path.join(app_dir, "docker", ".env")

    parts = [
        "docker compose",
        f"--project-directory {shlex.quote(app_dir)}",
        f"--env-file {shlex.quote(env_file)}",
        f"-f {shlex.quote(scenescape_compose)}",
        f"-f {shlex.quote(lp_compose)}",
    ]
    # Layer in cameras override if it exists
    cameras_override = os.path.join(app_dir, "docker", "docker-compose.cameras.yaml")
    if os.path.isfile(cameras_override):
        parts.append(f"-f {shlex.quote(cameras_override)}")
    return " ".join(parts)


def _docker_compose(app_dir: str, action: str, timeout: int = 60) -> int:
    cmd = f"{_compose_cmd(app_dir)} {action}"
    logger.info("Running: %s", cmd)
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0 and "down" not in action:
        logger.warning("docker compose stderr:\n%s", result.stderr[-500:])
    return result.returncode


def _run_cmd(cmd: str) -> subprocess.CompletedProcess:
    """Run a shell command, log it, and return the result."""
    logger.debug("Running: %s", cmd)
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)


def _delete_cloned_scenes(app_dir: str, num_scenes: int) -> None:
    """Delete previously-cloned scenes from SceneScape via REST API."""
    import re as _re
    import ssl
    import urllib.request
    import urllib.error

    env_file = os.path.join(app_dir, "docker", ".env")
    supass = ""
    if os.path.isfile(env_file):
        for line in open(env_file):
            if line.startswith("SUPASS="):
                supass = line.strip().split("=", 1)[1]
                break
    if not supass:
        logger.warning("Could not read SUPASS from .env — skipping scene cleanup")
        return

    base_url = "https://localhost/api/v1"
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    # Authenticate
    auth_data = json.dumps({"username": "admin", "password": supass}).encode()
    req = urllib.request.Request(
        f"{base_url}/auth", data=auth_data,
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, context=ctx) as resp:
            token = json.loads(resp.read()).get("token", "")
    except Exception as e:
        logger.warning("Failed to authenticate with SceneScape API: %s", e)
        return
    if not token:
        return

    auth_header = {"Authorization": f"Token {token}"}

    # List scenes
    req = urllib.request.Request(f"{base_url}/scenes", headers=auth_header)
    try:
        with urllib.request.urlopen(req, context=ctx) as resp:
            scenes = json.loads(resp.read()).get("results", [])
    except Exception:
        return

    # Delete cloned scenes (those with -N suffix in name)
    for scene in scenes:
        name = scene.get("name", "")
        uid = scene.get("uid", "")
        if _re.search(r'-\d+$', name):
            logger.info("Deleting cloned scene: %s (%s)", name, uid)
            req = urllib.request.Request(
                f"{base_url}/scene/{uid}",
                method="DELETE", headers=auth_header)
            try:
                urllib.request.urlopen(req, context=ctx)
            except urllib.error.HTTPError as e:
                logger.warning("  DELETE failed (%s): %s", e.code, e.reason)

    # Delete orphaned cameras (cloned camera names with -N suffix)
    # These must be cleaned up or SceneScape rejects re-import with
    # "orphaned camera with the name '...' already exists."
    req = urllib.request.Request(f"{base_url}/cameras", headers=auth_header)
    try:
        with urllib.request.urlopen(req, context=ctx) as resp:
            cameras = json.loads(resp.read()).get("results", [])
    except Exception:
        cameras = []
    for cam in cameras:
        cam_name = cam.get("name", "")
        cam_id = cam.get("uid", cam.get("id", cam.get("sensor_id", "")))
        if _re.search(r'-\d+$', cam_name) and cam_id:
            logger.info("Deleting orphaned camera: %s (%s)", cam_name, cam_id)
            req = urllib.request.Request(
                f"{base_url}/camera/{cam_id}",
                method="DELETE", headers=auth_header)
            try:
                urllib.request.urlopen(req, context=ctx)
            except urllib.error.HTTPError as e:
                logger.warning("  DELETE camera failed (%s): %s", e.code, e.reason)


# ---------------------------------------------------------------------------
# Helpers – cameras override generation  (REAL pipeline scaling)
# ---------------------------------------------------------------------------

def _read_base_config(app_dir: str) -> dict:
    """Read base camera/scene/video config from zone_config.json."""
    cfg = _read_zone_config(app_dir)
    scenes = cfg.get("scenes", [])
    if scenes:
        s = scenes[0]
        camera = (s.get("cameras") or [s.get("camera_name", "lp-camera1")])[0]
        video = s.get("video_file", "lp-camera1.mp4")
    else:
        camera = cfg.get("camera_name", "lp-camera1")
        video = cfg.get("video_file", "lp-camera1.mp4")
    return {"camera_name": camera, "video_file": video}


def _generate_cameras_override(app_dir: str, num_scenes: int) -> None:
    """
    Generate ``docker-compose.cameras.yaml`` that adds real RTSP camera
    streams for each additional camera beyond the base one.

    For N scenes, we create lp-cams-2 through lp-cams-N services, each
    streaming the same video on a unique RTSP path.
    """
    override_path = Path(app_dir) / "docker" / "docker-compose.cameras.yaml"
    base = _read_base_config(app_dir)
    base_camera = base["camera_name"]
    base_video = base["video_file"]

    with open(override_path, "w") as f:
        f.write("# Auto-generated by swlp_stream_density.py — do not edit\n")
        f.write(f"# Stream density: {num_scenes} scenes\n\n")
        f.write("services:\n")

        # Additional RTSP camera streams (camera 2..N)
        for i in range(2, num_scenes + 1):
            cam_name = f"{base_camera}-{i}"
            svc_name = f"lp-cams-{i}"
            f.write(f"  {svc_name}:\n")
            f.write(f"    image: linuxserver/ffmpeg:version-8.0-cli\n")
            f.write(f'    command: "-nostdin -re -stream_loop -1 '
                    f'-i /workspace/media/{base_video} '
                    f'-c:v copy -an -f rtsp -rtsp_transport tcp '
                    f'rtsp://mediaserver:8554/{cam_name}"\n')
            f.write(f"    volumes:\n")
            f.write(f"      - vol-sample-data:/workspace/media\n")
            f.write(f"    networks:\n")
            f.write(f"      - storewide-lp\n")
            f.write(f"    depends_on:\n")
            f.write(f"      - mediaserver\n")
            f.write(f'    restart: "no"\n')
            f.write(f"\n")

        # Override swlp-service to set STREAM_DENSITY
        f.write(f"  swlp-service:\n")
        f.write(f"    environment:\n")
        f.write(f"      STREAM_DENSITY: \"{num_scenes}\"\n")

    logger.info("Generated cameras override: %s  (%d scenes, %d extra RTSP streams)",
                override_path, num_scenes, max(0, num_scenes - 1))


def _generate_dlstreamer_config(app_dir: str, num_scenes: int) -> None:
    """
    Generate a multi-pipeline DLStreamer config.json for N cameras.

    Reads the **rendered** config (output of init.sh, with all pipeline
    variables like DECODE, QUEUE_OPTIONS etc. already resolved) and
    duplicates the base pipeline entry for each additional camera.

    IMPORTANT: This must run AFTER _reinit_env() so that init.sh has
    already produced a fully-resolved single-camera config.
    """
    scenescape_dir = Path(app_dir) / ".." / "scenescape"
    app_name = Path(app_dir).name
    output_path = scenescape_dir / "dlstreamer-pipeline-server" / f"{app_name}-pipeline-config.json"

    base = _read_base_config(app_dir)
    base_camera = base["camera_name"]

    # Read the rendered config that init.sh just produced (all {{VAR}} resolved)
    if not output_path.exists():
        logger.error("Rendered DLStreamer config not found at %s — did init.sh run?", output_path)
        return

    with open(output_path) as fh:
        rendered_cfg = json.load(fh)

    base_pipeline = rendered_cfg["config"]["pipelines"][0]

    pipelines = []
    for i in range(1, num_scenes + 1):
        cam_name = base_camera if i == 1 else f"{base_camera}-{i}"
        # Deep-copy pipeline and update for this camera
        pipeline_str = json.dumps(base_pipeline)
        pipeline_str = pipeline_str.replace(base_camera, cam_name)
        pipeline = json.loads(pipeline_str)
        pipeline["name"] = f"reid_{cam_name}"
        pipelines.append(pipeline)

    output_cfg = {
        "config": {
            "logging": rendered_cfg["config"].get("logging", {
                "C_LOG_LEVEL": "INFO",
                "PY_LOG_LEVEL": "INFO"
            }),
            "pipelines": pipelines,
        }
    }

    with open(output_path, "w") as fh:
        json.dump(output_cfg, fh, indent=2)

    logger.info("Generated DLStreamer config: %s  (%d pipelines)", output_path, len(pipelines))


def _reinit_env(app_dir: str) -> None:
    """
    Re-run init.sh to regenerate .env with the updated STREAM_DENSITY.
    This ensures scene-import picks up the new density value.
    """
    init_script = Path(app_dir) / ".." / "scenescape" / "scripts" / "init.sh"
    if not init_script.exists():
        logger.warning("init.sh not found at %s — skipping .env regeneration", init_script)
        return

    cmd = f"bash {shlex.quote(str(init_script))} {shlex.quote(app_dir)}"
    logger.info("Re-running init.sh to update .env …")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        logger.warning("init.sh returned non-zero:\n%s", result.stderr[-500:])
    else:
        logger.info("init.sh completed — .env updated")


def _wait_for_web_healthy(timeout: int = 300) -> None:
    """Block until storewide-lp-web-1 is healthy or timeout expires."""
    for attempt in range(timeout // 5):
        result = subprocess.run(
            "docker inspect storewide-lp-web-1 --format '{{.State.Health.Status}}'",
            shell=True, capture_output=True, text=True)
        status = result.stdout.strip()
        if status == "healthy":
            logger.info("Web container is healthy (after %ds)", attempt * 5)
            return
        if attempt % 6 == 0:
            logger.info("  web status: %s  (waiting…)", status)
        time.sleep(5)
    logger.warning("Web container did not become healthy after %ds — continuing anyway", timeout)


def _scale_pipeline_services(app_dir: str, num_scenes: int, wait: int = 90) -> None:
    """
    Scale the actual video pipeline to N cameras.

    Steps:
      1. Generate DLStreamer config with N pipelines
      2. Generate docker-compose.cameras.yaml with N-1 extra RTSP streams
      3. Re-run init.sh to update .env (STREAM_DENSITY)
      4. Restart scene-import → imports cloned scenes
      5. Recreate lp-video → picks up new DLStreamer config
      6. Recreate swlp-service → subscribes to new scene topics
    """
    logger.info("Scaling to %d scene(s) …", num_scenes)

    # Update config files
    # Note: _reinit_env must run BEFORE _generate_dlstreamer_config because
    # init.sh overwrites the DLStreamer config with a single-camera template.
    _set_stream_density(app_dir, num_scenes)
    _generate_cameras_override(app_dir, num_scenes)
    _reinit_env(app_dir)
    _generate_dlstreamer_config(app_dir, num_scenes)

    # Bring up any new camera services defined in the override
    logger.info("Starting new camera streams …")
    _docker_compose(app_dir, "up -d --no-recreate")

    # Wait for the web container to become healthy (needed after cold start)
    _wait_for_web_healthy()

    # Clean up leftover extract directories inside the web container so
    # SceneScape's ImportScene.extractZip() doesn't find stale JSON files
    # from previous iterations (it uses exist_ok=True without clearing).
    logger.info("Cleaning stale scene-import extract dirs in web container …")
    _run_cmd("docker exec storewide-lp-web-1 bash -c "
             "'rm -rf /workspace/media/storewide-loss-prevention-[0-9]*'")

    # Also delete previously-cloned scenes from SceneScape DB via REST API
    # so re-import succeeds (scenes with duplicate names are rejected).
    _delete_cloned_scenes(app_dir, num_scenes)

    # Re-run scene-import to import cloned scenes for new cameras
    logger.info("Re-running scene-import for %d scenes …", num_scenes)
    _docker_compose(app_dir, "rm -f -s scene-import")
    _docker_compose(app_dir, "up -d scene-import")
    # Wait for scene-import to complete (it's a one-shot container)
    time.sleep(15)

    # Force-recreate lp-video so updated DLStreamer config is mounted
    logger.info("Recreating lp-video (DLStreamer) with new config …")
    _docker_compose(app_dir, "up -d --force-recreate lp-video")

    # Force-recreate swlp-service to pick up new env / scene subscriptions
    logger.info("Recreating swlp-service …")
    _docker_compose(app_dir, "up -d --force-recreate swlp-service")

    logger.info("Waiting %ds for services to initialise …", wait)
    time.sleep(wait)


def _clean_cameras_override(app_dir: str) -> None:
    override_path = Path(app_dir) / "docker" / "docker-compose.cameras.yaml"
    if override_path.exists():
        override_path.unlink()
        logger.info("Removed %s", override_path)


# ---------------------------------------------------------------------------
# Latency collection
# ---------------------------------------------------------------------------

def _collect_latency_from_docker_logs(app_dir: str, duration_secs: int = 30, num_scenes: int = 1) -> Dict[str, float]:
    """
    Extract end-to-end latency from swlp-service docker logs.

    Measures the time between "Published BA request" (last_frame_ts) and the
    corresponding "BA queue: status update" (timestamp) for the same person_id
    + region_id pair.

    Returns a dict with latency statistics.
    """
    container = "storewide-lp-swlp-service-1"
    since_arg = f"--since={duration_secs + 30}s"
    cmd = f"docker logs {container} {since_arg} 2>&1"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        logger.warning("Failed to get docker logs: %s", result.stderr[:200])
        return {}

    # Parse structured JSON log lines
    ba_requests: Dict[str, datetime] = {}  # key: person_id+region_id -> last_frame_ts
    latencies: list = []

    for line in result.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue

        event = entry.get("event", "")
        person_id = entry.get("person_id", "")
        region_id = entry.get("region_id", "")
        key = f"{person_id}:{region_id}"

        if event == "Published BA request":
            last_frame_ts = entry.get("last_frame_ts", "")
            if last_frame_ts:
                try:
                    ts = datetime.fromisoformat(last_frame_ts.replace("Z", "+00:00"))
                    ba_requests[key] = ts
                except (ValueError, TypeError):
                    pass

        elif event == "BA queue: status update" and key in ba_requests:
            ts_str = entry.get("timestamp", "")
            if ts_str:
                try:
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    start = ba_requests.pop(key)
                    delta_ms = (ts - start).total_seconds() * 1000
                    if delta_ms > 0:
                        latencies.append(delta_ms)
                except (ValueError, TypeError):
                    pass

    stats: Dict[str, float] = {}
    if latencies:
        stats["e2e_latency_count"] = len(latencies)
        stats["e2e_latency_avg_ms"] = mean(latencies)
        stats["e2e_latency_median_ms"] = median(latencies)
        stats["e2e_latency_min_ms"] = min(latencies)
        stats["e2e_latency_max_ms"] = max(latencies)
        # Use the configured scene count (log-based counting is unreliable
        # because cloned scenes sharing the same video may not all produce
        # detections within the collection window).
        stats["active_scenes"] = num_scenes
        logger.info("Collected %d BA round-trip latency samples  (avg=%.0fms, active_scenes=%d)",
                     len(latencies), stats["e2e_latency_avg_ms"], stats["active_scenes"])
    else:
        logger.warning("No BA round-trip samples found in docker logs")

    # Also try the vlm_application_metrics files as a secondary source
    file_stats = _collect_latency_from_files(os.path.join(app_dir, "results"))
    stats.update(file_stats)

    return stats


def _collect_latency_from_files(results_dir: str) -> Dict[str, float]:
    """
    Use ``get_vlm_application_latency_stream_density`` to extract latency
    from the most recent vlm_application_metrics file, using only the
    last 20 completed start/end pairs to reflect current-iteration
    performance.
    """
    all_stats: Dict[str, float] = {}
    search_dirs = [results_dir, "/tmp"]
    # Also search immediate subdirectories (e.g. results/swlp/, results/ba/)
    if os.path.isdir(results_dir):
        for entry in os.listdir(results_dir):
            sub = os.path.join(results_dir, entry)
            if os.path.isdir(sub):
                search_dirs.append(sub)
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        try:
            stats = get_vlm_application_latency_stream_density(d, last_n_pairs=20)
            if stats:
                # Convert to a consistent format
                for app_id, avg_ms in stats.items():
                    all_stats[f"vlm_{app_id}_avg_ms"] = avg_ms
                logger.info("VLM file-based latency: %s", stats)
        except Exception as e:
            logger.warning("Failed to parse vlm metrics in %s: %s", d, e)
    return all_stats


def _extract_latency_value(stats: Dict[str, float], metric: str) -> float:
    """
    Given the latency stats dict, return a single representative latency (ms).

    Prefers ``get_vlm_application_latency_stream_density`` file-based values,
    falls back to docker-log E2E latency.
    """
    # Primary: vlm_application_metrics file values
    vlm_values = [v for k, v in stats.items()
                  if k.startswith("vlm_") and isinstance(v, (int, float)) and v > 0]
    if vlm_values:
        if metric == "max":
            return max(vlm_values)
        return mean(vlm_values)

    # Fallback: docker-log based E2E latency
    if metric == "max" and "e2e_latency_max_ms" in stats:
        return stats["e2e_latency_max_ms"]
    if "e2e_latency_avg_ms" in stats:
        return stats["e2e_latency_avg_ms"]

    return 0.0


def _clean_metrics(results_dir: str) -> None:
    """
    Archive previous-run metrics files into a timestamped subdirectory
    so they are retained for later analysis, then remove any leftover
    copies from /tmp.
    """
    patterns = [
        "vlm_application_metrics*.txt",
        "vlm_performance_metrics*.txt",
    ]

    # Archive files from results_dir (and its subdirectories) instead of deleting
    archive_dirs = [results_dir]
    if os.path.isdir(results_dir):
        for entry in os.listdir(results_dir):
            sub = os.path.join(results_dir, entry)
            if os.path.isdir(sub) and not entry.startswith("archived_"):
                archive_dirs.append(sub)

    archived_count = 0
    for d in archive_dirs:
        for pat in patterns:
            for f in glob.glob(os.path.join(d, pat)):
                archive_subdir = os.path.join(
                    d, f"archived_{time.strftime('%Y%m%d_%H%M%S')}")
                try:
                    os.makedirs(archive_subdir, exist_ok=True)
                    shutil.move(f, os.path.join(archive_subdir, os.path.basename(f)))
                    archived_count += 1
                except PermissionError:
                    logger.warning("Cannot archive %s (permission denied) — skipping", f)
                except OSError:
                    pass

    if archived_count:
        logger.info("Archived %d metrics file(s) from previous run", archived_count)

    # Only truly delete from /tmp (transient, no need to retain)
    for pat in patterns:
        for f in glob.glob(os.path.join("/tmp", pat)):
            try:
                os.remove(f)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Stream Density Runner
# ---------------------------------------------------------------------------

class SADStreamDensity:
    """
    Iteratively increases the number of camera pipelines until end-to-end
    latency exceeds *target_latency_ms*.

    What gets scaled at each iteration:
      - RTSP camera streams  (lp-cams-N)  → real video feed
      - DLStreamer pipelines (lp-video)    → inference per camera
      - SceneScape scenes    (scene-import) → scene clones in SceneScape DB
      - swlp-service subscriptions         → MQTT topics for each scene

    What stays running untouched:
      - SceneScape core  (web, controller, broker, ntpserv, pgserver, vdms)
      - ovms-vlm          (model server — stateless)
      - behavioral-analysis  (receives per-request work — stateless)
      - seaweedfs          (object storage)
      - alert-service
      - mediaserver        (RTSP relay)
    """

    MEMORY_SAFETY_PERCENT = 90

    def __init__(
        self,
        app_dir: str,
        target_latency_ms: float,
        latency_metric: str,
        scene_increment: int,
        init_duration: int,
        stabilise_duration: int,
        results_dir: str,
        max_iterations: int,
        single_run: bool = False,
        single_run_scenes: int = 1,
        min_throughput_ratio: float = 0.5,
    ):
        self.app_dir = os.path.abspath(app_dir)
        self.target_latency_ms = target_latency_ms
        self.latency_metric = latency_metric
        self.scene_increment = scene_increment
        self.init_duration = init_duration
        self.stabilise_duration = stabilise_duration
        self.results_dir = os.path.abspath(results_dir)
        self.max_iterations = max_iterations
        self.single_run = single_run
        self.single_run_scenes = single_run_scenes
        self.min_throughput_ratio = min_throughput_ratio
        os.makedirs(self.results_dir, exist_ok=True)

    # ---- public API -------------------------------------------------------

    def run(self) -> StreamDensityResult:
        """Execute the stream-density loop."""
        self._print_header()
        result = StreamDensityResult(target_latency_ms=self.target_latency_ms)

        num_scenes = self.single_run_scenes if self.single_run else 1
        max_iter = 1 if self.single_run else self.max_iterations
        best: Optional[IterationResult] = None

        for iteration in range(1, max_iter + 1):
            print(f"\n{'='*70}")
            print(f"Iteration {iteration}: Testing {num_scenes} scene(s)")
            print(f"{'='*70}")

            # On the first iteration, restart BA and OVMS to get a clean
            # memory baseline (they may carry bloated state from prior runs).
            if iteration == 1:
                logger.info("Restarting behavioral-analysis and ovms-vlm for clean memory baseline …")
                _docker_compose(self.app_dir, "restart ovms-vlm")
                # Wait for OVMS model to load
                logger.info("Waiting 60s for OVMS model to load …")
                time.sleep(60)
                _docker_compose(self.app_dir, "restart behavioral-analysis")
                logger.info("Waiting 30s for BA to initialise …")
                time.sleep(30)

            if not self._memory_safe():
                logger.warning("Memory threshold exceeded – stopping.")
                break

            # Clean old metrics before each measurement
            _clean_metrics(self.results_dir)

            # Scale to desired scene count (real pipeline scaling)
            _scale_pipeline_services(self.app_dir, num_scenes, wait=self.init_duration)

            # Wait for pipeline to stabilise and collect data
            logger.info("Collecting data for %ds …", self.stabilise_duration)
            time.sleep(self.stabilise_duration)

            # Collect latency from docker logs (primary) + metrics files (secondary)
            stats = _collect_latency_from_docker_logs(self.app_dir, self.stabilise_duration, num_scenes)
            latency = _extract_latency_value(stats, self.latency_metric)

            # Throughput: compare actual samples to expected
            actual_samples = int(stats.get("e2e_latency_count", 0))
            # Baseline: ~12 samples per scene per 30s collection window
            if not hasattr(self, '_baseline_rate'):
                self._baseline_rate = max(actual_samples, 1)
            expected_samples = num_scenes * self._baseline_rate
            throughput_ratio = actual_samples / expected_samples if expected_samples > 0 else 1.0
            samples_per_scene = actual_samples / num_scenes if num_scenes > 0 else 0

            it_result = IterationResult(
                num_scenes=num_scenes,
                latency_ms=latency,
                latency_details=stats,
                memory_percent=psutil.virtual_memory().percent,
                cpu_percent=psutil.cpu_percent(interval=1),
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                actual_samples=actual_samples,
                expected_samples=expected_samples,
                throughput_ratio=throughput_ratio,
                samples_per_scene=samples_per_scene,
            )

            self._print_iteration(it_result)

            # Pass/fail: based on latency only
            latency_ok = latency > 0 and latency <= self.target_latency_ms

            if latency == 0:
                print("  ⚠ NO DATA – no latency metrics collected yet")
                if iteration > 1:
                    break
            elif latency_ok:
                it_result.passed = True
                best = it_result
                print(f"  ✓ PASSED  (latency={latency:.0f}ms, throughput={throughput_ratio:.0%})")
            else:
                it_result.passed = False
                print(f"  ✗ FAILED  (latency {latency:.0f}ms > {self.target_latency_ms:.0f}ms)")
                result.iterations.append(it_result)
                break

            result.iterations.append(it_result)
            num_scenes += self.scene_increment

        result.best_iteration = best
        result.max_scenes = best.num_scenes if best else 0
        result.met_target = best is not None

        self._export(result)
        self._print_summary(result)
        return result

    # ---- internal ---------------------------------------------------------

    def _memory_safe(self) -> bool:
        mem = psutil.virtual_memory()
        if mem.percent > self.MEMORY_SAFETY_PERCENT:
            logger.warning("Memory at %.1f%% (threshold %d%%)",
                           mem.percent, self.MEMORY_SAFETY_PERCENT)
            return False
        return True

    # ---- output -----------------------------------------------------------

    def _print_header(self) -> None:
        print("=" * 70)
        print("SAD Stream Density – Scene-Based Latency Scaling")
        print("=" * 70)
        print(f"  Target Latency:    {self.target_latency_ms:.0f}ms")
        print(f"  Latency Metric:    {self.latency_metric}")
        print(f"  Scene Increment:   +{self.scene_increment}")
        print(f"  Init Duration:     {self.init_duration}s")
        print(f"  Stabilise:         {self.stabilise_duration}s")
        print(f"  Results Dir:       {self.results_dir}")
        print(f"  Single-run Mode:   {self.single_run}")
        print("=" * 70)

    def _print_iteration(self, it: IterationResult) -> None:
        print(f"\n  Scenes:   {it.num_scenes}")
        print(f"  Latency:  {it.latency_ms:.0f}ms")
        print(f"  Throughput: {it.actual_samples}/{it.expected_samples} samples "
              f"({it.throughput_ratio:.0%}, {it.samples_per_scene:.1f}/scene)")
        print(f"  Memory:   {it.memory_percent:.1f}%")
        print(f"  CPU:      {it.cpu_percent:.1f}%")
        if it.latency_details:
            for k, v in it.latency_details.items():
                print(f"    {k}: {v:.2f}")

    def _print_summary(self, result: StreamDensityResult) -> None:
        print("\n" + "=" * 70)
        print("STREAM DENSITY RESULTS")
        print("=" * 70)
        print(f"  Target Latency:  {result.target_latency_ms:.0f}ms")
        print(f"  Max Scenes:      {result.max_scenes}")
        print(f"  Met Target:      {'Yes' if result.met_target else 'No'}")
        if result.best_iteration:
            print(f"  Best Latency:    {result.best_iteration.latency_ms:.0f}ms "
                  f"@ {result.best_iteration.num_scenes} scene(s)")
        print()
        print(f"{'Scenes':<10}{'Latency':<12}{'Throughput':<14}{'Mem %':<10}{'CPU %':<10}{'Status':<10}")
        print("-" * 66)
        for it in result.iterations:
            status = "✓ PASS" if it.passed else "✗ FAIL"
            tp = f"{it.throughput_ratio:.0%}"
            print(f"{it.num_scenes:<10}{it.latency_ms:<12.0f}{tp:<14}"
                  f"{it.memory_percent:<10.1f}{it.cpu_percent:<10.1f}{status}")
        print("=" * 70)

    def _export(self, result: StreamDensityResult) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # JSON
        json_path = os.path.join(self.results_dir, f"swlp_stream_density_{ts}.json")
        data = {
            "target_latency_ms": result.target_latency_ms,
            "max_scenes": result.max_scenes,
            "met_target": result.met_target,
            "iterations": [
                {
                    "num_scenes": it.num_scenes,
                    "latency_ms": round(it.latency_ms, 2),
                    "passed": it.passed,
                    "memory_percent": round(it.memory_percent, 1),
                    "cpu_percent": round(it.cpu_percent, 1),
                    "timestamp": it.timestamp,
                    "throughput_ratio": round(it.throughput_ratio, 3),
                    "actual_samples": it.actual_samples,
                    "expected_samples": it.expected_samples,
                    "samples_per_scene": round(it.samples_per_scene, 1),
                    "latency_details": {
                        k: round(v, 2) if isinstance(v, float) else v
                        for k, v in it.latency_details.items()
                    },
                }
                for it in result.iterations
            ],
        }
        if result.best_iteration:
            data["best_iteration"] = {
                "num_scenes": result.best_iteration.num_scenes,
                "latency_ms": round(result.best_iteration.latency_ms, 2),
            }
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nJSON results: {json_path}")

        # CSV
        csv_path = os.path.join(self.results_dir, f"swlp_stream_density_{ts}.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["scenes", "latency_ms", "throughput_ratio", "actual_samples",
                        "expected_samples", "samples_per_scene", "passed", "memory_pct", "cpu_pct"])
            for it in result.iterations:
                w.writerow([it.num_scenes, f"{it.latency_ms:.0f}",
                            f"{it.throughput_ratio:.3f}", it.actual_samples,
                            it.expected_samples, f"{it.samples_per_scene:.1f}",
                            it.passed, f"{it.memory_percent:.1f}",
                            f"{it.cpu_percent:.1f}"])
        print(f"CSV results:  {csv_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_run(args) -> None:
    app_dir = args.app_dir
    # Default results_dir to <app_dir>/results if not explicitly set
    results_dir = args.results_dir
    if results_dir == "./results":
        results_dir = os.path.join(app_dir, "results")
    tester = SADStreamDensity(
        app_dir=app_dir,
        target_latency_ms=args.target_latency_ms,
        latency_metric=args.latency_metric,
        scene_increment=args.scene_increment,
        init_duration=args.init_duration,
        stabilise_duration=args.stabilise_duration,
        results_dir=results_dir,
        max_iterations=args.max_iterations,
        single_run=args.single_run,
        single_run_scenes=args.scenes,
        min_throughput_ratio=args.min_throughput_ratio,
    )
    result = tester.run()
    sys.exit(0 if result.met_target else 1)


def cmd_generate(args) -> None:
    app_dir = args.app_dir
    num = args.scenes
    _set_stream_density(app_dir, num)
    _generate_cameras_override(app_dir, num)
    _reinit_env(app_dir)
    _generate_dlstreamer_config(app_dir, num)
    print(f"Generated overrides for {num} scene(s).  Run 'make demo' to start.")


def cmd_clean(args) -> None:
    app_dir = args.app_dir
    # Restore zone_config.json from backup
    bak = _zone_config_path(app_dir).with_suffix(".json.bak")
    if bak.exists():
        shutil.copy2(bak, _zone_config_path(app_dir))
        bak.unlink()
        logger.info("Restored zone_config.json from backup")
    else:
        _set_stream_density(app_dir, 1)
    # Remove cameras override
    _clean_cameras_override(app_dir)
    # Re-run init to render single-pipeline config
    _reinit_env(app_dir)
    # Restore single-pipeline DLStreamer config (reads rendered output)
    _generate_dlstreamer_config(app_dir, 1)
    print("Cleaned up – stream_density reset to 1.")


def cmd_down(args) -> None:
    app_dir = args.app_dir
    _docker_compose(app_dir, "down -t 30 --volumes --remove-orphans")
    cmd_clean(args)


def main() -> None:
    target_latency = _env_float("TARGET_LATENCY_MS", 30000)
    latency_metric = _env_str("LATENCY_METRIC", "avg")
    scene_increment = _env_int("SCENE_INCREMENT", 1)
    init_duration = _env_int("INIT_DURATION", 90)
    stabilise_duration = _env_int("STABILISE_DURATION", 30)
    results_dir = _env_str("RESULTS_DIR", "./results")
    max_iterations = _env_int("MAX_ITERATIONS", 50)

    parser = argparse.ArgumentParser(
        description="SAD Stream Density Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- run ---
    p_run = sub.add_parser("run", help="Run stream-density loop")
    p_run.add_argument("app_dir", help="Path to suspicious-activity-detection/")
    p_run.add_argument("--target_latency_ms", type=float, default=target_latency)
    p_run.add_argument("--latency_metric", choices=["avg", "max"], default=latency_metric)
    p_run.add_argument("--scene_increment", type=int, default=scene_increment)
    p_run.add_argument("--init_duration", type=int, default=init_duration)
    p_run.add_argument("--stabilise_duration", type=int, default=stabilise_duration)
    p_run.add_argument("--results_dir", default=results_dir)
    p_run.add_argument("--max_iterations", type=int, default=max_iterations)
    p_run.add_argument("--min_throughput_ratio", type=float,
                       default=float(os.environ.get("BENCHMARK_MIN_THROUGHPUT_RATIO", "0.5")),
                       help="Min ratio of actual/expected BA samples (0-1)")
    p_run.add_argument("--single_run", action="store_true",
                       help="Run once with --scenes scenes (benchmark mode)")
    p_run.add_argument("--scenes", type=int, default=1,
                       help="Number of scenes for single-run mode")
    p_run.set_defaults(func=cmd_run)

    # --- generate ---
    p_gen = sub.add_parser("generate", help="Generate overrides for N scenes")
    p_gen.add_argument("app_dir")
    p_gen.add_argument("--scenes", type=int, default=1)
    p_gen.set_defaults(func=cmd_generate)

    # --- clean ---
    p_clean = sub.add_parser("clean", help="Revert to single scene")
    p_clean.add_argument("app_dir")
    p_clean.set_defaults(func=cmd_clean)

    # --- down ---
    p_down = sub.add_parser("down", help="Stop all services and clean up")
    p_down.add_argument("app_dir")
    p_down.set_defaults(func=cmd_down)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
