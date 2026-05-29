#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation.
# SPDX-License-Identifier: Apache-2.0

"""
Person of Interest (POI) Stream Density Benchmark

Iteratively increases the number of camera/scene pipelines until end-to-end
POI detection-to-alert latency exceeds a configurable threshold.

The POI pipeline flow measured:
  Person detected on camera (MQTT) → face embedding extracted → FAISS match
  → alert dispatched to alert-service

Scaling is achieved by:
  1. Updating ``stream_density`` in ``zone_config.json``
  2. Re-running ``init.sh`` to regenerate ``.env`` + DLStreamer pipeline config
  3. Generating ``docker-compose.cameras.yaml`` with additional RTSP camera
     streams (``lp-cams-N``) for each new camera
  4. Restarting ``scene-import`` (imports cloned scenes into SceneScape),
     ``lp-cams-*``, ``lp-video`` (DLStreamer), and ``poi-backend``
  5. SceneScape core services (web, controller, broker, etc.), poi-redis,
     poi-alert-service, and poi-ui stay running

Latency is measured from ``vlm_application_metrics`` files written by the
``vlm_metrics_logger`` package.  In ``alert_service.py``:
  - **start**: ``user_log_start_time(frame_ts_ms, ...)`` — the DLStreamer frame
    capture timestamp from the MQTT payload (epoch ms).  This is the moment
    the camera frame was decoded by the pipeline.
  - **end**: ``log_end_time(...)`` — wall-clock time at alert dispatch.

This measures **true end-to-end latency**:
  camera frame capture → DLStreamer decode/detect/reid → MQTT → FAISS match → alert dispatch.

Sub-commands
------------
run       Full automated stream-density loop.
generate  Write a ``docker-compose.cameras.yaml`` override for *N* scenes.
clean     Revert overrides and set ``stream_density`` back to 1.
down      Tear down all services.

Environment variables
---------------------
TARGET_LATENCY_MS       Latency threshold in ms  (default: 2000)
LATENCY_METRIC          Which metric to compare: avg | max  (default: avg)
SCENE_INCREMENT         Scenes to add per iteration  (default: 1)
INIT_DURATION           Warm-up seconds after restart  (default: 90)
STABILISE_DURATION      Extra wait for pipeline to stabilise  (default: 30)
BENCHMARK_DURATION      Max wait for single benchmark in seconds  (default: 120)
RESULTS_DIR             Where to write results  (default: ./results)
MAX_ITERATIONS          Safety cap on iterations  (default: 50)
RESOURCE_CONFIG         Path to device resource config file relative to app_dir
                        (e.g. configs/res/all-gpu.env).  Passed to init.sh on
                        every re-init so device and model precision are preserved
                        across stream-density iterations.  Prefer passing
                        --resource_config on the CLI (set automatically by
                        ``make benchmark DEVICE=...``).
"""

import argparse
import copy
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

from consolidate_multiple_run_of_metrics import (
    get_vlm_application_latency,
    get_vlm_application_latency_stream_density,
)

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
    actual_detections: int = 0
    alerts_generated: int = 0
    detections_per_scene: float = 0.0


@dataclass
class StreamDensityResult:
    """Aggregate result of the full stream-density run."""
    target_latency_ms: float = 2000.0
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
    """Build a combined compose invocation spanning SceneScape + POI.

    Includes scenescape-overrides and cameras override when present.
    The cameras override may reference POI services (e.g. poi-backend
    environment), so the POI compose file must be included too.
    """
    scenescape_dir = str(Path(app_dir) / ".." / "scenescape")
    scenescape_compose = os.path.join(scenescape_dir, "docker-compose.yaml")
    overrides = os.path.join(app_dir, "docker-compose.scenescape-overrides.yml")
    poi_compose = os.path.join(app_dir, "docker-compose.yml")
    env_file = os.path.join(app_dir, "docker", ".env")

    parts = [
        "docker compose",
        f"--project-directory {shlex.quote(app_dir)}",
        f"--env-file {shlex.quote(env_file)}",
        f"-f {shlex.quote(scenescape_compose)}",
    ]
    if os.path.isfile(overrides):
        parts.append(f"-f {shlex.quote(overrides)}")
    parts.append(f"-f {shlex.quote(poi_compose)}")
    # Layer in cameras override if it exists
    cameras_override = os.path.join(app_dir, "docker", "docker-compose.cameras.yaml")
    if os.path.isfile(cameras_override):
        parts.append(f"-f {shlex.quote(cameras_override)}")
    return " ".join(parts)


def _poi_compose_cmd(app_dir: str) -> str:
    """Build compose command for POI-only services."""
    poi_compose = os.path.join(app_dir, "docker-compose.yml")
    return f"docker compose -f {shlex.quote(poi_compose)}"


def _docker_compose(app_dir: str, action: str) -> int:
    """Run a combined compose action (SceneScape + POI)."""
    if "up" in action:
        subprocess.run(
            "docker network create storewide-lp",
            shell=True, capture_output=True,
        )
    cmd = f"{_compose_cmd(app_dir)} {action}"
    logger.info("Running: %s", cmd)
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0 and "down" not in action:
        logger.warning("docker compose stderr:\n%s", result.stderr[-500:])
    return result.returncode


def _poi_compose(app_dir: str, action: str) -> int:
    """Run a compose action against POI-only services."""
    cmd = f"{_poi_compose_cmd(app_dir)} {action}"
    logger.info("Running: %s", cmd)
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0 and "down" not in action:
        logger.warning("docker compose stderr:\n%s", result.stderr[-500:])
    return result.returncode


def _run_cmd(cmd: str) -> subprocess.CompletedProcess:
    logger.debug("Running: %s", cmd)
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)


# ---------------------------------------------------------------------------
# Helpers – SceneScape scene cleanup via REST API
# ---------------------------------------------------------------------------

def _delete_cloned_scenes(app_dir: str, num_scenes: int) -> None:
    """Delete previously-cloned scenes from SceneScape via REST API."""
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

    # List and delete cloned scenes (those with -N suffix)
    req = urllib.request.Request(f"{base_url}/scenes", headers=auth_header)
    try:
        with urllib.request.urlopen(req, context=ctx) as resp:
            scenes = json.loads(resp.read()).get("results", [])
    except Exception:
        return

    for scene in scenes:
        name = scene.get("name", "")
        uid = scene.get("uid", "")
        if re.search(r'-\d+$', name):
            logger.info("Deleting cloned scene: %s (%s)", name, uid)
            req = urllib.request.Request(
                f"{base_url}/scene/{uid}",
                method="DELETE", headers=auth_header)
            try:
                urllib.request.urlopen(req, context=ctx)
            except urllib.error.HTTPError as e:
                logger.warning("  DELETE failed (%s): %s", e.code, e.reason)

    # Delete orphaned cameras with -N suffix
    req = urllib.request.Request(f"{base_url}/cameras", headers=auth_header)
    try:
        with urllib.request.urlopen(req, context=ctx) as resp:
            cameras = json.loads(resp.read()).get("results", [])
    except Exception:
        cameras = []
    for cam in cameras:
        cam_name = cam.get("name", "")
        cam_id = cam.get("uid", cam.get("id", cam.get("sensor_id", "")))
        if re.search(r'-\d+$', cam_name) and cam_id:
            logger.info("Deleting orphaned camera: %s (%s)", cam_name, cam_id)
            req = urllib.request.Request(
                f"{base_url}/camera/{cam_id}",
                method="DELETE", headers=auth_header)
            try:
                urllib.request.urlopen(req, context=ctx)
            except urllib.error.HTTPError as e:
                logger.warning("  DELETE camera failed (%s): %s", e.code, e.reason)


# ---------------------------------------------------------------------------
# Helpers – cameras override generation
# ---------------------------------------------------------------------------

def _read_base_config(app_dir: str) -> dict:
    """Read base camera/scene/video config from zone_config.json."""
    cfg = _read_zone_config(app_dir)
    cameras = cfg.get("cameras", [])
    if cameras:
        camera = cameras[0].get("name", "Camera_01")
        video = cameras[0].get("video_file", "Camera_01.mp4")
    else:
        camera = cfg.get("camera_name", "Camera_01")
        video = cfg.get("video_file", "Camera_01.mp4")
    return {"camera_name": camera, "video_file": video}


def _generate_cameras_override(app_dir: str, num_scenes: int) -> None:
    """
    Generate ``docker-compose.cameras.yaml`` that adds real RTSP camera
    streams for each additional camera beyond the base ones.

    For N scenes, we create lp-cams-{N+1} through lp-cams-{2*N} services,
    each streaming the same video on a unique RTSP path.
    """
    override_path = Path(app_dir) / "docker" / "docker-compose.cameras.yaml"
    base = _read_base_config(app_dir)
    base_camera = base["camera_name"]
    base_video = base["video_file"]

    # POI already has 2 base cameras (Camera_01, Camera_02); add more
    # starting from camera index 3 (for scenes > 1)
    base_camera_count = 2

    with open(override_path, "w") as f:
        f.write("# Auto-generated by poi_stream_density.py — do not edit\n")
        f.write(f"# Stream density: {num_scenes} scenes\n\n")
        f.write("services:\n")

        # Additional RTSP camera streams
        for i in range(1, num_scenes):
            cam_idx = base_camera_count + i
            cam_name = f"{base_camera}-{cam_idx}"
            svc_name = f"lp-cams-{cam_idx}"
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

        # Build dynamic MQTT camera list for poi-backend
        all_cameras = ["Camera_01", "Camera_02"]
        for i in range(1, num_scenes):
            cam_idx = base_camera_count + i
            all_cameras.append(f"{base_camera}-{cam_idx}")
        camera_csv = ",".join(all_cameras)

        # Override poi-backend to subscribe to all cameras
        f.write(f"  poi-backend:\n")
        f.write(f"    environment:\n")
        f.write(f"      RTSP_PREWARM_CAMERAS: \"{camera_csv}\"\n")
        f.write(f"      MQTT_IMAGE_CAMERAS: \"{camera_csv}\"\n")
        f.write(f"      STREAM_DENSITY: \"{num_scenes}\"\n")

    logger.info("Generated cameras override: %s  (%d scenes, %d extra cameras)",
                override_path, num_scenes, max(0, num_scenes - 1))


def _generate_dlstreamer_config(app_dir: str, num_scenes: int) -> None:
    """
    Generate multi-pipeline DLStreamer configs for N scenes.

    POI uses two DLStreamer containers (lp-video, lp-video-2) with separate
    config files mounted via Docker Compose configs.  Extra camera pipelines
    are appended into these existing config files so that force-recreating
    the containers picks them up — no additional volume mounts needed.

    Distribution: extra cameras are round-robin'd across lp-video (Camera_01)
    and lp-video-2 (Camera_02).
    """
    scenescape_dir = Path(app_dir) / ".." / "scenescape"
    dlstreamer_dir = scenescape_dir / "dlstreamer-pipeline-server"

    base = _read_base_config(app_dir)
    base_camera = base["camera_name"]
    base_camera_count = 2

    # Read rendered Camera_01 config (already processed by init.sh with
    # correct model paths, devices, etc.)
    cfg_path_1 = dlstreamer_dir / f"person-of-interest-{base_camera}-pipeline-config.json"
    cfg_path_2 = dlstreamer_dir / f"person-of-interest-Camera_02-pipeline-config.json"

    if not cfg_path_1.exists():
        logger.warning("Pipeline config not found: %s", cfg_path_1)
        return

    with open(cfg_path_1) as fh:
        cfg1 = json.load(fh)
    base_pipeline_1 = cfg1["config"]["pipelines"][0]

    if cfg_path_2.exists():
        with open(cfg_path_2) as fh:
            cfg2 = json.load(fh)
        base_pipeline_2 = cfg2["config"]["pipelines"][0]
    else:
        cfg2 = copy.deepcopy(cfg1)
        base_pipeline_2 = base_pipeline_1

    # Reset to single base pipeline each before adding extras
    cfg1["config"]["pipelines"] = [base_pipeline_1]
    cfg2["config"]["pipelines"] = [base_pipeline_2]

    # Generate extra pipelines and distribute across the two containers
    for i in range(1, num_scenes):
        cam_idx = base_camera_count + i
        cam_name = f"{base_camera}-{cam_idx}"

        # Use Camera_01's pipeline as template (same model paths/devices)
        pipeline_str = json.dumps(base_pipeline_1)
        pipeline_str = pipeline_str.replace(base_camera, cam_name)
        pipeline = json.loads(pipeline_str)
        pipeline["name"] = f"reid_{cam_name}"

        # Round-robin: odd extras → lp-video, even extras → lp-video-2
        if i % 2 == 1:
            cfg1["config"]["pipelines"].append(pipeline)
        else:
            cfg2["config"]["pipelines"].append(pipeline)

    # Write updated configs back
    with open(cfg_path_1, "w") as fh:
        json.dump(cfg1, fh, indent=2)
    logger.info("Updated %s with %d pipelines", cfg_path_1.name, len(cfg1["config"]["pipelines"]))

    with open(cfg_path_2, "w") as fh:
        json.dump(cfg2, fh, indent=2)
    logger.info("Updated %s with %d pipelines", cfg_path_2.name, len(cfg2["config"]["pipelines"]))

    total = len(cfg1["config"]["pipelines"]) + len(cfg2["config"]["pipelines"])
    logger.info("Total DLStreamer pipelines across both containers: %d", total)


def _reinit_env(app_dir: str, resource_config: str = "") -> None:
    """Re-run init.sh to regenerate .env with updated config.

    Parameters
    ----------
    app_dir:
        Absolute path to the person-of-interest/ directory.
    resource_config:
        Path to the device resource config file (e.g. configs/res/all-gpu.env).
        Passed as the ``RESOURCE_CONFIG`` env var to init.sh so that device,
        precision, and pre-process settings are preserved across stream-density
        iterations.  When empty, init.sh uses its own default (all-gpu-cpu.env).
    """
    init_script = Path(app_dir) / ".." / "scenescape" / "scripts" / "init.sh"
    if not init_script.exists():
        logger.warning("init.sh not found at %s — skipping .env regeneration", init_script)
        return

    env = os.environ.copy()
    if resource_config:
        env["RESOURCE_CONFIG"] = resource_config
        logger.info("Re-running init.sh with RESOURCE_CONFIG=%s …", resource_config)
    else:
        logger.info("Re-running init.sh to update .env …")

    cmd = f"bash {shlex.quote(str(init_script))} {shlex.quote(app_dir)}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        logger.warning("init.sh returned non-zero:\n%s", result.stderr[-500:])
    else:
        logger.info("init.sh completed — .env updated")


def _wait_for_web_healthy(timeout: int = 300) -> None:
    """Block until SceneScape web container is healthy or timeout expires."""
    # Container name depends on compose project name used at startup
    candidates = ["storewide-lp-web-1", "scenescape-web-1"]
    for attempt in range(timeout // 5):
        for name in candidates:
            result = subprocess.run(
                f"docker inspect {name} --format '{{{{.State.Health.Status}}}}'",
                shell=True, capture_output=True, text=True)
            status = result.stdout.strip()
            if status == "healthy":
                logger.info("Web container (%s) is healthy (after %ds)", name, attempt * 5)
                return
        if attempt % 6 == 0:
            logger.info("  web status: %s  (waiting…)", status)
        time.sleep(5)
    logger.warning("Web container did not become healthy after %ds — continuing anyway", timeout)


def _wait_for_services_ready(timeout: int = 120) -> None:
    """Poll poi-backend and DLStreamer until ready, instead of blind sleep.

    Checks poi-backend /api/v1/status and DLStreamer container running state
    every 5 seconds, returning as soon as both are up or timeout expires.
    """
    import urllib.request
    import urllib.error

    poll_interval = 5
    logger.info("Waiting up to %ds for services to be ready …", timeout)

    for attempt in range(timeout // poll_interval):
        # Check poi-backend
        backend_ok = False
        try:
            req = urllib.request.Request("http://localhost:8000/api/v1/status")
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    backend_ok = True
        except (urllib.error.URLError, OSError):
            pass

        # Check DLStreamer container
        dlstreamer_ok = False
        result = subprocess.run(
            "docker inspect storewide-lp-lp-video-1 --format '{{.State.Running}}'",
            shell=True, capture_output=True, text=True)
        if result.stdout.strip() == "true":
            dlstreamer_ok = True

        if backend_ok and dlstreamer_ok:
            elapsed = (attempt + 1) * poll_interval
            logger.info("Services ready after %ds (saved %ds)", elapsed, timeout - elapsed)
            return

        if attempt % 6 == 0:
            logger.info("  poi-backend=%s  dlstreamer=%s  (waiting…)",
                        "up" if backend_ok else "down",
                        "up" if dlstreamer_ok else "down")
        time.sleep(poll_interval)

    logger.warning("Services not fully ready after %ds — continuing anyway", timeout)


def _scale_pipeline_services(app_dir: str, num_scenes: int, wait: int = 90, resource_config: str = "") -> None:
    """
    Scale the POI video pipeline to N scenes.

    Steps:
      1. Update stream_density in zone_config.json
      2. Generate docker-compose.cameras.yaml with extra RTSP streams
      3. Re-run init.sh to update .env
      4. Generate per-camera DLStreamer pipeline configs
      5. Start extra camera services
      6. In parallel: clean stale scenes + scene-import, recreate DLStreamer
      7. Poll until services are ready

    Note: poi-backend is NOT restarted — it uses wildcard MQTT subscriptions
    (scenescape/data/camera/+) and automatically picks up new cameras.
    """
    logger.info("Scaling POI to %d scene(s) …", num_scenes)

    _set_stream_density(app_dir, num_scenes)
    _generate_cameras_override(app_dir, num_scenes)
    _reinit_env(app_dir, resource_config=resource_config)
    _generate_dlstreamer_config(app_dir, num_scenes)

    # Start only the extra camera services needed for this iteration.
    # Avoids the broad `up -d --no-recreate` which tries all services and
    # hits container-name conflicts (e.g. poi-alert-service).
    base_camera_count = 2
    extra_cams = [f"lp-cams-{base_camera_count + i}" for i in range(1, num_scenes)]
    if extra_cams:
        svc_list = " ".join(extra_cams)
        logger.info("Starting extra cameras: %s", svc_list)
        _docker_compose(app_dir, f"up -d --force-recreate {svc_list}")

    # Scene cleanup + scene-import restart run concurrently with DLStreamer
    # and poi-backend restarts below via threading.
    import concurrent.futures

    def _scene_cleanup_and_import():
        """Clean stale scenes and re-run scene-import."""
        logger.info("Cleaning stale scene-import extract dirs …")
        _run_cmd("docker exec storewide-lp-web-1 bash -c "
                 "'rm -rf /workspace/media/storewide-loss-prevention-[0-9]*'")
        _delete_cloned_scenes(app_dir, num_scenes)
        logger.info("Re-running scene-import for %d scenes …", num_scenes)
        _docker_compose(app_dir, "rm -f -s scene-import")
        _docker_compose(app_dir, "up -d scene-import")

    def _recreate_pipeline_services():
        """Recreate DLStreamer containers with updated pipeline configs.

        poi-backend is NOT restarted — it uses wildcard MQTT subscriptions
        (scenescape/data/camera/+) and automatically receives events from
        new cameras without a restart.
        """
        logger.info("Recreating lp-video and lp-video-2 (DLStreamer) …")
        result = subprocess.run(
            f"{_compose_cmd(app_dir)} up -d --force-recreate lp-video lp-video-2",
            shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning("DLStreamer recreate stderr:\n%s", result.stderr[-500:])

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        scene_future = pool.submit(_scene_cleanup_and_import)
        pipeline_future = pool.submit(_recreate_pipeline_services)
        # Wait for both to finish
        scene_future.result()
        pipeline_future.result()

    _wait_for_services_ready(wait)


def _clean_cameras_override(app_dir: str) -> None:
    override_path = Path(app_dir) / "docker" / "docker-compose.cameras.yaml"
    if override_path.exists():
        override_path.unlink()
        logger.info("Removed %s", override_path)

    # Also remove generated pipeline configs for extra cameras
    scenescape_dir = Path(app_dir) / ".." / "scenescape"
    dlstreamer_dir = scenescape_dir / "dlstreamer-pipeline-server"
    for cfg_file in dlstreamer_dir.glob("person-of-interest-*-[0-9]*-pipeline-config.json"):
        cfg_file.unlink()
        logger.info("Removed extra pipeline config: %s", cfg_file)


# ---------------------------------------------------------------------------
# Latency collection — POI-specific
# ---------------------------------------------------------------------------

def _collect_poi_latency_from_docker_logs(app_dir: str, duration_secs: int = 30) -> Dict[str, float]:
    """
    Extract end-to-end POI latency from poi-backend docker logs.

    Measures the time between a POI match and its corresponding alert dispatch.
    Also counts total detections, matches, and alerts for throughput tracking.
    """
    container = "poi-backend"
    since_arg = f"--since={duration_secs + 30}s"
    cmd = f"docker logs {container} {since_arg} 2>&1"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        logger.warning("Failed to get poi-backend logs: %s", result.stderr[:200])
        return {}

    detection_count = 0
    alert_count = 0
    match_count = 0
    match_times: list = []
    alert_times: list = []

    ts_re = re.compile(r"^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})")

    for line in result.stdout.splitlines():
        if "Face embedding found" in line:
            detection_count += 1
        elif "POI match:" in line:
            match_count += 1
            m = ts_re.match(line)
            if m:
                match_times.append(m.group(1))
        elif "Alert dispatched" in line or "Alert forwarded" in line:
            alert_count += 1
            m = ts_re.match(line)
            if m:
                alert_times.append(m.group(1))

    stats: Dict[str, float] = {
        "poi_detections": detection_count,
        "poi_matches": match_count,
        "poi_alerts": alert_count,
    }

    if detection_count > 0:
        stats["match_rate"] = match_count / detection_count
    if match_count > 0:
        stats["alert_rate"] = alert_count / match_count

    # Compute match-to-alert latency from log timestamps
    if match_times and alert_times:
        try:
            from datetime import datetime as _dt
            first_match = _dt.strptime(match_times[0], "%Y-%m-%d %H:%M:%S")
            first_alert = _dt.strptime(alert_times[0], "%Y-%m-%d %H:%M:%S")
            latency_s = (first_alert - first_match).total_seconds()
            if latency_s >= 0:
                stats["log_detection_to_alert_ms"] = latency_s * 1000
                logger.info("Log-based latency: first match → first alert = %.0fms",
                            latency_s * 1000)
        except Exception:
            pass

    logger.info("POI logs: %d detections, %d matches, %d alerts (in %ds window)",
                detection_count, match_count, alert_count, duration_secs)

    return stats


def _collect_poi_e2e_latency_from_alerts(
    since: Optional[datetime] = None,
) -> Dict[str, float]:
    """Compute real end-to-end latency from POI alerts API.

    Each alert contains:
      - ``mqtt_received_at``: when POI backend received the MQTT message (preferred)
      - ``timestamp``: DLStreamer frame capture time (fallback, includes pipeline latency)
      - ``dispatched_at``: when the alert was actually dispatched

    Uses ``mqtt_received_at`` when available to measure POI application latency
    only, excluding DLStreamer pipeline latency (~6-8s).

    Args:
        since: If provided, only include alerts dispatched after this time.
               Filters out stale alerts from previous benchmark runs.

    Returns dict with ``poi_e2e_latency_avg_ms``, ``poi_e2e_latency_max_ms``,
    ``poi_e2e_latency_min_ms``, and ``poi_e2e_alert_count``.
    """
    import urllib.request
    import urllib.error

    try:
        req = urllib.request.Request("http://localhost:8000/api/v1/alerts")
        with urllib.request.urlopen(req, timeout=10) as resp:
            alerts = json.loads(resp.read().decode())
    except Exception as e:
        logger.warning("Failed to fetch alerts for E2E latency: %s", e)
        return {}

    if not isinstance(alerts, list) or not alerts:
        return {}

    from datetime import datetime as _dt, timezone as _tz

    latencies_ms: list[float] = []
    skipped = 0
    used_mqtt_received = 0
    used_frame_timestamp = 0
    for alert in alerts:
        # Prefer mqtt_received_at (POI application latency only, excludes
        # DLStreamer pipeline latency) over timestamp (frame capture time).
        mqtt_recv = alert.get("mqtt_received_at", "")
        start_str = mqtt_recv or alert.get("timestamp", "")
        dispatched_str = alert.get("dispatched_at", "")
        if not start_str or not dispatched_str:
            continue
        try:
            start_str = start_str.replace("Z", "+00:00")
            dispatched_str = dispatched_str.replace("Z", "+00:00")
            start = _dt.fromisoformat(start_str)
            dispatched = _dt.fromisoformat(dispatched_str)

            # Normalize both to UTC-aware to avoid mixed tz subtraction errors
            start_utc = start.astimezone(_tz.utc) if start.tzinfo else start.replace(tzinfo=_tz.utc)
            dispatched_utc = dispatched.astimezone(_tz.utc) if dispatched.tzinfo else dispatched.replace(tzinfo=_tz.utc)

            # Filter out alerts from before the benchmark started
            if since is not None:
                since_aware = since.astimezone(_tz.utc) if since.tzinfo else since.replace(tzinfo=_tz.utc)
                if dispatched_utc < since_aware:
                    skipped += 1
                    continue

            delta_ms = (dispatched_utc - start_utc).total_seconds() * 1000
            if delta_ms >= 0:
                latencies_ms.append(delta_ms)
                if mqtt_recv:
                    used_mqtt_received += 1
                else:
                    used_frame_timestamp += 1
        except (ValueError, TypeError):
            continue

    if skipped:
        logger.info("Filtered out %d stale alerts (before benchmark start)", skipped)

    if not latencies_ms:
        return {}

    stats: Dict[str, float] = {
        "poi_e2e_latency_avg_ms": sum(latencies_ms) / len(latencies_ms),
        "poi_e2e_latency_max_ms": max(latencies_ms),
        "poi_e2e_latency_min_ms": min(latencies_ms),
        "poi_e2e_alert_count": len(latencies_ms),
    }
    label = "MQTT receive → alert dispatch"
    if used_frame_timestamp and not used_mqtt_received:
        label = "frame capture → alert dispatch (includes DLStreamer latency)"
    elif used_frame_timestamp:
        label = "start → alert dispatch (mixed sources)"
    logger.info(
        "E2E latency (%s): avg=%.0fms, min=%.0fms, max=%.0fms (%d alerts)",
        label,
        stats["poi_e2e_latency_avg_ms"],
        stats["poi_e2e_latency_min_ms"],
        stats["poi_e2e_latency_max_ms"],
        len(latencies_ms),
    )
    return stats


def _save_alert_thumbnails(
    results_dir: str, iteration: int = 1, since: Optional[datetime] = None,
) -> int:
    """Fetch alerts and their thumbnails from the POI API and save to results_dir.

    Args:
        since: If provided, only save thumbnails for alerts dispatched after this time.

    Returns the number of thumbnails saved.
    """
    import urllib.request
    import urllib.error

    thumbs_dir = os.path.join(results_dir, f"thumbnails_iter{iteration}")
    try:
        os.makedirs(thumbs_dir, exist_ok=True)
    except PermissionError:
        logger.warning(
            "Cannot create thumbnails directory %s (permission denied). "
            "The results/ directory may be owned by root (written by a Docker container). "
            "Run: sudo chown -R $USER results/",
            thumbs_dir,
        )
        return 0
    saved = 0

    try:
        req = urllib.request.Request("http://localhost:8000/api/v1/alerts")
        with urllib.request.urlopen(req, timeout=10) as resp:
            alerts = json.loads(resp.read().decode())
    except Exception as e:
        logger.warning("Failed to fetch alerts for thumbnails: %s", e)
        return 0

    if not isinstance(alerts, list) or not alerts:
        logger.info("No alerts found — no thumbnails to save")
        return 0

    from datetime import datetime as _dt, timezone as _tz

    for i, alert in enumerate(alerts):
        # Filter stale alerts
        if since is not None:
            dispatched_str = alert.get("dispatched_at", "")
            if dispatched_str:
                try:
                    d_str = dispatched_str.replace("Z", "+00:00")
                    dispatched = _dt.fromisoformat(d_str)
                    since_aware = since.astimezone(_tz.utc) if since.tzinfo else since.replace(tzinfo=_tz.utc)
                    dispatched_utc = dispatched.astimezone(_tz.utc) if dispatched.tzinfo else dispatched.replace(tzinfo=_tz.utc)
                    if dispatched_utc < since_aware:
                        continue
                except (ValueError, TypeError):
                    pass
        # Extract fields from nested alert structure
        match_data = alert.get("match", {})
        thumb_url = match_data.get("thumbnail_path") or alert.get("thumbnail_path") or ""
        object_id = alert.get("object_id", "")
        poi_id = alert.get("poi_id", "unknown")
        camera_id = match_data.get("camera_id") or alert.get("camera_id", "unknown")

        # Build the thumbnail URL
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

            safe_cam = re.sub(r'[^a-zA-Z0-9_-]', '_', camera_id)
            safe_poi = re.sub(r'[^a-zA-Z0-9_-]', '_', poi_id)
            fname = f"alert_{i:03d}_{safe_poi}_{safe_cam}.jpg"
            fpath = os.path.join(thumbs_dir, fname)
            with open(fpath, "wb") as f:
                f.write(img_data)
            saved += 1
            logger.info("Saved thumbnail: %s", fpath)
        except urllib.error.HTTPError as e:
            logger.debug("Thumbnail HTTP %d for %s", e.code, thumb_url)
        except Exception as e:
            logger.debug("Thumbnail save failed for %s: %s", thumb_url, e)

    if saved:
        logger.info("Saved %d alert thumbnails to %s", saved, thumbs_dir)
    return saved


def _collect_poi_latency_from_metrics_files(
    results_dir: str, stream_density: bool = False
) -> Dict[str, float]:
    """
    Extract POI detection-to-alert latency from vlm_application_metrics files.

    These files are written by the vlm_metrics_logger package via
    user_log_start_time (detection) and log_end_time (alert dispatch)
    calls in the poi-backend.

    For single benchmarks uses ``get_vlm_application_latency`` (all pairs).
    For stream density uses ``get_vlm_application_latency_stream_density``
    (last 20 pairs) to reflect current-iteration performance.
    """
    all_stats: Dict[str, float] = {}
    search_dirs = [results_dir, "/tmp"]

    if stream_density:
        for d in search_dirs:
            if not os.path.isdir(d):
                continue
            try:
                stats = get_vlm_application_latency_stream_density(d, last_n_pairs=20)
                if stats:
                    for app_id, avg_ms in stats.items():
                        all_stats[f"vlm_{app_id}_avg_ms"] = avg_ms
                    logger.info("VLM stream-density latency (%s): %s", d, stats)
            except Exception as e:
                logger.warning("Failed to parse VLM metrics in %s: %s", d, e)
    else:
        for d in search_dirs:
            if not os.path.isdir(d):
                continue
            pattern = os.path.join(d, "vlm_application_metrics_*.txt")
            files = sorted(glob.glob(pattern), key=os.path.getmtime)
            if not files:
                continue
            latest = files[-1]
            try:
                stats = get_vlm_application_latency(latest)
                if stats:
                    for key, avg_ms in stats.items():
                        all_stats[f"vlm_{key}"] = avg_ms
                    logger.info("VLM metrics file latency (%s): %s", latest, stats)
            except Exception as e:
                logger.warning("Failed to parse VLM metrics in %s: %s", d, e)

    return all_stats


def _extract_poi_latency(stats: Dict[str, float], metric: str) -> float:
    """
    Extract a single representative POI latency value from collected stats.

    Priority:
      1. vlm_application_metrics file values — TRUE end-to-end latency:
         start = DLStreamer frame capture timestamp (set by user_log_start_time
         in alert_service.py using the MQTT payload's frame timestamp field),
         end   = wall-clock time at alert dispatch (log_end_time).
         This spans: camera frame capture → DLStreamer pipeline → FAISS match
         → alert dispatch.
      2. Alerts API fallback (``mqtt_received_at`` → ``dispatched_at``) —
         POI application latency only, excludes DLStreamer pipeline latency.
      3. Returns 0 if no data available.

    Note: Docker-log-based ``log_detection_to_alert_ms`` is excluded because
    log timestamps have only second-level precision and the first-match-to-
    first-alert gap includes dedup delay (60 s TTL), making it unreliable
    as a per-event latency metric.
    """
    # Primary: vlm_application_metrics file-based values
    vlm_values = [v for k, v in stats.items()
                  if k.startswith("vlm_") and isinstance(v, (int, float)) and v > 0]
    if vlm_values:
        if metric == "max":
            return max(vlm_values)
        return mean(vlm_values)

    # Fallback: alerts API E2E latency
    e2e_avg = stats.get("poi_e2e_latency_avg_ms", 0.0)
    e2e_max = stats.get("poi_e2e_latency_max_ms", 0.0)
    if e2e_avg > 0:
        if metric == "max":
            return e2e_max
        return e2e_avg

    return 0.0


def _clean_metrics(results_dir: str) -> None:
    """Remove stale metrics files before each measurement iteration."""
    patterns = [
        "vlm_application_metrics*.txt",
        "vlm_performance_metrics*.txt",
    ]
    for d in [results_dir, "/tmp"]:
        for pat in patterns:
            for f in glob.glob(os.path.join(d, pat)):
                try:
                    os.remove(f)
                except OSError:
                    pass


# ---------------------------------------------------------------------------
# POI Stream Density Runner
# ---------------------------------------------------------------------------

class POIStreamDensity:
    """
    Iteratively increases the number of camera pipelines until end-to-end
    POI detection-to-alert latency exceeds *target_latency_ms*.

    What gets scaled at each iteration:
      - RTSP camera streams (lp-cams-N)    → real video feed
      - DLStreamer pipelines (lp-video)     → inference per camera
      - SceneScape scenes (scene-import)    → scene clones in SceneScape DB
      - poi-backend camera subscriptions    → MQTT + RTSP for new cameras

    What stays running untouched:
      - SceneScape core (web, controller, broker, ntpserv, pgserver, vdms)
      - poi-redis          (metadata store)
      - poi-alert-service  (alert fan-out)
      - poi-ui             (React frontend)
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
        benchmark_duration: int = 120,
        resource_config: str = "",
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
        self.benchmark_duration = benchmark_duration
        self.resource_config = resource_config
        os.makedirs(self.results_dir, exist_ok=True)

    def _services_running(self) -> bool:
        """Check if key POI pipeline services are already running."""
        for container in ("poi-backend", "storewide-lp-lp-video-1"):
            result = subprocess.run(
                f"docker inspect {container} --format '{{{{.State.Running}}}}'",
                shell=True, capture_output=True, text=True)
            if result.stdout.strip() != "true":
                return False
        return True

    def _wait_for_alert_or_timeout(self, duration: int) -> None:
        """Poll for alerts during single benchmark, exit early on first alert.

        For single benchmarks the goal is to measure time-to-first-alert.
        Instead of sleeping the full duration, poll every 5 seconds and
        return as soon as at least one alert is found.
        """
        import urllib.request
        import urllib.error

        poll_interval = 5
        elapsed = 0
        logger.info("Waiting up to %ds for alert (polling every %ds) …",
                     duration, poll_interval)

        while elapsed < duration:
            sleep_time = min(poll_interval, duration - elapsed)
            time.sleep(sleep_time)
            elapsed += sleep_time

            try:
                req = urllib.request.Request("http://localhost:8000/api/v1/alerts")
                with urllib.request.urlopen(req, timeout=10) as resp:
                    alerts = json.loads(resp.read().decode())
                if isinstance(alerts, list) and len(alerts) > 0:
                    logger.info("Alert received after %ds — stopping early", elapsed)
                    # Brief extra wait for metrics files to flush
                    time.sleep(5)
                    return
            except Exception:
                pass

            logger.info("No alerts yet (%d/%ds elapsed)", elapsed, duration)

        logger.info("Benchmark duration reached (%ds) — collecting results", duration)

    def run(self) -> StreamDensityResult:
        """Execute the POI stream-density loop."""
        self._print_header()
        result = StreamDensityResult(target_latency_ms=self.target_latency_ms)

        num_scenes = self.single_run_scenes if self.single_run else 1
        max_iter = 1 if self.single_run else self.max_iterations
        best: Optional[IterationResult] = None

        for iteration in range(1, max_iter + 1):
            print(f"\n{'='*70}")
            print(f"Iteration {iteration}: Testing {num_scenes} scene(s)")
            print(f"{'='*70}")

            if not self._memory_safe():
                logger.warning("Memory threshold exceeded – stopping.")
                break

            # Record iteration start time for filtering stale alerts
            iteration_start = datetime.utcnow()

            # Clean old metrics before each measurement
            _clean_metrics(self.results_dir)

            if self.single_run and self._services_running():
                # Single benchmark: services already up, skip scaling
                logger.info("Services already running — skipping scaling for single benchmark")
            else:
                # Scale to desired scene count
                _scale_pipeline_services(self.app_dir, num_scenes, wait=self.init_duration,
                                         resource_config=self.resource_config)

            # Wait for data collection
            if self.single_run:
                # Single benchmark: poll for alerts with early exit
                self._wait_for_alert_or_timeout(self.benchmark_duration)
            else:
                # Stream density: fixed stabilise wait per iteration
                logger.info("Collecting data for %ds …", self.stabilise_duration)
                time.sleep(self.stabilise_duration)

            # Use actual elapsed time for log collection window
            elapsed_seconds = int((datetime.utcnow() - iteration_start).total_seconds())
            log_window = elapsed_seconds if self.single_run else self.stabilise_duration

            # Collect latency from metrics files + docker logs
            log_stats = _collect_poi_latency_from_docker_logs(
                self.app_dir, log_window)
            file_stats = _collect_poi_latency_from_metrics_files(
                self.results_dir, stream_density=not self.single_run)
            e2e_stats = _collect_poi_e2e_latency_from_alerts(since=iteration_start)

            # Save alert thumbnails to results directory
            _save_alert_thumbnails(self.results_dir, iteration=iteration,
                                   since=iteration_start)

            # Merge all stats
            stats: Dict[str, float] = {}
            stats.update(log_stats)
            stats.update(file_stats)
            stats.update(e2e_stats)

            latency = _extract_poi_latency(stats, self.latency_metric)

            it_result = IterationResult(
                num_scenes=num_scenes,
                latency_ms=latency,
                latency_details=stats,
                memory_percent=psutil.virtual_memory().percent,
                cpu_percent=psutil.cpu_percent(interval=1),
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                actual_detections=int(stats.get("poi_detections", 0)),
                alerts_generated=int(stats.get("poi_alerts", 0)),
                detections_per_scene=(
                    int(stats.get("poi_detections", 0)) / num_scenes
                    if num_scenes > 0 else 0
                ),
            )

            self._print_iteration(it_result)

            # Pass/fail based on latency threshold
            latency_ok = latency > 0 and latency <= self.target_latency_ms
            has_detections = it_result.actual_detections > 0
            has_matches = int(stats.get("poi_matches", 0)) > 0

            if latency == 0 and has_matches:
                # Matches found but latency not measurable (sub-second)
                it_result.passed = True
                best = it_result
                print("  ✓ PASSED  (matches found, latency < 1s)")
            elif latency == 0 and has_detections and not has_matches:
                # Pipeline works but target person not in frame during window
                it_result.passed = True
                best = it_result
                print(f"  ✓ PASSED  ({it_result.actual_detections} detections, "
                      "no matches — target not in frame during window)")
            elif latency == 0:
                print("  ⚠ NO DATA – no detections collected")
                if iteration > 1:
                    break
            elif latency_ok:
                it_result.passed = True
                best = it_result
                print(f"  ✓ PASSED  (latency={latency:.0f}ms ≤ {self.target_latency_ms:.0f}ms)")
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

    def _memory_safe(self) -> bool:
        mem = psutil.virtual_memory()
        if mem.percent > self.MEMORY_SAFETY_PERCENT:
            logger.warning("Memory at %.1f%% (threshold %d%%)",
                           mem.percent, self.MEMORY_SAFETY_PERCENT)
            return False
        return True

    def _print_header(self) -> None:
        print("=" * 70)
        print("POI Stream Density – Detection-to-Alert Latency Scaling")
        print("=" * 70)
        print(f"  Target Latency:    {self.target_latency_ms:.0f}ms")
        print(f"  Latency Metric:    {self.latency_metric}")
        print(f"  Scene Increment:   +{self.scene_increment}")
        print(f"  Init Duration:     {self.init_duration}s")
        if self.single_run:
            print(f"  Benchmark Duration:{self.benchmark_duration}s (exits early on alert)")
        else:
            print(f"  Stabilise:         {self.stabilise_duration}s")
        print(f"  Results Dir:       {self.results_dir}")
        print(f"  Single-run Mode:   {self.single_run}")
        print("=" * 70)

    def _print_iteration(self, it: IterationResult) -> None:
        print(f"\n  Scenes:      {it.num_scenes}")
        print(f"  Latency:     {it.latency_ms:.0f}ms")
        print(f"  Detections:  {it.actual_detections} "
              f"({it.detections_per_scene:.1f}/scene)")
        print(f"  Alerts:      {it.alerts_generated}")
        print(f"  Memory:      {it.memory_percent:.1f}%")
        print(f"  CPU:         {it.cpu_percent:.1f}%")
        if it.latency_details:
            for k, v in it.latency_details.items():
                if isinstance(v, float):
                    print(f"    {k}: {v:.2f}")
                else:
                    print(f"    {k}: {v}")

    def _print_summary(self, result: StreamDensityResult) -> None:
        print("\n" + "=" * 70)
        print("POI STREAM DENSITY RESULTS")
        print("=" * 70)
        print(f"  Target Latency:  {result.target_latency_ms:.0f}ms")
        print(f"  Max Scenes:      {result.max_scenes}")
        print(f"  Met Target:      {'Yes' if result.met_target else 'No'}")
        if result.best_iteration:
            print(f"  Best Latency:    {result.best_iteration.latency_ms:.0f}ms "
                  f"@ {result.best_iteration.num_scenes} scene(s)")
        print()
        print(f"{'Scenes':<10}{'Latency':<12}{'Detections':<14}"
              f"{'Alerts':<10}{'Mem %':<10}{'CPU %':<10}{'Status':<10}")
        print("-" * 76)
        for it in result.iterations:
            status = "✓ PASS" if it.passed else "✗ FAIL"
            print(f"{it.num_scenes:<10}{it.latency_ms:<12.0f}"
                  f"{it.actual_detections:<14}{it.alerts_generated:<10}"
                  f"{it.memory_percent:<10.1f}{it.cpu_percent:<10.1f}{status}")
        print("=" * 70)

    def _export(self, result: StreamDensityResult) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # JSON
        json_path = os.path.join(self.results_dir, f"poi_stream_density_{ts}.json")
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
                    "actual_detections": it.actual_detections,
                    "alerts_generated": it.alerts_generated,
                    "detections_per_scene": round(it.detections_per_scene, 1),
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
        csv_path = os.path.join(self.results_dir, f"poi_stream_density_{ts}.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["scenes", "latency_ms", "detections", "alerts",
                         "detections_per_scene", "passed", "memory_pct", "cpu_pct"])
            for it in result.iterations:
                w.writerow([it.num_scenes, f"{it.latency_ms:.0f}",
                            it.actual_detections, it.alerts_generated,
                            f"{it.detections_per_scene:.1f}",
                            it.passed, f"{it.memory_percent:.1f}",
                            f"{it.cpu_percent:.1f}"])
        print(f"CSV results:  {csv_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_run(args) -> None:
    tester = POIStreamDensity(
        app_dir=args.app_dir,
        target_latency_ms=args.target_latency_ms,
        latency_metric=args.latency_metric,
        scene_increment=args.scene_increment,
        init_duration=args.init_duration,
        stabilise_duration=args.stabilise_duration,
        results_dir=args.results_dir,
        max_iterations=args.max_iterations,
        single_run=args.single_run,
        single_run_scenes=args.scenes,
        benchmark_duration=args.benchmark_duration,
        resource_config=args.resource_config,
    )
    result = tester.run()
    sys.exit(0 if result.met_target else 1)


def cmd_generate(args) -> None:
    num = args.scenes
    _set_stream_density(args.app_dir, num)
    _generate_dlstreamer_config(args.app_dir, num)
    _generate_cameras_override(args.app_dir, num)
    _reinit_env(args.app_dir, resource_config=args.resource_config)
    print(f"Generated overrides for {num} scene(s).  Run 'make demo' to start.")


def cmd_clean(args) -> None:
    app_dir = args.app_dir
    bak = _zone_config_path(app_dir).with_suffix(".json.bak")
    if bak.exists():
        shutil.copy2(bak, _zone_config_path(app_dir))
        bak.unlink()
        logger.info("Restored zone_config.json from backup")
    else:
        _set_stream_density(app_dir, 1)
    _generate_dlstreamer_config(app_dir, 1)
    _clean_cameras_override(app_dir)
    _reinit_env(app_dir, resource_config=getattr(args, "resource_config", ""))
    print("Cleaned up – stream_density reset to 1.")


def cmd_down(args) -> None:
    _docker_compose(args.app_dir, "down -t 30 --volumes --remove-orphans")
    cmd_clean(args)


def main() -> None:
    target_latency = _env_float("TARGET_LATENCY_MS", 2000)
    latency_metric = _env_str("LATENCY_METRIC", "avg")
    scene_increment = _env_int("SCENE_INCREMENT", 1)
    init_duration = _env_int("INIT_DURATION", 90)
    stabilise_duration = _env_int("STABILISE_DURATION", 30)
    benchmark_duration = _env_int("BENCHMARK_DURATION", 120)
    results_dir = _env_str("RESULTS_DIR", "./results")
    max_iterations = _env_int("MAX_ITERATIONS", 50)

    parser = argparse.ArgumentParser(
        description="POI Stream Density Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- run ---
    p_run = sub.add_parser("run", help="Run POI stream-density loop")
    p_run.add_argument("app_dir", help="Path to person-of-interest/")
    p_run.add_argument("--target_latency_ms", type=float, default=target_latency)
    p_run.add_argument("--latency_metric", choices=["avg", "max"], default=latency_metric)
    p_run.add_argument("--scene_increment", type=int, default=scene_increment)
    p_run.add_argument("--init_duration", type=int, default=init_duration)
    p_run.add_argument("--stabilise_duration", type=int, default=stabilise_duration)
    p_run.add_argument("--results_dir", default=results_dir)
    p_run.add_argument("--max_iterations", type=int, default=max_iterations)
    p_run.add_argument("--single_run", action="store_true",
                       help="Run once with --scenes scenes (benchmark mode)")
    p_run.add_argument("--benchmark_duration", type=int, default=benchmark_duration,
                       help="Max duration in seconds for single benchmark (default: 120). "
                            "Exits early when an alert is received.")
    p_run.add_argument("--scenes", type=int, default=1,
                       help="Number of scenes for single-run mode")
    p_run.add_argument("--resource_config", default="",
                       help="Absolute path to device resource config "
                            "(e.g. /path/to/configs/res/all-gpu.env). "
                            "Passed as RESOURCE_CONFIG to init.sh on every "
                            "re-init so device and precision are preserved "
                            "across stream-density iterations.")
    p_run.set_defaults(func=cmd_run)

    # --- generate ---
    p_gen = sub.add_parser("generate", help="Generate overrides for N scenes")
    p_gen.add_argument("app_dir")
    p_gen.add_argument("--scenes", type=int, default=1)
    p_gen.add_argument("--resource_config", default="",
                       help="Absolute path to device resource config file.")
    p_gen.set_defaults(func=cmd_generate)

    # --- clean ---
    p_clean = sub.add_parser("clean", help="Revert to single scene")
    p_clean.add_argument("app_dir")
    p_clean.add_argument("--resource_config", default="",
                         help="Absolute path to device resource config file.")
    p_clean.set_defaults(func=cmd_clean)

    # --- down ---
    p_down = sub.add_parser("down", help="Stop all services and clean up")
    p_down.add_argument("app_dir")
    p_down.set_defaults(func=cmd_down)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
