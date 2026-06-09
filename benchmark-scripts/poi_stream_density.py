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
INIT_DURATION           Seconds per alert-wait window per retry  (default: 45)
MAX_ALERT_WAIT          Max total seconds to wait for alert from new camera
                        (retries in INIT_DURATION windows until received or
                        timeout).  Default: 180 — covers ~3 video cycles (~55s
                        each).  Increase for CPU mode where inference is slower.
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
import csv
import glob
import calendar
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
    # Camera under test for this iteration (empty for baseline)
    new_camera: str = ""


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

    Includes scenescape-overrides, NPU overlay (when NPU device is active),
    and cameras override when present.
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
    # Mirror Makefile logic: include NPU overlay when RESOURCE_CONFIG contains "npu".
    # Without this, force-recreating lp-video strips /dev/accel, causing
    # "[NPU_VCL] Unrecognized device ID! 0x0x0" on every stream-density iteration.
    npu_overlay = os.path.join(app_dir, "docker-compose.npu-overrides.yml")
    if os.path.isfile(npu_overlay) and _is_npu_device(app_dir):
        parts.append(f"-f {shlex.quote(npu_overlay)}")
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

# ---------------------------------------------------------------------------
# SceneScape REST API helpers
# ---------------------------------------------------------------------------

def _scenescape_get_client(app_dir: str):
    """Authenticate with SceneScape and return (base_url, ssl_ctx, token).

    Reads SUPASS from docker/.env and authenticates as admin.
    Returns (None, None, None) on failure — callers must handle gracefully.
    """
    import ssl
    import urllib.request

    env_file = os.path.join(app_dir, "docker", ".env")
    supass = ""
    if os.path.isfile(env_file):
        for line in open(env_file):
            if line.startswith("SUPASS="):
                supass = line.strip().split("=", 1)[1]
                break
    if not supass:
        logger.warning("Could not read SUPASS from docker/.env — SceneScape API unavailable")
        return None, None, None

    # Read base_url from zone_config.json; default to https://localhost
    try:
        zone_cfg = _read_zone_config(app_dir)
        base_url = zone_cfg.get("scenescape_api", {}).get("base_url", "https://localhost").rstrip("/")
        base_url = base_url + "/api/v1"
    except Exception:
        base_url = "https://localhost/api/v1"

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    auth_data = json.dumps({"username": "admin", "password": supass}).encode()
    req = urllib.request.Request(
        f"{base_url}/auth", data=auth_data,
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=15) as resp:
            token = json.loads(resp.read()).get("token", "")
    except Exception as e:
        logger.warning("SceneScape authentication failed: %s", e)
        return None, None, None

    if not token:
        logger.warning("SceneScape auth returned empty token")
        return None, None, None

    return base_url, ctx, token


def _clone_scene_zip(base_zip_path: str, scene_name: str, camera_name: str) -> bytes:
    """Clone *base_zip_path* with a new scene name and camera name.

    Returns the cloned ZIP as raw bytes (suitable for multipart upload).
    Replicates the logic in scenescape/webserver/stream_density.py so that the
    benchmark can call the SceneScape import-scene API directly from the host
    without spinning up a Docker sidecar container.

    The import-scene endpoint matches the background image by checking whether
    the scene name appears in the image filename, so we rename the image to
    ``<scene_name>.<ext>`` in the output ZIP.
    """
    import io
    import uuid
    import zipfile

    with zipfile.ZipFile(base_zip_path, "r") as zf:
        json_name = None
        base_json = None
        other_files: dict = {}
        for name in zf.namelist():
            data = zf.read(name)
            if name.endswith(".json"):
                json_name = name
                base_json = json.loads(data)
            else:
                other_files[name] = data

    if not json_name or base_json is None:
        raise ValueError(f"No scene JSON found in {base_zip_path}")

    # Deep-copy and patch the scene JSON
    scene_data = json.loads(json.dumps(base_json))
    new_scene_uid = str(uuid.uuid4())
    scene_data["uid"] = new_scene_uid
    scene_data["name"] = scene_name

    for cam in scene_data.get("cameras", []):
        cam["uid"] = camera_name
        cam["name"] = camera_name
        cam["scene"] = new_scene_uid

    for region in scene_data.get("regions", []):
        region["uid"] = str(uuid.uuid4())
        region["scene"] = new_scene_uid

    # SceneScape's import-scene matches the resource file by checking whether
    # the scene name is a substring of the filename.  Rename the image so it
    # matches: "<scene_name><ext>" (e.g. "conference room-2.jpg").
    safe_name = scene_name.replace("/", "_")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf_out:
        zf_out.writestr(f"{safe_name}.json", json.dumps(scene_data))
        for orig_name, data in other_files.items():
            import os as _os
            ext = _os.path.splitext(orig_name)[1]
            zf_out.writestr(f"{safe_name}{ext}", data)
    return buf.getvalue()


def _scenescape_import_scene(
    app_dir: str,
    scene_name: str,
    camera_name: str,
) -> tuple:
    """Register a new scene + camera in SceneScape via POST /api/v1/import-scene/.

    Clones the base scene ZIP in-memory with the new scene/camera names and
    uploads it using the same multipart endpoint that scene-import.sh uses.
    This replaces the Docker sidecar approach while being ~10× faster (~200ms).

    Returns (scene_uid, camera_name) on success, (None, None) on failure.
    Failure is non-fatal — callers fall back to the scene-import sidecar.
    """
    import io
    import urllib.error
    import urllib.request

    base_url, ctx, token = _scenescape_get_client(app_dir)
    if not token:
        return None, None

    # Locate base scene ZIP
    try:
        zone_cfg = _read_zone_config(app_dir)
        scene_zip_name = zone_cfg.get("scene_zip", "conference-room.zip")
    except Exception:
        scene_zip_name = "conference-room.zip"

    zip_path = str(Path(app_dir) / ".." / "scenescape" / "webserver" / scene_zip_name)
    if not Path(zip_path).exists():
        logger.warning("Base scene ZIP not found at %s — falling back to scene-import", zip_path)
        return None, None

    # Clone ZIP in-memory
    try:
        zip_bytes = _clone_scene_zip(zip_path, scene_name, camera_name)
    except Exception as e:
        logger.warning("Failed to clone scene ZIP: %s", e)
        return None, None

    # Build multipart/form-data body — SceneScape expects field name "zipFile"
    boundary = "----BenchmarkFormBoundary"
    filename = f"{scene_name.replace(' ', '-')}.zip"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="zipFile"; filename="{filename}"\r\n'
        f"Content-Type: application/zip\r\n\r\n"
    ).encode() + zip_bytes + f"\r\n--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        f"{base_url}/import-scene/",
        data=body,
        headers={
            "Authorization": f"Token {token}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=60) as resp:
            resp_data = json.loads(resp.read())
            # import-scene response: {"scene": null=success | {errors}, "cameras": ...}
            # scene: null means scene was created successfully (null = no errors).
            # The scene UID lives inside the camera objects in the cameras list.
            scene_errors = resp_data.get("scene")
            if scene_errors is not None:
                # scene field is non-null → contains error details
                logger.warning("SceneScape import-scene scene error: %s", scene_errors)
                return None, None

            # Scene created — extract UID from cameras data
            scene_uid = ""
            cameras = resp_data.get("cameras") or []
            for cam_entry in cameras:
                # cam_entry may be a list (one per camera) of [errors_or_None, cam_obj]
                entries = cam_entry if isinstance(cam_entry, list) else [cam_entry]
                for entry in entries:
                    if isinstance(entry, dict) and entry.get("scene"):
                        scene_uid = entry["scene"]
                        break
                if scene_uid:
                    break

            if not scene_uid:
                # Fall back to listing scenes by name
                try:
                    list_req = urllib.request.Request(
                        f"{base_url}/scenes", headers={"Authorization": f"Token {token}"})
                    with urllib.request.urlopen(list_req, context=ctx, timeout=10) as r:
                        for s in json.loads(r.read()).get("results", []):
                            if s.get("name") == scene_name:
                                scene_uid = s["uid"]
                                break
                except Exception:
                    pass

            logger.info("SceneScape scene+camera imported: %s / %s (uid=%s)",
                        scene_name, camera_name, scene_uid)
            return scene_uid or scene_name, camera_name
    except urllib.error.HTTPError as e:
        logger.warning("SceneScape import-scene → HTTP %s: %s", e.code, e.read().decode()[:200])
        return None, None
    except Exception as e:
        logger.warning("SceneScape import-scene → %s", e)
        return None, None


def _delete_cloned_scenes(app_dir: str, num_scenes: int) -> None:
    """Delete previously-cloned scenes from SceneScape via REST API."""
    import urllib.request
    import urllib.error

    base_url, ctx, token = _scenescape_get_client(app_dir)
    if not token:
        logger.warning("Could not authenticate — skipping scene cleanup")
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
    streams and DLStreamer pipeline containers for each additional camera
    beyond the base two (Camera_01, Camera_02).

    For N scenes, we create:
      - lp-cams-{cam_idx}:   ffmpeg RTSP server for the new camera stream
      - lp-video-{cam_idx}:  DLStreamer container running the inference pipeline
      - lp-config-{cam_idx}: Docker config pointing to the generated pipeline JSON

    Architecture note: each camera must have its own DLStreamer (lp-video-N)
    container because each container mounts a single config.json (via Docker
    configs) that hardcodes the RTSP source and camera name for that pipeline.
    Sharing one container across cameras is not supported by this service design.
    """
    override_path = Path(app_dir) / "docker" / "docker-compose.cameras.yaml"
    base = _read_base_config(app_dir)
    base_camera = base["camera_name"]
    base_video = base["video_file"]

    scenescape_dir = (Path(app_dir) / ".." / "scenescape").resolve()
    dlstreamer_dir = scenescape_dir / "dlstreamer-pipeline-server"

    # POI already has 2 base cameras (Camera_01, Camera_02); add more
    # starting from camera index 3 (for scenes > 1)
    base_camera_count = 2

    # When NPU is selected, extra DLStreamer containers need /dev/accel access.
    # Without it OpenVINO reports "[NPU_VCL] Unrecognized device ID! 0x0x0"
    # and the inference pipeline fails to initialise.
    is_npu = _is_npu_device(app_dir)

    with open(override_path, "w") as f:
        f.write("# Auto-generated by poi_stream_density.py — do not edit\n")
        f.write(f"# Stream density: {num_scenes} scenes\n\n")
        f.write("services:\n")

        # Additional RTSP camera streams + DLStreamer instances
        for i in range(1, num_scenes):
            cam_idx = base_camera_count + i
            cam_name = f"{base_camera}-{cam_idx}"
            cams_svc = f"lp-cams-{cam_idx}"
            video_svc = f"lp-video-{cam_idx}"
            config_name = f"lp-config-{cam_idx}"

            # RTSP camera stream (ffmpeg)
            f.write(f"  {cams_svc}:\n")
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

            # DLStreamer pipeline server for this camera.
            # Mirrors the lp-video-2 service from scenescape/docker-compose.yaml
            # but uses lp-config-{cam_idx} to load this camera's pipeline config.
            f.write(f"  {video_svc}:\n")
            f.write(f"    image: docker.io/intel/dlstreamer-pipeline-server:${{DLSTREAMER_VERSION:-2026.1.0-20260331-weekly-ubuntu24}}\n")
            f.write(f"    networks:\n")
            f.write(f"      storewide-lp:\n")
            f.write(f"    tty: true\n")
            f.write(f"    entrypoint: [\"./run.sh\"]\n")
            f.write(f"    devices:\n")
            f.write(f"      - \"/dev/dri:/dev/dri\"\n")
            if is_npu:
                # Intel NPU (MTL/WCL) is exposed as /dev/accel (major 261).
                # Without this mapping the VCL compiler reads device ID 0x0x0
                # and fails: "[NPU_VCL] Unrecognized device ID! 0x0x0"
                f.write(f"      - \"/dev/accel:/dev/accel\"\n")
            f.write(f"    group_add:\n")
            f.write(f"      - \"109\"\n")
            f.write(f"      - \"110\"\n")
            f.write(f"      - \"992\"\n")
            f.write(f"    device_cgroup_rules:\n")
            f.write(f"      - \"c 189:* rmw\"\n")
            f.write(f"      - \"c 209:* rmw\"\n")
            f.write(f"      - \"a 189:* rwm\"\n")
            if is_npu:
                f.write(f"      - \"c 261:* rmw\"  # Intel NPU accel devices\n")
            f.write(f"    depends_on:\n")
            f.write(f"      broker:\n")
            f.write(f"        condition: service_started\n")
            f.write(f"      ntpserv:\n")
            f.write(f"        condition: service_started\n")
            f.write(f"      {cams_svc}:\n")
            f.write(f"        condition: service_started\n")
            f.write(f"    healthcheck:\n")
            f.write(f'      test: ["CMD", "curl", "-I", "-s", "http://localhost:8080/pipelines"]\n')
            f.write(f"      interval: 10s\n")
            f.write(f"      timeout: 5s\n")
            f.write(f"      retries: 5\n")
            f.write(f"      start_period: 10s\n")
            f.write(f"    environment:\n")
            f.write(f"      - RUN_MODE=EVA\n")
            f.write(f"      - GENICAM=Balluff\n")
            f.write(f"      - GST_DEBUG=1,gencamsrc:2\n")
            f.write(f"      - ADD_UTCTIME_TO_METADATA=true\n")
            f.write(f"      - APPEND_PIPELINE_NAME_TO_PUBLISHER_TOPIC=false\n")
            f.write(f"      - MQTT_HOST=broker.scenescape.intel.com\n")
            f.write(f"      - MQTT_PORT=1883\n")
            f.write(f"      - REST_SERVER_PORT=8080\n")
            f.write(f"      - HTTPS_PROXY=${{HTTPS_PROXY}}\n")
            f.write(f"      - https_proxy=${{https_proxy}}\n")
            f.write(f"      - HTTP_PROXY=${{HTTP_PROXY}}\n")
            f.write(f"      - http_proxy=${{http_proxy}}\n")
            f.write(f"      - NO_PROXY=mediaserver,${{NO_PROXY}}\n")
            f.write(f"      - no_proxy=mediaserver,${{no_proxy}}\n")
            f.write(f"    configs:\n")
            f.write(f"      - source: {config_name}\n")
            f.write(f"        target: /home/pipeline-server/config.json\n")
            f.write(f"    volumes:\n")
            f.write(f"      - ../scenescape/dlstreamer-pipeline-server/user_scripts:/home/pipeline-server/user_scripts\n")
            f.write(f"      - vol-dlstreamer-pipeline-root-{cam_idx}:/var/cache/pipeline_root:uid=1999,gid=1999\n")
            f.write(f"      - vol-sample-data:/home/pipeline-server/videos\n")
            f.write(f"      - vol-models:/home/pipeline-server/models\n")
            f.write(f"    secrets:\n")
            f.write(f"      - source: root-cert\n")
            f.write(f"        target: certs/scenescape-ca.pem\n")
            f.write(f"    restart: always\n")
            f.write(f"    pids_limit: 1000\n")
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
        f.write(f"\n")

        # Docker configs for each new DLStreamer pipeline
        if num_scenes > 1:
            f.write(f"configs:\n")
            for i in range(1, num_scenes):
                cam_idx = base_camera_count + i
                cam_name = f"{base_camera}-{cam_idx}"
                config_name = f"lp-config-{cam_idx}"
                env_var = f"PIPELINE_CONFIG_{cam_idx}"
                default_path = dlstreamer_dir / f"person-of-interest-{cam_name}-pipeline-config.json"
                f.write(f"  {config_name}:\n")
                f.write(f"    file: ${{{env_var}:-{default_path}}}\n")
            f.write(f"\n")

            # Named volumes for each new DLStreamer container's pipeline cache
            f.write(f"volumes:\n")
            for i in range(1, num_scenes):
                cam_idx = base_camera_count + i
                f.write(f"  vol-dlstreamer-pipeline-root-{cam_idx}:\n")

    logger.info("Generated cameras override: %s  (%d scenes, %d extra cameras+DLStreamer instances)",
                override_path, num_scenes, max(0, num_scenes - 1))


def _generate_dlstreamer_config(app_dir: str, num_scenes: int) -> None:
    """
    Generate a multi-pipeline DLStreamer config for N scenes.

    Reads the base pipeline config template and replicates it for each
    additional camera, updating the camera name in each pipeline.
    Also writes PIPELINE_CONFIG_{cam_idx} to docker/.env so that the
    lp-config-{cam_idx} Docker config defined in docker-compose.cameras.yaml
    can resolve the correct pipeline JSON path.
    """
    scenescape_dir = Path(app_dir) / ".." / "scenescape"
    dlstreamer_dir = scenescape_dir / "dlstreamer-pipeline-server"
    env_file = os.path.join(app_dir, "docker", ".env")

    base = _read_base_config(app_dir)
    base_camera = base["camera_name"]
    base_camera_count = 2

    # Read existing Camera_01 pipeline config as template
    template_path = dlstreamer_dir / f"person-of-interest-{base_camera}-pipeline-config.json"
    if not template_path.exists():
        logger.warning("Pipeline template not found: %s", template_path)
        return

    with open(template_path) as fh:
        template_cfg = json.load(fh)

    # Generate config for each additional camera
    for i in range(1, num_scenes):
        cam_idx = base_camera_count + i
        cam_name = f"{base_camera}-{cam_idx}"
        output_path = dlstreamer_dir / f"person-of-interest-{cam_name}-pipeline-config.json"

        # Deep-copy and substitute camera name
        cfg_str = json.dumps(template_cfg)
        cfg_str = cfg_str.replace(base_camera, cam_name)
        cfg = json.loads(cfg_str)

        # Update pipeline name
        if "config" in cfg and "pipelines" in cfg["config"]:
            for pipeline in cfg["config"]["pipelines"]:
                pipeline["name"] = f"reid_{cam_name}"

        with open(output_path, "w") as fh:
            json.dump(cfg, fh, indent=2)
        logger.info("Generated pipeline config: %s", output_path)

        # Write PIPELINE_CONFIG_{cam_idx} so docker-compose.cameras.yaml can
        # resolve the lp-config-{cam_idx} Docker config file path.
        env_key = f"PIPELINE_CONFIG_{cam_idx}"
        _write_env_var(env_file, env_key, str(output_path.resolve()))

    logger.info("Generated DLStreamer configs for %d total cameras", base_camera_count + num_scenes - 1)


def _is_npu_device(app_dir: str) -> bool:
    """Return True when the active resource config selects NPU.

    Reads RESOURCE_CONFIG from docker/.env to mirror the Makefile logic:
    ``$(if $(findstring npu,$(DEVICE)),-f $(NPU_OVERLAY),)``.
    """
    env_file = os.path.join(app_dir, "docker", ".env")
    if os.path.isfile(env_file):
        with open(env_file) as fh:
            for line in fh:
                if line.startswith("RESOURCE_CONFIG=") and "npu" in line.lower():
                    return True
    return False


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
        # init.sh constructs RESOURCE_CONFIG_PATH as "${APP_DIR}/${RESOURCE_CONFIG}",
        # so it expects a path relative to app_dir, not an absolute path.
        # Convert absolute → relative so the path resolves correctly inside init.sh.
        try:
            rel_rc = str(Path(resource_config).relative_to(Path(app_dir)))
        except ValueError:
            rel_rc = resource_config  # already relative or outside app_dir
        env["RESOURCE_CONFIG"] = rel_rc
        logger.info("Re-running init.sh with RESOURCE_CONFIG=%s …", rel_rc)
    else:
        logger.info("Re-running init.sh to update .env …")

    cmd = f"bash {shlex.quote(str(init_script))} {shlex.quote(app_dir)}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        # init.sh writes errors to stdout (no >&2), so log both streams
        output = (result.stderr + result.stdout)[-500:]
        logger.warning("init.sh returned non-zero:\n%s", output)
    else:
        logger.info("init.sh completed — .env updated")


def _wait_for_first_detection(timeout: int = 60, poll_interval: int = 3,
                              camera_filter: Optional[str] = None) -> bool:
    """Poll poi-backend logs until the first face detection arrives or timeout.

    Args:
        camera_filter: If set, only detections from this specific camera count
                       as "warm". This prevents Camera_01/02 detections from
                       masking the fact that Camera_01-3's pipeline hasn't
                       started yet (false-positive "pipeline is warm").

    Returns True if a detection was seen within *timeout* seconds, False otherwise.
    Uses --since to only read logs produced after this function is called, avoiding
    false positives from stale log lines from previous iterations.
    """
    camera_label = f" from camera={camera_filter}" if camera_filter else ""
    since = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    deadline = time.time() + timeout
    attempt = 0
    while time.time() < deadline:
        result = subprocess.run(
            f"docker logs --since {since} poi-backend 2>&1",
            shell=True, capture_output=True, text=True)
        output = result.stdout + result.stderr
        is_detection = (
            "POI match" in output or
            "face embedding" in output.lower() or
            "detections" in output.lower() or
            "poi_detections" in output
        )
        if is_detection:
            if camera_filter is None or f"camera={camera_filter}" in output:
                elapsed = int(timeout - (deadline - time.time()))
                logger.info("First detection seen%s after ~%ds — pipeline is warm",
                            camera_label, elapsed)
                return True
        if attempt % 4 == 0:
            remaining = int(deadline - time.time())
            logger.info("  Waiting for first detection%s … (%ds remaining)",
                        camera_label, remaining)
        attempt += 1
        time.sleep(poll_interval)
    logger.warning("No detection%s seen within %ds — proceeding anyway",
                   camera_label, timeout)
    return False


# ---------------------------------------------------------------------------
# Camera-specific helpers — new camera detection for stream-density
# ---------------------------------------------------------------------------

def _get_new_camera_name(app_dir: str, num_scenes: int) -> Optional[str]:
    """Return the camera name newly added in this stream-density iteration.

    Returns ``None`` for *num_scenes* == 1 (baseline iteration: Camera_01 and
    Camera_02 are always present, no camera is "newly added").

    For *num_scenes* > 1 each increment adds exactly one extra RTSP stream
    whose index is ``base_camera_count + (num_scenes - 1)``.

    Example (base_camera = "Camera_01"):
      num_scenes=1 → None         (baseline — Camera_01 / Camera_02)
      num_scenes=2 → "Camera_01-3"
      num_scenes=3 → "Camera_01-4"
      num_scenes=N → f"Camera_01-{N+1}"
    """
    if num_scenes <= 1:
        return None
    base = _read_base_config(app_dir)
    base_camera = base["camera_name"]       # e.g. "Camera_01"
    base_camera_count = 2                   # POI always starts with 2 base cameras
    cam_idx = base_camera_count + (num_scenes - 1)
    return f"{base_camera}-{cam_idx}"


def _wait_for_alert_from_camera(
    camera_id: str,
    duration: int,
    since: Optional[datetime] = None,
    poll_interval: int = 5,
) -> Optional[dict]:
    """Poll /api/v1/alerts until a fresh alert from *camera_id* is dispatched.

    "Fresh" means ``dispatched_at`` is strictly after *since* (defaults to the
    moment this function is called).

    Returns the matched alert dict so the caller can compute latency directly
    without a second API round-trip.  Returns ``None`` if *duration* expires
    without a matching alert.

    Up to 100 alerts are fetched per poll to avoid missing the new camera's
    alert when many alerts have already accumulated.
    """
    import urllib.request
    from datetime import datetime as _dt, timezone as _tz

    elapsed = 0
    since_ts = since or datetime.utcnow()
    since_aware = (
        since_ts.astimezone(_tz.utc)
        if since_ts.tzinfo
        else since_ts.replace(tzinfo=_tz.utc)
    )
    logger.info(
        "Waiting up to %ds for alert from camera=%s (since=%s) …",
        duration, camera_id, since_ts.strftime("%H:%M:%S"),
    )

    while elapsed < duration:
        sleep_time = min(poll_interval, duration - elapsed)
        time.sleep(sleep_time)
        elapsed += sleep_time

        try:
            req = urllib.request.Request(
                "http://localhost:8000/api/v1/alerts?limit=100"
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                alerts = json.loads(resp.read().decode())
            if not isinstance(alerts, list):
                continue

            for alert in alerts:
                dispatched_str = alert.get("dispatched_at") or alert.get("timestamp", "")
                if not dispatched_str:
                    continue
                try:
                    d_str = dispatched_str.replace("Z", "+00:00")
                    dispatched_utc = _dt.fromisoformat(d_str)
                    if not dispatched_utc.tzinfo:
                        dispatched_utc = dispatched_utc.replace(tzinfo=_tz.utc)
                    else:
                        dispatched_utc = dispatched_utc.astimezone(_tz.utc)
                    if dispatched_utc < since_aware:
                        continue
                except (ValueError, TypeError):
                    continue

                alert_camera = (
                    alert.get("match", {}).get("camera_id")
                    or alert.get("camera_id", "")
                )
                if alert_camera == camera_id:
                    logger.info(
                        "Alert from camera=%s received after %ds — pipeline active",
                        camera_id, elapsed,
                    )
                    time.sleep(5)   # brief flush so metrics files catch up
                    return alert
        except Exception:
            pass

        if elapsed % 30 < poll_interval + 1:
            logger.info(
                "Waiting for alert from camera=%s … (%d/%ds)",
                camera_id, elapsed, duration,
            )

    logger.warning(
        "No alert from camera=%s within %ds — continuing anyway",
        camera_id, duration,
    )
    return None


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


def _wait_for_scene_import_completion(timeout: int = 180) -> None:
    """Wait until the scene-import one-shot container exits.

    scene-import must complete before lp-video (DLStreamer) restarts so that
    SceneScape has the newly cloned camera registered in its database before
    DLStreamer starts publishing data for it.

    Without this wait, SceneScape's controller has no record of Camera_01-3
    and silently drops all DLStreamer output for that camera — poi-backend
    never receives face embeddings → no FAISS match → no alert.
    """
    logger.info("Waiting for scene-import to complete (timeout=%ds) …", timeout)
    deadline = time.time() + timeout
    while time.time() < deadline:
        running = subprocess.run(
            "docker ps -q --filter 'name=scene-import' --filter 'status=running'",
            shell=True, capture_output=True, text=True,
        ).stdout.strip()
        if running:
            elapsed = int(timeout - (deadline - time.time()))
            if elapsed % 30 < 6:
                logger.info("  scene-import still running … (%ds elapsed)", elapsed)
            time.sleep(5)
            continue

        exited = subprocess.run(
            "docker ps -aq --filter 'name=scene-import' --filter 'status=exited'",
            shell=True, capture_output=True, text=True,
        ).stdout.strip()
        if exited:
            # Get exit code to surface errors
            code_result = subprocess.run(
                f"docker inspect {exited.splitlines()[0]} "
                f"--format '{{{{.State.ExitCode}}}}'",
                shell=True, capture_output=True, text=True,
            )
            exit_code = code_result.stdout.strip()
            if exit_code == "0":
                logger.info("scene-import completed successfully")
            else:
                logger.warning("scene-import exited with code %s — "
                               "camera registration may be incomplete", exit_code)
            return
        # Container not found yet — give it a moment to start
        time.sleep(3)

    logger.warning("scene-import did not complete within %ds — "
                   "SceneScape may not have the new camera registered yet", timeout)


def _wait_for_camera_rtsp_ready(camera_name: str, timeout: int = 60) -> bool:
    """Poll until the new camera's RTSP stream is being served by MediaMTX.

    DLStreamer (lp-video) connects to RTSP at startup.  If lp-cams-N is still
    initialising when lp-video is force-recreated, DLStreamer silently fails to
    open the RTSP source and the Camera_01-N pipeline never starts — no
    embeddings, no alerts.

    Strategy:
      1. Query MediaMTX path-list API via docker exec (port 9997 is NOT mapped
         to host — only 8554/8889 are — so localhost:9997 always fails).
      2. Fallback: check that the compose-named container is running.
         Container name format: storewide-lp-lp-cams-{N}-1
    """
    logger.info("Waiting for RTSP stream camera=%s to be ready (timeout=%ds) …",
                camera_name, timeout)
    # Camera_01-3 → suffix "3" → service "lp-cams-3" → container "storewide-lp-lp-cams-3-1"
    svc_idx = camera_name.split("-")[-1] if "-" in camera_name else "3"
    container_name = f"storewide-lp-lp-cams-{svc_idx}-1"

    deadline = time.time() + timeout
    while time.time() < deadline:
        # Primary: MediaMTX path-list API via docker exec (avoids host-port issue)
        try:
            result = subprocess.run(
                "docker exec storewide-lp-mediaserver-1 "
                "wget -qO- 'http://localhost:9997/v3/paths/list'",
                shell=True, capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0 and camera_name in result.stdout:
                logger.info("RTSP stream camera=%s is ready (confirmed via MediaMTX API)",
                            camera_name)
                return True
        except Exception:
            pass

        # Fallback: check the lp-cams-N container is running
        running = subprocess.run(
            f"docker inspect {container_name} "
            f"--format '{{{{.State.Running}}}}'",
            shell=True, capture_output=True, text=True,
        ).stdout.strip()
        if running == "true":
            # Container is up — give ffmpeg 3s to open the RTSP session
            time.sleep(3)
            logger.info("RTSP container %s is running — stream likely ready",
                        container_name)
            return True

        elapsed = int(timeout - (deadline - time.time()))
        if elapsed % 15 < 4:
            logger.info("  RTSP camera=%s not ready yet … (%ds elapsed)",
                        camera_name, elapsed)
        time.sleep(3)

    logger.warning("RTSP stream camera=%s not confirmed within %ds — "
                   "DLStreamer may fail to connect", camera_name, timeout)
    return False


def _write_env_var(env_file: str, key: str, value: str) -> None:
    """Write or update a KEY=VALUE line in an env file.

    If the key already exists it is updated in-place; otherwise the line is
    appended.  This ensures variables injected by the benchmark (e.g.
    STREAM_DENSITY) survive ``init.sh`` regeneration and are visible to all
    docker-compose services that read the same env file.
    """
    lines: list[str] = []
    found = False
    if os.path.isfile(env_file):
        with open(env_file) as fh:
            lines = fh.readlines()
        for i, line in enumerate(lines):
            if line.startswith(f"{key}=") or line.startswith(f"{key} ="):
                lines[i] = f"{key}={value}\n"
                found = True
                break
    if not found:
        lines.append(f"{key}={value}\n")
    with open(env_file, "w") as fh:
        fh.writelines(lines)
    logger.info("Set %s=%s in %s", key, value, env_file)


def _scale_pipeline_services(app_dir: str, num_scenes: int, wait: int = 90, resource_config: str = "") -> None:
    """
    Scale the POI video pipeline to N scenes.

    Steps:
      1. Update stream_density in zone_config.json
      2. Generate docker-compose.cameras.yaml with extra RTSP streams
      3. Re-run init.sh to update .env
      4. Write STREAM_DENSITY + BASE_CAMERA_COUNT to docker/.env
      5. Generate per-camera DLStreamer pipeline configs
      6. Bring up new camera services
      7. Wait for web container healthy
      8. Clean stale scenes, restart scene-import
      9. Recreate lp-video (DLStreamer)
    """
    logger.info("Scaling POI to %d scene(s) …", num_scenes)

    _set_stream_density(app_dir, num_scenes)
    _generate_cameras_override(app_dir, num_scenes)
    _reinit_env(app_dir, resource_config=resource_config)

    env_file = os.path.join(app_dir, "docker", ".env")
    _write_env_var(env_file, "STREAM_DENSITY", str(num_scenes))
    # BASE_CAMERA_COUNT is still written for backward-compat with any
    # scene-import fallback path (tells clone-zip to start at Camera_01-3)
    _write_env_var(env_file, "BASE_CAMERA_COUNT", "2")

    _generate_dlstreamer_config(app_dir, num_scenes)

    # Determine new camera service names for this iteration
    base_camera_count = 2
    new_cam_services: list[str] = []
    if num_scenes > 1:
        new_cam = _get_new_camera_name(app_dir, num_scenes)
        if new_cam:
            cam_idx = int(new_cam.split("-")[-1])   # "Camera_01-3" → 3
            new_cam_services = [f"lp-cams-{cam_idx}", f"lp-video-{cam_idx}"]

    # Remove stale containers for new services before (re)creating them.
    # Containers left in "Created" or "Exited" state from a previous iteration
    # hold stale Docker network IDs.  Starting them without --force-recreate
    # causes "network ... not found" errors.  Removing them first ensures
    # docker compose creates fresh containers with the current network.
    if new_cam_services:
        logger.info("Removing stale containers for new services: %s …",
                    " ".join(new_cam_services))
        _docker_compose(app_dir, f"rm -f {' '.join(new_cam_services)}")

    # Bring up all services.  --remove-orphans cleans up containers from
    # previous iterations (e.g. lp-cams-4..7 left over from earlier runs)
    # that would otherwise hold stale network references and cause conflicts.
    # This also creates any new named volumes (vol-dlstreamer-pipeline-root-N).
    logger.info("Starting new camera streams …")
    _docker_compose(app_dir, "up -d --no-recreate --remove-orphans")

    # Pre-initialise the DLStreamer pipeline cache volume for the NEW camera only.
    # New volumes are created by docker compose as root:root with no sticky bit.
    # DLStreamer runs as uid=1999 and needs to create user_defined_pipelines/
    # inside the mount — this fails with PermissionError on a fresh volume.
    # Only the volume for the camera added THIS iteration is new; previous
    # iterations' volumes are already initialised — no need to re-run alpine.
    project = "storewide-lp"
    if num_scenes > 1:
        new_vol_idx = base_camera_count + (num_scenes - 1)
        vol_name = f"{project}_vol-dlstreamer-pipeline-root-{new_vol_idx}"
        logger.info("Initialising volume %s for DLStreamer uid=1999 …", vol_name)
        subprocess.run(
            f"docker run --rm -v {vol_name}:/data alpine sh -c "
            f"'chmod a+rwxt /data && "
            f"mkdir -p /data/user_defined_pipelines && "
            f"chown 1999:1999 /data/user_defined_pipelines'",
            shell=True, capture_output=True, text=True,
        )

    # Explicitly force-recreate the new camera services to guarantee fresh
    # containers with a valid network attachment (--no-recreate skips them if
    # they were already recreated by the rm above but not yet started).
    if new_cam_services:
        cam_svc = new_cam_services[0]   # lp-cams-{N}
        logger.info("Force-starting camera stream service %s …", cam_svc)
        _docker_compose(app_dir, f"up -d --force-recreate {cam_svc}")

    _wait_for_web_healthy()

    # Delete any scenes/cameras cloned in previous iterations, then register
    # the new scene + camera directly via the SceneScape REST API.
    #
    # OLD flow: generate ZIP → docker compose up scene-import → wait ~10s
    # NEW flow: clone ZIP in-memory → POST /api/v1/import-scene/ → ~200ms
    #
    # SceneScape must have Camera_01-N registered BEFORE lp-video-N connects
    # to the RTSP source; otherwise the controller drops all DLStreamer output
    # for the new camera (no embeddings → no FAISS match → no alert).
    _delete_cloned_scenes(app_dir, num_scenes)

    if num_scenes > 1:
        new_cam = _get_new_camera_name(app_dir, num_scenes)
        zone_cfg = _read_zone_config(app_dir)
        base_scene_name = zone_cfg.get("scene_name", "conference room")
        new_scene_name = f"{base_scene_name}-{num_scenes}"
        if new_cam:
            logger.info("Registering scene=%s camera=%s via SceneScape REST API …",
                        new_scene_name, new_cam)
            scene_uid, cam_uid = _scenescape_import_scene(
                app_dir, new_scene_name, new_cam)
            if not scene_uid:
                # Direct import failed — fall back to scene-import sidecar
                logger.warning("SceneScape API import failed — falling back to scene-import sidecar")
                _docker_compose(app_dir, "rm -f -s scene-import")
                _docker_compose(app_dir, "up -d scene-import")
                _wait_for_scene_import_completion(timeout=180)


    # Wait for the new camera's RTSP stream to be served by MediaMTX.
    # DLStreamer connects to RTSP at startup; if lp-cams-N is still initialising
    # when lp-video is force-recreated, DLStreamer silently skips that pipeline.
    if num_scenes > 1:
        new_cam = _get_new_camera_name(app_dir, num_scenes)
        if new_cam:
            _wait_for_camera_rtsp_ready(new_cam, timeout=60)

    # Recreate only lp-video (Camera_01 baseline) and the NEW lp-video-{N}.
    # Already-running lp-video-3, lp-video-4, … from previous iterations have
    # unchanged pipeline configs and keep their RTSP connections — no need to
    # restart them.  Restarting N containers on every iteration adds O(N) setup
    # time and disrupts healthy pipelines unnecessarily.
    # --remove-orphans cleans leftover lp-video-{old} containers from prior runs.
    logger.info("Recreating DLStreamer container(s) for %d scene(s) …", num_scenes)
    if num_scenes == 1:
        # Baseline: only Camera_01's container
        video_services = "lp-video"
    else:
        # Incremental: baseline container + the single new camera container only
        new_vid_idx = base_camera_count + (num_scenes - 1)
        video_services = f"lp-video lp-video-{new_vid_idx}"
    _docker_compose(app_dir, f"up -d --force-recreate --remove-orphans {video_services}")

    # poi-backend subscribes to scenescape/data/camera/+ (wildcard) so it
    # receives embeddings from all cameras without restart.  MQTT_IMAGE_CAMERAS
    # only controls the thumbnail-grab strategy (MQTT vs RTSP), not embedding
    # ingestion.  Skip the force-recreate to avoid resetting FAISS/Redis state
    # and wasting the 90-second stabilisation window.

    # Wait for DLStreamer to produce first detection (replaces fixed 90s sleep).
    # For incremental iterations, wait for the NEW camera specifically — Camera_01/02
    # detections would give a false-positive "pipeline warm" signal while
    # Camera_01-N's lp-video-N is still initialising.
    new_cam_for_warmup = _get_new_camera_name(app_dir, num_scenes)
    _wait_for_first_detection(timeout=wait, poll_interval=3,
                              camera_filter=new_cam_for_warmup)
    logger.info("Pipeline warm — adding 10s stabilisation buffer …")
    time.sleep(10)


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
    camera_filter: Optional[str] = None,
) -> Dict[str, float]:
    """Compute real end-to-end latency from POI alerts API.

    Uses ``timestamp`` (DLStreamer frame capture time from the MQTT payload)
    as the start time and ``dispatched_at`` as the end time, giving true
    end-to-end latency from frame capture → alert dispatch (includes FAISS
    match overhead but NOT DLStreamer pipeline inference time — that is
    captured by the VLM metrics logger which uses the same frame timestamp).

    NOTE: In stream-density mode this function is a fallback only.  The
    primary path reads per-camera latency directly from the VLM metrics file
    (``vlm_Person-of-Interest_{camera_id}_avg_ms``) which is written by
    ``alert_service.py`` via ``user_log_start_time`` / ``log_end_time``.

    Args:
        since: If provided, only include alerts dispatched after this time.
               Filters out stale alerts from previous benchmark runs.
        camera_filter: If provided, only include alerts from this specific camera.
                       Used in stream-density mode to measure the newly added camera's
                       latency independently.  When ``None``, all cameras are included.

    Returns dict with ``poi_e2e_latency_avg_ms``, ``poi_e2e_latency_max_ms``,
    ``poi_e2e_latency_min_ms``, and ``poi_e2e_alert_count``.
    """
    import urllib.request
    import urllib.error

    try:
        req = urllib.request.Request("http://localhost:8000/api/v1/alerts?limit=100")
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
    filtered_camera = 0
    for alert in alerts:
        # Filter by camera when requested (stream-density: test new camera only)
        if camera_filter:
            alert_camera = (
                alert.get("match", {}).get("camera_id")
                or alert.get("camera_id", "")
            )
            if alert_camera != camera_filter:
                filtered_camera += 1
                continue

        # Use DLStreamer frame capture timestamp as start (true E2E start).
        # alert["timestamp"] = payload["timestamp"] from MQTT = frame capture time.
        # alert["dispatched_at"] = wall-clock when alert was fired after FAISS match.
        frame_ts = alert.get("timestamp", "")
        dispatched_str = alert.get("dispatched_at", "")
        if not frame_ts or not dispatched_str:
            continue
        try:
            frame_ts = frame_ts.replace("Z", "+00:00")
            dispatched_str = dispatched_str.replace("Z", "+00:00")
            start = _dt.fromisoformat(frame_ts)
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
        except (ValueError, TypeError):
            continue

    if skipped:
        logger.info("Filtered out %d stale alerts (before benchmark start)", skipped)
    if filtered_camera:
        logger.debug("Filtered out %d alerts from other cameras (camera_filter=%s)",
                     filtered_camera, camera_filter)

    if not latencies_ms:
        return {}

    stats: Dict[str, float] = {
        "poi_e2e_latency_avg_ms": sum(latencies_ms) / len(latencies_ms),
        "poi_e2e_latency_max_ms": max(latencies_ms),
        "poi_e2e_latency_min_ms": min(latencies_ms),
        "poi_e2e_alert_count": len(latencies_ms),
    }
    label = "frame capture → alert dispatch (alerts API)"
    if camera_filter:
        label = f"camera={camera_filter} " + label
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
    results_dir: str, stream_density: bool = False,
    since_ms: Optional[int] = None
) -> Dict[str, float]:
    """
    Extract POI detection-to-alert latency from vlm_application_metrics files.

    These files are written by the vlm_metrics_logger package via
    user_log_start_time (detection) and log_end_time (alert dispatch)
    calls in the poi-backend.

    For single benchmarks uses ``get_vlm_application_latency`` (all pairs).
    For stream density uses ``get_vlm_application_latency_stream_density``
    with ``since_ms`` so only pairs from the current iteration are measured.
    The file is never deleted between iterations — poi-backend holds an open
    RotatingFileHandler to it.
    """
    all_stats: Dict[str, float] = {}
    search_dirs = [results_dir, "/tmp"]

    if stream_density:
        for d in search_dirs:
            if not os.path.isdir(d):
                continue
            try:
                stats = get_vlm_application_latency_stream_density(
                    d, last_n_pairs=20, since_ms=since_ms)
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
         Two unique_id variants are written per alert:
           * ``person-of-interest`` — aggregate across all cameras
           * ``{camera_id}``        — per-camera (for stream-density isolation)
      2. Alerts API fallback (``timestamp`` frame capture → ``dispatched_at``) —
         same semantic as (1) but parsed from stored alert JSON rather than
         the live metrics file.
      3. Returns 0 if no data available.

    Note: Docker-log-based ``log_detection_to_alert_ms`` is excluded because
    log timestamps have only second-level precision and the first-match-to-
    first-alert gap includes dedup delay (60 s TTL), making it unreliable
    as a per-event latency metric.
    """
    # Primary: vlm_application_metrics file-based values.
    # Exclude the aggregate ``person-of-interest`` key — per-camera entries
    # (e.g., ``vlm_Person-of-Interest_Camera_01-3_avg_ms``) are more precise
    # in stream-density mode because they isolate each camera's contribution.
    # If no per-camera entries exist (e.g., first/single run), fall back to
    # the aggregate ``person-of-interest`` entry.
    per_camera_values = [
        v for k, v in stats.items()
        if k.startswith("vlm_") and "_person-of-interest_" not in k
        and isinstance(v, (int, float)) and v > 0
    ]
    aggregate_values = [
        v for k, v in stats.items()
        if k.startswith("vlm_") and "_person-of-interest_" in k
        and isinstance(v, (int, float)) and v > 0
    ]
    vlm_values = per_camera_values or aggregate_values
    if vlm_values:
        if metric == "max":
            return max(vlm_values)
        return mean(vlm_values)

    # Fallback: alerts API E2E latency (frame capture → dispatched_at)
    e2e_avg = stats.get("poi_e2e_latency_avg_ms", 0.0)
    e2e_max = stats.get("poi_e2e_latency_max_ms", 0.0)
    if e2e_avg > 0:
        if metric == "max":
            return e2e_max
        return e2e_avg

    return 0.0


def _clean_metrics(results_dir: str) -> None:
    """Remove stale metrics files before each measurement iteration.

    Note: vlm_application_metrics files are NOT deleted here because poi-backend
    holds an open RotatingFileHandler to them.  Deleting the file causes the
    handler to write to an unlinked inode (invisible to rglob) for the rest of
    the run.  Instead, callers pass a ``since_ms`` filter so each iteration
    reads only its own pairs from the cumulative file.
    """
    patterns = [
        "vlm_performance_metrics*.txt",
    ]
    for d in [results_dir, "/tmp"]:
        for pat in patterns:
            for f in glob.glob(os.path.join(d, pat)):
                try:
                    os.remove(f)
                except OSError:
                    pass


def _reset_alert_dedup() -> None:
    """Clear Redis alert dedup and history via the poi-backend REST API.

    Between stream-density iterations, the Redis dedup keys (``alert:sent:*``)
    must be reset so the newly added camera can fire fresh alerts.

    Background: the dedup key is the object UUID — shared across all cameras
    tracking the same physical person.  If Camera_01 already alerted on
    UUID-X within ALERT_DEDUP_TTL seconds, Camera_01-3 would be silently
    suppressed when it detects the same person, preventing any alert from
    the new camera regardless of its latency.

    ``DELETE /api/v1/alerts`` clears both the recent-alerts list AND all
    ``alert:sent:*`` / ``alert:*`` Redis keys so every iteration starts fresh.
    """
    import urllib.request
    try:
        req = urllib.request.Request(
            "http://localhost:8000/api/v1/alerts",
            method="DELETE",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = resp.read().decode()
        logger.info("Alert dedup reset: %s", result)
    except Exception as e:
        logger.warning("Could not reset alert dedup (non-fatal): %s", e)


# ---------------------------------------------------------------------------
# POI Stream Density Runner
# ---------------------------------------------------------------------------

class POIStreamDensity:
    """
    Iteratively increases the number of camera/scene pipelines until the
    newly added camera's end-to-end detection-to-alert latency exceeds
    *target_latency_ms*.  This determines the maximum number of cameras
    the hardware can support for this application.

    Iteration logic
    ---------------
    *Baseline* (num_scenes=1): Camera_01 and Camera_02 are already running.
    Any alert confirms the pipeline is functional.

    *Incremental* (num_scenes > 1): each iteration adds exactly one new
    camera (Camera_01-3, Camera_01-4, …) and one cloned scene.  The
    benchmark waits specifically for an alert from that new camera, then
    measures its E2E latency.  Only if latency ≤ target does the next
    camera get added.

    What gets added per iteration (num_scenes > 1)
    ------------------------------------------------
      - One new RTSP stream  (lp-cams-N Docker service)
      - One extra DLStreamer pipeline config
      - One new scene clone in SceneScape (scene-import re-run)
      - lp-video (DLStreamer) restarted with the updated config

    What stays running untouched
    -----------------------------
      - poi-backend  ← subscribes via MQTT wildcard ``scenescape/data/camera/+``
        so it automatically processes every new camera without a restart.
      - SceneScape core (web, controller, broker, ntpserv, pgserver, vdms)
      - poi-redis, poi-alert-service, poi-ui, mediaserver
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
        max_alert_wait: int = 180,
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
        # max total seconds to wait for an alert from the new camera per iteration;
        # should cover at least 2-3 full video cycles to handle phase offset.
        # Increase via MAX_ALERT_WAIT env var or --max_alert_wait flag for CPU mode.
        self.max_alert_wait = max_alert_wait
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

    def _wait_for_alert_or_timeout(self, duration: int,
                                    since: Optional[datetime] = None) -> bool:
        """Poll for new alerts, exit early on first alert after *since*.

        Used both in single-benchmark mode (time-to-first-alert) and in
        stream-density mode (ensure at least one alert before data collection).

        Returns True if a fresh alert was found, False if duration expired.
        """
        import urllib.request
        import urllib.error

        poll_interval = 5
        elapsed = 0
        since_ts = since or datetime.utcnow()
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
                # Filter to alerts created after iteration start
                fresh = [
                    a for a in (alerts if isinstance(alerts, list) else [])
                    if a.get("created_at", a.get("timestamp", "")) > since_ts.strftime("%Y-%m-%dT%H:%M:%S")
                ]
                if fresh:
                    logger.info("Fresh alert received after %ds — stopping early", elapsed)
                    # Brief extra wait for metrics files to flush
                    time.sleep(5)
                    return True
            except Exception:
                pass

            logger.info("No fresh alerts yet (%d/%ds elapsed)", elapsed, duration)

        logger.info("Alert wait duration reached (%ds) — continuing anyway", duration)
        return False

    def run(self) -> StreamDensityResult:
        """Execute the POI stream-density loop.

        For each iteration:
          1. Add one new camera + scene (num_scenes > 1) or use base cameras (num_scenes = 1).
          2. Wait for an alert specifically from the NEW camera.
          3. Compute E2E latency filtered to that camera's alerts.
          4. If latency ≤ target → add next camera and repeat.
             If no alert received or latency > target → report max_scenes and stop.
        """
        self._print_header()
        result = StreamDensityResult(target_latency_ms=self.target_latency_ms)

        num_scenes = self.single_run_scenes if self.single_run else 1
        max_iter = 1 if self.single_run else self.max_iterations
        best: Optional[IterationResult] = None

        for iteration in range(1, max_iter + 1):
            # Determine which camera is newly added in this iteration.
            # None → baseline (num_scenes=1, Camera_01 + Camera_02 already present).
            new_camera = _get_new_camera_name(self.app_dir, num_scenes)

            print(f"\n{'='*70}")
            print(f"Iteration {iteration}: Testing {num_scenes} scene(s)  "
                  f"[new camera: {new_camera or 'N/A (baseline)'}]")
            print(f"{'='*70}")

            if not self._memory_safe():
                logger.warning("Memory threshold exceeded – stopping.")
                break

            # Record iteration start time for filtering stale alerts
            iteration_start = datetime.utcnow()

            # Reset Redis alert dedup so the new camera can fire fresh alerts.
            # Without this, UUID-based dedup from previous iterations would
            # suppress alerts from Camera_01-3 if the same person was already
            # alerted on Camera_01 (dedup key = object UUID, not camera-specific).
            _reset_alert_dedup()

            # Clean old metrics before each measurement
            _clean_metrics(self.results_dir)

            if self.single_run and self._services_running():
                # Single benchmark: services already up, skip scaling
                logger.info("Services already running — skipping scaling for single benchmark")
            else:
                # Scale to desired scene count (adds new camera + scene)
                _scale_pipeline_services(self.app_dir, num_scenes, wait=self.init_duration,
                                         resource_config=self.resource_config)

            # ── Wait for alert from the newly added camera ──────────────────
            # For num_scenes=1 (baseline) there is no "new" camera, so we fall
            # back to the generic any-camera wait.  For num_scenes>1 we wait
            # specifically for the new camera to prove it is fully active.
            #
            # The wait retries in init_duration windows until either an alert
            # is received OR total wait exceeds max_alert_wait.  This is
            # important for CPU mode where inference is slower and the POI
            # face may only appear once per video loop (~55s), so a single
            # 45s window is not guaranteed to cover a full video cycle.
            if new_camera:
                # Camera-specific wait: must see an alert from the new camera
                if self.single_run:
                    _wait_for_alert_from_camera(
                        new_camera, self.benchmark_duration, since=iteration_start,
                    )
                else:
                    max_alert_wait = self.max_alert_wait
                    total_waited = 0
                    got_alert = False
                    while not got_alert and total_waited < max_alert_wait:
                        window = min(self.init_duration, max_alert_wait - total_waited)
                        got_alert = _wait_for_alert_from_camera(
                            new_camera, window, since=iteration_start,
                        )
                        total_waited += window
                        if not got_alert and total_waited < max_alert_wait:
                            logger.info(
                                "No alert from camera=%s yet — retrying "
                                "(%ds elapsed, max=%ds) …",
                                new_camera, total_waited, max_alert_wait,
                            )
                    if not got_alert:
                        logger.warning(
                            "No alert from new camera=%s within max_alert_wait=%ds",
                            new_camera, max_alert_wait,
                        )
                    logger.info("Collecting data for %ds …", self.stabilise_duration)
                    time.sleep(self.stabilise_duration)
            else:
                # Baseline iteration: any alert is acceptable
                if self.single_run:
                    self._wait_for_alert_or_timeout(self.benchmark_duration,
                                                    since=iteration_start)
                else:
                    self._wait_for_alert_or_timeout(self.init_duration,
                                                    since=iteration_start)
                    logger.info("Collecting data for %ds …", self.stabilise_duration)
                    time.sleep(self.stabilise_duration)

            # Use actual elapsed time for log collection window
            elapsed_seconds = int((datetime.utcnow() - iteration_start).total_seconds())
            log_window = elapsed_seconds if self.single_run else self.stabilise_duration

            # Collect latency from metrics files + docker logs
            # Use calendar.timegm to convert naive UTC datetime → epoch ms correctly.
            # datetime.timestamp() treats naive datetimes as LOCAL time, giving a
            # 5.5h offset on IST systems that makes since_ms far too early.
            iter_start_ms = int(calendar.timegm(iteration_start.timetuple()) * 1000)
            log_stats = _collect_poi_latency_from_docker_logs(
                self.app_dir, log_window)
            file_stats = _collect_poi_latency_from_metrics_files(
                self.results_dir, stream_density=not self.single_run,
                since_ms=iter_start_ms if not self.single_run else None)

            # E2E latency: filtered to the new camera when in incremental mode,
            # so pass/fail reflects the NEW camera's responsiveness — not the
            # aggregate of all previously-running cameras.
            e2e_stats = _collect_poi_e2e_latency_from_alerts(
                since=iteration_start,
                camera_filter=new_camera,   # None → all cameras (baseline)
            )

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
                new_camera=new_camera or "",
            )

            self._print_iteration(it_result)

            # ── Pass / fail decision ────────────────────────────────────────
            # When a new_camera is defined the benchmark REQUIRES an alert from
            # that specific camera.  A missing alert means the system cannot
            # keep up with the additional pipeline load — treat as FAIL.
            has_camera_alert = bool(e2e_stats)          # camera-filtered → non-empty means alert received
            has_detections   = it_result.actual_detections > 0
            has_matches      = int(stats.get("poi_matches", 0)) > 0

            if new_camera:
                # ── Incremental camera mode ──────────────────────────────────
                if not has_camera_alert:
                    it_result.passed = False
                    print(f"  ✗ NO ALERT from new camera={new_camera} — "
                          "system cannot process additional pipeline load")
                    result.iterations.append(it_result)
                    break
                elif latency <= 0:
                    # Alert received but latency sub-measurable (< 1 s)
                    it_result.passed = True
                    best = it_result
                    print(f"  ✓ PASSED  (camera={new_camera} alert received, latency < 1s)")
                elif latency <= self.target_latency_ms:
                    it_result.passed = True
                    best = it_result
                    print(f"  ✓ PASSED  (camera={new_camera} "
                          f"latency={latency:.0f}ms ≤ {self.target_latency_ms:.0f}ms)")
                else:
                    it_result.passed = False
                    print(f"  ✗ FAILED  (camera={new_camera} "
                          f"latency={latency:.0f}ms > {self.target_latency_ms:.0f}ms)")
                    result.iterations.append(it_result)
                    break
            else:
                # ── Baseline iteration (num_scenes=1) — any alert ────────────
                latency_ok = latency > 0 and latency <= self.target_latency_ms
                if latency == 0 and has_matches:
                    it_result.passed = True
                    best = it_result
                    print("  ✓ PASSED  (baseline: matches found, latency < 1s)")
                elif latency == 0 and has_detections and not has_matches:
                    it_result.passed = True
                    best = it_result
                    print(f"  ✓ PASSED  (baseline: {it_result.actual_detections} detections, "
                          "no matches — target not in frame during window)")
                elif latency == 0:
                    print("  ⚠ NO DATA – no detections collected")
                    if iteration > 1:
                        break
                elif latency_ok:
                    it_result.passed = True
                    best = it_result
                    print(f"  ✓ PASSED  (baseline: latency={latency:.0f}ms ≤ {self.target_latency_ms:.0f}ms)")
                else:
                    it_result.passed = False
                    print(f"  ✗ FAILED  (baseline: latency {latency:.0f}ms > {self.target_latency_ms:.0f}ms)")
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
        if it.new_camera:
            print(f"  New Camera:  {it.new_camera}")
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
                  f"@ {result.best_iteration.num_scenes} scene(s)"
                  + (f"  [{result.best_iteration.new_camera}]"
                     if result.best_iteration.new_camera else ""))
        print()
        print(f"{'Scenes':<10}{'Camera':<16}{'Latency':<12}{'Detections':<14}"
              f"{'Alerts':<10}{'Mem %':<10}{'CPU %':<10}{'Status':<10}")
        print("-" * 92)
        for it in result.iterations:
            status = "✓ PASS" if it.passed else "✗ FAIL"
            cam = it.new_camera or "baseline"
            print(f"{it.num_scenes:<10}{cam:<16}{it.latency_ms:<12.0f}"
                  f"{it.actual_detections:<14}{it.alerts_generated:<10}"
                  f"{it.memory_percent:<10.1f}{it.cpu_percent:<10.1f}{status}")
        print("=" * 92)

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
                    "new_camera": it.new_camera,
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
            w.writerow(["scenes", "new_camera", "latency_ms", "detections", "alerts",
                         "detections_per_scene", "passed", "memory_pct", "cpu_pct"])
            for it in result.iterations:
                w.writerow([it.num_scenes, it.new_camera or "baseline",
                            f"{it.latency_ms:.0f}",
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
        max_alert_wait=args.max_alert_wait,
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
    init_duration = _env_int("INIT_DURATION", 45)
    stabilise_duration = _env_int("STABILISE_DURATION", 30)
    benchmark_duration = _env_int("BENCHMARK_DURATION", 120)
    results_dir = _env_str("RESULTS_DIR", "./results")
    max_iterations = _env_int("MAX_ITERATIONS", 50)
    # Default max_alert_wait covers ~3 full video cycles (video ≈ 55s) plus
    # inference warmup, giving enough time for CPU-mode pipelines to generate
    # a face match on the new camera regardless of video phase offset.
    max_alert_wait = _env_int("MAX_ALERT_WAIT", 180)

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
    p_run.add_argument("--max_alert_wait", type=int, default=max_alert_wait,
                       help="Max total seconds to wait for an alert from the "
                            "new camera per iteration (default: 180). "
                            "The wait retries in init_duration windows until "
                            "an alert is received or this timeout expires. "
                            "Increase for CPU mode where inference is slower "
                            "and a full video cycle (~55s) must complete.")
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
