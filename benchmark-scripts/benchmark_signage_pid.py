#
# Apache v2 license
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


"""Sample the running PID pipeline through its MQTT output and Docker metrics."""

import argparse
import csv
import json
import os
import re
import select
import subprocess
import sys
import time
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean


SERVICE = "dlstreamer-pipeline-server"
BROKER = "mqtt-broker"
TOPIC = "yolo_od_results"
FPS_PATTERN = re.compile(r"FpsCounter\(last [^)]*\): total=([\d.]+) fps")


def docker(*args):
    return subprocess.check_output(["docker", *args], text=True, stderr=subprocess.STDOUT).strip()


def average(values):
    return round(mean(values), 3) if values else None


def parse_size(value):
    match = re.match(r"\s*([\d.]+)\s*([KMGT]?i?B)", value)
    if not match:
        return None
    unit = match.group(2)
    factors = {"B": 1, "KB": 1000, "MB": 1000**2, "GB": 1000**3,
               "KiB": 1024, "MiB": 1024**2, "GiB": 1024**3, "TiB": 1024**4}
    return round(float(match.group(1)) * factors[unit] / 1024**2, 2)


def parse_frame(line, arrival_ns=None):
    payload = json.loads(line)
    metadata = payload["metadata"]
    status = metadata.get("pipeline", {}).get("status", {})
    detections = metadata.get("gva_meta", [])
    frame_time_ns = metadata.get("time")
    arrival_delay_ms = None
    if (isinstance(frame_time_ns, int) and not isinstance(frame_time_ns, bool)
            and arrival_ns is not None and 0 <= arrival_ns - frame_time_ns < 60_000_000_000):
        arrival_delay_ms = (arrival_ns - frame_time_ns) / 1_000_000
    return {
        "frame_id": metadata.get("frame_id"),
        "frame_timestamp_ns": frame_time_ns,
        "arrival_timestamp_ns": arrival_ns,
        "detections": len(detections),
        "frame_fps": status.get("frame_fps"),
        "pipeline_latency_reported": status.get("avg_pipeline_latency"),
        "mqtt_arrival_delay_ms": arrival_delay_ms,
    }


def gpu_compute_usage(path):
    if path is None:
        return None
    data = json.loads(path.read_text())
    samples = []
    for entry in data:
        value = entry.get("CCS %")
        if value is not None:
            samples.append(float(value))
    return average(samples)


def run(duration, output, restart, gpu_metrics=None, device="CPU"):
    for container in (SERVICE, BROKER):
        if docker("inspect", "--format", "{{.State.Running}}", container) != "true":
            raise RuntimeError(f"{container} is not running; start the stack with make up")

    frames = []
    cpu = []
    memory = []
    started_at = None
    if restart:
        # Restart by container name, not `compose restart`, so the script does not depend on the caller's Compose project.
        subprocess.run(["docker", "restart", SERVICE], check=True)
        started_at = datetime.fromisoformat(
            docker("inspect", "--format", "{{.State.StartedAt}}", SERVICE).replace("Z", "+00:00")
        ).timestamp()
    subscriber = subprocess.Popen(
        ["docker", "exec", BROKER, "mosquitto_sub", "-h", "localhost", "-t", TOPIC],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        sample_start_wall = time.time()
        sample_start = time.monotonic()
        deadline = sample_start + duration
        next_stats = sample_start
        first_frame_at = None
        buffered = b""
        with ThreadPoolExecutor(max_workers=1) as pool:
            stats_future = None
            while time.monotonic() < deadline:
                if subscriber.poll() is not None:
                    raise RuntimeError(f"MQTT subscriber exited: {subscriber.stderr.read().decode().strip()}")
                now = time.monotonic()
                if stats_future is not None and stats_future.done():
                    stats = json.loads(stats_future.result())
                    cpu.append(float(stats["CPUPerc"].strip("%")))
                    used = parse_size(stats["MemUsage"].split("/")[0])
                    if used is not None:
                        memory.append(used)
                    stats_future = None
                if now >= next_stats and stats_future is None:
                    stats_future = pool.submit(docker, "stats", "--no-stream", "--format", "{{json .}}", SERVICE)
                    next_stats = now + 1
                ready, _, _ = select.select([subscriber.stdout], [], [], min(0.2, max(0, deadline - time.monotonic())))
                if ready:
                    chunk = os.read(subscriber.stdout.fileno(), 65536)
                    arrived_ns = time.time_ns()
                    buffered += chunk
                    while b"\n" in buffered:
                        line, buffered = buffered.split(b"\n", 1)
                        try:
                            frame = parse_frame(line, arrived_ns)
                        except (ValueError, KeyError, TypeError):
                            continue
                        if first_frame_at is None:
                            first_frame_at = arrived_ns / 1_000_000_000
                        frames.append(frame)
            if stats_future is not None:
                stats = json.loads(stats_future.result())
                cpu.append(float(stats["CPUPerc"].strip("%")))
                used = parse_size(stats["MemUsage"].split("/")[0])
                if used is not None:
                    memory.append(used)
        if not frames:
            raise RuntimeError("No PID frames received on MQTT; check the pipeline and topic")
    finally:
        subscriber.terminate()
        try:
            subscriber.communicate(timeout=3)
        except subprocess.TimeoutExpired:
            subscriber.kill()
            subscriber.communicate()

    log_lines = docker("logs", "--since", datetime.fromtimestamp(
        started_at if started_at else sample_start_wall, timezone.utc
    ).isoformat(), SERVICE).splitlines()
    log_fps = [float(match.group(1)) for line in log_lines if (match := FPS_PATTERN.search(line))]
    latency = [float(frame["pipeline_latency_reported"]) for frame in frames
               if isinstance(frame["pipeline_latency_reported"], (int, float))]
    # Frame 0 proves the subscription beat the restarted pipeline; without it the startup number would be inflated.
    captured_opening_frame = frames[0]["frame_id"] == 0
    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "duration_seconds": duration,
        "frames": len(frames),
        "fps_dlstreamer": average(log_fps),
        "fps_mqtt_reported": average([float(frame["frame_fps"]) for frame in frames
                                       if isinstance(frame["frame_fps"], (int, float))]),
        "detections_per_frame": average([frame["detections"] for frame in frames]),
        "detections_total": sum(frame["detections"] for frame in frames),
        "cpu_percent": average(cpu),
        "memory_mib": average(memory),
        "pipeline_latency_reported": average(latency),
        "mqtt_arrival_delay_ms": average([frame["mqtt_arrival_delay_ms"] for frame in frames
                          if frame["mqtt_arrival_delay_ms"] is not None]),
        "startup_to_first_frame_ms": round((first_frame_at - started_at) * 1000, 3)
                                     if started_at and captured_opening_frame else None,
        "gpu_compute_percent": gpu_compute_usage(gpu_metrics),
    }
    output.mkdir(parents=True, exist_ok=True)
    name = datetime.now().strftime("pid_benchmark_%Y%m%d_%H%M%S")
    (output / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")
    with (output / f"{name}.csv").open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=result.keys())
        writer.writeheader()
        writer.writerow(result)
    (output / f"{name}_frames.json").write_text(json.dumps(frames, indent=2) + "\n")
    with (output / f"{name}_frames.csv").open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=frames[0].keys())
        writer.writeheader()
        writer.writerows(frames)
    print(json.dumps(result, indent=2))
    print(f"Saved {output / name}.json, .csv, _frames.json and _frames.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=int, default=300)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--device", choices=("CPU", "GPU", "NPU"), default="CPU")
    parser.add_argument("--no-restart", action="store_true",
                        help="sample without restarting PID (no startup metric)")
    parser.add_argument("--gpu-metrics", type=Path, help="qmassa parsed JSON from the same measurement window")
    args = parser.parse_args()
    if args.duration < 1:
        parser.error("--duration must be positive")
    try:
        run(args.duration, args.output, not args.no_restart, args.gpu_metrics, args.device)
    except (subprocess.CalledProcessError, OSError, RuntimeError, ValueError, TypeError, KeyError) as error:
        print(f"PID benchmark failed: {error}", file=sys.stderr)
        sys.exit(1)