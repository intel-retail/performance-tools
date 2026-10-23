#
# Apache v2 license
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


"""Sample Docker CPU and memory usage for running Compose services."""

import argparse
import csv
import json
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from threading import Event


MEMORY_PATTERN = re.compile(r"\s*([\d.]+)\s*([KMGT]?i?B)")
MEMORY_FACTORS = {
    "B": 1 / 1024**2,
    "KB": 1000 / 1024**2,
    "MB": 1000**2 / 1024**2,
    "GB": 1000**3 / 1024**2,
    "TB": 1000**4 / 1024**2,
    "KiB": 1 / 1024,
    "MiB": 1,
    "GiB": 1024,
    "TiB": 1024**2,
}


def average(values):
    values = [float(value) for value in values if isinstance(value, (int, float))]
    return round(mean(values), 3) if values else None


def parse_percent(value):
    try:
        return float(value.strip().rstrip("%"))
    except (AttributeError, TypeError, ValueError):
        return None


def parse_memory_mib(value):
    match = MEMORY_PATTERN.match(value or "")
    if not match:
        return None
    return round(float(match.group(1)) * MEMORY_FACTORS[match.group(2)], 3)


def running_services():
    output = subprocess.check_output(
        ["docker", "compose", "ps", "--format", "json"],
        text=True, stderr=subprocess.STDOUT,
    )
    containers = []
    for line in output.splitlines():
        try:
            container = json.loads(line)
        except json.JSONDecodeError:
            continue
        if container.get("State", "").lower() == "running" and container.get("ID"):
            containers.append(container)
    return containers


def sample_services():
    containers = running_services()
    if not containers:
        return []

    services_by_id = {container["ID"][:12]: container.get("Service", "unknown")
                      for container in containers}
    services_by_name = {container.get("Name"): container.get("Service", "unknown")
                        for container in containers}
    command = ["docker", "stats", "--no-stream", "--format", "{{json .}}"]
    command.extend(container["ID"] for container in containers)
    output = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)
    timestamp = datetime.now(timezone.utc).isoformat()
    samples = []
    for line in output.splitlines():
        try:
            stats = json.loads(line)
        except json.JSONDecodeError:
            continue
        container_id = (stats.get("ID") or stats.get("Container") or "")[:12]
        name = stats.get("Name") or ""
        service = services_by_id.get(container_id) or services_by_name.get(name)
        if not service:
            continue
        memory_used, _, memory_limit = (stats.get("MemUsage", "").partition("/"))
        samples.append({
            "timestamp": timestamp,
            "service": service,
            "container": name,
            "cpu_percent": parse_percent(stats.get("CPUPerc")),
            "memory_percent": parse_percent(stats.get("MemPerc")),
            "memory_usage_mib": parse_memory_mib(memory_used),
            "memory_limit_mib": parse_memory_mib(memory_limit),
        })
    return samples


def host_metric_averages(path):
    if not path or not path.is_file():
        return {
            "cpu_utilization_percent": None,
            "gpu_utilization_percent": None,
            "memory_utilization_percent": None,
        }
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {
            "cpu_utilization_percent": None,
            "gpu_utilization_percent": None,
            "memory_utilization_percent": None,
        }
    return {
        "cpu_utilization_percent": average([row[1] for row in payload.get("cpu_utilization", [])
                                             if len(row) > 1]),
        "gpu_utilization_percent": average([row[1] for row in payload.get("gpu_utilization", [])
                                             if len(row) > 1]),
        "memory_utilization_percent": average([row[4] for row in payload.get("memory", [])
                                                if len(row) > 4]),
    }


def write_results(output_prefix, samples, interval, host_metrics):
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = Path(f"{output_prefix}.csv")
    json_path = Path(f"{output_prefix}.json")
    fields = ("timestamp", "service", "container", "cpu_percent", "memory_percent",
              "memory_usage_mib", "memory_limit_mib")
    with csv_path.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerows(samples)

    by_service = {}
    for sample in samples:
        by_service.setdefault(sample["service"], []).append(sample)
    service_averages = {}
    for service, service_samples in sorted(by_service.items()):
        service_averages[service] = {
            "samples": len(service_samples),
            "containers": sorted({sample["container"] for sample in service_samples}),
            "average_cpu_percent": average([sample["cpu_percent"] for sample in service_samples]),
            "average_memory_percent": average([sample["memory_percent"] for sample in service_samples]),
            "average_memory_usage_mib": average([sample["memory_usage_mib"] for sample in service_samples]),
            "average_memory_limit_mib": average([sample["memory_limit_mib"] for sample in service_samples]),
        }
    result = {
        "sample_interval_seconds": interval,
        "services": service_averages,
        "host_metric_averages": host_metric_averages(host_metrics),
    }
    json_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved Docker service averages to {json_path} and samples to {csv_path}")
    return bool(samples)


def run(output_prefix, interval, duration, host_metrics, ready_file):
    stop = Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    samples = []
    deadline = time.monotonic() + duration if duration else None

    try:
        sample_start = time.monotonic()
        try:
            samples.extend(sample_services())
        except (OSError, subprocess.CalledProcessError) as error:
            print(f"Docker stats sample failed: {error}", file=sys.stderr)
        if ready_file:
            ready_file.write_text("ready\n")
        while not stop.is_set() and (deadline is None or time.monotonic() < deadline):
            sample_start = time.monotonic()
            try:
                samples.extend(sample_services())
            except (OSError, subprocess.CalledProcessError) as error:
                print(f"Docker stats sample failed: {error}", file=sys.stderr)
            wait = max(0, interval - (time.monotonic() - sample_start))
            if deadline is not None:
                wait = min(wait, max(0, deadline - time.monotonic()))
            if wait:
                stop.wait(wait)
        return write_results(output_prefix, samples, interval, host_metrics)
    finally:
        if ready_file:
            ready_file.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--host-metrics", type=Path)
    parser.add_argument("--ready-file", type=Path)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--duration", type=float, default=0,
                        help="stop after this many seconds; defaults to signal-controlled sampling")
    args = parser.parse_args()
    if args.interval <= 0 or args.duration < 0:
        parser.error("interval must be positive and duration cannot be negative")
    if not run(args.output_prefix, args.interval, args.duration, args.host_metrics, args.ready_file):
        print("Docker stats produced no samples for running Compose services", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())