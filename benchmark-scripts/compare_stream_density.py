"""Capture run provenance and generate a cross-accelerator Markdown report."""

import argparse
import csv
from datetime import datetime, time, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys
import tempfile


METADATA_FILE = "run_metadata.json"
SAMPLES_FILE = "stream_density_latency_samples.jsonl"
DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "benchmark"


def read_json(path):
    with Path(path).open(encoding="utf-8") as source:
        return json.load(source)


def active_stages(camera, pipeline):
    workloads = set()
    for entry in camera.get("lane_config", {}).get("cameras", []):
        names = entry.get("workloads", [])
        workloads.update([names] if isinstance(names, str) else names)
    return {
        name: stages for name, stages in pipeline.get("workload_pipeline_map", {}).items()
        if name in workloads
    }


def configuration_id(camera, pipeline, settings):
    stages = {
        name: [{key: value for key, value in stage.items()
                if key not in ("device", "vlm_device")} for stage in entries]
        for name, entries in active_stages(camera, pipeline).items()
    }
    payload = json.dumps([camera, stages, settings], sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def pci_hardware(root=Path("/sys/bus/pci/devices")):
    devices = []
    for path in sorted(root.glob("*")):
        try:
            device_class = (path / "class").read_text().strip()
            if not device_class.startswith(("0x03", "0x0b", "0x12")):
                continue
            vendor = (path / "vendor").read_text().strip()
            device = (path / "device").read_text().strip()
            driver = (path / "driver").resolve().name if (path / "driver").exists() else "unknown"
            devices.append(f"PCI {path.name}: vendor={vendor}, device={device}, class={device_class}, driver={driver}")
        except OSError:
            continue
    return devices


def capture_run(results_dir, repo_root, env, settings):
    """Snapshot inputs before execution; never infer historical data from the host."""
    configs = Path(repo_root) / "configs"
    camera = read_json(configs / env.get("CAMERA_STREAM", "camera_to_workload.json"))
    pipeline = read_json(configs / env.get("WORKLOAD_DIST", "workload_to_pipeline.json"))
    stages = active_stages(camera, pipeline)
    devices = sorted({str(stage.get("device") or stage.get("vlm_device")).upper()
                      for entries in stages.values() for stage in entries
                      if stage.get("device") or stage.get("vlm_device")})
    cpu = platform.processor() or platform.machine()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.partition(":")[2].strip()
                break
    try:
        hardware = subprocess.run(
            ["lspci"], capture_output=True, text=True, check=False,
            timeout=10).stdout.splitlines()
        hardware = [line for line in hardware if any(
            kind in line.lower() for kind in ("vga", "display", "3d controller", "neural", "processing accelerators", "signal processing"))]
    except (OSError, subprocess.TimeoutExpired):
        hardware = []
    if not hardware:
        hardware = pci_hardware()
    if not hardware and any(device != "CPU" for device in devices):
        hardware = ["Accelerator hardware identity unavailable: neither PCI names nor sysfs IDs were accessible"]
    try:
        os_info = platform.freedesktop_os_release() if hasattr(platform, "freedesktop_os_release") else {}
    except OSError:
        os_info = {}
    metadata = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "config_id": configuration_id(camera, pipeline, settings),
        "accelerator": "+".join(devices) or env.get("DEVICE", "unknown"),
        "hardware": {"cpu": cpu, "accelerators": hardware},
        "os": os_info.get("PRETTY_NAME", platform.system()),
        "kernel": platform.release(),
        "camera_config": camera,
        "pipeline_config": pipeline,
        "settings": settings,
    }
    path = Path(results_dir) / METADATA_FILE
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def finish_run(results_dir, metadata):
    metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
    metadata["duration_seconds"] = (
        parse_date(metadata["finished_at"]) - parse_date(metadata["started_at"])).total_seconds()
    (Path(results_dir) / METADATA_FILE).write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def parse_date(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def parse_until(value):
    parsed = parse_date(value)
    return datetime.combine(parsed.date(), time.max, timezone.utc) if len(value) == 10 else parsed


def select_runs(paths, config_id=None, since=None, until=None):
    candidates = []
    for path in sorted({Path(entry).resolve() for entry in paths}):
        metadata_path = path / METADATA_FILE
        if not metadata_path.exists():
            print(f"WARNING: skipping {path}: no {METADATA_FILE}", file=sys.stderr)
            continue
        try:
            metadata = read_json(metadata_path)
            if not isinstance(metadata, dict):
                raise ValueError("metadata must be an object")
            if not metadata.get("finished_at"):
                continue
            started = parse_date(metadata["started_at"])
            finished = parse_date(metadata["finished_at"])
            for key in ("config_id", "accelerator", "hardware", "os", "kernel",
                        "camera_config", "pipeline_config", "settings"):
                if key not in metadata:
                    raise ValueError(f"missing {key}")
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
            print(f"WARNING: skipping {path}: invalid metadata: {error}", file=sys.stderr)
            continue
        if since and started < since:
            continue
        if until and started > until:
            continue
        candidates.append((finished, path, metadata))
    if not candidates:
        raise ValueError("No completed runs found in the selected directories/date range")
    candidates.sort(key=lambda entry: (entry[0], str(entry[1])))
    config_id = config_id or candidates[-1][2]["config_id"]
    selected = {}
    for _, path, metadata in candidates:
        if metadata["config_id"] == config_id:
            selected[metadata["accelerator"]] = (path, metadata)
    if not selected:
        raise ValueError(f"No completed runs for configuration {config_id}")
    return [selected[key] for key in sorted(selected)]


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def matches_accelerator(metric, accelerator):
    indexed = re.search(r"\bGPU_(\d+)\b", metric, re.IGNORECASE)
    for device in accelerator.upper().split("+"):
        if device == "GPU" and "GPU" in metric.upper():
            return True
        if device.startswith("GPU."):
            if indexed and indexed.group(1) == device.partition(".")[2]:
                return True
        elif device in ("CPU", "NPU") and device in metric.upper():
            return True
    return False


def snapshot_hardware_logs(results_dir):
    snapshots = {}
    for path in Path(results_dir).rglob("*"):
        if not path.is_file() or not re.search(
                r"(?:cpu_usage\.log|power_usage\.log|npu_usage\.csv|pcm\.csv|(?:qmassa.*(?:parsed|tool-generated)|xpum.*)\.json)$", path.name):
            continue
        try:
            stat = path.stat()
            snapshot = {"size": stat.st_size, "inode": stat.st_ino}
            if path.suffix == ".json":
                if path.name.endswith("tool-generated.json"):
                    from parse_qmassa_metrics_to_json import load_qmassa_json
                    data = load_qmassa_json(str(path)).get("states", [])
                else:
                    data = read_json(path)
                if not isinstance(data, list):
                    continue
                snapshot["records"] = data
            snapshots[str(path)] = snapshot
        except (OSError, ValueError):
            continue
    return snapshots


def window_hardware_metrics(start, end):
    from consolidate_multiple_run_of_metrics import KPIExtractor_OPTION

    metrics = {}
    with tempfile.TemporaryDirectory() as temporary:
        for filename, before in start.items():
            after = end.get(filename)
            if not after or before["inode"] != after["inode"] or after["size"] < before["size"]:
                continue
            path = Path(filename)
            target = Path(temporary) / path.name
            try:
                if path.suffix == ".json":
                    old_records = before["records"]
                    new_records = after["records"]
                    if new_records[:len(old_records)] != old_records:
                        continue
                    samples = new_records[len(old_records):]
                    if not samples:
                        continue
                    if path.name.endswith("tool-generated.json"):
                        from parse_qmassa_metrics_to_json import parse_qmassa_files
                        target.write_text(json.dumps({"states": samples}), encoding="utf-8")
                        parse_qmassa_files(temporary, path.stem)
                        target = target.with_name(target.name.replace("tool-generated", "parsed"))
                        if not target.exists():
                            continue
                    else:
                        target.write_text(json.dumps(samples), encoding="utf-8")
                else:
                    with path.open("rb") as source:
                        headers = b""
                        if path.suffix == ".csv":
                            headers = source.readline()
                            if path.name == "pcm.csv":
                                headers += source.readline()
                        boundary = before["size"]
                        source.seek(max(0, boundary - 1))
                        partial = boundary > 0 and source.read(1) != b"\n"
                        source.seek(boundary)
                        content = source.read(after["size"] - boundary)
                    lines = content.splitlines(keepends=True)
                    if partial:
                        lines = lines[1:]
                    content = b"".join(line for line in lines if line.endswith(b"\n"))
                    if not content:
                        continue
                    target.write_bytes(headers + content)
                for pattern, extractor in KPIExtractor_OPTION.items():
                    if re.search(pattern, target.name):
                        metrics.update(extractor().extract_data(str(target)))
                        break
            except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
                print(f"WARNING: Cannot extract window metrics from {path}: {error}", file=sys.stderr)
    return metrics


def run_metrics(path, metadata):
    records = []
    sample_path = path / SAMPLES_FILE
    if sample_path.exists():
        with sample_path.open(encoding="utf-8") as source:
            records = [json.loads(line) for line in source if line.strip()]
    sweeps = {record.get("container_name"): record for record in records
              if record.get("record_type") == "sweep_result"}
    confirmed = bool(sweeps) and all(record.get("status") == "pass" for record in sweeps.values())
    windows = {}
    for record in records:
        container = record.get("container_name")
        if (confirmed and "stream_fps" in record
                and record.get("pipeline_count") == sweeps.get(container, {}).get("lane_count")):
            windows[container] = record
    if windows.keys() != sweeps.keys():
        confirmed = False
        windows = {}
    fps = [value for record in windows.values()
           for raw in record.get("stream_fps", {}).values()
           if (value := number(raw)) is not None and value > 0]
    latency = [value for record in windows.values()
               for values in record.get("samples_by_stream", {}).values() for raw in values
               if (value := number(raw)) is not None and value >= 0]
    metric_path = path / "metrics.csv"
    if not metric_path.exists():
        from consolidate_multiple_run_of_metrics import consolidate_metrics
        with tempfile.TemporaryDirectory() as temporary:
            metric_path = Path(temporary) / "metrics.csv"
            consolidate_metrics(str(path), str(metric_path))
            metrics = read_metrics(metric_path)
    else:
        metrics = read_metrics(metric_path)
    accelerator = metadata["accelerator"]
    power = {}
    window_utilisation = {}
    for container, record in windows.items():
        stream_count = sum(number(raw) is not None and number(raw) > 0
                           for raw in record.get("stream_fps", {}).values())
        if not stream_count:
            continue
        for key, raw in record.get("hardware_metrics", {}).items():
            value = number(raw)
            if value is None:
                continue
            label = f"{container}: {key}" if len(windows) > 1 else key
            if (("Power Draw W" in key or "GPU Power (W)" in key)
                    and ("GPU" not in key.upper() or matches_accelerator(key, accelerator))):
                power[label] = value / stream_count
            if "Utilization %" in key and "Memory" not in key and matches_accelerator(key, accelerator):
                window_utilisation[label] = value
    utilisation = {key: value for key, raw in metrics.items()
                   if "Utilization %" in key and "Memory" not in key
                   and matches_accelerator(key, accelerator)
                   and (value := number(raw)) is not None}
    return {
        "status": "pass" if confirmed and fps else "no confirmed density",
        "streams": len(fps) if fps else None,
        "throughput": sum(fps) if fps else None,
        "rate": statistics.mean(fps) if fps else None,
        "latency": statistics.mean(latency) if latency else None,
        "power": power,
        "utilisation": window_utilisation or utilisation,
        "utilisation_period": "confirming window" if window_utilisation else "collector period (not window-aligned)",
    }


def read_metrics(path):
    with path.open(newline="", encoding="utf-8") as source:
        return {row[0]: row[1] for row in csv.reader(source) if len(row) >= 2}


def cell(value):
    if value is None or value == "":
        return "N/A"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def labelled(values):
    return "; ".join(f"{cell(key)}: {cell(value)}" for key, value in values.items()) or "N/A"


def generate_report(runs):
    metadata = runs[0][1]
    about = metadata["camera_config"].get("_about", metadata["camera_config"].get("about", {}))
    lines = ["# Stream Density Accelerator Comparison", "",
             f"Configuration: {metadata['config_id']}", ""]
    if isinstance(about, dict):
        lines.extend(f"**{cell(key)}:** {cell(value)}" for key, value in about.items())
    elif about:
        lines.append(cell(about))
    lines.extend(["", "## Comparison", "",
                  "| Accelerator | Source run | Duration (s) | Max confirmed streams | Throughput (FPS) | FPS/stream | Mean latency (ms) | Power/stream (W, source scope) | Utilisation (%, period) | Result |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"])
    for path, run in runs:
        metrics = run_metrics(path, run)
        values = [run["accelerator"], path.name, run.get("duration_seconds"),
                  metrics["streams"], metrics["throughput"], metrics["rate"],
                  metrics["latency"], labelled(metrics["power"]),
                  f"{labelled(metrics['utilisation'])} ({metrics['utilisation_period']})", metrics["status"]]
        lines.append("| " + " | ".join(cell(value) for value in values) + " |")
    lines.extend(["", "FPS and latency use the final confirming measurement window. N/A means not recorded.",
                  "Power per stream uses only confirming-window samples and retains the collector's scope; package power is not accelerator-only power.",
                  "N/A power means no aligned power samples were saved; whole-run averages are not substituted.", ""])
    for path, run in runs:
        lines.extend([f"## {cell(run['accelerator'])}", "",
                      f"Source directory: {cell(path)}", "",
                      f"Started: {cell(run['started_at'])}; finished: {cell(run['finished_at'])}", "",
                      f"Hardware: {cell(run['hardware']['cpu'])}"])
        lines.extend(cell(device) for device in run["hardware"].get("accelerators", []))
        lines.extend([f"OS: {cell(run['os'])}", f"Kernel: {cell(run['kernel'])}", "",
                      "### Pipeline Stages", "",
                      "| Workload | Stage | Model | Device | Precision |",
                      "| --- | --- | --- | --- | --- |"])
        for workload, stages in active_stages(run["camera_config"], run["pipeline_config"]).items():
            for stage in stages:
                values = [workload, stage.get("type"), stage.get("model"),
                          stage.get("device") or stage.get("vlm_device"), stage.get("precision")]
                lines.append("| " + " | ".join(cell(value) for value in values) + " |")
        lines.extend(["", "### Test Configuration", "", "```json",
                      json.dumps({"camera": run["camera_config"], "settings": run["settings"]}, indent=2),
                      "```", ""])
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="*", type=Path, help="Explicit run directories")
    parser.add_argument("--results-dir", type=Path,
                        default=Path(os.environ.get("RESULTS_DIR", DEFAULT_ROOT)))
    parser.add_argument("--config", help="Configuration ID from run_metadata.json")
    parser.add_argument("--since", type=parse_date, help="Inclusive ISO date/time (UTC by default)")
    parser.add_argument("--until", type=parse_until, help="Inclusive ISO date/time; a date includes the whole UTC day")
    parser.add_argument("--output", type=Path, help="Markdown output; default: results-dir/stream_density_report.md")
    args = parser.parse_args(argv)
    if args.since and args.until and args.since > args.until:
        parser.error("--since must not be after --until")
    try:
        paths = args.runs or [path.parent for path in args.results_dir.rglob(METADATA_FILE)]
        runs = select_runs(paths, args.config, args.since, args.until)
        report = generate_report(runs)
        output = args.output or args.results_dir / "stream_density_report.md"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report, encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Report generation failed: {error}\n")
    print(f"Report: {output}")


if __name__ == "__main__":
    main()