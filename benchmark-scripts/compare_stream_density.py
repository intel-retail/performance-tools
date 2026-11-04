"""Capture run provenance and generate cross-accelerator HTML and Markdown reports."""

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


def gpu_metric_devices():
    mapping = {}
    try:
        from openvino import Core
        core = Core()
        cards = {path.resolve().name.lower(): path.parent.name.removeprefix("card")
                 for path in Path("/sys/class/drm").glob("card[0-9]*/device")}
        for device in core.available_devices:
            if device == "GPU":
                device = "GPU.0"
            if not device.startswith("GPU."):
                continue
            try:
                pci = core.get_property(device, "DEVICE_PCI_INFO")
                address = (pci if isinstance(pci, str) else
                           f"{pci.domain:04x}:{pci.bus:02x}:{pci.device:02x}.{pci.function:x}")
                if address.lower() in cards:
                    mapping[device] = f"GPU_{cards[address.lower()]}"
            except (RuntimeError, AttributeError, TypeError, ValueError):
                continue
    except (ImportError, RuntimeError, OSError):
        pass
    return mapping


def read_power_limits(powercap_dir=Path("/sys/class/powercap/intel-rapl:0")):
    limits = {}
    for constraint in (0, 1):
        key = f"constraint_{constraint}_power_limit_uw"
        try:
            value = int((Path(powercap_dir) / key).read_text().strip())
        except (OSError, ValueError):
            continue
        limits[key] = value
    return limits


def read_memory_details(edac_root=Path("/sys/devices/system/edac/mc")):
    devices = []
    source = None
    try:
        result = subprocess.run(["dmidecode", "--type", "17"], capture_output=True,
                                text=True, check=False, timeout=10)
        output = result.stdout if result.returncode == 0 else ""
        if output:
            for block in re.split(r"\n\s*\n", output):
                fields = dict(line.strip().split(":", 1) for line in block.splitlines()
                              if ":" in line)
                fields = {key: value.strip() for key, value in fields.items()}
                size = fields.get("Size", "")
                if not re.fullmatch(r"\d+\s+(?:MB|GB|TB)", size):
                    continue
                speed = fields.get("Configured Memory Speed", fields.get("Configured Clock Speed", ""))
                if not re.fullmatch(r"\d+\s+(?:MT/s|MHz)", speed):
                    speed = None
                devices.append({"size": size, "speed": speed,
                                "locator": fields.get("Locator"),
                                "bank_locator": fields.get("Bank Locator")})
            if devices:
                source = "dmidecode type 17"
    except (OSError, subprocess.TimeoutExpired):
        pass
    channels = set()
    if devices:
        for device in devices:
            location = f'{device.get("bank_locator") or ""} {device.get("locator") or ""}'
            channel = re.search(r"\bchannel[\s_-]*([a-z0-9]+)\b", location, re.IGNORECASE)
            if channel:
                socket = re.search(r"\b(?:cpu|p|socket)[\s_-]*(\d+)\b", location, re.IGNORECASE)
                channels.add((socket.group(1) if socket else "0", channel.group(1).upper()))
    else:
        for dimm in sorted(Path(edac_root).glob("mc*/dimm*")):
            try:
                size_mb = int((dimm / "size").read_text().strip())
                if size_mb <= 0:
                    continue
                location = (dimm / "dimm_location").read_text().strip()
                channel = re.search(r"\bchannel\s+(\d+)\b", location)
                if channel:
                    channels.add((dimm.parent.name, channel.group(1)))
                size = f"{size_mb // 1024} GB" if size_mb % 1024 == 0 else f"{size_mb} MB"
                devices.append({"size": size, "speed": None,
                                "locator": location, "bank_locator": dimm.parent.name})
            except (OSError, ValueError):
                continue
        if devices:
            source = "EDAC sysfs"
    sizes = sorted({device["size"] for device in devices})
    speeds = sorted({device["speed"] for device in devices if device["speed"]})
    edac = source == "EDAC sysfs"
    return {"memory_devices": devices, "memory_info_source": source,
            "mem_channels": f"{len(channels)}-channel" + (" (EDAC logical)" if edac else "")
            if channels else None,
            "mem_per_dimm": "; ".join(sizes) + (" (EDAC device)" if edac else "") if sizes else None,
            "mem_speed": "; ".join(speeds) or None}


def read_npu_driver(module_root=Path("/sys/module/intel_vpu")):
    module_root = Path(module_root)
    try:
        version = (module_root / "version").read_text().strip()
        if version:
            return f"intel_vpu v{version.split()[0]}"
    except OSError:
        pass
    if module_root.is_dir():
        return f"intel_vpu (kernel {platform.release()}; module version unavailable)"
    return None


def camera_targets(camera):
    targets = {}
    for index, entry in enumerate(camera.get("lane_config", {}).get("cameras", [])):
        value = next((entry[key] for key in ("targetFps", "fps")
                      if number(entry.get(key)) is not None and number(entry[key]) > 0), None)
        targets[str(entry.get("camera_id", f"camera_{index + 1}"))] = value
    return targets


def effective_targets(targets, settings, environment, fallback=None):
    explicit = settings.get("target_fps") or []
    containers = settings.get("container_names") or []
    if explicit:
        if len(explicit) == 1:
            return {name: explicit[0] for name in targets}
        return {f"{container} / {name}": target
                for container, target in zip(containers, explicit) for name in targets}
    override = number(environment.get("TARGET_FPS"))
    if override is not None and override > 0:
        return {name: override for name in targets}
    return {name: value if value is not None else fallback for name, value in targets.items()}


def capture_run(results_dir, repo_root, env, settings):
    """Snapshot inputs before execution; never infer historical data from the host."""
    configs = Path(repo_root) / "configs"
    camera = read_json(configs / env.get("CAMERA_STREAM", "camera_to_workload.json"))
    pipeline = read_json(configs / env.get("WORKLOAD_DIST", "workload_to_pipeline.json"))
    stages = active_stages(camera, pipeline)
    import psutil
    import stream_density

    settings = dict(settings)
    for key, environment_key, default, convert in (
            ("pass_tolerance_ratio", stream_density.PASS_TOLERANCE_RATIO_KEY,
             stream_density.DEFAULT_PASS_TOLERANCE_RATIO, float),
            ("consecutive_passes", stream_density.CONSECUTIVE_PASS_WINDOWS_KEY,
             stream_density.DEFAULT_CONSECUTIVE_PASS_WINDOWS, int),
            ("consecutive_fails", stream_density.CONSECUTIVE_FAIL_WINDOWS_KEY,
             stream_density.DEFAULT_CONSECUTIVE_FAIL_WINDOWS, int)):
        settings[key] = convert(env.get(environment_key) or default)
    settings["camera_target_fps"] = camera_targets(camera)
    settings["effective_target_fps"] = effective_targets(
        settings["camera_target_fps"], settings, env, stream_density.DEFAULT_TARGET_FPS)
    devices = sorted({str(device).upper()
                      for entries in stages.values() for stage in entries
                      for device in (stage.get("device"), stage.get("vlm_device"))
                      if device})
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
    accelerator_names = {"gpu": [], "npu": []}
    for line in hardware:
        if any(kind in line.lower() for kind in ("vga", "display", "3d controller")):
            accelerator_names["gpu"].append(line)
        if any(kind in line.lower() for kind in ("neural", "processing accelerators", "signal processing")):
            accelerator_names["npu"].append(line)
    try:
        from openvino import Core
        core = Core()
        for device in core.available_devices:
            kind = device.split(".")[0].lower()
            if kind in accelerator_names:
                try:
                    name = str(core.get_property(device, "FULL_DEVICE_NAME"))
                    accelerator_names[kind].append(f"{device}: {name}")
                except RuntimeError:
                    continue
    except (ImportError, RuntimeError, OSError):
        pass
    system = {}
    for key, filename in (("vendor", "sys_vendor"), ("model", "product_name")):
        try:
            system[key] = (Path("/sys/class/dmi/id") / filename).read_text().strip() or None
        except OSError:
            system[key] = None
    try:
        os_info = platform.freedesktop_os_release() if hasattr(platform, "freedesktop_os_release") else {}
    except OSError:
        os_info = {}
    metadata = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "config_id": configuration_id(camera, pipeline, settings),
        "accelerator": "+".join(devices) or env.get("DEVICE", "unknown"),
        "hardware": {
            "cpu": cpu, "accelerators": hardware,
            "hostname": platform.node(), "system_vendor": system["vendor"],
            "system_model": system["model"], "cpu_cores": psutil.cpu_count(logical=False),
            "logical_cpus": psutil.cpu_count(), "ram_bytes": psutil.virtual_memory().total,
            "gpu_names": accelerator_names["gpu"], "npu_names": accelerator_names["npu"],
            "arch": platform.machine() or None,
            "npu_driver": read_npu_driver(),
            **read_memory_details(),
            **read_power_limits(),
        },
        "gpu_metric_devices": gpu_metric_devices(),
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
            hardware = metadata["hardware"]
            if (not isinstance(hardware, dict)
                    or not isinstance(hardware.get("cpu"), str)
                    or not hardware["cpu"].strip()
                    or not isinstance(hardware.get("accelerators", []), list)):
                raise ValueError("hardware must contain a CPU identity and an accelerator list")
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


def matches_accelerator(metric, accelerator, gpu_devices=None):
    indexed = re.search(r"\bGPU_(\d+)\b", metric, re.IGNORECASE)
    for device in accelerator.upper().split("+"):
        if device == "GPU" and "GPU" in metric.upper():
            return True
        if device.startswith("GPU."):
            if indexed and (gpu_devices or {}).get(device) == f"GPU_{indexed.group(1)}":
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
        if (confirmed and record.get("record_type") != "measurement_interval"
            and "stream_fps" in record
                and record.get("pipeline_count") == sweeps.get(container, {}).get("lane_count")):
            windows[container] = record
    if windows.keys() != sweeps.keys():
        confirmed = False
        windows = {}
    fps = [value for record in windows.values()
           for raw in record.get("stream_fps", {}).values()
           if (value := number(raw)) is not None and value > 0]
    metric_path = path / "metrics.csv"
    if not metric_path.exists():
        from consolidate_multiple_run_of_metrics import consolidate_metrics
        with tempfile.TemporaryDirectory() as temporary:
            metric_path = Path(temporary) / "metrics.csv"
            consolidate_metrics(str(path), str(metric_path))
            metrics = read_metrics(metric_path)
    else:
        metrics = read_metrics(metric_path)
    from consolidate_multiple_run_of_metrics import build_window_latency_metrics

    latency_metrics = build_window_latency_metrics(list(windows.values())) if windows else {}
    latency_metrics.update(metrics)
    percentiles = {
        f"latency_p{percentile}": number(latency_metrics.get(
            f"System Latency p{percentile} (ms)")) if confirmed else None
        for percentile in (50, 95, 99)
    }
    stream_p95 = latency_metrics.get("Per-stream Latency p95 (ms)", {})
    if isinstance(stream_p95, str):
        stream_p95 = {
            label: value for entry in stream_p95.split(", ")
            if ": " in entry for label, value in [entry.rsplit(": ", 1)]
        }
    highest_p95 = max((value for raw in stream_p95.values()
                       if (value := number(raw)) is not None), default=None) if confirmed else None
    statuses = {record.get("status") for record in sweeps.values()}
    status = ("fail" if "fail" in statuses else
              "inconclusive" if "inconclusive" in statuses else
              "pass" if statuses == {"pass"} else "unknown")
    accelerator = metadata["accelerator"]
    gpu_devices = metadata.get("gpu_metric_devices", {})
    power = {}
    gpu_power = {}
    package_power = {}
    window_utilisation = {}
    interval_utilisation = {}
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
            if "Power Draw W" in key and "GPU" not in key.upper():
                package_power[label] = value
            if ("GPU Power (W)" in key
                    and matches_accelerator(key, accelerator, gpu_devices)):
                gpu_power[label] = value
            if (("Power Draw W" in key or "GPU Power (W)" in key)
                    and ("GPU" not in key.upper() or matches_accelerator(key, accelerator, gpu_devices))):
                power[label] = value / stream_count
            if "Utilization %" in key and "Memory" not in key:
                if "GPU" not in key.upper() or matches_accelerator(key, accelerator, gpu_devices):
                    interval_utilisation[label] = value
                if matches_accelerator(key, accelerator, gpu_devices):
                    window_utilisation[label] = value
    utilisation = {key: value for key, raw in metrics.items()
                   if "Utilization %" in key and "Memory" not in key
                   and matches_accelerator(key, accelerator, gpu_devices)
                   and (value := number(raw)) is not None}
    return {
        "status": status,
        "confirmed": bool(confirmed and fps),
        "last_passing_lanes": {
            container: record.get("lane_count") for container, record in sweeps.items()
            if record.get("status") == "inconclusive" and record.get("lane_count", 0) > 0
        },
        "streams": len(fps) if fps else None,
        "throughput": sum(fps) if fps else None,
        "rate": statistics.mean(fps) if fps else None,
        **percentiles,
        "highest_stream_p95": highest_p95,
        "power": power,
        "gpu_power": gpu_power,
        "package_power": package_power,
        "interval_utilisation": interval_utilisation,
        "utilisation": window_utilisation or utilisation,
        "utilisation_period": "measurement interval" if window_utilisation else "collector period (not interval-aligned)",
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


def test_configuration(metadata):
    settings = metadata["settings"]
    environment = settings.get("environment", {})
    targets = settings.get("camera_target_fps") or camera_targets(metadata["camera_config"])
    resolved = settings.get("effective_target_fps") or effective_targets(
        targets, settings, environment)
    ratio = number(settings.get("pass_tolerance_ratio", environment.get("PASS_TOLERANCE_RATIO")))
    thresholds = {name: value * ratio if ratio is not None and
                  (value := number(raw)) is not None else None
                  for name, raw in resolved.items()}

    def compact(values):
        unique = {cell(value) for value in values.values()}
        return next(iter(unique)) if len(unique) == 1 else labelled(values)

    passes = settings.get("consecutive_passes", environment.get("CONSECUTIVE_PASS_WINDOWS"))
    return {
        "Target FPS per camera": compact(targets),
        "Effective target FPS per camera": compact(resolved),
        "Throughput threshold (FPS)": compact(thresholds),
        "Pass tolerance ratio": ratio,
        "Warmup period (s)": settings.get("init_duration"),
        "Measurement interval (s)": settings.get("measurement_window_seconds"),
        "Run acceptance criterion": (
            "two consecutive passes" if str(passes) == "2" else
            f"{passes} consecutive passes" if passes is not None else None),
        "Consecutive failing intervals": settings.get(
            "consecutive_fails", environment.get("CONSECUTIVE_FAIL_WINDOWS")),
    }


def hardware_details(metadata):
    hardware = metadata["hardware"]
    ram = number(hardware.get("ram_bytes"))
    pl1 = number(hardware.get("constraint_0_power_limit_uw"))
    pl2 = number(hardware.get("constraint_1_power_limit_uw"))
    return {
        "Host": hardware.get("hostname"), "System vendor": hardware.get("system_vendor"),
        "System model": hardware.get("system_model"), "CPU": hardware["cpu"],
        "CPU cores (physical)": hardware.get("cpu_cores"),
        "Logical CPUs": hardware.get("logical_cpus"),
        "RAM (GiB)": ram / 1024 ** 3 if ram is not None else None,
        "GPU": "; ".join(hardware.get("gpu_names", [])) or None,
        "NPU": "; ".join(hardware.get("npu_names", [])) or None,
        "PL1 power limit (W)": pl1 / 1_000_000 if pl1 is not None else None,
        "PL2 power limit (W)": pl2 / 1_000_000 if pl2 is not None else None,
        "Accelerator identities": "; ".join(hardware.get("accelerators", [])) or None,
    }


def density_result(metrics):
    if metrics["confirmed"]:
        return cell(metrics["streams"])
    if metrics["last_passing_lanes"]:
        return f"N/A; last passing lanes: {labelled(metrics['last_passing_lanes'])} (not confirmed)"
    return "N/A"


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
                  "| Accelerator | Source run | Duration (s) | Stream density (streams) | Throughput (FPS) | FPS/stream | Pooled latency p50 (ms) | Pooled latency p95 (ms) | Pooled latency p99 (ms) | Highest per-stream p95 (ms) | Power/stream (W, source scope) | Utilisation (%, period) | Result |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"])
    for path, run in runs:
        metrics = run_metrics(path, run)
        values = [run["accelerator"], path.name, run.get("duration_seconds"),
                  density_result(metrics), metrics["throughput"], metrics["rate"],
                  metrics["latency_p50"], metrics["latency_p95"], metrics["latency_p99"],
                  metrics["highest_stream_p95"], labelled(metrics["power"]),
                  f"{labelled(metrics['utilisation'])} ({metrics['utilisation_period']})", metrics["status"]]
        lines.append("| " + " | ".join(cell(value) for value in values) + " |")
    lines.extend(["", "FPS and latency use the final confirmed measurement interval. N/A means not recorded or not confirmed.",
                  "Power per stream uses only measurement interval samples and retains the collector's scope; package power is not accelerator-only power.",
                  "N/A power means no aligned power samples were saved; whole-run averages are not substituted.", ""])
    for path, run in runs:
        lines.extend([f"## {cell(run['accelerator'])}", "",
                      f"Source directory: {cell(path)}", "",
                      f"Started: {cell(run['started_at'])}; finished: {cell(run['finished_at'])}", "",
                      f"Hardware: {cell(run['hardware']['cpu'])}"])
        lines.extend(cell(device) for device in run["hardware"].get("accelerators", []))
        lines.extend(f"{key}: {cell(value)}" for key, value in hardware_details(run).items())
        lines.extend(["", "### Test & Software", ""])
        lines.extend(f"{key}: {cell(value)}" for key, value in test_configuration(run).items())
        lines.extend([f"OS: {cell(run['os'])}", f"Kernel: {cell(run['kernel'])}", "",
                      "### Pipeline Stages", "",
                      "| Workload | Stage | Model | Device | Precision |",
                      "| --- | --- | --- | --- | --- |"])
        for workload, stages in active_stages(run["camera_config"], run["pipeline_config"]).items():
            for stage in stages:
                values = [workload, stage.get("type"), stage.get("model"),
                          stage.get("device") or stage.get("vlm_device"), stage.get("precision")]
                lines.append("| " + " | ".join(cell(value) for value in values) + " |")
                if stage.get("vlm_model"):
                    values = [workload, "vlm", stage["vlm_model"],
                              stage.get("vlm_device"), stage.get("vlm_precision")]
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
    parser.add_argument("--output", type=Path, help="Output .md or .html path; both formats are written with the same stem")
    args = parser.parse_args(argv)
    if args.since and args.until and args.since > args.until:
        parser.error("--since must not be after --until")
    try:
        paths = args.runs or [path.parent for path in args.results_dir.rglob(METADATA_FILE)]
        runs = select_runs(paths, args.config, args.since, args.until)
        report = generate_report(runs)
        output = args.output or args.results_dir / "stream_density_report.md"
        if output.suffix.lower() not in (".md", ".html"):
            raise ValueError("Output must use a .md or .html extension")
        output.parent.mkdir(parents=True, exist_ok=True)
        from stream_density_report_html import generate_html_report
        html = generate_html_report(runs)
        output.with_suffix(".md").write_text(report, encoding="utf-8")
        output.with_suffix(".html").write_text(html, encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Report generation failed: {error}\n")
    print(f"Report: {output.with_suffix('.html')}")
    print(f"Markdown: {output.with_suffix('.md')}")


if __name__ == "__main__":
    main()