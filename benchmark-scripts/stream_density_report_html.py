"""Map recorded benchmark data to the ASC presentation with issue #249 changes."""

from datetime import datetime
from html import escape
import json
from pathlib import Path
import re
import statistics
import tempfile

import asc_benchmark_report as presentation
from compare_stream_density import (
    SAMPLES_FILE, active_stages, cell, hardware_details, labelled, number,
    run_metrics, test_configuration,
)


def text(value):
    return escape(cell(value))


def formatted(value, pattern="{:.1f}", unit=""):
    return pattern.format(value) + unit if value is not None else "N/A"


def hardware_target(target):
    target = target.upper()
    if "+" in target or "HETERO" in target:
        return "HETERO"
    return next((name for name in ("CPU", "GPU", "NPU") if name in target), target)


def sample_records(path):
    sample_path = path / SAMPLES_FILE
    if not sample_path.exists():
        return []
    with sample_path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def presentation_metrics(path, metadata, metrics):
    package = sum(metrics["package_power"].values()) if metrics["package_power"] else None
    utilisation = metrics["interval_utilisation"]

    def average(values):
        return statistics.mean(values) if values else None

    cpu = [value for key, value in utilisation.items() if "CPU" in key.upper()]
    gpu = [value for key, value in utilisation.items() if "GPU" in key.upper()]
    npu = [value for key, value in utilisation.items() if "NPU" in key.upper()]
    environment = metadata.get("settings", {}).get("environment", {})
    if any(path.glob("**/qmassa*.json")):
        power_source = "qmassa"
    elif any(path.glob("**/xpum*.json")):
        power_source = "XPUM"
    elif str(environment.get("WSL2", "")).lower() == "true":
        power_source = "Windows metrics"
    else:
        power_source = "collector not recorded"
    return {
        "stream_density": metrics["streams"], "total_fps": metrics["throughput"],
        "avg_fps": metrics["rate"],
        **{key: metrics[key] for key in ("latency_p50", "latency_p95", "latency_p99",
                           "highest_stream_p95", "status")},
        "last_passing": text(labelled(metrics["last_passing_lanes"]))
        if metrics["last_passing_lanes"] else None,
        "cpu_util_pct": average(cpu), "gpu_engine_pct": average(gpu),
        "npu_util_pct": average(npu), "proc_power_w": package,
        "gpu_power_reported": text(f'{power_source}: {labelled(metrics["gpu_power"])}')
        if metrics["gpu_power"] else "N/A",
        "power_per_stream_w": package / metrics["streams"]
        if package is not None and metrics["streams"] else None,
        "power_scope": text(labelled({
            f'{key} ({power_source})' if "GPU" in key.upper() else key: value
            for key, value in metrics["power"].items()})),
        "utilisation_scope": text(labelled(utilisation)),
    }


def fps_progression(path):
    records = [record for record in sample_records(path)
               if record.get("record_type") == "measurement_interval"]
    identities = {(record.get("container_name", "unknown"), name)
                  for record in records for name in record.get("stream_fps", {})}

    def stream_index(identity):
        match = re.fullmatch(r"(?:pipeline_)?stream(\d+)", identity[1])
        return int(match.group(1)) if match else None

    ordered = sorted(identities, key=lambda identity: (
        identity[0], stream_index(identity) is None,
        stream_index(identity) if stream_index(identity) is not None else identity[1]))
    single_container = len({container for container, _ in identities}) == 1
    indices = {}
    next_index = max((stream_index(identity) for identity in identities
                      if stream_index(identity) is not None), default=-1) + 1
    for index, identity in enumerate(ordered):
        numeric_index = stream_index(identity)
        if not single_container:
            indices[identity] = index
        elif numeric_index is not None:
            indices[identity] = numeric_index
        else:
            indices[identity] = next_index
            next_index += 1
    return [
        {"n_streams": len(record.get("stream_fps", {})),
         "fps_map": {indices[(record.get("container_name", "unknown"), name)]: value
                     for name, raw in record.get("stream_fps", {}).items()
                     if (value := number(raw)) is not None},
         "status": record.get("status")}
        for record in records
        if any(number(raw) is not None for raw in record.get("stream_fps", {}).values())
    ]


def run_timing(metadata):
    duration = number(metadata.get("duration_seconds"))
    duration_text = "N/A"
    if duration is not None:
        hours, remainder = divmod(int(duration), 3600)
        minutes, seconds = divmod(remainder, 60)
        duration_text = (f"{hours}h {minutes:02d}m {seconds:02d}s" if hours
                         else f"{minutes}m {seconds:02d}s")

    def timestamp(value):
        if value:
            return text(datetime.fromisoformat(value).strftime("%Y-%m-%d %H:%M:%S"))
        return "N/A"

    return {"start": timestamp(metadata.get("started_at")),
            "end": timestamp(metadata.get("finished_at")), "duration_str": duration_text}


def pipeline_steps(metadata):
    steps = []
    for pipeline in active_stages(metadata["camera_config"], metadata["pipeline_config"]).values():
        for stage in pipeline:
            if stage.get("type") in ("gvadetect", "gvaclassify", "gvainference"):
                steps.append({key: text(stage.get(key))
                              for key in ("type", "device", "model", "precision")})
            if stage.get("vlm_model"):
                steps.append({"type": "vlm", "model": text(stage["vlm_model"]),
                              "device": text(stage.get("vlm_device")),
                              "precision": text(stage.get("vlm_precision"))})
    return steps


def system_info(metadata):
    hardware = metadata["hardware"]
    details = hardware_details(metadata)
    return {
        "hostname": text(details["Host"]), "cpu_model": text(details["CPU"]),
        "system_vendor": text(hardware.get("system_vendor")),
        "system_model": text(hardware.get("system_model")),
        "cpu_cores": text(hardware.get("cpu_cores")),
        "logical_cpus": text(hardware.get("logical_cpus")),
        "total_ram_gb": formatted(details["RAM (GiB)"], "{:.0f}", " GB"),
        "gpu_model": text(details["GPU"]), "npu_model": text(details["NPU"]),
        "npu_driver": text(hardware.get("npu_driver")),
        "mem_channels": text(hardware.get("mem_channels")),
        "mem_per_dimm": text(hardware.get("mem_per_dimm")),
        "mem_speed": text(hardware.get("mem_speed")),
        "pl1_w": formatted(details["PL1 power limit (W)"], "{:.2f}", " W"),
        "pl2_w": formatted(details["PL2 power limit (W)"], "{:.2f}", " W"),
        "os_name": text(metadata.get("os")), "kernel": text(metadata.get("kernel")),
        "arch": text(hardware.get("arch")),
    }


def generate_html_report(runs):
    all_kpis = []
    for path, metadata in runs:
        settings = test_configuration(metadata)
        config_file = (metadata.get("camera_config_file") or
                   metadata.get("settings", {}).get("environment", {}).get("CAMERA_STREAM"))
        config_file = Path(config_file).name if config_file else None
        rows = [
            ("Test configuration", config_file),
            ("Target FPS per camera (fps)", settings["Target FPS per camera"]),
            ("Effective target FPS per camera (fps)", settings["Effective target FPS per camera"]),
            ("Throughput threshold (fps)", settings["Throughput threshold (FPS)"]),
            ("Warmup period (s)", settings["Warmup period (s)"]),
            ("Measurement interval (s)", settings["Measurement interval (s)"]),
            ("Run acceptance criterion", settings["Run acceptance criterion"]),
            ("Consecutive failing intervals", settings["Consecutive failing intervals"]),
            ("Run duration", run_timing(metadata)["duration_str"]),
            ("Source run", str(path)),
        ]
        all_kpis.append({
            **presentation_metrics(path, metadata, run_metrics(path, metadata)),
            "hw": text(metadata["accelerator"]),
            "color_target": hardware_target(metadata["accelerator"]).lower(),
            "source_run": text(str(path)),
            "timing": run_timing(metadata), "pipeline_steps": pipeline_steps(metadata),
            "fps_progression": fps_progression(path),
            "target_fps": number(settings["Effective target FPS per camera"]),
            "software_rows": [(text(label), text(value)) for label, value in rows],
            "system_info": system_info(metadata),
        })
    first_path, first_metadata = runs[0]
    config = first_metadata.get("settings", {}).get("environment", {}).get("CAMERA_STREAM", "")
    about = first_metadata["camera_config"].get("_about", {})
    if not isinstance(about, dict):
        about = {"Description": about}
    description = "".join(
        f'<p><b>{text(key.replace("_", " ").capitalize())}:</b> {text(value)}</p>'
        for key, value in about.items() if value)
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / text(first_path.parent.name)
        output.mkdir()
        presentation.plot_charts(all_kpis, str(output))
        for kpis in all_kpis:
            hardware = kpis["hw"].lower()
            presentation.plot_fps_heatmap(
                kpis["fps_progression"], hardware,
                str(output / presentation.HEATMAP_FNAME.format(hw=hardware)), target_fps=kpis["target_fps"])
        report_path = presentation.generate_html_report(
            all_kpis, str(output), system_info(first_metadata),
            test_config=text(config) if config else "", description_html=description)
        return Path(report_path).read_text(encoding="utf-8")