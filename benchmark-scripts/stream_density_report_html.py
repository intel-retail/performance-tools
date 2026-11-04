"""Self-contained accelerator comparison report with embedded Matplotlib charts."""

import base64
from html import escape
from io import BytesIO
import json

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
import numpy as np

from compare_stream_density import (
    SAMPLES_FILE, active_stages, cell, density_result, hardware_details,
    labelled, number, run_metrics, test_configuration,
)


def text(value):
    return escape(cell(value))


def table(headers, rows, css_class=""):
    head = "".join(f"<th scope='col'>{text(header)}</th>" for header in headers)
    body = "".join("<tr>" + "".join(f"<td>{text(value)}</td>" for value in row) + "</tr>"
                   for row in rows)
    return f"<div class='table-scroll'><table class='{css_class}'><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def image(figure, title):
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=130, bbox_inches="tight")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    figure.clear()
    return f"<figure><figcaption>{text(title)}</figcaption><img alt='{text(title)}' src='data:image/png;base64,{encoded}'></figure>"


def comparison_chart(title, unit, entries, series, horizontal=False):
    figure = Figure(figsize=(6, 3.5) if horizontal else (7, 4.5), layout="constrained")
    axis = figure.subplots()
    positions = np.arange(len(entries))
    width = 0.65 / max(len(series), 1)
    hw_colors = {"CPU": "#4e79a7", "GPU": "#f28e2b", "NPU": "#59a14f", "HETERO": "#b07aa1"}
    series_colors = ("#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2")
    targets = [run["accelerator"] for _, run, _ in entries]
    all_values = [value for _, values in series for raw in values
                  if (value := number(raw)) is not None]
    for index, (label, values) in enumerate(series):
        offsets = positions + (index - (len(series) - 1) / 2) * width
        numeric = [number(value) for value in values]
        colors = ([next((color for name, color in hw_colors.items()
                         if name in target.upper()), "#888888") for target in targets]
                  if len(series) == 1 else series_colors[index % len(series_colors)])
        if horizontal:
            bars = axis.barh(offsets, [value if value is not None else 0 for value in numeric],
                             height=width * 0.9, label=label, color=colors,
                             edgecolor="white", linewidth=0.5)
            for bar, value in zip(bars, numeric):
                if value is not None:
                    axis.annotate(f"{value:.1f}",
                                  (bar.get_width(), bar.get_y() + bar.get_height() / 2),
                                  xytext=(4, 0), textcoords="offset points",
                                  ha="left", va="center", fontsize=8)
        else:
            bars = axis.bar(offsets, [value if value is not None else 0 for value in numeric],
                            width=width * 0.9, label=label,
                            color=colors, edgecolor="white", linewidth=0.5)
            for bar, value in zip(bars, numeric):
                if value is not None:
                    axis.annotate(f"{value:.1f}",
                                  (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                                  xytext=(0, 3), textcoords="offset points",
                                  ha="center", va="bottom", fontsize=8)
    axis.set_title(title, fontsize=12, fontweight="bold")
    if horizontal:
        axis.set_xlabel(unit)
        axis.set_yticks(positions, targets)
        axis.invert_yaxis()
        axis.grid(axis="x", alpha=0.3, linestyle="--")
        if all_values:
            axis.set_xlim(0, max(all_values) * 1.18)
    else:
        axis.set_ylabel(unit)
        axis.set_xticks(positions, targets)
        axis.grid(axis="y", alpha=0.3, linestyle="--")
        if all_values:
            axis.set_ylim(0, max(all_values) * 1.18)
    axis.set_axisbelow(True)
    if len(series) > 1:
        axis.legend(fontsize=8)
    elif len(targets) > 1:
        from matplotlib.patches import Patch
        axis.legend(handles=[Patch(color=next((color for name, color in hw_colors.items()
                                               if name in target.upper()), "#888888"),
                                  label=target) for target in targets], fontsize=8)
    if not all_values:
        axis.text(0.5, 0.5, "N/A - no recorded measurements",
                  transform=axis.transAxes, ha="center", va="center", color="#59636b")
    return image(figure, title)


def fps_heatmaps(path, metadata):
    sample_path = path / SAMPLES_FILE
    if not sample_path.exists():
        return "<p>No measurement interval history recorded.</p>"
    containers = {}
    with sample_path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("record_type") == "measurement_interval":
                containers.setdefault(record.get("container_name", "unknown"), []).append(record)
    if not containers:
        return "<p>No measurement interval history recorded.</p>"
    output = []
    for container, records in containers.items():
        streams = sorted({name for record in records for name in record.get("stream_fps", {})})
        if not streams:
            output.append(f"<p>{text(container)}: no per-stream FPS samples recorded.</p>")
            continue
        matrix = np.full((len(streams), len(records)), np.nan)
        for column, record in enumerate(records):
            for row, name in enumerate(streams):
                value = number(record.get("stream_fps", {}).get(name))
                if value is not None:
                    matrix[row, column] = value
        figure = Figure(figsize=(10, min(12, max(4, len(streams) * 0.2))), layout="constrained")
        axis = figure.subplots()
        cmap = matplotlib.colormaps["RdYlGn"].copy()
        cmap.set_bad("#e1e4e6")
        plotted = axis.imshow(np.ma.masked_invalid(matrix), aspect="auto", cmap=cmap,
                              interpolation="nearest", origin="lower")
        columns = list(range(0, len(records), max(1, len(records) // 15)))
        axis.set_xticks(columns, [
            f"{records[index].get('interval_id', index + 1)} / {records[index].get('lane_count', '?')} lanes\n{records[index].get('status', 'unknown')}"
            for index in columns], rotation=45, ha="right", fontsize=8)
        rows = list(range(0, len(streams), max(1, len(streams) // 20)))
        axis.set_yticks(rows, [streams[index] for index in rows], fontsize=8)
        axis.set_xlabel("Measurement interval / lane count / status")
        axis.set_ylabel("Stream")
        figure.colorbar(plotted, ax=axis, label="Measured FPS")
        output.append(image(figure, f"{metadata['accelerator']} - {container}: per-stream FPS"))
    return "".join(output)


def generate_html_report(runs):
    entries = [(path, metadata, run_metrics(path, metadata)) for path, metadata in runs]
    about = runs[0][1]["camera_config"].get("_about", {})
    description = about if isinstance(about, dict) else {"Description": about}
    config_name = description.get("name", "Stream Density Benchmark")
    description_html = "".join(
        f"<p><b>{text(key.replace('_', ' ').capitalize())}:</b> {text(value)}</p>"
        for key, value in description.items() if key != "name" and value
    )
    parts = ["<!doctype html><html lang='en'><head><meta charset='utf-8'>",
             "<meta name='viewport' content='width=device-width, initial-scale=1'>",
             "<title>Stream Density Benchmark Report</title><style>",
             "body{font-family:'Segoe UI',Arial,sans-serif;margin:0;padding:24px;background:#f5f5f5;color:#222}"
             "h1{color:#0071c5;border-bottom:3px solid #0071c5;padding-bottom:8px}h2{color:#333;margin-top:36px}"
             "h3{margin:0 0 6px;font-size:15px}.meta{background:#fff;border-radius:8px;padding:14px 20px;margin-bottom:20px;box-shadow:0 1px 4px #0002}"
             ".meta-cols{display:flex;gap:0}.meta-half{flex:1;padding:4px 12px;min-width:0}.meta-divider{width:1px;background:#e0e0e0;margin:4px 8px}"
             ".meta-half table{box-shadow:none}.meta-heading{font-size:13px;font-weight:700;color:#0071c5;text-transform:uppercase;margin-bottom:8px}"
             ".summary-cards{display:flex;gap:18px;flex-wrap:wrap;margin-bottom:28px}.card{background:#fff;border-radius:8px;padding:18px 24px;box-shadow:0 2px 8px #0002;min-width:230px;flex:1}"
             ".big{font-size:48px;font-weight:700;line-height:1.1}.sub{font-size:13px;color:#555;margin-top:5px}.status{font-weight:700;text-transform:capitalize}"
             ".table-scroll{overflow:auto}table{border-collapse:collapse;width:100%;background:#fff;font-size:13px}"
             "table.kpi th,table.kpi td{border:1px solid #ddd;padding:9px 14px;text-align:right}table.kpi th{background:#0071c5;color:#fff}"
             "table.kpi td:first-child{text-align:left}table.kpi tr:nth-child(even){background:#f0f7ff}td,th{padding:7px 10px;border:1px solid #ddd;text-align:left;overflow-wrap:anywhere}"
             ".unit{color:#888;font-size:.9em}.charts{display:flex;flex-wrap:wrap;gap:20px;margin-top:16px}.chart{background:#fff;padding:12px;box-shadow:0 1px 4px #0002;border-radius:6px}"
             "figure{margin:0;min-width:0}figcaption{font-size:13px;color:#555;font-weight:700;padding:0 0 6px}img{display:block;max-width:620px;width:100%;height:auto}"
             "footer{margin-top:40px;color:#999;font-size:.85em;text-align:center}p{overflow-wrap:anywhere}"
             "@media(max-width:760px){body{padding:16px}.meta-cols{display:block}.meta-divider{height:1px;width:auto}.card{min-width:0}}",
             "</style></head><body><h1>Intel Loss Prevention - Stream Density Benchmark Report</h1>",
             f"<p><b>{text(config_name)}</b> &nbsp; | &nbsp; Configuration: <code>{text(runs[0][1]['config_id'])}</code></p>",
             description_html,
             "<div class='meta'><div class='meta-cols'><div class='meta-half'><div class='meta-heading'>Hardware</div>"]
    for path, metadata, metrics in entries:
        hardware = list(hardware_details(metadata).items())
        hardware.extend([("OS", metadata["os"]), ("Kernel", metadata["kernel"])])
        parts.append(f"<h3>{text(metadata['accelerator'])} - {text(path.name)}</h3>")
        parts.append(table(["Field", "Recorded value"], hardware))
    parts.append("</div><div class='meta-divider'></div><div class='meta-half'><div class='meta-heading'>Test &amp; Software</div>")
    for path, metadata, metrics in entries:
        settings = {
            "Test configuration": metadata.get("camera_config", {}).get("_about", {}).get("name", metadata["config_id"]),
            "Source run": str(path),
            "Started": metadata["started_at"], "Finished": metadata["finished_at"],
            "Run duration (s)": metadata.get("duration_seconds"), **test_configuration(metadata),
        }
        parts.append(f"<h3>{text(metadata['accelerator'])}</h3>")
        parts.append(table(["Field", "Recorded value"], settings.items()))
    parts.append("</div></div><h2>Stream Density Summary</h2><div class='summary-cards'>")
    for path, metadata, metrics in entries:
        parts.extend([f"<div class='card' style='border-top:4px solid #0071c5'><h3>{text(metadata['accelerator'])}</h3>",
                      f"<p class='status'>{text(metrics['status'])}</p>",
                      f"<div class='big'>{text(metrics['streams'])}</div>",
                      "<div class='sub'>stream density (streams)</div>",
                      f"<div class='sub'>Run: <b>{text(path.name)}</b></div>"])
        if metrics["last_passing_lanes"]:
            parts.append(f"<p>Last passing lanes: {text(labelled(metrics['last_passing_lanes']))} (not confirmed)</p>")
        parts.append(table(["KPI", "Value"], [
            ("Throughput (FPS)", metrics["throughput"]), ("FPS/stream", metrics["rate"]),
            ("Pooled latency p50 / p95 / p99 (ms)", "/".join(
                cell(metrics[key]) for key in ("latency_p50", "latency_p95", "latency_p99"))),
            ("Highest per-stream p95 (ms)", metrics["highest_stream_p95"]),
            ("GPU power reported by qmassa (W)", labelled(metrics["gpu_power"])),
            ("Power/stream (W, source scope)", labelled(metrics["power"])),
        ]))
        stages = []
        for workload, pipeline in active_stages(metadata["camera_config"], metadata["pipeline_config"]).items():
            for stage in pipeline:
                stages.append((workload, stage.get("type"), stage.get("model"),
                               stage.get("device"), stage.get("precision")))
                if stage.get("vlm_model"):
                    stages.append((workload, "vlm", stage["vlm_model"],
                                   stage.get("vlm_device"), stage.get("vlm_precision")))
        parts.append("<h4>Pipeline stages</h4>" + table(
            ["Workload", "Stage", "Model", "Device", "Precision"], stages))
        parts.append(f"</div>")
    parts.append("</div><h2>KPI Comparison Table</h2>")
    fields = [
        ("Sweep status", "status"), ("Stream density (streams)", None),
        ("Throughput (FPS)", "throughput"), ("FPS/stream", "rate"),
        ("Pooled latency p50 (ms)", "latency_p50"),
        ("Pooled latency p95 (ms)", "latency_p95"),
        ("Pooled latency p99 (ms)", "latency_p99"),
        ("Highest per-stream p95 (ms)", "highest_stream_p95"),
        ("Package power (W, source scope)", "package_power"),
        ("GPU power reported by qmassa (W)", "gpu_power"),
        ("Power/stream (W, source scope)", "power"),
        ("Utilisation (%, source scope)", "utilisation"),
        ("Utilisation sampling period", "utilisation_period"),
    ]
    rows = []
    for label, key in fields:
        values = []
        for path, metadata, metrics in entries:
            value = density_result(metrics) if key is None else metrics[key]
            values.append(labelled(value) if isinstance(value, dict) else value)
        rows.append([label, *values])
    parts.append(table(["KPI", *[metadata["accelerator"] for _, metadata, _ in entries]], rows))
    parts.append("<p>FPS and latency use the final confirmed measurement interval. Package power retains its source scope; it is not accelerator-only power. Missing values are N/A, not zero.</p>")
    parts.append("<h2>Comparison Charts</h2><div class='charts'>")
    charts = [
        ("Max Stream Density @ Target FPS (higher is better)", "Streams", [("Stream Density", [metrics["streams"] for _, _, metrics in entries])], True),
        ("Processor Package Power (lower is better)", "Watts (W)", [
            (key, [metrics["package_power"].get(key) for _, _, metrics in entries])
            for key in sorted({key for _, _, metrics in entries for key in metrics["package_power"]})], True),
        ("Efficiency: Power per Stream (lower is better)", "W / stream", [
            (key, [metrics["power"].get(key) for _, _, metrics in entries])
            for key in sorted({key for _, _, metrics in entries for key in metrics["power"]})], True),
        ("Accelerator Utilization", "Utilization (%)", [
            (key, [metrics["utilisation"].get(key) for _, _, metrics in entries])
            for key in sorted({key for _, _, metrics in entries for key in metrics["utilisation"]})], False),
        ("Per-Frame Latency Percentiles (lower is better)", "Latency (ms)", [
            (label, [metrics[key] for _, _, metrics in entries]) for label, key in (
                ("Pooled p50", "latency_p50"), ("Pooled p95", "latency_p95"),
                ("Pooled p99", "latency_p99"), ("Highest stream p95", "highest_stream_p95"))], False),
    ]
    for title, unit, series, horizontal in charts:
        parts.append(f"<div class='chart'>{comparison_chart(title, unit, entries, series, horizontal)}</div>")
    parts.append("</div><h2>Stream Density FPS Progression Heatmaps (Per-Stream FPS Heatmaps)</h2><p>Each cell shows measured FPS for a stream at a measurement interval. Grey means no recorded value; interval labels retain status and lane count.</p><div class='charts'>")
    for path, metadata, metrics in entries:
        parts.append(f"<div class='chart'><h3>{text(metadata['accelerator'])}</h3>{fps_heatmaps(path, metadata)}</div>")
    parts.append("</div><footer>Generated from recorded run metadata and measurement intervals by compare_stream_density.py.</footer></body></html>")
    return "".join(parts)