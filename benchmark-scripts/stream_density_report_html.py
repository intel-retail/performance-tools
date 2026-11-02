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


def comparison_chart(title, unit, entries, series):
    figure = Figure(figsize=(8, 4), layout="constrained")
    axis = figure.subplots()
    positions = np.arange(len(entries))
    width = 0.8 / max(len(series), 1)
    colors = ("#0071c5", "#26834a", "#d39b16", "#bd4750")
    for index, (label, values) in enumerate(series):
        offsets = positions + (index - (len(series) - 1) / 2) * width
        numeric = [number(value) for value in values]
        bars = axis.bar(offsets, [value if value is not None else np.nan for value in numeric],
                        width, label=label, color=colors[index % len(colors)])
        for bar, value, offset in zip(bars, numeric, offsets):
            axis.annotate(cell(value), (offset, value if value is not None else 0),
                          xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8)
    axis.set_xticks(positions, [run["accelerator"] for _, run, _ in entries])
    axis.set_ylabel(unit)
    axis.set_ylim(bottom=0)
    axis.margins(y=0.2)
    axis.grid(axis="y", alpha=0.2)
    axis.set_axisbelow(True)
    if not any(number(value) is not None for _, values in series for value in values):
        axis.text(0.5, 0.5, "N/A - no recorded measurements",
                  transform=axis.transAxes, ha="center", va="center", color="#59636b")
    if series:
        axis.legend(fontsize=8, loc="upper left", bbox_to_anchor=(0, 1.16), ncols=2)
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
    name = about.get("name") if isinstance(about, dict) else about
    parts = ["<!doctype html><html lang='en'><head><meta charset='utf-8'>",
             "<meta name='viewport' content='width=device-width, initial-scale=1'>",
             "<title>Stream Density Benchmark Report</title><style>",
             "body{font-family:Verdana,sans-serif;margin:0;color:#242b30;background:#f4f6f7;letter-spacing:0}"
             "main{max-width:1400px;margin:auto;padding:28px}h1{font-size:28px;color:#0071c5;border-bottom:3px solid #0071c5;padding-bottom:14px}"
             "h2{font-size:21px;margin-top:32px}h3{font-size:16px;margin-top:0}p{line-height:1.6;overflow-wrap:anywhere}"
             ".metadata,.charts{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:22px}"
             "section{padding:10px 0;border-bottom:1px solid #d5dce0}.summary{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:18px}"
             "article{background:white;border:1px solid #d5dce0;border-top:4px solid #0071c5;border-radius:6px;padding:20px;min-width:0}"
             ".density{font-size:36px;font-weight:bold;color:#0071c5}.status{font-weight:bold}.table-scroll{overflow:auto}"
             "table{border-collapse:collapse;width:100%;font-size:13px;background:white}td,th{padding:10px;border:1px solid #dce1e4;text-align:left;overflow-wrap:anywhere}"
             "th{background:#0071c5;color:white}tbody tr:nth-child(even){background:#eef4f7}"
             "figure{margin:0;min-width:0}figcaption{font-size:15px;font-weight:bold;padding:10px 0}img{display:block;width:100%;height:auto}"
             "small{display:block;overflow-wrap:anywhere}footer{margin-top:30px;font-size:12px}"
             "@media(max-width:760px){main{padding:16px}.metadata,.charts{grid-template-columns:1fr}h1{font-size:23px}}",
             "</style></head><body><main><h1>Stream Density Benchmark Report</h1>",
             f"<p>{text(name)} | Configuration: {text(runs[0][1]['config_id'])}</p>",
             "<div class='metadata'><section><h2>Hardware</h2>"]
    for path, metadata, metrics in entries:
        parts.append(f"<h3>{text(metadata['accelerator'])} - {text(path.name)}</h3>")
        parts.append(table(["Field", "Recorded value"], hardware_details(metadata).items()))
    parts.append("</section><section><h2>Test &amp; Software</h2>")
    for path, metadata, metrics in entries:
        settings = {
            "Source run": str(path), "OS": metadata["os"], "Kernel": metadata["kernel"],
            "Started": metadata["started_at"], "Finished": metadata["finished_at"],
            "Duration (s)": metadata.get("duration_seconds"), **test_configuration(metadata),
        }
        parts.append(f"<h3>{text(metadata['accelerator'])}</h3>")
        parts.append(table(["Field", "Recorded value"], settings.items()))
    parts.append("</section></div><h2>Stream Density Summary</h2><div class='summary'>")
    for path, metadata, metrics in entries:
        parts.extend([f"<article><h3>{text(metadata['accelerator'])}</h3>",
                      f"<p class='status'>{text(metrics['status'])}</p>",
                      f"<div class='density'>{text(metrics['streams'])}</div>",
                      "<p>Stream density (streams)</p>"])
        if metrics["last_passing_lanes"]:
            parts.append(f"<p>Last passing lanes: {text(labelled(metrics['last_passing_lanes']))} (not confirmed)</p>")
        parts.append(table(["KPI", "Value"], [
            ("Throughput (FPS)", metrics["throughput"]), ("FPS/stream", metrics["rate"]),
            ("Pooled latency p95 (ms)", metrics["latency_p95"]),
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
        parts.append(f"<p>Duration: {text(metadata.get('duration_seconds'))} s</p><small>{text(path)}</small></article>")
    parts.append("</div><h2>KPI Comparison Table</h2>")
    fields = [
        ("Result", "status"), ("Stream density (streams)", None),
        ("Throughput (FPS)", "throughput"), ("FPS/stream", "rate"),
        ("Pooled latency p50 (ms)", "latency_p50"),
        ("Pooled latency p95 (ms)", "latency_p95"),
        ("Pooled latency p99 (ms)", "latency_p99"),
        ("Highest per-stream p95 (ms)", "highest_stream_p95"),
        ("Package power (W, source scope)", "package_power"),
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
        ("Stream density (streams)", "Streams", [("Streams", [metrics["streams"] for _, _, metrics in entries])]),
        ("Processor Package Power", "W", [
            (key, [metrics["package_power"].get(key) for _, _, metrics in entries])
            for key in sorted({key for _, _, metrics in entries for key in metrics["package_power"]})]),
        ("Power per Stream (source scope)", "W/stream", [
            (key, [metrics["power"].get(key) for _, _, metrics in entries])
            for key in sorted({key for _, _, metrics in entries for key in metrics["power"]})]),
        ("Accelerator Utilisation", "%", [
            (key, [metrics["utilisation"].get(key) for _, _, metrics in entries])
            for key in sorted({key for _, _, metrics in entries for key in metrics["utilisation"]})]),
        ("Latency Percentiles", "ms", [
            (label, [metrics[key] for _, _, metrics in entries]) for label, key in (
                ("Pooled p50", "latency_p50"), ("Pooled p95", "latency_p95"),
                ("Pooled p99", "latency_p99"), ("Highest stream p95", "highest_stream_p95"))]),
    ]
    for title, unit, series in charts:
        parts.append(comparison_chart(title, unit, entries, series))
    parts.append("</div><h2>Per-Stream FPS Heatmaps</h2><p>Measured FPS by measurement interval. Grey cells have no recorded value; interval labels retain status and lane count.</p>")
    for path, metadata, metrics in entries:
        parts.append(f"<section><h3>{text(metadata['accelerator'])}</h3>{fps_heatmaps(path, metadata)}</section>")
    parts.append("<footer>Generated from recorded run metadata and measurement intervals by compare_stream_density.py.</footer></main></body></html>")
    return "".join(parts)