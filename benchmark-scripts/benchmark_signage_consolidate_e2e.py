'''
* Copyright (C) 2026 Intel Corporation.
*
* SPDX-License-Identifier: Apache-2.0
'''


"""Consolidate PID, Web UI, AIG, Qdrant and host-metrics results into an end-to-end stage table."""

import argparse
import csv
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path


FIELDS = ("stage", "source", "samples", "avg_ms", "p50_ms", "p95_ms", "max_ms", "value", "note", "file")


def latest(directory, pattern):
    files = sorted(directory.glob(pattern), key=lambda path: path.stat().st_mtime)
    return files[-1] if files else None


def load(path):
    return json.loads(path.read_text()) if path else None


def stats(values):
    values = sorted(float(value) for value in values if isinstance(value, (int, float)))
    if not values:
        return {"samples": 0}
    return {"samples": len(values), "avg_ms": round(statistics.mean(values), 3),
            "p50_ms": round(statistics.median(values), 3),
            "p95_ms": round(values[math.ceil(0.95 * len(values)) - 1], 3),
            "max_ms": round(values[-1], 3)}


def row(stage, source, path, values=None, value=None, note=""):
    entry = {"stage": stage, "source": source, "note": note, "file": path.name if path else None}
    entry.update(stats(values or []) if value is None else {"value": value})
    return entry


def consolidate(results):
    rows = []
    pid_path = latest(results, "pid_benchmark_*[0-9].json")
    pid = load(pid_path)
    pid_frames_path = pid_path.with_name(pid_path.stem + "_frames.json") if pid_path else None
    pid_frames = load(pid_frames_path) if pid_frames_path and pid_frames_path.exists() else []
    web_path = latest(results, "web_ui_benchmark_*[0-9].json")
    web_frames = load(web_path.with_name(web_path.stem + "_frames.json")) if web_path else []
    ads_path = web_path.with_name(web_path.stem + "_ads.json") if web_path else None
    ads = load(ads_path) if ads_path and ads_path.exists() else []
    aig_path = latest(results, "aig_benchmark_*[0-9].json")
    aig = load(aig_path) or []
    qdrant_path = latest(results, "qdrant_benchmark_*[0-9].json")
    qdrant = load(qdrant_path) or []

    def ad_values(field, path=None):
        return [ad[field] for ad in ads if path is None or ad.get("path") == path]

    if pid:
        rows.append(row("1. Video frame -> PID inference", "PID", pid_path,
                        [frame.get("pipeline_latency_reported") for frame in pid_frames],
                        note="DLSPS avg_pipeline_latency; empty when DLSPS does not report it"))
        rows.append(row("1. PID throughput (FPS)", "PID", pid_path, value=pid.get("fps_dlstreamer"),
                        note="Throughput, not latency"))
        rows.append(row("2. PID -> MQTT -> benchmark subscriber", "PID", pid_path,
                        [frame.get("mqtt_arrival_delay_ms") for frame in pid_frames]))
    if web_path:
        rows.append(row("2-3. PID frame timestamp -> Web UI receive", "Web UI", web_path,
                        [frame.get("mqtt_to_web_ui_ms") for frame in web_frames],
                        note="Publish and broker delivery combined; DLSPS does not expose publish time"))
        rows.append(row("4. Web UI filtering (callback)", "Web UI", web_path,
                        [frame.get("callback_total_ms") for frame in web_frames]))
        rows.append(row("4b. Queue wait before ad decision", "Web UI", web_path,
                [frame.get("queue_wait_ms") for frame in web_frames],
                note="Candidate enqueue to generator dequeue; excludes cooldown suppression"))
        rows.append(row("4c. Cooldown remaining at suppression", "Web UI", web_path,
                [frame.get("cooldown_remaining_ms") for frame in web_frames],
                note="Remaining cooldown when a candidate is dropped; it is not deferred"))
        rows.append(row("4d. Cooldown-suppressed candidates", "Web UI", web_path,
                value=sum(frame.get("cooldown_suppressed") is True for frame in web_frames),
                note="Candidates dropped during cooldown instead of queued for later"))
        rows.append(row("5. Product selection", "Web UI", web_path,
                        [frame.get("selector_total_ms") for frame in web_frames]))
        rows.append(row("6. ASe query (live)", "Web UI", web_path, ad_values("ase_query_ms")))
        rows.append(row("7. AIG generation on ASe miss (live)", "Web UI", web_path,
                        ad_values("aig_generation_ms", "dynamic")))
        rows.append(row("8a. Ad ready -> served to browser", "Web UI", web_path, ad_values("ready_to_served_ms"),
                        note="Includes the browser's 2 s polling interval"))
        rows.append(row("8b. Browser fetch", "Browser", web_path, ad_values("browser_fetch_ms")))
        rows.append(row("9. Browser render", "Browser", web_path, ad_values("browser_render_ms"),
                        note="Requires an open Web UI page; paint-opportunity estimate, not measured display time"))
        rows.append(row("E2E frame -> ad ready", "Web UI", web_path, ad_values("frame_to_ad_ready_ms")))
        rows.append(row("E2E frame -> ad rendered", "Web UI + Browser", web_path, ad_values("frame_to_render_ms")))
        rows.append(row("Ads generated (predefined / dynamic)", "Web UI", web_path,
                        value=f"{len(ad_values('ad_id', 'predefined'))} / {len(ad_values('ad_id', 'dynamic'))}"))
    if aig:
        searches = [entry for entry in aig if entry["kind"] == "predefined"]
        generations = [entry for entry in aig if entry["kind"] == "dynamic"]
        rows.append(row("6. ASe query (synthetic)", "AIG benchmark", aig_path,
                        [entry["http_elapsed_ms"] for entry in searches]))
        rows.append(row("6. ASe vector search (server)", "AIG benchmark", aig_path,
                        [entry.get("vector_search_ms") for entry in searches]))
        rows.append(row("7. AIG first call (cold if run with BENCHMARK_COLD_START)", "AIG benchmark", aig_path,
                        [entry["http_elapsed_ms"] for entry in generations if entry["call"] == 1]))
        rows.append(row("7. AIG warm calls", "AIG benchmark", aig_path,
                        [entry["http_elapsed_ms"] for entry in generations if entry["call"] > 1]))
        rows.append(row("7. AIG inference (server)", "AIG benchmark", aig_path,
                        [entry.get("inference_ms") for entry in generations]))
    for entry in qdrant:
        if entry["kind"] == "vector_search":
            rows.append(row(f"Qdrant search @ {entry['points']} points", "Qdrant benchmark", qdrant_path,
                            value=entry["avg_query_ms"], note=f"{entry['queries_per_sec']} queries/s"))
        else:
            rows.append(row("Qdrant predefined-ad indexing", "Qdrant benchmark", qdrant_path,
                            value=entry["index_http_ms"]))

    for prefix in ("e2e", "pid", "web_ui", "aig", "qdrant"):
        metrics_path = latest(results, f"{prefix}_metrics*.json")
        metrics = load(metrics_path)
        if not metrics:
            continue
        for key, label, column in (("cpu_utilization", "Host CPU %", 1), ("gpu_utilization", "GPU busy %", 1),
                                   ("npu_utilization", "NPU %", 1), ("memory", "Host memory %", 4),
                                   ("power", "Package power W", 1)):
            values = [sample[column] for sample in metrics.get(key, []) if len(sample) > column]
            if values:
                rows.append(row(f"Resources during {prefix}: {label}", "performance-tools", metrics_path,
                                value=round(statistics.mean(values), 2)))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    rows = consolidate(args.results)
    if not rows:
        print(f"No benchmark results found in {args.results}", file=sys.stderr)
        sys.exit(1)
    output = args.output or args.results
    output.mkdir(parents=True, exist_ok=True)
    name = f"e2e_consolidated_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    (output / f"{name}.json").write_text(json.dumps(
        {"timestamp": datetime.now(timezone.utc).isoformat(), "stages": rows}, indent=2) + "\n")
    with (output / f"{name}.csv").open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    for entry in rows:
        summary = entry.get("value", entry.get("avg_ms"))
        print(f"{entry['stage']:<62} {str(summary):>12}  samples={entry.get('samples', '-')}")
    print(f"Saved {output / name}.json and .csv")


if __name__ == "__main__":
    main()
