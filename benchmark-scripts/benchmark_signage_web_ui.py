'''
* Copyright (C) 2026 Intel Corporation.
*
* SPDX-License-Identifier: Apache-2.0
'''


"""Collect per-frame Web UI product selection timing from the running container."""

import argparse
import csv
import json
import os
import select
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean


CALLBACK_MARKER = "WEB_UI_SELECTION "
SELECTOR_MARKER = "WEB_UI_PRODUCT_SELECTION "
AD_TRACE_MARKER = "WEB_UI_AD_TRACE "
COOLDOWN_MARKER = "WEB_UI_AD_COOLDOWN "
AD_SERVED_MARKER = "WEB_UI_AD_SERVED "
AD_RENDERED_MARKER = "WEB_UI_AD_RENDERED "
MARKERS = (CALLBACK_MARKER, SELECTOR_MARKER, AD_TRACE_MARKER, COOLDOWN_MARKER,
           AD_SERVED_MARKER, AD_RENDERED_MARKER)
FIELDS = ("timestamp", "frame_id", "recency_frames", "confidence_threshold",
          "mqtt_to_web_ui_ms", "label_normalization_ms", "recency_filtering_ms", "confidence_gating_ms",
          "callback_total_ms", "product_key_normalization_ms", "csv_lookup_ms",
          "product_deduplication_ms", "selector_total_ms", "candidate_count", "queue_wait_ms",
          "cooldown_remaining_ms", "cooldown_suppressed",
          "total_selection_logic_ms")
AD_FIELDS = ("timestamp", "ad_id", "frame_id", "item", "path", "queue_wait_ms", "ase_query_ms",
             "aig_generation_ms", "ad_generation_total_ms", "response_size_kb", "frame_to_ad_ready_ms",
             "ready_to_served_ms", "browser_fetch_ms", "browser_render_ms", "frame_to_render_ms")


def parse_log_line(line):
    for marker in MARKERS:
        if marker in line:
            try:
                payload = json.loads(line.split(marker, 1)[1])
            except json.JSONDecodeError:
                return None
            return marker, line.split(" ", 1)[0], payload
    return None


def combine_events(events):
    frames = []
    selections = {}
    queue_waits = {}
    cooldowns = {}
    for marker, timestamp, payload in events:
        frame_id = payload.get("frame_id")
        if marker == CALLBACK_MARKER:
            row = {field: payload.get(field) for field in FIELDS}
            row["timestamp"] = timestamp
            frames.append(row)
        elif marker == SELECTOR_MARKER and frame_id is not None:
            selections[frame_id] = payload
        elif marker == AD_TRACE_MARKER and frame_id is not None:
            queue_waits[frame_id] = payload.get("queue_wait_ms")
        elif marker == COOLDOWN_MARKER and frame_id is not None:
            queue_waits[frame_id] = payload.get("queue_wait_ms")
            cooldowns[frame_id] = payload
    for row in frames:
        frame_id = row["frame_id"]
        payload = selections.get(frame_id)
        if payload is not None:
            row.update({field: payload.get(field) for field in
                        ("product_key_normalization_ms", "csv_lookup_ms",
                         "product_deduplication_ms", "selector_total_ms", "candidate_count")})
            row["total_selection_logic_ms"] = round(
                row["callback_total_ms"] + row["selector_total_ms"], 6)
        row["queue_wait_ms"] = queue_waits.get(frame_id)
        cooldown = cooldowns.get(frame_id)
        row["cooldown_suppressed"] = cooldown is not None if frame_id is not None else None
        if cooldown is not None:
            row["cooldown_remaining_ms"] = cooldown.get("cooldown_remaining_ms")
    return frames


def combine_ad_events(events):
    ads = {}
    served = {}
    rendered = {}
    for marker, timestamp, payload in events:
        ad_id = payload.get("ad_id")
        if marker == AD_TRACE_MARKER:
            ads[ad_id] = {field: payload.get(field) for field in AD_FIELDS}
            ads[ad_id]["timestamp"] = timestamp
        elif marker == AD_SERVED_MARKER:
            served.setdefault(ad_id, payload)
        elif marker == AD_RENDERED_MARKER:
            rendered.setdefault(ad_id, payload)
    for ad_id, row in ads.items():
        row.update({key: value for key, value in served.get(ad_id, {}).items() if key in AD_FIELDS})
        row.update({key: value for key, value in rendered.get(ad_id, {}).items() if key in AD_FIELDS})
        parts = [row[key] for key in ("frame_to_ad_ready_ms", "ready_to_served_ms",
                                      "browser_fetch_ms", "browser_render_ms")]
        if all(isinstance(part, (int, float)) for part in parts):
            row["frame_to_render_ms"] = round(sum(parts), 3)
    return list(ads.values())

def run(duration, output):
    if subprocess.check_output(
        ["docker", "inspect", "--format", "{{.State.Running}}", "web-ui"], text=True
    ).strip() != "true":
        raise RuntimeError("web-ui is not running; start the stack with make up")
    since = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    process = subprocess.Popen(
        ["docker", "logs", "--follow", "--timestamps", "--since", since, "web-ui"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    events = []
    buffer = b""
    try:
        while time.monotonic() - started < duration:
            ready, _, _ = select.select([process.stdout], [], [], 0.5)
            if ready:
                chunk = os.read(process.stdout.fileno(), 65536)
                if not chunk:
                    raise RuntimeError("web-ui log stream ended during capture")
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    event = parse_log_line(line.decode("utf-8", errors="replace"))
                    if event is not None:
                        events.append(event)
    finally:
        process.terminate()
        try:
            process.communicate(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()

    frames = combine_events(events)
    ads = combine_ad_events(events)
    if not frames:
        raise RuntimeError("No WEB_UI_SELECTION frames found; restart Web UI with instrumented code")
    summary = {"timestamp": datetime.now(timezone.utc).isoformat(),
               "capture_seconds": round(time.monotonic() - started, 3),
               "frames": len(frames),
               "frames_selected": sum(row["selector_total_ms"] is not None for row in frames),
               "cooldown_suppressed_frames": sum(row["cooldown_suppressed"] is True for row in frames),
               "recency_frames": frames[0]["recency_frames"],
               "confidence_threshold": frames[0]["confidence_threshold"]}
    for field in FIELDS:
        if field.endswith("_ms"):
            values = [row[field] for row in frames if isinstance(row[field], (int, float))]
            summary[f"avg_{field}"] = round(mean(values), 6) if values else None
    summary["ads"] = len(ads)
    for field in AD_FIELDS:
        if field.endswith("_ms"):
            values = [row[field] for row in ads if isinstance(row[field], (int, float))]
            summary[f"avg_ad_{field}"] = round(mean(values), 3) if values else None
    # Predefined ads skip AIG, so a combined average mixes two different pipelines.
    for path in ("predefined", "dynamic"):
        subset = [row for row in ads if row.get("path") == path]
        summary[f"ads_{path}"] = len(subset)
        for field in AD_FIELDS:
            if field.endswith("_ms"):
                values = [row[field] for row in subset if isinstance(row[field], (int, float))]
                summary[f"avg_{path}_{field}"] = round(mean(values), 3) if values else None

    output.mkdir(parents=True, exist_ok=True)
    name = f"web_ui_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    (output / f"{name}.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output / f"{name}.csv").open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=summary.keys())
        writer.writeheader()
        writer.writerow(summary)
    (output / f"{name}_frames.json").write_text(json.dumps(frames, indent=2) + "\n")
    with (output / f"{name}_frames.csv").open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(frames)
    (output / f"{name}_ads.json").write_text(json.dumps(ads, indent=2) + "\n")
    with (output / f"{name}_ads.csv").open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=AD_FIELDS)
        writer.writeheader()
        writer.writerows(ads)
    print(json.dumps(summary, indent=2))
    print(f"Saved {output / name}.json, .csv, _frames.json/.csv and _ads.json/.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=int, default=300)
    parser.add_argument("--output", type=Path, default=Path("results"))
    options = parser.parse_args()
    if options.duration < 1:
        parser.error("--duration must be positive")
    try:
        run(options.duration, options.output)
    except (OSError, subprocess.CalledProcessError, RuntimeError) as error:
        print(f"Web UI benchmark failed: {error}", file=sys.stderr)
        sys.exit(1)