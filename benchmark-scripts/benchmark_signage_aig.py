'''
* Copyright (C) 2026 Intel Corporation.
*
* SPDX-License-Identifier: Apache-2.0
'''


"""Benchmark ASe search and AIG image generation through the HTTP API."""

import argparse
import base64
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

import requests


SEARCH_STAGES = ("query_build", "vector_search", "ranking_retrieval")
GENERATION_STAGES = ("model_load", "inference", "overlay", "jpeg_encode", "model_unload", "model_cleanup")
DEFAULT_BASE_URL = "https://localhost:5000/aig-api"


def parse_server_timing(header):
    timings = {}
    for entry in header.split(","):
        name, _, timing = entry.strip().partition(";dur=")
        if timing:
            try:
                timings[f"{name}_ms"] = float(timing)
            except ValueError:
                pass
    return timings


def request_ad(session, url, payload, kind, call, timeout, cold_start=False):
    start = time.perf_counter()
    response = session.post(url, json=payload, headers={"X-Benchmark-Timing": "true"}, timeout=timeout)
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.raise_for_status()
    if kind == "predefined":
        ads = response.json()
        if not isinstance(ads, list):
            raise ValueError("ASe did not return a list of ads")
        image_size = sum(len(base64.b64decode(ad["imgb64"])) for ad in ads if ad.get("imgb64"))
        result_count = len(ads)
        stages = SEARCH_STAGES
    else:
        if not response.headers.get("Content-Type", "").startswith("image/jpeg"):
            raise ValueError("AIG did not return a JPEG image")
        image_size = len(response.content)
        result_count = 1
        stages = GENERATION_STAGES
    timings = parse_server_timing(response.headers.get("Server-Timing", ""))
    sla_ms = 5000 if kind == "predefined" else (60000 if call == 1 and cold_start else 15000)
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "kind": kind,
        "call": call,
        "http_elapsed_ms": round(elapsed_ms, 3),
        "sla_ms": sla_ms,
        "sla_met": elapsed_ms < sla_ms,
        "response_size_kb": round(len(response.content) / 1024, 3),
        "image_size_kb": round(image_size / 1024, 3),
        "result_count": result_count,
        **{f"{stage}_ms": timings.get(f"{stage}_ms") for stage in stages},
    }


def summarize(rows, device, capture_elapsed_ms):
    """Average each call type so predefined (ASe) and dynamic (AIG) runs compare directly."""
    def average(values):
        values = [value for value in values if isinstance(value, (int, float))]
        return round(mean(values), 3) if values else None

    summary = {"timestamp": datetime.now(timezone.utc).isoformat(), "device": device,
               "capture_elapsed_ms": capture_elapsed_ms}
    for kind, stages in (("predefined", SEARCH_STAGES), ("dynamic", GENERATION_STAGES)):
        subset = [row for row in rows if row["kind"] == kind]
        summary[f"{kind}_calls"] = len(subset)
        summary[f"{kind}_sla_met"] = sum(bool(row["sla_met"]) for row in subset)
        for field in ("http_elapsed_ms", "response_size_kb", "image_size_kb",
                      *(f"{stage}_ms" for stage in stages)):
            summary[f"avg_{kind}_{field}"] = average([row.get(field) for row in subset])
    # The first dynamic call carries model load, so warm calls are the comparable number.
    dynamic = [row for row in rows if row["kind"] == "dynamic"]
    first = [row["http_elapsed_ms"] for row in dynamic if row["call"] == 1]
    summary["dynamic_first_call_http_elapsed_ms"] = first[0] if first else None
    summary["avg_dynamic_warm_http_elapsed_ms"] = average(
        [row["http_elapsed_ms"] for row in dynamic if row["call"] > 1])
    return summary


def run(args):
    base_url = args.base_url.rstrip("/")
    local_self_signed = (base_url == DEFAULT_BASE_URL
                         and not os.environ.get("REQUESTS_CA_BUNDLE")
                         and not os.environ.get("CURL_CA_BUNDLE"))
    session = requests.Session()
    session.verify = not (args.insecure or local_self_signed)
    if not session.verify:
        requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)
        print("TLS verification disabled for local AIG proxy" if local_self_signed and not args.insecure
              else "TLS verification disabled by --insecure", file=sys.stderr)
    if args.cold_start:
        import subprocess
        # Restart by container name, not `compose restart`, so the script does not depend on the caller's Compose project.
        subprocess.run(["docker", "restart", "aig-server"], check=True)
    deadline = time.monotonic() + min(args.timeout, 120)
    while True:
        try:
            status = session.get(f"{base_url}/aig/hstatus/1", timeout=3)
            status.raise_for_status()
            break
        except requests.exceptions.SSLError as error:
            raise RuntimeError(
                "TLS certificate verification failed; use --insecure for the local self-signed "
                "certificate or set REQUESTS_CA_BUNDLE to a trusted CA file"
            ) from error
        except requests.RequestException:
            if time.monotonic() >= deadline:
                raise RuntimeError("AIG did not become ready")
            time.sleep(1)
    search_payload = {"query": args.query, "n_results": 1, "use_default_ad_onempty": False}
    generation_payload = {
        "description": args.description,
        "device": args.device,
        "promo_details": {"promo_text": args.slogan},
        "logo_details": {"logo_percentage": 15},
    }
    rows = []
    benchmark_start = time.perf_counter()
    for call in range(1, args.iterations + 1):
        scheduled_at = benchmark_start + (call - 1) * args.duration / args.iterations
        if scheduled_at > time.perf_counter():
            time.sleep(scheduled_at - time.perf_counter())
        start = time.perf_counter()
        search = request_ad(session, f"{base_url}/ase/predef/query/ad", search_payload,
                            "predefined", call, args.timeout)
        if search["result_count"] == 0:
            raise RuntimeError(f"No predefined image matched {args.query!r}; choose an indexed query")
        rows.append(search)
        generation = request_ad(session, f"{base_url}/aig/minf/", generation_payload,
                                "dynamic", call, args.timeout, args.cold_start)
        rows.append(generation)
        for row in (search, generation):
            row["workflow_elapsed_ms"] = round((time.perf_counter() - start) * 1000, 3)
    if benchmark_start + args.duration > time.perf_counter():
        time.sleep(benchmark_start + args.duration - time.perf_counter())
    capture_elapsed_ms = round((time.perf_counter() - benchmark_start) * 1000, 3)
    for row in rows:
        row["device"] = args.device
        row["capture_elapsed_ms"] = capture_elapsed_ms

    args.output.mkdir(parents=True, exist_ok=True)
    name = f"aig_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    fields = ("timestamp", "kind", "call", "device", "http_elapsed_ms", "sla_ms", "sla_met",
              "response_size_kb", "image_size_kb", "result_count", "workflow_elapsed_ms",
              "capture_elapsed_ms",
              *(f"{stage}_ms" for stage in SEARCH_STAGES + GENERATION_STAGES))
    (args.output / f"{name}.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (args.output / f"{name}.csv").open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows, args.device, capture_elapsed_ms)
    (args.output / f"{name}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.output / f"{name}_summary.csv").open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=summary.keys())
        writer.writeheader()
        writer.writerow(summary)
    print(json.dumps(summary, indent=2))
    print(f"Saved {args.output / name}.json, .csv and _summary.json/.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--insecure", action="store_true", help="disable TLS verification for a custom HTTPS endpoint")
    parser.add_argument("--query", default="broccoli and carrot")
    parser.add_argument("--description", default="A studio photograph of a freshly brewed cup of coffee on a clean table")
    parser.add_argument("--slogan", default="Fresh coffee")
    parser.add_argument("--device", choices=("CPU", "GPU"), default="GPU")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--duration", type=int, default=300, help="minimum capture duration in seconds")
    parser.add_argument("--timeout", type=int, default=400)
    parser.add_argument("--cold-start", action="store_true", help="restart only aig-server before measuring")
    parser.add_argument("--output", type=Path, default=Path("results"))
    options = parser.parse_args()
    if options.iterations < 1 or options.timeout < 1 or options.duration < 0:
        parser.error("iterations and timeout must be positive; duration cannot be negative")
    try:
        run(options)
    except (requests.RequestException, subprocess.CalledProcessError, ValueError, OSError, RuntimeError) as error:
        print(f"AIG benchmark failed: {error}", file=sys.stderr)
        sys.exit(1)