"""Benchmark Qdrant collection scaling and ASe predefined-ad indexing."""

import argparse
import base64
import csv
import json
import math
import random
import re
import secrets
import statistics
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import requests


DEFAULT_ASE_URL = "https://localhost:5000/aig-api"
IMAGE_PATH = Path(__file__).resolve().parents[1] / "web-ui/pre-defined-ads/brocoli_carrot.jpg"


def memory_mib():
    stats = json.loads(subprocess.check_output(
        ["docker", "stats", "--no-stream", "--format", "{{json .}}", "ase-qdrant"], text=True
    ))
    value = stats["MemUsage"].split("/")[0].strip()
    match = re.fullmatch(r"([\d.]+)\s*([KMGT]?i?B)", value)
    if match is None:
        raise ValueError(f"Unrecognized Qdrant memory usage: {value}")
    amount, unit = match.groups()
    factors = {"KiB": 1 / 1024, "MiB": 1, "GiB": 1024, "TiB": 1024**2,
               "kB": 1000 / 1024**2, "MB": 1000**2 / 1024**2, "GB": 1000**3 / 1024**2}
    return round(float(amount) * factors[unit], 3)


def api(session, method, url, **kwargs):
    response = session.request(method, url, timeout=120, **kwargs)
    response.raise_for_status()
    data = response.json()
    if data.get("status") != "ok":
        raise RuntimeError(f"Qdrant error: {data}")
    return data["result"]


def benchmark(args):
    address = subprocess.check_output(
        ["docker", "inspect", "--format", "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}",
         "ase-qdrant"], text=True
    ).strip()
    if not address:
        raise RuntimeError("ase-qdrant is not running")
    base = f"http://{address}:6333"
    qdrant = requests.Session()
    qdrant.trust_env = False
    config = api(qdrant, "GET", f"{base}/collections/{args.collection}")
    vectors = config["config"]["params"]["vectors"]
    if "size" not in vectors:
        raise RuntimeError("Only a single unnamed vector collection is supported")
    dimensions = vectors["size"]
    rng = random.Random(42)
    query = [rng.random() for _ in range(dimensions)]
    name = f"benchmark-{uuid.uuid4().hex[:12]}"
    url = f"{base}/collections/{name}"
    rows = []
    created = False
    try:
        api(qdrant, "PUT", url, json={"vectors": {"size": dimensions, "distance": vectors["distance"]}})
        created = True
        next_id = 1
        for size in args.sizes:
            memory_before = memory_mib()
            upsert_start = time.perf_counter()
            while next_id <= size:
                points = [{"id": point_id, "vector": [rng.random() for _ in range(dimensions)]}
                          for point_id in range(next_id, min(next_id + args.batch_size, size + 1))]
                api(qdrant, "PUT", f"{url}/points?wait=true", json={"points": points})
                next_id += len(points)
            upsert_ms = (time.perf_counter() - upsert_start) * 1000
            info = api(qdrant, "GET", url)
            memory_after = memory_mib()
            samples = []
            query_loop_start = time.perf_counter()
            for _ in range(args.queries):
                started = time.perf_counter()
                api(qdrant, "POST", f"{url}/points/query", json={
                    "query": query, "limit": 1, "with_payload": False,
                })
                samples.append((time.perf_counter() - started) * 1000)
            query_loop_seconds = time.perf_counter() - query_loop_start
            rows.append({"kind": "vector_search", "collection": name,
                         "points": info["points_count"], "indexed_vectors": info.get("indexed_vectors_count"),
                         "queries": args.queries, "avg_query_ms": round(statistics.mean(samples), 3),
                         "p95_query_ms": round(sorted(samples)[math.ceil(0.95 * len(samples)) - 1], 3),
                         "queries_per_sec": round(args.queries / query_loop_seconds, 3),
                         "upsert_ms": round(upsert_ms, 3), "qdrant_memory_before_mib": memory_before,
                         "qdrant_memory_after_mib": memory_after,
                         "qdrant_memory_delta_mib": round(memory_after - memory_before, 3)})
    finally:
        if created:
            api(qdrant, "DELETE", url)

    ase = requests.Session()
    ase.verify = not args.insecure and args.ase_url.rstrip("/") != DEFAULT_ASE_URL
    if not ase.verify:
        requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)
    point_id = 1_000_000_000_000 + secrets.randbelow(1_000_000_000_000)
    live_url = f"{base}/collections/{args.collection}/points/{point_id}"
    if qdrant.get(live_url, timeout=10).status_code != 404:
        raise RuntimeError("Temporary ad ID is already in use; no ad was written")
    ad_url = f"{args.ase_url.rstrip('/')}/ase/predef/"
    payload = {"id": point_id, "description": f"Qdrant benchmark temporary ad {point_id}",
               "source": "benchmark", "imgb64": base64.b64encode(IMAGE_PATH.read_bytes()).decode("ascii")}
    try:
        memory_before = memory_mib()
        started = time.perf_counter()
        response = ase.post(ad_url, json=payload, timeout=120)
        index_ms = (time.perf_counter() - started) * 1000
        response.raise_for_status()
        if response.json().get("message") != "Success":
            raise RuntimeError(f"ASe add failed: {response.text[:200]}")
        memory_after = memory_mib()
        rows.append({"kind": "predefined_ad_index", "collection": args.collection,
                     "points": api(qdrant, "GET", f"{base}/collections/{args.collection}")["points_count"],
                     "index_http_ms": round(index_ms, 3), "image_kb": round(len(IMAGE_PATH.read_bytes()) / 1024, 3),
                     "qdrant_memory_before_mib": memory_before,
                     "qdrant_memory_after_mib": memory_after,
                     "qdrant_memory_delta_mib": round(memory_after - memory_before, 3)})
    finally:
        if qdrant.get(live_url, timeout=10).status_code == 200:
            response = ase.delete(f"{ad_url}{point_id}", timeout=120)
            response.raise_for_status()

    args.output.mkdir(parents=True, exist_ok=True)
    name = f"qdrant_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    fields = ("kind", "collection", "points", "indexed_vectors", "queries", "avg_query_ms",
              "p95_query_ms", "queries_per_sec", "upsert_ms", "index_http_ms", "image_kb",
              "qdrant_memory_before_mib", "qdrant_memory_after_mib", "qdrant_memory_delta_mib")
    (args.output / f"{name}.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (args.output / f"{name}.csv").open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))
    print(f"Saved {args.output / name}.json and .csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", default="ase-collection")
    parser.add_argument("--sizes", type=int, nargs="+", default=[100, 1000])
    parser.add_argument("--queries", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--ase-url", default=DEFAULT_ASE_URL)
    parser.add_argument("--insecure", action="store_true", help="disable TLS verification for custom ASe endpoints")
    parser.add_argument("--output", type=Path, default=Path("results"))
    options = parser.parse_args()
    if options.queries < 1 or options.batch_size < 1 or any(size < 1 for size in options.sizes):
        parser.error("queries, batch size and sizes must be positive")
    if options.sizes != sorted(set(options.sizes)):
        parser.error("sizes must be unique and sorted in ascending order")
    try:
        benchmark(options)
    except (requests.RequestException, subprocess.CalledProcessError, RuntimeError, ValueError, OSError) as error:
        print(f"Qdrant benchmark failed: {error}", file=sys.stderr)
        sys.exit(1)