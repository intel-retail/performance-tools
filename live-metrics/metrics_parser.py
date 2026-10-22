'''
* Copyright (C) 2026 Intel Corporation.
*
* SPDX-License-Identifier: Apache-2.0
'''

"""
Parses the metric files written by docker/scripts/collect_platform.sh,
collect_gpu.sh and collect_npu.py into JSON time series.

Used by metrics_api.py, served from inside the Docker image (see
docker/supervisord.conf's [program:metrics_api]). Only ever reads files
under $RESULTS_DIR; paths are re-read from the environment on every call
(not cached at import time) so a long-lived process can serve multiple
sessions/RESULTS_DIR values over its lifetime.
"""

import csv
import glob
import json
import math
import os
import shutil
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    _GPU_MAX_JSON_BYTES = int(float(os.getenv("METRICS_GPU_MAX_JSON_MB", "256")) * 1024 * 1024)
except (TypeError, ValueError):
    _GPU_MAX_JSON_BYTES = 256 * 1024 * 1024

try:
    _GPU_MAX_POINTS = int(os.getenv("METRICS_GPU_MAX_POINTS", "300"))
except (TypeError, ValueError):
    _GPU_MAX_POINTS = 300


def _results_dir() -> Path:
    return Path(os.getenv("RESULTS_DIR", "/tmp/results"))


def _cpu_log() -> Path:
    return _results_dir() / "cpu_usage.log"


def _mem_log() -> Path:
    return _results_dir() / "memory_usage.log"


def _npu_csv() -> Path:
    return Path(os.getenv("NPU_LOG", str(_results_dir() / "npu_usage.csv")))


def _pcm_csv() -> Path:
    return _results_dir() / "pcm.csv"


def build_cpu_series() -> List[List]:
    """
    Parse cpu_usage.log (sar -u 1 output). Each data row ends with %idle;
    usage = 100 - idle. Timestamps are approximated backward from now,
    assuming the 1-second sampling interval sar was started with.
    Returns [[timestamp_iso, usage_percent], ...]
    """
    try:
        lines = [l.strip() for l in _cpu_log().open() if l.strip()]
    except (FileNotFoundError, OSError):
        return []

    samples: List[float] = []
    for line in lines:
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            idle = float(parts[-1])
            samples.append(max(0.0, min(100.0, 100.0 - idle)))
        except ValueError:
            continue  # header / label rows from sar -- skip

    if not samples:
        return []
    start = datetime.now() - timedelta(seconds=len(samples) - 1)
    return [[(start + timedelta(seconds=i)).isoformat(), v] for i, v in enumerate(samples)]


def build_memory_series() -> List[List]:
    """
    Parse memory_usage.log (free -s 1 output). Each 'Mem:' row is
    total used free ... (KiB). Returns
    [[timestamp_iso, total_gb, used_gb, free_gb, usage_percent], ...]
    """
    try:
        lines = [l.strip() for l in _mem_log().open() if l.lstrip().startswith("Mem:")]
    except (FileNotFoundError, OSError):
        return []

    samples = []
    for line in lines:
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            total_kib, used_kib, free_kib = float(parts[1]), float(parts[2]), float(parts[3])
            pct = (used_kib / total_kib * 100.0) if total_kib > 0 else 0.0
            samples.append((total_kib / 1024 ** 2, used_kib / 1024 ** 2, free_kib / 1024 ** 2, pct))
        except ValueError:
            continue

    if not samples:
        return []
    start = datetime.now() - timedelta(seconds=len(samples) - 1)
    return [[(start + timedelta(seconds=i)).isoformat(), *s] for i, s in enumerate(samples)]


def build_npu_series() -> List[List]:
    """
    Parse npu_usage.csv written by collect_npu.py: header + rows of
    timestamp_iso,percent_usage. Returns [[timestamp_iso, usage_percent], ...]
    """
    try:
        lines = [l.strip() for l in _npu_csv().open() if l.strip()]
    except (FileNotFoundError, OSError):
        return []
    if len(lines) <= 1:
        return []

    series: List[List] = []
    for line in lines[1:]:  # skip header row
        try:
            ts, usage = line.split(",", 1)
            series.append([ts.strip(), float(usage)])
        except ValueError:
            continue
    return series


def _load_qmassa_json(path: str) -> Dict[str, Any]:
    """
    qmassa's on-disk format changed between versions: v1.0 wrote a single
    JSON document; v1.3+ (the version docker/Dockerfile pins) writes JSONL
    instead -- one JSON object per line (a version line, a config line,
    then one state line per sample). Try the simple case first, fall back
    to JSONL -- mirrors
    benchmark-scripts/parse_qmassa_metrics_to_json.py::load_qmassa_json.
    """
    with open(path) as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            pass

    states = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and "devs_state" in obj:
                states.append(obj)
    if not states:
        raise ValueError(f"no qmassa state data found in {path}")
    return {"states": states}


def build_gpu_series() -> List[List]:
    """
    Parse the qmassa tool-generated JSON (docker/scripts/collect_gpu.sh output):
      qmassa<card>-<deviceid>-<driver>-tool-generated.json
    Returns [[timestamp_iso, busy_percent], ...] -- the busiest engine per
    sample, since qmassa's engine key names differ by driver (i915 uses
    'compute'/'render'/'copy'/'video'/'video-enhance'; xe uses 'ccs'/'rcs'/
    'bcs'/'vcs'/'vecs').

    Bounded by size/point-count: qmassa's JSON grows unbounded while it
    keeps running (observed ~600 MB/hour in a 24/7 deployment), so a
    pathologically large or mid-write file is skipped rather than risking
    an OOM on json.load(). Use QMASSA_CYCLE_SECONDS in collect_gpu.sh (see
    docker/scripts/collect_gpu.sh) to keep the file itself bounded too.
    """
    candidates = glob.glob(str(_results_dir() / "qmassa*-tool-generated.json"))
    if not candidates:
        return []
    latest = max(candidates, key=os.path.getmtime)

    try:
        if os.path.getsize(latest) > _GPU_MAX_JSON_BYTES:
            return []
        data = _load_qmassa_json(latest)
    except (OSError, ValueError):
        return []  # qmassa was mid-write, or wrote neither known format

    states = data.get("states") or []
    if not isinstance(states, list) or not states:
        return []
    if len(states) > _GPU_MAX_POINTS:
        states = states[-_GPU_MAX_POINTS:]

    try:
        ms_interval = int(data.get("args", {}).get("ms_interval", 1500))
    except (TypeError, ValueError):
        ms_interval = 1500
    dt_seconds = max(ms_interval / 1000.0, 0.1)

    samples: List[float] = []
    for state in states:
        try:
            devs_state = state.get("devs_state") or []
            if not devs_state:
                continue
            eng_usage = (devs_state[0].get("dev_stats") or {}).get("eng_usage") or {}
            busy = max((float(v[-1]) for v in eng_usage.values() if v), default=0.0)
            samples.append(max(0.0, min(100.0, busy)))
        except (IndexError, TypeError, ValueError):
            continue

    if not samples:
        return []
    start = datetime.now() - timedelta(seconds=dt_seconds * (len(samples) - 1))
    return [
        [(start + timedelta(seconds=dt_seconds * i)).isoformat(), v]
        for i, v in enumerate(samples)
    ]


def build_power_series() -> List[List]:
    """
    Parse pcm.csv (Intel PCM output from `pcm 1 -silent -r -nc -nsys -csv=...`,
    see docker/scripts/collect_platform.sh). Mirrors the column-matching
    approach benchmark-scripts/consolidate_multiple_run_of_metrics.py's
    PCMExtractor already uses against real captured PCM output: the first
    CSV row already contains directly-usable column names for simple,
    ungrouped columns such as 'Proc Energy (Joules)' (only per-socket
    breakdown columns need the second header row, which this function does
    not need). PCM samples once per second (`pcm 1 ...`), so each row's
    Joules value is numerically already Watts for that interval -- no
    differencing between rows is needed. Returns
    [[timestamp_iso, package0_watts, package1_watts, ...], ...]
    Returns [] if pcm.csv is unavailable (e.g. PCM not installed/no MSR access).
    """
    try:
        with _pcm_csv().open() as f:
            reader = csv.reader(f)
            header = next(reader, None)
            rows = list(reader)
    except (FileNotFoundError, OSError):
        return []
    if not header or len(rows) < 2:
        return []

    energy_cols = [i for i, c in enumerate(header) if "Proc Energy" in c]
    if not energy_cols:
        return []

    series: List[List] = []
    total_rows = len(rows)
    start = datetime.now() - timedelta(seconds=total_rows - 1)
    for i, row in enumerate(rows):
        try:
            watts = [float(row[c]) for c in energy_cols]
        except (ValueError, IndexError):
            continue
        series.append([(start + timedelta(seconds=i)).isoformat(), *watts])
    return series


def parse_memory_usage() -> Optional[Dict[str, Any]]:
    """Return a single latest memory snapshot (for GET /memory)."""
    series = build_memory_series()
    if not series:
        return None
    ts, total_gb, used_gb, free_gb, pct = series[-1]
    return {
        "timestamp": ts,
        "total_gb": total_gb,
        "used_gb": used_gb,
        "free_gb": free_gb,
        "usage_percent": pct,
    }


def get_platform_info() -> Dict[str, Any]:
    """Return a hardware summary: Processor, iGPU, NPU, Memory, Storage."""

    def _format_gb(size_bytes: int) -> str:
        return f"{math.ceil(size_bytes / 1024 ** 3)} GB"

    processor = "Intel Processor"
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    processor = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass

    igpu, npu = "Not detected", "Not detected"
    try:
        out = subprocess.run(["lspci", "-nn"], capture_output=True, text=True, timeout=5).stdout
        for line in out.splitlines():
            if "VGA" in line and "Intel" in line:
                igpu = line.split(": ", 1)[-1].strip()
            if "AI Boost" in line or "NPU" in line.upper():
                npu = line.split(": ", 1)[-1].strip()
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        pass

    memory = "--"
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    memory = _format_gb(int(line.split()[1]) * 1024)
                    break
    except OSError:
        pass

    storage = "--"
    try:
        storage = f"{shutil.disk_usage('/').total / 1024 ** 3 / 931:.2f} TB"
    except OSError:
        pass

    return {"Processor": processor, "iGPU": igpu, "NPU": npu, "Memory": memory, "Storage": storage}


def build_device_config_payload() -> Dict[str, Any]:
    """
    Optional, generic per-workload device summary. Reads simple KEY=VALUE
    lines (e.g. DETECT_DEVICE=GPU) from the file at $DEVICE_CONFIG_PATH and
    resolves each value against get_platform_info() where it names a class
    of device (CPU/GPU/NPU). Consumers that don't set DEVICE_CONFIG_PATH
    (e.g. education-ai-suite) simply get {} back -- this endpoint carries
    no required behavior.
    """
    path = os.getenv("DEVICE_CONFIG_PATH")
    if not path or not os.path.isfile(path):
        return {}

    raw: Dict[str, str] = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                raw[key.strip()] = value.strip().strip('"').strip("'")
    except OSError:
        return {}

    platform_info = get_platform_info()
    resolved = {}
    for key, device_code in raw.items():
        code = (device_code or "").strip().upper()
        resolved[key] = {
            "CPU": platform_info.get("Processor", "CPU"),
            "GPU": platform_info.get("iGPU", "GPU"),
            "NPU": platform_info.get("NPU", "NPU"),
        }.get(code, device_code)

    return {"devices": raw, "resolved": resolved}


def build_metrics_payload(window: int = 60) -> Dict[str, Any]:
    """
    Assemble the full payload returned by GET /metrics. Only the last
    `window` samples of each series are returned so the JSON stays small
    (comfortably under typical UI poll timeouts) even if a collector has
    been running a long time.
    """

    def tail(series: List[List], n: int) -> List[List]:
        return series[-n:] if len(series) > n else series

    return {
        "cpu_utilization": tail(build_cpu_series(), window),
        "gpu_utilization": tail(build_gpu_series(), window),
        "npu_utilization": tail(build_npu_series(), window),
        "memory": tail(build_memory_series(), window),
        "power": tail(build_power_series(), window),
    }
