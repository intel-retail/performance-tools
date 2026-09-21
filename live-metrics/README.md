# live-metrics

A small HTTP API that turns the flat files written by
`docker/scripts/collect_platform.sh`, `collect_gpu.sh` and `collect_npu.py`
into a live, pollable JSON time series (CPU / GPU / NPU / memory / power).

`performance-tools` already ran those collectors inside a privileged Docker
container (`docker/Dockerfile` + `docker/docker-compose.yaml`, used today by
`loss-prevention` as a submodule) for **offline benchmarking** --
`benchmark-scripts/consolidate_multiple_run_of_metrics.py` only parses the
results *after* a run finishes. `live-metrics/` adds a second way to consume
the same collectors: a **live HTTP API** (`metrics_api.py` /
`metrics_parser.py`), served from inside the same container via an
additional `supervisord` program, so a UI can poll `GET /metrics`
continuously while the collectors keep running -- not just after a
benchmark completes.

Both use cases run **the exact same collector scripts** under
`docker/scripts/` -- `live-metrics/` only adds a reader on top, it doesn't
change what's collected or how.

## Running it

```bash
cd docker
log_dir=./results docker compose build   # picks up live-metrics/ automatically, see docker/Dockerfile
log_dir=./results docker compose up -d
curl http://localhost:9000/metrics | jq
```
`docker-compose.yaml` uses `network_mode: host`, so port 9000 is reachable
directly on the host once the container is up — no port publishing needed.

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `RESULTS_DIR` | `/tmp/results` | Where the collectors write, and where `metrics_parser.py` reads from. |
| `NPU_LOG` | `${RESULTS_DIR}/npu_usage.csv` | Override if you need the NPU CSV somewhere else. |
| `METRICS_HTTP_PORT` | `9000` | Port `metrics_api.py` listens on. |
| `DEVICE_CONFIG_PATH` | unset | Optional path to a `KEY=VALUE` file for `GET /device-config`; returns `{}` if unset. |
| `QMASSA_CYCLE_SECONDS` | `0` (disabled) | If set > 0, `collect_gpu.sh` restarts qmassa every N seconds and deletes its output file first, keeping the JSON bounded for long-running live dashboards. `0` reproduces the original, unbounded, single-invocation behavior used by existing offline-benchmarking consumers. **Set this for any 24/7 live-dashboard deployment** (e.g. `180`) -- qmassa never rotates its own output file, and it will otherwise grow unbounded (observed ~600 MB/hour in one production deployment). |
| `METRICS_GPU_MAX_JSON_MB` | `256` | `build_gpu_series()` refuses to `json.load()` a qmassa file larger than this (defense in depth alongside `QMASSA_CYCLE_SECONDS`). |
| `METRICS_GPU_MAX_POINTS` | `300` | Caps how many qmassa samples are parsed per request (only the tail is ever plotted). |

## Endpoints

| Method & path | Returns |
|---|---|
| `GET /health` | `{"status": "ok"}` |
| `GET /metrics` | `{"cpu_utilization": [...], "gpu_utilization": [...], "npu_utilization": [...], "memory": [...], "power": [...]}` |
| `GET /platform-info` | `{"Processor": "...", "iGPU": "...", "NPU": "...", "Memory": "...", "Storage": "..."}` |
| `GET /memory` | Latest single memory snapshot |
| `GET /device-config` | `{}` unless `DEVICE_CONFIG_PATH` is set (see above) |