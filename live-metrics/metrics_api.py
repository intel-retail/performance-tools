'''
* Copyright (C) 2026 Intel Corporation.
*
* SPDX-License-Identifier: Apache-2.0
'''

"""
Live metrics HTTP API for performance-tools.

Served inside the Docker image alongside the collectors (see
docker/supervisord.conf's [program:metrics_api]); this process only ever
reads $RESULTS_DIR, it has no other dependency on how the collectors were
started.

Endpoints
---------
GET /health          -> 200 {"status": "ok"}
GET /metrics         -> time-series JSON (cpu / gpu / npu / memory / power)
GET /platform-info   -> hardware summary (processor, iGPU, NPU, memory, storage)
GET /memory          -> latest single memory snapshot
GET /device-config   -> optional per-workload device summary (see metrics_parser.build_device_config_payload)
"""

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from metrics_parser import (
    build_device_config_payload,
    build_metrics_payload,
    get_platform_info,
    parse_memory_usage,
)

app = FastAPI(title="performance-tools live-metrics")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
         o.strip()
         for o in os.getenv("METRICS_CORS_ORIGINS", "*").split(",")
         if o.strip()
     ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/metrics")
def metrics() -> dict:
    """Time-series utilization data: CPU, GPU, NPU, memory, power."""
    return build_metrics_payload()


@app.get("/platform-info")
def platform_info() -> dict:
    """Hardware summary: processor, iGPU, NPU, memory, storage."""
    return get_platform_info()


@app.get("/memory")
def memory() -> dict:
    """Latest single memory snapshot."""
    data = parse_memory_usage()
    return data if data is not None else {"error": "no memory data"}


@app.get("/device-config")
def device_config() -> dict:
    """Optional per-workload device summary; {} if not configured."""
    return build_device_config_payload()


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("METRICS_HTTP_PORT", "9000"))
    print(
        f"[live-metrics] starting on 0.0.0.0:{port}  "
        f"endpoints: /health /metrics /platform-info /memory /device-config",
        flush=True,
    )
    uvicorn.run("metrics_api:app", host="0.0.0.0", port=port, reload=False)
