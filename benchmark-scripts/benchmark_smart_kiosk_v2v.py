"""
Smart Kiosk Voice-to-Voice (V2V) Latency Benchmark

Copyright (C) 2025 Intel Corporation.

SPDX-License-Identifier: Apache-2.0

Orchestrates a voice-pipeline benchmark run for the Smart AI Kiosk the same way
``benchmark_order_accuracy.py`` orchestrates the vision pipelines: bring the
application stack up, run a fixed workload against it, collect metrics into
``results/``, tear the stack down.

NOTE ON LOCATION: this script lives here, alongside
``benchmark_order_accuracy.py``, rather than in the Smart Kiosk application
repository, so that it can bring the Smart Kiosk stack up the same way that
script brings the vision-pipeline stacks up, and so
``consolidate_multiple_run_of_metrics.py``/``usage_graph_plot.py`` in this
same directory can be chained straight after it without crossing repos.

It therefore depends on an application repository it does not live in. The
application is located via ``--app_dir`` and must provide:

* a ``Makefile`` exposing ``up``/``down`` targets (used to start and stop the
  stack, so this script does not duplicate the application's tested compose
  profile and env handling), and
* ``tests/benchmarks/v2v_scripted_conversation_benchmark.py``, which performs
  the actual measurement.

Both are validated up front, and the service URLs the health check polls are
overridable (see ``--kiosk_url``/``--rag_url``/``--ovms_url``), so an
application-side port or layout change does not require a change here.

Differences from the vision-pipeline benchmarks, and why:

* **Run-count based, not duration based.** ``benchmark_order_accuracy.py``
  sleeps for ``--duration`` while RTSP workers churn continuously. A voice turn
  is a discrete request/response, and kiosk-core's session model is one
  turn per session, so this script runs an exact number of scripted
  conversations instead. A wall-clock duration would only control *how
  many* turns happened to fit, making runs non-comparable.

* **Stack is started with the CPU-hungry vision services OFF.** rtsp-streamer +
  queue-service (YOLOv8 person counting) consume ~650-750% + ~350-500% CPU
  continuously whether or not a voice session is active, and contend with the
  voice pipeline's only CPU-bound stage (TTS). Speaker diarization is disabled
  for the same reason. See ``configure_stack_env()``.

* **A warmup run is discarded.** First-turn latency is materially higher than
  steady state (measured: ~726ms cold vs ~754ms warm on a turn whose TTS opener
  cache was still empty), so the first run would otherwise bias the median.

Usage::

    # Run from this directory, against a sibling checkout of the application:
    python benchmark_smart_kiosk_v2v.py \
        --app_dir /path/to/smart-kiosk-assistant
    python benchmark_smart_kiosk_v2v.py \
        --app_dir /path/to/smart-kiosk-assistant \
        --script order-multi --runs 5 --results_dir /tmp/v2v-results

``--app_dir`` defaults to ``../../smart-kiosk-assistant``, which resolves
correctly when this repository is checked out as a sibling of the application
(or vendored into it as a ``performance-tools/`` submodule, the layout the
application's own ``make benchmark`` uses -- it passes ``--app_dir``
explicitly in any case). Pass the path yourself for any other layout.

By default (no ``--script``), the conversation replayed is whatever the
application's own ``Sample_data/conversation.jsonl`` contains -- edit that
file in the application repo to change the workload without touching this
orchestrator. ``--script NAME`` opts back into one of the application's
legacy hardcoded named conversations (e.g. ``order-simple``, ``order-multi``)
instead.
"""

import argparse
import csv
import glob
import json
import math
import os
import re
import statistics
import subprocess  # nosec B404
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict, deque
from typing import Deque, Dict, List, Optional


def percentile(ordered: List[int], fraction: float) -> int:
    """Nearest-rank percentile of an already-sorted, non-empty sequence.

    Uses the standard nearest-rank definition, ``ceil(fraction * n)``, which
    for n=20 and fraction=0.95 selects the 19th of 20 samples. The simpler
    ``int(n * fraction)`` indexes one past that whenever ``n * fraction`` is
    a whole number, reporting the maximum as the p95.

    Args:
        ordered: Samples sorted ascending. Must not be empty.
        fraction: Percentile as a fraction in (0, 1].

    Returns:
        The sample at the nearest rank.
    """
    if not ordered:
        raise ValueError("percentile() requires at least one sample")
    rank = max(1, math.ceil(fraction * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


class SmartKioskV2VBenchmark:
    """
    Benchmark orchestrator for the Smart AI Kiosk voice-to-voice pipeline.

    Brings the kiosk stack up via the application's own Makefile (reusing its
    tested compose-profile and env handling rather than re-implementing it),
    runs the scripted conversation benchmark that ships with the application,
    and collects both that benchmark's own JSON report and the
    ``vlm_application_metrics_*.txt`` files the shared performance-tools
    collectors understand.
    """

    DEFAULT_RUNS = 8
    DEFAULT_WARMUP_RUNS = 1
    DEFAULT_INIT_DURATION = 30  # seconds, after health checks pass
    DEFAULT_HEALTH_TIMEOUT = 600  # seconds

    # Endpoints that must answer before a run is meaningful. kiosk-core is the
    # orchestrator; rag-service and OVMS back the generation stage. A run
    # started before OVMS has loaded the LLM measures model load, not latency.
    # These are defaults only -- override per run via --kiosk_url/--rag_url/
    # --ovms_url (or the SMART_KIOSK_V2V_* environment variables) so an
    # application-side port change does not require a change here.
    DEFAULT_HEALTH_ENDPOINTS = {
        "kiosk-core": "http://localhost:8012/health",
        "rag-service": "http://localhost:8020/health",
        "ovms-llm": "http://localhost:8000/v3/models",
    }

    # Passed to every `make` invocation. QUEUE=false is the application
    # Makefile's switch for leaving the vision profile out of compose
    # entirely; see configure_stack_env() for why that matters.
    DEFAULT_MAKE_ARGS = ["QUEUE=false"]

    def __init__(
        self,
        app_dir: str,
        results_dir: str,
        skip_stack: bool = False,
        keep_stack: bool = False,
        health_endpoints: Optional[Dict[str, str]] = None,
        make_args: Optional[List[str]] = None,
    ):
        self.app_dir = os.path.abspath(app_dir)
        self.results_dir = os.path.abspath(results_dir)
        self.skip_stack = skip_stack
        self.keep_stack = keep_stack
        self.health_endpoints = dict(
            health_endpoints or self.DEFAULT_HEALTH_ENDPOINTS
        )
        self.make_args = list(
            self.DEFAULT_MAKE_ARGS if make_args is None else make_args
        )

        if not os.path.isdir(self.app_dir):
            raise ValueError(
                f"Application directory not found: {self.app_dir}"
            )
        if not os.path.isfile(os.path.join(self.app_dir, "Makefile")):
            raise ValueError(
                f"No Makefile in {self.app_dir} -- is this the "
                f"smart-kiosk-assistant dir?"
            )

        os.makedirs(self.results_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Stack lifecycle
    # ------------------------------------------------------------------
    def configure_stack_env(self) -> Dict[str, str]:
        """Build the environment the stack is started with.

        Both flags are passed through the process environment (not just as make
        arguments) because docker compose gives shell environment variables
        precedence over the application's ``.env`` file -- so this forces the
        intended configuration regardless of what ``.env`` currently says.
        """
        env = os.environ.copy()
        # Frees ~10-12 CPU cores otherwise burned by rtsp-streamer +
        # queue-service. The application Makefile reads this to decide whether
        # to pass `--profile queue` to compose at all, so the containers are
        # never started rather than merely being idle.
        env["KIOSK_CORE_QUEUE_SERVICE_ENABLED"] = "false"
        # Speaker diarization (pyannote) adds analyzer latency and CPU load
        # that is irrelevant to a single-speaker scripted benchmark. compose
        # maps this onto AUDIO_ANALYZER__MODELS__ASR__DIARIZATION.
        env["KIOSK_CORE_DIARIZATION_ENABLED"] = "false"
        env["KIOSK_CORE_DIARIZATION_INTERMEDIATE_ENABLED"] = "false"
        return env

    def _run_make(
        self, target: str, extra_args: Optional[List[str]] = None
    ) -> int:
        cmd = ["make", target] + self.make_args + (extra_args or [])
        print(f"\n$ {' '.join(cmd)}  (cwd={self.app_dir})")
        # Fixed argv built from a literal target and operator-supplied make
        # arguments; no shell, so no injection surface.
        proc = subprocess.run(  # nosec B404, B603
            cmd, cwd=self.app_dir, env=self.configure_stack_env()
        )
        return proc.returncode

    def start_stack(self) -> None:
        if self.skip_stack:
            print(
                "--skip_stack set: assuming the kiosk stack is already "
                "running."
            )
            return
        print("\n" + "=" * 70)
        print(
            "Starting Smart Kiosk stack "
            "(queue-service/rtsp-streamer OFF, diarization OFF)"
        )
        print("=" * 70)
        rc = self._run_make("up")
        if rc != 0:
            raise RuntimeError(f"'make up' failed with exit code {rc}")

    def stop_stack(self) -> None:
        if self.skip_stack or self.keep_stack:
            print("Leaving the stack running (--skip_stack/--keep_stack).")
            return
        print("\nStopping stack...")
        self._run_make("down")

    def wait_for_health(self, timeout: int) -> None:
        """Block until every required endpoint answers, or raise."""
        print(f"\nWaiting up to {timeout}s for services to become healthy...")
        deadline = time.time() + timeout
        pending = dict(self.health_endpoints)

        while pending and time.time() < deadline:
            for name, url in list(pending.items()):
                try:
                    # Health URLs are http(s) endpoints supplied by the
                    # operator, not untrusted input.
                    with urllib.request.urlopen(  # nosec B310
                        url, timeout=5
                    ) as resp:
                        if 200 <= resp.status < 300:
                            elapsed = int(timeout - (deadline - time.time()))
                            print(f"  [{elapsed:>4}s] {name} healthy")
                            pending.pop(name)
                except (urllib.error.URLError, OSError, ValueError):
                    pass
            if pending:
                time.sleep(5)

        if pending:
            raise RuntimeError(
                f"Timed out after {timeout}s waiting for: "
                f"{', '.join(sorted(pending))}. "
                f"Check 'make logs' in {self.app_dir}."
            )
        print("All services healthy.")

    # ------------------------------------------------------------------
    # Workload
    # ------------------------------------------------------------------
    def run_benchmark_script(
        self,
        script: Optional[str],
        runs: int,
        label: str,
        emit_vlm_metrics: bool = True,
        explicit_end_mark: bool = False,
    ) -> int:
        """Invoke the application's own scripted conversation benchmark.

        Deliberately shells out to the in-application script rather than
        re-implementing the voice-to-voice measurement here: that script (and
        the ``v2v_fixture_benchmark`` it delegates to) reads kiosk-core's own
        server-side pipeline trace, which is the authoritative measurement.
        Duplicating it in performance-tools would create a second, divergent
        definition of "voice to voice latency".

        ``explicit_end_mark`` defaults to False. It used to be hardcoded on,
        which meant this orchestrator never measured endpointing delay at
        all: the child harness told kiosk-core the turn was over instead of
        letting it detect the pause. Endpointing is part of voice-to-voice
        latency by definition, and on the hands-free path it is the whole of
        what the customer waits through, so forcing it off produced a number
        no customer ever experiences. It is now opt-in, for isolating
        processing latency while debugging.

        Expect more run-to-run spread as a result: endpointing is bimodal
        (the completeness shortcut fires at ~0.15s, or the turn falls back to
        the full silence timeout at ~1.1s). That spread is a real property of
        the product, so it belongs in the measurement; the fix for an
        unstable p95 is more turns, not a shorter span.

        ``script`` is optional: when falsy, ``--script`` is omitted entirely
        and the child script falls back to its own default conversation
        source (``Sample_data/conversation.jsonl`` in the application repo),
        rather than this orchestrator hardcoding a named scripted
        conversation.
        """
        benchmark_path = os.path.join(
            self.app_dir,
            "tests",
            "benchmarks",
            "v2v_scripted_conversation_benchmark.py",
        )
        if not os.path.isfile(benchmark_path):
            raise FileNotFoundError(
                f"Benchmark script not found: {benchmark_path}"
            )

        cmd = [
            sys.executable,
            benchmark_path,
            "--runs", str(runs),
            "--label", label,
            "--results-dir", self.results_dir,
        ]
        if script:
            cmd += ["--script", script]
        if emit_vlm_metrics:
            cmd.append("--emit-vlm-metrics")
        if explicit_end_mark:
            cmd.append("--explicit-end-mark")

        print(f"\n$ {' '.join(cmd)}")
        # Fixed argv: the interpreter, a path this class validated above, and
        # typed/validated arguments. No shell.
        proc = subprocess.run(  # nosec B404, B603
            cmd, cwd=self.app_dir, env=self.configure_stack_env()
        )
        return proc.returncode

    def _clean_previous_metrics(self) -> None:
        """Remove stale metrics from earlier runs in this results dir.

        VLMMetricsLogger deletes files matching ``vlm_application_metrics*``
        when it first writes, but only inside the directory it was configured
        with, so a results dir that has been reused can still hold an older
        run's files.

        Deliberately a no-op under ``--skip_stack``: that mode benchmarks a
        stack this script did not start and will not stop, so kiosk-core
        already holds these files open through a RotatingFileHandler.
        Unlinking them there does not give a clean slate -- it leaves the
        handler writing to an unlinked inode, so every subsequent event
        vanishes and the run ends with "No transactions". The ``since_ms``
        filter in ``collect_vlm_logger_metrics`` already excludes the older
        run's turns, which is what this cleanup was for.
        """
        if self.skip_stack:
            print(
                "--skip_stack set: leaving existing metrics files in place "
                "(kiosk-core holds them open); stale turns are excluded by "
                "timestamp instead."
            )
            return
        patterns = (
            "vlm_application_metrics_*.txt",
            "vlm_performance_metrics_*.txt",
        )
        for pattern in patterns:
            for path in glob.glob(os.path.join(self.results_dir, pattern)):
                try:
                    os.remove(path)
                except OSError:
                    pass

    # ------------------------------------------------------------------
    # Collection
    # ------------------------------------------------------------------
    def collect_vlm_logger_metrics(
        self, since_ms: Optional[int] = None
    ) -> Dict:
        """Parse vlm_application_metrics_*.txt into latency stats.

        Mirrors benchmark_order_accuracy.py's _collect_vlm_logger_metrics so a
        voice run and a vision run produce the same shape.

        ``since_ms``: ignore any start/end event stamped before this epoch-ms
        instant. kiosk-core's own vlm_metrics_logger hook
        (kiosk_core.audio_session._emit_vlm_metrics) emits for every
        completed turn unconditionally -- including this script's own
        discarded warmup run, which shares the same long-lived kiosk-core
        process and log file as the measured run that follows it. Passing the
        wall-clock instant the measured run started isolates its turns
        without needing to touch (or be able to rotate) that file.
        """
        metrics: Dict = {
            "total_transactions": 0,
            "avg_latency_ms": 0,
            "p95_latency_ms": 0,
            "transactions": [],
        }

        log_files = glob.glob(
            os.path.join(self.results_dir, "vlm_application_metrics_*.txt")
        )
        if not log_files:
            print(f"No vlm_metrics_logger files found in {self.results_dir}")
            return metrics

        # Events per id, paired as a queue rather than stored one-per-id.
        # kiosk-core stamps every turn with the same stream id (see
        # kiosk_core.config.VLM_METRICS_STREAM_ID) so that the downstream
        # consolidator produces one aggregate row instead of one row per turn.
        # Keying "{id: timestamp}" here would therefore keep only the last
        # turn and report a single transaction for the whole run. Sorting by
        # timestamp and pairing each end with the OLDEST unmatched start
        # pairs every turn correctly, and still works for the older
        # per-turn ids.
        events: Dict[str, List[Dict]] = defaultdict(list)

        for log_file in log_files:
            try:
                with open(log_file, "r") as f:
                    for line in f:
                        id_match = re.search(r"id=(\S+)", line)
                        event_match = re.search(r"event=(\w+)", line)
                        ts_match = re.search(r"timestamp_ms=(\d+)", line)
                        if not (id_match and event_match and ts_match):
                            continue
                        event = event_match.group(1)
                        if event not in ("start", "end"):
                            continue
                        timestamp = int(ts_match.group(1))
                        if since_ms is not None and timestamp < since_ms:
                            continue
                        events[id_match.group(1)].append(
                            {"event": event, "timestamp_ms": timestamp}
                        )
            except (IOError, OSError) as e:
                print(f"Warning: could not read {log_file}: {e}")

        latencies: List[int] = []
        for unique_id, entries in events.items():
            open_starts: Deque[int] = deque()
            for entry in sorted(entries, key=lambda e: e["timestamp_ms"]):
                if entry["event"] == "start":
                    open_starts.append(entry["timestamp_ms"])
                elif open_starts:
                    # FIFO. Turns under the shared stream id are sequential
                    # today, so this is equivalent to LIFO; it is the safer
                    # default if they ever overlap, because the oldest
                    # unmatched start is the one an arriving end belongs to.
                    latency_ms = entry["timestamp_ms"] - open_starts.popleft()
                    latencies.append(latency_ms)
                    metrics["transactions"].append(
                        {"id": unique_id, "latency_ms": latency_ms}
                    )

        metrics["total_transactions"] = len(latencies)
        if latencies:
            ordered = sorted(latencies)
            metrics["avg_latency_ms"] = round(sum(ordered) / len(ordered), 1)
            metrics["median_latency_ms"] = round(statistics.median(ordered), 1)
            metrics["p95_latency_ms"] = percentile(ordered, 0.95)
            # A p95 drawn from a handful of turns is just the slowest turn:
            # with n samples the nearest-rank index only moves below the
            # maximum once n > 20. Report n alongside it so the number is
            # never read as a stable tail estimate when it cannot be one.
            metrics["p95_sample_count"] = len(ordered)
            metrics["p95_is_max"] = len(ordered) <= 20
        return metrics

    def collect_benchmark_report(self, label: str) -> Dict:
        """Read the benchmark's own JSON report (the authoritative one)."""
        path = os.path.join(self.results_dir, f"{label}.json")
        if not os.path.isfile(path):
            print(f"Benchmark report not found: {path}")
            return {}
        try:
            with open(path, "r") as f:
                report = json.load(f)
        except (IOError, ValueError) as e:
            print(f"Warning: could not parse {path}: {e}")
            return {}

        turns = report.get("turns", []) or []
        return {
            "label": report.get("label"),
            "started_at": report.get("started_at"),
            "finished_at": report.get("finished_at"),
            "turns_total": len(turns),
            "summary": report.get("summary", {}),
        }

    @staticmethod
    def cross_check(vlm: Dict, report: Dict) -> Dict:
        """Verify the emitted metrics agree with the authoritative report.

        The vlm_metrics_logger pair is synthesised from the same
        voice_to_voice_ms the benchmark report carries, so a mismatch beyond
        rounding means the emit path is broken (wrong field, wrong clock, or
        turns silently dropped) -- exactly the class of bug that would
        otherwise go unnoticed and quietly misreport the headline number.
        """
        summary = report.get("summary", {}) or {}
        reported = (
            summary.get("voice_to_voice_ms", {}).get("median")
            if isinstance(summary.get("voice_to_voice_ms"), dict)
            else summary.get("voice_to_voice_ms")
        )
        emitted = vlm.get("median_latency_ms")
        result = {"reported_median_ms": reported, "emitted_median_ms": emitted}

        if reported is None or emitted is None:
            result["status"] = "unavailable"
            return result

        delta = abs(float(reported) - float(emitted))
        result["delta_ms"] = round(delta, 1)
        # 1ms tolerance: the emitted pair is built from int-rounded epoch
        # milliseconds, so a sub-millisecond difference is expected.
        result["status"] = "ok" if delta <= 1.0 else "MISMATCH"
        return result

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def export_results(
        self, results: Dict, prefix: str = "smart_kiosk_v2v"
    ) -> None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        json_path = os.path.join(
            self.results_dir, f"{prefix}_results_{timestamp}.json"
        )
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results exported to: {json_path}")

        csv_path = os.path.join(
            self.results_dir, f"{prefix}_summary_{timestamp}.csv"
        )
        self._write_csv_summary(results, csv_path)
        print(f"Summary exported to: {csv_path}")

    @staticmethod
    def _write_csv_summary(results: Dict, csv_path: str) -> None:
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "Value"])

            def flatten(d, prefix=""):
                for k, v in d.items():
                    key = f"{prefix}{k}" if prefix else k
                    if isinstance(v, dict):
                        yield from flatten(v, f"{key}_")
                    elif isinstance(v, list):
                        yield (key, len(v))
                    else:
                        yield (key, v)

            for metric, value in flatten(results):
                writer.writerow([metric, value])

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------
    def run(
        self,
        script: Optional[str],
        runs: int,
        warmup_runs: int,
        init_duration: int,
        health_timeout: int,
        label: str,
        explicit_end_mark: bool = False,
    ) -> Dict:
        print("\n" + "=" * 70)
        print("Smart Kiosk Voice-to-Voice Benchmark")
        default_src = '(default: Sample_data/conversation.jsonl)'
        print(f"Script: {script or default_src}")
        print(f"Runs: {runs} (plus {warmup_runs} discarded warmup)")
        print(f"Results: {self.results_dir}")
        print("=" * 70)

        self._clean_previous_metrics()
        # Bound before the try block so the `finally` below can tell "the run
        # never reached the measured phase" from "it did", without raising a
        # NameError on top of whatever actually went wrong.
        measured_run_started_ms = None

        try:
            # Inside the try so that a stack which came up partially before
            # `make up` failed is still torn down by the `finally`.
            # stop_stack() is safe to call when nothing started.
            self.start_stack()
            self.wait_for_health(health_timeout)
            if init_duration > 0:
                print(f"Settling for {init_duration}s before warmup...")
                time.sleep(init_duration)

            if warmup_runs > 0:
                print(f"\n--- Warmup ({warmup_runs} run(s), discarded) ---")
                warmup_rc = self.run_benchmark_script(
                    script=script,
                    runs=warmup_runs,
                    label=f"{label}-warmup",
                    emit_vlm_metrics=False,
                    explicit_end_mark=explicit_end_mark,
                )
                # The warmup exercises exactly the same path as the measured
                # run, so a failure here is a failure of the whole run. Left
                # unchecked it surfaces much later as a bare "No
                # transactions", hiding the real cause.
                if warmup_rc != 0:
                    raise RuntimeError(
                        f"Warmup run failed with exit code {warmup_rc}; "
                        f"not starting the measured run."
                    )

            # kiosk-core's own vlm_metrics_logger hook (kiosk_core.
            # audio_session._emit_vlm_metrics) emits for EVERY completed turn
            # unconditionally, including the warmup run above -- production
            # instrumentation has no concept of "warmup". Deleting
            # vlm_application_metrics_*.txt here (as an earlier version of
            # this fix tried) does NOT work: kiosk-core is one long-lived
            # process across both runs, and Python's RotatingFileHandler keeps
            # writing to the now-unlinked inode once its underlying file is
            # removed from under it -- every subsequent write silently
            # vanishes (nothing in --results-dir, no error). So instead:
            # remember the wall-clock instant the measured run starts and
            # have collect_vlm_logger_metrics() below ignore any entry
            # stamped before it, which isolates the measured turns without
            # ever touching the file kiosk-core has open.
            measured_run_started_ms = int(time.time() * 1000)

            print(f"\n--- Measured run ({runs} run(s)) ---")
            rc = self.run_benchmark_script(
                script=script,
                runs=runs,
                label=label,
                explicit_end_mark=explicit_end_mark,
            )

            measured_run_ended_ms = int(time.time() * 1000)
            # The hardware counters in ./metrics/ cover the collector's whole
            # lifetime -- image pulls, the health wait, the warmup -- so an
            # average taken over the raw files describes a mostly-idle stack
            # rather than the measured turns. Record the window so the
            # application side can trim them before consolidating.
            self.write_measured_window(
                measured_run_started_ms, measured_run_ended_ms
            )

            vlm = self.collect_vlm_logger_metrics(
                since_ms=measured_run_started_ms
            )
            report = self.collect_benchmark_report(label)

            results = {
                "benchmark": "smart_kiosk_v2v",
                "script": script,
                "runs": runs,
                "warmup_runs": warmup_runs,
                "explicit_end_mark": explicit_end_mark,
                "stack": {
                    "queue_service": "disabled",
                    "rtsp_streamer": "disabled",
                    "diarization": "disabled",
                },
                "benchmark_report": report,
                "vlm_metrics": vlm,
                "cross_check": self.cross_check(vlm, report),
                "benchmark_exit_code": rc,
            }

            self.export_results(results)
            self._print_summary(results)
            return results
        finally:
            self.stop_stack()
            # Only safe once the stack is down. kiosk-core holds these files
            # open through a RotatingFileHandler for the whole run, so they
            # cannot be rewritten (or removed) while it is alive without its
            # subsequent writes vanishing into an unlinked inode.
            if measured_run_started_ms is not None:
                self.prune_warmup_from_vlm_logs(measured_run_started_ms)

    def write_measured_window(self, start_ms: int, end_ms: int) -> None:
        """Record the epoch-ms span the measured turns ran over.

        Consumed by the application's scripts/trim_metrics_to_window.py so the
        CPU/GPU/NPU counters can be narrowed to the measured turns instead of
        being averaged over the collector's entire lifetime.

        Args:
            start_ms: Epoch ms the measured run started.
            end_ms: Epoch ms the measured run finished.
        """
        path = os.path.join(self.results_dir, "measured_window.json")
        try:
            with open(path, "w") as f:
                json.dump(
                    {
                        "start_ms": start_ms,
                        "end_ms": end_ms,
                        "duration_s": round((end_ms - start_ms) / 1000, 1),
                    },
                    f,
                    indent=2,
                )
            print(f"Wrote {path}")
        except (IOError, OSError) as e:
            # Best-effort: without this the counters stay un-trimmed, which is
            # the old behaviour, not a failed run.
            print(f"Warning: could not write {path}: {e}")

    def prune_warmup_from_vlm_logs(self, since_ms: int) -> None:
        """Drop pre-measurement events from the emitted vlm metrics logs.

        kiosk-core emits a start/end pair for *every* completed turn --
        production instrumentation has no concept of a warmup -- so the
        discarded warmup turns land in the same log file as the measured ones.
        This script filters them out of its own summary via ``since_ms``, but
        the downstream ``make consolidate-metrics`` step reads these files
        directly and has no way to tell the two apart, so warmup turns would
        otherwise be averaged into consolidated_metrics.csv.

        The unfiltered original is kept alongside with a ``.with-warmup``
        suffix -- deliberately not ``.txt``, so the consolidator's
        ``vlm_application_metrics_*.txt`` glob does not pick it back up.

        Args:
            since_ms: Epoch-ms instant the measured run started. Events
                stamped before this belong to the warmup.
        """
        for log_file in glob.glob(
            os.path.join(self.results_dir, "vlm_application_metrics_*.txt")
        ):
            try:
                with open(log_file, "r") as f:
                    lines = f.readlines()

                kept = []
                dropped = 0
                for line in lines:
                    ts_match = re.search(r"timestamp_ms=(\d+)", line)
                    # Lines without a timestamp are not measurements; keep
                    # them rather than silently discarding unknown content.
                    if ts_match and int(ts_match.group(1)) < since_ms:
                        dropped += 1
                        continue
                    kept.append(line)

                if not dropped:
                    continue

                # Write the filtered copy to a temporary file FIRST, then
                # preserve the original and move the new file into place.
                # Renaming the original out of the way before writing would
                # leave no *.txt at all if the write then failed, and the
                # consolidator globs *.txt -- so a tidying step could destroy
                # the very metrics the run produced.
                tmp_path = log_file + ".pruned.tmp"
                with open(tmp_path, "w") as f:
                    f.writelines(kept)
                os.replace(log_file, log_file + ".with-warmup")
                os.replace(tmp_path, log_file)
                print(
                    f"Pruned {dropped} warmup event(s) from "
                    f"{os.path.basename(log_file)} "
                    f"(original kept as *.txt.with-warmup)"
                )
            except (IOError, OSError) as e:
                # Never fail a completed run over a metrics-tidying step.
                print(f"Warning: could not prune {log_file}: {e}")
                try:
                    os.remove(log_file + ".pruned.tmp")
                except OSError:
                    pass

    @staticmethod
    def _print_summary(results: Dict) -> None:
        vlm = results.get("vlm_metrics", {})
        check = results.get("cross_check", {})
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Transactions   : {vlm.get('total_transactions', 0)}")
        print(f"Median v2v     : {vlm.get('median_latency_ms')} ms")
        print(f"Average v2v    : {vlm.get('avg_latency_ms')} ms")
        p95_note = " (= max; too few samples)" if vlm.get("p95_is_max") else ""
        print(f"p95 v2v        : {vlm.get('p95_latency_ms')} ms{p95_note}")
        print(
            f"Cross-check    : {check.get('status')} "
            f"(delta={check.get('delta_ms')} ms)"
        )
        if check.get("status") == "MISMATCH":
            print(
                "  WARNING: emitted metrics disagree with the "
                "benchmark's own report.\n"
                "  Treat the report JSON as authoritative and "
                "investigate the emit path."
            )
        print("=" * 70)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Smart Kiosk voice-to-voice latency benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--app_dir",
        default="../../smart-kiosk-assistant",
        help=(
            "Path to the smart-kiosk-assistant directory "
            "(default: ../../smart-kiosk-assistant)"
        ),
    )
    parser.add_argument(
        "--results_dir",
        default=None,
        help="Where to write results (default: <app_dir>/results)",
    )
    parser.add_argument(
        "--script",
        default=None,
        help=(
            "Legacy named scripted conversation to replay (e.g. order-simple, "
            "order-multi). If omitted (the default), the application's own "
            "Sample_data/conversation.jsonl is used instead."
        ),
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=SmartKioskV2VBenchmark.DEFAULT_RUNS,
        help="Measured repetitions of the script (default: 8)",
    )
    parser.add_argument(
        "--explicit_end_mark",
        action="store_true",
        help=(
            "Signal end-of-turn explicitly instead of letting kiosk-core "
            "detect it. This REMOVES endpointing delay from the measurement, "
            "so the reported voice-to-voice latency is no longer what a "
            "customer waits. Use it only to isolate processing latency when "
            "debugging the pipeline; never for a reported benchmark number."
        ),
    )
    parser.add_argument(
        "--warmup_runs",
        type=int,
        default=SmartKioskV2VBenchmark.DEFAULT_WARMUP_RUNS,
        help="Discarded warmup repetitions (default: 1)",
    )
    parser.add_argument(
        "--init_duration",
        type=int,
        default=SmartKioskV2VBenchmark.DEFAULT_INIT_DURATION,
        help="Settle time after health checks pass, in seconds (default: 30)",
    )
    parser.add_argument(
        "--health_timeout",
        type=int,
        default=SmartKioskV2VBenchmark.DEFAULT_HEALTH_TIMEOUT,
        help=(
            "Max seconds to wait for services to become healthy "
            "(default: 600)"
        ),
    )
    parser.add_argument(
        "--label",
        default="v2v-perf-tools",
        help="Result filename label",
    )
    parser.add_argument(
        "--skip_stack",
        action="store_true",
        help=(
            "Do not start/stop the stack -- benchmark whatever is "
            "already running"
        ),
    )
    parser.add_argument(
        "--keep_stack",
        action="store_true",
        help="Start the stack but leave it running afterwards",
    )
    parser.add_argument(
        "--kiosk_url",
        default=os.environ.get("SMART_KIOSK_V2V_KIOSK_URL"),
        help=(
            "kiosk-core health URL (default: "
            "http://localhost:8012/health, env SMART_KIOSK_V2V_KIOSK_URL)"
        ),
    )
    parser.add_argument(
        "--rag_url",
        default=os.environ.get("SMART_KIOSK_V2V_RAG_URL"),
        help=(
            "rag-service health URL (default: "
            "http://localhost:8020/health, env SMART_KIOSK_V2V_RAG_URL)"
        ),
    )
    parser.add_argument(
        "--ovms_url",
        default=os.environ.get("SMART_KIOSK_V2V_OVMS_URL"),
        help=(
            "OVMS model-list URL (default: "
            "http://localhost:8000/v3/models, env SMART_KIOSK_V2V_OVMS_URL)"
        ),
    )
    parser.add_argument(
        "--make_arg",
        action="append",
        dest="make_args",
        default=None,
        metavar="VAR=VALUE",
        help=(
            "Argument passed to every `make` invocation. Repeatable. "
            "Replaces the default QUEUE=false, so re-pass it if you still "
            "want the vision profile left out."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    app_dir = os.path.abspath(args.app_dir)
    results_dir = args.results_dir or os.path.join(app_dir, "results")

    health_endpoints = dict(
        SmartKioskV2VBenchmark.DEFAULT_HEALTH_ENDPOINTS
    )
    for name, override in (
        ("kiosk-core", args.kiosk_url),
        ("rag-service", args.rag_url),
        ("ovms-llm", args.ovms_url),
    ):
        if override:
            health_endpoints[name] = override

    try:
        benchmark = SmartKioskV2VBenchmark(
            app_dir=app_dir,
            results_dir=results_dir,
            skip_stack=args.skip_stack,
            keep_stack=args.keep_stack,
            health_endpoints=health_endpoints,
            make_args=args.make_args,
        )
        results = benchmark.run(
            script=args.script,
            runs=args.runs,
            warmup_runs=args.warmup_runs,
            init_duration=args.init_duration,
            health_timeout=args.health_timeout,
            explicit_end_mark=args.explicit_end_mark,
            label=args.label,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"\nBenchmark failed: {exc}", file=sys.stderr)
        return 1

    if results.get("vlm_metrics", {}).get("total_transactions", 0) == 0:
        print(
            "\nNo transactions were recorded -- treating as failure.",
            file=sys.stderr,
        )
        return 1
    return results.get("benchmark_exit_code", 0)


if __name__ == "__main__":
    sys.exit(main())
