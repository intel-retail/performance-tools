"""
Smart Kiosk Voice-to-Voice (V2V) Latency Benchmark

Orchestrates a voice-pipeline benchmark run for the Smart AI Kiosk the same way
``benchmark_order_accuracy.py`` (in the ``performance-tools`` submodule)
orchestrates the vision pipelines: bring the application stack up, run a fixed
workload against it, collect metrics into ``results/``, tear the stack down.

NOTE ON LOCATION: this script lives inside the ``performance-tools``
submodule (``performance-tools/benchmark-scripts/``), alongside
``benchmark_order_accuracy.py``, so that it can bring the Smart Kiosk stack up
the same way that script brings the vision-pipeline stacks up, and so
``consolidate_multiple_run_of_metrics.py``/``usage_graph_plot.py`` in this
same directory can be chained straight after it without crossing repos.

This is a deliberate trade-off: it is NOT upstreamed into
``intel-retail/performance-tools``, so from this submodule's perspective it is
an untracked file sitting on top of a pinned upstream commit. Concretely:

* ``git status`` inside ``performance-tools/`` will show this file as ``??``.
* ``make update-submodules`` (``git submodule update --init --recursive
  --remote performance-tools``) resets the submodule's working tree to the
  pinned upstream commit and DOES wipe this file if the pinned commit changes.
  A copy of this exact file is tracked in the application repo at
  ``smart-kiosk-assistant/tests/benchmarks/perf-tools-orchestrator/`` purely
  as a recovery backup (not the copy that runs) -- if this file goes missing,
  copy it back from there. ``make update-submodules`` does this
  automatically; see the Makefile.
* If this is ever upstreamed into ``intel-retail/performance-tools`` proper,
  delete both the backup and this note.

Differences from the vision-pipeline benchmarks, and why:

* **Run-count based, not duration based.** ``benchmark_order_accuracy.py``
  sleeps for ``--duration`` while RTSP workers churn continuously. A voice turn
  is a discrete request/response, and kiosk-core's session model is one turn per
  session, so this script runs an exact number of scripted conversations
  instead. A wall-clock duration would only control *how many* turns happened
  to fit, making runs non-comparable.

* **Stack is started with the CPU-hungry vision services OFF.** rtsp-streamer +
  queue-service (YOLOv8 person counting) consume ~650-750% + ~350-500% CPU
  continuously whether or not a voice session is active, and contend with the
  voice pipeline's only CPU-bound stage (TTS). Speaker diarization is disabled
  for the same reason. See ``configure_stack_env()``.

* **A warmup run is discarded.** First-turn latency is materially higher than
  steady state (measured: ~726ms cold vs ~754ms warm on a turn whose TTS opener
  cache was still empty), so the first run would otherwise bias the median.

Usage::

    python benchmark_smart_kiosk_v2v.py --app_dir ../../..
    python benchmark_smart_kiosk_v2v.py --app_dir ../../.. \
        --script order-multi --runs 5 --results_dir /tmp/v2v-results

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
import os
import re
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Dict, List, Optional


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

    DEFAULT_RUNS = 12
    DEFAULT_WARMUP_RUNS = 1
    DEFAULT_INIT_DURATION = 30  # seconds, after health checks pass
    DEFAULT_HEALTH_TIMEOUT = 600  # seconds

    # Endpoints that must answer before a run is meaningful. kiosk-core is the
    # orchestrator; rag-service and OVMS back the generation stage. A run
    # started before OVMS has loaded the LLM measures model load, not latency.
    HEALTH_ENDPOINTS = {
        "kiosk-core": "http://localhost:8012/health",
        "rag-service": "http://localhost:8020/health",
        "ovms-llm": "http://localhost:8000/v3/models",
    }

    def __init__(
        self,
        app_dir: str,
        results_dir: str,
        skip_stack: bool = False,
        keep_stack: bool = False,
    ):
        self.app_dir = os.path.abspath(app_dir)
        self.results_dir = os.path.abspath(results_dir)
        self.skip_stack = skip_stack
        self.keep_stack = keep_stack

        if not os.path.isdir(self.app_dir):
            raise ValueError(f"Application directory not found: {self.app_dir}")
        if not os.path.isfile(os.path.join(self.app_dir, "Makefile")):
            raise ValueError(f"No Makefile in {self.app_dir} -- is this the smart-kiosk-assistant dir?")

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

    def _run_make(self, target: str, extra_args: Optional[List[str]] = None) -> int:
        cmd = ["make", target, "QUEUE=false"] + (extra_args or [])
        print(f"\n$ {' '.join(cmd)}  (cwd={self.app_dir})")
        proc = subprocess.run(cmd, cwd=self.app_dir, env=self.configure_stack_env())
        return proc.returncode

    def start_stack(self) -> None:
        if self.skip_stack:
            print("--skip_stack set: assuming the kiosk stack is already running.")
            return
        print("\n" + "=" * 70)
        print("Starting Smart Kiosk stack (queue-service/rtsp-streamer OFF, diarization OFF)")
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
        pending = dict(self.HEALTH_ENDPOINTS)

        while pending and time.time() < deadline:
            for name, url in list(pending.items()):
                try:
                    with urllib.request.urlopen(url, timeout=5) as resp:
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
                f"Timed out after {timeout}s waiting for: {', '.join(sorted(pending))}. "
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
    ) -> int:
        """Invoke the application's own scripted conversation benchmark.

        Deliberately shells out to the in-application script rather than
        re-implementing the voice-to-voice measurement here: that script (and
        the ``v2v_fixture_benchmark`` it delegates to) reads kiosk-core's own
        server-side pipeline trace, which is the authoritative measurement.
        Duplicating it in performance-tools would create a second, divergent
        definition of "voice to voice latency".

        ``script`` is optional: when falsy, ``--script`` is omitted entirely
        and the child script falls back to its own default conversation
        source (``Sample_data/conversation.jsonl`` in the application repo),
        rather than this orchestrator hardcoding a named scripted
        conversation.
        """
        benchmark_path = os.path.join(
            self.app_dir, "tests", "benchmarks", "v2v_scripted_conversation_benchmark.py"
        )
        if not os.path.isfile(benchmark_path):
            raise FileNotFoundError(f"Benchmark script not found: {benchmark_path}")

        cmd = [
            sys.executable,
            benchmark_path,
            "--runs", str(runs),
            "--explicit-end-mark",
            "--label", label,
            "--results-dir", self.results_dir,
        ]
        if script:
            cmd += ["--script", script]
        if emit_vlm_metrics:
            cmd.append("--emit-vlm-metrics")

        print(f"\n$ {' '.join(cmd)}")
        proc = subprocess.run(cmd, cwd=self.app_dir, env=self.configure_stack_env())
        return proc.returncode

    def _clean_previous_metrics(self) -> None:
        """Remove stale metrics from earlier runs in this results dir.

        VLMMetricsLogger itself deletes files matching
        ``vlm_application_metrics*`` when it first writes, but only inside the
        directory it was configured with -- doing it here too keeps a
        --skip_stack re-run from silently mixing in an older run's turns.
        """
        for pattern in ("vlm_application_metrics_*.txt", "vlm_performance_metrics_*.txt"):
            for path in glob.glob(os.path.join(self.results_dir, pattern)):
                try:
                    os.remove(path)
                except OSError:
                    pass

    # ------------------------------------------------------------------
    # Collection
    # ------------------------------------------------------------------
    def collect_vlm_logger_metrics(self, since_ms: Optional[int] = None) -> Dict:
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

        log_files = glob.glob(os.path.join(self.results_dir, "vlm_application_metrics_*.txt"))
        if not log_files:
            print(f"No vlm_metrics_logger files found in {self.results_dir}")
            return metrics

        start_times: Dict[str, int] = {}
        end_times: Dict[str, int] = {}

        for log_file in log_files:
            try:
                with open(log_file, "r") as f:
                    for line in f:
                        id_match = re.search(r"id=(\S+)", line)
                        event_match = re.search(r"event=(\w+)", line)
                        ts_match = re.search(r"timestamp_ms=(\d+)", line)
                        if not (id_match and event_match and ts_match):
                            continue
                        unique_id = id_match.group(1)
                        event = event_match.group(1)
                        timestamp = int(ts_match.group(1))
                        if since_ms is not None and timestamp < since_ms:
                            continue
                        if event == "start":
                            start_times[unique_id] = timestamp
                        elif event == "end":
                            end_times[unique_id] = timestamp
            except (IOError, OSError) as e:
                print(f"Warning: could not read {log_file}: {e}")

        latencies: List[int] = []
        for unique_id, start in start_times.items():
            if unique_id in end_times:
                latency_ms = end_times[unique_id] - start
                latencies.append(latency_ms)
                metrics["transactions"].append({"id": unique_id, "latency_ms": latency_ms})

        metrics["total_transactions"] = len(latencies)
        if latencies:
            ordered = sorted(latencies)
            metrics["avg_latency_ms"] = round(sum(ordered) / len(ordered), 1)
            metrics["median_latency_ms"] = round(statistics.median(ordered), 1)
            p95_idx = int(len(ordered) * 0.95)
            metrics["p95_latency_ms"] = ordered[p95_idx] if p95_idx < len(ordered) else ordered[-1]
        return metrics

    def collect_benchmark_report(self, label: str) -> Dict:
        """Read the scripted benchmark's own JSON report (the authoritative one)."""
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
    def export_results(self, results: Dict, prefix: str = "smart_kiosk_v2v") -> None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        json_path = os.path.join(self.results_dir, f"{prefix}_results_{timestamp}.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results exported to: {json_path}")

        csv_path = os.path.join(self.results_dir, f"{prefix}_summary_{timestamp}.csv")
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
    ) -> Dict:
        print("\n" + "=" * 70)
        print("Smart Kiosk Voice-to-Voice Benchmark")
        print(f"Script: {script or '(default: Sample_data/conversation.jsonl)'}")
        print(f"Runs: {runs} (plus {warmup_runs} discarded warmup)")
        print(f"Results: {self.results_dir}")
        print("=" * 70)

        self._clean_previous_metrics()
        self.start_stack()

        try:
            self.wait_for_health(health_timeout)
            if init_duration > 0:
                print(f"Settling for {init_duration}s before warmup...")
                time.sleep(init_duration)

            if warmup_runs > 0:
                print(f"\n--- Warmup ({warmup_runs} run(s), discarded) ---")
                self.run_benchmark_script(
                    script=script,
                    runs=warmup_runs,
                    label=f"{label}-warmup",
                    emit_vlm_metrics=False,
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
            rc = self.run_benchmark_script(script=script, runs=runs, label=label)

            vlm = self.collect_vlm_logger_metrics(since_ms=measured_run_started_ms)
            report = self.collect_benchmark_report(label)

            results = {
                "benchmark": "smart_kiosk_v2v",
                "script": script,
                "runs": runs,
                "warmup_runs": warmup_runs,
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
        print(f"p95 v2v        : {vlm.get('p95_latency_ms')} ms")
        print(f"Cross-check    : {check.get('status')} (delta={check.get('delta_ms')} ms)")
        if check.get("status") == "MISMATCH":
            print(
                "  WARNING: emitted metrics disagree with the benchmark's own report.\n"
                "  Treat the report JSON as authoritative and investigate the emit path."
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
        help="Path to the smart-kiosk-assistant directory (default: ../../smart-kiosk-assistant)",
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
        help="Measured repetitions of the script (default: 12)",
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
        help="Max seconds to wait for services to become healthy (default: 600)",
    )
    parser.add_argument("--label", default="v2v-perf-tools", help="Result filename label")
    parser.add_argument(
        "--skip_stack",
        action="store_true",
        help="Do not start/stop the stack -- benchmark whatever is already running",
    )
    parser.add_argument(
        "--keep_stack",
        action="store_true",
        help="Start the stack but leave it running afterwards",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    app_dir = os.path.abspath(args.app_dir)
    results_dir = args.results_dir or os.path.join(app_dir, "results")

    try:
        benchmark = SmartKioskV2VBenchmark(
            app_dir=app_dir,
            results_dir=results_dir,
            skip_stack=args.skip_stack,
            keep_stack=args.keep_stack,
        )
        results = benchmark.run(
            script=args.script,
            runs=args.runs,
            warmup_runs=args.warmup_runs,
            init_duration=args.init_duration,
            health_timeout=args.health_timeout,
            label=args.label,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"\nBenchmark failed: {exc}", file=sys.stderr)
        return 1

    if results.get("vlm_metrics", {}).get("total_transactions", 0) == 0:
        print("\nNo transactions were recorded -- treating as failure.", file=sys.stderr)
        return 1
    return results.get("benchmark_exit_code", 0)


if __name__ == "__main__":
    sys.exit(main())
