"""
Smart Kiosk Voice-to-Voice (V2V) Latency Benchmark

Copyright (C) 2026 Intel Corporation.

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

* **A warmup run is discarded.** The first turn after the stack comes up pays
  one-off costs the steady state does not: the TTS opener cache is empty and
  the model paths have not been exercised. Including it would bias the
  median of a short run.

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
from datetime import datetime
from typing import Deque, Dict, List, Optional, Tuple


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


# Precompiled once, and shared by every parser below so the two cannot
# drift apart. The negative lookbehind matters: vlm_metrics_logger's
# log_custom_event() appends arbitrary **kwargs to the line, so a future
# `session_id=` or `request_id=` field would otherwise be matched by a bare
# `id=` pattern and silently regroup turns under the wrong identifier.
_ID_RE = re.compile(r"(?<!\w)id=(\S+)")
_EVENT_RE = re.compile(r"(?<!\w)event=(\w+)")
_TIMESTAMP_RE = re.compile(r"(?<!\w)timestamp_ms=(\d+)")


def parse_event_line(line: str) -> Optional[Tuple[str, str, int]]:
    """Parse one vlm_metrics_logger line into (id, event, timestamp_ms).

    Returns None for anything that is not a complete start/end event, so
    callers can keep unrecognised lines rather than discard them.
    """
    id_match = _ID_RE.search(line)
    event_match = _EVENT_RE.search(line)
    ts_match = _TIMESTAMP_RE.search(line)
    if not (id_match and event_match and ts_match):
        return None
    event = event_match.group(1)
    if event not in ("start", "end"):
        return None
    return id_match.group(1), event, int(ts_match.group(1))


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
        # Validated here rather than at the point of use: by then the stack
        # has been started and waited on, so a mistyped --app_dir would cost
        # a bring-up plus the health timeout before failing.
        self.benchmark_script = os.path.join(
            self.app_dir,
            "tests",
            "benchmarks",
            "v2v_scripted_conversation_benchmark.py",
        )
        if not os.path.isfile(self.benchmark_script):
            raise ValueError(
                f"Benchmark script not found: {self.benchmark_script} -- "
                f"is {self.app_dir} the smart-kiosk-assistant dir?"
            )

        os.makedirs(self.results_dir, exist_ok=True)

    @property
    def app_results_dir(self) -> str:
        """Where kiosk-core itself writes ``vlm_application_metrics_*.txt``.

        The service writes from inside its container to a bind mount the
        application's compose file defines as
        ``${KIOSK_RESULTS_HOST_DIR:-./results}``, relative to ``app_dir`` --
        it does not know about this script's ``--results_dir``. When the two
        differ and this script starts the stack, it points that bind mount at
        ``--results_dir`` (see ``configure_stack_env``) so the logs land where
        the collectors look.
        """
        return os.path.join(self.app_dir, "results")

    # ------------------------------------------------------------------
    # Stack lifecycle
    # ------------------------------------------------------------------
    def configure_stack_env(self) -> Dict[str, str]:
        """Build the environment the stack is started with.

        The flags are passed through the process environment (not just as make
        arguments) because docker compose gives shell environment variables
        precedence over the application's ``.env`` file -- so this forces the
        intended configuration regardless of what ``.env`` currently says.
        """
        env = os.environ.copy()
        # Frees ~10-12 CPU cores otherwise burned by rtsp-streamer +
        # queue-service when disabled.
        #
        # The application Makefile owns this relationship: it defaults
        # QUEUE from KIOSK_CORE_QUEUE_SERVICE_ENABLED (`QUEUE ?=`), uses
        # QUEUE to decide whether compose gets `--profile queue` at all,
        # and then re-exports KIOSK_CORE_QUEUE_SERVICE_ENABLED derived from
        # QUEUE. So a `QUEUE=` make argument always wins, and setting this
        # env var to a value that disagrees with it would be ignored at
        # best and misleading in the results at worst. Mirror the make
        # argument instead, so the two can never contradict each other.
        env["KIOSK_CORE_QUEUE_SERVICE_ENABLED"] = (
            "true" if self.queue_enabled else "false"
        )
        # Speaker diarization (pyannote) adds analyzer latency and CPU load
        # that is irrelevant to a single-speaker scripted benchmark. compose
        # maps this onto AUDIO_ANALYZER__MODELS__ASR__DIARIZATION.
        env["KIOSK_CORE_DIARIZATION_ENABLED"] = "false"
        env["KIOSK_CORE_DIARIZATION_INTERMEDIATE_ENABLED"] = "false"
        # kiosk-core emits vlm_application_metrics_*.txt from inside its
        # container into a bind mount the application's compose file defines
        # as ${KIOSK_RESULTS_HOST_DIR:-./results}. Without this, a custom
        # --results_dir would collect nothing: the service would keep writing
        # to <app_dir>/results while this script looked somewhere else, and
        # the run would end with a misleading "No transactions".
        env["KIOSK_RESULTS_HOST_DIR"] = self.results_dir
        return env

    def warn_if_results_dir_unreachable(self) -> None:
        """Warn when the stack cannot be pointed at ``--results_dir``.

        Under ``--skip_stack`` the containers are already running with
        whatever bind mount they were started with, so this script cannot
        redirect where kiosk-core writes.
        """
        if not self.skip_stack:
            return
        if os.path.realpath(self.results_dir) == os.path.realpath(
            self.app_results_dir
        ):
            return
        print(
            f"\nWARNING: --skip_stack with --results_dir "
            f"{self.results_dir}, but the already-running kiosk-core writes "
            f"its metrics to the bind mount it was started with (normally "
            f"{self.app_results_dir}). Metrics collection will likely find "
            f"nothing. Restart the stack without --skip_stack, or point "
            f"--results_dir at the directory the stack already uses."
        )

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
        queue_state = "ON" if self.queue_enabled else "OFF"
        print("\n" + "=" * 70)
        print(
            f"Starting Smart Kiosk stack "
            f"(queue-service/rtsp-streamer {queue_state}, diarization OFF)"
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
        rc = self._run_make("down")
        if rc != 0:
            # Not fatal -- the measurements are already collected -- but a
            # failed teardown leaves containers holding the ports, and the
            # next run's health check would then pass against the old stack.
            print(
                f"Warning: 'make down' exited {rc}. Containers may still be "
                f"running; check `docker compose ps` in {self.app_dir} "
                f"before the next run, or its health checks will pass "
                f"against the old stack.",
                file=sys.stderr,
            )

    @staticmethod
    def _endpoint_is_ready(name: str, body: bytes) -> bool:
        """Decide whether a 2xx response really means "ready".

        For OVMS a 200 from /v3/models is not enough: it answers while the
        LLM is still loading, with an empty or not-yet-AVAILABLE model
        list. Starting a run then would charge model-load time to the
        first measured turn, which is precisely what the warmup and this
        wait exist to avoid.

        Anything unrecognised is treated as ready, so an operator pointing
        --ovms_url at a plain health endpoint is not locked out.
        """
        if name != "ovms-llm":
            return True
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            return True

        entries = payload.get("data") if isinstance(payload, dict) else None
        if entries is None and isinstance(payload, list):
            entries = payload
        if not isinstance(entries, list):
            return True
        if not entries:
            return False

        def _available(entry) -> bool:
            if not isinstance(entry, dict):
                return True
            state = entry.get("state") or entry.get("status")
            if state is None:
                return True
            return str(state).upper() in ("AVAILABLE", "READY", "LOADED")

        return any(_available(entry) for entry in entries)

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
                        if not 200 <= resp.status < 300:
                            continue
                        body = resp.read()
                    if not self._endpoint_is_ready(name, body):
                        continue
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
        # Existence was validated in __init__, before the stack was
        # started, so a bad --app_dir fails immediately.
        benchmark_path = self.benchmark_script

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

    def _kiosk_core_is_running(self) -> bool:
        """Best-effort check for an already-running kiosk-core.

        Returns False on any error: this only guards an optional cleanup
        step, so an unreachable endpoint must never fail the run.
        """
        url = self.health_endpoints.get("kiosk-core")
        if not url:
            return False
        try:
            # Operator-supplied health URL, not untrusted input.
            with urllib.request.urlopen(url, timeout=2) as resp:  # nosec B310
                return 200 <= resp.status < 300
        except (urllib.error.URLError, OSError, ValueError):
            return False

    def _clean_previous_metrics(self) -> None:
        """Remove stale metrics from earlier runs in this results dir.

        VLMMetricsLogger deletes files matching ``vlm_application_metrics*``
        when it first writes, but only inside the directory it was configured
        with, so a results dir that has been reused can still hold an older
        run's files.

        Deliberately a no-op whenever kiosk-core may already be running --
        under ``--skip_stack``, or when its health endpoint answers before
        the stack is started (which happens after an earlier
        ``--keep_stack`` run). In those cases kiosk-core holds these files
        open through a RotatingFileHandler, so unlinking them does not give
        a clean slate: it leaves the handler writing to an unlinked inode,
        every subsequent event vanishes, and the run ends with "No
        transactions". The ``since_ms`` filter in
        ``collect_vlm_logger_metrics`` already excludes the older run's
        turns, which is what this cleanup was for.
        """
        if self.skip_stack:
            print(
                "--skip_stack set: leaving existing metrics files in place "
                "(kiosk-core holds them open); stale turns are excluded by "
                "timestamp instead."
            )
            return
        if self._kiosk_core_is_running():
            print(
                "kiosk-core is already running (likely a previous "
                "--keep_stack run): leaving existing metrics files in place "
                "so its open log handles stay valid; stale turns are "
                "excluded by timestamp instead."
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
                    lines = f.readlines()
                # One definition of "warmup", shared with the pruner, so
                # the summary and the pruned log files can never disagree
                # about which turns belong to the measured window. Filtering
                # each event independently would split a turn straddling
                # since_ms and leave an orphan end behind.
                warmup = (
                    self._warmup_line_indices(lines, since_ms)
                    if since_ms is not None
                    else set()
                )
                for i, line in enumerate(lines):
                    if i in warmup:
                        continue
                    parsed = parse_event_line(line)
                    if parsed is None:
                        continue
                    unique_id, event, timestamp = parsed
                    events[unique_id].append(
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
            # with the nearest-rank definition the p95 only drops below the
            # maximum at n >= 20. Derive the flag by comparison rather than
            # from a sample-count threshold, so it stays correct whatever
            # the percentile definition is -- and so ties (where the second
            # -slowest turn equals the slowest) are reported honestly.
            metrics["p95_sample_count"] = len(ordered)
            metrics["p95_is_max"] = metrics["p95_latency_ms"] == ordered[-1]
        return metrics

    def _clean_previous_reports(self, label: str) -> None:
        """Remove earlier runs' JSON reports from this results dir.

        The default label is fixed, so without this a measured run that
        crashes before writing its report leaves the previous run's
        ``{label}.json`` in place -- and the cross-check would silently
        validate this run's emitted metrics against last run's numbers.

        Unlike the vlm logs these are written and closed by a short-lived
        child process, not held open by kiosk-core, so removing them is
        safe even when the stack is running.
        """
        stale = [
            os.path.join(self.results_dir, f"{label}.json"),
            os.path.join(self.results_dir, f"{label}-warmup.json"),
            os.path.join(self.results_dir, "measured_window.json"),
        ]
        for path in stale:
            try:
                os.remove(path)
            except OSError:
                pass

    def collect_benchmark_report(
        self, label: str, not_before_ms: Optional[int] = None
    ) -> Dict:
        """Read the benchmark's own JSON report (the authoritative one).

        Args:
            label: Report basename, without the ``.json`` suffix.
            not_before_ms: When given, a report that finished before this
                epoch-ms instant is rejected as stale rather than returned.
                Belt-and-braces alongside ``_clean_previous_reports``: a
                stale report is worse than no report, because it makes the
                cross-check vouch for numbers it never saw.
        """
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

        if not_before_ms is not None:
            # Deliberately the file's mtime rather than its finished_at
            # field. Both are produced on this host, so they share a clock,
            # and mtime cannot be misread the way a timestamp string can:
            # an ISO value without an offset would be parsed as local time
            # and a fresh report could look hours old on any host not on
            # UTC, silently downgrading the cross-check to "unavailable".
            try:
                written_ms = int(os.path.getmtime(path) * 1000)
            except OSError:
                written_ms = None
            if written_ms is not None and written_ms < not_before_ms:
                print(
                    f"Warning: ignoring stale report {path} "
                    f"(last written before this run started). The measured "
                    f"run most likely failed before writing its report."
                )
                return {}

        turns = report.get("turns", []) or []
        # Turns that errored (for example a TTS synthesis failure) are
        # recorded in the report but never emit a start/end pair, so
        # counting them would fail the cross-check on a turn-count
        # mismatch and point at the metrics path instead of the real cause.
        successful = [
            t for t in turns if not (isinstance(t, dict) and t.get("error"))
        ]
        return {
            "label": report.get("label"),
            "started_at": report.get("started_at"),
            "finished_at": report.get("finished_at"),
            "turns_total": len(successful),
            "turns_recorded": len(turns),
            "turns_failed": len(turns) - len(successful),
            "summary": report.get("summary", {}),
        }

    @staticmethod
    def _queue_value_is_on(value: str) -> bool:
        """Apply the application Makefile's exact truthiness rule.

        That Makefile uses ``$(filter true TRUE,$(QUEUE))``, so only the
        literals ``true`` and ``TRUE`` enable the vision profile. Anything
        else -- including ``1``, ``yes`` and ``True`` -- leaves it off.
        Accepting a broader set here would record a run as queue-enabled
        while the services were never started.
        """
        return value.strip() in ("true", "TRUE")

    @property
    def queue_enabled(self) -> bool:
        """Whether the vision (queue) profile will be started.

        Resolved the same way the application Makefile resolves it: an
        explicit ``QUEUE=`` make argument wins, otherwise the inherited
        ``KIOSK_CORE_QUEUE_SERVICE_ENABLED`` acts as the default, otherwise
        the Makefile's own default of true.
        """
        for arg in reversed(self.make_args):
            if arg.startswith("QUEUE="):
                return self._queue_value_is_on(arg.split("=", 1)[1])
        inherited = os.environ.get("KIOSK_CORE_QUEUE_SERVICE_ENABLED", "")
        if inherited:
            return self._queue_value_is_on(inherited)
        return True

    def describe_stack(self) -> Dict[str, str]:
        """Report the stack configuration this run actually used.

        Derived from the effective settings rather than hardcoded: the
        defaults can be overridden with --make_arg, and a results file that
        misstates its own configuration is worse than one that omits it.
        """
        env = self.configure_stack_env()

        def _flag(name: str) -> str:
            return (
                "disabled"
                if env.get(name, "").lower() == "false"
                else "enabled"
            )

        # The Makefile derives both the compose profile and kiosk-core's
        # own flag from the same QUEUE value, so queue-service and
        # rtsp-streamer are always either both present or both absent.
        # Reporting them from one resolved value keeps the results file
        # from ever claiming a combination that cannot occur.
        queue_state = "enabled" if self.queue_enabled else "disabled"

        return {
            "queue_service": queue_state,
            "rtsp_streamer": queue_state,
            "diarization": _flag("KIOSK_CORE_DIARIZATION_ENABLED"),
            "make_args": " ".join(self.make_args) or "(none)",
            "stack_managed_by_script": "no" if self.skip_stack else "yes",
        }

    @staticmethod
    def cross_check(vlm: Dict, report: Dict) -> Dict:
        """Verify the emitted metrics agree with the authoritative report.

        The vlm_metrics_logger pair is synthesised from the same
        voice_to_voice_ms the benchmark report carries, so a mismatch beyond
        rounding means the emit path is broken (wrong field, wrong clock, or
        turns silently dropped) -- exactly the class of bug that would
        otherwise go unnoticed and quietly misreport the headline number.

        Both the median and the turn count are compared. A median alone can
        hide a dropped or duplicated turn, whereas a count mismatch is the
        clearest possible signal that the emit path lost data.
        """
        summary = report.get("summary", {}) or {}
        reported = (
            summary.get("voice_to_voice_ms", {}).get("median")
            if isinstance(summary.get("voice_to_voice_ms"), dict)
            else summary.get("voice_to_voice_ms")
        )
        emitted = vlm.get("median_latency_ms")
        result = {"reported_median_ms": reported, "emitted_median_ms": emitted}

        reported_turns = report.get("turns_total")
        emitted_turns = vlm.get("total_transactions")
        result["reported_turns"] = reported_turns
        result["emitted_turns"] = emitted_turns
        counts_disagree = (
            reported_turns is not None
            and emitted_turns is not None
            and int(reported_turns) != int(emitted_turns)
        )
        result["turn_count_status"] = (
            "MISMATCH" if counts_disagree else "ok"
        )

        if reported is None or emitted is None:
            result["status"] = (
                "MISMATCH" if counts_disagree else "unavailable"
            )
            return result

        delta = abs(float(reported) - float(emitted))
        result["delta_ms"] = round(delta, 1)
        # 1ms tolerance: the emitted pair is built from int-rounded epoch
        # milliseconds, so a sub-millisecond difference is expected.
        medians_disagree = delta > 1.0
        result["status"] = (
            "MISMATCH" if (medians_disagree or counts_disagree) else "ok"
        )
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

        self.warn_if_results_dir_unreachable()
        self._clean_previous_metrics()
        self._clean_previous_reports(label)
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
            report = self.collect_benchmark_report(
                label, not_before_ms=measured_run_started_ms
            )

            results = {
                "benchmark": "smart_kiosk_v2v",
                "script": script,
                "runs": runs,
                "warmup_runs": warmup_runs,
                "explicit_end_mark": explicit_end_mark,
                "stack": self.describe_stack(),
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
            # Pruning rewrites the log files, so it is only safe once
            # kiosk-core is actually gone. stop_stack() is a no-op under
            # --skip_stack/--keep_stack, and os.replace() would then swap
            # the file out from under kiosk-core's still-open
            # RotatingFileHandler: the service would keep writing into
            # *.txt.with-warmup, which the consolidator's glob ignores.
            if measured_run_started_ms is None:
                pass
            elif self.skip_stack or self.keep_stack:
                print(
                    "\nSkipping warmup pruning: the stack is still running "
                    "(--skip_stack/--keep_stack) and rewriting the log "
                    "files would detach kiosk-core's open handles. The "
                    "summary above already excludes warmup turns; the log "
                    "files still contain them."
                )
            else:
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

    @staticmethod
    def _warmup_line_indices(lines: List[str], since_ms: int) -> set:
        """Indices of log lines belonging to warmup turns.

        A turn is warmup if its ``start`` is stamped before ``since_ms``. Its
        matching ``end`` is dropped with it even when that end is stamped
        after ``since_ms``, so no orphan ends are left behind.

        Lines that are not parseable start/end events are never dropped:
        unknown content is kept rather than silently discarded.

        Args:
            lines: The log file's lines, in file order.
            since_ms: Epoch-ms instant the measured run started.

        Returns:
            Set of indices into ``lines`` to remove.
        """
        # Per id, indices of starts still awaiting an end, oldest first.
        open_starts: Dict[str, Deque[tuple]] = defaultdict(deque)
        drop: set = set()

        for i, line in enumerate(lines):
            parsed = parse_event_line(line)
            if parsed is None:
                continue
            stream_id, event, timestamp = parsed

            if event == "start":
                open_starts[stream_id].append((i, timestamp))
                continue

            if not open_starts[stream_id]:
                # Orphan end: its start is not in this file. Drop it only if
                # it is itself older than the measured window.
                if timestamp < since_ms:
                    drop.add(i)
                continue

            start_i, start_ts = open_starts[stream_id].popleft()
            if start_ts < since_ms:
                drop.add(start_i)
                drop.add(i)

        # Starts never matched by an end: drop the warmup-era ones so the
        # consolidator cannot pair them with a later measured end.
        for pending in open_starts.values():
            for start_i, start_ts in pending:
                if start_ts < since_ms:
                    drop.add(start_i)
        return drop

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

        Whole start/end pairs are dropped together. Filtering purely on
        timestamp would split a turn that straddles ``since_ms`` -- its start
        is older, its end is not -- leaving an orphan ``end`` in the file.
        This parser ignores unmatched ends, but the downstream consolidator
        may pair such an end with the next turn's start and invent a wildly
        wrong latency.

        Args:
            since_ms: Epoch-ms instant the measured run started. Turns whose
                ``start`` is stamped before this belong to the warmup.
        """
        for log_file in glob.glob(
            os.path.join(self.results_dir, "vlm_application_metrics_*.txt")
        ):
            try:
                with open(log_file, "r") as f:
                    lines = f.readlines()

                drop_idx = self._warmup_line_indices(lines, since_ms)
                kept = [
                    line
                    for i, line in enumerate(lines)
                    if i not in drop_idx
                ]
                dropped = len(drop_idx)

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
        p95_note = (
            f" (= max of {vlm.get('p95_sample_count')} turns)"
            if vlm.get("p95_is_max")
            else ""
        )
        print(f"p95 v2v        : {vlm.get('p95_latency_ms')} ms{p95_note}")
        print(
            f"Turns reported : {check.get('reported_turns')} "
            f"(emitted {check.get('emitted_turns')})"
        )
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


def positive_int(value: str) -> int:
    """argparse type: an integer >= 1."""
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError(
            f"must be 1 or greater, got {number}"
        )
    return number


def non_negative_int(value: str) -> int:
    """argparse type: an integer >= 0."""
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError(
            f"must be 0 or greater, got {number}"
        )
    return number


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
        type=positive_int,
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
        type=non_negative_int,
        default=SmartKioskV2VBenchmark.DEFAULT_WARMUP_RUNS,
        help="Discarded warmup repetitions (default: 1)",
    )
    parser.add_argument(
        "--init_duration",
        type=non_negative_int,
        default=SmartKioskV2VBenchmark.DEFAULT_INIT_DURATION,
        help="Settle time after health checks pass, in seconds (default: 30)",
    )
    parser.add_argument(
        "--health_timeout",
        type=positive_int,
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
    parser.add_argument(
        "--allow_cross_check_mismatch",
        action="store_true",
        help=(
            "Exit 0 even when the cross-check does not pass. By default "
            "both MISMATCH (the two measurements of the same turns "
            "disagree) and unavailable (the application's report was "
            "missing or stale, so nothing verified the number) fail the "
            "run."
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

    # A MISMATCH means the two independent measurements of the same turns
    # disagree, so the headline number is untrustworthy. "unavailable" is
    # no better: it means the application's report was missing or stale, so
    # nothing checked the number at all. Exiting 0 in either case would let
    # CI publish an unverified result, which is exactly what the
    # cross-check exists to prevent.
    status = results.get("cross_check", {}).get("status")
    reasons = {
        "MISMATCH": (
            "the emitted metrics disagree with the application's own report"
        ),
        "unavailable": (
            "the application's report was missing or stale, so the emitted "
            "metrics were never verified against it"
        ),
    }
    if status in reasons:
        if args.allow_cross_check_mismatch:
            print(
                f"\nCross-check {status} ignored "
                f"(--allow_cross_check_mismatch): {reasons[status]}.",
                file=sys.stderr,
            )
        else:
            print(
                f"\nCross-check {status}: {reasons[status]} -- treating as "
                f"failure. Pass --allow_cross_check_mismatch to override.",
                file=sys.stderr,
            )
            return 1
    return results.get("benchmark_exit_code", 0)


if __name__ == "__main__":
    sys.exit(main())
