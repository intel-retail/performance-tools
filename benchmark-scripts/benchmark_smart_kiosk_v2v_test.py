'''
* Copyright (C) 2026 Intel Corporation.
*
* SPDX-License-Identifier: Apache-2.0
'''

import json
import os
import shutil
import subprocess  # nosec B404
import tempfile
import unittest
import unittest.mock as mock
from datetime import datetime

import benchmark_smart_kiosk_v2v as v2v


def _event_line(stream_id, event, timestamp_ms):
    return (
        f"id={stream_id} event={event} timestamp_ms={timestamp_ms}\n"
    )


class PercentileTest(unittest.TestCase):
    """Nearest-rank percentile, including the off-by-one it replaced."""

    def test_rejects_empty(self):
        with self.assertRaises(ValueError):
            v2v.percentile([], 0.95)

    def test_single_sample(self):
        self.assertEqual(v2v.percentile([7], 0.95), 7)

    def test_nearest_rank_at_exact_multiple(self):
        # n=20 is where int(n * 0.95) == 19 indexes the maximum, while the
        # nearest rank is the 19th of 20 samples -> index 18.
        ordered = list(range(1, 21))
        self.assertEqual(v2v.percentile(ordered, 0.95), 19)
        self.assertNotEqual(v2v.percentile(ordered, 0.95), ordered[-1])

    def test_small_sample_is_max(self):
        ordered = [10, 20, 30, 40, 50, 60, 70, 80]
        self.assertEqual(v2v.percentile(ordered, 0.95), 80)

    def test_never_indexes_past_the_end(self):
        for n in range(1, 60):
            ordered = list(range(n))
            self.assertEqual(v2v.percentile(ordered, 1.0), ordered[-1])


class BenchmarkTempDirTest(unittest.TestCase):
    """Shared fixture: a real app_dir and results_dir on disk."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.app_dir = os.path.join(self.tmp, "app")
        self.results_dir = os.path.join(self.tmp, "results")
        os.makedirs(self.app_dir)
        os.makedirs(self.results_dir)
        # The constructor validates both of these up front, before the stack
        # is started, so the fixture has to provide them.
        with open(os.path.join(self.app_dir, "Makefile"), "w") as f:
            f.write("up:\n\t@true\n")
        self.script_dir = os.path.join(self.app_dir, "tests", "benchmarks")
        os.makedirs(self.script_dir)
        self.app_script = os.path.join(
            self.script_dir, "v2v_scripted_conversation_benchmark.py"
        )
        with open(self.app_script, "w") as f:
            f.write("# stub\n")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def make(self, **kwargs):
        return v2v.SmartKioskV2VBenchmark(
            app_dir=self.app_dir,
            results_dir=self.results_dir,
            **kwargs,
        )

    def write_log(self, name, lines):
        path = os.path.join(self.results_dir, name)
        with open(path, "w") as f:
            f.writelines(lines)
        return path


class CollectVlmLoggerMetricsTest(BenchmarkTempDirTest):

    def test_no_files_returns_empty_metrics(self):
        metrics = self.make().collect_vlm_logger_metrics()
        self.assertEqual(metrics["total_transactions"], 0)
        self.assertEqual(metrics["transactions"], [])

    def test_pairs_multiple_turns_under_one_stream_id(self):
        # kiosk-core stamps every turn with the SAME stream id, so a naive
        # {id: timestamp} map would collapse this to one transaction.
        self.write_log(
            "vlm_application_metrics_0.txt",
            [
                _event_line("stream0", "start", 1000),
                _event_line("stream0", "end", 1500),
                _event_line("stream0", "start", 2000),
                _event_line("stream0", "end", 2800),
            ],
        )
        metrics = self.make().collect_vlm_logger_metrics()
        self.assertEqual(metrics["total_transactions"], 2)
        self.assertEqual(
            sorted(t["latency_ms"] for t in metrics["transactions"]),
            [500, 800],
        )

    def test_fifo_pairing_for_overlapping_turns(self):
        # Two starts before either end. FIFO pairs end@1500 with start@1000
        # and end@1800 with start@1200 -> 500 and 600.
        self.write_log(
            "vlm_application_metrics_0.txt",
            [
                _event_line("stream0", "start", 1000),
                _event_line("stream0", "start", 1200),
                _event_line("stream0", "end", 1500),
                _event_line("stream0", "end", 1800),
            ],
        )
        metrics = self.make().collect_vlm_logger_metrics()
        self.assertEqual(
            sorted(t["latency_ms"] for t in metrics["transactions"]),
            [500, 600],
        )

    def test_since_ms_excludes_warmup_turns(self):
        self.write_log(
            "vlm_application_metrics_0.txt",
            [
                _event_line("stream0", "start", 1000),
                _event_line("stream0", "end", 1900),
                _event_line("stream0", "start", 5000),
                _event_line("stream0", "end", 5400),
            ],
        )
        metrics = self.make().collect_vlm_logger_metrics(since_ms=4000)
        self.assertEqual(metrics["total_transactions"], 1)
        self.assertEqual(metrics["transactions"][0]["latency_ms"], 400)

    def test_ignores_unparseable_and_unknown_events(self):
        self.write_log(
            "vlm_application_metrics_0.txt",
            [
                "this line has no fields at all\n",
                _event_line("stream0", "heartbeat", 900),
                _event_line("stream0", "start", 1000),
                _event_line("stream0", "end", 1100),
            ],
        )
        metrics = self.make().collect_vlm_logger_metrics()
        self.assertEqual(metrics["total_transactions"], 1)

    def test_unmatched_end_is_not_counted(self):
        self.write_log(
            "vlm_application_metrics_0.txt",
            [_event_line("stream0", "end", 1100)],
        )
        self.assertEqual(
            self.make().collect_vlm_logger_metrics()["total_transactions"], 0
        )

    def test_reports_p95_sample_count(self):
        lines = []
        for i in range(8):
            base = 10000 + i * 1000
            lines.append(_event_line("stream0", "start", base))
            lines.append(_event_line("stream0", "end", base + 100 + i))
        self.write_log("vlm_application_metrics_0.txt", lines)
        metrics = self.make().collect_vlm_logger_metrics()
        self.assertEqual(metrics["p95_sample_count"], 8)
        self.assertTrue(metrics["p95_is_max"])

    def _write_n_turns(self, n):
        lines = []
        for i in range(n):
            base = 10000 + i * 1000
            lines.append(_event_line("stream0", "start", base))
            # Strictly increasing latencies, so there are no ties.
            lines.append(_event_line("stream0", "end", base + 100 + i))
        self.write_log("vlm_application_metrics_0.txt", lines)

    def test_p95_is_max_true_at_19_samples(self):
        # ceil(0.95 * 19) = 19 -> the 19th of 19 samples is the maximum.
        self._write_n_turns(19)
        metrics = self.make().collect_vlm_logger_metrics()
        self.assertTrue(metrics["p95_is_max"])
        self.assertEqual(
            metrics["p95_latency_ms"],
            max(t["latency_ms"] for t in metrics["transactions"]),
        )

    def test_p95_is_max_false_at_20_samples(self):
        # ceil(0.95 * 20) = 19 -> the 19th of 20 samples is NOT the maximum.
        self._write_n_turns(20)
        metrics = self.make().collect_vlm_logger_metrics()
        self.assertEqual(metrics["p95_sample_count"], 20)
        self.assertFalse(metrics["p95_is_max"])
        self.assertLess(
            metrics["p95_latency_ms"],
            max(t["latency_ms"] for t in metrics["transactions"]),
        )


class CleanPreviousMetricsTest(BenchmarkTempDirTest):

    def test_removes_stale_files_when_managing_the_stack(self):
        os.makedirs(self.results_dir, exist_ok=True)
        path = self.write_log("vlm_application_metrics_0.txt", ["stale\n"])
        # _kiosk_core_is_running() is patched rather than left to make a
        # real request to localhost:8012: unpatched, this test fails on a
        # developer box that happens to have the kiosk running, and costs
        # up to the probe timeout in CI.
        with mock.patch.object(
            v2v.SmartKioskV2VBenchmark,
            "_kiosk_core_is_running",
            return_value=False,
        ):
            self.make()._clean_previous_metrics()
        self.assertFalse(os.path.exists(path))

    def test_is_a_no_op_when_kiosk_core_already_answers(self):
        # A stack left up by an earlier --keep_stack run still holds these
        # files open, so unlinking them would send every later write to an
        # unlinked inode and the run would end with "No transactions".
        os.makedirs(self.results_dir, exist_ok=True)
        path = self.write_log("vlm_application_metrics_0.txt", ["live\n"])
        with mock.patch.object(
            v2v.SmartKioskV2VBenchmark,
            "_kiosk_core_is_running",
            return_value=True,
        ):
            self.make()._clean_previous_metrics()
        self.assertTrue(os.path.exists(path))

    def test_skip_stack_short_circuits_before_probing(self):
        os.makedirs(self.results_dir, exist_ok=True)
        self.write_log("vlm_application_metrics_0.txt", ["live\n"])
        with mock.patch.object(
            v2v.SmartKioskV2VBenchmark, "_kiosk_core_is_running"
        ) as probe:
            self.make(skip_stack=True)._clean_previous_metrics()
        probe.assert_not_called()

    def test_is_a_no_op_under_skip_stack(self):
        # kiosk-core already holds the file open; unlinking it would send
        # every later write to an unlinked inode and lose the whole run.
        os.makedirs(self.results_dir, exist_ok=True)
        path = self.write_log("vlm_application_metrics_0.txt", ["live\n"])
        self.make(skip_stack=True)._clean_previous_metrics()
        self.assertTrue(os.path.exists(path))

    def test_removes_stale_reports(self):
        for name in (
            "v2v-perf-tools.json",
            "v2v-perf-tools-warmup.json",
            "measured_window.json",
        ):
            with open(os.path.join(self.results_dir, name), "w") as f:
                f.write("{}")
        self.make()._clean_previous_reports("v2v-perf-tools")
        for name in (
            "v2v-perf-tools.json",
            "v2v-perf-tools-warmup.json",
            "measured_window.json",
        ):
            self.assertFalse(
                os.path.exists(os.path.join(self.results_dir, name)), name
            )

    def test_cleaning_reports_leaves_other_labels_alone(self):
        keep = os.path.join(self.results_dir, "some-other-label.json")
        with open(keep, "w") as f:
            f.write("{}")
        self.make()._clean_previous_reports("v2v-perf-tools")
        self.assertTrue(os.path.exists(keep))


class PruneWarmupFromVlmLogsTest(BenchmarkTempDirTest):

    def test_drops_warmup_events_and_keeps_original(self):
        path = self.write_log(
            "vlm_application_metrics_0.txt",
            [
                _event_line("stream0", "start", 1000),
                _event_line("stream0", "end", 1500),
                _event_line("stream0", "start", 9000),
                _event_line("stream0", "end", 9400),
            ],
        )
        self.make().prune_warmup_from_vlm_logs(since_ms=5000)

        with open(path) as f:
            kept = f.readlines()
        self.assertEqual(len(kept), 2)
        self.assertTrue(all("timestamp_ms=9" in line for line in kept))

        with open(path + ".with-warmup") as f:
            self.assertEqual(len(f.readlines()), 4)

    def test_backup_is_not_picked_up_by_the_consolidator_glob(self):
        self.write_log(
            "vlm_application_metrics_0.txt",
            [
                _event_line("stream0", "start", 1000),
                _event_line("stream0", "end", 1500),
                _event_line("stream0", "start", 9000),
                _event_line("stream0", "end", 9400),
            ],
        )
        self.make().prune_warmup_from_vlm_logs(since_ms=5000)
        # The consolidator globs *.txt; the backup must not match.
        txt_files = [
            n for n in os.listdir(self.results_dir) if n.endswith(".txt")
        ]
        self.assertEqual(txt_files, ["vlm_application_metrics_0.txt"])

    def test_keeps_lines_without_a_timestamp(self):
        path = self.write_log(
            "vlm_application_metrics_0.txt",
            [
                "# header line\n",
                _event_line("stream0", "start", 1000),
                _event_line("stream0", "start", 9000),
                _event_line("stream0", "end", 9400),
            ],
        )
        self.make().prune_warmup_from_vlm_logs(since_ms=5000)
        with open(path) as f:
            self.assertIn("# header line\n", f.readlines())

    def test_nothing_to_drop_leaves_file_untouched(self):
        path = self.write_log(
            "vlm_application_metrics_0.txt",
            [
                _event_line("stream0", "start", 9000),
                _event_line("stream0", "end", 9400),
            ],
        )
        self.make().prune_warmup_from_vlm_logs(since_ms=5000)
        self.assertFalse(os.path.exists(path + ".with-warmup"))
        with open(path) as f:
            self.assertEqual(len(f.readlines()), 2)

    def test_original_survives_a_failed_write(self):
        # The main log must never disappear: the consolidator globs *.txt,
        # so losing it silently drops the run's metrics.
        path = self.write_log(
            "vlm_application_metrics_0.txt",
            [
                _event_line("stream0", "start", 1000),
                _event_line("stream0", "end", 1500),
                _event_line("stream0", "start", 9000),
                _event_line("stream0", "end", 9400),
            ],
        )
        bench = self.make()
        real_open = open

        def failing_open(file, mode="r", *args, **kwargs):
            if str(file).endswith(".pruned.tmp"):
                raise IOError("disk full")
            return real_open(file, mode, *args, **kwargs)

        # Patch only the module under test, and only for this call, rather
        # than reassigning builtins.open for the whole process.
        with mock.patch.object(
            v2v, "open", failing_open, create=True
        ):
            bench.prune_warmup_from_vlm_logs(since_ms=5000)

        self.assertTrue(os.path.exists(path))
        with open(path) as f:
            self.assertEqual(len(f.readlines()), 4)
        self.assertFalse(os.path.exists(path + ".pruned.tmp"))


class StraddlingWarmupTurnTest(BenchmarkTempDirTest):
    """A turn whose start is warmup but whose end lands after since_ms."""

    def setUp(self):
        super().setUp()
        self.path = self.write_log(
            "vlm_application_metrics_0.txt",
            [
                # Warmup turn that straddles the measured-window boundary.
                _event_line("stream0", "start", 4900),
                _event_line("stream0", "end", 5200),
                # A genuine measured turn.
                _event_line("stream0", "start", 9000),
                _event_line("stream0", "end", 9400),
            ],
        )

    def test_whole_pair_is_dropped_leaving_no_orphan_end(self):
        self.make().prune_warmup_from_vlm_logs(since_ms=5000)
        with open(self.path) as f:
            kept = f.readlines()
        # Both warmup lines must go, including the end stamped after
        # since_ms, or the consolidator pairs it with the next start.
        self.assertEqual(len(kept), 2)
        self.assertNotIn("timestamp_ms=5200", "".join(kept))
        self.assertEqual(
            sum(1 for line in kept if "event=start" in line), 1
        )
        self.assertEqual(sum(1 for line in kept if "event=end" in line), 1)

    def test_pruned_file_reparses_to_only_the_measured_turn(self):
        self.make().prune_warmup_from_vlm_logs(since_ms=5000)
        metrics = self.make().collect_vlm_logger_metrics()
        self.assertEqual(metrics["total_transactions"], 1)
        self.assertEqual(metrics["transactions"][0]["latency_ms"], 400)

    def test_unmatched_warmup_start_is_dropped(self):
        # A warmup turn whose end never made it to the log. Leaving the
        # start behind would let the consolidator pair it with a later
        # measured end and report an absurd latency.
        self.write_log(
            "vlm_application_metrics_1.txt",
            [
                _event_line("stream1", "start", 1000),
                _event_line("stream2", "start", 9000),
                _event_line("stream2", "end", 9400),
            ],
        )
        self.make().prune_warmup_from_vlm_logs(since_ms=5000)
        path = os.path.join(self.results_dir, "vlm_application_metrics_1.txt")
        with open(path) as f:
            kept = f.readlines()
        self.assertEqual(len(kept), 2)
        self.assertNotIn("timestamp_ms=1000", "".join(kept))

    def test_unparseable_lines_are_never_dropped(self):
        self.write_log(
            "vlm_application_metrics_2.txt",
            [
                "some header line without events\n",
                _event_line("stream0", "start", 1000),
                _event_line("stream0", "end", 1400),
            ],
        )
        self.make().prune_warmup_from_vlm_logs(since_ms=5000)
        path = os.path.join(self.results_dir, "vlm_application_metrics_2.txt")
        with open(path) as f:
            kept = f.readlines()
        self.assertEqual(kept, ["some header line without events\n"])


class DescribeStackTest(BenchmarkTempDirTest):
    """Results must state the configuration the run actually used."""

    def test_defaults_report_everything_disabled(self):
        stack = self.make().describe_stack()
        self.assertEqual(stack["queue_service"], "disabled")
        self.assertEqual(stack["rtsp_streamer"], "disabled")
        self.assertEqual(stack["diarization"], "disabled")
        self.assertEqual(stack["stack_managed_by_script"], "yes")

    def test_queue_true_is_not_reported_as_disabled(self):
        # --make_arg QUEUE=true starts the vision profile, so claiming
        # rtsp_streamer is disabled would misreport the configuration.
        stack = self.make(make_args=["QUEUE=true"]).describe_stack()
        self.assertEqual(stack["rtsp_streamer"], "enabled")
        self.assertEqual(stack["make_args"], "QUEUE=true")

    def test_records_make_args_and_skip_stack(self):
        stack = self.make(
            skip_stack=True, make_args=["QUEUE=false", "TARGET_DEVICE=GPU"]
        ).describe_stack()
        self.assertIn("TARGET_DEVICE=GPU", stack["make_args"])
        self.assertEqual(stack["stack_managed_by_script"], "no")


class ResultsDirWiringTest(BenchmarkTempDirTest):
    """--results_dir must reach the service that writes the metrics."""

    def test_stack_env_points_the_bind_mount_at_results_dir(self):
        env = self.make().configure_stack_env()
        self.assertEqual(env["KIOSK_RESULTS_HOST_DIR"], self.results_dir)

    def test_warns_under_skip_stack_with_a_divergent_results_dir(self):
        bench = self.make(skip_stack=True)
        with mock.patch("builtins.print") as printed:
            bench.warn_if_results_dir_unreachable()
        output = " ".join(str(c) for c in printed.call_args_list)
        self.assertIn("WARNING", output)

    def test_no_warning_when_results_dir_matches_the_app(self):
        bench = v2v.SmartKioskV2VBenchmark(
            app_dir=self.app_dir,
            results_dir=os.path.join(self.app_dir, "results"),
            skip_stack=True,
        )
        with mock.patch("builtins.print") as printed:
            bench.warn_if_results_dir_unreachable()
        output = " ".join(str(c) for c in printed.call_args_list)
        self.assertNotIn("WARNING", output)


class CrossCheckTest(unittest.TestCase):

    def test_ok_within_tolerance(self):
        result = v2v.SmartKioskV2VBenchmark.cross_check(
            {"median_latency_ms": 750.4},
            {"summary": {"voice_to_voice_ms": {"median": 750.0}}},
        )
        self.assertEqual(result["status"], "ok")

    def test_mismatch_outside_tolerance(self):
        result = v2v.SmartKioskV2VBenchmark.cross_check(
            {"median_latency_ms": 900.0},
            {"summary": {"voice_to_voice_ms": {"median": 750.0}}},
        )
        self.assertEqual(result["status"], "MISMATCH")
        self.assertEqual(result["delta_ms"], 150.0)

    def test_accepts_a_scalar_summary_value(self):
        result = v2v.SmartKioskV2VBenchmark.cross_check(
            {"median_latency_ms": 750.0},
            {"summary": {"voice_to_voice_ms": 750.0}},
        )
        self.assertEqual(result["status"], "ok")

    def test_unavailable_when_either_side_is_missing(self):
        self.assertEqual(
            v2v.SmartKioskV2VBenchmark.cross_check({}, {})["status"],
            "unavailable",
        )
        self.assertEqual(
            v2v.SmartKioskV2VBenchmark.cross_check(
                {"median_latency_ms": 750.0}, {}
            )["status"],
            "unavailable",
        )

    def test_turn_count_mismatch_fails_even_when_medians_agree(self):
        # A dropped or duplicated turn can leave the median untouched, so
        # comparing medians alone would miss a broken metrics path.
        result = v2v.SmartKioskV2VBenchmark.cross_check(
            {"median_latency_ms": 750.0, "total_transactions": 7},
            {
                "summary": {"voice_to_voice_ms": {"median": 750.0}},
                "turns_total": 8,
            },
        )
        self.assertEqual(result["status"], "MISMATCH")
        self.assertEqual(result["turn_count_status"], "MISMATCH")
        self.assertEqual(result["reported_turns"], 8)
        self.assertEqual(result["emitted_turns"], 7)

    def test_matching_turn_counts_keep_the_run_ok(self):
        result = v2v.SmartKioskV2VBenchmark.cross_check(
            {"median_latency_ms": 750.0, "total_transactions": 8},
            {
                "summary": {"voice_to_voice_ms": {"median": 750.0}},
                "turns_total": 8,
            },
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["turn_count_status"], "ok")


class ConstructorValidationTest(BenchmarkTempDirTest):
    """Bad paths must fail immediately, not after a ~10 minute bring-up."""

    def test_missing_makefile_is_rejected(self):
        os.remove(os.path.join(self.app_dir, "Makefile"))
        with self.assertRaises(ValueError):
            self.make()

    def test_missing_app_benchmark_script_is_rejected_up_front(self):
        # Previously this was only noticed inside run_benchmark_script, i.e.
        # after make up and the health wait had already been paid for.
        os.remove(self.app_script)
        with self.assertRaisesRegex(ValueError, "Benchmark script not found"):
            self.make()

    def test_missing_app_dir_is_rejected(self):
        with self.assertRaises(ValueError):
            v2v.SmartKioskV2VBenchmark(
                app_dir=os.path.join(self.tmp, "nope"),
                results_dir=self.results_dir,
            )


class WriteCsvSummaryTest(BenchmarkTempDirTest):

    def test_flattens_nested_values_and_counts_lists(self):
        os.makedirs(self.results_dir, exist_ok=True)
        csv_path = os.path.join(self.results_dir, "summary.csv")
        v2v.SmartKioskV2VBenchmark._write_csv_summary(
            {
                "runs": 8,
                "vlm_metrics": {
                    "median_latency_ms": 750.0,
                    "transactions": [1, 2, 3],
                },
            },
            csv_path,
        )
        with open(csv_path) as f:
            rows = [line.strip() for line in f if line.strip()]
        self.assertEqual(rows[0], "Metric,Value")
        self.assertIn("runs,8", rows)
        self.assertIn("vlm_metrics_median_latency_ms,750.0", rows)
        # Lists are summarised by length, not serialised.
        self.assertIn("vlm_metrics_transactions,3", rows)


class ConfigurationTest(BenchmarkTempDirTest):

    def test_health_endpoints_are_overridable(self):
        bench = self.make(
            health_endpoints={"kiosk-core": "http://host:9999/health"}
        )
        self.assertEqual(
            bench.health_endpoints, {"kiosk-core": "http://host:9999/health"}
        )

    def test_default_health_endpoints_are_not_shared_between_instances(self):
        bench = self.make()
        bench.health_endpoints["kiosk-core"] = "mutated"
        self.assertEqual(
            v2v.SmartKioskV2VBenchmark.DEFAULT_HEALTH_ENDPOINTS["kiosk-core"],
            "http://localhost:8012/health",
        )

    def test_make_args_default_to_queue_false(self):
        self.assertEqual(self.make().make_args, ["QUEUE=false"])

    def test_make_args_are_overridable_including_empty(self):
        self.assertEqual(self.make(make_args=[]).make_args, [])
        self.assertEqual(
            self.make(make_args=["QUEUE=true"]).make_args, ["QUEUE=true"]
        )

    def test_rejects_a_directory_without_a_makefile(self):
        empty = os.path.join(self.tmp, "empty")
        os.makedirs(empty)
        with self.assertRaises(ValueError):
            v2v.SmartKioskV2VBenchmark(
                app_dir=empty, results_dir=self.results_dir
            )

    def test_rejects_a_missing_directory(self):
        with self.assertRaises(ValueError):
            v2v.SmartKioskV2VBenchmark(
                app_dir=os.path.join(self.tmp, "nope"),
                results_dir=self.results_dir,
            )


class WriteMeasuredWindowTest(BenchmarkTempDirTest):

    def test_writes_the_window_with_a_duration(self):
        self.make().write_measured_window(1_000_000, 1_060_000)
        path = os.path.join(self.results_dir, "measured_window.json")
        with open(path) as f:
            payload = json.load(f)
        self.assertEqual(payload["start_ms"], 1_000_000)
        self.assertEqual(payload["end_ms"], 1_060_000)
        self.assertEqual(payload["duration_s"], 60.0)


class StaleReportTest(BenchmarkTempDirTest):
    """A crashed measured run must not cross-check last run's numbers."""

    def write_report(self, label, finished_at, turns):
        path = os.path.join(self.results_dir, label + ".json")
        with open(path, "w") as f:
            json.dump(
                {
                    "label": label,
                    "started_at": finished_at,
                    "finished_at": finished_at,
                    "turns": turns,
                    "summary": {"voice_to_voice_ms": {"median": 750.0}},
                },
                f,
            )
        return path

    def test_report_predating_the_measured_run_is_rejected(self):
        self.write_report("lbl", "2026-01-01T00:00:00", [{"run": 1}])
        not_before = int(
            datetime(2026, 6, 1, 0, 0, 0).timestamp() * 1000
        )
        report = self.make().collect_benchmark_report(
            "lbl", not_before_ms=not_before
        )
        self.assertEqual(report, {})

    def test_fresh_report_is_accepted(self):
        self.write_report("lbl", "2026-06-02T00:00:00", [{"run": 1}])
        not_before = int(
            datetime(2026, 6, 1, 0, 0, 0).timestamp() * 1000
        )
        report = self.make().collect_benchmark_report(
            "lbl", not_before_ms=not_before
        )
        self.assertEqual(report["turns_total"], 1)

    def test_errored_turns_are_not_counted_as_measured(self):
        # Failed turns never emit a start/end pair, so counting them would
        # fail the cross-check and blame the metrics path.
        self.write_report(
            "lbl",
            "2026-06-02T00:00:00",
            [{"run": 1}, {"run": 2, "error": "tts synth failed"}],
        )
        report = self.make().collect_benchmark_report("lbl")
        self.assertEqual(report["turns_total"], 1)
        self.assertEqual(report["turns_recorded"], 2)
        self.assertEqual(report["turns_failed"], 1)


class QueueFlagConsistencyTest(BenchmarkTempDirTest):
    """The app Makefile derives both the profile and the env var from QUEUE."""

    def test_default_disables_queue_consistently(self):
        bench = self.make()
        self.assertFalse(bench.queue_enabled)
        env = bench.configure_stack_env()
        self.assertEqual(env["KIOSK_CORE_QUEUE_SERVICE_ENABLED"], "false")
        stack = bench.describe_stack()
        self.assertEqual(stack["queue_service"], "disabled")
        self.assertEqual(stack["rtsp_streamer"], "disabled")

    def test_queue_true_is_reported_and_exported_consistently(self):
        bench = self.make(make_args=["QUEUE=true"])
        self.assertTrue(bench.queue_enabled)
        env = bench.configure_stack_env()
        # Must not contradict the make argument: the Makefile re-exports
        # this derived from QUEUE, so a false here would be a lie.
        self.assertEqual(env["KIOSK_CORE_QUEUE_SERVICE_ENABLED"], "true")
        stack = bench.describe_stack()
        self.assertEqual(stack["queue_service"], "enabled")
        self.assertEqual(stack["rtsp_streamer"], "enabled")

    def test_last_queue_argument_wins(self):
        bench = self.make(make_args=["QUEUE=true", "QUEUE=false"])
        self.assertFalse(bench.queue_enabled)


class OvmsReadinessTest(unittest.TestCase):

    def ready(self, name, payload):
        return v2v.SmartKioskV2VBenchmark._endpoint_is_ready(
            name, json.dumps(payload).encode()
        )

    def test_empty_model_list_is_not_ready(self):
        self.assertFalse(self.ready("ovms-llm", {"data": []}))

    def test_loading_model_is_not_ready(self):
        self.assertFalse(
            self.ready("ovms-llm", {"data": [{"state": "LOADING"}]})
        )

    def test_available_model_is_ready(self):
        self.assertTrue(
            self.ready("ovms-llm", {"data": [{"state": "AVAILABLE"}]})
        )

    def test_model_without_a_state_field_is_assumed_ready(self):
        self.assertTrue(self.ready("ovms-llm", {"data": [{"id": "qwen"}]}))

    def test_unparseable_body_does_not_block_startup(self):
        self.assertTrue(
            v2v.SmartKioskV2VBenchmark._endpoint_is_ready(
                "ovms-llm", b"not json"
            )
        )

    def test_other_services_are_ready_on_any_2xx(self):
        self.assertTrue(
            v2v.SmartKioskV2VBenchmark._endpoint_is_ready(
                "kiosk-core", b""
            )
        )


class WaitForHealthTest(BenchmarkTempDirTest):

    def test_raises_and_names_the_pending_services(self):
        bench = self.make()
        with mock.patch.object(
            v2v.urllib.request,
            "urlopen",
            side_effect=v2v.urllib.error.URLError("refused"),
        ), mock.patch.object(v2v.time, "sleep"):
            with self.assertRaises(RuntimeError) as ctx:
                bench.wait_for_health(timeout=1)
        message = str(ctx.exception)
        self.assertIn("kiosk-core", message)
        self.assertIn("rag-service", message)


class _FakeResponse:
    """Minimal urlopen context manager stand-in."""

    def __init__(self, status=200, body=b'{"data": [{"state": "AVAILABLE"}]}'):
        self.status = status
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class RunEndToEndTest(BenchmarkTempDirTest):
    """run() with every external effect mocked out."""

    def setUp(self):
        super().setUp()
        self.made = []

    def _fake_run(self, cmd, **kwargs):
        self.made.append(cmd)
        # Emit the per-turn events the child benchmark would have caused
        # kiosk-core to log, plus the report it writes itself.
        if any("v2v_scripted" in str(part) for part in cmd):
            label = cmd[cmd.index("--label") + 1]
            now = int(v2v.time.time() * 1000)
            path = os.path.join(
                self.results_dir, "vlm_application_metrics_0.txt"
            )
            with open(path, "a") as f:
                for i in range(2):
                    f.write(_event_line("kiosk-voice", "start", now + i * 10))
                    f.write(
                        _event_line("kiosk-voice", "end", now + i * 10 + 700)
                    )
            with open(
                os.path.join(self.results_dir, label + ".json"), "w"
            ) as f:
                json.dump(
                    {
                        "label": label,
                        "started_at": datetime.now().isoformat(),
                        "finished_at": datetime.now().isoformat(),
                        "turns": [{"run": 1}, {"run": 2}],
                        "summary": {
                            "voice_to_voice_ms": {"median": 700.0}
                        },
                    },
                    f,
                )
        return subprocess.CompletedProcess(cmd, 0)

    def run_benchmark(self, bench, **kwargs):
        params = dict(
            script=None,
            runs=2,
            warmup_runs=0,
            init_duration=0,
            health_timeout=1,
            explicit_end_mark=False,
            label="unit-label",
        )
        params.update(kwargs)
        with mock.patch.object(
            v2v.subprocess, "run", side_effect=self._fake_run
        ), mock.patch.object(
            v2v.urllib.request, "urlopen", return_value=_FakeResponse()
        ), mock.patch.object(
            v2v.time, "sleep"
        ):
            return bench.run(**params)

    def test_happy_path_collects_and_cross_checks(self):
        results = self.run_benchmark(self.make())
        self.assertEqual(results["vlm_metrics"]["total_transactions"], 2)
        self.assertEqual(results["cross_check"]["status"], "ok")

    def test_stack_is_torn_down_and_logs_pruned(self):
        bench = self.make()
        with mock.patch.object(bench, "prune_warmup_from_vlm_logs") as prune:
            self.run_benchmark(bench)
        joined = [" ".join(str(p) for p in cmd) for cmd in self.made]
        self.assertTrue(any("down" in c for c in joined), joined)
        prune.assert_called_once()

    def test_keep_stack_does_not_prune_the_live_logs(self):
        # os.replace() under a live RotatingFileHandler would leave
        # kiosk-core writing into *.with-warmup, which the consolidator's
        # glob ignores -- the exact lost-writes bug guarded elsewhere.
        bench = self.make(keep_stack=True)
        with mock.patch.object(bench, "prune_warmup_from_vlm_logs") as prune:
            self.run_benchmark(bench)
        prune.assert_not_called()

    def test_skip_stack_neither_starts_nor_prunes(self):
        bench = self.make(skip_stack=True)
        with mock.patch.object(bench, "prune_warmup_from_vlm_logs") as prune:
            self.run_benchmark(bench)
        prune.assert_not_called()
        joined = [" ".join(str(p) for p in cmd) for cmd in self.made]
        self.assertFalse(
            any("make" in c and " up" in c for c in joined), joined
        )

    def test_warmup_failure_stops_before_the_measured_run(self):
        def failing(cmd, **kwargs):
            if any("v2v_scripted" in str(p) for p in cmd):
                return subprocess.CompletedProcess(cmd, 3)
            return subprocess.CompletedProcess(cmd, 0)

        bench = self.make()
        with mock.patch.object(
            v2v.subprocess, "run", side_effect=failing
        ), mock.patch.object(
            v2v.urllib.request, "urlopen", return_value=_FakeResponse()
        ), mock.patch.object(v2v.time, "sleep"):
            with self.assertRaises(RuntimeError):
                bench.run(
                    script=None,
                    runs=2,
                    warmup_runs=1,
                    init_duration=0,
                    health_timeout=1,
                    explicit_end_mark=False,
                    label="unit-label",
                )

    def test_stack_is_stopped_even_when_make_up_fails(self):
        def failing_up(cmd, **kwargs):
            self.made.append(cmd)
            if "up" in cmd:
                return subprocess.CompletedProcess(cmd, 2)
            return subprocess.CompletedProcess(cmd, 0)

        bench = self.make()
        with mock.patch.object(
            v2v.subprocess, "run", side_effect=failing_up
        ), mock.patch.object(v2v.time, "sleep"):
            with self.assertRaises(RuntimeError):
                bench.run(
                    script=None,
                    runs=1,
                    warmup_runs=0,
                    init_duration=0,
                    health_timeout=1,
                    explicit_end_mark=False,
                    label="unit-label",
                )
        joined = [" ".join(str(p) for p in cmd) for cmd in self.made]
        self.assertTrue(any("down" in c for c in joined), joined)


class MainExitCodeTest(unittest.TestCase):

    def _main_with(self, results, extra_argv=()):
        argv = ["prog", "--app_dir", "/app"] + list(extra_argv)
        with mock.patch.object(v2v.sys, "argv", argv), mock.patch.object(
            v2v.SmartKioskV2VBenchmark, "__init__", return_value=None
        ), mock.patch.object(
            v2v.SmartKioskV2VBenchmark, "run", return_value=results
        ):
            return v2v.main()

    def test_zero_transactions_fails(self):
        rc = self._main_with(
            {"vlm_metrics": {"total_transactions": 0}, "cross_check": {}}
        )
        self.assertEqual(rc, 1)

    def test_cross_check_mismatch_fails(self):
        rc = self._main_with(
            {
                "vlm_metrics": {"total_transactions": 8},
                "cross_check": {"status": "MISMATCH"},
                "benchmark_exit_code": 0,
            }
        )
        self.assertEqual(rc, 1)

    def test_mismatch_can_be_downgraded(self):
        rc = self._main_with(
            {
                "vlm_metrics": {"total_transactions": 8},
                "cross_check": {"status": "MISMATCH"},
                "benchmark_exit_code": 0,
            },
            extra_argv=["--allow_cross_check_mismatch"],
        )
        self.assertEqual(rc, 0)

    def test_child_exit_code_is_propagated(self):
        rc = self._main_with(
            {
                "vlm_metrics": {"total_transactions": 8},
                "cross_check": {"status": "ok"},
                "benchmark_exit_code": 7,
            }
        )
        self.assertEqual(rc, 7)

    def test_happy_path_returns_zero(self):
        rc = self._main_with(
            {
                "vlm_metrics": {"total_transactions": 8},
                "cross_check": {"status": "ok"},
                "benchmark_exit_code": 0,
            }
        )
        self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
