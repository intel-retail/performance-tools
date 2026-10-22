'''
* Copyright (C) 2025 Intel Corporation.
*
* SPDX-License-Identifier: Apache-2.0
'''

import json
import os
import shutil
import tempfile
import unittest

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
        # The constructor requires a Makefile to exist.
        with open(os.path.join(self.app_dir, "Makefile"), "w") as f:
            f.write("up:\n\t@true\n")

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


class CleanPreviousMetricsTest(BenchmarkTempDirTest):

    def test_removes_stale_files_when_managing_the_stack(self):
        os.makedirs(self.results_dir, exist_ok=True)
        path = self.write_log("vlm_application_metrics_0.txt", ["stale\n"])
        self.make()._clean_previous_metrics()
        self.assertFalse(os.path.exists(path))

    def test_is_a_no_op_under_skip_stack(self):
        # kiosk-core already holds the file open; unlinking it would send
        # every later write to an unlinked inode and lose the whole run.
        os.makedirs(self.results_dir, exist_ok=True)
        path = self.write_log("vlm_application_metrics_0.txt", ["live\n"])
        self.make(skip_stack=True)._clean_previous_metrics()
        self.assertTrue(os.path.exists(path))


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

        import builtins
        builtins.open = failing_open
        try:
            bench.prune_warmup_from_vlm_logs(since_ms=5000)
        finally:
            builtins.open = real_open

        self.assertTrue(os.path.exists(path))
        with open(path) as f:
            self.assertEqual(len(f.readlines()), 4)
        self.assertFalse(os.path.exists(path + ".pruned.tmp"))


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


if __name__ == "__main__":
    unittest.main()
