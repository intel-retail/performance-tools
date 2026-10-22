"""Unit tests for end-to-end benchmark result consolidation."""

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import benchmark_signage_consolidate_e2e as consolidate


class TestConsolidationHelpers(unittest.TestCase):

    def test_latest_load_stats_and_row(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertIsNone(consolidate.latest(root, "*.json"))
            first = root / "first.json"
            first.write_text('{"ok": true}')
            second = root / "second.json"
            second.write_text('{"ok": false}')
            self.assertEqual(consolidate.latest(root, "*.json"), second)
            self.assertEqual(consolidate.load(second), {"ok": False})
            self.assertIsNone(consolidate.load(None))

        self.assertEqual(consolidate.stats([]), {"samples": 0})
        self.assertEqual(consolidate.stats([1, 2, 3]), {
            "samples": 3, "avg_ms": 2, "p50_ms": 2, "p95_ms": 3, "max_ms": 3,
        })
        self.assertEqual(
            consolidate.row("stage", "source", None, [1, 3])["avg_ms"], 2,
        )
        self.assertEqual(
            consolidate.row("count", "source", None, value=4)["value"], 4,
        )


class TestConsolidate(unittest.TestCase):

    @staticmethod
    def write_json(directory, name, payload):
        path = Path(directory) / name
        path.write_text(json.dumps(payload))
        return path

    def test_combines_all_sources_and_host_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pid_path = self.write_json(root, "pid_benchmark_20260101_000000.json",
                                       {"fps_dlstreamer": 30})
            self.write_json(root, "pid_benchmark_20260101_000000_frames.json", [
                {"pipeline_latency_reported": 10, "mqtt_arrival_delay_ms": 2},
            ])
            web_path = self.write_json(root, "web_ui_benchmark_20260101_000000.json", {})
            self.write_json(root, "web_ui_benchmark_20260101_000000_frames.json", [
                {"mqtt_to_web_ui_ms": 1, "callback_total_ms": 2, "queue_wait_ms": 3,
                 "cooldown_remaining_ms": 4, "cooldown_suppressed": True, "selector_total_ms": 5},
            ])
            self.write_json(root, "web_ui_benchmark_20260101_000000_ads.json", [
                {"ad_id": "p", "path": "predefined", "ase_query_ms": 6,
                 "aig_generation_ms": None, "ready_to_served_ms": 7, "browser_fetch_ms": 8,
                 "browser_render_ms": 9, "frame_to_ad_ready_ms": 10, "frame_to_render_ms": 34},
                {"ad_id": "d", "path": "dynamic", "ase_query_ms": 11,
                 "aig_generation_ms": 12, "ready_to_served_ms": 13, "browser_fetch_ms": 14,
                 "browser_render_ms": 15, "frame_to_ad_ready_ms": 16, "frame_to_render_ms": 58},
            ])
            self.write_json(root, "aig_benchmark_20260101_000000.json", [
                {"kind": "predefined", "http_elapsed_ms": 20, "vector_search_ms": 4},
                {"kind": "dynamic", "call": 1, "http_elapsed_ms": 30, "inference_ms": 8},
                {"kind": "dynamic", "call": 2, "http_elapsed_ms": 10, "inference_ms": 3},
            ])
            self.write_json(root, "qdrant_benchmark_20260101_000000.json", [
                {"kind": "vector_search", "points": 100, "queries_per_sec": 5, "avg_query_ms": 2},
                {"kind": "predefined_ad_index", "index_http_ms": 7},
            ])
            self.write_json(root, "e2e_metrics_20260101.json", {
                "cpu_utilization": [[0, 10]], "gpu_utilization": [[0, 20]],
                "npu_utilization": [[0, 30]], "memory": [[0, 0, 0, 0, 40]],
                "power": [[0, 50]],
            })

            rows = consolidate.consolidate(root)

        stages = [entry["stage"] for entry in rows]
        self.assertIn("1. PID throughput (FPS)", stages)
        self.assertIn("E2E frame -> ad rendered", stages)
        self.assertIn("7. AIG warm calls", stages)
        self.assertIn("Qdrant search @ 100 points", stages)
        self.assertIn("Resources during e2e: Package power W", stages)
        self.assertEqual(next(row for row in rows if row["stage"] == "Ads generated (predefined / dynamic)")["value"],
                         "1 / 1")
        self.assertEqual(next(row for row in rows if row["stage"] == "7. AIG warm calls")["avg_ms"], 10)

    def test_empty_directory_returns_no_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(consolidate.consolidate(Path(directory)), [])

    def test_main_writes_json_csv_and_reports_empty_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_json(root, "pid_benchmark_20260101_000000.json", {"fps_dlstreamer": 25})
            self.write_json(root, "pid_benchmark_20260101_000000_frames.json", [])
            output = root / "out"
            with (
                mock.patch.object(consolidate.sys, "argv", ["consolidate", "--results", directory,
                                                              "--output", str(output)]),
                mock.patch("builtins.print"),
            ):
                consolidate.main()
            json_files = list(output.glob("e2e_consolidated_*.json"))
            csv_files = list(output.glob("e2e_consolidated_*.csv"))
            self.assertEqual(len(json_files), 1)
            self.assertEqual(len(csv_files), 1)
            self.assertGreater(len(json.loads(json_files[0].read_text())["stages"]), 0)
            with csv_files[0].open(newline="") as source:
                self.assertGreater(len(list(csv.DictReader(source))), 0)

        with tempfile.TemporaryDirectory() as directory:
            with (
                mock.patch.object(consolidate.sys, "argv", ["consolidate", "--results", directory]),
                mock.patch("builtins.print") as output,
            ):
                with self.assertRaises(SystemExit) as error:
                    consolidate.main()
            self.assertEqual(error.exception.code, 1)
            self.assertEqual(output.call_args.kwargs["file"], consolidate.sys.stderr)


if __name__ == "__main__":
    unittest.main()
