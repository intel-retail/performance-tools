"""Unit tests for the ASe/AIG benchmark helpers."""

import base64
import csv
import json
import os
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import benchmark_signage_aig as benchmark


class TestParseServerTiming(unittest.TestCase):

    def test_parses_valid_durations_and_ignores_invalid_entries(self):
        header = (
            "query_build;dur=1.25, vector_search;dur=2, "
            "ranking_retrieval;dur=invalid, cache;desc=hit"
        )

        self.assertEqual(
            benchmark.parse_server_timing(header),
            {"query_build_ms": 1.25, "vector_search_ms": 2.0},
        )

    def test_empty_header_returns_empty_timings(self):
        self.assertEqual(benchmark.parse_server_timing(""), {})


class TestRequestAd(unittest.TestCase):

    @staticmethod
    def response(content=b"image", headers=None, body=None):
        response = mock.Mock()
        response.content = content
        response.headers = headers or {}
        response.json.return_value = body
        return response

    def test_predefined_response_decodes_images_and_applies_search_sla(self):
        session = mock.Mock()
        session.post.return_value = self.response(
            content=b"response",
            headers={"Server-Timing": "query_build;dur=4"},
            body=[{"imgb64": base64.b64encode(b"image-bytes").decode("ascii")}, {}],
        )

        with mock.patch.object(benchmark.time, "perf_counter", side_effect=[1.0, 1.025]):
            result = benchmark.request_ad(
                session, "https://example.test/search", {"query": "coffee"},
                "predefined", 2, 10,
            )

        session.post.assert_called_once_with(
            "https://example.test/search",
            json={"query": "coffee"},
            headers={"X-Benchmark-Timing": "true"},
            timeout=10,
        )
        self.assertEqual(result["kind"], "predefined")
        self.assertEqual(result["call"], 2)
        self.assertEqual(result["http_elapsed_ms"], 25.0)
        self.assertEqual(result["sla_ms"], 5000)
        self.assertTrue(result["sla_met"])
        self.assertEqual(result["result_count"], 2)
        self.assertEqual(result["image_size_kb"], round(len(b"image-bytes") / 1024, 3))
        self.assertEqual(result["query_build_ms"], 4.0)
        self.assertIsNone(result["vector_search_ms"])

    def test_dynamic_response_requires_jpeg_and_uses_cold_start_sla(self):
        session = mock.Mock()
        session.post.return_value = self.response(
            content=b"jpeg-data",
            headers={
                "Content-Type": "image/jpeg; charset=binary",
                "Server-Timing": "inference;dur=12",
            },
        )

        with mock.patch.object(benchmark.time, "perf_counter", side_effect=[3.0, 3.01]):
            result = benchmark.request_ad(
                session, "https://example.test/generate", {}, "dynamic", 1, 30,
                cold_start=True,
            )

        self.assertEqual(result["sla_ms"], 60000)
        self.assertEqual(result["result_count"], 1)
        self.assertEqual(result["image_size_kb"], round(len(b"jpeg-data") / 1024, 3))
        self.assertEqual(result["inference_ms"], 12.0)
        self.assertIsNone(result["model_load_ms"])

    def test_dynamic_warm_call_uses_warm_sla(self):
        session = mock.Mock()
        session.post.return_value = self.response(headers={"Content-Type": "image/jpeg"})

        with mock.patch.object(benchmark.time, "perf_counter", side_effect=[0.0, 0.001]):
            result = benchmark.request_ad(
                session, "https://example.test/generate", {}, "dynamic", 2, 30,
            )

        self.assertEqual(result["sla_ms"], 15000)

    def test_predefined_response_must_be_a_list(self):
        session = mock.Mock()
        session.post.return_value = self.response(body={"imgb64": "aW1hZ2U="})

        with self.assertRaisesRegex(ValueError, "did not return a list"):
            benchmark.request_ad(session, "https://example.test/search", {}, "predefined", 1, 10)

    def test_dynamic_response_must_be_jpeg(self):
        session = mock.Mock()
        session.post.return_value = self.response(headers={"Content-Type": "application/json"})

        with self.assertRaisesRegex(ValueError, "did not return a JPEG"):
            benchmark.request_ad(session, "https://example.test/generate", {}, "dynamic", 1, 10)

    def test_http_errors_are_propagated(self):
        session = mock.Mock()
        response = self.response()
        response.raise_for_status.side_effect = RuntimeError("HTTP failure")
        session.post.return_value = response

        with self.assertRaisesRegex(RuntimeError, "HTTP failure"):
            benchmark.request_ad(session, "https://example.test/search", {}, "predefined", 1, 10)


class TestSummarize(unittest.TestCase):

    def test_averages_calls_by_kind_and_separates_first_from_warm_dynamic(self):
        rows = [
            {
                "kind": "predefined", "call": 1, "sla_met": True,
                "http_elapsed_ms": 10, "response_size_kb": 2, "image_size_kb": 1,
                "query_build_ms": 4, "vector_search_ms": 6, "ranking_retrieval_ms": None,
            },
            {
                "kind": "predefined", "call": 2, "sla_met": False,
                "http_elapsed_ms": 20, "response_size_kb": 4, "image_size_kb": 3,
                "query_build_ms": 8, "vector_search_ms": None, "ranking_retrieval_ms": 12,
            },
            {
                "kind": "dynamic", "call": 1, "sla_met": True,
                "http_elapsed_ms": 100, "response_size_kb": 10, "image_size_kb": 9,
                "model_load_ms": 70, "inference_ms": 20, "overlay_ms": 5,
                "jpeg_encode_ms": 3, "model_unload_ms": 1, "model_cleanup_ms": 1,
            },
            {
                "kind": "dynamic", "call": 2, "sla_met": True,
                "http_elapsed_ms": 50, "response_size_kb": 8, "image_size_kb": 7,
                "model_load_ms": None, "inference_ms": 30, "overlay_ms": 10,
                "jpeg_encode_ms": 5, "model_unload_ms": 2, "model_cleanup_ms": 3,
            },
        ]

        summary = benchmark.summarize(rows, "GPU", 150)

        self.assertEqual(summary["device"], "GPU")
        self.assertEqual(summary["capture_elapsed_ms"], 150)
        self.assertEqual(summary["predefined_calls"], 2)
        self.assertEqual(summary["predefined_sla_met"], 1)
        self.assertEqual(summary["avg_predefined_http_elapsed_ms"], 15.0)
        self.assertEqual(summary["avg_predefined_query_build_ms"], 6.0)
        self.assertEqual(summary["avg_predefined_ranking_retrieval_ms"], 12.0)
        self.assertEqual(summary["dynamic_calls"], 2)
        self.assertEqual(summary["avg_dynamic_inference_ms"], 25.0)
        self.assertEqual(summary["dynamic_first_call_http_elapsed_ms"], 100)
        self.assertEqual(summary["avg_dynamic_warm_http_elapsed_ms"], 50.0)

    def test_empty_rows_have_zero_counts_and_no_averages(self):
        summary = benchmark.summarize([], "CPU", 0)

        self.assertEqual(summary["predefined_calls"], 0)
        self.assertEqual(summary["dynamic_calls"], 0)
        self.assertIsNone(summary["avg_predefined_http_elapsed_ms"])
        self.assertIsNone(summary["dynamic_first_call_http_elapsed_ms"])
        self.assertIsNone(summary["avg_dynamic_warm_http_elapsed_ms"])


class TestRun(unittest.TestCase):

    @staticmethod
    def args(directory, **overrides):
        values = {
            "base_url": "https://example.test/api",
            "insecure": False,
            "cold_start": False,
            "timeout": 5,
            "query": "coffee",
            "description": "A cup of coffee",
            "device": "CPU",
            "slogan": "Fresh",
            "iterations": 1,
            "duration": 0,
            "output": Path(directory),
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_runs_mocked_workflow_and_writes_json_and_csv_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(directory, base_url="https://example.test/api/")
            session = mock.Mock()
            session.get.return_value = mock.Mock()

            def make_row(_session, _url, _payload, kind, call, _timeout, _cold_start=False):
                row = {
                    "timestamp": "2026-01-01T00:00:00+00:00",
                    "kind": kind,
                    "call": call,
                    "http_elapsed_ms": 10.0,
                    "sla_ms": 5000 if kind == "predefined" else 15000,
                    "sla_met": True,
                    "response_size_kb": 2.0,
                    "image_size_kb": 1.0,
                    "result_count": 1,
                }
                stages = benchmark.SEARCH_STAGES if kind == "predefined" else benchmark.GENERATION_STAGES
                row.update({f"{stage}_ms": None for stage in stages})
                return row

            with (
                mock.patch.dict(os.environ, {"REQUESTS_CA_BUNDLE": "", "CURL_CA_BUNDLE": ""}),
                mock.patch.object(benchmark.requests, "Session", return_value=session),
                mock.patch.object(benchmark, "request_ad", side_effect=make_row) as request_ad,
                mock.patch.object(benchmark.requests.packages.urllib3, "disable_warnings"),
                mock.patch("builtins.print"),
            ):
                benchmark.run(args)

            self.assertTrue(session.verify)
            session.get.assert_called_once_with(
                "https://example.test/api/aig/hstatus/1", timeout=3,
            )
            session.get.return_value.raise_for_status.assert_called_once_with()
            self.assertEqual(request_ad.call_count, 2)

            json_files = [
                path for path in Path(directory).glob("aig_benchmark_*.json")
                if not path.name.endswith("_summary.json")
            ]
            summary_files = list(Path(directory).glob("aig_benchmark_*_summary.json"))
            csv_files = [
                path for path in Path(directory).glob("aig_benchmark_*.csv")
                if not path.name.endswith("_summary.csv")
            ]
            summary_csv_files = list(Path(directory).glob("aig_benchmark_*_summary.csv"))
            self.assertEqual(len(json_files), 1)
            self.assertEqual(len(summary_files), 1)
            self.assertEqual(len(csv_files), 1)
            self.assertEqual(len(summary_csv_files), 1)

            rows = json.loads(json_files[0].read_text())
            summary = json.loads(summary_files[0].read_text())
            self.assertEqual([row["kind"] for row in rows], ["predefined", "dynamic"])
            self.assertTrue(all(row["device"] == "CPU" for row in rows))
            self.assertEqual(summary["predefined_calls"], 1)
            self.assertEqual(summary["dynamic_calls"], 1)
            with csv_files[0].open(newline="") as source:
                csv_rows = list(csv.DictReader(source))
            self.assertEqual(len(csv_rows), 2)

    def test_tls_readiness_error_gets_actionable_message(self):
        args = self.args(tempfile.gettempdir())
        session = mock.Mock()
        session.get.side_effect = benchmark.requests.exceptions.SSLError("bad certificate")

        with mock.patch.object(benchmark.requests, "Session", return_value=session):
            with self.assertRaisesRegex(RuntimeError, "TLS certificate verification failed"):
                benchmark.run(args)

    def test_readiness_retries_until_deadline_then_reports_failure(self):
        args = self.args(tempfile.gettempdir(), timeout=1)
        session = mock.Mock()
        session.get.side_effect = benchmark.requests.RequestException("not ready")

        with (
            mock.patch.object(benchmark.requests, "Session", return_value=session),
            mock.patch.object(benchmark.time, "monotonic", side_effect=[0, 0, 1]),
            mock.patch.object(benchmark.time, "sleep") as sleep,
        ):
            with self.assertRaisesRegex(RuntimeError, "AIG did not become ready"):
                benchmark.run(args)

        self.assertEqual(session.get.call_count, 2)
        sleep.assert_called_once_with(1)

    def test_cold_start_restarts_server_and_empty_search_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(directory, cold_start=True)
            session = mock.Mock()
            session.get.return_value = mock.Mock()
            empty_result = {"result_count": 0}

            with (
                mock.patch.object(benchmark.requests, "Session", return_value=session),
                mock.patch.object(benchmark.subprocess, "run") as docker_restart,
                mock.patch.object(benchmark, "request_ad", return_value=empty_result),
            ):
                with self.assertRaisesRegex(RuntimeError, "No predefined image matched"):
                    benchmark.run(args)

            docker_restart.assert_called_once_with(["docker", "restart", "aig-server"], check=True)
            self.assertEqual(session.get.call_count, 1)

    def test_cli_rejects_invalid_iteration_count(self):
        script = Path(benchmark.__file__)
        with (
            mock.patch.object(sys, "argv", [str(script), "--iterations", "0"]),
            mock.patch("builtins.print"),
        ):
            with self.assertRaises(SystemExit) as error:
                runpy.run_path(str(script), run_name="__main__")

        self.assertEqual(error.exception.code, 2)

    def test_cli_reports_runtime_failure_with_nonzero_exit(self):
        script = Path(benchmark.__file__)
        with (
            mock.patch.object(sys, "argv", [str(script)]),
            mock.patch.object(benchmark.requests, "Session", side_effect=RuntimeError("test failure")),
            mock.patch("builtins.print") as print_output,
        ):
            with self.assertRaises(SystemExit) as error:
                runpy.run_path(str(script), run_name="__main__")

        self.assertEqual(error.exception.code, 1)
        self.assertIn("AIG benchmark failed: test failure", print_output.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
