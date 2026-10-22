"""Unit tests for Qdrant and ASe benchmark helpers."""

import json
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import benchmark_signage_qdrant as qdrant_benchmark


class TestQdrantHelpers(unittest.TestCase):

    @mock.patch.object(qdrant_benchmark.subprocess, "check_output")
    def test_memory_mib_parses_memory_and_rejects_unknown_units(self, check_output):
        check_output.return_value = json.dumps({"MemUsage": "1.5GiB / 2GiB"})
        self.assertEqual(qdrant_benchmark.memory_mib(), 1536)
        check_output.return_value = json.dumps({"MemUsage": "unknown / 2GiB"})
        with self.assertRaisesRegex(ValueError, "Unrecognized Qdrant memory usage"):
            qdrant_benchmark.memory_mib()

    def test_api_returns_result_and_raises_for_http_or_qdrant_errors(self):
        session = mock.Mock()
        response = mock.Mock()
        response.json.return_value = {"status": "ok", "result": {"points_count": 3}}
        session.request.return_value = response
        self.assertEqual(qdrant_benchmark.api(session, "GET", "url"), {"points_count": 3})
        session.request.assert_called_once_with("GET", "url", timeout=120)

        response.json.return_value = {"status": "error"}
        with self.assertRaisesRegex(RuntimeError, "Qdrant error"):
            qdrant_benchmark.api(session, "GET", "url")
        response.raise_for_status.side_effect = ValueError("http error")
        with self.assertRaisesRegex(ValueError, "http error"):
            qdrant_benchmark.api(session, "GET", "url")

    def test_summarize_flattens_search_sizes_and_ad_index(self):
        summary = qdrant_benchmark.summarize([
            {"kind": "vector_search", "points": 100, "queries": 4, "avg_query_ms": 2,
             "p95_query_ms": 3, "queries_per_sec": 5, "upsert_ms": 6,
             "qdrant_memory_delta_mib": 7},
            {"kind": "predefined_ad_index", "index_http_ms": 8, "image_kb": 9,
             "qdrant_memory_delta_mib": 10},
        ])
        self.assertEqual(summary["sizes"], "100")
        self.assertEqual(summary["queries_per_size"], 4)
        self.assertEqual(summary["search_100_avg_query_ms"], 2)
        self.assertEqual(summary["predefined_ad_index_http_ms"], 8)
        self.assertEqual(qdrant_benchmark.summarize([])["sizes"], "")


class TestQdrantBenchmark(unittest.TestCase):

    def test_benchmark_runs_search_and_ad_index_workflow_and_writes_results(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "sample.jpg"
            image.write_bytes(b"fake-image")
            args = SimpleNamespace(
                collection="live-collection", sizes=[1], batch_size=2, queries=1,
                ase_url="https://ase.example.test", insecure=False,
                output=Path(directory) / "results",
            )
            qdrant_session = mock.Mock()

            def api_result(_method, url, **_kwargs):
                if url.endswith("/collections/live-collection"):
                    result = {"config": {"params": {"vectors": {"size": 2, "distance": "Cosine"}}},
                              "points_count": 1}
                elif url.endswith("/points/query"):
                    result = {}
                else:
                    result = {"points_count": 1, "indexed_vectors_count": 1}
                response = mock.Mock()
                response.raise_for_status.return_value = None
                response.json.return_value = {"status": "ok", "result": result}
                return response

            qdrant_session.request.side_effect = api_result
            qdrant_session.get.return_value = mock.Mock(status_code=404)
            ase_session = mock.Mock()
            ase_response = mock.Mock()
            ase_response.json.return_value = {"message": "Success"}
            ase_session.post.return_value = ase_response

            with (
                mock.patch.object(qdrant_benchmark.subprocess, "check_output", return_value="172.18.0.2"),
                mock.patch.object(qdrant_benchmark.requests, "Session",
                                  side_effect=[qdrant_session, ase_session]),
                mock.patch.object(qdrant_benchmark, "memory_mib", side_effect=[100, 110, 120, 130]),
                mock.patch.object(qdrant_benchmark.time, "perf_counter",
                                  side_effect=[0, 1, 2, 2.01, 2.02, 3, 4, 4.01]),
                mock.patch.object(qdrant_benchmark, "IMAGE_PATH", image),
                mock.patch("builtins.print"),
            ):
                qdrant_benchmark.benchmark(args)

            self.assertFalse(qdrant_session.trust_env)
            self.assertTrue(ase_session.verify)
            self.assertEqual(ase_session.post.call_args.args[0], "https://ase.example.test/ase/predef/")
            qdrant_session.request.assert_any_call(
                "PUT", mock.ANY, timeout=120,
                json={"vectors": {"size": 2, "distance": "Cosine"}},
            )
            json_files = [
                path for path in args.output.glob("qdrant_benchmark_*.json")
                if not path.name.endswith("_summary.json")
            ]
            summary_files = list(args.output.glob("qdrant_benchmark_*_summary.json"))
            self.assertEqual(len(json_files), 1)
            self.assertEqual(len(summary_files), 1)
            rows = json.loads(json_files[0].read_text())
            self.assertEqual([row["kind"] for row in rows],
                             ["vector_search", "predefined_ad_index"])
            self.assertEqual(rows[0]["points"], 1)
            self.assertEqual(rows[1]["image_kb"], round(len(b"fake-image") / 1024, 3))

    def test_benchmark_rejects_missing_container_address(self):
        args = SimpleNamespace()
        with mock.patch.object(qdrant_benchmark.subprocess, "check_output", return_value=" "):
            with self.assertRaisesRegex(RuntimeError, "ase-qdrant is not running"):
                qdrant_benchmark.benchmark(args)

    def test_cli_rejects_unsorted_collection_sizes(self):
        with (
            mock.patch.object(sys, "argv", [qdrant_benchmark.__file__, "--sizes", "2", "1"]),
            mock.patch("builtins.print"),
        ):
            with self.assertRaises(SystemExit) as error:
                runpy.run_path(qdrant_benchmark.__file__, run_name="__main__")
        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
