"""Unit tests for Docker service resource sampling."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import benchmark_signage_docker_stats as stats


class TestDockerStatsHelpers(unittest.TestCase):

    def test_average_and_percent_parsing(self):
        self.assertEqual(stats.average([1, 2.25, None]), 1.625)
        self.assertIsNone(stats.average([None, "2"]))
        self.assertEqual(stats.parse_percent(" 12.5% "), 12.5)
        self.assertIsNone(stats.parse_percent(None))
        self.assertIsNone(stats.parse_percent("unknown"))

    def test_memory_parsing_supports_binary_and_decimal_units(self):
        self.assertEqual(stats.parse_memory_mib("1.5GiB"), 1536)
        self.assertEqual(stats.parse_memory_mib("1000 KB"), 0.954)
        self.assertIsNone(stats.parse_memory_mib("unknown"))

    @mock.patch.object(stats.subprocess, "check_output")
    def test_running_services_ignores_bad_json_and_stopped_containers(self, check_output):
        check_output.return_value = "\n".join([
            "{bad json",
            json.dumps({"State": "running", "ID": "abc", "Service": "web"}),
            json.dumps({"State": "exited", "ID": "def", "Service": "db"}),
            json.dumps({"State": "running", "Service": "missing-id"}),
        ])

        result = stats.running_services()

        self.assertEqual(result, [{"State": "running", "ID": "abc", "Service": "web"}])
        check_output.assert_called_once()

    @mock.patch.object(stats, "running_services", return_value=[])
    def test_sample_services_returns_empty_without_containers(self, _running):
        self.assertEqual(stats.sample_services(), [])

    @mock.patch.object(stats.subprocess, "check_output")
    @mock.patch.object(stats, "running_services")
    def test_sample_services_matches_ids_and_names_and_skips_unknown(self, running, check_output):
        running.return_value = [
            {"ID": "123456789012345", "Name": "one", "Service": "api"},
            {"ID": "abcdefghijklmnop", "Name": "two", "Service": "worker"},
        ]
        check_output.return_value = "\n".join([
            json.dumps({"ID": "123456789012", "Name": "renamed", "CPUPerc": "2.5%",
                        "MemPerc": "10%", "MemUsage": "100MiB / 1GiB"}),
            json.dumps({"Container": "unknown-id", "Name": "two", "CPUPerc": "3%",
                        "MemPerc": "20%", "MemUsage": "1GB / 2GB"}),
            "{bad json",
            json.dumps({"ID": "not-known", "Name": "stranger"}),
        ])

        samples = stats.sample_services()

        self.assertEqual([sample["service"] for sample in samples], ["api", "worker"])
        self.assertEqual(samples[0]["memory_usage_mib"], 100)
        self.assertEqual(samples[0]["memory_limit_mib"], 1024)
        self.assertEqual(samples[1]["memory_usage_mib"], 953.674)

    def test_host_metrics_handles_missing_invalid_and_valid_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            absent = stats.host_metric_averages(root / "missing.json")
            invalid_path = root / "invalid.json"
            invalid_path.write_text("{")
            invalid = stats.host_metric_averages(invalid_path)
            valid_path = root / "metrics.json"
            valid_path.write_text(json.dumps({
                "cpu_utilization": [[1, 10], [2, 20]],
                "gpu_utilization": [[1, 30]],
                "memory": [[0, 0, 0, 0, 50], [1, 0, 0, 0, 70], [2]],
            }))
            valid = stats.host_metric_averages(valid_path)

        self.assertIsNone(absent["cpu_utilization_percent"])
        self.assertIsNone(invalid["gpu_utilization_percent"])
        self.assertEqual(valid, {
            "cpu_utilization_percent": 15,
            "gpu_utilization_percent": 30,
            "memory_utilization_percent": 60,
        })


class TestDockerStatsOutputAndRun(unittest.TestCase):

    def test_writes_csv_and_json_and_returns_sample_presence(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / "nested" / "capture"
            sample = {
                "timestamp": "now", "service": "api", "container": "api-1",
                "cpu_percent": 10, "memory_percent": 20,
                "memory_usage_mib": 100, "memory_limit_mib": 512,
            }
            with (
                mock.patch.object(stats, "host_metric_averages", return_value={"cpu": 5}),
                mock.patch("builtins.print"),
            ):
                has_samples = stats.write_results(prefix, [sample], 2, None)

            self.assertTrue(has_samples)
            payload = json.loads(Path(f"{prefix}.json").read_text())
            self.assertEqual(payload["services"]["api"]["samples"], 1)
            self.assertEqual(payload["services"]["api"]["average_cpu_percent"], 10)
            self.assertEqual(payload["host_metric_averages"], {"cpu": 5})
            self.assertTrue(Path(f"{prefix}.csv").exists())
            with mock.patch("builtins.print"):
                self.assertFalse(stats.write_results(prefix, [], 2, None))

    def test_run_samples_writes_ready_file_and_removes_it(self):
        with tempfile.TemporaryDirectory() as directory:
            ready = Path(directory) / "ready"
            stop = mock.Mock()
            stop.is_set.side_effect = [False, True]
            sample = [{"service": "api", "container": "api-1", "cpu_percent": 1}]
            with (
                mock.patch.object(stats, "Event", return_value=stop),
                mock.patch.object(stats.signal, "signal"),
                mock.patch.object(stats, "sample_services", return_value=sample) as sample_services,
                mock.patch.object(stats, "write_results", return_value=True) as write_results,
            ):
                result = stats.run(Path(directory) / "out", 0.1, 0, None, ready)

            self.assertTrue(result)
            self.assertEqual(sample_services.call_count, 2)
            stop.wait.assert_called_once()
            write_results.assert_called_once()
            self.assertFalse(ready.exists())

    def test_run_logs_sampling_errors_and_cleans_ready_file(self):
        with tempfile.TemporaryDirectory() as directory:
            ready = Path(directory) / "ready"
            stop = mock.Mock()
            stop.is_set.side_effect = [False, True]
            with (
                mock.patch.object(stats, "Event", return_value=stop),
                mock.patch.object(stats.signal, "signal"),
                mock.patch.object(stats, "sample_services",
                                  side_effect=[OSError("docker unavailable"), []]),
                mock.patch.object(stats, "write_results", return_value=False) as write_results,
                mock.patch("builtins.print") as output,
            ):
                result = stats.run(Path(directory) / "out", 0, 0, None, ready)

            self.assertFalse(result)
            write_results.assert_called_once()
            self.assertEqual(output.call_count, 1)
            self.assertFalse(ready.exists())

    def test_main_rejects_invalid_interval(self):
        with mock.patch.object(stats.sys, "argv", ["docker_stats", "--output-prefix", "x", "--interval", "0"]):
            with self.assertRaises(SystemExit) as error:
                stats.main()
        self.assertEqual(error.exception.code, 2)

    def test_main_reports_no_samples(self):
        with (
            mock.patch.object(stats.sys, "argv", ["docker_stats", "--output-prefix", "x"]),
            mock.patch.object(stats, "run", return_value=False),
            mock.patch("builtins.print") as output,
        ):
            self.assertEqual(stats.main(), 1)
        self.assertEqual(output.call_args.kwargs["file"], stats.sys.stderr)


if __name__ == "__main__":
    unittest.main()
