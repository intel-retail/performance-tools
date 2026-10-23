#
# Apache v2 license
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""Unit tests for PID pipeline benchmark parsing and capture."""

import json
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import benchmark_signage_pid as pid


class TestPidParsers(unittest.TestCase):

    def test_parse_size_binary_decimal_and_invalid_values(self):
        self.assertEqual(pid.parse_size("1.5GiB"), 1536)
        self.assertEqual(pid.parse_size("100 MB"), 95.37)
        self.assertIsNone(pid.parse_size("n/a"))

    def test_parse_frame_extracts_metadata_and_validates_arrival_delay(self):
        payload = {"metadata": {
            "frame_id": 4, "time": 1_000_000_000,
            "pipeline": {"status": {"frame_fps": 25, "avg_pipeline_latency": 8}},
            "gva_meta": [{}, {}],
        }}
        frame = pid.parse_frame(json.dumps(payload), 1_005_000_000)
        self.assertEqual(frame["frame_id"], 4)
        self.assertEqual(frame["detections"], 2)
        self.assertEqual(frame["mqtt_arrival_delay_ms"], 5)
        self.assertEqual(frame["frame_fps"], 25)
        self.assertIsNone(pid.parse_frame(json.dumps(payload), 0)["mqtt_arrival_delay_ms"])
        payload["metadata"]["time"] = True
        self.assertIsNone(pid.parse_frame(json.dumps(payload), 1_005_000_000)["mqtt_arrival_delay_ms"])

    def test_gpu_compute_usage_handles_missing_path_and_samples(self):
        self.assertIsNone(pid.gpu_compute_usage(None))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gpu.json"
            path.write_text(json.dumps([{"CCS %": "10"}, {}, {"CCS %": 30}]))
            self.assertEqual(pid.gpu_compute_usage(path), 20)

    def test_parse_frame_propagates_invalid_json(self):
        with self.assertRaises(json.JSONDecodeError):
            pid.parse_frame("{")


class TestPidRun(unittest.TestCase):

    def test_run_fails_when_required_container_is_stopped(self):
        with mock.patch.object(pid, "docker", side_effect=["false"]):
            with self.assertRaisesRegex(RuntimeError, "dlstreamer-pipeline-server is not running"):
                pid.run(1, Path(tempfile.gettempdir()), False)

    def test_run_captures_frame_and_writes_summary_and_frame_files(self):
        with tempfile.TemporaryDirectory() as directory:
            stdout = mock.Mock()
            stdout.fileno.return_value = 4
            subscriber = mock.Mock(stdout=stdout, stderr=mock.Mock())
            subscriber.poll.return_value = None
            subscriber.stderr.read.return_value = b""
            frame = {"metadata": {
                "frame_id": 0, "time": 1,
                "pipeline": {"status": {"frame_fps": 20, "avg_pipeline_latency": 5}},
                "gva_meta": [{}, {}],
            }}
            sample_line = json.dumps(frame).encode() + b"\n"

            def docker_output(*args):
                if args[0] == "inspect":
                    return "true"
                if args[0] == "stats":
                    return json.dumps({"CPUPerc": "12.5%", "MemUsage": "100MiB / 1GiB"})
                if args[0] == "logs":
                    return "FpsCounter(last 1s): total=24 fps"
                raise AssertionError(args)

            with (
                mock.patch.object(pid, "docker", side_effect=docker_output),
                mock.patch.object(pid.subprocess, "Popen", return_value=subscriber),
                mock.patch.object(pid.select, "select", return_value=([stdout], [], [])),
                mock.patch.object(pid.os, "read", return_value=sample_line),
                mock.patch.object(pid.time, "monotonic", side_effect=[0, 0, 0, 0, 2]),
                mock.patch("builtins.print"),
            ):
                pid.run(1, Path(directory), False, device="CPU")

            summary_files = list(Path(directory).glob("pid_benchmark_*.json"))
            frame_files = list(Path(directory).glob("pid_benchmark_*_frames.json"))
            self.assertEqual(len(summary_files), 2)
            self.assertEqual(len(frame_files), 1)
            summaries = [path for path in summary_files if not path.name.endswith("_frames.json")]
            summary = json.loads(summaries[0].read_text())
            self.assertEqual(summary["frames"], 1)
            self.assertEqual(summary["detections_total"], 2)
            self.assertEqual(summary["cpu_percent"], 12.5)
            self.assertEqual(summary["fps_dlstreamer"], 24)
            self.assertEqual(json.loads(frame_files[0].read_text())[0]["frame_id"], 0)
            subscriber.terminate.assert_called_once()
            subscriber.communicate.assert_called_once_with(timeout=3)

    def test_run_raises_if_no_frames_are_received_and_terminates_subscriber(self):
        stdout = mock.Mock()
        subscriber = mock.Mock(stdout=stdout, stderr=mock.Mock())
        with (
            mock.patch.object(pid, "docker", return_value="true"),
            mock.patch.object(pid.subprocess, "Popen", return_value=subscriber),
            mock.patch.object(pid.time, "monotonic", side_effect=[0, 2]),
        ):
            with self.assertRaisesRegex(RuntimeError, "No PID frames received"):
                pid.run(1, Path(tempfile.gettempdir()), False)
        subscriber.terminate.assert_called_once()

    def test_cli_rejects_non_positive_duration(self):
        with (
            mock.patch.object(sys, "argv", [pid.__file__, "--duration", "0"]),
            mock.patch("builtins.print"),
        ):
            with self.assertRaises(SystemExit) as error:
                runpy.run_path(pid.__file__, run_name="__main__")
        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
