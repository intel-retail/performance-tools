'''
* Copyright (C) 2026 Intel Corporation.
*
* SPDX-License-Identifier: Apache-2.0
'''

import json
import os
import csv
import tempfile
import unittest
from unittest import mock

import consolidate_multiple_run_of_metrics as metrics
from windows_metrics import blank_metrics


class WSLConsolidationTest(unittest.TestCase):
    def consolidate(self, wsl2, windows_metrics=None, records=None):
        with tempfile.TemporaryDirectory() as results_dir:
            if windows_metrics is not None:
                with open(os.path.join(results_dir, "windows_metrics.json"),
                          "w", encoding="utf-8-sig") as metrics_file:
                    json.dump(windows_metrics, metrics_file)
            if records is not None:
                with open(os.path.join(
                        results_dir, "stream_density_latency_samples.jsonl"),
                        "w", encoding="utf-8") as samples_file:
                    samples_file.writelines(
                        json.dumps(record) + "\n" for record in records)
            output = os.path.join(results_dir, "metrics.csv")
            with mock.patch.dict(os.environ, {"WSL2": wsl2}), \
                    mock.patch.object(metrics, "write_metrics_csv",
                                      wraps=metrics.write_metrics_csv) as writer:
                metrics.consolidate_metrics(results_dir, output)
                writer.assert_called_once()
            with open(output, newline="", encoding="utf-8") as csv_file:
                return list(csv.reader(csv_file))

    def test_native_ignores_windows_metrics(self):
        rows = self.consolidate("false", {
            "metrics": {"GPU_1 GPU Power (W)": 12.5}})
        self.assertEqual(rows, [])

    def test_wsl_missing_windows_metrics_uses_na(self):
        rows = dict(self.consolidate("true"))
        self.assertEqual(rows, {
            **blank_metrics(), metrics.AVG_NPU_USAGE_CONSTANT: "NA"})

    def test_wsl_merges_valid_hardware_metrics_only(self):
        rows = dict(self.consolidate("TRUE", {"metrics": {
            "GPU_1 GPU Power (W)": 12.345,
            "GPU_1 Compute[CCS] Utilization %": 0,
            "GPU_1 Render/3D[RCS] Utilization %": -1,
            "GPU_1 Video[VCS] Utilization %": True,
            "GPU_1 VideoEnhance[VECS] Utilization %": float("nan"),
            "GPU_1 Blitter Copy Engine %": "NA",
            "S0 Power Draw W": float("inf"),
            "S1 Memory Bandwidth Usage MB/s": 1234.567,
            "Camera_0 FPS": 999,
        }}))
        self.assertEqual(rows["GPU_1 GPU Power (W)"], "12.35")
        self.assertEqual(rows["GPU_1 Compute[CCS] Utilization %"], "0")
        self.assertEqual(rows["S1 Memory Bandwidth Usage MB/s"], "1234.57")
        self.assertNotIn("Camera_0 FPS", rows)
        for key in (
                "GPU_1 Render/3D[RCS] Utilization %",
                "GPU_1 Video[VCS] Utilization %",
                "GPU_1 VideoEnhance[VECS] Utilization %",
                "GPU_1 Blitter Copy Engine %", "S0 Power Draw W"):
            self.assertEqual(rows[key], "NA")

    def test_wsl_invalid_metrics_object_uses_na(self):
        with mock.patch("builtins.print") as warning:
            rows = dict(self.consolidate("true", {"metrics": []}))
        self.assertEqual(rows["GPU_1 GPU Power (W)"], "NA")
        warning.assert_called_once()

    def test_wsl_skips_unsupported_extractors(self):
        unsupported = (
            metrics.NPUUsageExtractor, metrics.PCMExtractor,
            metrics.MemBandwidthExtractor, metrics.PowerUsageExtractor,
            metrics.QMASSAGPUUsageExtractor, metrics.XPUMUsageExtractor)
        with mock.patch.object(metrics, "KPIExtractor_OPTION", {
                extractor.__name__: extractor for extractor in unsupported}):
            with mock.patch.object(metrics.os, "walk") as walk:
                self.consolidate("true")
                walk.assert_not_called()
                self.consolidate("false")
                self.assertEqual(walk.call_count, len(unsupported))

    def test_wsl_preserves_confirmed_window_and_csv_exclusions(self):
        rows = dict(self.consolidate("true", records=[{
            "container_name": "gst",
            "stream_fps": {"pipeline_stream0": 15.0},
            "samples_by_stream": {"source0 -> sink0": [10, 20]},
        }, {
            "record_type": "sweep_result", "container_name": "gst",
            "status": "pass", "lane_count": 1,
        }]))
        self.assertEqual(rows["Camera_0 FPS"], "15.0")
        self.assertEqual(rows["System Latency p99 (ms)"], "19.9")
        self.assertNotIn("Per-stream Latency Sample Count", rows)
        self.assertEqual(rows["GPU_1 GPU Power (W)"], "NA")

    def test_wsl_preserves_unconfirmed_marker_and_hardware_metrics(self):
        rows = self.consolidate("true", {
            "metrics": {"GPU_1 GPU Power (W)": 12.5}}, records=[{
                "container_name": "gst",
                "stream_fps": {"pipeline_stream0": 999},
                "samples_by_stream": {"source0 -> sink0": [999]},
            }, {
                "record_type": "sweep_result", "container_name": "gst",
                "status": "inconclusive", "lane_count": 1,
            }])
        self.assertEqual(rows[0], ["Result: no confirmed density"])
        self.assertIn(["GPU_1 GPU Power (W)", "12.5"], rows)
        self.assertFalse(any(metrics.is_fps_or_latency_metric(row[0])
                             for row in rows))


class WindowLatencyMetricsTest(unittest.TestCase):
    def test_pipeline_latency_extractor_builds_whole_log_record(self):
        log_lines = [
            "latency_tracer_pipeline, source_name=(string)source0, "
            "sink_name=(string)sink0, frame_num=(uint)1, "
            "frame_latency=(double)10\n",
            # Duplicate frame number: the last value is kept once.
            "latency_tracer_pipeline, source_name=(string)source0, "
            "sink_name=(string)sink0, frame_num=(uint)1, "
            "frame_latency=(double)12\n",
            "latency_tracer_pipeline, source_name=(string)source0, "
            "sink_name=(string)sink0, frame_num=(uint)2, "
            "frame_latency=(double)20\n",
        ]
        with tempfile.TemporaryDirectory() as results_dir:
            log_path = os.path.join(results_dir, "gst-launch_gst0.log")
            with open(log_path, "w", encoding="utf-8") as log_file:
                log_file.writelines(log_lines)

            extractor = metrics.PipelineLatencyExtractor()
            result = extractor.extract_data(log_path)

        self.assertEqual(result, {})
        self.assertEqual(
            extractor.record,
            {"container_name": "gst0",
             "samples_by_stream": {"source0 -> sink0": [12.0, 20.0]}})

        summary = metrics.build_window_latency_metrics([extractor.record])
        stream_key = "gst0 | source0 -> sink0"
        self.assertEqual(
            summary["Per-stream Latency p50 (ms)"][stream_key], 16.0)
        self.assertEqual(summary["System Latency p99 (ms)"], 19.92)

    def test_pipeline_fps_extractor_is_time_weighted_over_whole_log(self):
        with tempfile.TemporaryDirectory() as results_dir:
            log_path = os.path.join(results_dir, "pipeline_stream2_gst0.log")
            with open(log_path, "w", encoding="utf-8") as log_file:
                log_file.write(
                    "fps,duration_seconds\n10,1\n20,3\nbad,row\n15,0\n")

            result = metrics.PIPELINEFPSExtractor().extract_data(log_path)

        # (10*1 + 20*3) / (1 + 3); a plain mean would be 15.
        self.assertEqual(result, {"Camera_2 FPS": 17.5})

    def test_build_window_latency_metrics_merges_stream_samples_across_windows(self):
        records = [
            {
                "container_name": "gst",
                "pipeline_count": 1,
                "window_id": "window-1",
                "samples_by_stream": {
                    "source0 -> sink0": [0, 10],
                    "source1 -> sink1": [100],
                },
            },
            {
                "container_name": "gst",
                "pipeline_count": 2,
                "window_id": "window-2",
                "samples_by_stream": {
                    "source0 -> sink0": [20, 30],
                },
            },
        ]

        result = metrics.build_window_latency_metrics(records)

        stream_key = "gst | source0 -> sink0"
        self.assertEqual(
            set(result["Per-stream Latency Sample Count"]),
            {"gst | source0 -> sink0", "gst | source1 -> sink1"})
        self.assertEqual(
            result["Per-stream Latency p50 (ms)"][stream_key], 15.0)
        self.assertEqual(
            result["Per-stream Latency p95 (ms)"][stream_key], 28.5)
        self.assertEqual(
            result["Per-stream Latency p99 (ms)"][stream_key], 29.7)
        self.assertEqual(
            result["Per-stream Latency Average (ms)"][stream_key], 15.0)
        self.assertNotIn("Per-stream Latency Median (ms)", result)
        self.assertEqual(
            result["Per-stream Latency Sample Count"][stream_key], 4)
        self.assertEqual(result["System Latency p50 (ms)"], 20.0)
        self.assertEqual(result["System Latency p95 (ms)"], 86.0)
        self.assertEqual(result["System Latency p99 (ms)"], 97.2)
        self.assertEqual(result["System Latency Sample Count"], 5)
        self.assertNotIn("Latency Percentile Method", result)

    def test_build_window_fps_metrics_uses_final_window_stream_fps(self):
        records = [
            {
                "container_name": "gst",
                "pipeline_count": 2,
                "stream_fps": {
                    "pipeline_stream0": 14.876,
                    "pipeline_stream1": 15.0,
                },
                "samples_by_stream": {},
            },
        ]

        result = metrics.build_window_fps_metrics(records)

        self.assertEqual(
            result, {"Camera_0 FPS": 14.88, "Camera_1 FPS": 15.0})

    def test_build_window_fps_metrics_labels_multiple_containers(self):
        records = [
            {"container_name": "gst0",
             "stream_fps": {"pipeline_stream0": 15.0}},
            {"container_name": "gst1",
             "stream_fps": {"pipeline_stream0": 14.0}},
        ]

        result = metrics.build_window_fps_metrics(records)

        self.assertEqual(
            result,
            {"Camera_0 FPS (gst0)": 15.0, "Camera_0 FPS (gst1)": 14.0})

    def test_load_window_latency_records_reads_jsonl_recursively(self):
        record = {
            "container_name": "gst",
            "pipeline_count": 2,
            "window_id": "window-1",
            "samples_by_stream": {"source0 -> sink0": [12.5]},
        }
        with tempfile.TemporaryDirectory() as root_directory:
            nested = os.path.join(root_directory, "run")
            os.makedirs(nested)
            sample_path = os.path.join(
                nested, "stream_density_latency_samples.jsonl")
            with open(sample_path, "w", encoding="utf-8") as sample_file:
                sample_file.write(json.dumps(record) + "\n")

            result = metrics.load_window_latency_records(root_directory)

        self.assertEqual(result, [record])

    def test_load_stream_density_records_separates_sweep_results(self):
        samples = {
            "container_name": "gst",
            "pipeline_count": 2,
            "samples_by_stream": {"source0 -> sink0": [12.5]},
        }
        result = {
            "record_type": "sweep_result",
            "container_name": "gst",
            "status": "inconclusive",
            "lane_count": 2,
            "result_id": "2026-10-22T10:00:00+05:30",
        }
        with tempfile.TemporaryDirectory() as root_directory:
            sample_path = os.path.join(
                root_directory, "stream_density_latency_samples.jsonl")
            with open(sample_path, "w", encoding="utf-8") as sample_file:
                sample_file.write(json.dumps(samples) + "\n")
                sample_file.write(json.dumps(result) + "\n")

            window_records, sweep_results = (
                metrics.load_stream_density_records(root_directory))

        self.assertEqual(window_records, [samples])
        self.assertEqual(sweep_results, [result])
        self.assertTrue(metrics.has_unconfirmed_density(sweep_results))
        self.assertFalse(metrics.has_unconfirmed_density([
            {**result, "status": "pass"}]))

    def test_unconfirmed_result_filters_fps_latency_and_marks_csv(self):
        metrics_to_write = metrics.filter_unconfirmed_fps_latency_metrics({
            "Camera_0 FPS": 15.0,
            "System Latency p99 (ms)": 20.0,
            "CPU Utilization %": 55.0,
        })

        with tempfile.TemporaryDirectory() as results_dir:
            output = os.path.join(results_dir, "metrics.csv")
            metrics.write_metrics_csv(
                output, metrics_to_write, no_confirmed_density=True)
            with open(output, newline="", encoding="utf-8") as csv_file:
                rows = list(csv.reader(csv_file))

        self.assertEqual(rows[0], ["Result: no confirmed density"])
        self.assertEqual(rows[1], ["CPU Utilization %", "55.0"])
        self.assertFalse(any("FPS" in ",".join(row) for row in rows))
        self.assertFalse(any("Latency" in ",".join(row) for row in rows))

    def test_consolidation_omits_whole_log_fps_and_latency_when_unconfirmed(self):
        with tempfile.TemporaryDirectory() as results_dir:
            with open(os.path.join(results_dir, "pipeline_stream0_gst.log"),
                      "w", encoding="utf-8") as log_file:
                log_file.write("fps,duration_seconds\n999,1\n")
            with open(os.path.join(results_dir, "gst-launch_gst.log"),
                      "w", encoding="utf-8") as log_file:
                log_file.write(
                    "latency_tracer_pipeline, source_name=(string)source0, "
                    "sink_name=(string)sink0, frame_num=(uint)1, "
                    "frame_latency=(double)999\n")
            with open(os.path.join(
                    results_dir, "stream_density_latency_samples.jsonl"),
                    "w", encoding="utf-8") as sample_file:
                sample_file.write(json.dumps({
                    "record_type": "sweep_result",
                    "container_name": "gst",
                    "status": "inconclusive",
                    "lane_count": 1,
                }) + "\n")

            output = os.path.join(results_dir, "metrics.csv")
            metrics.consolidate_metrics(results_dir, output)

            with open(output, newline="", encoding="utf-8") as csv_file:
                rows = list(csv.reader(csv_file))

        self.assertEqual(rows, [["Result: no confirmed density"]])

    def test_consolidation_uses_confirmed_window_metrics_when_passed(self):
        with tempfile.TemporaryDirectory() as results_dir:
            with open(os.path.join(results_dir, "pipeline_stream0_gst.log"),
                      "w", encoding="utf-8") as log_file:
                log_file.write("fps,duration_seconds\n999,1\n")
            with open(os.path.join(results_dir, "gst-launch_gst.log"),
                      "w", encoding="utf-8") as log_file:
                log_file.write(
                    "latency_tracer_pipeline, source_name=(string)source0, "
                    "sink_name=(string)sink0, frame_num=(uint)1, "
                    "frame_latency=(double)999\n")
            records = [
                {
                    "container_name": "gst",
                    "pipeline_count": 1,
                    "stream_fps": {"pipeline_stream0": 15.0},
                    "samples_by_stream": {"source0 -> sink0": [10.0, 20.0]},
                },
                {
                    "record_type": "sweep_result",
                    "container_name": "gst",
                    "status": "pass",
                    "lane_count": 1,
                },
            ]
            with open(os.path.join(
                    results_dir, "stream_density_latency_samples.jsonl"),
                    "w", encoding="utf-8") as sample_file:
                sample_file.writelines(
                    json.dumps(record) + "\n" for record in records)

            output = os.path.join(results_dir, "metrics.csv")
            metrics.consolidate_metrics(results_dir, output)

            with open(output, newline="", encoding="utf-8") as csv_file:
                rows = list(csv.reader(csv_file))

        self.assertIn(["Camera_0 FPS", "15.0"], rows)
        self.assertIn(["System Latency p99 (ms)", "19.9"], rows)
        self.assertNotIn(["Camera_0 FPS", "999.0"], rows)
        self.assertNotIn(["System Latency p99 (ms)", "999.0"], rows)


if __name__ == "__main__":
    unittest.main()
