import json
import re
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import compare_stream_density as report


class ComparisonTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.camera = {"_about": {"name": "Checkout"}, "lane_config": {
            "cameras": [{"workloads": ["detect"], "targetFps": 30}]}}

    def make_run(self, name, accelerator="CPU", day=1, config="checkout"):
        path = self.root / name
        path.mkdir()
        metadata = {
            "config_id": config, "accelerator": accelerator,
            "started_at": f"2026-10-{day:02d}T10:00:00+00:00",
            "finished_at": f"2026-10-{day:02d}T10:01:00+00:00",
            "duration_seconds": 60, "camera_config": self.camera,
            "pipeline_config": {"workload_pipeline_map": {"detect": [
                {"type": "gvadetect", "model": "yolo", "precision": "INT8", "device": accelerator}]}},
            "hardware": {"cpu": "Test CPU", "accelerators": ["Test GPU"]},
            "os": "Test Linux", "kernel": "6.8", "settings": {},
        }
        (path / report.METADATA_FILE).write_text(json.dumps(metadata))
        records = [
            {"container_name": "gst", "pipeline_count": 2,
             "stream_fps": {"stream0": 29, "stream1": 31},
             "hardware_metrics": {"S0 Power Draw W": 20, "CPU Utilization %": 50},
             "samples_by_stream": {"stream0": [10, 20], "stream1": [30]}},
            {"record_type": "sweep_result", "container_name": "gst", "status": "pass", "lane_count": 2},
        ]
        (path / report.SAMPLES_FILE).write_text("\n".join(json.dumps(record) for record in records) + "\n")
        (path / "metrics.csv").write_text("S0 Power Draw W,20\nCPU Utilization %,50\nGPU Utilization %,80\n")
        return path, metadata

    def test_latest_per_accelerator_and_default_config(self):
        old, _ = self.make_run("old", day=1)
        new, _ = self.make_run("new", day=3)
        gpu, _ = self.make_run("gpu", "GPU", day=2)
        other, _ = self.make_run("other", "NPU", day=2, config="other")
        selected = report.select_runs([old, new, gpu, other])
        self.assertEqual([path for path, _ in selected], [new, gpu])

    def test_dates_and_config_filter(self):
        old, _ = self.make_run("old", day=1)
        new, _ = self.make_run("new", day=3)
        selected = report.select_runs([old, new], "checkout", until=report.parse_date("2026-10-02"))
        self.assertEqual(selected[0][0], old)

    def test_metrics_and_provenance(self):
        path, metadata = self.make_run("cpu")
        metrics = report.run_metrics(path, metadata)
        self.assertEqual(metrics["streams"], 2)
        self.assertEqual(metrics["throughput"], 60)
        self.assertEqual(metrics["rate"], 30)
        self.assertEqual(metrics["latency_p50"], 20)
        self.assertEqual(metrics["power"], {"S0 Power Draw W": 10})
        self.assertEqual(metrics["utilisation"], {"CPU Utilization %": 50})
        text = report.generate_report([(path, metadata)])
        for expected in ["Checkout", "Test CPU", "Test Linux", "6.8", "INT8", str(path)]:
            self.assertIn(expected, text)

    def test_configuration_identity_ignores_devices_not_precision(self):
        _, cpu = self.make_run("cpu")
        _, gpu = self.make_run("gpu", "GPU")
        identity = report.configuration_id(self.camera, cpu["pipeline_config"], {})
        self.assertEqual(identity, report.configuration_id(self.camera, gpu["pipeline_config"], {}))
        gpu["pipeline_config"]["workload_pipeline_map"]["detect"][0]["precision"] = "FP16"
        self.assertNotEqual(identity, report.configuration_id(self.camera, gpu["pipeline_config"], {}))

    def test_unconfirmed_does_not_report_density(self):
        path, metadata = self.make_run("failed")
        sample_path = path / report.SAMPLES_FILE
        sample_path.write_text(sample_path.read_text().replace('"pass"', '"inconclusive"'))
        metrics = report.run_metrics(path, metadata)
        self.assertIsNone(metrics["streams"])
        self.assertIsNone(metrics["latency_p50"])

    def test_csv_percentiles_are_not_replaced_by_mean_latency(self):
        path, metadata = self.make_run("percentiles")
        with (path / "metrics.csv").open("a") as source:
            source.write(
                'System Latency p50 (ms),11\n'
                'System Latency p95 (ms),91\n'
                'System Latency p99 (ms),99\n'
                'Per-stream Latency p95 (ms),"gst | source0 -> sink0: 80, gst | source1 -> sink1: 110"\n')
        metrics = report.run_metrics(path, metadata)
        self.assertEqual(metrics["latency_p50"], 11)
        self.assertEqual(metrics["latency_p95"], 91)
        self.assertEqual(metrics["latency_p99"], 99)
        self.assertEqual(metrics["highest_stream_p95"], 110)
        markdown = report.generate_report([(path, metadata)])
        self.assertIn("Pooled latency p95 (ms)", markdown)
        self.assertNotIn("Mean latency", markdown)
        self.assertNotIn("confirming window", markdown)

    def test_inconclusive_retains_last_passing_lanes_but_not_fps_or_latency(self):
        path, metadata = self.make_run("inconclusive")
        sample_path = path / report.SAMPLES_FILE
        sample_path.write_text(sample_path.read_text().replace('"pass"', '"inconclusive"'))
        metrics = report.run_metrics(path, metadata)
        self.assertEqual(metrics["status"], "inconclusive")
        self.assertEqual(metrics["last_passing_lanes"], {"gst": 2})
        for key in ("streams", "throughput", "rate", "latency_p50",
                    "latency_p95", "latency_p99", "highest_stream_p95"):
            self.assertIsNone(metrics[key])
        self.assertIn("not confirmed", report.generate_report([(path, metadata)]))
        sample_path.write_text(sample_path.read_text().replace('"inconclusive"', '"fail"'))
        failed = report.run_metrics(path, metadata)
        self.assertEqual(failed["status"], "fail")
        self.assertEqual(failed["last_passing_lanes"], {})

    def test_history_is_persisted_without_contaminating_final_metrics(self):
        import stream_density
        from consolidate_multiple_run_of_metrics import load_stream_density_records

        path, metadata = self.make_run("history")
        for interval, status, fps in ((1, "pass", 999), (2, "fail", 3)):
            stream_density.save_measurement_interval(
                path, "gst", 2, interval, status,
                {"pipeline_stream0": fps}, 100)
        stream_density.save_measurement_interval(path, "gst", 3, 3, "inconclusive", {}, 200)
        records = [json.loads(line) for line in (path / report.SAMPLES_FILE).read_text().splitlines()]
        history = [record for record in records if record.get("record_type") == "measurement_interval"]
        self.assertEqual([record["interval_id"] for record in history], [1, 2, 3])
        self.assertEqual([record["status"] for record in history], ["pass", "fail", "inconclusive"])
        samples, sweeps = load_stream_density_records(path)
        self.assertEqual(len(samples), 1)
        self.assertEqual(len(sweeps), 1)
        self.assertEqual(report.run_metrics(path, metadata)["throughput"], 60)

    def test_capture_records_hardware_and_resolved_acceptance_defaults(self):
        import psutil

        path, template = self.make_run("metadata")
        configs = self.root / "configs"
        configs.mkdir()
        (configs / "camera.json").write_text(json.dumps(self.camera))
        (configs / "pipeline.json").write_text(json.dumps(template["pipeline_config"]))
        core = SimpleNamespace(available_devices=["GPU.0", "NPU"],
                               get_property=lambda device, key: f"Test {device}")
        with patch.object(report.platform, "node", return_value="lab-host"), \
                patch.object(report.subprocess, "run", side_effect=FileNotFoundError), \
                patch.object(report, "gpu_metric_devices", return_value={}), \
                patch.object(report.platform, "machine", return_value="x86_64"), \
                patch.object(report, "read_npu_driver", return_value="intel_vpu v1.2.3"), \
                patch.object(report, "read_memory_details", return_value={
                    "mem_channels": "2-channel", "mem_per_dimm": "32 GB", "mem_speed": "5600 MT/s"}), \
                patch.object(report, "read_power_limits", return_value={
                    "constraint_0_power_limit_uw": 30000000,
                    "constraint_1_power_limit_uw": 64000000}), \
                patch.dict("sys.modules", {"openvino": SimpleNamespace(Core=lambda: core)}), \
                patch.object(psutil, "cpu_count", side_effect=lambda logical=True: 16 if logical else 8), \
                patch.object(psutil, "virtual_memory", return_value=SimpleNamespace(total=32 * 1024 ** 3)):
            metadata = report.capture_run(path, self.root, {
                "CAMERA_STREAM": "camera.json", "WORKLOAD_DIST": "pipeline.json"}, {})
        self.assertEqual(metadata["hardware"]["hostname"], "lab-host")
        self.assertEqual(metadata["camera_config_file"], "camera.json")
        self.assertEqual(metadata["hardware"]["cpu_cores"], 8)
        self.assertEqual(metadata["hardware"]["logical_cpus"], 16)
        self.assertEqual(metadata["hardware"]["ram_bytes"], 32 * 1024 ** 3)
        self.assertEqual(metadata["hardware"]["arch"], "x86_64")
        self.assertEqual(metadata["hardware"]["npu_driver"], "intel_vpu v1.2.3")
        self.assertEqual(metadata["hardware"]["mem_speed"], "5600 MT/s")
        self.assertEqual(metadata["hardware"]["mem_per_dimm"], "32 GB")
        self.assertEqual(metadata["hardware"]["mem_channels"], "2-channel")
        saved = json.loads((path / report.METADATA_FILE).read_text())
        self.assertEqual(saved["hardware"], metadata["hardware"])
        self.assertIn("GPU.0: Test GPU.0", metadata["hardware"]["gpu_names"])
        self.assertIn("NPU: Test NPU", metadata["hardware"]["npu_names"])
        self.assertEqual(metadata["hardware"]["constraint_0_power_limit_uw"], 30000000)
        self.assertEqual(metadata["hardware"]["constraint_1_power_limit_uw"], 64000000)
        self.assertEqual(metadata["settings"]["pass_tolerance_ratio"], 0.95)
        self.assertEqual(metadata["settings"]["consecutive_passes"], 2)
        self.assertEqual(metadata["settings"]["consecutive_fails"], 2)
        self.assertEqual(report.test_configuration(metadata)["Throughput threshold (FPS)"], "28.50")

    def test_memory_dmi_ignores_empty_slots_and_counts_channels_not_dimms(self):
        blocks = []
        for locator, size in (("ChannelA-DIMM0", "32 GB"), ("ChannelA-DIMM1", "16 GB"),
                              ("ChannelB-DIMM0", "32 GB"), ("ChannelB-DIMM1", "No Module Installed")):
            blocks.append(f'Memory Device\n\tSize: {size}\n\tLocator: {locator}\n'
                          '\tConfigured Memory Speed: 5600 MT/s\n')
        with patch.object(report.subprocess, "run", return_value=SimpleNamespace(
                returncode=0, stdout="\n".join(blocks))) as launch:
            result = report.read_memory_details(self.root / "missing")
        self.assertEqual(result["mem_channels"], "2-channel")
        self.assertEqual(len(result["memory_devices"]), 3)
        self.assertEqual(result["mem_per_dimm"], "16 GB; 32 GB")
        self.assertEqual(result["mem_speed"], "5600 MT/s")
        self.assertEqual(launch.call_args.args[0], ["dmidecode", "--type", "17"])

    def test_memory_edac_fallback_and_unavailable_sources(self):
        dimm = self.root / "edac" / "mc0" / "dimm0"
        dimm.mkdir(parents=True)
        (dimm / "size").write_text("8192")
        (dimm / "dimm_location").write_text("channel 0 slot 0")
        with patch.object(report.subprocess, "run", side_effect=FileNotFoundError):
            result = report.read_memory_details(self.root / "edac")
            absent = report.read_memory_details(self.root / "missing")
        self.assertEqual(result["mem_channels"], "1-channel (EDAC logical)")
        self.assertEqual(result["mem_per_dimm"], "8 GB (EDAC device)")
        self.assertIsNone(result["mem_speed"])
        self.assertEqual(absent["memory_devices"], [])
        self.assertIsNone(absent["mem_channels"])
        with patch.object(report.subprocess, "run", side_effect=report.subprocess.TimeoutExpired("dmidecode", 10)):
            self.assertIsNone(report.read_memory_details(self.root / "missing")["mem_speed"])

    def test_npu_driver_version_and_builtin_module_fallback(self):
        module = self.root / "intel_vpu"
        self.assertIsNone(report.read_npu_driver(module))
        module.mkdir()
        with patch.object(report.platform, "release", return_value="6.8-test"):
            self.assertEqual(report.read_npu_driver(module),
                             "intel_vpu (kernel 6.8-test; module version unavailable)")
        (module / "version").write_text("1.0.0 6.8-test\n")
        self.assertEqual(report.read_npu_driver(module), "intel_vpu v1.0.0")

    def test_power_limits_are_read_and_displayed_in_watts(self):
        path, metadata = self.make_run("power-limits")
        (path / "constraint_0_power_limit_uw").write_text("30000000\n")
        (path / "constraint_1_power_limit_uw").write_text("64000000\n")
        self.assertEqual(report.read_power_limits(path), {
            "constraint_0_power_limit_uw": 30000000,
            "constraint_1_power_limit_uw": 64000000,
        })
        metadata["hardware"].update(report.read_power_limits(path))
        details = report.hardware_details(metadata)
        self.assertEqual(details["PL1 power limit (W)"], 30)
        self.assertEqual(details["PL2 power limit (W)"], 64)

    def test_test_configuration_preserves_per_camera_targets_and_unknown_history(self):
        _, metadata = self.make_run("targets")
        metadata["camera_config"]["lane_config"]["cameras"] = [
            {"camera_id": "cam1", "fps": 15},
            {"camera_id": "cam2", "fps": 15, "targetFps": 30}]
        self.assertIsNone(report.test_configuration(metadata)["Pass tolerance ratio"])
        metadata["settings"].update(pass_tolerance_ratio=0.9, consecutive_passes=3)
        settings = report.test_configuration(metadata)
        self.assertEqual(settings["Target FPS per camera"], "cam1: 15; cam2: 30")
        self.assertEqual(settings["Throughput threshold (FPS)"], "cam1: 13.50; cam2: 27.00")
        self.assertEqual(settings["Run acceptance criterion"], "3 consecutive passes")

    def test_effective_targets_follow_camera_fallback_and_override_precedence(self):
        _, metadata = self.make_run("override-targets")
        metadata["camera_config"]["lane_config"]["cameras"] = [
            {"camera_id": "cam1", "targetFps": 0, "fps": 15},
            {"camera_id": "cam2", "targetFps": "bad", "fps": 20},
            {"camera_id": "cam3", "targetFps": None, "fps": 25}]
        targets = report.camera_targets(metadata["camera_config"])
        self.assertEqual(targets, {"cam1": 15, "cam2": 20, "cam3": 25})
        metadata["settings"] = {
            "pass_tolerance_ratio": 0.9, "environment": {"TARGET_FPS": "10"}}
        display = report.test_configuration(metadata)
        self.assertEqual(display["Effective target FPS per camera"], "10.00")
        self.assertEqual(display["Throughput threshold (FPS)"], "9.00")
        metadata["settings"]["target_fps"] = [30]
        self.assertEqual(report.test_configuration(metadata)["Throughput threshold (FPS)"], "27.00")
        metadata["settings"].update(target_fps=[30, 40], container_names=["gst0", "gst1"])
        resolved = report.effective_targets(targets, metadata["settings"], {"TARGET_FPS": "10"})
        self.assertEqual(resolved["gst0 / cam1"], 30)
        self.assertEqual(resolved["gst1 / cam3"], 40)
        metadata["settings"] = {"pass_tolerance_ratio": 0.9, "environment": {"TARGET_FPS": "-1"}}
        self.assertEqual(report.test_configuration(metadata)["Effective target FPS per camera"],
                         "cam1: 15; cam2: 20; cam3: 25")

    def test_html_is_self_contained_escaped_and_contains_five_charts_and_heatmap(self):
        import base64
        from html.parser import HTMLParser
        from io import BytesIO
        from PIL import Image
        from stream_density_report_html import generate_html_report
        import stream_density

        path, metadata = self.make_run("html")
        metadata["hardware"]["hostname"] = '<script>alert("host")</script>'
        metadata["accelerator"] = "GPU"
        metadata["hardware"]["gpu_names"] = ["Test GPU"]
        records = [json.loads(line) for line in (path / report.SAMPLES_FILE).read_text().splitlines()]
        records[0]["hardware_metrics"]["GPU_1 GPU Power (W)"] = 12.5
        records[0]["hardware_metrics"]["GPU_1 Compute[CCS] Utilization %"] = 75
        (path / "qmassa-parsed.json").write_text("[]")
        (path / report.SAMPLES_FILE).write_text(
            "\n".join(json.dumps(record) for record in records) + "\n")
        stream_density.save_measurement_interval(
            path, "gst", 2, 1, "pass", {"stream0": 29, "stream1": 31}, 100)
        document = generate_html_report([(path, metadata)])
        self.assertIn("&lt;script&gt;", document)
        self.assertNotIn("<script>", document)
        self.assertNotIn('<div class="threshold-note">', document)
        self.assertNotIn("Avg latency", document)
        for title in ("Hardware", "Test &amp; Software", "Stream Density Summary",
                      "KPI Comparison Table", "FPS/stream",
                      "GPU power (reported source)", "Stream Density FPS Progression Heatmaps",
                      "Per-Frame Latency Percentiles", "stream density (streams)"):
            self.assertIn(title, document)
        self.assertIn("Pooled latency", document)
        self.assertNotIn("Mean p90", document)
        self.assertNotIn("30% warm-up discarded", document)
        self.assertNotIn("Recorded value", document)
        self.assertIn("#f28e2b", document)
        self.assertIn("qmassa: GPU_1 GPU Power (W): 12.50", document)

        class Images(HTMLParser):
            def __init__(self):
                super().__init__()
                self.sources = []

            def handle_starttag(self, tag, attributes):
                if tag == "img":
                    self.sources.append(dict(attributes)["src"])

        images = Images()
        images.feed(document)
        self.assertEqual(len(images.sources), 6)
        for source in images.sources:
            self.assertTrue(source.startswith("data:image/png;base64,"))
            bitmap = Image.open(BytesIO(base64.b64decode(source.split(",", 1)[1])))
            self.assertGreater(bitmap.width, 100)
            self.assertIsNotNone(bitmap.convert("RGB").getextrema())

    def test_html_output_option_also_writes_markdown(self):
        self.make_run("cpu")
        output = self.root / "comparison.html"
        report.main(["--results-dir", str(self.root), "--output", str(output)])
        self.assertTrue(output.exists())
        self.assertTrue(output.with_suffix(".md").exists())

    def test_html_matches_reference_kpi_rows_and_section_order(self):
        from stream_density_report_html import generate_html_report

        path, metadata = self.make_run("reference")
        document = generate_html_report([(path, metadata)])
        self.assertEqual(re.findall(r"<strong>(.*?)</strong>", document), [
            "Stream density", "Throughput", "FPS/stream", "Pooled latency p50",
            "Pooled latency p95", "Pooled latency p99", "Highest per-stream p95",
            "Avg CPU Utilization", "Avg NPU Utilization",
            "Avg GPU Engine Utilization", "GPU power (reported source)",
            "Avg Processor Package Power", "Power/stream (source scope)",
            "Utilization (source scope)", "Sweep status",
        ])
        self.assertEqual(re.findall(r"<h2>(.*?)</h2>", document), [
            "Stream Density Summary", "KPI Comparison Table", "Comparison Charts",
            "Stream Density FPS Progression Heatmaps",
        ])
        for label in ("Hostname", "CPU", "CPU Cores", "Total RAM", "GPU", "NPU",
                  "PL1 / PL2", "OS", "Kernel", "Target FPS per camera (fps)",
                      "Throughput threshold (fps)", "Test configuration", "Archive", "Generated",
                      "System vendor", "System model", "Warmup period (s)",
                      "Measurement interval (s)", "Run acceptance criterion", "Run duration", "Source run"):
            self.assertIn(f"<b>{label}:</b>", document)

    def test_reference_metrics_do_not_relabel_average_fps_or_gpu_power(self):
        from stream_density_report_html import presentation_metrics

        path, metadata = self.make_run("reference-metrics")
        result = presentation_metrics(path, metadata, report.run_metrics(path, metadata))
        self.assertEqual(result["latency_p50"], 20)
        self.assertEqual(result["proc_power_w"], 20)
        self.assertEqual(result["power_per_stream_w"], 10)
        self.assertEqual(result["cpu_util_pct"], 50)
        self.assertEqual(result["avg_fps"], 30)
        self.assertEqual(result["gpu_power_reported"], "N/A")
        sample_path = path / report.SAMPLES_FILE
        sample_path.write_text(sample_path.read_text().replace('"pass"', '"fail"'))
        result = presentation_metrics(path, metadata, report.run_metrics(path, metadata))
        for key in ("stream_density", "total_fps", "avg_fps", "latency_p50",
                    "proc_power_w", "power_per_stream_w"):
            self.assertIsNone(result[key])

    def test_reference_chart_colors_and_dimensions(self):
        import asc_benchmark_report as presentation
        from matplotlib.figure import Figure
        from matplotlib.colors import to_rgba
        from stream_density_report_html import hardware_target

        for target, color in (("CPU", "#4e79a7"), ("GPU.0", "#f28e2b"),
                              ("NPU", "#59a14f"), ("CPU+GPU", "#b07aa1"),
                              ("HETERO:GPU,CPU", "#b07aa1")):
            self.assertEqual(presentation.HW_COLORS[hardware_target(target).lower()], color)
        with patch.object(Figure, "savefig", autospec=True) as save:
            presentation.plot_charts([{"hw": "CPU", "stream_density": 2},
                                      {"hw": "GPU", "stream_density": 4}], str(self.root))
        self.assertEqual(save.call_count, 1)
        figure = save.call_args.args[0]
        self.assertEqual(list(figure.get_size_inches()), [6, 3.5])
        self.assertEqual(figure.axes[0].get_xlabel(), "Streams")
        self.assertTrue(figure.axes[0].yaxis_inverted())
        self.assertEqual(figure.axes[0].get_title(),
                         "Stream density (streams) @ Target FPS (higher is better)")
        self.assertEqual([bar.get_facecolor() for bar in figure.axes[0].patches],
                         [to_rgba("#4e79a7"), to_rgba("#f28e2b")])
        self.assertEqual(save.call_args.kwargs, {"dpi": 150})

    def test_missing_measurements_are_not_plotted_as_zero(self):
        import asc_benchmark_report as presentation
        from matplotlib.figure import Figure

        with patch.object(Figure, "savefig", autospec=True) as save:
            presentation.plot_charts([
                {"hw": "CPU", "stream_density": 2, "proc_power_w": None},
                {"hw": "GPU", "stream_density": 4, "proc_power_w": 20},
            ], str(self.root))
        figure = next(call.args[0] for call in save.call_args_list
                      if str(call.args[1]).endswith("compare_power.png"))
        bars = figure.axes[0].patches
        self.assertEqual(len(bars), 1)
        self.assertEqual(bars[0].get_width(), 20)
        self.assertEqual(bars[0].get_y() + bars[0].get_height() / 2, 1)

    def test_gpu_run_keeps_interval_cpu_npu_and_only_matching_gpu(self):
        from stream_density_report_html import generate_html_report, presentation_metrics, system_info

        path, metadata = self.make_run("all-interval-devices", "GPU.0")
        metadata["gpu_metric_devices"] = {"GPU.0": "GPU_1"}
        metadata["hardware"].update(cpu_cores=16, logical_cpus=22)
        sample_path = path / report.SAMPLES_FILE
        records = [json.loads(line) for line in sample_path.read_text().splitlines()]
        records[0]["hardware_metrics"].update({
            "CPU Utilization %": 20.68, "NPU Utilization %": 0,
            "GPU_0 Compute[CCS] Utilization %": 99,
            "GPU_1 Compute[CCS] Utilization %": 56.63})
        sample_path.write_text("\n".join(json.dumps(record) for record in records))
        metrics = report.run_metrics(path, metadata)
        result = presentation_metrics(path, metadata, metrics)
        self.assertEqual(result["cpu_util_pct"], 20.68)
        self.assertEqual(result["npu_util_pct"], 0)
        self.assertEqual(result["gpu_engine_pct"], 56.63)
        self.assertNotIn("GPU_0", result["utilisation_scope"])
        expected_utilisation = "CPU Utilization %: 20.68<br>NPU Utilization %: 0.00<br>GPU_1 Compute[CCS] Utilization %: 56.63"
        self.assertIn(expected_utilisation,
                  result["utilisation_scope"])
        document = generate_html_report([(path, metadata)])
        self.assertIn(expected_utilisation, document)
        self.assertEqual(system_info(metadata)["cpu_cores"], "16")
        self.assertEqual(system_info(metadata)["logical_cpus"], "22")

    def test_reference_css_and_metadata_markup(self):
        from stream_density_report_html import generate_html_report

        path, metadata = self.make_run("css")
        metadata["settings"]["environment"] = {"CAMERA_STREAM": "camera.json"}
        metadata["hardware"]["arch"] = "x86_64"
        document = generate_html_report([(path, metadata)])
        expected = (Path(__file__).parent / "test_src" / "asc_reference_report.css").read_text()
        actual = re.search(r"<style>(.*?)</style>", document, re.DOTALL).group(1)
        reference_styles = actual.split("  .meta-half, .card, .chart")[0]
        self.assertEqual(re.sub(r"\s+", "", reference_styles), re.sub(r"\s+", "", expected))
        self.assertIn('@media (max-width: 760px)', actual)
        self.assertIn('overflow-wrap: anywhere', actual)
        self.assertIn('<b>Test configuration:</b></td><td>camera.json</td>', document)
        self.assertIn('<td><code>' + self.root.name + '</code></td>', document)
        self.assertIn('6.8 (x86_64)', document)
        self.assertIn('<span style="color:#888;font-size:12px;">(N/A)</span>', document)
        self.assertNotIn('algorithm internal', document)
        self.assertIn('<footer>Generated by compare_stream_density.py &nbsp;|&nbsp; Intel Corporation</footer>', document)
        self.assertIn('<div class="table-scroll">', document)

    def test_reference_heatmap_keeps_sparse_ids_and_last_event_per_count(self):
        import asc_benchmark_report as presentation
        from matplotlib.figure import Figure
        import numpy as np

        steps = [
            {"n_streams": 2, "fps_map": {2: 12, 10: 13}},
            {"n_streams": 1, "fps_map": {2: 15}},
            {"n_streams": 2, "fps_map": {2: 14, 10: 14.5}},
        ]
        with patch.object(Figure, "savefig", autospec=True) as save:
            self.assertTrue(presentation.plot_fps_heatmap(
                steps, "cpu", str(self.root / "heatmap.png"), target_fps=15))
        figure = save.call_args.args[0]
        axis = figure.axes[0]
        plotted = axis.images[0]
        matrix = plotted.get_array()
        self.assertEqual(matrix.shape, (11, 2))
        self.assertEqual(matrix[2, 0], 15)
        self.assertEqual(matrix[2, 1], 14)
        self.assertEqual(matrix[10, 1], 14.5)
        self.assertTrue(np.ma.is_masked(matrix[0, 0]))
        self.assertEqual([label.get_text() for label in axis.get_xticklabels()], ["1", "2"])
        self.assertEqual(plotted.get_clim(), (15 * 0.945 * 0.85, 15))
        self.assertEqual(list(figure.axes[1].lines[0].get_ydata()), [1, 1])
        self.assertEqual(save.call_args.kwargs, {"dpi": 150, "bbox_inches": "tight"})

    def test_progression_adapter_preserves_numeric_stream_ids(self):
        import stream_density
        from stream_density_report_html import fps_progression

        path, _ = self.make_run("sparse-streams")
        stream_density.save_measurement_interval(
            path, "gst", 2, 1, "pass", {"pipeline_stream2": 14, "pipeline_stream10": 15}, 100)
        self.assertEqual(fps_progression(path), [
            {"n_streams": 2, "fps_map": {2: 14, 10: 15}, "status": "pass"},
        ])

    def test_reference_does_not_draw_empty_charts_or_unconfirmed_heatmaps(self):
        from stream_density_report_html import generate_html_report

        path, metadata = self.make_run("empty")
        sample_path = path / report.SAMPLES_FILE
        sample_path.write_text(sample_path.read_text().replace('"pass"', '"fail"'))
        document = generate_html_report([(path, metadata)])
        self.assertNotIn('<img ', document)

    def test_report_supports_recorded_per_camera_targets_and_thresholds(self):
        from stream_density_report_html import generate_html_report

        path, metadata = self.make_run("different-threshold")
        metadata["settings"]["pass_tolerance_ratio"] = 0.9
        metadata["settings"]["effective_target_fps"] = {"cam1": 15, "cam2": 30}
        document = generate_html_report([(path, metadata)])
        self.assertIn("cam1: 15; cam2: 30", document)
        self.assertIn("cam1: 13.50; cam2: 27.00", document)
        self.assertNotIn('95% of target', document)

    def test_old_html_runs_show_missing_metadata_and_history(self):
        from stream_density_report_html import generate_html_report

        path, metadata = self.make_run("legacy")
        document = generate_html_report([(path, metadata)])
        self.assertNotIn("No measurement interval history recorded", document)
        self.assertIn("N/A", document)

    def test_issue_249_inconclusive_status_and_source_runs(self):
        from stream_density_report_html import generate_html_report

        cpu_path, cpu = self.make_run("cpu-source")
        gpu_path, gpu = self.make_run("gpu-source", "GPU+NPU")
        sample_path = gpu_path / report.SAMPLES_FILE
        sample_path.write_text(sample_path.read_text().replace('"pass"', '"inconclusive"'))
        gpu["settings"].update(init_duration=12, measurement_window_seconds=80,
                               pass_tolerance_ratio=0.9, consecutive_passes=3)
        gpu["hardware"].update(system_vendor="Test Vendor", system_model="Test Model")
        document = generate_html_report([(cpu_path, cpu), (gpu_path, gpu)])
        self.assertIn('Sweep status: <b>pass</b>', document)
        self.assertIn('Sweep status: <b>inconclusive</b>', document)
        self.assertIn('Last passing lanes: <b>gst: 2</b> (not confirmed)', document)
        self.assertIn('style="color:#b07aa1;">GPU+NPU</h3>', document)
        self.assertIn('style="color:#b07aa1;">N/A</div>', document)
        for expected in (str(cpu_path), str(gpu_path), "Checkout", "Test Vendor", "Test Model",
                         "3 consecutive passes", "27.00", "INT8"):
            self.assertIn(expected, document)
        self.assertEqual(document.count('<th>CPU</th>'), 1)
        self.assertEqual(document.count('<th>GPU+NPU</th>'), 1)

    def test_issue_249_uses_percentiles_and_interval_only_metrics(self):
        from stream_density_report_html import presentation_metrics, generate_html_report

        path, metadata = self.make_run("interval-only")
        (path / "metrics.csv").write_text(
            'System Latency p50 (ms),11\nSystem Latency p95 (ms),91\n'
            'System Latency p99 (ms),99\nCPU Utilization %,99\nS0 Power Draw W,999\n')
        result = presentation_metrics(path, metadata, report.run_metrics(path, metadata))
        self.assertEqual([result[key] for key in ("latency_p50", "latency_p95", "latency_p99")],
                         [11, 91, 99])
        self.assertEqual(result["highest_stream_p95"], 30)
        self.assertEqual(result["cpu_util_pct"], 50)
        self.assertEqual(result["proc_power_w"], 20)
        sample_path = path / report.SAMPLES_FILE
        records = [json.loads(line) for line in sample_path.read_text().splitlines()]
        records[0]["hardware_metrics"] = {}
        sample_path.write_text("\n".join(json.dumps(record) for record in records))
        result = presentation_metrics(path, metadata, report.run_metrics(path, metadata))
        self.assertIsNone(result["cpu_util_pct"])
        self.assertIsNone(result["proc_power_w"])
        self.assertEqual(result["utilisation_scope"], "N/A")
        document = generate_html_report([(path, metadata)])
        self.assertNotIn("999.0", document)
        self.assertIn("Pooled latency p95", document)

    def test_issue_249_mixed_camera_targets_still_render_heatmaps(self):
        import stream_density
        from stream_density_report_html import generate_html_report

        path, metadata = self.make_run("mixed-targets")
        metadata["settings"].update(
            effective_target_fps={"cam1": 15, "cam2": 30}, pass_tolerance_ratio=0.9)
        stream_density.save_measurement_interval(
            path, "gst", 2, 1, "pass", {"pipeline_stream0": 15, "pipeline_stream1": 30}, 100)
        document = generate_html_report([(path, metadata)])
        self.assertIn('alt="CPU \u2014 Stream FPS Progression"', document)
        self.assertIn("cam1: 13.50; cam2: 27.00", document)

    def test_issue_249_configuration_description_is_escaped(self):
        from stream_density_report_html import generate_html_report

        path, metadata = self.make_run("description")
        metadata["camera_config"]["_about"] = {
            "name": "Checkout", "summary": '<script>alert("description")</script>'}
        document = generate_html_report([(path, metadata)])
        self.assertIn("&lt;script&gt;", document)
        self.assertNotIn("<script>", document)
        self.assertLess(document.index("<b>Summary:</b>"), document.index('<div class="meta">'))

    def test_test_configuration_uses_only_saved_filename_not_name_or_hash(self):
        from stream_density_report_html import generate_html_report

        path, metadata = self.make_run("configuration-name")
        metadata["camera_config"]["_about"] = {"name": "Saved checkout name"}
        document = generate_html_report([(path, metadata)])
        self.assertIn('<b>Test configuration:</b></td><td>N/A</td>', document)
        metadata["settings"]["environment"] = {"CAMERA_STREAM": "camera.json"}
        document = generate_html_report([(path, metadata)])
        self.assertIn('<b>Test configuration:</b></td><td>camera.json</td>', document)
        metadata["camera_config_file"] = "/configs/camera_saved.json"
        document = generate_html_report([(path, metadata)])
        self.assertIn('<b>Test configuration:</b></td><td>camera_saved.json</td>', document)
        metadata.pop("camera_config_file")
        metadata["settings"]["environment"] = {}
        document = generate_html_report([(path, metadata)])
        self.assertIn('<b>Test configuration:</b></td><td>N/A</td>', document)

    def test_requested_heading_labels_units_and_saved_filename(self):
        from stream_density_report_html import generate_html_report

        path, metadata = self.make_run("heading-labels", "GPU")
        metadata["camera_config_file"] = "camera_checkout.json"
        document = generate_html_report([(path, metadata)])
        self.assertEqual(document.count('<h3 style="color:#0071c5;">GPU</h3>'), 2)
        self.assertIn('<h3 style="color:#f28e2b;">GPU</h3>', document)
        self.assertIn('<b>Hostname:</b>', document)
        self.assertNotIn('<b>Host:</b>', document)
        self.assertIn('<b>Test configuration:</b></td><td>camera_checkout.json</td>', document)
        self.assertIn('<b>Target FPS per camera (fps):</b>', document)
        self.assertIn('<b>Effective target FPS per camera (fps):</b>', document)
        self.assertIn('Power/stream (W/stream, source scope)', document)
        self.assertIn('FPS/stream (fps)', document)
        self.assertIn('Utilization (source scope)', document)
        self.assertNotIn('Last passing lanes (not confirmed)', document)

    def test_cli_discovers_runs_without_positional_arguments(self):
        self.make_run("cpu")
        self.make_run("gpu", "GPU")
        report.main(["--results-dir", str(self.root)])
        text = (self.root / "stream_density_report.md").read_text()
        self.assertIn("| CPU | cpu |", text)
        self.assertIn("| GPU | gpu |", text)

    def test_no_argument_cli_uses_results_dir_environment(self):
        self.make_run("cpu")
        with patch.dict("os.environ", {"RESULTS_DIR": str(self.root)}):
            report.main([])
        self.assertTrue((self.root / "stream_density_report.md").exists())

    def test_explicit_run_and_inclusive_date(self):
        cpu, _ = self.make_run("cpu")
        self.make_run("gpu", "GPU")
        output = self.root / "explicit.md"
        report.main([str(cpu), "--until", "2026-10-01", "--output", str(output)])
        text = output.read_text()
        self.assertIn("| CPU | cpu |", text)
        self.assertNotIn("| GPU | gpu |", text)

    def test_missing_power_and_latency_are_not_zero(self):
        path, metadata = self.make_run("cpu")
        (path / "metrics.csv").write_text("")
        sample_path = path / report.SAMPLES_FILE
        records = [json.loads(line) for line in sample_path.read_text().splitlines()]
        records[0]["samples_by_stream"] = {}
        records[0]["hardware_metrics"] = {}
        sample_path.write_text("\n".join(json.dumps(record) for record in records))
        metrics = report.run_metrics(path, metadata)
        self.assertIsNone(metrics["latency_p50"])
        self.assertEqual(metrics["power"], {})
        self.assertIn("N/A", report.generate_report([(path, metadata)]))

    def test_gpu_collector_power_and_utilisation(self):
        path, metadata = self.make_run("gpu", "GPU.0")
        metadata["gpu_metric_devices"] = {"GPU.0": "GPU_1"}
        (path / "metrics.csv").write_text(
            "GPU_0 GPU Power (W),12\nGPU_0 Compute[CCS] Utilization %,75\n"
            "GPU_1 GPU Power (W),90\nGPU_1 Compute[CCS] Utilization %,99\n")
        sample_path = path / report.SAMPLES_FILE
        records = [json.loads(line) for line in sample_path.read_text().splitlines()]
        records[0]["hardware_metrics"] = report.read_metrics(path / "metrics.csv")
        sample_path.write_text("\n".join(json.dumps(record) for record in records))
        metrics = report.run_metrics(path, metadata)
        self.assertEqual(metrics["power"], {"GPU_1 GPU Power (W)": 45})
        self.assertEqual(metrics["utilisation"], {"GPU_1 Compute[CCS] Utilization %": 99})
        metadata.pop("gpu_metric_devices")
        metrics = report.run_metrics(path, metadata)
        self.assertEqual(metrics["power"], {})
        self.assertEqual(metrics["utilisation"], {})

    def test_gpu_mapping_matches_pci_identity_not_card_index(self):
        card = self.root / "card1" / "device"
        card.parent.mkdir()
        pci = self.root / "0000:03:00.0"
        pci.mkdir()
        card.symlink_to(pci, target_is_directory=True)
        def get_property(device, key):
            self.assertEqual(device, "GPU.0")
            self.assertEqual(key, "DEVICE_PCI_INFO")
            return SimpleNamespace(domain=0, bus=3, device=0, function=0)

        for alias in ("GPU.0", "GPU"):
            with self.subTest(alias=alias):
                core = SimpleNamespace(available_devices=[alias], get_property=get_property)
                with patch.dict("sys.modules", {"openvino": SimpleNamespace(Core=lambda: core)}), \
                        patch.object(Path, "glob", return_value=[card]):
                    self.assertEqual(report.gpu_metric_devices(), {"GPU.0": "GPU_1"})

    def test_invalid_hardware_is_skipped(self):
        valid, _ = self.make_run("valid")
        for index, hardware in enumerate((None, {}, {"cpu": ""}, {"cpu": "Test", "accelerators": None})):
            with self.subTest(hardware=hardware):
                path, metadata = self.make_run(f"invalid{index}", day=3)
                metadata["hardware"] = hardware
                (path / report.METADATA_FILE).write_text(json.dumps(metadata))
                self.assertEqual(report.select_runs([valid, path])[0][0], valid)

    def test_vlm_targets_and_component_details_are_preserved(self):
        configs = self.root / "configs"
        configs.mkdir()
        (configs / "camera.json").write_text(json.dumps(self.camera))
        paths = []
        for device in ("GPU", "NPU"):
            path, template = self.make_run(device)
            stage = {"type": "vlm", "model": "detector", "device": "CPU", "precision": "FP16",
                     "vlm_model": "language-model", "vlm_device": device, "vlm_precision": "INT8"}
            pipeline = {"workload_pipeline_map": {"detect": [stage]}}
            (configs / "pipeline.json").write_text(json.dumps(pipeline))
            with patch.object(report, "gpu_metric_devices", return_value={}), \
                    patch.object(report.subprocess, "run", side_effect=FileNotFoundError):
                metadata = report.capture_run(path, self.root, {
                    "CAMERA_STREAM": "camera.json", "WORKLOAD_DIST": "pipeline.json"}, {})
            self.assertEqual(metadata["accelerator"], f"CPU+{device}")
            report.finish_run(path, metadata)
            paths.append(path)
        selected = report.select_runs(paths)
        self.assertEqual(len(selected), 2)
        text = report.generate_report(selected)
        self.assertIn("| detect | vlm | detector | CPU | FP16 |", text)
        self.assertIn("| detect | vlm | language-model | GPU | INT8 |", text)
        self.assertIn("| detect | vlm | language-model | NPU | INT8 |", text)

    def test_missing_container_window_cannot_report_partial_density(self):
        path, metadata = self.make_run("partial")
        with (path / report.SAMPLES_FILE).open("a") as samples:
            samples.write("\n" + json.dumps({"record_type": "sweep_result",
                "container_name": "other", "status": "pass", "lane_count": 2}))
        self.assertIsNone(report.run_metrics(path, metadata)["streams"])

    def test_raw_metrics_are_consolidated_per_run_without_csv(self):
        path, metadata = self.make_run("npu", "NPU")
        (path / "metrics.csv").unlink()
        (path / "npu_usage.csv").write_text("percent_usage\n50\n70\n")
        metrics = report.run_metrics(path, metadata)
        self.assertEqual(metrics["utilisation"], {"NPU Utilization %": 60})
        self.assertEqual(metrics["streams"], 2)
        self.assertFalse((path / "metrics.csv").exists())

    def test_capture_and_finish_snapshot(self):
        _, template = self.make_run("template", "GPU")
        configs = self.root / "configs"
        configs.mkdir()
        (configs / "camera.json").write_text(json.dumps(self.camera))
        (configs / "pipeline.json").write_text(json.dumps(template["pipeline_config"]))
        run = self.root / "new"
        run.mkdir()
        with patch.object(report.subprocess, "run", side_effect=FileNotFoundError):
            metadata = report.capture_run(run, self.root, {
                "CAMERA_STREAM": "camera.json", "WORKLOAD_DIST": "pipeline.json"}, {})
        self.assertEqual(metadata["accelerator"], "GPU")
        self.assertEqual(metadata["camera_config"], self.camera)
        self.assertNotIn("finished_at", metadata)
        self.assertEqual(report.select_runs([run, self.root / "template"])[0][0], self.root / "template")
        report.finish_run(run, metadata)
        saved = report.read_json(run / report.METADATA_FILE)
        self.assertGreaterEqual(saved["duration_seconds"], 0)
        self.assertTrue(saved["os"])
        self.assertTrue(saved["kernel"])
        self.assertTrue(saved["hardware"]["cpu"])

    def test_pci_fallback_identifies_accelerator_without_lspci(self):
        device = self.root / "0000:00:02.0"
        device.mkdir()
        for name, value in {"class": "0x030000", "vendor": "0x8086", "device": "0x7d55"}.items():
            (device / name).write_text(value)
        hardware = report.pci_hardware(self.root)
        self.assertIn("vendor=0x8086, device=0x7d55", hardware[0])

    def test_power_without_window_samples_is_not_estimated(self):
        path, metadata = self.make_run("legacy")
        sample_path = path / report.SAMPLES_FILE
        records = [json.loads(line) for line in sample_path.read_text().splitlines()]
        records[0].pop("hardware_metrics")
        sample_path.write_text("\n".join(json.dumps(record) for record in records))
        self.assertEqual(report.run_metrics(path, metadata)["power"], {})

    def test_hardware_window_excludes_warmup_and_later_samples(self):
        npu = self.root / "npu_usage.csv"
        gpu = self.root / "qmassa0_parsed.json"
        npu.write_text("percent_usage\n100\n")
        gpu.write_text(json.dumps([{"Power Usage (W)": 100, "CCS %": 100}]))
        start = report.snapshot_hardware_logs(self.root)
        with npu.open("a") as output:
            output.write("40\n60\n")
        gpu.write_text(json.dumps([{"Power Usage (W)": 100, "CCS %": 100},
                                   {"Power Usage (W)": 12, "CCS %": 50}]))
        end = report.snapshot_hardware_logs(self.root)
        with npu.open("a") as output:
            output.write("100\n")
        metrics = report.window_hardware_metrics(start, end)
        self.assertEqual(metrics["NPU Utilization %"], 50)
        self.assertEqual(metrics["GPU_0 GPU Power (W)"], 12)

    def test_hardware_window_rejects_truncated_file(self):
        path = self.root / "npu_usage.csv"
        path.write_text("percent_usage\n100\n100\n100\n")
        start = report.snapshot_hardware_logs(self.root)
        path.write_text("percent_usage\n1\n")
        self.assertEqual(report.window_hardware_metrics(start, report.snapshot_hardware_logs(self.root)), {})

    def test_raw_qmassa_window_reuses_existing_parser(self):
        path = self.root / "qmassa0-7d55-xe-tool-generated.json"
        def state(power):
            return {"devs_state": [{"dev_stats": {
                "power": [{"pkg_cur_power": power}],
                "eng_usage": {engine: [50] for engine in ("ccs", "vecs", "rcs", "bcs", "vcs")}}}]}
        path.write_text(json.dumps({"states": [state(100)]}))
        start = report.snapshot_hardware_logs(self.root)
        path.write_text(json.dumps({"states": [state(100), state(12)]}))
        metrics = report.window_hardware_metrics(start, report.snapshot_hardware_logs(self.root))
        self.assertEqual(metrics["GPU_0 GPU Power (W)"], 12)

    def test_description_from_real_age_verification_config(self):
        path, metadata = self.make_run("age")
        config = Path(__file__).resolve().parents[2] / "configs" / "camera_to_workload_asc_age_verification.json"
        if not config.exists():
            self.skipTest("Loss-prevention scenario configs are not installed")
        metadata["camera_config"] = report.read_json(config)
        text = report.generate_report([(path, metadata)])
        self.assertIn(metadata["camera_config"]["_about"]["name"], text)
        self.assertIn(metadata["camera_config"]["_about"]["why_run_it"], text)

    def test_malformed_metadata_does_not_block_valid_runs(self):
        valid, _ = self.make_run("valid")
        broken = self.root / "broken"
        broken.mkdir()
        (broken / report.METADATA_FILE).write_text("{")
        self.assertEqual(report.select_runs([valid, broken])[0][0], valid)

    def test_saved_window_persists_hardware_metrics(self):
        import stream_density
        stream_density.save_window_latency_samples(self.root, "gst", 1,
            {"cam": [10]}, {"stream0": 30}, {"S0 Power Draw W": 12})
        record = json.loads((self.root / report.SAMPLES_FILE).read_text())
        self.assertEqual(record["hardware_metrics"], {"S0 Power Draw W": 12})


if __name__ == "__main__":
    unittest.main()