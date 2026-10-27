import json
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
        (path / report.SAMPLES_FILE).write_text("\n".join(json.dumps(record) for record in records))
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
        self.assertEqual(metrics["latency"], 20)
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
        self.assertIsNone(metrics["latency"])

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
        self.assertIsNone(metrics["latency"])
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