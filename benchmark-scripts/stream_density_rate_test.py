import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import stream_density


class StreamDensityRateTest(unittest.TestCase):
    def test_weighted_rate_uses_only_measurement_rows(self):
        samples = [(14.97, 1.00), (14.99, 1.00), (14.94, 1.00),
                   (15.76, 1.02), (14.97, 1.00), (14.42, 1.04),
                   (14.97, 1.00), (14.96, 1.00)]
        with tempfile.TemporaryDirectory() as results_dir, \
                patch.dict(os.environ, {'TIMESTAMP': ''}), \
                patch('stream_density.get_pipeline_stream_count', return_value=1):
            log_file = os.path.join(results_dir, 'pipeline_stream0_123_gst.log')
            with open(log_file, 'w') as output:
                output.write('fps,duration_seconds\n1.00,1.00\n')
            start_offsets = stream_density.snapshot_stream_log_offsets(results_dir, 'gst')
            with open(log_file, 'a') as output:
                for fps, seconds in samples:
                    output.write(f'{fps:.2f},{seconds:.2f}\n')
            end_offsets = stream_density.snapshot_stream_log_offsets(results_dir, 'gst')
            with open(log_file, 'a') as output:
                output.write('99.00,1.00\n')

            calculation_log = io.StringIO()
            with redirect_stdout(calculation_log):
                total, minimum, rates, fps_samples = stream_density.calculate_multi_stream_fps(
                    1, results_dir, 'gst', start_offsets=start_offsets, end_offsets=end_offsets)

        expected_rate = sum(fps * seconds for fps, seconds in samples) / sum(
            seconds for _, seconds in samples)
        self.assertAlmostEqual(expected_rate, 120.872 / 8.06)
        self.assertAlmostEqual(rates['pipeline_stream0'], expected_rate, places=6)
        self.assertAlmostEqual(total, expected_rate, places=6)
        self.assertAlmostEqual(minimum, expected_rate, places=6)
        self.assertEqual(fps_samples['pipeline_stream0'], samples)
        self.assertIn('weighted_frames_estimate=120.872000', calculation_log.getvalue())
        self.assertIn('total_duration_seconds=8.060000', calculation_log.getvalue())
        self.assertIn('measured_fps=14.996526', calculation_log.getvalue())

    def test_report_uses_weighted_rate_and_displays_p10_and_p90(self):
        fps_samples = [10.0, 10.0] + [15.0] * 8
        with tempfile.TemporaryDirectory() as results_dir, \
                patch.dict(os.environ, {'TIMESTAMP': ''}), \
                patch('stream_density.get_pipeline_stream_count', return_value=1):
            log_file = os.path.join(results_dir, 'pipeline_stream0_123_gst.log')
            with open(log_file, 'w') as output:
                output.write('fps,duration_seconds\n')
                for fps in fps_samples:
                    output.write(f'{fps:.2f},1.00\n')
            _, _, rates, samples = stream_density.calculate_multi_stream_fps(
                1, results_dir, 'gst')

        output = io.StringIO()
        with redirect_stdout(output):
            stream_density.print_stream_density_report(
                1, rates, {'pipeline_stream0': 15.0},
                {'pipeline_stream0': 14.25}, 100, 2, 2, 0.95,
                stream_samples=samples)

        report = output.getvalue()
        self.assertEqual(rates['pipeline_stream0'], 14.0)
        self.assertIn('throughput threshold', report)
        self.assertIn('measured (avg)', report)
        self.assertIn('p10', report)
        self.assertIn('p90', report)
        self.assertIn('14.0000', report)
        self.assertIn('10.00', report)
        self.assertIn('15.00', report)
        self.assertIn('seconds below throughput threshold', report)
        self.assertIn('2.00 s (20%)', report)
        self.assertIn('Result: Fail', report)
        self.assertIn('measuring 14.000000 FPS', report)

    def test_sweep_fails_on_weighted_rate_even_when_p90_passes(self):
        responses = [
            (15.0, 15.0, {'pipeline_stream0': 15.0}, {'pipeline_stream0': [15.0] * 100}),
            (14.0, 14.0, {'pipeline_stream0': 14.0},
             {'pipeline_stream0': [10.0] * 20 + [15.0] * 80}),
            (15.0, 15.0, {'pipeline_stream0': 15.0}, {'pipeline_stream0': [15.0] * 100}),
        ]
        env_vars = {'INIT_DURATION': '0', 'PIPELINE_INC': '1',
                    'CONSECUTIVE_FAIL_WINDOWS': '1',
                    'CONSECUTIVE_PASS_WINDOWS': '1'}
        with patch('stream_density.measure_pipeline_memory', return_value=100), \
                patch('stream_density.clean_up_pipeline_logs'), \
                patch('stream_density.check_can_add_pipelines', return_value=True), \
                patch('stream_density.benchmark.docker_compose_containers'), \
                patch('stream_density.time.sleep'), \
                patch('stream_density.check_non_empty_result_logs'), \
                patch('stream_density.collect_measurement_window',
                      return_value=({}, {}, 100.0, {'pipeline_stream0.log': 100}, True)), \
                patch('stream_density.calculate_multi_stream_fps', side_effect=responses) as calculate, \
                patch('stream_density.calculate_pipeline_latency', return_value=(0.0, 0.0)), \
                patch('stream_density.build_per_stream_target_fps',
                      return_value={'pipeline_stream0': 15.0}), \
                patch('stream_density.build_per_stream_camera_meta', return_value={}), \
                patch('stream_density.print_stream_density_report') as report:
            with redirect_stdout(io.StringIO()):
                num_pipelines, passed, _ = stream_density.run_pipeline_iterations(
                    env_vars, ['docker-compose.yml'], '/tmp/results', 'gst', 15.0)

        self.assertEqual((num_pipelines, passed), (1, True))
        self.assertEqual(calculate.call_count, 3)
        self.assertEqual(report.call_args.args[1], {'pipeline_stream0': 15.0})


if __name__ == '__main__':
    unittest.main()