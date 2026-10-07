'''
* Copyright (C) 2024 Intel Corporation.
*
* SPDX-License-Identifier: Apache-2.0
'''

import mock
import subprocess  # nosec B404
import unittest
import tempfile
import json
from unittest.mock import patch, mock_open, MagicMock
import stream_density
from stream_density import validate_and_setup_env, ArgumentError
from stream_density import (
    RESULTS_DIR_KEY,
    PIPELINE_INCR_KEY,
    INIT_DURATION_KEY,
    DEFAULT_TARGET_FPS
)
import os
import io
from contextlib import redirect_stdout


class Testing(unittest.TestCase):
    def test_build_per_stream_target_fps_mixed_values(self):
        stream_fps_dict = {
            'pipeline_stream0': 10.0,
            'pipeline_stream1': 10.0,
            'pipeline_stream2': 10.0,
            'pipeline_stream3': 10.0,
            'pipeline_stream4': 10.0,
        }
        camera_config = {
            'lane_config': {
                'cameras': [
                    {'targetFps': 30},
                    {'targetFps': '15.5'},
                    {'targetFps': 0},
                    {'targetFps': -7},
                    {'targetFps': 'bad'},
                ]
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            json.dump(camera_config, temp_file)
            temp_file.flush()
            temp_config_path = temp_file.name

        try:
            with patch.dict(os.environ, {'CAMERA_STREAM': temp_config_path}, clear=False):
                result = stream_density.build_per_stream_target_fps(
                    stream_fps_dict, default_target_fps=15
                )

            expected = {
                'pipeline_stream0': 30.0,
                'pipeline_stream1': 15.5,
                'pipeline_stream2': 15,
                'pipeline_stream3': 15,
                'pipeline_stream4': 15,
            }
            self.assertEqual(result, expected)
        finally:
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)

    def test_build_per_stream_target_fps_missing_config_uses_default(self):
        stream_fps_dict = {
            'pipeline_stream0': 10.0,
            'pipeline_stream7': 10.0,
            'stream_without_index': 10.0,
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            missing_path = os.path.join(tmpdir, 'non-existent-camera-config-for-test.json')
        
        with patch.dict(os.environ, {'CAMERA_STREAM': missing_path}, clear=False):
            result = stream_density.build_per_stream_target_fps(
                stream_fps_dict, default_target_fps=22.0
            )

        expected = {
            'pipeline_stream0': 22.0,
            'pipeline_stream7': 22.0,
            'stream_without_index': 22.0,
        }
        self.assertEqual(result, expected)

    def test_is_env_non_empty(self):
        sys_env = os.environ.copy()
        sys_env["EMPTY"] = ""
        test_cases = [
            # Testcase: empty env_vars
            (None, 'USER', False),
            # Testcase: system env with USER key
            (sys_env, 'USER', True),
            # Testcase: system env with NON_EXISTING_A key
            (sys_env, 'NON_EXISTING_A', False),
            (sys_env, 'EMPTY', False),
        ]
        for env_vars, key, expected in test_cases:
            with self.subTest(env_vars=env_vars, key=key):
                self.assertEqual(stream_density.is_env_non_empty(
                    env_vars, key), expected)

    def test_check_non_empty_result_logs_max_tries(self):
        # no file at all case:
        try:
            stream_density.check_non_empty_result_logs(
                1, './non-existing-results', 'abc')
            self.fail('expected ValueError exception')
        except ValueError as ex:
            self.assertTrue("""ERROR: cannot find all pipeline log files
                    after max retries""" in str(ex))
        # 1 file only but 2 pipelines:
        test_results_dir = './test_results'
        testFile = os.path.join(
            test_results_dir,
            'pipeline12345656_abc.log')
        try:
            os.makedirs(test_results_dir)
            # create a non empty temporary log file
            with open(testFile, 'w') as file:
                file.write('this is a test')
            stream_density.check_non_empty_result_logs(
                2, test_results_dir, 'abc')
            self.fail('expected ValueError exception')
        except ValueError as ex:
            self.assertTrue("""ERROR: cannot find all pipeline log files
                    after max retries""" in str(ex))
        finally:
            if os.path.exists(testFile):
                os.remove(testFile)
            if not os.listdir(test_results_dir):
                os.rmdir(test_results_dir)

    def test_check_non_empty_result_logs_success(self):
        test_results_dir = './test_results'
        testFile1 = os.path.join(
            test_results_dir, 'pipeline12345656_abc.log')
        testFile2 = os.path.join(
            test_results_dir, 'pipeline98765432_abc.log')
        try:
            os.makedirs(test_results_dir)
            # create two non empty temporary log files
            with open(testFile1, 'w') as file:
                file.write('this is a test')
            with open(testFile2, 'w') as file:
                file.write('another file for testing')

            stream_density.check_non_empty_result_logs(
                2, test_results_dir, 'abc')
            stream_density.check_non_empty_result_logs(
                2, test_results_dir, 'abc')
        except ValueError as ex:
            self.fail("""ERROR: cannot find all pipeline log files
                    after max retries""")
        finally:
            if os.path.exists(testFile1):
                os.remove(testFile1)
            if os.path.exists(testFile2):
                os.remove(testFile2)
            if not os.listdir(test_results_dir):
                os.rmdir(test_results_dir)

    def test_calculate_multi_stream_fps_success(self):
        samples = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        with tempfile.TemporaryDirectory() as test_results_dir:
            log_file = os.path.join(
                test_results_dir, 'pipeline_stream0_123_gst.log')
            with open(log_file, 'w') as file:
                file.write('fps,duration_seconds\n')
                file.write(''.join(f'{s},1.00\n' for s in samples))

            with patch('stream_density.get_pipeline_stream_count',
                       return_value=1):
                total_fps, min_stream_fps, stream_fps, stream_samples = (
                    stream_density.calculate_multi_stream_fps(
                        1, test_results_dir, 'gst'))

            self.assertEqual(total_fps, 5.5)
            self.assertEqual(min_stream_fps, 5.5)
            self.assertEqual(stream_fps, {'pipeline_stream0': 5.5})
            self.assertEqual(
                stream_samples,
                {'pipeline_stream0': [(float(s), 1.0) for s in samples]})

    def test_calculate_multi_stream_fps_measurement_window(self):
        with tempfile.TemporaryDirectory() as test_results_dir, \
                patch.dict(os.environ, {'TIMESTAMP': ''}), \
                patch('stream_density.get_pipeline_stream_count',
                      return_value=1):
            log_file = os.path.join(
                test_results_dir, 'pipeline_stream0_123_gst.log')
            with open(log_file, 'w') as file:
                file.write('fps,duration_seconds\n1,1.00\n2,1.00\n3,1.00\n')
            start_offsets = stream_density.snapshot_stream_log_offsets(
                test_results_dir, 'gst')
            with open(log_file, 'a') as file:
                file.write('14,1.00\n15,1.00\n16,1.00\n')
            end_offsets = stream_density.snapshot_stream_log_offsets(
                test_results_dir, 'gst')
            with open(log_file, 'a') as file:
                file.write('99,1.00\n100,1.00\n')

            total_fps, min_stream_fps, stream_fps, stream_samples = (
                stream_density.calculate_multi_stream_fps(
                    1, test_results_dir, 'gst',
                    start_offsets=start_offsets,
                    end_offsets=end_offsets))

        self.assertEqual(
            stream_samples,
            {'pipeline_stream0': [(14.0, 1.0), (15.0, 1.0), (16.0, 1.0)]})
        self.assertEqual(total_fps, 15.0)
        self.assertEqual(stream_fps, {'pipeline_stream0': 15.0})
        self.assertEqual(min_stream_fps, 15.0)

    def test_extract_fps_samples_requires_duration_column(self):
        with tempfile.TemporaryDirectory() as test_results_dir:
            log_file = os.path.join(test_results_dir, 'pipeline_stream0.log')
            with open(log_file, 'w') as file:
                file.write('fps,duration_seconds\n')
                file.write('15\n')
                file.write('14.5,1.25\n')

            self.assertEqual(
                stream_density.extract_fps_samples(log_file),
                [(14.5, 1.25)])

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

    def test_count_measurement_samples_counts_generated_stream_logs_once(self):
        with tempfile.TemporaryDirectory() as test_results_dir, \
                patch.dict(os.environ, {'TIMESTAMP': ''}), \
                patch('stream_density.get_pipeline_stream_count',
                      return_value=2):
            first_log = os.path.join(
                test_results_dir, 'pipeline_stream0_lane1_gst.log')
            second_log = os.path.join(
                test_results_dir, 'pipeline_stream1_lane2_gst.log')
            with open(first_log, 'w') as file:
                file.write('fps,duration_seconds\n' + '1,1.00\n' * 99)
            with open(second_log, 'w') as file:
                file.write('fps,duration_seconds\n' + '1,1.00\n' * 100)

            end_offsets = {
                first_log: os.path.getsize(first_log),
                second_log: os.path.getsize(second_log),
            }
            counts, expected_count = stream_density.count_measurement_samples(
                2, test_results_dir, 'gst', {}, end_offsets)

        self.assertEqual(expected_count, 2)
        self.assertEqual(sorted(counts.values()), [99, 100])

    def test_collect_measurement_window_extends_to_sample_floor(self):
        elapsed = [0.0]

        def advance_time(seconds):
            elapsed[0] += seconds

        with patch('stream_density.time.monotonic',
                   side_effect=lambda: elapsed[0]), \
                patch('stream_density.time.sleep', side_effect=advance_time), \
                patch('stream_density.snapshot_stream_log_offsets',
                      side_effect=[{}, {'log': 99}, {'log': 100}]), \
                patch('stream_density.count_measurement_samples',
                      side_effect=[({'log': 99}, 1), ({'log': 100}, 1)]):
            result = stream_density.collect_measurement_window(
                1, '/results', 'gst', minimum_duration_seconds=2)

        start_offsets, end_offsets, duration, counts, enough_samples = result
        self.assertEqual(start_offsets, {})
        self.assertEqual(end_offsets, {'log': 100})
        self.assertEqual(duration, 3)
        self.assertEqual(counts, {'log': 100})
        self.assertTrue(enough_samples)

    def test_collect_measurement_window_caps_extension_at_configured_duration(self):
        elapsed = [0.0]

        def advance_time(seconds):
            elapsed[0] += seconds

        with patch('stream_density.time.monotonic',
                   side_effect=lambda: elapsed[0]), \
                patch('stream_density.time.sleep', side_effect=advance_time), \
                patch('stream_density.snapshot_stream_log_offsets',
                      return_value={}), \
                patch('stream_density.count_measurement_samples',
                      return_value=({'log': 99}, 1)):
            result = stream_density.collect_measurement_window(
                1, '/results', 'gst', minimum_duration_seconds=2)

        self.assertEqual(result[2], 4)
        self.assertFalse(result[4])

    def test_print_stream_density_report_summary(self):
        stream_fps = {'pipeline_stream0': 14.6, 'pipeline_stream1': 15.0}
        targets = {'pipeline_stream0': 15.0, 'pipeline_stream1': 15.0}
        pass_marks = {'pipeline_stream0': 14.25, 'pipeline_stream1': 14.25}
        samples = {
            'pipeline_stream0': [(fps, 1.0) for fps in
                                 [16, 14, 15, 13, 17, 14, 16, 12, 15, 14]],
            'pipeline_stream1': [(15, 1.0)] * 10,
        }
        meta = {
            'pipeline_stream0': {'camera': 'cam1', 'workload': 'wl'},
            'pipeline_stream1': {'camera': 'cam2', 'workload': 'wl'},
        }

        output = io.StringIO()
        with redirect_stdout(output):
            stream_density.print_stream_density_report(
                18, stream_fps, targets, pass_marks, 100, 2, 2, 0.95,
                meta, 10, samples, 103.4,
                {'pipeline_stream0_lane1_gst.log': 100,
                 'pipeline_stream1_lane1_gst.log': 103})
        report = output.getvalue()

        self.assertIn('Use case density (lanes) 18', report)
        self.assertIn('Stream density (streams) 2', report)
        self.assertIn('Warmup period            10 s', report)
        self.assertIn('Measurement interval     100 s minimum', report)
        self.assertIn('run acceptance criterion (two consecutive passes)', report)
        self.assertIn('Actual measurement time  103.4 s (minimum 100 s)', report)
        self.assertIn(
            'Samples per pipeline log 100-103 across 2 logs (minimum 100)',
            report)
        header = next(line for line in report.splitlines()
                      if line.startswith('stream'))
        for column in ('target', 'throughput threshold', 'measured (avg)',
                       'p10', 'p90', 'seconds below throughput threshold', 'result'):
            self.assertIn(column, header)
        rows = {line.split()[0]: line for line in report.splitlines()
                if line.startswith('pipeline_stream')}
        self.assertIn('14.25', rows['pipeline_stream0'])
        self.assertIn('14.600000', rows['pipeline_stream0'])
        self.assertIn('12.00', rows['pipeline_stream0'])
        self.assertIn('16.00', rows['pipeline_stream0'])
        self.assertIn('5.00 s (50%)', rows['pipeline_stream0'])
        self.assertIn('pass', rows['pipeline_stream0'])
        self.assertIn('0.00 s (0%)', rows['pipeline_stream1'])
        self.assertIn(
            'The lowest-throughput stream was pipeline_stream0', report)
        self.assertIn('targeting 15.00 FPS', report)
        self.assertIn('measuring 14.600000 FPS', report)
        self.assertIn('against its 14.25 FPS throughput threshold', report)

    def test_count_valid_streams(self):
        stream_fps_dict = {
            'pipeline_stream0': 15.0,
            'pipeline_stream1': 12.5,
            'pipeline_stream2': 0.0,
        }

        self.assertEqual(
            stream_density.count_valid_streams(stream_fps_dict), 2)

    def test_count_valid_streams_five_lanes_of_six_cameras(self):
        lane_count = 5
        cameras_per_lane = 6
        stream_fps_dict = {
            f'pipeline_stream{stream_index}': 15.0
            for stream_index in range(lane_count * cameras_per_lane)
        }

        self.assertEqual(
            stream_density.count_valid_streams(stream_fps_dict), 30)

    def test_clean_up_pipeline_logs(self):
        test_results_dir = './test_results_clean'
        testFile1 = os.path.join(
            test_results_dir, 'pipeline12345656_abc.log')
        testFile2 = os.path.join(
            test_results_dir, 'pipeline98765432_def.log')
        try:
            os.makedirs(test_results_dir)
            with open(testFile1, 'w') as file:
                file.write('this is a test')
            with open(testFile2, 'w') as file:
                file.write('another file for testing')
            stream_density.clean_up_pipeline_logs(
                test_results_dir)
            self.assertFalse(
                os.path.exists(testFile1),
                f"file still exists: {testFile1}")
            self.assertFalse(
                os.path.exists(testFile2),
                f"file still exists: {testFile2}")
        except Exception as ex:
            self.fail(f"ERROR: found exception {ex}")
        finally:
            if os.path.exists(testFile1):
                os.remove(testFile1)
            if os.path.exists(testFile2):
                os.remove(testFile2)
            if not os.listdir(test_results_dir):
                os.rmdir(test_results_dir)

    def test_validate_and_setup_env(self):
        test_cases = [
            # Test case 1: Valid environment, valid target_fps_list
            {
                "env_vars": {RESULTS_DIR_KEY: "/some/path"},
                "target_fps_list": [20.0],
                "expect_exception": False,
                "expected_target_fps_list": [20.0],
                "expected_env_vars": {
                    RESULTS_DIR_KEY: "/some/path",
                    INIT_DURATION_KEY: "10"
                },
            },
            # Test case 2: Missing RESULTS_DIR_KEY in env_vars
            {
                "env_vars": {},
                "target_fps_list": [20.0],
                "expect_exception": True,
                "exception_type": ArgumentError,
            },
            # Test case 3: Empty target_fps_list (should set to default)
            {
                "env_vars": {RESULTS_DIR_KEY: "/some/path"},
                "target_fps_list": [],
                "expect_exception": False,
                "expected_target_fps_list": [DEFAULT_TARGET_FPS],
                "expected_env_vars": {
                    RESULTS_DIR_KEY: "/some/path",
                    INIT_DURATION_KEY: "10"
                },
            },
            # Test case 4: Negative target_fps value in target_fps_list
            {
                "env_vars": {RESULTS_DIR_KEY: "/some/path"},
                "target_fps_list": [-5.0],
                "expect_exception": True,
                "exception_type": ArgumentError,
            },
            # Test case 5: Missing INIT_DURATION_KEY in env_vars-
            # should default to "10"
            {
                "env_vars": {RESULTS_DIR_KEY: "/some/path"},
                "target_fps_list": [20.0],
                "expect_exception": False,
                "expected_target_fps_list": [20.0],
                "expected_env_vars": {
                    RESULTS_DIR_KEY: "/some/path",
                    INIT_DURATION_KEY: "10"
                },
            },
            # Test case 6: PIPELINE_INCR_KEY <= 0 (should raise exception)
            {
                "env_vars": {
                    RESULTS_DIR_KEY: "/some/path",
                    PIPELINE_INCR_KEY: "0"
                },
                "target_fps_list": [20.0],
                "expect_exception": True,
                "exception_type": ArgumentError,
            },
        ]

        for i, test_case in enumerate(test_cases):
            with self.subTest(f"Test case {i + 1}"):
                # Make a copy to avoid mutation
                env_vars = test_case["env_vars"].copy()
                target_fps_list = test_case["target_fps_list"].copy()

                if test_case["expect_exception"]:
                    with self.assertRaises(test_case["exception_type"]):
                        validate_and_setup_env(env_vars, target_fps_list)
                else:
                    try:
                        validate_and_setup_env(env_vars, target_fps_list)
                        # Verify the target_fps_list was updated as expected
                        self.assertEqual(
                            target_fps_list,
                            test_case["expected_target_fps_list"])
                        # Verify env_vars was updated as expected
                        expected_env_vars = test_case["expected_env_vars"]
                        for key, value in expected_env_vars.items():
                            self.assertEqual(env_vars.get(key), value)
                    except Exception as ex:
                        self.fail(f"Unexpected exception raised: {ex}")

    @patch('time.sleep', return_value=None)
    @patch('stream_density.collect_measurement_window',
           return_value=({}, {}, 100.0,
                         {'pipeline_stream0_lane_gst.log': 100}, True))
    @patch('benchmark.docker_compose_containers')
    @patch('stream_density.calculate_multi_stream_fps')
    @patch('stream_density.calculate_pipeline_latency',
           return_value=(0.0, 0.0))
    @patch('stream_density.snapshot_stream_log_offsets', return_value={})
    @patch('stream_density.check_can_add_pipelines', return_value=True)
    @patch('stream_density.measure_pipeline_memory', return_value=100.0)
    @patch('stream_density.check_non_empty_result_logs')
    @patch('stream_density.clean_up_pipeline_logs')
    def test_pipeline_iterations(
        self,
        mock_clean_logs,
        mock_check_logs,
        mock_measure_memory,
        mock_check_can_add,
        mock_snapshot_offsets,
        mock_calculate_latency,
        mock_calculate_fps,
        mock_docker_compose,
        mock_collect_window,
        mock_sleep
    ):
        test_cases = [
            # Test case 1: fail at two pipelines, then pass at one pipeline.
            {
                "env_vars": {
                    "INIT_DURATION": "10",
                    "PIPELINE_INC": "1",
                    "CONSECUTIVE_PASS_WINDOWS": "1",
                    "CONSECUTIVE_FAIL_WINDOWS": "1",
                },
                "compose_files": ["docker-compose.yml"],
                "results_dir": "/path/to/results",
                "container_name": "above_fps_target",
                "target_fps": 14.0,
                "expected_num_pipelines": 1,
                "expected_meet_target_fps": True,
                "expected_streams_sustained": 1,
                "calculate_fps_side_effect": [
                    (15.0, 15.0, {"pipeline_stream0": 15.0}, {}),
                    (10.0, 10.0, {"pipeline_stream0": 10.0}, {}),
                    (15.0, 15.0, {"pipeline_stream0": 15.0}, {}),
                ]
            },
            # Test case 2: fail at the minimum pipeline count.
            {
                "env_vars": {
                    "INIT_DURATION": "10",
                    "CONSECUTIVE_FAIL_WINDOWS": "1",
                },
                "compose_files": ["docker-compose.yml"],
                "results_dir": "/path/to/results",
                "container_name": "below_fps_target",
                "target_fps": 15.0,
                "expected_num_pipelines": 1,
                "expected_meet_target_fps": False,
                "expected_streams_sustained": 1,
                "calculate_fps_side_effect": [
                    (10.0, 10.0, {"pipeline_stream0": 10.0}, {}),
                    (10.0, 10.0, {"pipeline_stream0": 10.0}, {}),
                ]
            },
            {
                "env_vars": {"INIT_DURATION": "10"},
                "compose_files": ["docker-compose.yml"],
                "results_dir": "/path/to/results",
                "container_name": "insufficient_samples",
                "target_fps": 15.0,
                "expected_num_pipelines": 1,
                "expected_meet_target_fps": False,
                "expected_streams_sustained": 0,
                "measurement_window_result": (
                    {}, {}, 200.0, {"pipeline_stream0_lane1_gst.log": 99},
                    False),
                "calculate_fps_side_effect": [],
                "expect_no_fps_calculation": True,
            },
        ]

        for i, test_case in enumerate(test_cases):
            with self.subTest(f"Test case {i + 1}"):
                env_vars = test_case["env_vars"]
                compose_files = test_case["compose_files"]
                results_dir = test_case["results_dir"]
                container_name = test_case["container_name"]
                target_fps = test_case["target_fps"]

                mock_calculate_fps.reset_mock()
                mock_calculate_fps.side_effect = test_case[
                    "calculate_fps_side_effect"]
                mock_collect_window.return_value = test_case.get(
                    "measurement_window_result",
                    ({}, {}, 100.0,
                     {'pipeline_stream0_lane_gst.log': 100}, True))
                num_pipelines, meet_target_fps, streams_sustained = (
                    stream_density.run_pipeline_iterations(
                        env_vars, compose_files, results_dir,
                        container_name, target_fps)
                )

                self.assertEqual(
                    num_pipelines, test_case["expected_num_pipelines"])
                self.assertEqual(
                    meet_target_fps, test_case["expected_meet_target_fps"])
                self.assertEqual(
                    streams_sustained, test_case["expected_streams_sustained"])
                if test_case.get("expect_no_fps_calculation"):
                    mock_calculate_fps.assert_not_called()

    @patch('time.sleep', return_value=None)
    @patch('stream_density.validate_and_setup_env')
    @patch('stream_density.run_pipeline_iterations')
    @patch('stream_density.benchmark.docker_compose_containers')
    @patch('builtins.open', new_callable=mock_open)
    def test_run_stream_density(
        self,
        mock_open_file,
        mock_docker_compose,
        mock_run_pipeline_iterations,
        mock_validate_env,
        mock_sleep
    ):
        test_cases = [
            # Test case 1: Valid scenario where all parameters are correct
            {
                "env_vars": {RESULTS_DIR_KEY: "/some/path"},
                "compose_files": ["docker-compose.yml"],
                "target_fps_list": [15.0, 25.0],
                "container_names_list": ["container1", "container2"],
                "run_pipeline_side_effect": [
                    (5, True, 10),  # For container1
                    (7, False, 12)  # For container2
                ],
                "expected_results": [
                    (15.0, "container1", 5, True, 10),
                    (25.0, "container2", 7, False, 12)
                ],
                # Expected number of compose down calls
                "expected_down_call_count": 2
            },
            # Test case 2: Exception occurs during run_pipeline_iterations()
            {
                "env_vars": {RESULTS_DIR_KEY: "/some/path"},
                "compose_files": ["docker-compose.yml"],
                "target_fps_list": [15.0],
                "container_names_list": ["container1"],
                "run_pipeline_side_effect": Exception("Test exception"),
                "expect_exception": True,
                # Expected number of compose down calls
                # even if an exception occurs, it should call down
                "expected_down_call_count": 1
            }
        ]

        for i, test_case in enumerate(test_cases):
            with self.subTest(f"Test case {i + 1}"):
                mock_docker_compose.reset_mock()
                env_vars = test_case["env_vars"].copy()
                compose_files = test_case["compose_files"]
                target_fps_list = test_case["target_fps_list"]
                container_names_list = test_case["container_names_list"]

                # Mock the behavior of run_pipeline_iterations
                # based on the test case
                if isinstance(test_case["run_pipeline_side_effect"], list):
                    mock_run_pipeline_iterations.side_effect = test_case[
                        "run_pipeline_side_effect"]
                else:
                    mock_run_pipeline_iterations.side_effect = test_case[
                        "run_pipeline_side_effect"]

                # Run the function and verify results or exceptions
                if test_case.get("expect_exception"):
                    print('expecting exception test case')
                    with self.assertRaises(Exception) as context:
                        stream_density.run_stream_density(
                            env_vars, compose_files,
                            target_fps_list, container_names_list)
                    self.assertTrue(isinstance(context.exception, Exception))
                else:
                    results = stream_density.run_stream_density(
                        env_vars, compose_files,
                        target_fps_list, container_names_list)
                    self.assertEqual(results, test_case["expected_results"])

                # Verify that validate_and_setup_env was called correctly
                mock_validate_env.assert_called_with(
                    env_vars, target_fps_list
                )

                expected_down_call_count = test_case[
                    "expected_down_call_count"
                ]
                actual_down_calls = [
                    call for call in mock_docker_compose.call_args_list
                    if call[0][0] == 'down'
                ]
                self.assertEqual(
                    len(actual_down_calls),
                    expected_down_call_count,
                    f"Expected {expected_down_call_count} 'down' calls, "
                    f"but found {len(actual_down_calls)}"
                )


if __name__ == '__main__':
    unittest.main()
