'''
* Copyright (C) 2024 Intel Corporation.
*
* SPDX-License-Identifier: Apache-2.0
'''

import mock
import subprocess  # nosec B404
import unittest
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


class Testing(unittest.TestCase):
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

    def test_calculate_total_fps_success(self):
        test_results_dir = './test_stream_density_results'
        try:
            fps, avg_fps = stream_density.calculate_total_fps(
                2, test_results_dir, 'gst')
            self.assertTrue(
                fps > 0.0,
                f"total_fps is expected > 0.0 but found {fps}")
            self.assertTrue(
                avg_fps > 0.0,
                f"total_fps_per_stream is expected > 0.0 but found "
                f"{avg_fps}")
        except Exception as ex:
            self.fail(f'ERROR: got exception {type(ex).__name__}')

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
                    INIT_DURATION_KEY: "120"
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
                    INIT_DURATION_KEY: "120"
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
            # should default to "120"
            {
                "env_vars": {RESULTS_DIR_KEY: "/some/path"},
                "target_fps_list": [20.0],
                "expect_exception": False,
                "expected_target_fps_list": [20.0],
                "expected_env_vars": {
                    RESULTS_DIR_KEY: "/some/path",
                    INIT_DURATION_KEY: "120"
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
    @patch('benchmark.docker_compose_containers')
    @patch('stream_density.calculate_total_fps')
    @patch('stream_density.check_non_empty_result_logs')
    @patch('stream_density.clean_up_pipeline_logs')
    def test_pipeline_iterations(
        self,
        mock_clean_logs,
        mock_check_logs,
        mock_calculate_fps,
        mock_docker_compose,
        mock_sleep
    ):
        test_cases = [
            # Test case 1: Succeed on the first iteration
            # with target_fps achieved
            {
                "env_vars": {"INIT_DURATION": "10"},
                "compose_files": ["docker-compose.yml"],
                "results_dir": "/path/to/results",
                "container_name": "above_fps_target",
                "target_fps": 14.0,
                "expected_num_pipelines": 10,
                "expected_meet_target_fps": True,
                # Mock returns FPS below target
                "calculate_fps_side_effect": [
                    (50, 20.0), (70, 16.0), (100, 12.0), (120, 14.8)
                ]
            },
            # Test case 2: Reach minimum pipeline count
            # without achieving target_fps
            {
                "env_vars": {"INIT_DURATION": "10"},
                "compose_files": ["docker-compose.yml"],
                "results_dir": "/path/to/results",
                "container_name": "below_fps_target",
                "target_fps": 15.0,
                "expected_num_pipelines": 1,
                "expected_meet_target_fps": False,
                # Mock returns FPS below target
                "calculate_fps_side_effect": [(100, 10.0), (50, 5.0)]
            },
            # Test case 3: Below target_fps and come back
            {
                "env_vars": {"INIT_DURATION": "10", "PIPELINE_INC": "1"},
                "compose_files": ["docker-compose.yml"],
                "results_dir": "/path/to/results",
                "container_name": "below_comeback_target",
                "target_fps": 14.0,
                "expected_num_pipelines": 1,
                "expected_meet_target_fps": True,
                # Mock returns FPS below target
                "calculate_fps_side_effect": [
                    (100, 60.0), (190, 10.0), (320, 14.2)
                ]
            },
            # Test case 4: Below target_fps and reach minimum
            {
                "env_vars": {"INIT_DURATION": "10", "PIPELINE_INC": "1"},
                "compose_files": ["docker-compose.yml"],
                "results_dir": "/path/to/results",
                "container_name": "below_comeback_target",
                "target_fps": 14.0,
                "expected_num_pipelines": 1,
                "expected_meet_target_fps": False,
                # Mock returns FPS below target
                "calculate_fps_side_effect": [
                    (100, 60.0), (190, 10.0), (220, 13.2)
                ]
            },
            # Test case 5: check_non_empty_result_logs raises ValueError
            {
                "env_vars": {"INIT_DURATION": "10"},
                "compose_files": ["docker-compose.yml"],
                "results_dir": "/path/to/results",
                "container_name": "value_error",
                "target_fps": 14.0,
                "check_logs_side_effect": ValueError(
                    "Expecting ValueError for check_non_empty_result_logs"),
                "expected_num_pipelines": 1,
                "expected_meet_target_fps": False
            },
        ]

        for i, test_case in enumerate(test_cases):
            with self.subTest(f"Test case {i + 1}"):
                env_vars = test_case["env_vars"]
                compose_files = test_case["compose_files"]
                results_dir = test_case["results_dir"]
                container_name = test_case["container_name"]
                target_fps = test_case["target_fps"]

                # Set up the side effect for
                # check_non_empty_result_logs if specified
                if "check_logs_side_effect" in test_case:
                    mock_check_logs.side_effect = test_case[
                        "check_logs_side_effect"]

                # Set up the side effect for
                # calculate_total_fps using a finite list
                if "calculate_fps_side_effect" in test_case:
                    mock_calculate_fps.side_effect = test_case[
                        "calculate_fps_side_effect"]

                # Run the function with the test case inputs
                if "check_logs_side_effect" in test_case:
                    num_pipelines, meet_target_fps = (
                        stream_density.run_pipeline_iterations(
                            env_vars, compose_files, results_dir,
                            container_name, target_fps)
                    )

                    # Verify the returned values after ValueError is handled
                    self.assertEqual(
                        num_pipelines,
                        test_case["expected_num_pipelines"])
                    self.assertEqual(
                        meet_target_fps,
                        test_case["expected_meet_target_fps"])

                    # Ensure that check_non_empty_result_logs was
                    # called and raised ValueError
                    mock_check_logs.assert_called()
                else:
                    num_pipelines, meet_target_fps = (
                        stream_density.run_pipeline_iterations(
                            env_vars, compose_files, results_dir,
                            container_name, target_fps)
                    )

                    # Verify the output
                    self.assertEqual(
                        num_pipelines,
                        test_case["expected_num_pipelines"])
                    self.assertEqual(
                        meet_target_fps,
                        test_case["expected_meet_target_fps"])

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
                    (5, True),  # For container1
                    (7, False)  # For container2
                ],
                "expected_results": [
                    (15.0, "container1", 5, True),
                    (25.0, "container2", 7, False)
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
