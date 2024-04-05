'''
* Copyright (C) 2024 Intel Corporation.
*
* SPDX-License-Identifier: Apache-2.0
'''

import mock
import subprocess  # nosec B404
import unittest
import stream_density
import os


class Testing(unittest.TestCase):
    def test_is_env_non_empty(self):
        sys_env = os.environ.copy()
        sys_env["EMPTY"]=""
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
                1,'./non-existing-results')
            self.fail('expected ValueError exception')
        except ValueError as ex:
            self.assertTrue("""ERROR: cannot find all pipeline log files
                    after max retries""" in str(ex))
        # 1 file only but 2 pipelines:
        test_results_dir='./test_results'
        testFile = os.path.join(test_results_dir, 
              'pipeline12345656_abc.log')
        try:
            os.makedirs(test_results_dir)
            # create a non empty temporary log file
            with open(testFile, 'w') as file:
                file.write('this is a test')
            stream_density.check_non_empty_result_logs(
                2, test_results_dir)
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
        test_results_dir='./test_results'
        testFile1 = os.path.join(test_results_dir, 
              'pipeline12345656_abc.log')
        testFile2 = os.path.join(test_results_dir,
              'pipeline98765432_def.log')
        try:
            os.makedirs(test_results_dir)
            # create two non empty temporary log files
            with open(testFile1, 'w') as file:
                file.write('this is a test')
            with open(testFile2, 'w') as file:
                file.write('another file for testing')

            stream_density.check_non_empty_result_logs(
                2, test_results_dir)
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
        test_results_dir='./test_stream_density_results'
        try:
            fps, avg_fps = stream_density.calculate_total_fps(
                2, test_results_dir)
            self.assertTrue(fps > 0.0,
              f"total_fps is expected > 0.0 but found {fps}")
            self.assertTrue(avg_fps > 0.0,
              f"total_fps_per_stream is expected > 0.0 but found "
              f"{avg_fps}")
        except Exception as ex:
            self.fail(f'ERROR: got exception {type(ex).__name__}')


    def test_clean_up_pipeline_logs(self):
        test_results_dir='./test_results_clean'
        testFile1 = os.path.join(test_results_dir, 
              'pipeline12345656_abc.log')
        testFile2 = os.path.join(test_results_dir,
              'pipeline98765432_def.log')
        try:
            os.makedirs(test_results_dir)
            with open(testFile1, 'w') as file:
                file.write('this is a test')
            with open(testFile2, 'w') as file:
                file.write('another file for testing')
            stream_density.clean_up_pipeline_logs(
                test_results_dir)
            self.assertFalse(os.path.exists(testFile1), 
                f"file still exists: {testFile1}")
            self.assertFalse(os.path.exists(testFile2), 
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


if __name__ == '__main__':
    unittest.main()
