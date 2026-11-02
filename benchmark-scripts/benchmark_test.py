'''
* Copyright (C) 2024 Intel Corporation.
*
* SPDX-License-Identifier: Apache-2.0
'''

import unittest.mock as mock
import subprocess  # nosec B404
import unittest
import benchmark
import stream_density
import io
import os
from contextlib import redirect_stdout
from types import SimpleNamespace
import tempfile


class Testing(unittest.TestCase):

    class MockPopen(object):
        def __init__(self):
            pass

        def communicate(self, input=None):
            pass

        @property
        def returncode(self):
            pass

    def test_docker_compose_containers_success(self):
        mock_popen = Testing.MockPopen()
        mock_popen.communicate = mock.Mock(
            return_value=('', '1Starting camera: rtsp://127.0.0.1:8554/' +
                          'camera_0 from *.mp4'))
        mock_returncode = mock.PropertyMock(return_value=0)
        type(mock_popen).returncode = mock_returncode

        setattr(subprocess, 'Popen', lambda *args, **kargs: mock_popen)
        res = benchmark.docker_compose_containers('up')

        self.assertEqual(res, ('',
                               '1Starting camera: rtsp://127.0.0.1:8554/' +
                               'camera_0 from *.mp4', 0))
        mock_popen.communicate.assert_called_once_with()
        mock_returncode.assert_called()

    def test_docker_compose_containers_fail(self):
        mock_popen = Testing.MockPopen()
        mock_popen.communicate = mock.Mock(return_value=('',
                                                         'an error occurred'))
        mock_returncode = mock.PropertyMock(return_value=1)
        type(mock_popen).returncode = mock_returncode

        setattr(subprocess, 'Popen', lambda *args, **kargs: mock_popen)
        res = benchmark.docker_compose_containers('up')

        self.assertEqual(res, ('', 'an error occurred', 1))
        mock_popen.communicate.assert_called_once_with()
        mock_returncode.assert_called()

    def _result_output(self, status, num_pipelines, streams, limit_note=None):
        output = io.StringIO()
        with redirect_stdout(output):
            benchmark.print_stream_density_result(
                status, num_pipelines, streams, '/results', limit_note)
        return output.getvalue()

    def test_print_stream_density_result_pass(self):
        output = self._result_output(stream_density.STATUS_PASS, 11, 22)
        self.assertIn('Result: Pass', output)
        self.assertIn(
            'use case density 11 lanes, stream density 22 streams', output)
        self.assertNotIn('sustained', output)
        self.assertNotIn('Note:', output)

    def test_print_stream_density_result_pass_with_limit_note(self):
        output = self._result_output(
            stream_density.STATUS_PASS, 3, 6,
            'available memory set the limit; one more lane beyond '
            '3 lane(s) did not fit')
        self.assertIn('Result: Pass', output)
        self.assertIn(
            'use case density 3 lanes, stream density 6 streams', output)
        self.assertIn(
            'Note: available memory set the limit; one more lane beyond '
            '3 lane(s) did not fit.', output)

    def test_print_stream_density_result_fail(self):
        output = self._result_output(stream_density.STATUS_FAIL, 1, 2)
        self.assertIn('Result: Fail', output)
        self.assertIn('did not meet the target', output)

    def test_failed_sweep_does_not_publish_unconfirmed_density(self):
        output = self._result_output(stream_density.STATUS_FAIL, 10, 20)
        self.assertIn('Result: Fail', output)
        self.assertNotIn('Stream density completed', output)
        self.assertNotIn('stream density 20 streams', output)

    def test_failed_sweep_does_not_publish_limit_note(self):
        output = self._result_output(
            stream_density.STATUS_FAIL, 0, 0, 'unconfirmed memory limit')
        self.assertIn('Result: Fail', output)
        self.assertNotIn('Note:', output)

    def test_main_captures_metadata_around_density_sweep(self):
        with tempfile.TemporaryDirectory() as results_dir:
            args = SimpleNamespace(
                target_fps=[30], container_names=['gst'], results_dir=results_dir,
                compose_file=['compose.yml'], benchmark_type='default',
                target_device='CPU', retail_use_case_root='.',
                measurement_window_seconds=100, init_duration=10,
                density_increment=1, parser_script='parser.py', parser_args='')
            calls = mock.Mock()
            calls.capture.return_value = {'started_at': 'test'}
            calls.sweep.return_value = [(30, 'gst', 2, stream_density.STATUS_PASS, 2, None)]
            with mock.patch.object(benchmark, 'parse_args', return_value=args), \
                    mock.patch.object(benchmark, 'capture_run', calls.capture), \
                    mock.patch.object(benchmark, 'finish_run', calls.finish), \
                    mock.patch.object(stream_density, 'run_stream_density', calls.sweep), \
                    mock.patch.object(subprocess, 'run'), redirect_stdout(io.StringIO()):
                benchmark.main()
            self.assertEqual([entry[0] for entry in calls.mock_calls],
                             ['capture', 'sweep', 'finish'])
            calls.finish.assert_called_once_with(results_dir, {'started_at': 'test'})
            self.assertEqual(calls.capture.call_args.args[3]['target_fps'], [30])

if __name__ == '__main__':
    unittest.main()
