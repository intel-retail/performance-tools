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

    def _result_output(self, status, num_pipelines, streams):
        output = io.StringIO()
        with redirect_stdout(output):
            benchmark.print_stream_density_result(
                status, num_pipelines, streams, '/results')
        return output.getvalue()

    def test_print_stream_density_result_pass(self):
        output = self._result_output(stream_density.STATUS_PASS, 11, 22)
        self.assertIn('Result: Pass', output)
        self.assertIn(
            'use case density 11 lanes, stream density 22 streams', output)
        self.assertNotIn('sustained', output)

    def test_print_stream_density_result_fail(self):
        output = self._result_output(stream_density.STATUS_FAIL, 1, 2)
        self.assertIn('Result: Fail', output)
        self.assertIn('did not meet the target', output)

    def test_print_stream_density_result_inconclusive(self):
        output = self._result_output(stream_density.STATUS_INCONCLUSIVE, 10, 20)
        self.assertIn('Result: Inconclusive', output)
        self.assertNotIn('Result: Fail', output)
        self.assertIn(
            'Last passing density: use case density 10 lanes, '
            'stream density 20 streams', output)

    def test_print_stream_density_result_inconclusive_without_pass(self):
        output = self._result_output(stream_density.STATUS_INCONCLUSIVE, 0, 0)
        self.assertIn('Result: Inconclusive', output)
        self.assertIn('No measurement interval had passed.', output)
        self.assertNotIn('Last passing density', output)

if __name__ == '__main__':
    unittest.main()
