'''
* Copyright (C) 2024 Intel Corporation.
*
* SPDX-License-Identifier: Apache-2.0
'''

import mock
import subprocess
import unittest
import benchmark


class Testing(unittest.TestCase):

    class MockPopen(object):
        def __init__(self):
            pass
        def communicate(self, input=None):
            pass
        @property
        def returncode(self):
            pass

    def test_start_camera_simulator_success(self):
        mock_popen = Testing.MockPopen()
        mock_popen.communicate = mock.Mock(return_value=('1Starting camera: rtsp://127.0.0.1:8554/camera_0 from *.mp4', ''))
        mock_returncode = mock.PropertyMock(return_value=0)
        type(mock_popen).returncode = mock_returncode

        setattr(subprocess, 'Popen', lambda *args, **kargs: mock_popen)
        res = benchmark.start_camera_simulator()

        self.assertEqual(res, ('1Starting camera: rtsp://127.0.0.1:8554/camera_0 from *.mp4', '', 0))
        mock_popen.communicate.assert_called_once_with()
        mock_returncode.assert_called_once_with()

    def test_start_camera_simulator_fail(self):
        mock_popen = Testing.MockPopen()
        mock_popen.communicate = mock.Mock(return_value=('', b'an error occurred'))
        mock_returncode = mock.PropertyMock(return_value=1)
        type(mock_popen).returncode = mock_returncode

        setattr(subprocess, 'Popen', lambda *args, **kargs: mock_popen)
        res = benchmark.start_camera_simulator()

        self.assertEqual(res, ('', b'an error occurred', 1))
        mock_popen.communicate.assert_called_once_with()
        mock_returncode.assert_called_once_with()

if __name__ == '__main__':
    unittest.main()