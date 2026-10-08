'''
* Copyright (C) 2024 Intel Corporation.
*
* SPDX-License-Identifier: Apache-2.0
'''

import unittest.mock as mock
import subprocess  # nosec B404
import unittest
import benchmark
import os
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import windows_metrics


class WindowsMetricsTesting(unittest.TestCase):
    def test_configured_python_path_with_spaces(self):
        executable = '/mnt/c/Program Files/Python311/python.exe'
        collector = windows_metrics.WindowsMetricsCollector()
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(windows_metrics.subprocess, 'check_output',
                                  side_effect=['C:\\helper.py', 'C:\\results']) as paths, \
                mock.patch.object(windows_metrics.subprocess, 'Popen') as launch:
            launch.return_value.returncode = 0
            collector.start({'WSL2': 'true', 'RESULTS_DIR': directory,
                             'WINDOWS_PYTHON': executable})
            self.assertEqual(launch.call_args.args[0][0], executable)
            self.assertEqual(paths.call_count, 2)
            collector.stop()

    def test_configured_python_failure_does_not_try_fallback(self):
        collector = windows_metrics.WindowsMetricsCollector()
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(windows_metrics.subprocess, 'check_output',
                                  side_effect=['C:\\helper.py', 'C:\\results']) as paths, \
                mock.patch.object(windows_metrics.subprocess, 'Popen',
                                  side_effect=FileNotFoundError()) as launch:
            collector.start({'WSL2': 'true', 'RESULTS_DIR': directory,
                             'WINDOWS_PYTHON': '/missing/python.exe'})
            launch.assert_called_once()
            self.assertEqual(paths.call_count, 2)
            self.assertIsNone(collector.process)

    def test_gpu_power_rejects_cpu_and_preserves_missing_values(self):
        reader = windows_metrics.GpuPowerReader.__new__(windows_metrics.GpuPowerReader)
        sensor = SimpleNamespace(SensorType='Power', Value=None,
                                 Name='GPU Power', Identifier='/gpu/power/0')
        cpu_sensor = SimpleNamespace(SensorType='Power', Value=100,
                                     Name='CPU Package', Identifier='/cpu/power/0')
        reader.computer = SimpleNamespace(Hardware=[
            SimpleNamespace(HardwareType='Cpu', Sensors=[cpu_sensor]),
            SimpleNamespace(HardwareType='GpuIntel', Name='GPU', Identifier='/gpu',
                            Sensors=[sensor], Update=lambda: None)])
        self.assertIsNone(reader.read('/cpu/power/0'))
        self.assertIsNone(reader.read('/gpu/power/0'))
        sensor.Value = 0
        self.assertEqual(reader.read('/gpu/power/0'), 0)
        sensor.Value = 12.5
        self.assertEqual(reader.read('/gpu/power/0'), 12.5)
        sensor.Value = float('nan')
        self.assertIsNone(reader.read('/gpu/power/0'))

    def test_gpu_power_without_activity_counters(self):
        stop = mock.MagicMock()
        stop.wait.side_effect = [False, False, True]
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(windows_metrics, 'GpuPowerReader') as provider, \
                mock.patch.dict('sys.modules', {'win32pdh': None}), \
                mock.patch.object(windows_metrics.shutil, 'which', return_value=None):
            provider.return_value.read.return_value = 12.5
            windows_metrics.collect(directory, stop, lhm_dll='library.dll',
                                    power_sensor='/gpu/power/0',
                                    power_adapter='luid_0x0_0x1_phys_0')
            metrics = json.loads((Path(directory) / 'windows_metrics.json').read_text())['metrics']
            self.assertEqual(metrics['GPU_1 GPU Power (W)'], 12.5)
            self.assertEqual(metrics['S0 Power Draw W'], 'NA')
            provider.return_value.close.assert_called_once()

    def test_gpu_engines(self):
        counters = {
            f'pid_{process}_luid_0x0_0x1_phys_0_eng_{engine}_engtype_{kind}': value
            for process, engine, kind, value in (
                (1, 0, '3D', 20), (2, 0, '3D', 30), (1, 1, '3D', 40),
                (1, 2, 'Compute_0', 0), (1, 3, 'Copy', float('nan')))
        }
        result = windows_metrics.gpu_sample(counters)
        self.assertEqual(result, {
            ('luid_0x0_0x1_phys_0', 'Render/3D[RCS] Utilization %'): 50,
            ('luid_0x0_0x1_phys_0', 'Compute[CCS] Utilization %'): 0})

    def test_pcm_units(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'windows_pcm_raw.log'
            source.write_text(
                'System,,Socket 0,,Proc Energy (Joules)\n'
                'Date,Time,READ,WRITE,SKT0\n'
                '2026-09-21,10:00:00,1,2,20\n'
                '2026-09-21,10:00:02,2,3,40\n')
            self.assertEqual(windows_metrics.read_pcm(source), {
                'S0 Memory Bandwidth Usage MB/s': 5000,
                'S0 Power Draw W': 20})

    def test_native_noop_and_wsl_launch_failure(self):
        collector = windows_metrics.WindowsMetricsCollector()
        with mock.patch.object(windows_metrics.subprocess, 'Popen') as launch:
            collector.start({'WSL2': 'false'})
            launch.assert_not_called()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'windows_metrics.json'
            output.write_text('{"metrics":{"stale":100}}')
            with mock.patch.object(windows_metrics.subprocess, 'check_output',
                                   side_effect=FileNotFoundError('wslpath')):
                collector.start({'WSL2': 'true', 'RESULTS_DIR': directory})
            self.assertIsNone(collector.process)
            self.assertEqual(json.loads(output.read_text())['metrics'], windows_metrics.blank_metrics())

    def test_worker_missing_pcm_and_invalid_gpu_sample(self):
        pdh = mock.MagicMock(PDH_FMT_DOUBLE=512)
        pdh.ExpandCounterPath.return_value = [
            r'\GPU Engine(pid_1_luid_0x0_0x1_phys_0_eng_0_engtype_3D)\Utilization Percentage']
        pdh.GetFormattedCounterValue.side_effect = [RuntimeError('invalid'), (0, 25)]
        stop = mock.MagicMock()
        stop.wait.side_effect = [False, False, False, False, True]
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.dict('sys.modules', {'win32pdh': pdh}), \
                mock.patch.object(windows_metrics.shutil, 'which', return_value=None):
            windows_metrics.collect(directory, stop)
            metrics = json.loads((Path(directory) / 'windows_metrics.json').read_text())['metrics']
        self.assertEqual(metrics['GPU_1 Render/3D[RCS] Utilization %'], 25)
        self.assertEqual(metrics['GPU_1 GPU Power (W)'], 'NA')
        self.assertEqual(metrics['S0 Power Draw W'], 'NA')
        self.assertEqual(pdh.GetFormattedCounterValue.call_count, 2)
        pdh.ExpandCounterPath.assert_called_with(
            r'\GPU Engine(*)\Utilization Percentage')
        pdh.GetCounterInfo.assert_not_called()

    def test_vanished_gpu_instances_sample_as_idle(self):
        path = (r'\GPU Engine(pid_1_luid_0x0_0x1_phys_0_eng_0_engtype_3D)'
                r'\Utilization Percentage')
        pdh = mock.MagicMock(PDH_FMT_DOUBLE=512)
        pdh.ExpandCounterPath.side_effect = [[path], [path], []]
        pdh.GetFormattedCounterValue.return_value = (0, 40)
        stop = mock.MagicMock()
        stop.wait.side_effect = [False, False, False, False, True]
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.dict('sys.modules', {'win32pdh': pdh}), \
                mock.patch.object(windows_metrics.shutil, 'which', return_value=None):
            windows_metrics.collect(directory, stop)
            metrics = json.loads((Path(directory) / 'windows_metrics.json').read_text())['metrics']
        self.assertEqual(metrics['GPU_1 Render/3D[RCS] Utilization %'], 20)

    def test_compose_lifecycle(self):
        with mock.patch.object(benchmark, 'windows_collector') as collector, \
                mock.patch.object(benchmark.subprocess, 'Popen') as launch:
            launch.return_value.communicate.return_value = (b'ok', b'')
            launch.return_value.returncode = 0
            env = {'WSL2': 'true', 'RESULTS_DIR': '/benchmark'}
            benchmark.docker_compose_containers('up', env_vars=env)
            collector.start.assert_called_once_with(env)
            benchmark.docker_compose_containers('down', env_vars=env)
            collector.stop.assert_called_once()
            collector.reset_mock()
            launch.return_value.returncode = 1
            benchmark.docker_compose_containers('up', env_vars=env)
            collector.start.assert_not_called()

    def test_native_compose_never_calls_collector(self):
        with mock.patch.object(benchmark, 'windows_collector') as collector, \
                mock.patch.object(benchmark.subprocess, 'Popen') as launch:
            launch.return_value.communicate.return_value = (b'ok', b'')
            launch.return_value.returncode = 0
            for env in ({}, {'WSL2': 'false'}, {'WSL': 'true'}):
                benchmark.docker_compose_containers('up', env_vars=env)
                benchmark.docker_compose_containers('down', env_vars=env)
            collector.start.assert_not_called()
            collector.stop.assert_not_called()


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

if __name__ == '__main__':
    unittest.main()
