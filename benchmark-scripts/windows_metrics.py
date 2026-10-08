"""Automatic Windows-host telemetry for WSL benchmarks."""

import atexit
import argparse
import csv
import datetime
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
from collections import defaultdict


GPU_FIELDS = (
    'Compute[CCS] Utilization %', 'Render/3D[RCS] Utilization %',
    'Video[VCS] Utilization %', 'VideoEnhance[VECS] Utilization %',
    'Blitter Copy Engine %', 'GPU Power (W)',
)


def blank_metrics():
    metrics = {f'GPU_1 {field}': 'NA' for field in GPU_FIELDS}
    metrics.update({'S0 Power Draw W': 'NA', 'S0 Memory Bandwidth Usage MB/s': 'NA'})
    return metrics


class GpuPowerReader:
    def __init__(self, dll):
        import clr
        library = Path(dll).resolve(strict=True)
        sys.path.append(str(library.parent))
        clr.AddReference(str(library))
        from LibreHardwareMonitor.Hardware import Computer
        self.computer = Computer()
        self.computer.IsCpuEnabled = True
        self.computer.IsGpuEnabled = True
        try:
            self.computer.Open()
        except Exception:
            self.computer.Close()
            raise

    def sensors(self):
        result = []
        for hardware in self.computer.Hardware:
            if not str(hardware.HardwareType).startswith('Gpu'):
                continue
            hardware.Update()
            for sensor in hardware.Sensors:
                if str(sensor.SensorType) != 'Power':
                    continue
                raw = sensor.Value
                value = float(raw) if raw is not None else None
                if value is not None and (not math.isfinite(value) or value < 0):
                    value = None
                result.append({'hardware': str(hardware.Name),
                               'hardware_id': str(hardware.Identifier),
                               'sensor': str(sensor.Name),
                               'sensor_id': str(sensor.Identifier), 'watts': value})
        return result

    def read(self, sensor_id):
        return next((sensor['watts'] for sensor in self.sensors()
                     if sensor['sensor_id'] == sensor_id), None)

    def close(self):
        self.computer.Close()


def gpu_sample(instances):
    engines = defaultdict(float)
    for name, value in instances.items():
        match = re.search(r'(luid_.+?_phys_\d+)_eng_(\d+)_engtype_(.+)', name)
        if not match or not math.isfinite(value) or value < 0:
            continue
        adapter, engine, kind = match.groups()
        kind = kind.lower()
        if kind.startswith('compute'):
            field = GPU_FIELDS[0]
        else:
            field = {'3d': GPU_FIELDS[1], 'videodecode': GPU_FIELDS[2],
                     'videoencode': GPU_FIELDS[2], 'videoprocessing': GPU_FIELDS[3],
                     'copy': GPU_FIELDS[4]}.get(kind)
        if field:
            engines[adapter, field, engine] += value
    result = {}
    for (adapter, field, engine), value in engines.items():
        result[adapter, field] = max(result.get((adapter, field), 0), min(value, 100))
    return result


def read_pcm(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as source:
        rows = csv.reader(source)
        groups = next(rows, [])
        # PCM labels a group once and leaves the remaining cells of it blank.
        current_group = ''
        for index, group in enumerate(groups):
            if group.strip():
                current_group = group.strip()
            groups[index] = current_group
        headers = next(rows, [])
        if headers[:2] != ['Date', 'Time']:
            return {}
        values = defaultdict(list)
        previous = None
        for row in rows:
            if len(row) < len(headers):
                continue
            try:
                timestamp = datetime.datetime.fromisoformat(f'{row[0]}T{row[1]}')
            except ValueError:
                continue
            interval = (timestamp - previous).total_seconds() if previous else 0
            previous = timestamp
            if interval <= 0:
                continue
            for index, (group, header) in enumerate(zip(groups, headers)):
                try:
                    if header == 'READ' and group.startswith('Socket '):
                        socket = int(group.split()[-1])
                        write_index = next(position for position, pair in enumerate(zip(groups, headers))
                                           if pair == (group, 'WRITE'))
                        read, write = float(row[index]), float(row[write_index])
                        if min(read, write) < 0:
                            continue
                        value = (read + write) * 1000
                        key = f'S{socket} Memory Bandwidth Usage MB/s'
                    elif group == 'Proc Energy (Joules)' and re.fullmatch(r'SKT\d+', header):
                        key = f'S{int(header[3:])} Power Draw W'
                        value = float(row[index]) / interval
                    else:
                        continue
                    if math.isfinite(value) and value >= 0:
                        values[key].append((value, interval))
                except (ValueError, StopIteration, IndexError):
                    continue
    return {key: round(sum(value * duration for value, duration in samples) /
                       sum(duration for value, duration in samples), 2)
            for key, samples in values.items()}


def collect(directory, stop, init_duration=0, pcm_exe=None,
            lhm_dll=None, power_sensor=None, power_adapter=None):
    directory = Path(directory)
    metrics = blank_metrics()
    adapters, samples = {}, defaultdict(list)
    observed = set()
    query, pcm, pcm_log = None, None, None
    power_reader = None
    try:
        if stop.wait(init_duration):
            return
        if lhm_dll:
            try:
                if not power_sensor or not re.fullmatch(r'luid_.+_phys_\d+', power_adapter or ''):
                    raise ValueError('GPU power requires WINDOWS_GPU_POWER_SENSOR and WINDOWS_GPU_POWER_ADAPTER')
                power_reader = GpuPowerReader(lhm_dll)
                power_reader.sensors()
                adapters[power_adapter] = 1
            except Exception as error:
                print(f'WARN: GPU power unavailable: {error}', flush=True)
                if power_reader is not None:
                    power_reader.close()
                power_reader = None
        try:
            import win32pdh
            query = win32pdh.OpenQuery()
            counter = win32pdh.AddEnglishCounter(
                query, r'\GPU Engine(*)\Utilization Percentage')
            counter_path = r'\GPU Engine(*)\Utilization Percentage'
            win32pdh.RemoveCounter(counter)
            counters = {}
        except Exception as error:
            print(f'WARN: GPU counters unavailable: {error}', flush=True)
            if query is not None:
                win32pdh.CloseQuery(query)
            query = None
        try:
            executable = pcm_exe or shutil.which('pcm.exe')
            if not executable:
                raise FileNotFoundError('pcm.exe not found; set WINDOWS_PCM_EXE to its Windows path')
            pcm_log = (directory / 'windows_pcm.log').open('w')
            pcm = subprocess.Popen(
                [executable, '1', f'-csv={directory / "windows_pcm_raw.log"}', '-nc'],
                stdin=subprocess.DEVNULL, stdout=pcm_log, stderr=subprocess.STDOUT)
        except OSError as error:
            print(f'WARN: Windows PCM unavailable: {error}', flush=True)
        while not stop.wait(1):
            if power_reader is not None:
                try:
                    watts = power_reader.read(power_sensor)
                    if watts is not None:
                        samples[f'GPU_{adapters[power_adapter]} GPU Power (W)'].append(watts)
                except Exception as error:
                    print(f'WARN: GPU power sample unavailable: {error}', flush=True)
            if query is None:
                continue
            try:
                paths = set(win32pdh.ExpandCounterPath(counter_path))
                ready = set(counters) & paths
                for path in set(counters) - paths:
                    win32pdh.RemoveCounter(counters.pop(path))
                for path in paths - set(counters):
                    counters[path] = win32pdh.AddCounter(query, path)
                win32pdh.CollectQueryData(query)
                instances = {}
                for path in sorted(ready):
                    try:
                        instances[path.split('(', 1)[1].rsplit(')', 1)[0]] = (
                            win32pdh.GetFormattedCounterValue(counters[path], win32pdh.PDH_FMT_DOUBLE)[1])
                    except Exception:
                        continue
                current = gpu_sample(instances)
                # Vanished instances mean the engine went idle, not missing data.
                for adapter, field in observed - current.keys():
                    samples[f'GPU_{adapters[adapter]} {field}'].append(0.0)
                for (adapter, field), value in current.items():
                    if adapter not in adapters:
                        adapters[adapter] = len(adapters) + 1
                    observed.add((adapter, field))
                    samples[f'GPU_{adapters[adapter]} {field}'].append(value)
            except Exception as error:
                print(f'WARN: GPU sample unavailable: {error}', flush=True)
    finally:
        if power_reader is not None:
            try:
                power_reader.close()
            except Exception as error:
                print(f'WARN: GPU power cleanup: {error}', flush=True)
        if query is not None:
            try:
                win32pdh.CloseQuery(query)
            except Exception as error:
                print(f'WARN: GPU counter cleanup: {error}', flush=True)
        if pcm is not None:
            try:
                if pcm.poll() is None:
                    pcm.terminate()
                pcm.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pcm.kill()
                pcm.wait(timeout=2)
            except OSError as error:
                print(f'WARN: PCM cleanup: {error}', flush=True)
        if pcm_log is not None:
            pcm_log.close()
        for number in adapters.values():
            metrics.update({f'GPU_{number} {field}': 'NA' for field in GPU_FIELDS})
        metrics.update({key: round(sum(values) / len(values), 2) for key, values in samples.items()})
        try:
            metrics.update(read_pcm(directory / 'windows_pcm_raw.log'))
        except (OSError, csv.Error) as error:
            print(f'WARN: PCM measurements unavailable: {error}', flush=True)
        output = {'metrics': metrics, 'adapters': adapters,
                  'gpu_source': 'Windows GPU Engine counters (host-wide)',
                  'gpu_aggregation': 'sum processes per engine, then busiest engine per category',
                  'gpu_power_source': ({'provider': 'LibreHardwareMonitor',
                                        'sensor_id': power_sensor, 'adapter': power_adapter}
                                       if power_reader is not None else None),
                  'pcm_source': 'native Windows PCM; package energy / sample interval; READ+WRITE GB/s * 1000'}
        temporary = directory / 'windows_metrics.json.tmp'
        temporary.write_text(json.dumps(output, indent=2), encoding='utf-8')
        temporary.replace(directory / 'windows_metrics.json')


class WindowsMetricsCollector:
    def __init__(self):
        self.process = None
        atexit.register(self.stop)

    def start(self, env_vars):
        if env_vars.get('WSL2', '').lower() != 'true':
            return
        self.stop()
        try:
            directory = Path(env_vars['RESULTS_DIR']).resolve()
            directory.mkdir(parents=True, exist_ok=True)
            output = directory / 'windows_metrics.json'
            output.write_text(json.dumps({'metrics': blank_metrics()}), encoding='utf-8')
            for name in ('windows_pcm_raw.log', 'windows_pcm.log', 'windows_metrics.json.tmp'):
                (directory / name).unlink(missing_ok=True)
            paths = []
            for path in (Path(__file__).resolve(), directory):
                paths.append(subprocess.check_output(
                    ['wslpath', '-w', str(path)], text=True, timeout=5).strip())
            executable = env_vars.get('WINDOWS_PYTHON', 'python.exe').strip()
            if not executable:
                raise ValueError('WINDOWS_PYTHON must name a Windows Python executable')
            command = [executable, '-u',
                       paths[0], '--output-dir', paths[1]]
            if env_vars.get('WINDOWS_PCM_EXE'):
                command.extend(['--pcm-exe', env_vars['WINDOWS_PCM_EXE']])
            for variable, option in (
                    ('WINDOWS_LHM_DLL', '--lhm-dll'),
                    ('WINDOWS_GPU_POWER_SENSOR', '--gpu-power-sensor'),
                    ('WINDOWS_GPU_POWER_ADAPTER', '--gpu-power-adapter')):
                if env_vars.get(variable):
                    command.extend([option, env_vars[variable]])
            command.extend(['--init-duration', env_vars.get('INIT_DURATION', '0')])
            with (directory / 'windows_metrics.log').open('w') as log:
                self.process = subprocess.Popen(
                    command, stdin=subprocess.PIPE, stdout=log,
                    stderr=subprocess.STDOUT, env=env_vars)
            print(f'Windows hardware collector started: {directory}')
        except (OSError, subprocess.SubprocessError, KeyError, ValueError) as error:
            print(f'WARN: Windows hardware collection unavailable: {error}')

    def stop(self):
        process, self.process = self.process, None
        if process is None:
            return
        try:
            process.stdin.close()
            process.wait(timeout=15)
            if process.returncode:
                print('WARN: Windows hardware collector failed; see windows_metrics.log')
        except (OSError, subprocess.SubprocessError) as error:
            print(f'WARN: Windows hardware collector shutdown: {error}')
            try:
                process.kill()
                process.wait(timeout=5)
            except (OSError, subprocess.SubprocessError):
                pass


collector = WindowsMetricsCollector()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--init-duration', type=float, default=0)
    parser.add_argument('--pcm-exe')
    parser.add_argument('--lhm-dll')
    parser.add_argument('--gpu-power-sensor')
    parser.add_argument('--gpu-power-adapter')
    parser.add_argument('--list-gpu-power-sensors', action='store_true')
    args = parser.parse_args()
    if os.name != 'nt':
        parser.error('This worker must run under Windows Python, launched by the WSL benchmark')
    if args.list_gpu_power_sensors:
        if not args.lhm_dll:
            parser.error('--list-gpu-power-sensors requires --lhm-dll')
        reader = GpuPowerReader(args.lhm_dll)
        try:
            reader.sensors()
            threading.Event().wait(1)
            sensors = reader.sensors()
            directory = Path(args.output_dir)
            directory.mkdir(parents=True, exist_ok=True)
            destination = directory / 'windows_gpu_power_sensors.json'
            destination.write_text(json.dumps(sensors, indent=2), encoding='utf-8')
            print(json.dumps(sensors, indent=2))
        finally:
            reader.close()
        sys.exit(0)
    stopped = threading.Event()

    def wait_for_parent():
        sys.stdin.buffer.read()
        stopped.set()

    threading.Thread(target=wait_for_parent, daemon=True).start()
    collect(args.output_dir, stopped, max(0, args.init_duration), args.pcm_exe,
            args.lhm_dll, args.gpu_power_sensor, args.gpu_power_adapter)