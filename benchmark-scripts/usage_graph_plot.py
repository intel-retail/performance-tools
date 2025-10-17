import csv
import json
import matplotlib.pyplot as plt
import os
import subprocess
import argparse
import glob

MAX_POINTS = 180

def downsample(x, y, max_points=MAX_POINTS):
    step = max(1, len(x) // max_points)
    return x[::step], y[::step]

def plot_cpu_usage(ax, filepath):
    cpu_usage = []
    time_seconds = []
    if os.path.isfile(filepath) and os.path.getsize(filepath) > 0:
        with open(filepath, 'r') as file:
            for line in file:
                parts = line.split()
                if len(parts) >= 8 and parts[1] == 'all':
                    try:
                        idle = float(parts[7])
                        usage = 100 - idle
                        cpu_usage.append(usage)
                        time_seconds.append(len(cpu_usage) - 1)
                    except ValueError:
                        continue

        if cpu_usage:
            time_ds, usage_ds = downsample(time_seconds, cpu_usage)
            ax.plot(time_ds, usage_ds, marker='o', color='blue', label='CPU Usage %')
            ax.set_title('CPU Usage Over Time')
            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel('Usage (%)')
            ax.set_ylim(bottom=0)
            ax.margins(x=0)
            ax.grid(True)
            ax.legend()
    else:
        ax.text(0.5, 0.5, "No CPU data", ha='center', va='center' , fontsize=12)
        ax.axis('off')

def plot_npu_usage(ax, filepath):
    usage_values = []
    if os.path.isfile(filepath) and os.path.getsize(filepath) > 0:
        with open(filepath, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                try:
                    usage = float(row['percent_usage'])
                    usage_values.append(usage)
                except ValueError:
                    continue

        time_intervals = list(range(len(usage_values)))
        if usage_values:
            time_ds, usage_ds = downsample(time_intervals, usage_values)
            ax.plot(time_ds, usage_ds, color='darkorange', marker='o', linestyle='-', linewidth=2, label='NPU Usage (%)')
            ax.set_title('NPU Usage Over Time')
            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel('Usage (%)')
            ax.set_ylim(bottom=0)
            ax.margins(x=0)
            ax.grid(True)
            ax.legend()
    else:
        ax.text(0.5, 0.5, "No NPU data", ha='center', va='center' , fontsize=12)
        ax.axis('off')

def plot_gpu_metrics(ax, filepath):
    desc_map = {
        'CCS %': 'Compute[CCS]',
        'RCS %': 'Render/3D[RCS]',
        'VCS %': 'Video[VCS]',
        'VECS %': 'VideoEnhance[VECS]',
        'Power Usage (W)': 'Power (W)',
        'BCS %': 'Blitter Copy Engine %'
    }

    metric_series = {metric: [] for metric in desc_map}
    time_series = []

    if os.path.isfile(filepath) and os.path.getsize(filepath) > 0:
        with open(filepath, 'r') as f:
            data = json.load(f)

        for i, entry in enumerate(data):
            time_series.append(i)
            for metric in desc_map:
                try:
                    val = float(str(entry.get(metric, '0')).replace('%', '').strip())
                except ValueError:
                    val = 0.0
                metric_series[metric].append(val)

        step = max(1, len(time_series) // MAX_POINTS)
        time_series_ds = time_series[::step]

        styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 10))]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c','#d62728', '#9467bd', '#8c564b']
        markers = ['o', 's', '^', 'x', 'D', '*']

        for idx, (metric, values) in enumerate(metric_series.items()):
            values_ds = values[::step]
            ax.plot(
                time_series_ds, values_ds,
                label=desc_map[metric],
                linestyle=styles[idx % len(styles)],
                color=colors[idx % len(colors)],
                marker=markers[idx % len(markers)],
                markersize=3,
                linewidth=1.5
            )

        # Get device ID from filename like "igt0-7D55.json" → "7D55"
        basename = os.path.basename(filepath)
        parts = basename.split('-')
        if len(parts) >= 3:
            device_part = parts[1]  # Second part is the device ID
        else:
            device_part = "unknown"
        ax.set_title(f'GPU Usage Over Time (device={device_part})')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Usage (%)')
        ax.set_ylim(bottom=0)
        ax.margins(x=0)
        ax.grid(True)
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No GPU data", ha='center', va='center', fontsize=12)
        ax.axis('off')

def plot_memory_usage(ax, filepath):
    
    memory_usage = []
    time_seconds = []

    if os.path.isfile(filepath) and os.path.getsize(filepath) > 0:
        with open(filepath, 'r') as file:
            for line in file:
                if line.strip().startswith('Mem:'):
                    parts = line.split()
                    if len(parts) >= 7:
                        try:
                            total_mem = float(parts[1])
                            used_mem = float(parts[2])
                            usage = (used_mem / total_mem) * 100
                            memory_usage.append(usage)
                            time_seconds.append(len(memory_usage) - 1)
                        except (ValueError, ZeroDivisionError):
                            continue

        if memory_usage:
            time_ds, usage_ds = downsample(time_seconds, memory_usage)
            ax.plot(time_ds, usage_ds, marker='o', color='green', label='Memory Usage %', linewidth=2)
            ax.set_title('Memory Usage Over Time')
            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel('Usage (%)')
            ax.set_ylim(bottom=0)
            ax.margins(x=0)
            ax.grid(True)
            ax.legend()
    else:
        ax.text(0.5, 0.5, "No Memory data", ha='center', va='center', fontsize=12)
        ax.axis('off')

def main():
    parser = argparse.ArgumentParser(description="Generate single consolidated plot for CPU, NPU, and all GPU usage.")
    parser.add_argument('--dir', type=str, default='.', help='Root directory containing logs')
    args = parser.parse_args()

    root = os.path.abspath(args.dir)
    cpu_log = os.path.join(root, 'cpu_usage.log')
    npu_csv = os.path.join(root, 'npu_usage.csv')
    gpu_files = sorted(glob.glob(os.path.join(root, 'qmassa*parsed.json')))
    memory_log = os.path.join(root, 'memory_usage.log')

    if not gpu_files:
        print("No GPU files found (qmassa*parsed.json)")
        # Still create a plot with CPU, NPU, and a "No GPU data" message
        total_plots = 4  # CPU + NPU + 1 GPU placeholder
    else:
        total_plots = 3 + len(gpu_files)  # CPU + NPU + each GPU
        
    fig, axs = plt.subplots(total_plots, 1, figsize=(20, total_plots * 4))  # Wide + Tall

    print("📊 Generating single consolidated graph...")
    plot_cpu_usage(axs[0], cpu_log)
    plot_npu_usage(axs[1], npu_csv)
    plot_memory_usage(axs[2], memory_log)

    if gpu_files:
        for idx, gpu_file in enumerate(gpu_files):
            plot_gpu_metrics(axs[idx + 3], gpu_file)
    else:
        # Create a placeholder plot showing "No GPU data found"
        axs[3].text(0.5, 0.5, "No GPU Data", ha='center', va='center', fontsize=12)
        axs[3].axis('off')
        
    plt.tight_layout()
    output_image = os.path.join(root, 'plot_metrics.png')
    plt.savefig(output_image, dpi=300)
    plt.close()

    subprocess.run(["xdg-open", output_image])
    print(f"✅ Saved: {output_image}")

if __name__ == '__main__':
    main()
