"""Presentation port from the user-supplied analyze-lp_asc-benchmarks.py.

Source: jcork-intel/asc-benchmark-reporting. Data collection remains in the
calling adapter. Report-specific contents retain the reference visual design.
"""

from datetime import datetime
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

FMT = {
    "stream_density":    ("Stream density",             "streams", "{:.0f}"),
    "total_fps":         ("Throughput",                 "fps",     "{:.2f}"),
    "avg_fps":           ("FPS/stream",                 "fps",     "{:.2f}"),
    "latency_p50":       ("Pooled latency p50",         "ms",      "{:.1f}"),
    "latency_p95":       ("Pooled latency p95",         "ms",      "{:.1f}"),
    "latency_p99":       ("Pooled latency p99",         "ms",      "{:.1f}"),
    "highest_stream_p95":("Highest per-stream p95",     "ms",      "{:.1f}"),
    "cpu_util_pct":      ("Avg CPU Utilization",        "%",        "{:.1f}"),
    "npu_util_pct":      ("Avg NPU Utilization",        "%",        "{:.1f}"),
    "gpu_engine_pct":    ("Avg GPU Engine Utilization", "%",        "{:.1f}"),
    "gpu_power_reported":("GPU power (reported source)", "W",       "{}"),
    "proc_power_w":      ("Avg Processor Package Power","W",        "{:.1f}"),
    "power_scope":       ("Power/stream (source scope)", "W/stream", "{}"),
    "utilisation_scope": ("Utilization (source scope)", "%",        "{}"),
    "status":            ("Sweep status",               "",         "{}"),
    "last_passing":      ("Last passing lanes (not confirmed)", "", "{}"),
}

HW_COLORS = {
    "cpu":    "#4e79a7",
    "gpu":    "#f28e2b",
    "npu":    "#59a14f",
    "hetero": "#b07aa1",
}

SERIES_COLORS = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2"]

CHART_CONFIGS = [
    {
        "keys":       ["stream_density"],
        "title":      "Stream density (streams) @ Target FPS (higher is better)",
        "ylabel":     "Streams",
        "fname":      "compare_stream_density.png",
        "labels":     ["Stream density (streams)"],
        "horizontal": True,
    },
    {
        "keys":       ["proc_power_w"],
        "title":      "Processor Package Power (lower is better)",
        "ylabel":     "Watts (W)",
        "fname":      "compare_power.png",
        "labels":     ["Package Power (W)"],
        "horizontal": True,
    },
    {
        "keys":       ["power_per_stream_w"],
        "title":      "Efficiency: Package Power per Stream (lower is better)",
        "ylabel":     "W / stream",
        "fname":      "compare_power_per_stream.png",
        "labels":     ["Power / Stream (W)"],
        "horizontal": True,
    },
    {
        "keys":       ["cpu_util_pct", "gpu_engine_pct", "npu_util_pct"],
        "title":      "Accelerator Utilization",
        "ylabel":     "Utilization (%)",
        "fname":      "compare_utilization.png",
        "labels":     ["CPU Active %", "GPU Engine %", "NPU %"],
        "horizontal": False,
    },
    {
        "keys":       ["latency_p50", "latency_p95", "latency_p99", "highest_stream_p95"],
        "title":      "Per-Frame Latency Percentiles (lower is better)",
        "ylabel":     "Latency (ms)",
        "fname":      "compare_latency.png",
        "labels":     ["Pooled p50", "Pooled p95", "Pooled p99", "Highest stream p95"],
        "horizontal": False,
    },
]


def plot_charts(all_kpis: list[dict], out_dir: str) -> None:
    all_kpis = [k for k in all_kpis if k.get("stream_density") is not None]
    if not all_kpis:
        return

    targets = [k["hw"] for k in all_kpis]
    x = np.arange(len(targets))

    for cfg in CHART_CONFIGS:
        keys       = cfg["keys"]
        labels     = cfg["labels"]
        horizontal = cfg.get("horizontal", False)
        n_bars     = len(keys)
        width      = 0.65 / n_bars

        fig, ax = plt.subplots(figsize=(6, 3.5) if horizontal else (7, 4.5))
        has_data = False
        use_hw_colors = (n_bars == 1)

        for idx, (key, label) in enumerate(zip(keys, labels)):
            values = [k.get(key) for k in all_kpis]
            if all(v is None for v in values):
                continue
            has_data = True
            measured = [index for index, value in enumerate(values) if value is not None]
            plot_vals = [values[index] for index in measured]
            offsets = x[measured] + (idx - (n_bars - 1) / 2) * width

            if use_hw_colors:
                bar_colors = [HW_COLORS.get(k.get("color_target", k["hw"].lower()), "#888888")
                              for k in (all_kpis[index] for index in measured)]
            else:
                bar_colors = SERIES_COLORS[idx % len(SERIES_COLORS)]

            if horizontal:
                bars = ax.barh(offsets, plot_vals, height=width * 0.9,
                               color=bar_colors, label=label,
                               edgecolor="white", linewidth=0.5)
                for bar, raw in zip(bars, plot_vals):
                    if raw is not None:
                        ax.annotate(
                            f"{raw:.1f}",
                            xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
                            xytext=(4, 0), textcoords="offset points",
                            ha="left", va="center", fontsize=8,
                        )
            else:
                bars = ax.bar(offsets, plot_vals, width=width * 0.9,
                              color=bar_colors, label=label,
                              edgecolor="white", linewidth=0.5)
                for bar, raw in zip(bars, plot_vals):
                    if raw is not None:
                        ax.annotate(
                            f"{raw:.1f}",
                            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=8,
                        )

        if not has_data:
            plt.close(fig)
            continue

        ax.set_title(cfg["title"], fontsize=12, fontweight="bold")
        if horizontal:
            ax.set_xlabel(cfg["ylabel"])
            ax.set_yticks(x)
            ax.set_yticklabels(targets)
            ax.invert_yaxis()
            ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
            ax.grid(axis="x", alpha=0.3, linestyle="--")
            all_vals = [v for k in all_kpis for key in cfg["keys"]
                        for v in [k.get(key)] if v is not None]
            if all_vals:
                ax.set_xlim(0, max(all_vals) * 1.18)
        else:
            ax.set_ylabel(cfg["ylabel"])
            ax.set_xticks(x)
            ax.set_xticklabels(targets)
            ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
            ax.grid(axis="y", alpha=0.3, linestyle="--")
            all_vals = [v for k in all_kpis for key in cfg["keys"]
                        for v in [k.get(key)] if v is not None]
            if all_vals:
                ax.set_ylim(0, max(all_vals) * 1.18)
        ax.set_axisbelow(True)
        if n_bars > 1:
            ax.legend(fontsize=8)
        elif len(targets) > 1:
            from matplotlib.patches import Patch
            handles = [Patch(color=HW_COLORS.get(k.get("color_target", k["hw"].lower()), "#888"),
                             label=k["hw"]) for k in all_kpis]
            ax.legend(handles=handles, fontsize=8)
        fig.tight_layout()

        out_path = os.path.join(out_dir, cfg["fname"])
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  Saved chart: {out_path}")


def plot_fps_heatmap(steps: list[dict], hw: str, out_path: str, target_fps: float = 14.95) -> bool:
    """Plot per-stream FPS progression heatmap; returns True if saved."""
    if not steps:
        return False

    deduped: dict[int, dict] = {}
    for s in steps:
        deduped[s["n_streams"]] = s
    steps = sorted(deduped.values(), key=lambda s: s["n_streams"])

    max_stream = max(max(s["fps_map"].keys()) for s in steps)
    n_steps = len(steps)
    matrix = np.full((max_stream + 1, n_steps), np.nan)
    for col, step in enumerate(steps):
        for sid, fps in step["fps_map"].items():
            matrix[sid, col] = fps

    x_labels = [s["n_streams"] for s in steps]

    fig_w = max(10, n_steps * 0.18)
    fig_h = max(4, (max_stream + 1) * 0.22)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    fail_fps = target_fps * 0.945 if target_fps is not None else None
    vmin, vmax = (fail_fps * 0.85, target_fps) if target_fps is not None else (None, None)
    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad(color="#e0e0e0")

    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="nearest", origin="lower")

    ax.axhline(y=-0.5, color="none")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("fps", fontsize=9)
    if target_fps is not None:
        cbar.ax.axhline(y=(target_fps - vmin) / (vmax - vmin), color="white",
                        linewidth=1.5, linestyle="--")

    tick_step = max(1, n_steps // 20)
    ax.set_xticks(range(0, n_steps, tick_step))
    ax.set_xticklabels([x_labels[i] for i in range(0, n_steps, tick_step)],
                       fontsize=7, rotation=45, ha="right")

    ytick_step = max(1, (max_stream + 1) // 15)
    ax.set_yticks(range(0, max_stream + 1, ytick_step))
    ax.set_yticklabels([f"Stream {i}" for i in range(0, max_stream + 1, ytick_step)],
                       fontsize=7)

    ax.set_xlabel("Active Stream Count (step)", fontsize=9)
    ax.set_ylabel("Stream ID", fontsize=9)
    target_label = (f"white dashed = target {target_fps} fps, "
                    if target_fps is not None else "per-camera targets, ")
    ax.set_title(f"{hw.upper()} \u2014 Per-Stream FPS Progression\n"
                 f"({target_label}grey = stream not yet active)",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


CHART_FILES = [
    ("compare_stream_density.png",   "Stream density (streams)"),
    ("compare_power.png",             "Processor Package Power"),
    ("compare_power_per_stream.png",  "Package Power per Stream"),
    ("compare_utilization.png",       "Accelerator Utilization"),
    ("compare_latency.png",           "Per-Frame Latency Percentiles"),
]
HEATMAP_FNAME = "heatmap_fps_{hw}.png"


def _cell(v, fmt):
    return fmt.format(v) if v is not None else "N/A"


def _embed_png(fpath: str, title: str) -> str:
    import base64
    with open(fpath, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return (f'<div class="chart"><h3>{title}</h3>'
            f'<img src="data:image/png;base64,{b64}" alt="{title}"></div>\n')


def generate_html_report(all_kpis: list[dict], out_dir: str,
                         system_info: dict,
                         test_config: str = "", description_html: str = "") -> str:
    targets = [k["hw"] for k in all_kpis]
    archive_name = os.path.basename(os.path.abspath(out_dir))
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    hardware_html = ""
    software_html = ""
    for index, kpi in enumerate(all_kpis):
        si = kpi.get("system_info", system_info)
        hardware_rows = [
            ("Hostname", si["hostname"]), ("System vendor", si["system_vendor"]),
            ("System model", si["system_model"]), ("CPU", si["cpu_model"]),
            ("CPU Cores", si["cpu_cores"]),
            ("Logical CPUs", si["logical_cpus"]),
            ("Total RAM", f'{si["total_ram_gb"]} &nbsp;({si["mem_channels"]}, '
             f'{si["mem_per_dimm"]} per memory device, {si["mem_speed"]})'),
            ("GPU", si["gpu_model"]),
            ("NPU", f'{si["npu_model"]} &nbsp;<span style="color:#888;font-size:12px;">'
             f'({si["npu_driver"]})</span>'),
            ("PL1 / PL2", f'{si["pl1_w"]} &nbsp;/&nbsp; {si["pl2_w"]}'),
            ("OS", si["os_name"]), ("Kernel", f'{si["kernel"]} ({si["arch"]})'),
        ]
        hardware_html += f'<h3 style="color:#0071c5;">{kpi["hw"]}</h3><table>' + "".join(
            f'<tr><td><b>{label}:</b></td><td>{value}</td></tr>'
            for label, value in hardware_rows) + '</table>'
        software_rows = "".join(
            f'<tr><td><b>{label}:</b></td><td>{value}</td></tr>'
            for label, value in kpi.get("software_rows", []))
        stage_rows = "".join(
            f'<tr><td><b>{stage["type"]}:</b></td><td>{stage["device"]} &mdash; '
            f'{stage["model"]} ({stage["precision"]})</td></tr>'
            for stage in kpi.get("pipeline_steps", []))
        if index == len(all_kpis) - 1:
            stage_rows += (
                f'<tr><td><b>Archive:</b></td><td><code>{archive_name}</code></td></tr>'
                f'<tr><td><b>Generated:</b></td><td>{now_str}</td></tr>')
        software_html += f'<h3 style="color:#0071c5;">{kpi["hw"]}</h3><table>{software_rows}{stage_rows}</table>'
    sys_html = (
        '<div class="meta-cols"><div class="meta-half">'
        '<div class="meta-heading">&#9881; Hardware</div>'
        f'{hardware_html}</div><div class="meta-divider"></div><div class="meta-half">'
        '<div class="meta-heading">&#128196; Test &amp; Software</div>'
        f'{software_html}</div></div>'
    )

    cards_html = ""
    for k in all_kpis:
        hw   = k["hw"]
        col  = HW_COLORS.get(k.get("color_target", hw.lower()), "#0071c5")
        sd   = k.get("stream_density")
        fps  = k.get("avg_fps")
        cpu  = k.get("cpu_util_pct")
        npu  = k.get("npu_util_pct")
        gpu  = k.get("gpu_engine_pct")
        tm   = k.get("timing", {})
        t_start = tm.get("start", "N/A")
        t_end   = tm.get("end",   "N/A")
        t_dur   = tm.get("duration_str", "N/A")
        steps   = k.get("pipeline_steps", [])
        prec_rows = "".join(
            f'<tr><td style="color:#888;padding:1px 8px 1px 0;">'
            f'{s["type"].replace("gva","").capitalize()}:</td>'
            f'<td><b>{s["device"]}</b> &mdash; {s["model"]} <span style="background:#f0f4ff;'
            f'border-radius:3px;padding:1px 4px;font-size:11px;">{s["precision"]}</span></td></tr>'
            for s in steps
        )
        prec_html = (
            f'<table style="border-collapse:collapse;margin-top:4px;font-size:12px;">'
            f'{prec_rows}</table>' if prec_rows else "N/A"
        )
        last_passing = (f'<div class="sub">Last passing lanes: <b>{k["last_passing"]}</b> '
                        '(not confirmed)</div>' if k.get("last_passing") else "")
        latency_html = "".join(
            f'<div class="sub">{label}: <b>{_cell(k.get(key), "{:.1f}")} ms</b></div>'
            for key, label in (("latency_p50", "Pooled latency p50"),
                               ("latency_p95", "Pooled latency p95"),
                               ("latency_p99", "Pooled latency p99"),
                               ("highest_stream_p95", "Highest per-stream p95")))
        cards_html += f"""
  <div class="card" style="border-top: 4px solid {col};">
    <h3 style="color:{col};">{hw}</h3>
    <div class="big" style="color:{col};">{sd if sd is not None else 'N/A'}</div>
    <div class="sub">stream density (streams)</div>
    <div class="sub">Sweep status: <b>{k.get('status', 'unknown')}</b></div>
    {last_passing}
    <div class="sub">Source run: <b>{k.get('source_run', 'N/A')}</b></div>
    <div class="sub">Total throughput: <b>{'N/A' if k.get('total_fps') is None else f"{k['total_fps']:.1f} fps"}</b></div>
    <div class="sub">FPS/stream (fps): <b>{'N/A' if fps is None else f'{fps:.2f}'}</b></div>
    {latency_html}
    <div class="sub">Power/stream (W/stream, source scope): <b>{k.get('power_scope', 'N/A')}</b></div>
    <div class="sub">CPU util: <b>{'N/A' if cpu is None else f'{cpu:.1f}%'}</b> &nbsp;
      GPU: <b>{'N/A' if gpu is None else f'{gpu:.1f}%'}</b> &nbsp;
      NPU: <b>{'N/A' if npu is None else f'{npu:.1f}%'}</b></div>
    <hr style="border:none;border-top:1px solid #eee;margin:10px 0;">
    <div class="sub" style="margin-bottom:4px;"><b>Pipeline:</b></div>
    <div class="sub">{prec_html}</div>
    <hr style="border:none;border-top:1px solid #eee;margin:10px 0;">
    <div class="sub">&#9200; Run duration: <b>{t_dur}</b></div>
    <div class="sub" style="font-size:11px;color:#888;">{t_start} &rarr; {t_end}</div>
  </div>"""

    header_cells = "".join(f"<th>{t}</th>" for t in targets)
    rows_html = ""
    for key, (label, unit, fmt) in FMT.items():
        if key == "last_passing" and not any(k.get("last_passing") for k in all_kpis):
            continue
        cells = "".join(f'<td>{_cell(k.get(key), fmt)}</td>' for k in all_kpis)
        unit_html = f"<span class='unit'>({unit})</span>" if unit else ""
        rows_html += (f"<tr><td><strong>{label}</strong> "
                  f"{unit_html}</td>{cells}</tr>\n")

    charts_html = ""
    for fname, title in CHART_FILES:
        fpath = os.path.join(out_dir, fname)
        if os.path.isfile(fpath):
            charts_html += _embed_png(fpath, title)

    heatmaps_html = ""
    for k in all_kpis:
        fpath = os.path.join(out_dir, HEATMAP_FNAME.format(hw=k["hw"].lower()))
        if os.path.isfile(fpath):
            heatmaps_html += _embed_png(fpath, f"{k['hw']} \u2014 Stream FPS Progression")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LP Benchmark Report \u2014 {archive_name}</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 24px;
          background: #f5f5f5; color: #222; }}
  h1   {{ color: #0071c5; border-bottom: 3px solid #0071c5; padding-bottom: 8px; }}
  h2   {{ color: #333; margin-top: 36px; }}
  h3   {{ margin: 0 0 6px 0; font-size: 15px; }}
  .meta {{ background: #fff; border-radius: 8px; padding: 14px 20px;
           margin-bottom: 20px; box-shadow: 0 1px 4px rgba(0,0,0,.1); }}
  .meta-cols {{ display: flex; gap: 0; }}
  .meta-half {{ flex: 1; padding: 4px 12px; }}
  .meta-half table {{ border-collapse: collapse; width: 100%; box-shadow: none; }}
  .meta-half td {{ padding: 5px 12px 5px 0; font-size: 14px; border: none;
                   background: transparent; vertical-align: top; }}
  .meta-half td:first-child {{ white-space: nowrap; }}
  .meta-heading {{ font-size: 13px; font-weight: 700; color: #0071c5;
                   text-transform: uppercase; letter-spacing: .05em;
                   margin-bottom: 8px; }}
  .meta-divider {{ width: 1px; background: #e0e0e0; margin: 4px 8px; }}
  .summary-cards {{ display: flex; gap: 18px; flex-wrap: wrap; margin-bottom: 28px; }}
  .card {{ background: #fff; border-radius: 10px; padding: 18px 24px;
           box-shadow: 0 2px 8px rgba(0,0,0,.12); min-width: 190px; flex: 1; }}
  .big  {{ font-size: 48px; font-weight: 700; line-height: 1.1; }}
  .sub  {{ font-size: 13px; color: #555; margin-top: 5px; }}
  table.kpi {{ border-collapse: collapse; width: 100%; background: #fff;
               box-shadow: 0 1px 4px rgba(0,0,0,.1); }}
  table.kpi th, table.kpi td {{ border: 1px solid #ddd; padding: 9px 14px;
                                text-align: right; }}
  table.kpi th {{ background: #0071c5; color: #fff; }}
  table.kpi td:first-child {{ text-align: left; }}
  table.kpi tr:nth-child(even) {{ background: #f0f7ff; }}
  .unit {{ color: #888; font-weight: normal; font-size: .9em; }}
  .charts {{ display: flex; flex-wrap: wrap; gap: 20px; margin-top: 16px; }}
  .chart {{ background: #fff; padding: 12px; box-shadow: 0 1px 4px rgba(0,0,0,.1);
            border-radius: 6px; }}
  .chart h3 {{ font-size: 13px; color: #555; margin-bottom: 6px; }}
  .chart img {{ max-width: 620px; display: block; }}
  footer {{ margin-top: 40px; color: #999; font-size: .85em; text-align: center; }}
    .meta-half, .card, .chart {{ min-width: 0; }}
    .meta-half td, .sub, p {{ overflow-wrap: anywhere; }}
    .chart {{ max-width: 100%; box-sizing: border-box; }}
    .chart img {{ max-width: 100%; width: 620px; height: auto; }}
    .table-scroll {{ max-width: 100%; overflow-x: auto; }}
    @media (max-width: 760px) {{
        body {{ padding: 16px; }}
        .meta-cols {{ display: block; }}
        .meta-half {{ padding: 4px 0; }}
        .meta-half td:first-child {{ white-space: normal; }}
        .meta-divider {{ width: auto; height: 1px; margin: 12px 0; }}
        .card {{ flex-basis: 100%; }}
    }}
</style>
</head>
<body>
<h1>Intel Loss Prevention \u2014 Stream Density Benchmark Report</h1>

{description_html}

<div class="meta">
{sys_html}
</div>

<h2>Stream Density Summary</h2>
<div class="summary-cards">
{cards_html}
</div>

<h2>KPI Comparison Table</h2>
<div class="table-scroll">
<table class="kpi">
  <thead><tr><th>KPI</th>{header_cells}</tr></thead>
  <tbody>
{rows_html}  </tbody>
</table>
</div>

<h2>Comparison Charts</h2>
<div class="charts">
{charts_html}</div>

<h2>Stream Density FPS Progression Heatmaps</h2>
<p style="font-size:14px;">Each cell shows the measured FPS for a given stream at a given
step. Green \u2265 target FPS; red = below target. Grey = stream not yet active.
The white dashed line in the colour bar marks the target FPS threshold.</p>
<div class="charts">
{heatmaps_html}</div>

<footer>Generated by compare_stream_density.py &nbsp;|&nbsp; Intel Corporation</footer>
</body>
</html>
"""

    if test_config:
        base = os.path.splitext(os.path.basename(test_config))[0]
        test_name = re.sub(r'^camera_to_workload_', '', base)
    else:
        test_name = archive_name
    ts_match = re.search(r'(\d{8}_\d{6})$', archive_name)
    ts_suffix = ts_match.group(1) if ts_match else datetime.now().strftime("%Y%m%d_%H%M%S")
    hostname_slug = re.sub(r'[^\w.-]', '_', si.get('hostname', 'unknown'))
    out_path = os.path.join(out_dir, f"stream_density_{test_name}_{hostname_slug}_{ts_suffix}.html")
    with open(out_path, "w") as f:
        f.write(html)
    return out_path