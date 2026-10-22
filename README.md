# Performance Evaluation Service

![CodeQL](https://github.com/intel-retail/performance-tools/actions/workflows/codeql.yaml/badge.svg?branch=main) ![DockerImageBuild](https://github.com/intel-retail/performance-tools/actions/workflows/build.yaml/badge.svg?branch=main)  [![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/intel-retail/performance-tools/badge)](https://api.securityscorecards.dev/projects/github.com/intel-retail/performance-tools) [![OpenSSF Best Practices](https://www.bestpractices.dev/projects/10448/badge)](https://www.bestpractices.dev/projects/10448)

## Documentation

- [Open source code for setup, install, and execution of software, with complete developer documentation](https://intel-retail.github.io/documentation/performance-tools/benchmark.html)
- [Developer focused website to enable developers to engage and build our partner community](https://www.intel.com/content/www/us/en/developer/articles/reference-implementation/automated-self-checkout.html)

## Benchmark Target Device

Benchmark scripts support explicit target-device selection via `--target_device` or `TARGET_DEVICE`.

Accepted values:

- `CPU`
- `GPU`
- `GPU.<index>` (for example: `GPU.0`, `GPU.1`, `GPU.2`)
- `NPU`

Examples:

```bash
# Existing workflow (still supported)
make benchmark TARGET_DEVICE=GPU

# Explicit GPU device selection
make benchmark TARGET_DEVICE=GPU.1

# Direct script usage
python benchmark-scripts/benchmark.py \
	--compose_file ./docker/docker-compose.yaml \
	--target_device GPU.1
```

## Smart Kiosk Voice-to-Voice (V2V) Benchmark

`benchmark-scripts/benchmark_smart_kiosk_v2v.py` measures end-to-end
voice-to-voice latency for the Smart AI Kiosk, following the same pattern as
`benchmark_order_accuracy.py`: bring the application stack up, run a fixed
workload, collect metrics into `results/`, tear the stack down.

Unlike the vision benchmarks it is **run-count based rather than duration
based**, because a voice turn is a discrete request/response rather than a
continuously churning stream.

### Requirements

This orchestrator drives an application repository it does not contain. The
directory given by `--app_dir` must provide:

- a `Makefile` with `up` and `down` targets, and
- `tests/benchmarks/v2v_scripted_conversation_benchmark.py`, which performs
  the measurement.

Both are validated before the stack is started.

### Usage

```bash
cd benchmark-scripts

# Defaults: 8 measured runs plus 1 discarded warmup run
python benchmark_smart_kiosk_v2v.py --app_dir /path/to/smart-kiosk-assistant

# Fewer runs, a named conversation, and a separate results directory
python benchmark_smart_kiosk_v2v.py \
	--app_dir /path/to/smart-kiosk-assistant \
	--script order-multi --runs 5 --results_dir /tmp/v2v-results

# Benchmark a stack that is already running
python benchmark_smart_kiosk_v2v.py \
	--app_dir /path/to/smart-kiosk-assistant --skip_stack
```

`--app_dir` defaults to `../../smart-kiosk-assistant`, which resolves when this
repository sits beside the application or is vendored into it as a submodule.

Service URLs and `make` arguments are overridable, so an application-side port
or Makefile change does not require a change here:

```bash
python benchmark_smart_kiosk_v2v.py \
	--app_dir /path/to/smart-kiosk-assistant \
	--kiosk_url http://localhost:18012/health \
	--make_arg QUEUE=false --make_arg TARGET_DEVICE=GPU
```

The same URLs may be set through `SMART_KIOSK_V2V_KIOSK_URL`,
`SMART_KIOSK_V2V_RAG_URL` and `SMART_KIOSK_V2V_OVMS_URL`.

### Output

Written into `--results_dir` (default `<app_dir>/results`):

| File | Contents |
| --- | --- |
| `smart_kiosk_v2v_results_<ts>.json` | Full results, including the cross-check |
| `smart_kiosk_v2v_summary_<ts>.csv` | Flattened key/value summary |
| `measured_window.json` | Epoch-ms span of the measured turns, so hardware counters can be trimmed to it |
| `vlm_application_metrics_*.txt` | Per-turn events, with warmup turns pruned |
| `vlm_application_metrics_*.txt.with-warmup` | Unpruned original, kept for debugging |

The script cross-checks the metrics it emits against the application's own
report and prints `MISMATCH` if they disagree by more than 1 ms, which catches
a broken emit path rather than silently misreporting the headline number.

Warmup turns are excluded by timestamp rather than by deleting log files,
because the application holds those files open for the whole run.

Report p95 with its sample count: below 21 measured turns the nearest-rank p95
is just the slowest turn, and the summary labels it as such.

### Tests

```bash
cd benchmark-scripts
python -m unittest benchmark_smart_kiosk_v2v_test
```

## Disclaimer

GStreamer is an open source framework licensed under LGPL. See https://gstreamer.freedesktop.org/documentation/frequently-asked-questions/licensing.html?gi-language=c.  You are solely responsible for determining if your use of Gstreamer requires any additional licenses.  Intel is not responsible for obtaining any such licenses, nor liable for any licensing fees due, in connection with your use of Gstreamer

Certain third-party software or hardware identified in this document only may be used upon securing a license directly from the third-party software or hardware owner. The identification of non-Intel software, tools, or services in this document does not constitute a sponsorship, endorsement, or warranty by Intel.

## Datasets & Models Disclaimer:

To the extent that any data, datasets or models are referenced by Intel or accessed using tools or code on this site such data, datasets and models are provided by the third party indicated as the source of such content. Intel does not create the data, datasets, or models, provide a license to any third-party data, datasets, or models referenced, and does not warrant their accuracy or quality.  By accessing such data, dataset(s) or model(s) you agree to the terms associated with that content and that your use complies with the applicable license.

Intel expressly disclaims the accuracy, adequacy, or completeness of any data, datasets or models, and is not liable for any errors, omissions, or defects in such content, or for any reliance thereon. Intel also expressly disclaims any warranty of non-infringement with respect to such data, dataset(s), or model(s). Intel is not liable for any liability or damages relating to your use of such data, datasets or models.

