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

Both are validated in the constructor, before the stack is started, so a
mistyped `--app_dir` fails in seconds rather than after a bring-up and the
health timeout.

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

### Where the per-turn metrics come from, and `--results_dir`

The per-turn events are not written by this script. They are written by the
application's `kiosk-core` service from inside its container, into the
directory bound to `/app/results` by the application's compose file. That bind
mount is `${KIOSK_RESULTS_HOST_DIR:-./results}`, so:

- When this script starts the stack, it exports `KIOSK_RESULTS_HOST_DIR` set to
  `--results_dir`, and the service writes where this script reads.
- With `--skip_stack` the stack is already running with whatever mount it was
  started with. If `--results_dir` differs from that directory, no events will
  be found and the run ends with `No transactions`. The script warns about this
  before running. Either omit `--results_dir` or restart the stack.

Under `--skip_stack` or `--keep_stack` the warmup events are **not** pruned
from the log files. Rewriting them while kiosk-core still holds them open
would leave the service writing into `*.txt.with-warmup`, which the
consolidator ignores. The printed summary and the exported JSON still exclude
the warmup turns; only the raw log files keep them.

### Application-specific defaults

This orchestrator targets one application, so it carries that application's
names as defaults. If they change there, update them here:

| Setting | Default |
| --- | --- |
| `--app_dir` | `../../smart-kiosk-assistant` |
| kiosk-core health | `http://localhost:8012/health` |
| rag-service health | `http://localhost:8020/health` |
| OVMS model list | `http://localhost:8000/v3/models` |
| `make` arguments | `QUEUE=false` |
| Report basename | `--label`, default `v2v-perf-tools` |
| Measurement script | `tests/benchmarks/v2v_scripted_conversation_benchmark.py` |
| Metrics bind mount | `KIOSK_RESULTS_HOST_DIR` |

The run exports `KIOSK_CORE_DIARIZATION_ENABLED` and
`KIOSK_CORE_DIARIZATION_INTERMEDIATE_ENABLED` as `false`, to stop diarization
competing with text-to-speech for CPU.

Queue handling follows the application Makefile rather than fighting it. That
Makefile defaults `QUEUE` from `KIOSK_CORE_QUEUE_SERVICE_ENABLED`, uses
`QUEUE` to decide whether compose is given `--profile queue` at all, and then
re-exports `KIOSK_CORE_QUEUE_SERVICE_ENABLED` derived from `QUEUE`. A `QUEUE=`
argument therefore always wins, so this script mirrors it into the environment
instead of forcing a value that would be silently overridden. The Makefile
tests it with `$(filter true TRUE,$(QUEUE))`, so only the literals `true` and
`TRUE` enable the vision profile; this script applies the same rule, so a
`QUEUE=1` run is correctly recorded as queue-disabled. `queue-service`
and `rtsp-streamer` are consequently always both on or both off, and the
`stack` block of the results reports the resolved value rather than a
hardcoded one.

Before each run the script deletes its own `{label}.json`,
`{label}-warmup.json` and `measured_window.json`, and ignores any report that
finished before the measured run started. Without this, a measured run that
crashed before writing its report would be cross-checked against the previous
run's numbers.

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
report, comparing both the median latency and the turn count. Turns the report
marks as errored are excluded from that count, since they never produce a
start/end pair. A disagreement
of more than 1 ms, or any difference in turn count, is reported as `MISMATCH`
and **exits non-zero**, so CI cannot publish a headline number produced by a
broken emit path. A cross-check of `unavailable` -- the application's report
was missing or written before the measured run started -- fails the run for
the same reason: nothing verified the number. Pass
`--allow_cross_check_mismatch` to downgrade both to warnings.

Staleness is judged by the report file's modification time rather than its
`finished_at` field, so a timestamp written without a UTC offset cannot make a
fresh report look hours old on a host that is not on UTC.

Warmup turns are excluded by timestamp rather than by deleting log files,
because the application holds those files open for the whole run. Whole
start/end pairs are removed together, so a warmup turn whose end is logged
after the measured window begins cannot leave an orphan event behind for the
consolidator to mispair.

Report p95 with its sample count. The nearest-rank p95 of *n* samples is the
`ceil(0.95 * n)`-th, so with 19 or fewer measured turns it is simply the
slowest turn; at 20 turns it is the 19th. The summary and the results JSON
carry `p95_sample_count` and `p95_is_max`, the latter determined by comparing
the p95 against the observed maximum rather than assumed from the count.

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

