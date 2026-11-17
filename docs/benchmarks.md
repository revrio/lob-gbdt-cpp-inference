# Initial C++ engine benchmark

This is the straightforward evaluator baseline, measured on November 17, 2026.
It measures a warm in-memory fixture, not file replay or exchange-to-screen latency.
No model/evaluator optimization is included in this milestone.

## Reproduce

From the repository root:

```bash
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build -j2
ctest --test-dir cpp/build --output-on-failure
cpp/build/lob_bench --tau 0.31 --passes 20 --latency-passes 5 --output /tmp/lob-baseline.json
```

Run repetitions sequentially, never concurrently. JSON goes to stdout and optionally
`--output`; progress goes to stderr. `--model` and `--input` override the default
contract paths. The runner requires the canonical numerical CSV schema and a
single contiguous segment. Tau must be supplied explicitly from metadata.
This runner is a benchmark utility, not the general replay executable.

## Measurement method

1. Load and validate the model and snapshot CSV, and prepare inference features,
   before starting timing. Allocate output and timing storage in advance.
2. For each mode, run two untimed passes to warm code, model and output pages.
   Seed the feature engine with the first 300 snapshots outside each timed pass.
3. Time 20 whole passes of the remaining 4,700 observations for throughput.
   Report median pass wall time, amortized ns/update and its reciprocal in updates/s.
4. Separately time five passes one observation at a time: 23,500 latency samples
   per mode, using steady_clock with compiler fences. p50/p95/p99 use nearest rank.
5. Store every result and reduce all output fields into a printed checksum after
   the timed region. Timings include result stores and loop/dispatch overhead;
   they exclude checksum reduction, allocations, I/O and initial warm-up.
6. Measure 10,000 empty timer intervals. Report their p50/p99 without subtracting
   an estimate from the latency samples. For very fast features, prefer whole-pass
   amortized cost over interpreting each instrumented latency as pure engine time.

Features mode measures FeatureEngine::update. Inference mode measures all 549
trees, score accumulation, softmax, input checks and the class decision.
Combined mode measures both on the same sequential snapshots. Repetitions reset
state and replay from the beginning; they do not concatenate sessions.

## Recorded baseline

Hardware: Intel Core 5 210H, 12 reported hardware threads. Compiler: GCC 16.2.1,
C++20, Release, -O3 -DNDEBUG. Three sequential runs, normal laptop session; CPU
not pinned and frequency/background load uncontrolled. The table gives the
minimum–maximum of each run's statistic, not pooled percentiles or confidence
intervals.

| Mode | p50 µs | p95 µs | p99 µs | Updates/s from median pass |
|---|---:|---:|---:|---:|
| features | 0.115–0.166 | 0.139–0.190 | 0.169–0.229 | 16,734,435–22,515,941 |
| inference | 12.369–20.857 | 33.348–56.008 | 47.583–75.706 | 37,365–41,407 |
| combined | 15.634–22.357 | 42.764–63.831 | 61.798–83.484 | 36,954–53,796 |

The median empty timer interval ranged from 24 to 34ns. Maximum latency and
per-run values are retained in the raw JSON, including scheduling outliers.
Inference and combined checksums agree in all three repetitions.

The combined path takes most of its time in model evaluation. These measurements
suggest ample local compute capacity for a nominal 10-update/s feed, but cannot
establish live data freshness, network delay or trading profitability.

The separate modes are measured at different times: CPU scheduling, frequency,
cache state and instrumentation can make combined throughput appear faster than
inference alone. Do not subtract these independent measurements to infer an
exact feature cost. Use the separately measured feature mode.

Raw results: [run 1](benchmarks/baseline_20261117/run1.json),
[run 2](benchmarks/baseline_20261117/run2.json),
[run 3](benchmarks/baseline_20261117/run3.json).
[Provenance](benchmarks/baseline_20261117/provenance.json) records compiler context,
input/source hashes and the measured binary hash. The tiny fixed fixture is
cache-friendly and repeatedly revisits the same branches; later replay and
larger-dataset benchmarks must be reported separately.

## Next measurements

After the replay executable exists, measure loading/parsing and full-file replay
separately. Before and after any evaluator optimization, rerun parity and this
same baseline on the same inputs and build settings under comparable conditions.
Keep live arrival timing, queueing and dashboard publication measurements separate.
