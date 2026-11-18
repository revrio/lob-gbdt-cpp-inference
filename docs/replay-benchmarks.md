# Terminal replay baseline

Measured November 17, 2026, after the compensated rolling OFI correction.
This is a single-threaded offline replay of the complete locked dataset, using
the frozen 549-tree model and tau 0.31. It measures the terminal application's
CSV reading, validation, features, inference and optional prediction output.

## Full dataset results

| Mode | Replay wall time | Input snapshots/s | Predictions/s | Peak process RSS (KiB) |
|---|---:|---:|---:|---:|
| No prediction CSV | 71.883s | 53,635 | 47,293 | 4,812 |
| Prediction CSV | 80.250s | 48,043 | 42,363 | 4,764 |
| Prediction CSV, stage timers enabled | 79.744s | 48,348 | 42,631 | 5,024 |

Each mode was run once, sequentially, in a normal laptop session. The profiled
run being slightly faster than the unprofiled output run reflects uncontrolled
run-to-run variation; it does not show that profiling speeds up the application.
Use unprofiled repetitions under comparable conditions for optimization claims.

Hardware: Intel Core 5 210H, 12 reported hardware threads. GCC 16.2.1, C++20,
Release, `-O3 -DNDEBUG`. No CPU affinity, frequency, background-load or page-cache
controls were applied. These are neither controlled cold-disk measurements nor
live exchange latency measurements. RSS comes from `/usr/bin/time -v` and excludes
the operating system's file cache.

Input: 435 Parquet sources exported into a 2,990,692,594-byte CSV. Both C++ output
runs produced the same 536,527,078-byte prediction CSV. Large generated files
remain under ignored `instructions/replay_20261117/`; only the small measurements
and their provenance are included in Git.

## Counts and correctness

All three modes agree on:

| Quantity | Count |
|---|---:|
| Input snapshots | 3,855,454 |
| Warm-up rows | 455,864 |
| Predictions | 3,399,590 |
| Gaps greater than 1,000ms | 3,200 |
| Segments, including the first | 3,201 |
| Down / Flat / Up predictions | 898,949 / 1,680,319 / 820,322 |

The first 300 rows of each segment are warm-up. Short segments contribute no
predictions. Counts were independently derived from the input timestamps, and
the profiled/unprofiled output files have identical SHA-256 hashes. The no-output
mode still evaluates and consumes every score, probability and class in a
checksum; it does not bypass inference.

Replay retains ready rows at segment tails because it does not need future
prices. The labelled modelling dataset removes those tails to construct its
50-update target. These prediction counts therefore differ from training counts.

The executable's CI test compares every one of the 4,700 ready golden rows with
the checked-in Python scores/probabilities (absolute error below 1e-6) and exact
classes. An additional local check took the first three files from every
recording session: 23 files, 168,887 snapshots and 117,836 ready predictions.
On that sample, C++ scores, probabilities and classes matched Python exactly.
The sample spans multiple sessions and gaps; its artificial joins are segmented
using the same >1,000ms rule.

This wider check found two class differences in the previous OFI implementation.
Replacing `sum += new - old` with separate compensated removal/addition and
constant-window handling removed them. Tiny rounding differences can change a
tree branch even when a feature passes an absolute-tolerance check. The frozen
model and Python fixtures were not changed. The entire 3.86-million-row run was
checked for counts and mode agreement, not full numerical Python parity.

## Stage timing

The profiled output run reports aggregate stage time, including timer overhead:

| Stage | Time | Share of replay wall time |
|---|---:|---:|
| CSV reading, parsing and validation | 11.009s | 13.8% |
| Streaming features, including warm-up | 0.423s | 0.5% |
| Model inference on ready rows | 57.039s | 71.5% |
| CSV formatting, writing and final flush | 10.774s | 13.5% |
| Other loop bookkeeping and timer gaps | 0.500s | 0.6% |

Replay wall time starts before CSV header parsing and ends after the final output
flush. It includes input reads, validation, processing, counters/checksum and
serialization. It excludes input opening, model loading, temporary output setup,
output publication and summary printing. Model loading is reported separately
and took 12.6–13.6ms in these runs. Flush does not imply `fsync` durability.

The timers describe accumulated work, not per-snapshot p50/p99 latency. Inference
dominates this workload, so evaluator changes are the main candidate for future
compute optimization. The parser and output also have measurable costs.

## Current warm in-memory engine measurement

A separate run used the existing 5,000-row golden fixture: 300 warm-up rows,
4,700 ready rows, two untimed warming passes, 20 throughput passes and five
per-update latency passes. There are 23,500 latency samples per mode.

| Mode | p50 | p95 | p99 | Updates/s from median throughput pass |
|---|---:|---:|---:|---:|
| Features | 0.085µs | 0.102µs | 0.114µs | 24,259,317 |
| Inference | 9.989µs | 26.139µs | 32.598µs | 81,738 |
| Combined | 9.544µs | 24.757µs | 29.901µs | 83,987 |

The empty timer interval was p50 23ns / p99 26ns, without overhead subtraction.
For the very short feature operation, whole-pass throughput is more informative
than treating an instrumented latency as pure compute time. Modes run at
different times, so combined throughput can exceed inference-only throughput.

This fixture is smaller, repeatedly warmed and excludes parsing/output/checksum
reduction. Full replay covers different branches and includes warm-up rows in its
input rate. Do not subtract the two rates to calculate an exact parsing cost or
compare them as if they measured the same workload. The earlier three-run engine
session is retained in [the original report](benchmarks.md); session differences
cannot be attributed solely to the OFI correction.

These results show ample local processing capacity for one nominal 100ms stream.
They do not establish exchange-to-application freshness, network delay, order
execution performance or strategy profitability.

## Reproduce

Run from the repository root, with the locked raw Parquet files available for the
full export. Use fresh output paths: the exporter and replay refuse overwrites.
Run benchmark commands sequentially and avoid other compute-heavy work.

```bash
source .venv/bin/activate
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build -j2
ctest --test-dir cpp/build --output-on-failure
python -m unittest discover -s python/tests -v
python python/export_contract.py verify --bundle contract

python python/export_replay.py --manifest contract/training_manifest.json \
  --output /tmp/lob-replay-input.csv > /tmp/lob-export.json

/usr/bin/time -v -o /tmp/lob-no-output.resources.txt \
  cpp/build/lob_replay --input /tmp/lob-replay-input.csv --tau 0.31 \
  --no-output > /tmp/lob-no-output.json

/usr/bin/time -v -o /tmp/lob-output.resources.txt \
  cpp/build/lob_replay --input /tmp/lob-replay-input.csv --tau 0.31 \
  --output /tmp/lob-predictions.csv > /tmp/lob-output.json

/usr/bin/time -v -o /tmp/lob-profile.resources.txt \
  cpp/build/lob_replay --input /tmp/lob-replay-input.csv --tau 0.31 \
  --output /tmp/lob-profile-predictions.csv --profile > /tmp/lob-profile.json

cpp/build/lob_bench --tau 0.31 --passes 20 --latency-passes 5 \
  --output /tmp/lob-current-engine.json
```

Without the raw recordings, use `contract/golden_input_lob.csv` to exercise the
complete replay path; the full-data timings above will not apply to that sample.

Raw results: [no output](benchmarks/replay_20261117/no_output.json),
[CSV output](benchmarks/replay_20261117/output.json),
[profiled CSV output](benchmarks/replay_20261117/profile.json),
[current engine](benchmarks/replay_20261117/engine.json).
[Export checksums](benchmarks/replay_20261117/export.json),
[verification](benchmarks/replay_20261117/verification.json),
[sample parity](benchmarks/replay_20261117/sample_python_parity.json) and
[sample manifest](benchmarks/replay_20261117/sample_manifest.json) identify the
checked inputs. [Provenance](benchmarks/replay_20261117/provenance.json) includes
source/binary hashes, build context, package versions and exact measured commands.
