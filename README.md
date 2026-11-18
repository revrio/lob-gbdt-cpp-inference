# Limit Order Book Gradient Boosting & Low-Latency C++ Inference

Predicting short-horizon (`t + 50` updates, approximately 5s) mid-price direction on `BTCUSDT` using Limit Order Book snapshots and LightGBM. A terminal C++ engine replays recorded snapshots against a frozen Python reference bundle.

## Repository Structure
- `python/` — Async WebSocket LOB recorder, feature engineering, LightGBM training, and golden vector exporter.
- `contract/` — Frozen feature specification (`feature_spec.md`), exported LightGBM `model.txt`, and golden test vectors.
- `cpp/` — Implemented: all eleven streaming features, validated LightGBM loading, tree/ensemble evaluation, probabilities, decision rules, and Python/C++ feature and prediction parity. Also implemented: streaming CSV replay, a checksummed Parquet-to-CSV exporter, CLI integration tests, and in-memory baseline benchmarks. Remaining: live ingestion, dashboard integration, and measured optimization.
- `docs/` — Project workflow, benchmark plots, and write-up.

## Setup (Python Data & Modeling)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
systemd-inhibit --what=idle:sleep --why="Recording Binance LOB Data" python3 python/record_lob.py
```

## Frozen Python → C++ reference bundle

The `contract/` reference bundle contains a saved model and known inputs/outputs
for implementing C++ independently of ongoing recording or model experiments.
Verify it without retraining or access to raw recordings:

```bash
python python/export_contract.py verify --bundle contract
python -m unittest discover -s python/tests -v
```

The 5,000 inputs include 300 warm-up rows; the remaining 4,700 rows have expected
features, raw class scores, probabilities, and class IDs. Join files by `row_id`.
The frozen model has 183 boosting rounds (549 trees) and a validation-selected
conviction threshold of 0.31. See `contract/feature_spec.md` for exact column
order and numerical rules.

To deliberately recreate the bundle from its frozen source list (the listed raw
files must be available locally):

```bash
python python/export_contract.py export \
  --manifest contract/training_manifest.json --output /tmp/lob-reference
```

Export uses the established training settings, locked calendar split and
validation-only threshold selection. It refuses to overwrite an existing bundle
unless `--force` is supplied. CSVs are written at float64 round-trip precision
and the staged bundle is verified before it replaces any output files.

## C++ streaming features

```bash
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build -j 2
ctest --test-dir cpp/build --output-on-failure
```

`FeatureEngine` connects the stateless feature functions and fixed-capacity
rolling buffers. Supply snapshots one at a time:

```cpp
#include "lob/feature_engine.hpp"

lob::FeatureEngine engine;  // Create once; owns the recent history.
lob::FeatureVector features{};
// For each validated BookSnapshot named snapshot:
if (engine.update(snapshot, features)) {
  // All eleven features are ready here; pass them to the loaded model.
}
```

The first 300 snapshots of each segment return `false`, leaving the output
untouched; snapshot 301 writes a complete feature vector and returns `true`.
Gaps greater than 1000ms restart warm-up. Equal timestamps are accepted, while
negative or decreasing timestamps are rejected without changing engine state.
Call `reset()` before replaying a separate stream from the beginning.

The golden test checks every fixture readiness flag and all ready feature values
against Python with an absolute tolerance below `1e-6`. It uses a small test-only
CSV reader. The user-facing `lob_replay` executable uses the validated streaming
reader in `csv_reader.hpp`/`csv_reader.cpp`.

## Locked dataset and evaluation

The checksummed source list in `contract/training_manifest.json` freezes 435
files and 3,855,454 snapshots through November 7, 2026, 11:23 IST. New recordings
are excluded until the manifest is deliberately updated. Run `python python/pipeline.py`
to evaluate the locked data (raw recordings must be available).

Both Python and C++ reset history only for receive-time gaps **greater than
1,000ms**; exactly 1,000ms stays in the current segment. This is a continuity
cutoff, not an execution delay. The label horizon remains 50 updates, roughly
five seconds, and warm-up remains 300 updates.

Training uses data through November 3 IST; November 4 selects early stopping
and decision thresholds; November 5 onward is the inspected test period.
Targets crossing calendar boundaries are purged and the first 100 raw updates
of each subsequent period are embargoed. The current test has already been
reviewed; use new future data for a fresh final evaluation after further tuning.

Recorded execution totals in the pipeline are exploratory overlapping signals,
not a realistic account PnL. Classification scores do not establish profitability.

## C++ model loading and prediction

`tree.hpp` declares a validated decision tree. `model.hpp` declares the model
loader and prediction interface; their implementations live in `cpp/src/`.
The `lob` CMake target is a compiled static library.

```cpp
#include "lob/feature_engine.hpp"
#include "lob/model.hpp"

const auto model = lob::LightGBMModel::load("contract/model.txt"); // Once at startup.
const double tau = 0.31; // conviction_tau in contract/metadata.json.
lob::FeatureEngine engine;
lob::FeatureVector features{};
// On each validated BookSnapshot named snapshot:
if (engine.update(snapshot, features)) {
  const auto result = model.predict(features, tau);
  // result.raw_scores, result.probabilities: Down, Flat, Up.
  // result.class_id: 0=Down, 1=Flat, 2=Up.
}
```

Loading reconstructs the existing learned trees; it does not train them.
A numerical split takes its left child for `feature <= threshold`. Each tree
returns one leaf score; tree index modulo three chooses the class receiving it.
Stable softmax converts accumulated scores into probabilities. Leaf values
already contain learning-rate scaling and initial bias; do not apply them again.
The conviction threshold is passed explicitly; `model.txt` does not contain it.

The loader supports the frozen v4, three-class numerical model and exact feature
order. It rejects malformed arrays, invalid indices, cyclic/shared/unreachable
nodes, nonfinite parameters, incomplete iterations, categorical or linear trees,
averaged ensembles, and missing-value split modes. Prediction requires finite
features and a threshold in [0,1]. Training statistics and the post-tree footer
are not needed for evaluation. Load errors use exceptions; valid predictions
read immutable trees and do not allocate memory.

The parity test checks all 4,700 ready fixture rows twice: once from stored
Python features and once through the C++ FeatureEngine. It compares raw scores
and probabilities with absolute error below `1e-6`, and classes exactly.

Rolling volatility follows the compensated Welford operation order of the frozen
pandas 3.1.0 reference, including cancellation recovery. Equivalent formulas
can differ near zero and cross tree thresholds despite passing feature tolerance.
Avoid fast-math/reassociation compiler flags. Matching this fixture is a
correctness check, not a measured live-latency or throughput claim.

## Engine baseline benchmark

```bash
cmake --build cpp/build -j2
ctest --test-dir cpp/build --output-on-failure
cpp/build/lob_bench --tau 0.31 --passes 20 --latency-passes 5 --output /tmp/lob-baseline.json
```

The runner preloads the fixture/model and measures features, inference and both
combined. Throughput uses whole-pass timing; p50/p95/p99 use a separate per-update
run. Results include timer overhead and output stores, but exclude loading,
printing and initial warm-up. See [benchmark methodology and recorded results](docs/benchmarks.md).
These are warm in-memory results. Full-file replay is measured separately;
live/network measurements remain a later milestone.

## Terminal replay (iteration one)

Build and test from the repository root (Python 3 is also required for CLI tests):

```bash
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build -j2
ctest --test-dir cpp/build --output-on-failure
```

Replay the checked-in example without raw recording files:

```bash
cpp/build/lob_replay --input contract/golden_input_lob.csv --tau 0.31 \
  --output /tmp/lob-predictions.csv --summary /tmp/lob-replay-summary.json
```

This produces 4,700 predictions from 5,000 snapshots, after the first 300 warm-up
updates. Output includes input row ID, timestamp, update ID, three raw scores,
three probabilities, and class ID. The JSON summary goes to stdout and optionally
`--summary`. Existing output paths are refused; choose new paths for another run.
The model defaults to `contract/model.txt`; `--model` overrides it. Supply the
threshold from `contract/metadata.json` explicitly.

Export the complete locked dataset when its Parquet files are available locally:

```bash
python python/export_replay.py --manifest contract/training_manifest.json \
  --output /tmp/lob-recording.csv
cpp/build/lob_replay --input /tmp/lob-recording.csv --tau 0.31 \
  --output /tmp/lob-all-predictions.csv
```

CSV exports can be several GB. The exporter verifies hashes, row counts and
ordering, processes one batch at a time, and prints output/manifest checksums.
It exports raw snapshots only; C++ computes features and predictions. The reader
requires the exact canonical unquoted numeric CSV schema; it does not read
Parquet directly or accept arbitrary CSV layouts.

Replay runs at maximum speed. Receive-time gaps >1,000ms reset feature history
and start another 300-update warm-up. Equal timestamps are accepted when update
IDs increase. Invalid rows stop with a source filename and line number. No rows
are silently skipped. Predictions and summaries are each published as complete
files after successful EOF; temporary files are cleaned up on handled failures.
Publication uses same-filesystem hard links and is intended for local filesystems
supporting them. This is process-level publication, not a power-loss durability
guarantee or a transaction across both output files.

For full replay timing, `--no-output` disables CSV serialization/writing while
still performing every prediction and consuming its output in a checksum:

```bash
cpp/build/lob_replay --input /tmp/lob-recording.csv --tau 0.31 --no-output
cpp/build/lob_replay --input /tmp/lob-recording.csv --tau 0.31 \
  --output /tmp/lob-profile-predictions.csv --profile
```

`--profile` measures aggregate parsing/validation, feature, inference and output
time. Its per-stage timer overhead is included, so compare unprofiled runs for
throughput. Replay wall time includes CSV reading, validation, processing,
checksum bookkeeping, output formatting/writing and flush; model loading is
reported separately. Input opening, output staging setup/publication and summary
printing are outside replay wall time. Flush is not fsync. No playback pacing,
network latency or exchange execution is simulated.

The modelling frame removes the last 50 updates of a segment to construct future
labels; replay does not. Expect more predictions than labelled modelling rows.
The CTest `ReplayCLI` entry checks the actual executable with seven integration
cases, including Python golden probabilities/classes, gap boundaries and failure
cleanup. The CI C++ job runs it without access to raw recordings.

The engine now also matches the reference's compensated rolling OFI sums. A wider
multi-session check exposed threshold crossings from rounding in the earlier
`sum += new - old` implementation. Separate compensated additions/removals and
constant-window handling eliminate those differences on the checked sample.
The frozen Python model and prediction fixtures did not change.

The [full replay benchmark report](docs/replay-benchmarks.md) records the complete
3,855,454-snapshot replay, stage timings, memory use, correctness checks and
reproduction commands. Large generated CSVs stay outside Git.
