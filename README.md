# Limit Order Book Gradient Boosting & Low-Latency C++ Inference

Predicting short-horizon (`t + 50` updates, approximately 5s) mid-price direction on `BTCUSDT` using Limit Order Book snapshots and LightGBM. The C++ engine is being built against a frozen Python reference bundle.

## Repository Structure
- `python/` — Async WebSocket LOB recorder, feature engineering, LightGBM training, and golden vector exporter.
- `contract/` — Frozen feature specification (`feature_spec.md`), exported LightGBM `model.txt`, and golden test vectors.
- `cpp/` — Implemented: all eleven streaming features, validated LightGBM loading, tree/ensemble evaluation, probabilities, decision rules, and Python/C++ feature and prediction parity. Also implemented: a standalone in-memory baseline benchmark. Remaining: replay runner and its benchmarks, followed by live ingestion and dashboard integration.
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
CSV reader; the user-facing replay executable is not yet implemented.

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
These are warm in-memory results; full-file replay and live/network measurements
remain separate milestones.
