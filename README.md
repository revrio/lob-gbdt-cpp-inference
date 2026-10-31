# Limit Order Book Gradient Boosting & Low-Latency C++ Inference

Predicting short-horizon (`t + 50` updates, approximately 5s) mid-price direction on `BTCUSDT` using Limit Order Book snapshots and LightGBM. The C++ engine is being built against a frozen Python reference bundle.

## Repository Structure
- `python/` — Async WebSocket LOB recorder, feature engineering, LightGBM training, and golden vector exporter.
- `contract/` — Frozen feature specification (`feature_spec.md`), exported LightGBM `model.txt`, and golden test vectors.
- `cpp/` — Implemented: all eleven streaming features, ring buffers, rolling statistics, warm-up/reset handling, unit tests and Python/C++ feature parity. Remaining: model loading/evaluation, prediction parity, replay runner, and benchmarks.
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
See `contract/feature_spec.md` for exact column order and numerical rules.

To deliberately recreate the bundle from its frozen source list (the listed raw
files must be available locally):

```bash
python python/export_contract.py export \
  --manifest contract/training_manifest.json --output /tmp/lob-reference
```

Export uses the established training settings, chronological split and
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
  // All eleven features are ready here. Model inference is the next milestone.
}
```

The first 300 snapshots of each segment return `false`, leaving the output
untouched; snapshot 301 writes a complete feature vector and returns `true`.
Gaps greater than 500ms restart warm-up. Equal timestamps are accepted, while
negative or decreasing timestamps are rejected without changing engine state.
Call `reset()` before replaying a separate stream from the beginning.

The golden test checks every fixture readiness flag and all ready feature values
against Python with an absolute tolerance below `1e-6`. It uses a small test-only
CSV reader; the user-facing replay executable and model evaluator are not yet
implemented.
