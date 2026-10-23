# Limit Order Book Gradient Boosting & Low-Latency C++ Inference

Predicting short-horizon (`t + 5s`) mid-price direction on `BTCUSDT` using 100ms Limit Order Book (LOB) snapshots, LightGBM, and a C++ inference engine verified for numerical parity.

## Repository Structure
- `python/` — Async WebSocket LOB recorder, feature engineering, LightGBM training, and golden vector exporter.
- `contract/` — Frozen feature specification (`feature_spec.md`), exported LightGBM `model.txt`, and golden test vectors.
- `cpp/` — C++ feature calculator, tree evaluator, parity test suite, and latency benchmarks.
- `docs/` — Project workflow, benchmark plots, and write-up.

## Setup (Python Data & Modeling)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
systemd-inhibit --what=idle:sleep --why="Recording Binance LOB Data" python3 python/record_lob.py
```
