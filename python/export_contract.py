"""Freeze a LightGBM model and golden vectors for the future C++ implementation.

Export uses an explicit, checksummed input manifest. Verification needs only the
bundle, so it can run in CI without the recording data or a training operation.
"""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import tempfile

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

if __package__:
    from .lob_features import (
        EPS, MAX_GAP_MS, WARMUP_TICKS, HORIZON_TICKS, LABEL_THRESHOLD,
        EMBARGO_TICKS, FEATURE_COLS, RAW_COLS, segment_snapshots,
        engineer_features, modeling_frame,
    )
else:
    from lob_features import (
        EPS, MAX_GAP_MS, WARMUP_TICKS, HORIZON_TICKS, LABEL_THRESHOLD,
        EMBARGO_TICKS, FEATURE_COLS, RAW_COLS, segment_snapshots,
        engineer_features, modeling_frame,
    )

ROOT = Path(__file__).resolve().parents[1]
SCORES = ["raw_down", "raw_flat", "raw_up"]
PROBS = ["p_down", "p_flat", "p_up"]
PAYLOAD_FILES = ["model.txt", "golden_input_lob.csv", "golden_features.csv",
                 "golden_predictions.csv", "training_manifest.json"]
BUNDLE_FILES = PAYLOAD_FILES + ["metadata.json"]
TOLERANCE = 1e-6

# Frozen baseline settings, identical to the existing training pipeline. This
# exporter deliberately does not search hyperparameters or use test-set scores.
MODEL_PARAMS = {
    "objective": "multiclass", "num_class": 3, "metric": "multi_logloss",
    "boosting_type": "gbdt", "learning_rate": 0.05, "num_leaves": 31,
    "max_depth": 5, "min_child_samples": 50, "feature_fraction": 0.8,
    "verbose": -1, "seed": 42, "num_threads": 2,
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def read_reference_csv(path: Path, float_columns) -> pd.DataFrame:
    # Python's float parser preserves 17-digit round trips on pandas 2.x and
    # 3.x. This avoids relying on version-dependent default CSV precision.
    def parse_float(value):
        return float(value) if value else np.nan
    return pd.read_csv(path, converters={name: parse_float for name in float_columns})


def validate_raw(raw: pd.DataFrame) -> None:
    missing = set(RAW_COLS) - set(raw.columns)
    if missing:
        raise ValueError(f"Missing raw columns: {sorted(missing)}")
    if raw.empty or not np.isfinite(raw[RAW_COLS].to_numpy(dtype=np.float64)).all():
        raise ValueError("Snapshots must be nonempty and finite")
    for name in ("timestamp_ms", "last_update_id"):
        values = raw[name].to_numpy()
        if (values < 0).any() or (values != np.floor(values)).any():
            raise ValueError(f"{name} must contain nonnegative integers")
    if (raw.timestamp_ms.diff() < 0).any():
        raise ValueError("Timestamps must be nondecreasing")
    if (raw.last_update_id.diff() <= 0).any():
        raise ValueError("Update IDs must be unique and increasing")
    if (raw.bid_px_1 >= raw.ask_px_1).any():
        raise ValueError("Crossed or locked book")
    for side in ("bid", "ask"):
        prices = raw[[f"{side}_px_{i}" for i in range(1, 11)]].to_numpy()
        sizes = raw[[f"{side}_sz_{i}" for i in range(1, 11)]].to_numpy()
        if (prices <= 0).any() or (sizes <= 0).any():
            raise ValueError("Recorded prices and quantities must be positive")
        steps = np.diff(prices, axis=1)
        if ((steps >= 0) if side == "bid" else (steps <= 0)).any():
            raise ValueError(f"Unordered {side} price ladder")


def load_snapshot(manifest: dict, root: Path = ROOT) -> pd.DataFrame:
    """Read exactly the listed files, including checks against partial changes."""
    if not manifest.get("files"):
        raise ValueError("Manifest must contain a nonempty files list")
    frames, seen = [], set()
    for item in manifest["files"]:
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts or str(relative) in seen:
            raise ValueError(f"Invalid or repeated manifest path: {relative}")
        seen.add(str(relative))
        path = root / relative
        if digest(path) != item["sha256"]:
            raise ValueError(f"Input checksum mismatch: {relative}")
        before = path.stat()
        frame = pd.read_parquet(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f"Input changed while reading: {relative}")
        if len(frame) != item["rows"]:
            raise ValueError(f"Input row count mismatch: {relative}")
        frames.append(frame[RAW_COLS])
    raw = pd.concat(frames, ignore_index=True).sort_values("timestamp_ms").reset_index(drop=True)
    validate_raw(raw)
    return raw


def classify(prob: np.ndarray, tau: float) -> np.ndarray:
    result = np.ones(len(prob), dtype=np.int64)
    result[(prob[:, 2] > tau) & (prob[:, 2] > prob[:, 0])] = 2
    result[(prob[:, 0] > tau) & (prob[:, 0] > prob[:, 2])] = 0
    return result


def train_reference(raw: pd.DataFrame):
    """Recreate the established model once; tune the decision rule on validation."""
    features = engineer_features(segment_snapshots(raw))
    clean = modeling_frame(features)
    n = len(clean)
    train_end, val_end = int(n * .70), int(n * .85)
    train = clean.iloc[:train_end]
    val = clean.iloc[train_end + EMBARGO_TICKS:val_end]
    test = clean.iloc[val_end + EMBARGO_TICKS:]
    if train.empty or val.empty or test.empty or set(train.label.unique()) != {0, 1, 2}:
        raise ValueError("Insufficient labelled data for the established chronological split")
    x = train[FEATURE_COLS].to_numpy()
    xv = val[FEATURE_COLS].to_numpy()
    training = lgb.Dataset(x, label=train.label.to_numpy(), feature_name=FEATURE_COLS)
    validation = lgb.Dataset(xv, label=val.label.to_numpy(), reference=training, feature_name=FEATURE_COLS)
    model = lgb.train(
        MODEL_PARAMS, training, num_boost_round=300, valid_sets=[training, validation],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
    )
    probability = model.predict(xv, num_iteration=model.best_iteration)
    grid = np.linspace(.20, .45, 26)
    scores = [f1_score(val.label, classify(probability, tau), average="macro") for tau in grid]
    tau = float(grid[np.argmax(scores)])
    info = {
        "params": MODEL_PARAMS, "max_boost_rounds": 300, "early_stopping_rounds": 30,
        "best_iteration": model.best_iteration, "raw_rows": len(raw), "clean_rows": n,
        "train_rows": len(train), "validation_rows": len(val), "test_rows": len(test),
        "embargo_ticks": EMBARGO_TICKS, "validation_macro_f1": float(max(scores)),
        "threshold_search": {"min": .20, "max": .45, "step": .01, "split": "validation"},
    }
    # Begin at a true segment boundary so replay does not need hidden history.
    longest = features.segment_id.value_counts().sort_index().idxmax()
    segment = features.loc[features.segment_id == longest, RAW_COLS].reset_index(drop=True)
    info["golden_segment_id"] = int(longest)
    info["golden_segment_available_rows"] = len(segment)
    return model, tau, segment, info


def assert_close(actual, expected, name: str) -> float:
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError(f"Invalid {name} shape or nonfinite values")
    error = float(np.max(np.abs(a - b))) if a.size else 0.
    if error >= TOLERANCE:
        raise ValueError(f"{name} differs by {error}, tolerance is < {TOLERANCE}")
    return error


def write_bundle(model, raw, tau, training_info, manifest, output: Path, force=False):
    """Write, reload and verify in a staging directory before publishing files."""
    output = Path(output)
    if not 0 <= tau <= 1:
        raise ValueError("Threshold must be between 0 and 1")
    validate_raw(raw)
    if len(raw) <= WARMUP_TICKS:
        raise ValueError("Golden input must include at least one ready prediction")
    if model.feature_name() != FEATURE_COLS or model.num_model_per_iteration() != 3:
        raise ValueError("Model must use the fixed eleven features and three classes")
    existing = [name for name in BUNDLE_FILES if (output / name).exists()]
    if existing and not force:
        raise FileExistsError(f"Bundle already exists in {output}; use --force to replace it")
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".export-", dir=output) as temporary:
        staging = Path(temporary)
        model.save_model(staging / "model.txt", num_iteration=model.best_iteration or -1)
        reloaded = lgb.Booster(model_file=str(staging / "model.txt"))
        raw = raw[RAW_COLS].reset_index(drop=True)
        inputs = raw.copy()
        inputs.insert(0, "row_id", np.arange(len(raw), dtype=np.int64))
        features = engineer_features(segment_snapshots(raw))
        ready = features.ready.to_numpy()
        x = features.loc[ready, FEATURE_COLS].to_numpy()
        if not len(x):
            raise ValueError("Golden inputs have no ready rows after segment warm-up")
        expected = features[FEATURE_COLS].copy()
        expected.loc[~ready, FEATURE_COLS] = np.nan
        expected.insert(0, "ready", ready.astype(np.int64))
        expected.insert(0, "row_id", inputs.row_id)
        probabilities = reloaded.predict(x, num_threads=2)
        raw_scores = reloaded.predict(x, raw_score=True, num_threads=2)
        # Check serialization against the in-memory trained model, not just itself.
        assert_close(probabilities, model.predict(x, num_threads=2), "reloaded probabilities")
        assert_close(raw_scores, model.predict(x, raw_score=True, num_threads=2), "reloaded raw scores")
        predictions = pd.DataFrame(raw_scores, columns=SCORES)
        predictions[PROBS] = probabilities
        predictions.insert(0, "row_id", inputs.loc[ready, "row_id"].to_numpy())
        predictions["class_id"] = classify(probabilities, tau)
        for name, frame in [("golden_input_lob.csv", inputs), ("golden_features.csv", expected),
                            ("golden_predictions.csv", predictions)]:
            frame.to_csv(staging / name, index=False, float_format="%.17g", lineterminator="\n")
        write_json(staging / "training_manifest.json", manifest)
        metadata = {
            "schema_version": 1, "feature_names": FEATURE_COLS,
            "classes": {"0": "Down", "1": "Flat", "2": "Up"},
            "max_gap_ms": MAX_GAP_MS, "warmup_ticks": WARMUP_TICKS,
            "eps": EPS, "horizon_ticks": HORIZON_TICKS, "label_threshold": LABEL_THRESHOLD,
            "conviction_tau": tau, "parity_absolute_tolerance": TOLERANCE,
            "floating_point": "float64; CSV uses 17 significant digits",
            "input_rows": len(inputs), "ready_rows": int(ready.sum()),
            "warmup_rows": int((~ready).sum()), "warmup_feature_fields": "empty",
            "prediction_rows": "ready rows only, joined by row_id",
            "num_trees": reloaded.num_trees(), "training": training_info,
            "versions": {p: importlib.metadata.version(p) for p in
                         ["numpy", "pandas", "pyarrow", "lightgbm", "scikit-learn"]},
            "reference_source_sha256": digest(Path(__file__).with_name("lob_features.py")),
            "files": {name: {"sha256": digest(staging / name)} for name in PAYLOAD_FILES},
        }
        write_json(staging / "metadata.json", metadata)
        summary = verify_bundle(staging)
        # Metadata is the last file replaced: incomplete replacements are detected
        # by its hashes. Existing unrelated contract files are left untouched.
        for name in BUNDLE_FILES:
            (staging / name).replace(output / name)
    return summary


def verify_bundle(bundle: Path):
    """Check a portable bundle without loading raw recordings or fitting a model."""
    bundle = Path(bundle)
    meta = json.loads((bundle / "metadata.json").read_text())
    if meta["schema_version"] != 1 or meta["feature_names"] != FEATURE_COLS:
        raise ValueError("Incompatible bundle feature schema")
    if (meta["max_gap_ms"], meta["warmup_ticks"], meta["eps"]) != (MAX_GAP_MS, WARMUP_TICKS, EPS):
        raise ValueError("Bundle does not match the current feature contract")
    for name in PAYLOAD_FILES:
        if digest(bundle / name) != meta["files"][name]["sha256"]:
            raise ValueError(f"Bundle checksum mismatch: {name}")
    inputs = read_reference_csv(bundle / "golden_input_lob.csv", RAW_COLS[2:])
    features = read_reference_csv(bundle / "golden_features.csv", FEATURE_COLS)
    predictions = read_reference_csv(bundle / "golden_predictions.csv", SCORES + PROBS)
    if list(inputs.columns) != ["row_id"] + RAW_COLS:
        raise ValueError("Unexpected golden input columns")
    if list(features.columns) != ["row_id", "ready"] + FEATURE_COLS:
        raise ValueError("Unexpected golden feature columns")
    if list(predictions.columns) != ["row_id"] + SCORES + PROBS + ["class_id"]:
        raise ValueError("Unexpected golden prediction columns")
    if len(inputs) != meta["input_rows"] or len(features) != len(inputs):
        raise ValueError("Golden row count mismatch")
    ids = np.arange(len(inputs), dtype=np.int64)
    if not np.array_equal(inputs.row_id, ids) or not np.array_equal(features.row_id, ids):
        raise ValueError("Golden row IDs are misaligned")
    validate_raw(inputs)
    calculated = engineer_features(segment_snapshots(inputs[RAW_COLS]))
    ready = calculated.ready.to_numpy()
    if not np.array_equal(features.ready, ready.astype(int)):
        raise ValueError("Warm-up readiness flags differ")
    if int(ready.sum()) != meta["ready_rows"] or int((~ready).sum()) != meta["warmup_rows"]:
        raise ValueError("Readiness counts differ")
    if not features.loc[~ready, FEATURE_COLS].isna().all().all():
        raise ValueError("Warm-up features must be empty")
    if not np.array_equal(predictions.row_id, ids[ready]):
        raise ValueError("Prediction row IDs must match ready input rows")
    model = lgb.Booster(model_file=str(bundle / "model.txt"))
    if model.feature_name() != FEATURE_COLS or model.num_model_per_iteration() != 3 or model.num_trees() != meta["num_trees"]:
        raise ValueError("Exported model schema differs")
    x = calculated.loc[ready, FEATURE_COLS].to_numpy()
    error = {"features": assert_close(features.loc[ready, FEATURE_COLS], x, "features")}
    # Evaluate model on the stored CSV features as well as the recomputed ones:
    # small numeric text errors must not cross a tree split threshold silently.
    for label, matrix in [("recomputed", x), ("stored", features.loc[ready, FEATURE_COLS].to_numpy())]:
        probability = model.predict(matrix, num_threads=2)
        scores = model.predict(matrix, raw_score=True, num_threads=2)
        error[f"probabilities_{label}"] = assert_close(predictions[PROBS], probability, "probabilities")
        error[f"raw_scores_{label}"] = assert_close(predictions[SCORES], scores, "raw scores")
        stable = np.exp(scores - scores.max(axis=1, keepdims=True))
        stable /= stable.sum(axis=1, keepdims=True)
        assert_close(stable, probability, "softmax")
        if not np.array_equal(predictions.class_id, classify(probability, meta["conviction_tau"])):
            raise ValueError("Predicted class IDs differ")
    return {"input_rows": len(inputs), "warmup_rows": int((~ready).sum()),
            "ready_rows": int(ready.sum()), "num_trees": model.num_trees(),
            "conviction_tau": meta["conviction_tau"], "max_absolute_errors": error}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="Recreate and freeze the established reference model")
    export.add_argument("--manifest", type=Path, required=True)
    export.add_argument("--output", type=Path, default=ROOT / "contract")
    export.add_argument("--rows", type=int, default=5000)
    export.add_argument("--force", action="store_true")
    verify = commands.add_parser("verify", help="Verify a bundle without retraining")
    verify.add_argument("--bundle", type=Path, default=ROOT / "contract")
    args = parser.parse_args()
    try:
        if args.command == "verify":
            summary = verify_bundle(args.bundle)
        else:
            if args.rows <= WARMUP_TICKS:
                raise ValueError(f"--rows must exceed {WARMUP_TICKS}")
            if not args.force and any((args.output / n).exists() for n in BUNDLE_FILES):
                raise FileExistsError("Output bundle already exists; verify it or explicitly use --force")
            manifest = json.loads(args.manifest.read_text())
            raw = load_snapshot(manifest)
            print(f"Loaded frozen snapshot: {len(raw):,} rows from {len(manifest['files'])} files", flush=True)
            model, tau, segment, info = train_reference(raw)
            if len(segment) < args.rows:
                raise ValueError(f"Longest segment has {len(segment)} rows, fewer than --rows={args.rows}")
            summary = write_bundle(model, segment.iloc[:args.rows], tau, info, manifest, args.output, args.force)
        print(json.dumps(summary, indent=2))
    except (ValueError, OSError, KeyError, lgb.basic.LightGBMError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
