"""Contract boundaries and serialization checks, independent of recording files."""

import json
from pathlib import Path
import tempfile
import unittest

import lightgbm as lgb
import numpy as np
import pandas as pd

from python.export_contract import (
    FEATURE_COLS, RAW_COLS, classify, digest, load_snapshot,
    verify_bundle, write_bundle,
)
from python.lob_features import engineer_features, modeling_frame, segment_snapshots


def book_sequence(n=340):
    data = {"timestamp_ms": np.arange(n, dtype=np.int64) * 100,
            "last_update_id": np.arange(n, dtype=np.int64) + 1}
    for level in range(1, 11):
        data[f"bid_px_{level}"] = np.full(n, 101.0 - level)
        data[f"ask_px_{level}"] = np.full(n, 100.0 + level)
        data[f"bid_sz_{level}"] = np.full(n, 3.0 if level == 1 else 1.0 if level <= 3 else 2.0)
        data[f"ask_sz_{level}"] = np.full(n, 1.0 if level == 1 else 2.0 if level <= 3 else 1.0)
    return pd.DataFrame(data)[RAW_COLS]


class FeatureReferenceTests(unittest.TestCase):
    def test_hand_computed_book_and_one_price_jump(self):
        raw = book_sequence(301)
        prices = [name for name in RAW_COLS if "_px_" in name]
        raw.loc[300, prices] += 1.0
        f = engineer_features(segment_snapshots(raw))
        self.assertFalse(f.ready.iloc[:300].any())
        self.assertTrue(f.ready.iloc[300])
        row = f.iloc[300]
        self.assertAlmostEqual(row.spread, 1.)
        self.assertAlmostEqual(row.micro_minus_mid, .25, places=6)
        self.assertAlmostEqual(row.imb_l1, .5)
        self.assertAlmostEqual(row.imb_l2_3, -1 / 3)
        self.assertAlmostEqual(row.imb_l4_10, 1 / 3)
        self.assertEqual(row.span_asym, 0.)
        for window in (10, 50, 300):
            self.assertEqual(row[f"ofi_{window}"], 4.)
        self.assertEqual(row.ret_50, 1.)
        self.assertAlmostEqual(row.vol_50, .14)

    def test_gap_boundary_and_new_segment_warmup(self):
        raw = book_sequence(641)
        raw.loc[320:, "timestamp_ms"] += 400  # Exactly 500ms: stays in segment.
        self.assertTrue(engineer_features(segment_snapshots(raw)).ready.iloc[320])
        raw.loc[320:, "timestamp_ms"] += 1  # 501ms: reset.
        f = engineer_features(segment_snapshots(raw))
        self.assertFalse(f.ready.iloc[320:620].any())
        self.assertTrue(f.ready.iloc[620:].all())
        self.assertEqual(f.loc[620, "ofi_300"], 0.)

    def test_equal_timestamps_allowed_but_decreasing_rejected(self):
        raw = book_sequence()
        raw.loc[10, "timestamp_ms"] = raw.loc[9, "timestamp_ms"]
        segment_snapshots(raw)
        raw.loc[10, "timestamp_ms"] -= 1
        with self.assertRaisesRegex(ValueError, "nondecreasing"):
            segment_snapshots(raw)

    def test_inference_tail_does_not_need_future_labels(self):
        f = engineer_features(segment_snapshots(book_sequence(340)))
        self.assertEqual(int(f.ready.sum()), 40)
        self.assertTrue(f.ready.iloc[-1])
        self.assertTrue(modeling_frame(f).empty)

    def test_training_trim_matches_contract(self):
        f = engineer_features(segment_snapshots(book_sequence(500)))
        training = modeling_frame(f)
        self.assertEqual(len(training), 150)
        self.assertEqual(training.timestamp_ms.iloc[0], 30_000)
        self.assertEqual(training.timestamp_ms.iloc[-1], 44_900)
        self.assertTrue((training.label == 1).all())


class ExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(90, 11))
        y = np.tile([0, 1, 2], 30)
        cls.model = lgb.train(
            {"objective": "multiclass", "num_class": 3, "verbosity": -1,
             "num_threads": 1, "min_data_in_leaf": 1, "num_leaves": 4, "seed": 42},
            lgb.Dataset(x, label=y, feature_name=FEATURE_COLS), num_boost_round=3,
        )

    def test_roundtrip_and_explicit_warmup_alignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            summary = write_bundle(self.model, book_sequence(340), .39, {}, {"files": []}, out)
            self.assertEqual(summary["warmup_rows"], 300)
            self.assertEqual(summary["ready_rows"], 40)
            self.assertEqual(summary["num_trees"], 9)
            self.assertLess(max(summary["max_absolute_errors"].values()), 1e-12)
            features = pd.read_csv(out / "golden_features.csv")
            predictions = pd.read_csv(out / "golden_predictions.csv")
            self.assertTrue(features.loc[:299, FEATURE_COLS].isna().all().all())
            np.testing.assert_array_equal(predictions.row_id, np.arange(300, 340))
            # It refuses replacement and preserves unrelated contract documents.
            (out / "feature_spec.md").write_text("keep this document")
            with self.assertRaises(FileExistsError):
                write_bundle(self.model, book_sequence(), .39, {}, {}, out)
            self.assertEqual((out / "feature_spec.md").read_text(), "keep this document")

    def test_model_bundle_with_a_reset(self):
        raw = book_sequence(641)
        raw.loc[320:, "timestamp_ms"] += 401
        with tempfile.TemporaryDirectory() as tmp:
            summary = write_bundle(self.model, raw, .39, {}, {}, Path(tmp))
            self.assertEqual(summary["warmup_rows"], 600)
            self.assertEqual(summary["ready_rows"], 41)

    def test_corrupt_payload_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            write_bundle(self.model, book_sequence(), .39, {}, {}, out)
            with (out / "golden_predictions.csv").open("a") as stream:
                stream.write("corruption\n")
            with self.assertRaisesRegex(ValueError, "checksum"):
                verify_bundle(out)

    def test_prediction_alignment_checked_after_hash_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            write_bundle(self.model, book_sequence(), .39, {}, {}, out)
            path = out / "golden_predictions.csv"
            frame = pd.read_csv(path)
            frame.loc[0, "row_id"] = 299
            frame.to_csv(path, index=False)
            meta = json.loads((out / "metadata.json").read_text())
            meta["files"][path.name]["sha256"] = digest(path)
            (out / "metadata.json").write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, "Prediction row IDs"):
                verify_bundle(out)

    def test_input_manifest_detects_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "raw.parquet"
            book_sequence().to_parquet(path, index=False)
            manifest = {"files": [{"path": path.name, "rows": 340, "sha256": digest(path)}]}
            self.assertEqual(len(load_snapshot(manifest, root)), 340)
            path.write_bytes(path.read_bytes() + b"changed")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_snapshot(manifest, root)

    def test_direction_threshold_boundaries_and_ties(self):
        probability = np.array([[.39, .60, .01], [.4, .6, 0], [.3, .3, .4], [.45, .1, .45]])
        np.testing.assert_array_equal(classify(probability, .39), [1, 0, 2, 1])


if __name__ == "__main__":
    unittest.main()
