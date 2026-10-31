"""Python reference calculations shared by training and C++ fixture export.

Feature calculation includes every snapshot. Training alone trims unlabelled
tails; a live/replay predictor doesn't need future prices to emit a prediction.
"""

import numpy as np
import pandas as pd

EPS = 1e-9
MAX_GAP_MS = 500
WARMUP_TICKS = 300
HORIZON_TICKS = 50
LABEL_THRESHOLD = 0.50
EMBARGO_TICKS = 100

FEATURE_COLS = [
    "spread", "micro_minus_mid", "imb_l1", "imb_l2_3", "imb_l4_10",
    "span_asym", "ofi_10", "ofi_50", "ofi_300", "ret_50", "vol_50",
]
RAW_COLS = ["timestamp_ms", "last_update_id"] + [
    f"{side}_{field}_{level}"
    for level in range(1, 11)
    for side in ("bid", "ask")
    for field in ("px", "sz")
]


def segment_snapshots(raw: pd.DataFrame, max_gap_ms: int = MAX_GAP_MS) -> pd.DataFrame:
    """Mark segment boundaries in already ordered data; equal timestamps are valid."""
    df = raw.copy()
    df["time_diff_ms"] = df["timestamp_ms"].diff().fillna(0)
    if (df["time_diff_ms"] < 0).any():
        raise ValueError("Snapshots must have nondecreasing timestamps")
    df["segment_id"] = (df["time_diff_ms"] > max_gap_ms).cumsum()
    return df


def engineer_features(segmented: pd.DataFrame) -> pd.DataFrame:
    """Calculate all eleven features, preserving the caller's segment boundaries.

    `ready` is false for positions 0..299 of each segment. Warm-up feature values
    are implementation intermediates and must not be submitted to the model.
    """
    df = segmented.copy()
    df["mid_px"] = (df["bid_px_1"] + df["ask_px_1"]) / 2.0
    df["spread"] = df["ask_px_1"] - df["bid_px_1"]
    b1, a1 = df["bid_sz_1"], df["ask_sz_1"]
    df["micro_minus_mid"] = (
        (df["bid_px_1"] * a1 + df["ask_px_1"] * b1) / (b1 + a1 + EPS)
    ) - df["mid_px"]
    b23 = df[["bid_sz_2", "bid_sz_3"]].sum(axis=1)
    a23 = df[["ask_sz_2", "ask_sz_3"]].sum(axis=1)
    b410 = df[[f"bid_sz_{i}" for i in range(4, 11)]].sum(axis=1)
    a410 = df[[f"ask_sz_{i}" for i in range(4, 11)]].sum(axis=1)
    df["imb_l1"] = (b1 - a1) / (b1 + a1 + EPS)
    df["imb_l2_3"] = (b23 - a23) / (b23 + a23 + EPS)
    df["imb_l4_10"] = (b410 - a410) / (b410 + a410 + EPS)
    ask_span = df["ask_px_10"] - df["ask_px_1"]
    bid_span = df["bid_px_1"] - df["bid_px_10"]
    df["span_asym"] = (ask_span - bid_span) / (ask_span + bid_span + EPS)

    pieces = []
    for _, group in df.groupby("segment_id", sort=True):
        g = group.copy()
        bp, bs = g["bid_px_1"].to_numpy(), g["bid_sz_1"].to_numpy()
        ap, az = g["ask_px_1"].to_numpy(), g["ask_sz_1"].to_numpy()
        mid = g["mid_px"].to_numpy()
        wb, wa = np.zeros(len(g)), np.zeros(len(g))
        bd, ad = np.diff(bp), np.diff(ap)
        wb[1:] = np.where(bd > 0, bs[1:], np.where(bd == 0, bs[1:] - bs[:-1], -bs[:-1]))
        wa[1:] = np.where(ad < 0, az[1:], np.where(ad == 0, az[1:] - az[:-1], -az[:-1]))
        ofi = pd.Series(wb - wa, index=g.index)
        for window in (10, 50, 300):
            g[f"ofi_{window}"] = ofi.rolling(window).sum()
        dmid = pd.Series(np.r_[0.0, np.diff(mid)], index=g.index)
        g["ret_50"] = g["mid_px"] - g["mid_px"].shift(50)
        g["vol_50"] = dmid.rolling(50).std(ddof=0)
        g["ready"] = np.arange(len(g)) >= WARMUP_TICKS
        pieces.append(g)
    if not pieces:
        raise ValueError("No snapshots supplied")
    return pd.concat(pieces).reset_index(drop=True)


def modeling_frame(features: pd.DataFrame) -> pd.DataFrame:
    """Select ready rows whose 50-update future target stays inside the segment."""
    df = features.copy()
    future_mid = df.groupby("segment_id")["mid_px"].shift(-HORIZON_TICKS)
    df["future_ret_50"] = future_mid - df["mid_px"]
    df = df.loc[df["ready"] & df["future_ret_50"].notna()].copy()
    df["label"] = np.where(
        df["future_ret_50"] > LABEL_THRESHOLD, 2,
        np.where(df["future_ret_50"] < -LABEL_THRESHOLD, 0, 1),
    )
    return df.reset_index(drop=True)
