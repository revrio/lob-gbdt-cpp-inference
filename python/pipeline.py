# %% [markdown]
# # Limit Order Book Alpha Pipeline: Data Hygiene, Features, Split & Baselines
# This notebook/script executes Stages A through D of the project:
# 1. Load raw 100ms `BTCUSDT` Parquet chunks & verify book integrity.
# 2. Segment continuous blocks across WebSocket reconnect gaps (`> 500ms`).
# 3. Engineer the 11 custom microstructure features defined in `contract/feature_spec.md`.
# 4. Construct 3-class future 5s (`t + 50`) target labels with threshold $\theta = \$0.50$.
# 5. Perform a strict chronological Train / Val / Test split with a 100-tick Embargo.
# 6. Evaluate Baseline #1 (Tuned L1 Imbalance Heuristic) and Baseline #2 (Multinomial Logistic Regression).

# %%
import glob
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.preprocessing import StandardScaler

EPS = 1e-9
WARMUP_TICKS = 300       # 30s warm-up for ofi_300 rolling window
HORIZON_TICKS = 50       # 5s prediction horizon
LABEL_THRESHOLD = 0.50   # $0.50 move required for Up (+1) or Down (-1)
EMBARGO_TICKS = 100      # 10s buffer (> 50-tick label horizon) between splits

FEATURE_COLS = [
    "spread",
    "micro_minus_mid",
    "imb_l1",
    "imb_l2_3",
    "imb_l4_10",
    "span_asym",
    "ofi_10",
    "ofi_50",
    "ofi_300",
    "ret_50",
    "vol_50",
]

# %% [markdown]
# ## Step 1: Load Raw Parquet Chunks & Run Data Hygiene Audit

# %%
raw_files = sorted(glob.glob("data/raw/*.parquet"))
if not raw_files:
    raise FileNotFoundError("No .parquet files found in data/raw/. Run record_lob.py first.")

df = pd.concat([pd.read_parquet(f) for f in raw_files], ignore_index=True)
df = df.sort_values("timestamp_ms").reset_index(drop=True)

# Hygiene checks
crossed_books = (df["bid_px_1"] >= df["ask_px_1"]).sum()
null_count = df.isna().sum().sum()

# Partition into contiguous segments whenever gap > 500ms
df["time_diff_ms"] = df["timestamp_ms"].diff().fillna(0)
df["segment_id"] = (df["time_diff_ms"] > 500).cumsum()

print("=== STEP 1: DATA HYGIENE AUDIT ===")
print(f"Total raw snapshots loaded : {len(df):,}")
print(f"Crossed books (bid >= ask) : {crossed_books}")
print(f"Missing / NaN values       : {null_count}")
print(f"Contiguous segments found  : {df['segment_id'].nunique()}")
print("Top 5 largest contiguous segments (rows):")
print(df["segment_id"].value_counts().head(5).to_string())

# %% [markdown]
# ## Step 2: Engineer the 11 Custom LOB Features & $t+50$ Target Labels

# %%
# 2A. Stateless Snapshot Features (Vectorized across entire table)
df["mid_px"] = (df["bid_px_1"] + df["ask_px_1"]) / 2.0
df["spread"] = df["ask_px_1"] - df["bid_px_1"]

b1, a1 = df["bid_sz_1"], df["ask_sz_1"]
df["micro_minus_mid"] = ((df["bid_px_1"] * a1 + df["ask_px_1"] * b1) / (b1 + a1 + EPS)) - df["mid_px"]

# Non-overlapping volume imbalance zones: L1, L2..3, L4..10
b2_3 = df[["bid_sz_2", "bid_sz_3"]].sum(axis=1)
a2_3 = df[["ask_sz_2", "ask_sz_3"]].sum(axis=1)
b4_10 = df[[f"bid_sz_{i}" for i in range(4, 11)]].sum(axis=1)
a4_10 = df[[f"ask_sz_{i}" for i in range(4, 11)]].sum(axis=1)

df["imb_l1"] = (b1 - a1) / (b1 + a1 + EPS)
df["imb_l2_3"] = (b2_3 - a2_3) / (b2_3 + a2_3 + EPS)
df["imb_l4_10"] = (b4_10 - a4_10) / (b4_10 + a4_10 + EPS)

# Book price span asymmetry across all 10 levels
ask_span = df["ask_px_10"] - df["ask_px_1"]
bid_span = df["bid_px_1"] - df["bid_px_10"]
df["span_asym"] = (ask_span - bid_span) / (ask_span + bid_span + EPS)


# 2B. Stateful Rolling Features & Future Target Label (Strictly inside each segment_id)
def compute_segment_stateful(g: pd.DataFrame) -> pd.DataFrame:
    if len(g) <= WARMUP_TICKS + HORIZON_TICKS:
        return pd.DataFrame()

    bp, bs = g["bid_px_1"].to_numpy(), g["bid_sz_1"].to_numpy()
    ap, asz = g["ask_px_1"].to_numpy(), g["ask_sz_1"].to_numpy()
    mid = g["mid_px"].to_numpy()

    # 1-tick Cont-Kukanov-Stoikov OFI
    wb = np.zeros(len(g), dtype=np.float64)
    wa = np.zeros(len(g), dtype=np.float64)

    bp_diff = bp[1:] - bp[:-1]
    ap_diff = ap[1:] - ap[:-1]

    wb[1:] = np.where(bp_diff > 0, bs[1:], np.where(bp_diff == 0, bs[1:] - bs[:-1], -bs[:-1]))
    wa[1:] = np.where(ap_diff < 0, asz[1:], np.where(ap_diff == 0, asz[1:] - asz[:-1], -asz[:-1]))
    ofi_1 = pd.Series(wb - wa, index=g.index)

    g = g.copy()
    g["ofi_10"] = ofi_1.rolling(10).sum()
    g["ofi_50"] = ofi_1.rolling(50).sum()
    g["ofi_300"] = ofi_1.rolling(300).sum()

    # 50-tick past price change & 50-tick realized volatility (ddof=0)
    dmid = pd.Series(np.r_[0.0, np.diff(mid)], index=g.index)
    g["ret_50"] = g["mid_px"] - g["mid_px"].shift(50)
    g["vol_50"] = dmid.rolling(50).std(ddof=0)

    # Future 5-second (50-tick) mid-price change
    g["future_ret_50"] = g["mid_px"].shift(-HORIZON_TICKS) - g["mid_px"]

    # Trim warm-up ticks at start and unlabelled horizon ticks at end of segment
    return g.iloc[WARMUP_TICKS : len(g) - HORIZON_TICKS]


df_clean = (
    df.groupby("segment_id", group_keys=False)
    .apply(compute_segment_stateful)
    .reset_index(drop=True)
)

# Encode 3 classes: 0 = Down (< -$0.50), 1 = Flat ([-$0.50, +$0.50]), 2 = Up (> +$0.50)
df_clean["label"] = np.where(
    df_clean["future_ret_50"] > LABEL_THRESHOLD,
    2,
    np.where(df_clean["future_ret_50"] < -LABEL_THRESHOLD, 0, 1),
)

print("\n=== STEP 2: FEATURE & LABEL SUMMARY ===")
print(f"Clean modeling rows after segment warm-up/horizon trim: {len(df_clean):,}")
label_counts = df_clean["label"].value_counts(normalize=True).sort_index() * 100
print(
    f"Class distribution (theta = ${LABEL_THRESHOLD:.2f}): "
    f"Down (0): {label_counts.get(0, 0):.2f}% | "
    f"Flat (1): {label_counts.get(1, 0):.2f}% | "
    f"Up (2): {label_counts.get(2, 0):.2f}%"
)

# %% [markdown]
# ## Step 3: Time-Ordered Split with Embargo (70% Train / 15% Val / 15% Test)

# %%
n_total = len(df_clean)
train_end = int(n_total * 0.70)
val_start = train_end + EMBARGO_TICKS
val_end = int(n_total * 0.85)
test_start = val_end + EMBARGO_TICKS

train_df = df_clean.iloc[:train_end].reset_index(drop=True)
val_df = df_clean.iloc[val_start:val_end].reset_index(drop=True)
test_df = df_clean.iloc[test_start:].reset_index(drop=True)

X_train, y_train = train_df[FEATURE_COLS].to_numpy(), train_df["label"].to_numpy()
X_val, y_val = val_df[FEATURE_COLS].to_numpy(), val_df["label"].to_numpy()
X_test, y_test = test_df[FEATURE_COLS].to_numpy(), test_df["label"].to_numpy()

print("\n=== STEP 3: CHRONOLOGICAL SPLIT WITH EMBARGO ===")
print(f"Train Set : {len(train_df):,} rows (indices 0 .. {train_end - 1})")
print(f"Embargo 1 : {EMBARGO_TICKS} rows dropped")
print(f"Val Set   : {len(val_df):,} rows (indices {val_start} .. {val_end - 1})")
print(f"Embargo 2 : {EMBARGO_TICKS} rows dropped")
print(f"Test Set  : {len(test_df):,} rows (indices {test_start} .. {n_total - 1})")

# %% [markdown]
# ## Step 4: Baseline Models (Always-Flat, Heuristic Rule, and Logistic Regression)

# %%
# Baseline 0: Trivial Majority Class ("Always Predict Flat")
y_pred_flat = np.ones_like(y_test)
print("\n=== BASELINE 0: ALWAYS PREDICT FLAT ===")
print(f"Test Accuracy : {accuracy_score(y_test, y_pred_flat):.4f}")
print(f"Test Macro-F1 : {f1_score(y_test, y_pred_flat, average='macro'):.4f}")

# Baseline 1: Level-1 Imbalance Threshold Rule
# Tune threshold c on Validation Set to maximize Macro-F1
imb_val = val_df["imb_l1"].to_numpy()
best_c, best_val_f1 = 0.5, 0.0
for c in np.linspace(0.10, 0.95, 18):
    preds_val = np.where(imb_val > c, 2, np.where(imb_val < -c, 0, 1))
    score = f1_score(y_val, preds_val, average="macro")
    if score > best_val_f1:
        best_val_f1 = score
        best_c = c

imb_test = test_df["imb_l1"].to_numpy()
y_pred_rule = np.where(imb_test > best_c, 2, np.where(imb_test < -best_c, 0, 1))

print(f"\n=== BASELINE 1: LEVEL-1 IMBALANCE RULE (Tuned cutoff c = {best_c:.2f}) ===")
print(f"Test Accuracy : {accuracy_score(y_test, y_pred_rule):.4f}")
print(f"Test Macro-F1 : {f1_score(y_test, y_pred_rule, average='macro'):.4f}")
print(classification_report(y_test, y_pred_rule, target_names=["Down (0)", "Flat (1)", "Up (2)"], digits=4))

# Baseline 2: Multinomial Logistic Regression (Standardized Features)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Clip extreme outliers (> 5 std) so linear model doesn't blow up on $6 spread shocks
X_train_scaled = np.clip(X_train_scaled, -5.0, 5.0)
X_test_scaled = np.clip(X_test_scaled, -5.0, 5.0)

clf_lr = LogisticRegression(max_iter=500, random_state=42)
clf_lr.fit(X_train_scaled, y_train)

y_pred_lr = clf_lr.predict(X_test_scaled)
y_prob_lr = clf_lr.predict_proba(X_test_scaled)

print("\n=== BASELINE 2: MULTINOMIAL LOGISTIC REGRESSION ===")
print(f"Test Log-Loss : {log_loss(y_test, y_prob_lr):.4f}")
print(f"Test Accuracy : {accuracy_score(y_test, y_pred_lr):.4f}")
print(f"Test Macro-F1 : {f1_score(y_test, y_pred_lr, average='macro'):.4f}")
print(classification_report(y_test, y_pred_lr, target_names=["Down (0)", "Flat (1)", "Up (2)"], digits=4))

# %% [markdown]
# ## Step 5: LightGBM GBDT Multi-Class Classifier & Threshold Tuning

# %%
train_data = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_COLS)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, feature_name=FEATURE_COLS)

params = {
    "objective": "multiclass",
    "num_class": 3,
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": 5,
    "min_child_samples": 50,
    "feature_fraction": 0.8,
    "verbose": -1,
    "seed": 42,
}

model_lgb = lgb.train(
    params,
    train_data,
    num_boost_round=300,
    valid_sets=[train_data, val_data],
    callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
)

print("\n=== STEP 5: LIGHTGBM TRAINING ===")
print(f"Optimal iteration: {model_lgb.best_iteration}")

# Raw Argmax Predictions
probs_val = model_lgb.predict(X_val)
probs_test = model_lgb.predict(X_test)
y_pred_lgb_raw = np.argmax(probs_test, axis=1)

print("\n--- LightGBM (Raw Argmax on Test) ---")
print(f"Test Log-Loss : {log_loss(y_test, probs_test):.4f}")
print(f"Test Accuracy : {accuracy_score(y_test, y_pred_lgb_raw):.4f}")
print(f"Test Macro-F1 : {f1_score(y_test, y_pred_lgb_raw, average='macro'):.4f}")

# Tune Conviction Probability Threshold strictly on Validation Set
best_tau, best_val_f1 = 0.33, 0.0
for tau in np.linspace(0.20, 0.45, 26):
    preds_v = np.ones(len(y_val), dtype=int)
    up_m = (probs_val[:, 2] > tau) & (probs_val[:, 2] > probs_val[:, 0])
    down_m = (probs_val[:, 0] > tau) & (probs_val[:, 0] > probs_val[:, 2])
    preds_v[up_m] = 2
    preds_v[down_m] = 0
    score = f1_score(y_val, preds_v, average="macro")
    if score > best_val_f1:
        best_val_f1 = score
        best_tau = tau

# Apply Tuned Threshold to Test Set
y_pred_lgb_tuned = np.ones(len(y_test), dtype=int)
up_test = (probs_test[:, 2] > best_tau) & (probs_test[:, 2] > probs_test[:, 0])
down_test = (probs_test[:, 0] > best_tau) & (probs_test[:, 0] > probs_test[:, 2])
y_pred_lgb_tuned[up_test] = 2
y_pred_lgb_tuned[down_test] = 0

print(f"\n--- LightGBM (Tuned Conviction Threshold tau = {best_tau:.3f}) ---")
print(f"Val Macro-F1  : {best_val_f1:.4f}")
print(f"Test Accuracy : {accuracy_score(y_test, y_pred_lgb_tuned):.4f}")
print(f"Test Macro-F1 : {f1_score(y_test, y_pred_lgb_tuned, average='macro'):.4f}")
print(classification_report(y_test, y_pred_lgb_tuned, target_names=["Down (0)", "Flat (1)", "Up (2)"], digits=4))

# Feature Importances
importance_df = pd.DataFrame({
    "feature": FEATURE_COLS,
    "split_importance": model_lgb.feature_importance(importance_type="split"),
    "gain_importance": model_lgb.feature_importance(importance_type="gain"),
}).sort_values("gain_importance", ascending=False).reset_index(drop=True)

print("\nFeature Importances (by Gain):")
print(importance_df.to_string())

# %% [markdown]
# ## Step 6: Economic Backtest (5s Horizon Mid-Price Return Simulation)

# %%
future_ret_test = test_df["future_ret_50"].to_numpy()
half_spread_cost = (test_df["spread"] / 2.0).to_numpy()

def run_backtest(preds: np.ndarray, model_name: str):
    pos = np.where(preds == 2, 1.0, np.where(preds == 0, -1.0, 0.0))
    trades = np.count_nonzero(pos)
    if trades == 0:
        print(f"{model_name:32s} | Trades: {0:5d} ( 0.0%) | WinRate:   0.0% | Gross PnL: $     0.00 | Net PnL (Spread): $     0.00")
        return

    gross = np.sum(pos * future_ret_test)
    slippage = np.sum(np.abs(pos) * half_spread_cost)
    net = gross - slippage
    wins = np.sum((pos * future_ret_test) > 0)
    win_rate = (wins / trades) * 100

    print(f"{model_name:32s} | Trades: {trades:5d} ({trades/len(preds)*100:4.1f}%) | WinRate: {win_rate:5.1f}% | Gross PnL: ${gross:9.2f} | Net PnL (Spread): ${net:9.2f}")

print("\n=== STEP 6: ECONOMIC BACKTEST ON TEST SET (18,171 TICKS) ===")
run_backtest(y_pred_flat, "Baseline 0 (Always Flat)")
run_backtest(y_pred_rule, "Baseline 1 (L1 Imbalance Rule)")
run_backtest(y_pred_lr, "Baseline 2 (Logistic Regression)")
run_backtest(y_pred_lgb_raw, "LightGBM (Raw Argmax)")
run_backtest(y_pred_lgb_tuned, f"LightGBM (Tuned tau={best_tau:.2f})")