# %% [markdown]
# # Limit Order Book Alpha Pipeline: Data Hygiene, Features, Split & Baselines
# This notebook/script executes Stages A through D of the project:
# 1. Load raw 100ms `BTCUSDT` Parquet chunks & verify book integrity.
# 2. Segment continuous blocks across WebSocket reconnect gaps (`> 1000ms`).
# 3. Engineer the 11 custom microstructure features defined in `contract/feature_spec.md`.
# 4. Construct 3-class future 5s (`t + 50`) target labels with threshold $\theta = \$0.50$.
# 5. Perform a strict chronological Train / Val / Test split with a 100-tick Embargo.
# 6. Evaluate Baseline #1 (Tuned L1 Imbalance Heuristic) and Baseline #2 (Multinomial Logistic Regression).

# %%
import json
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

# Shared formulas also generate the C++ reference bundle.
if __package__:
    from .lob_features import (
        EPS, MAX_GAP_MS, WARMUP_TICKS, HORIZON_TICKS, LABEL_THRESHOLD, EMBARGO_TICKS,
        FEATURE_COLS, prepare_modeling_frame, calendar_split,
    )
else:
    from lob_features import (
        EPS, MAX_GAP_MS, WARMUP_TICKS, HORIZON_TICKS, LABEL_THRESHOLD, EMBARGO_TICKS,
        FEATURE_COLS, prepare_modeling_frame, calendar_split,
    )

if __package__:
    from .export_contract import load_snapshot, classify
else:
    from export_contract import load_snapshot, classify

# %% [markdown]
# ## Step 1: Load Raw Parquet Chunks & Run Data Hygiene Audit

# %%
manifest_path = Path(__file__).resolve().parents[1] / "contract/training_manifest.json"
manifest = json.loads(manifest_path.read_text())
df = load_snapshot(manifest)

# Hygiene checks
crossed_books = (df["bid_px_1"] >= df["ask_px_1"]).sum()
null_count = df.isna().sum().sum()

# Partition into contiguous segments whenever gap > 1000ms
df["time_diff_ms"] = df["timestamp_ms"].diff().fillna(0)
df["segment_id"] = (df["time_diff_ms"] > MAX_GAP_MS).cumsum()

print("  STEP 1: DATA HYGIENE AUDIT  ")
print(f"Total raw snapshots loaded : {len(df):,}")
print(f"Crossed books (bid >= ask) : {crossed_books}")
print(f"Missing / NaN values       : {null_count}")
print(f"Contiguous segments found  : {df['segment_id'].nunique()}")
print("Top 5 largest contiguous segments (rows):")
print(df["segment_id"].value_counts().head(5).to_string())

# %% [markdown]
# Step 2: Engineer the 11 Custom LOB Features & $t+50$ Target Labels

# %%
# Shared preparation applies the same segment rules as the reference exporter.
df_clean = prepare_modeling_frame(df)

print("\n  STEP 2: FEATURE & LABEL SUMMARY  ")
print(f"Clean modeling rows after segment warm-up/horizon trim: {len(df_clean):,}")
label_counts = df_clean["label"].value_counts(normalize=True).sort_index() * 100
print(
    f"Class distribution (theta = ${LABEL_THRESHOLD:.2f}): "
    f"Down (0): {label_counts.get(0, 0):.2f}% | "
    f"Flat (1): {label_counts.get(1, 0):.2f}% | "
    f"Up (2): {label_counts.get(2, 0):.2f}%"
)

# %% [markdown]
# Step 3: Locked Calendar Split with Embargo

# %%
train_df, val_df, test_df = calendar_split(df_clean, df.timestamp_ms.to_numpy())

X_train, y_train = train_df[FEATURE_COLS].to_numpy(), train_df["label"].to_numpy()
X_val, y_val = val_df[FEATURE_COLS].to_numpy(), val_df["label"].to_numpy()
X_test, y_test = test_df[FEATURE_COLS].to_numpy(), test_df["label"].to_numpy()

print("\n=== STEP 3: LOCKED CALENDAR SPLIT (IST) ===")
print(f"Train through Nov 3: {len(train_df):,}; validation Nov 4: {len(val_df):,}; test Nov 5 onward: {len(test_df):,}")
print(f"Cross-boundary targets purged; {EMBARGO_TICKS}-raw-update embargo at each boundary")

# %% [markdown]
#  Step 4: Baseline Models (Always-Flat, Heuristic Rule, and Logistic Regression)

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

print(f"\n   BASELINE 1: LEVEL-1 IMBALANCE RULE (Tuned cutoff c = {best_c:.2f})   ")
print(f"Test Accuracy : {accuracy_score(y_test, y_pred_rule):.4f}")
print(f"Test Macro-F1 : {f1_score(y_test, y_pred_rule, average='macro'):.4f}")
print(classification_report(y_test, y_pred_rule, target_names=["Down (0)", "Flat (1)", "Up (2)"], digits=4))

# Baseline 2: Multinomial Logistic Regression (Standardized Features)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = np.clip(scaler.transform(X_val), -5.0, 5.0)
X_test_scaled = scaler.transform(X_test)

# Clip extreme outliers (> 5 std) so linear model doesn't blow up on $6 spread shocks
X_train_scaled = np.clip(X_train_scaled, -5.0, 5.0)
X_test_scaled = np.clip(X_test_scaled, -5.0, 5.0)

clf_lr = LogisticRegression(max_iter=500, random_state=42)
clf_lr.fit(X_train_scaled, y_train)

y_pred_lr = clf_lr.predict(X_test_scaled)
y_prob_lr = clf_lr.predict_proba(X_test_scaled)

print("\n   BASELINE 2: MULTINOMIAL LOGISTIC REGRESSION   ")
print(f"Test Log-Loss : {log_loss(y_test, y_prob_lr):.4f}")
print(f"Test Accuracy : {accuracy_score(y_test, y_pred_lr):.4f}")
print(f"Test Macro-F1 : {f1_score(y_test, y_pred_lr, average='macro'):.4f}")
print(classification_report(y_test, y_pred_lr, target_names=["Down (0)", "Flat (1)", "Up (2)"], digits=4))

# Select the logistic decision threshold on validation only, just as for LightGBM.
lr_val_prob = clf_lr.predict_proba(X_val_scaled)
lr_grid = np.linspace(0.20, 0.45, 26)
lr_scores = [f1_score(y_val, classify(lr_val_prob, tau), average="macro") for tau in lr_grid]
lr_tau = float(lr_grid[np.argmax(lr_scores)])
y_pred_lr_tuned = classify(y_prob_lr, lr_tau)
print(f"\nLogistic tuned threshold {lr_tau:.2f}: test macro-F1 {f1_score(y_test, y_pred_lr_tuned, average='macro'):.4f}")

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
    "num_threads": 2,
}

model_lgb = lgb.train(
    params,
    train_data,
    num_boost_round=300,
    valid_sets=[train_data, val_data],
    callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
)

print("\n   STEP 5: LIGHTGBM TRAINING   ")
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

print("\n=== STEP 6: OVERLAPPING SIGNAL DIAGNOSTIC (NOT ACCOUNT PNL) ===")
run_backtest(y_pred_flat, "Baseline 0 (Always Flat)")
run_backtest(y_pred_rule, "Baseline 1 (L1 Imbalance Rule)")
run_backtest(y_pred_lr, "Baseline 2 (Logistic Regression)")
run_backtest(y_pred_lr_tuned, "Logistic (Tuned threshold)")
run_backtest(y_pred_lgb_raw, "LightGBM (Raw Argmax)")
run_backtest(y_pred_lgb_tuned, f"LightGBM (Tuned tau={best_tau:.2f})")
