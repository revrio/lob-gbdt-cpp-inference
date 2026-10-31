# Feature Specification Contract (`Python <-> C++`)

This document serves as the immutable contract between the Python offline modeling pipeline and the C++20 low-latency inference engine. Any change to feature ordering, window length, or edge-case handling must be updated here first and verified via `ctest`.

---

## 1. Stream & Horizon Parameters

| Parameter | Value | Notes |
| :--- | :--- | :--- |
| **Instrument** | `BTCUSDT` (Binance Spot) | `@depth10@100ms` WebSocket stream |
| **Nominal Tick Interval** | `100 ms` (`10 ticks = 1s`) | Rolling windows are defined strictly by integer tick count |
| **Contiguous Segment Rule** | `timestamp_ms[t] - timestamp_ms[t-1] <= 500` | Any gap `> 500 ms` begins a new segment (resets ring buffers) |
| **Warm-up Period** | `300 ticks` (`~30 seconds`) | First 300 ticks of any segment warm up buffers; no predictions emitted |
| **Prediction Horizon ($k$)** | `50 ticks` (`~5 seconds`) | Target evaluates `mid_px[t + 50] - mid_px[t]` |
| **Label Threshold ($\theta$)** | `$0.50` (`50 cents`) | `Down (0): < -0.50`, `Flat (1): [-0.50, +0.50]`, `Up (2): > +0.50` |

---

## 2. Numerical & Edge-Case Rules (Parity Guarantees)

1. **Floating-Point Type:** Double precision (`double` in C++, `float64` in Python/NumPy).
2. **Zero-Denominator Epsilon:** All bounded ratio features add `EPS = 1e-9` to the denominator (`(num) / (den + 1e-9)`) to guarantee zero division-by-zero branches in C++.
3. **First Tick of Segment (`t = 0`):** For tick `0` of any segment, 1-tick Order Flow Imbalance `ofi_1[0] = 0.0` and 1-tick mid-price change `dmid[0] = 0.0`.
4. **Standard Deviation Formula (`vol_50`):** Uses population standard deviation (`ddof = 0`), defined over $N = 50$ ticks of `dmid[t] = mid_px[t] - mid_px[t-1]` as:
   $$\sigma_{50} = \sqrt{\max\left(0.0, \frac{1}{50}\sum_{i=0}^{49} \Delta\text{mid}_{t-i}^2 - \left(\frac{1}{50}\sum_{i=0}^{49} \Delta\text{mid}_{t-i}\right)^2\right)}$$
5. **LightGBM Split Direction:** A node branches to `left_child` when `feature_value <= threshold`, and to `right_child` when `feature_value > threshold`.
6. **Parity Tolerance:** Maximum absolute difference between Python and C++ across all features and output probabilities must be `< 1e-6`.

---

## 3. Preliminary Notation

At tick index $t$, given top-10 bid prices $P_{1}^B \dots P_{10}^B$, bid sizes $q_{1}^B \dots q_{10}^B$, ask prices $P_{1}^A \dots P_{10}^A$, and ask sizes $q_{1}^A \dots q_{10}^A$:

* **Mid-Price:**
  $$P_{\text{mid}}(t) = \frac{P_1^B(t) + P_1^A(t)}{2.0}$$
* **1-Tick Cont-Kukanov-Stoikov Order Flow Imbalance ($e_t$):**
  $$\Delta W_t^B = \begin{cases} q_1^B(t), & P_1^B(t) > P_1^B(t-1) \\ q_1^B(t) - q_1^B(t-1), & P_1^B(t) = P_1^B(t-1) \\ -q_1^B(t-1), & P_1^B(t) < P_1^B(t-1) \end{cases}$$
  $$\Delta W_t^A = \begin{cases} q_1^A(t), & P_1^A(t) < P_1^A(t-1) \\ q_1^A(t) - q_1^A(t-1), & P_1^A(t) = P_1^A(t-1) \\ -q_1^A(t-1), & P_1^A(t) > P_1^A(t-1) \end{cases}$$
  $$e_t = \Delta W_t^B - \Delta W_t^A \quad (\text{with } e_0 = 0.0)$$

---

## 4. Feature Vector Specification (`NUM_FEATURES = 11`)

| Array Index | Feature Name | Exact Mathematical Formula | Window (Ticks) | C++ State Requirement |
| :--- | :--- | :--- | :--- | :--- |
| **`0`** | `spread` | $P_1^A(t) - P_1^B(t)$ | `1` (Stateless) | None |
| **`1`** | `micro_minus_mid` | $\frac{P_1^B(t) q_1^A(t) + P_1^A(t) q_1^B(t)}{q_1^B(t) + q_1^A(t) + 10^{-9}} - P_{\text{mid}}(t)$ | `1` (Stateless) | None |
| **`2`** | `imb_l1` | $\frac{q_1^B(t) - q_1^A(t)}{q_1^B(t) + q_1^A(t) + 10^{-9}}$ | `1` (Stateless) | None |
| **`3`** | `imb_l2_3` | $\frac{\sum_{i=2}^3 q_i^B(t) - \sum_{i=2}^3 q_i^A(t)}{\sum_{i=2}^3 q_i^B(t) + \sum_{i=2}^3 q_i^A(t) + 10^{-9}}$ | `1` (Stateless) | None |
| **`4`** | `imb_l4_10` | $\frac{\sum_{i=4}^{10} q_i^B(t) - \sum_{i=4}^{10} q_i^A(t)}{\sum_{i=4}^{10} q_i^B(t) + \sum_{i=4}^{10} q_i^A(t) + 10^{-9}}$ | `1` (Stateless) | None |
| **`5`** | `span_asym` | $\frac{(P_{10}^A(t) - P_1^A(t)) - (P_1^B(t) - P_{10}^B(t))}{(P_{10}^A(t) - P_1^A(t)) + (P_1^B(t) - P_{10}^B(t)) + 10^{-9}}$ | `1` (Stateless) | None |
| **`6`** | `ofi_10` | $\sum_{k=0}^{9} e_{t-k}$ | `10` (`~1s`) | `RingBuffer<double, 10>` + running sum |
| **`7`** | `ofi_50` | $\sum_{k=0}^{49} e_{t-k}$ | `50` (`~5s`) | `RingBuffer<double, 50>` + running sum |
| **`8`** | `ofi_300` | $\sum_{k=0}^{299} e_{t-k}$ | `300` (`~30s`) | `RingBuffer<double, 300>` + running sum |
| **`9`** | `ret_50` | $P_{\text{mid}}(t) - P_{\text{mid}}(t-50)$ | `51` (`~5s`) | `RingBuffer<double, 51>` of `mid_px` |
| **`10`** | `vol_50` | $\text{std}_{\text{pop}}(\Delta P_{\text{mid}}(t-49 \dots t), \text{ddof}=0)$ | `50` (`~5s`) | `RingBuffer<double, 50>` + $\sum x, \sum x^2$ |

---

## 5. Reference Bundle (Schema Version 1)

`python/export_contract.py` exports these files together:

| File | Contents |
| :--- | :--- |
| `model.txt` | Native LightGBM model, saved at its best validation iteration |
| `golden_input_lob.csv` | Zero-based `row_id`, integer `timestamp_ms`, integer `last_update_id`, then the 40 price/quantity columns in recorder order: `bid_px_1,bid_sz_1,ask_px_1,ask_sz_1,...,ask_sz_10` |
| `golden_features.csv` | `row_id,ready`, followed by the 11 features in section 4 order |
| `golden_predictions.csv` | Ready rows only: `row_id,raw_down,raw_flat,raw_up,p_down,p_flat,p_up,class_id` |
| `training_manifest.json` | Frozen source-file list with SHA-256 hashes and row counts |
| `metadata.json` | Schema version, feature order, class mapping, numerical constants, selected threshold, training settings, package versions, and payload hashes |

All numeric CSV outputs use 17 significant digits. Readers must preserve float64
round-trip precision and 64-bit integer IDs/timestamps. `ready` is integer 0/1.
For every segment, positions 0 through 299 have `ready=0` and empty feature
fields; position 300 is the first ready row. Compare model outputs only for row
IDs present in `golden_predictions.csv`. Equal timestamps are permitted, while
decreasing timestamps are invalid. Raw input validation requires finite values,
positive prices/quantities, ordered ladders and strictly increasing update IDs.

Inference does **not** discard the final 50 snapshots: that training-only trim
exists because future target labels are unavailable there. A ready input needs
only past/current snapshots to produce features and predictions.

The decision policy uses the validation-selected `conviction_tau` from metadata:
Up when `p_up > tau` and `p_up > p_down`; Down when `p_down > tau` and
`p_down > p_up`; Flat otherwise, including an exact directional probability tie.
Raw scores are before softmax. Export verifies both raw scores and probabilities
after loading the saved model again; it also checks predictions on CSV-read
features so serialization cannot silently change tree branches.

The default real-data fixture is the first 5,000 rows of the longest contiguous
segment, with ties resolved by the earliest segment. It begins at a real reset
boundary and includes all 300 warm-up inputs. Synthetic unit tests additionally
cover reset gaps and readiness boundaries. A bundle is a numerical reference,
not evidence that a C++ implementation has passed parity yet.

## 6. C++ Streaming Feature API

`FeatureEngine` in `lob/feature_engine.hpp` provides:

```cpp
bool update(const BookSnapshot& snapshot, FeatureVector& output);
void reset() noexcept;
```

The caller supplies finite, valid book snapshots in timestamp order. An update
consumes one snapshot, maintaining OFI, mid-price history and volatility in fixed
storage. It returns `false` for the first 300 snapshots of each segment and
leaves `output` untouched. Snapshot 301 returns `true` and writes all 11 features.
A gap strictly greater than 500ms clears history **before** processing the new
snapshot, which becomes the first warm-up snapshot of a new segment.

Equal timestamps still count as distinct updates. Negative or decreasing
timestamps throw `std::invalid_argument` without changing state or output.
Explicit `reset()` clears history and permits a new stream to start at any
nonnegative timestamp. Readiness is capped after warm-up so its counter cannot
overflow on a long-running stream. Valid updates perform no heap allocation;
the 51-mid-price buffer includes the current value and the value 50 updates ago.

The C++ golden test streams the complete input fixture through this API, checks
every readiness flag, and compares all ready features with the Python reference
using the strict `< 1e-6` absolute tolerance. It does not yet test C++ model
probabilities; model loading and evaluation are separate milestones.
