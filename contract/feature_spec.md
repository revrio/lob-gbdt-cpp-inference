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