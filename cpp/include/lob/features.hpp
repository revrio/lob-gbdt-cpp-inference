#pragma once
#include "lob/types.hpp"

namespace lob {

inline double mid_price(const BookSnapshot& s) {
  return (s.bid_px[0] + s.ask_px[0]) / 2.0;
}

// Fills indices 0..5 of f. Indices 6..10 (windowed) come later.
inline void compute_stateless(const BookSnapshot& s, FeatureVector& f) {
  const double mid = mid_price(s);
  const double b1 = s.bid_sz[0];
  const double a1 = s.ask_sz[0];

  f[kSpread] = s.ask_px[0] - s.bid_px[0];

  f[kMicroMinusMid] =
      (s.bid_px[0] * a1 + s.ask_px[0] * b1) / (b1 + a1 + kEps) - mid;

  f[kImbL1] = (b1 - a1) / (b1 + a1 + kEps);

  double b23 = 0.0, a23 = 0.0;
  for (std::size_t i = 1; i <= 2; ++i) { b23 += s.bid_sz[i]; a23 += s.ask_sz[i]; }
  f[kImbL2_3] = (b23 - a23) / (b23 + a23 + kEps);

  double b410 = 0.0, a410 = 0.0;
  for (std::size_t i = 3; i < kLevels; ++i) { b410 += s.bid_sz[i]; a410 += s.ask_sz[i]; }
  f[kImbL4_10] = (b410 - a410) / (b410 + a410 + kEps);

  const double ask_span = s.ask_px[kLevels - 1] - s.ask_px[0];
  const double bid_span = s.bid_px[0] - s.bid_px[kLevels - 1];
  f[kSpanAsym] = (ask_span - bid_span) / (ask_span + bid_span + kEps);
}

}  // namespace lob
