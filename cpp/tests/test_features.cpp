#include <gtest/gtest.h>
#include "lob/features.hpp"

// bids: 100, 99, ... 91   asks: 101, 102, ... 110
// bid sizes: 3, 1, 1, then 2 x7     ask sizes: 1, 2, 2, then 1 x7
static lob::BookSnapshot make_book() {
  lob::BookSnapshot s;
  for (std::size_t i = 0; i < lob::kLevels; ++i) {
    s.bid_px[i] = 100.0 - static_cast<double>(i);
    s.ask_px[i] = 101.0 + static_cast<double>(i);
    s.bid_sz[i] = (i == 0) ? 3.0 : (i <= 2 ? 1.0 : 2.0);
    s.ask_sz[i] = (i == 0) ? 1.0 : (i <= 2 ? 2.0 : 1.0);
  }
  return s;
}

TEST(StatelessFeatures, HandComputedBook) {
  lob::FeatureVector f{};
  lob::compute_stateless(make_book(), f);

  EXPECT_NEAR(f[lob::kSpread], 1.0, 1e-9);
  EXPECT_NEAR(f[lob::kMicroMinusMid], 0.25, 1e-6);   // (100*1+101*3)/4 - 100.5
  EXPECT_NEAR(f[lob::kImbL1], 0.5, 1e-6);            // (3-1)/4
  EXPECT_NEAR(f[lob::kImbL2_3], -1.0 / 3.0, 1e-6);   // (2-4)/6
  EXPECT_NEAR(f[lob::kImbL4_10], 1.0 / 3.0, 1e-6);   // (14-7)/21
  EXPECT_NEAR(f[lob::kSpanAsym], 0.0, 1e-6);         // spans 9 and 9
}

TEST(StatelessFeatures, EmptyBookDoesNotDivideByZero) {
  lob::BookSnapshot s;   // all zeros
  lob::FeatureVector f{};
  lob::compute_stateless(s, f);
  for (std::size_t i = 0; i <= lob::kSpanAsym; ++i) EXPECT_EQ(f[i], 0.0);
}
