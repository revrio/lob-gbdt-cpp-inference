#include <gtest/gtest.h>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#include "lob/feature_engine.hpp"

namespace {

lob::BookSnapshot book(std::int64_t timestamp = 0, double shift = 0.0) {
  lob::BookSnapshot s;
  s.timestamp_ms = timestamp;
  for (std::size_t i = 0; i < lob::kLevels; ++i) {
    s.bid_px[i] = 100.0 - static_cast<double>(i) + shift;
    s.ask_px[i] = 101.0 + static_cast<double>(i) + shift;
    s.bid_sz[i] = i == 0 ? 3.0 : i <= 2 ? 1.0 : 2.0;
    s.ask_sz[i] = i == 0 ? 1.0 : i <= 2 ? 2.0 : 1.0;
  }
  return s;
}

void warm_up(lob::FeatureEngine& engine, lob::FeatureVector& f) {
  for (int i = 0; i < 300; ++i) ASSERT_FALSE(engine.update(book(i * 100), f));
}

}  // namespace

TEST(FeatureEngine, First300UpdatesLeaveOutputUntouched) {
  lob::FeatureEngine engine;
  lob::FeatureVector f;
  f.fill(42.0);
  const auto sentinel = f;
  warm_up(engine, f);
  EXPECT_EQ(f, sentinel);
  ASSERT_TRUE(engine.update(book(30'000), f));
  EXPECT_NEAR(f[lob::kSpread], 1.0, 1e-9);
  EXPECT_NEAR(f[lob::kMicroMinusMid], .25, 1e-6);
  EXPECT_NEAR(f[lob::kImbL1], .5, 1e-6);
  EXPECT_NEAR(f[lob::kImbL2_3], -1.0 / 3.0, 1e-6);
  EXPECT_NEAR(f[lob::kImbL4_10], 1.0 / 3.0, 1e-6);
  EXPECT_DOUBLE_EQ(f[lob::kSpanAsym], 0.0);
  for (std::size_t i = lob::kOfi10; i < lob::kNumFeatures; ++i) EXPECT_DOUBLE_EQ(f[i], 0.0);
}

TEST(FeatureEngine, PriceJumpHasHandCalculatedStatefulFeatures) {
  lob::FeatureEngine engine;
  lob::FeatureVector f{};
  warm_up(engine, f);
  ASSERT_TRUE(engine.update(book(30'000, 1.0), f));
  // Both prices rise: bid contributes +3, ask contributes -1, so OFI = 4.
  EXPECT_DOUBLE_EQ(f[lob::kOfi10], 4.0);
  EXPECT_DOUBLE_EQ(f[lob::kOfi50], 4.0);
  EXPECT_DOUBLE_EQ(f[lob::kOfi300], 4.0);
  EXPECT_DOUBLE_EQ(f[lob::kRet50], 1.0);
  // Fifty price changes contain one 1 and forty-nine zeros: std = 0.14.
  EXPECT_NEAR(f[lob::kVol50], .14, 1e-12);

  for (int i = 301; i <= 600; ++i) {
    ASSERT_TRUE(engine.update(book(i * 100, 1.0), f));
    if (i == 309) { EXPECT_DOUBLE_EQ(f[lob::kOfi10], 4.0); }
    if (i == 310) { EXPECT_DOUBLE_EQ(f[lob::kOfi10], 0.0); }
    if (i == 349) {
      EXPECT_DOUBLE_EQ(f[lob::kOfi50], 4.0);
      EXPECT_DOUBLE_EQ(f[lob::kRet50], 1.0);
    }
    if (i == 350) {
      EXPECT_DOUBLE_EQ(f[lob::kOfi50], 0.0);
      EXPECT_DOUBLE_EQ(f[lob::kRet50], 0.0);
      EXPECT_NEAR(f[lob::kVol50], 0.0, 1e-12);
    }
    if (i == 599) { EXPECT_DOUBLE_EQ(f[lob::kOfi300], 4.0); }
    if (i == 600) { EXPECT_DOUBLE_EQ(f[lob::kOfi300], 0.0); }
  }
}

TEST(FeatureEngine, AllBidAndAskOFIBranches) {
  struct Case { double bid_px, bid_sz, ask_px, ask_sz, expected; };
  const std::array<Case, 7> cases{{
      {100.25, 7, 101, 1, 7}, {100, 5, 101, 1, 2}, {99.75, 7, 101, 1, -3},
      {100, 3, 100.75, 4, -4}, {100, 3, 101, 4, -3}, {100, 3, 101.25, 4, 1},
      {100, 3, 101, 1, 0},
  }};
  for (const auto& c : cases) {
    SCOPED_TRACE(c.expected);
    lob::FeatureEngine engine;
    lob::FeatureVector f{};
    warm_up(engine, f);
    auto s = book(30'000);
    s.bid_px[0] = c.bid_px; s.bid_sz[0] = c.bid_sz;
    s.ask_px[0] = c.ask_px; s.ask_sz[0] = c.ask_sz;
    ASSERT_TRUE(engine.update(s, f));
    EXPECT_DOUBLE_EQ(f[lob::kOfi10], c.expected);
    EXPECT_DOUBLE_EQ(f[lob::kOfi50], c.expected);
    EXPECT_DOUBLE_EQ(f[lob::kOfi300], c.expected);
  }
}

TEST(FeatureEngine, Exactly500msDoesNotResetBut501msDoes) {
  lob::FeatureEngine engine;
  lob::FeatureVector f{};
  warm_up(engine, f);  // Last timestamp = 29,900.
  ASSERT_TRUE(engine.update(book(30'400), f));
  f.fill(-7.0);
  const auto before = f;
  ASSERT_FALSE(engine.update(book(30'901, 25.0), f));
  EXPECT_EQ(f, before);
  for (int i = 1; i < 300; ++i) ASSERT_FALSE(engine.update(book(30'901 + i * 100, 25.0), f));
  ASSERT_TRUE(engine.update(book(60'901, 25.0), f));
  // The price jump across the gap must not leak into OFI, returns or volatility.
  for (std::size_t i = lob::kOfi10; i < lob::kNumFeatures; ++i) EXPECT_DOUBLE_EQ(f[i], 0.0);
}

TEST(FeatureEngine, ExplicitResetClearsHistoryAndTimestamp) {
  lob::FeatureEngine engine;
  lob::FeatureVector f{};
  warm_up(engine, f);
  ASSERT_TRUE(engine.update(book(30'000, 1.0), f));
  engine.reset();
  engine.reset();  // Idempotent, including before the first update.
  for (int i = 0; i < 300; ++i) ASSERT_FALSE(engine.update(book(i * 100, 50.0), f));
  ASSERT_TRUE(engine.update(book(30'000, 50.0), f));
  for (std::size_t i = lob::kOfi10; i < lob::kNumFeatures; ++i) EXPECT_DOUBLE_EQ(f[i], 0.0);
}

TEST(FeatureEngine, EqualTimestampsStillCountAsDistinctUpdates) {
  lob::FeatureEngine engine;
  lob::FeatureVector f{};
  for (int i = 0; i < 300; ++i) ASSERT_FALSE(engine.update(book(1000), f));
  auto s = book(1000);
  s.bid_sz[0] = 5.0;
  ASSERT_TRUE(engine.update(s, f));
  EXPECT_DOUBLE_EQ(f[lob::kOfi10], 2.0);
  EXPECT_DOUBLE_EQ(f[lob::kRet50], 0.0);
}

TEST(FeatureEngine, InvalidTimestampLeavesStateAndOutputUntouched) {
  lob::FeatureEngine engine;
  lob::FeatureVector f;
  f.fill(42.0);
  EXPECT_THROW(engine.update(book(-1), f), std::invalid_argument);
  warm_up(engine, f);
  ASSERT_TRUE(engine.update(book(30'000), f));
  const auto before = f;
  EXPECT_THROW(engine.update(book(29'999, 10.0), f), std::invalid_argument);
  EXPECT_EQ(f, before);
  ASSERT_TRUE(engine.update(book(30'100), f));
  EXPECT_EQ(f, before);
}

TEST(FeatureEngine, ExtremeNonnegativeTimestampsDoNotOverflow) {
  lob::FeatureEngine engine;
  lob::FeatureVector f{};
  ASSERT_FALSE(engine.update(book(0), f));
  const auto maximum = std::numeric_limits<std::int64_t>::max();
  ASSERT_FALSE(engine.update(book(maximum), f));  // Huge gap starts fresh warm-up.
  for (int i = 1; i < 300; ++i) ASSERT_FALSE(engine.update(book(maximum), f));
  ASSERT_TRUE(engine.update(book(maximum), f));
  EXPECT_DOUBLE_EQ(f[lob::kVol50], 0.0);
}

TEST(FeatureEngine, LongStreamMatchesNaiveHistoryCalculations) {
  lob::FeatureEngine engine;
  lob::FeatureVector f{};
  std::vector<double> ofi, mid, changes;
  lob::BookSnapshot previous;
  for (int i = 0; i < 20'000; ++i) {
    SCOPED_TRACE(i);
    auto s = book(i * 100);
    const double best = 1000.0 + (i % 211 - 105) * .01;
    for (std::size_t j = 0; j < lob::kLevels; ++j) {
      s.bid_px[j] = best - static_cast<double>(j) * .01;
      s.ask_px[j] = best + (i % 3 + 1) * .01 + static_cast<double>(j) * .01;
    }
    s.bid_sz[0] = 1.0 + (i % 17) * .001;
    s.ask_sz[0] = .5 + (i % 13) * .002;
    const double current_mid = (s.bid_px[0] + s.ask_px[0]) / 2.0;
    double flow = 0.0;
    if (i > 0) {
      // Independent indicator-form OFI reference, including equality on both terms.
      flow = (s.bid_px[0] >= previous.bid_px[0] ? s.bid_sz[0] : 0.0)
           - (s.bid_px[0] <= previous.bid_px[0] ? previous.bid_sz[0] : 0.0)
           - (s.ask_px[0] <= previous.ask_px[0] ? s.ask_sz[0] : 0.0)
           + (s.ask_px[0] >= previous.ask_px[0] ? previous.ask_sz[0] : 0.0);
    }
    changes.push_back(i == 0 ? 0.0 : current_mid - mid.back());
    mid.push_back(current_mid);
    ofi.push_back(flow);
    previous = s;
    const bool ready = engine.update(s, f);
    ASSERT_EQ(ready, i >= 300);
    if (!ready) continue;
    for (const auto [window, feature] : std::array<std::pair<int, lob::Feature>, 3>{{
        {10, lob::kOfi10}, {50, lob::kOfi50}, {300, lob::kOfi300}}}) {
      const double expected = std::accumulate(ofi.end() - window, ofi.end(), 0.0);
      ASSERT_NEAR(f[feature], expected, 1e-6);
    }
    ASSERT_NEAR(f[lob::kRet50], mid.back() - mid[mid.size() - 51], 1e-6);
    const double mean = std::accumulate(changes.end() - 50, changes.end(), 0.0) / 50;
    double variance = 0.0;
    for (auto it = changes.end() - 50; it != changes.end(); ++it) variance += (*it - mean) * (*it - mean);
    ASSERT_NEAR(f[lob::kVol50], std::sqrt(variance / 50), 1e-6);
  }
}
