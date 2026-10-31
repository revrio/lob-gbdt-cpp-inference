#pragma once
#include <array>
#include <cstddef>
#include <cstdint>

namespace lob {

constexpr std::size_t kLevels = 10;
constexpr std::size_t kNumFeatures = 11;
constexpr double kEps = 1e-9;
constexpr std::int64_t kMaxGapMs = 500;
constexpr std::size_t kWarmupTicks = 300;

// Index 0 = level 1 (best price), index 9 = level 10.
struct BookSnapshot {
  std::int64_t timestamp_ms = 0;
  std::array<double, kLevels> bid_px{};
  std::array<double, kLevels> bid_sz{};
  std::array<double, kLevels> ask_px{};
  std::array<double, kLevels> ask_sz{};
};

using FeatureVector = std::array<double, kNumFeatures>;

// Order must match contract/feature_spec.md section 4.
enum Feature : std::size_t {
  kSpread = 0,
  kMicroMinusMid,
  kImbL1,
  kImbL2_3,
  kImbL4_10,
  kSpanAsym,
  kOfi10,
  kOfi50,
  kOfi300,
  kRet50,
  kVol50
};

}  // namespace lob
