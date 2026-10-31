#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "lob/feature_engine.hpp"

namespace {

constexpr std::array<const char*, lob::kNumFeatures> feature_names{
    "spread", "micro_minus_mid", "imb_l1", "imb_l2_3", "imb_l4_10", "span_asym",
    "ofi_10", "ofi_50", "ofi_300", "ret_50", "vol_50"};

// Test-only reader for the canonical, unquoted numeric CSV fixture. A public
// replay CLI and its general input/error handling are a later milestone.
std::vector<std::string> cells(std::string line) {
  if (!line.empty() && line.back() == '\r') line.pop_back();
  std::vector<std::string> result;
  std::size_t start = 0;
  for (;;) {
    const auto end = line.find(',', start);
    if (end == std::string::npos) {
      result.push_back(line.substr(start));
      return result;
    }
    result.push_back(line.substr(start, end - start));
    start = end + 1;  // Preserve empty trailing warm-up feature fields.
  }
}

std::string input_header() {
  std::ostringstream result;
  result << "row_id,timestamp_ms,last_update_id";
  for (std::size_t i = 1; i <= lob::kLevels; ++i) {
    result << ",bid_px_" << i << ",bid_sz_" << i << ",ask_px_" << i << ",ask_sz_" << i;
  }
  return result.str();
}

std::string feature_header() {
  std::ostringstream result;
  result << "row_id,ready";
  for (const auto name : feature_names) result << ',' << name;
  return result.str();
}

lob::BookSnapshot snapshot(const std::vector<std::string>& fields) {
  lob::BookSnapshot s;
  s.timestamp_ms = std::stoll(fields[1]);
  for (std::size_t i = 0; i < lob::kLevels; ++i) {
    const auto first = 3 + 4 * i;
    s.bid_px[i] = std::stod(fields[first]);
    s.bid_sz[i] = std::stod(fields[first + 1]);
    s.ask_px[i] = std::stod(fields[first + 2]);
    s.ask_sz[i] = std::stod(fields[first + 3]);
  }
  return s;
}

}  // namespace

TEST(FeatureParity, CompletePythonGoldenStream) {
  const std::filesystem::path directory{LOB_CONTRACT_DIR};
  std::ifstream inputs(directory / "golden_input_lob.csv");
  std::ifstream expected(directory / "golden_features.csv");
  ASSERT_TRUE(inputs.is_open()) << "Missing golden_input_lob.csv in " << directory;
  ASSERT_TRUE(expected.is_open()) << "Missing golden_features.csv in " << directory;
  std::string input_line, feature_line;
  ASSERT_TRUE(static_cast<bool>(std::getline(inputs, input_line)));
  ASSERT_TRUE(static_cast<bool>(std::getline(expected, feature_line)));
  ASSERT_EQ(input_line, input_header());
  ASSERT_EQ(feature_line, feature_header());

  lob::FeatureEngine engine;
  std::array<double, lob::kNumFeatures> max_error{};
  std::array<std::size_t, lob::kNumFeatures> worst_row{};
  std::size_t rows = 0, ready_rows = 0;
  while (std::getline(inputs, input_line)) {
    SCOPED_TRACE(rows);
    ASSERT_TRUE(static_cast<bool>(std::getline(expected, feature_line))) << "Missing feature row";
    const auto input = cells(input_line);
    const auto reference = cells(feature_line);
    ASSERT_EQ(input.size(), 3 + 4 * lob::kLevels);
    ASSERT_EQ(reference.size(), 2 + lob::kNumFeatures);
    ASSERT_EQ(std::stoull(input[0]), rows);
    ASSERT_EQ(std::stoull(reference[0]), rows);
    ASSERT_TRUE(reference[1] == "0" || reference[1] == "1");

    lob::FeatureVector features;
    features.fill(-12345.0);
    const auto before = features;
    const bool ready = engine.update(snapshot(input), features);
    ASSERT_EQ(ready, reference[1] == "1");
    if (!ready) {
      EXPECT_EQ(features, before);
      for (std::size_t j = 0; j < lob::kNumFeatures; ++j) EXPECT_TRUE(reference[j + 2].empty());
    } else {
      ++ready_rows;
      for (std::size_t j = 0; j < lob::kNumFeatures; ++j) {
        const double value = std::stod(reference[j + 2]);
        ASSERT_TRUE(std::isfinite(value));
        ASSERT_TRUE(std::isfinite(features[j]));
        const double error = std::abs(value - features[j]);
        if (error > max_error[j]) {
          max_error[j] = error;
          worst_row[j] = rows;
        }
      }
    }
    ++rows;
  }
  EXPECT_FALSE(static_cast<bool>(std::getline(expected, feature_line))) << "Extra feature rows";
  ASSERT_GT(ready_rows, 0U);
  for (std::size_t j = 0; j < lob::kNumFeatures; ++j) {
    EXPECT_LT(max_error[j], 1e-6) << feature_names[j] << " at row " << worst_row[j];
  }
  std::cout << "Compared " << rows << " snapshots, " << ready_rows << " ready rows, "
            << ready_rows * lob::kNumFeatures << " feature values; max absolute error = "
            << std::setprecision(17) << *std::max_element(max_error.begin(), max_error.end()) << '\n';
}
