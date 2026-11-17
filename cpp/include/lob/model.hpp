#pragma once
#include <array>
#include <filesystem>
#include <istream>
#include <vector>
#include "lob/tree.hpp"

namespace lob {

struct Prediction {
  std::array<double, 3> raw_scores{};
  std::array<double, 3> probabilities{};
  int class_id = 1;  // Down=0, Flat=1, Up=2.
};

// Restricted loader for our v4, three-class, numerical GBDT contract.
// Load once; subsequent predictions read immutable trees without allocating.
class LightGBMModel {
 public:
  static LightGBMModel load(const std::filesystem::path& path);
  static LightGBMModel parse(std::istream& input);
  Prediction predict(const FeatureVector& features, double conviction_tau) const;
  std::size_t num_trees() const noexcept { return trees_.size(); }

 private:
  explicit LightGBMModel(std::vector<Tree> trees);
  std::vector<Tree> trees_;
};

}  // namespace lob
