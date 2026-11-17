#pragma once
#include <vector>
#include "lob/types.hpp"

namespace lob {

// Nonnegative children name split nodes; negative children encode leaf -child-1.
struct SplitNode {
  std::size_t feature;
  double threshold;
  int left;
  int right;
};

class LightGBMModel;

class Tree {
 public:
  // Validates ownership, reachability, indices and finite numerical parameters.
  Tree(std::vector<SplitNode> splits, std::vector<double> leaves);
  double predict(const FeatureVector& features) const;
  std::size_t num_leaves() const noexcept { return leaves_.size(); }

 private:
  friend class LightGBMModel;
  double evaluate(const FeatureVector& features) const noexcept;
  std::vector<SplitNode> splits_;
  std::vector<double> leaves_;
};

}  // namespace lob
