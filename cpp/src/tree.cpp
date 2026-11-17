#include "lob/tree.hpp"
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

namespace lob {

Tree::Tree(std::vector<SplitNode> splits, std::vector<double> leaves)
    : splits_(std::move(splits)), leaves_(std::move(leaves)) {
  if (leaves_.empty() || splits_.size() != leaves_.size() - 1 ||
      leaves_.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument("Tree needs N leaves and N-1 split nodes");
  }
  for (double leaf : leaves_) {
    if (!std::isfinite(leaf)) throw std::invalid_argument("Nonfinite leaf score");
  }
  if (splits_.empty()) return;  // A constant tree has exactly one leaf.
  std::vector<bool> seen_splits(splits_.size()), seen_leaves(leaves_.size());
  std::vector<int> pending{0};
  while (!pending.empty()) {
    const int child = pending.back();
    pending.pop_back();
    if (child < 0) {
      // Widen before negation so even INT_MIN is safe to validate.
      const auto leaf = static_cast<std::size_t>(-static_cast<std::int64_t>(child) - 1);
      if (leaf >= leaves_.size() || seen_leaves[leaf]) {
        throw std::invalid_argument("Invalid or repeated leaf child");
      }
      seen_leaves[leaf] = true;
    } else {
      const auto index = static_cast<std::size_t>(child);
      if (index >= splits_.size() || seen_splits[index]) {
        throw std::invalid_argument("Invalid child, cycle or shared split node");
      }
      seen_splits[index] = true;
      const auto& node = splits_[index];
      if (node.feature >= kNumFeatures || !std::isfinite(node.threshold)) {
        throw std::invalid_argument("Invalid split feature or threshold");
      }
      pending.push_back(node.right);
      pending.push_back(node.left);
    }
  }
  for (bool seen : seen_splits) {
    if (!seen) throw std::invalid_argument("Unreachable split node");
  }
  for (bool seen : seen_leaves) {
    if (!seen) throw std::invalid_argument("Unreachable leaf");
  }
}

double Tree::evaluate(const FeatureVector& features) const noexcept {
  if (splits_.empty()) return leaves_[0];
  int child = 0;
  while (child >= 0) {
    const auto& node = splits_[static_cast<std::size_t>(child)];
    child = features[node.feature] <= node.threshold ? node.left : node.right;
  }
  return leaves_[static_cast<std::size_t>(-static_cast<std::int64_t>(child) - 1)];
}

double Tree::predict(const FeatureVector& features) const {
  if (leaves_.empty()) throw std::logic_error("Cannot predict with a moved tree");
  for (double feature : features) {
    if (!std::isfinite(feature)) throw std::invalid_argument("Features must be finite");
  }
  return evaluate(features);
}

}  // namespace lob
