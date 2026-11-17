#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include "lob/tree.hpp"

TEST(Tree, AllLeavesAndExactThresholds) {
  const lob::Tree tree({{lob::kImbL1, .30, -1, 1}, {lob::kRet50, -.50, -2, -3}},
                       {-.20, .10, .70});
  lob::FeatureVector x{};
  x[lob::kImbL1] = .30;  // Equality goes left.
  EXPECT_DOUBLE_EQ(tree.predict(x), -.20);
  x[lob::kImbL1] = std::nextafter(.30, 1.0);
  x[lob::kRet50] = -.50;
  EXPECT_DOUBLE_EQ(tree.predict(x), .10);
  x[lob::kRet50] = std::nextafter(-.50, 0.0);
  EXPECT_DOUBLE_EQ(tree.predict(x), .70);
}

TEST(Tree, ConstantTreeAndNonfiniteInput) {
  const lob::Tree tree({}, {2.5});
  lob::FeatureVector x{};
  EXPECT_DOUBLE_EQ(tree.predict(x), 2.5);
  x[0] = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(tree.predict(x), std::invalid_argument);
  x[0] = std::numeric_limits<double>::infinity();
  EXPECT_THROW(tree.predict(x), std::invalid_argument);
}

TEST(Tree, RejectsInvalidShapesNumbersAndIndices) {
  EXPECT_THROW((lob::Tree({}, {})), std::invalid_argument);
  EXPECT_THROW((lob::Tree({}, {1, 2})), std::invalid_argument);
  EXPECT_THROW((lob::Tree({}, {std::numeric_limits<double>::infinity()})), std::invalid_argument);
  EXPECT_THROW((lob::Tree({{11, 0, -1, -2}}, {1, 2})), std::invalid_argument);
  EXPECT_THROW((lob::Tree({{0, std::numeric_limits<double>::quiet_NaN(), -1, -2}}, {1, 2})), std::invalid_argument);
  EXPECT_THROW((lob::Tree({{0, 0, -1, -3}}, {1, 2})), std::invalid_argument);
  EXPECT_THROW((lob::Tree({{0, 0, -1, std::numeric_limits<int>::min()}}, {1, 2})), std::invalid_argument);
  EXPECT_THROW((lob::Tree({{0, 0, -1, 1}}, {1, 2})), std::invalid_argument);
}

TEST(Tree, RejectsCyclesSharedAndUnreachableNodes) {
  EXPECT_THROW((lob::Tree({{0, 0, 0, -1}}, {1, 2})), std::invalid_argument);
  EXPECT_THROW((lob::Tree({{0, 0, -1, -1}}, {1, 2})), std::invalid_argument);
  EXPECT_THROW((lob::Tree({{0, 0, 1, 1}, {0, 0, -1, -2}}, {1, 2, 3})), std::invalid_argument);
  EXPECT_THROW((lob::Tree({{0, 0, -1, -2}, {0, 0, 1, -3}}, {1, 2, 3})), std::invalid_argument);
}

TEST(Tree, DeepTreeUsesIterativeValidationAndTraversal) {
  constexpr int depth = 10000;
  std::vector<lob::SplitNode> nodes;
  std::vector<double> leaves(static_cast<std::size_t>(depth) + 1, -1);
  for (int i = 0; i < depth; ++i) {
    nodes.push_back({0, 0, -i - 1, i + 1 == depth ? -depth - 1 : i + 1});
  }
  leaves.back() = 42;
  const lob::Tree tree(std::move(nodes), std::move(leaves));
  lob::FeatureVector x{};
  x[0] = 1;
  EXPECT_DOUBLE_EQ(tree.predict(x), 42);
}
