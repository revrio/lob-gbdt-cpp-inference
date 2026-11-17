#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include "lob/model.hpp"

namespace {
std::string fixture(double down = -2, double flat = 0, double up = 2) {
  std::ostringstream out;
  out << "tree\nversion=v4\nnum_class=3\nnum_tree_per_iteration=3\nmax_feature_idx=10\n"
      << "objective=multiclass num_class:3\n"
      << "feature_names=spread micro_minus_mid imb_l1 imb_l2_3 imb_l4_10 span_asym ofi_10 ofi_50 ofi_300 ret_50 vol_50\n"
      << "tree_sizes=1 1 1\n";
  int id = 0;
  for (double value : {down, flat, up}) {
    out << "Tree=" << id++ << "\nnum_leaves=2\nnum_cat=0\nis_linear=0\n"
        << "split_feature=2\nthreshold=0.3\ndecision_type=2\n"
        << "left_child=-1\nright_child=-2\nleaf_value=" << value << ' ' << value + .5
        << "\nshrinkage=0.05\n";
  }
  return out.str() + "end of trees\n\nparameters:\nignored training metadata\n";
}

lob::LightGBMModel parse(const std::string& text) {
  std::istringstream stream(text);
  return lob::LightGBMModel::parse(stream);
}

std::string replace(std::string text, const std::string& before, const std::string& after) {
  const auto at = text.find(before);
  if (at == std::string::npos) throw std::logic_error("Test mutation did not match");
  text.replace(at, before.size(), after);
  return text;
}
}  // namespace

TEST(Model, ParsesNumericalTreesAndDoesNotApplyShrinkageTwice) {
  const auto model = parse(fixture());
  EXPECT_EQ(model.num_trees(), 3U);
  lob::FeatureVector x{};
  auto p = model.predict(x, .31);
  EXPECT_EQ(p.raw_scores, (std::array<double, 3>{-2, 0, 2}));
  EXPECT_EQ(p.class_id, 2);
  EXPECT_NEAR(p.probabilities[0] + p.probabilities[1] + p.probabilities[2], 1, 1e-15);
  x[lob::kImbL1] = .31;
  EXPECT_EQ(model.predict(x, .31).raw_scores, (std::array<double, 3>{-1.5, .5, 2.5}));
}

TEST(Model, ClassGroupingAcrossIterationsAndStableSoftmax) {
  auto text = fixture(1000, 0, -1000);
  const auto at = text.find("end of trees");
  auto more = text.substr(text.find("Tree=0"), at - text.find("Tree=0"));
  for (int i = 2; i >= 0; --i) more = replace(more, "Tree=" + std::to_string(i), "Tree=" + std::to_string(i + 3));
  text.insert(at, more);
  text = replace(text, "tree_sizes=1 1 1", "tree_sizes=1 1 1 1 1 1");
  const auto p = parse(text).predict({}, .31);
  EXPECT_EQ(p.raw_scores, (std::array<double, 3>{2000, 0, -2000}));
  EXPECT_EQ(p.probabilities, (std::array<double, 3>{1, 0, 0}));
  EXPECT_EQ(p.class_id, 0);
}

TEST(Model, DirectionalTiesAndStrictThresholdPolicy) {
  const auto model = parse(fixture(0, 0, 0));
  EXPECT_EQ(model.predict({}, 0).class_id, 1);  // Directional tie stays Flat.
  const auto up = parse(fixture());
  const auto p = up.predict({}, 0);
  EXPECT_EQ(up.predict({}, p.probabilities[2]).class_id, 1);  // Equality fails strict >.
  EXPECT_EQ(up.predict({}, std::nextafter(p.probabilities[2], 0.0)).class_id, 2);
  EXPECT_EQ(up.predict({}, 1).class_id, 1);
}

TEST(Model, ConstantTreesWhitespaceAndCRLF) {
  auto text = fixture();
  std::size_t pos = 0;
  while ((pos = text.find("num_leaves=2", pos)) != std::string::npos) {
    const auto end = text.find("shrinkage=0.05", pos);
    text.replace(pos, end - pos, "num_leaves=1\nnum_cat=0\nis_linear=0\nsplit_feature=\nthreshold=\ndecision_type=\nleft_child=\nright_child=\nleaf_value=2\n");
    ++pos;
  }
  text = replace(text, "decision_type=", "decision_type= ");
  std::string crlf;
  for (char c : text) { if (c == '\n') crlf += '\r'; crlf += c; }
  EXPECT_EQ(parse(crlf).predict({}, .31).raw_scores, (std::array<double, 3>{2, 2, 2}));
}

TEST(Model, RejectsUnsupportedModelSemantics) {
  for (const auto& [before, after] : std::initializer_list<std::pair<std::string, std::string>>{
      {"version=v4", "version=v3"}, {"num_class=3", "num_class=2"},
      {"num_tree_per_iteration=3", "num_tree_per_iteration=1"},
      {"max_feature_idx=10", "max_feature_idx=11"},
      {"objective=multiclass num_class:3", "objective=multiclassova num_class:3"},
      {"feature_names=spread micro_minus_mid", "feature_names=micro_minus_mid spread"},
      {"num_cat=0", "num_cat=1"}, {"is_linear=0", "is_linear=1"},
      {"decision_type=2", "decision_type=1"}, {"decision_type=2", "decision_type=6"},
      {"version=v4", "version=v4\naverage_output"}}) {
    SCOPED_TRACE(after);
    EXPECT_THROW(parse(replace(fixture(), before, after)), std::invalid_argument);
  }
  EXPECT_NO_THROW(parse(replace(fixture(), "decision_type=2", "decision_type=0")));
}

TEST(Model, RejectsMalformedAndTruncatedModels) {
  for (const auto& [before, after] : std::initializer_list<std::pair<std::string, std::string>>{
      {"tree\n", "garbage\n"}, {"threshold=0.3", "threshold=nan"},
      {"threshold=0.3", "threshold=1e999"}, {"threshold=0.3", "threshold=0.3junk"},
      {"threshold=0.3", "threshold=0.3 0.4"}, {"threshold=0.3", "threshold="},
      {"num_leaves=2", "num_leaves=99999999999999999"},
      {"num_leaves=2", "num_leaves=-1"}, {"num_leaves=2", "num_leaves=65537"},
      {"split_feature=2", "split_feature=11"}, {"left_child=-1", "left_child=0"},
      {"right_child=-2", "right_child=-2147483648"},
      {"Tree=1", "Tree=8"}, {"tree_sizes=1 1 1", "tree_sizes=1 1"},
      {"tree_sizes=1 1 1", "tree_sizes=1 0 1"},
      {"version=v4", "version=v4\nversion=v4"},
      {"threshold=0.3", "threshold=0.3\nthreshold=0.4"},
      {"num_cat=0\n", ""}, {"end of trees", ""}}) {
    SCOPED_TRACE(after);
    EXPECT_THROW(parse(replace(fixture(), before, after)), std::invalid_argument);
  }
  auto text = fixture(); text.resize(text.find("Tree=2"));
  EXPECT_THROW(parse(text), std::invalid_argument);
  EXPECT_THROW(parse(""), std::invalid_argument);
}

TEST(Model, RejectsIncompleteIterationAndInvalidPredictionInputs) {
  auto text = fixture();
  text.erase(text.find("Tree=2"), text.find("end of trees") - text.find("Tree=2"));
  EXPECT_THROW(parse(text), std::invalid_argument);
  const auto model = parse(fixture());
  for (double tau : {-1.0, 1.1, std::numeric_limits<double>::quiet_NaN()}) {
    EXPECT_THROW(model.predict({}, tau), std::invalid_argument);
  }
  lob::FeatureVector x{};
  x[10] = std::numeric_limits<double>::infinity();
  EXPECT_THROW(model.predict(x, .31), std::invalid_argument);
  EXPECT_THROW(lob::LightGBMModel::load("/nonexistent/lob-model.txt"), std::runtime_error);
}
