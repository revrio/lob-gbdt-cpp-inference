#include "lob/model.hpp"
#include <algorithm>
#include <charconv>
#include <cmath>
#include <fstream>
#include <locale>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace lob {
namespace {
using Fields = std::map<std::string, std::string>;
constexpr std::size_t kMaxModelBytes = 64 * 1024 * 1024;
constexpr std::size_t kMaxTrees = 10000;
constexpr std::size_t kMaxLeaves = 65536;
constexpr std::string_view kFeatureNames =
    "spread micro_minus_mid imb_l1 imb_l2_3 imb_l4_10 span_asym "
    "ofi_10 ofi_50 ofi_300 ret_50 vol_50";

std::string trim(const std::string& text) {
  const auto first = text.find_first_not_of(" \t\r");
  if (first == std::string::npos) return {};
  return text.substr(first, text.find_last_not_of(" \t\r") - first + 1);
}

const std::string& require(const Fields& fields, const std::string& key) {
  const auto item = fields.find(key);
  if (item == fields.end()) throw std::invalid_argument("Missing field: " + key);
  return item->second;
}

int integer(const std::string& text) {
  int value = 0;
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
  if (error != std::errc{} || end != text.data() + text.size()) {
    throw std::invalid_argument("Invalid model integer: " + text);
  }
  return value;
}

double number(const std::string& text) {
  double value = 0;
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
  if (error != std::errc{} || end != text.data() + text.size() || !std::isfinite(value)) {
    throw std::invalid_argument("Invalid or nonfinite model number: " + text);
  }
  return value;
}

template <typename T, typename Parse>
std::vector<T> array(const Fields& fields, const std::string& key,
                     std::size_t count, Parse parse) {
  std::istringstream stream(require(fields, key));
  stream.imbue(std::locale::classic());
  std::vector<T> values;
  std::string token;
  while (stream >> token) {
    if (values.size() >= count) throw std::invalid_argument("Too many values: " + key);
    values.push_back(parse(token));
  }
  if (values.size() != count) throw std::invalid_argument("Wrong array length: " + key);
  return values;
}

Tree parse_tree(const Fields& fields) {
  const int leaves = integer(require(fields, "num_leaves"));
  if (leaves <= 0 || static_cast<std::size_t>(leaves) > kMaxLeaves) {
    throw std::invalid_argument("Invalid or excessive leaf count");
  }
  if (integer(require(fields, "num_cat")) != 0 ||
      integer(require(fields, "is_linear")) != 0) {
    throw std::invalid_argument("Categorical and linear trees are unsupported");
  }
  // LightGBM leaf_value already includes shrinkage and the initial class bias.
  // Never multiply these scores by learning_rate or shrinkage a second time.
  if (number(require(fields, "shrinkage")) < 0) {
    throw std::invalid_argument("Negative shrinkage");
  }
  const auto n = static_cast<std::size_t>(leaves - 1);
  const auto features = array<int>(fields, "split_feature", n, integer);
  const auto thresholds = array<double>(fields, "threshold", n, number);
  const auto decisions = array<int>(fields, "decision_type", n, integer);
  const auto left = array<int>(fields, "left_child", n, integer);
  const auto right = array<int>(fields, "right_child", n, integer);
  auto scores = array<double>(fields, "leaf_value", static_cast<std::size_t>(leaves), number);
  std::vector<SplitNode> splits;
  splits.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    // Support numerical <= splits with MissingType=None, either default-left
    // bit. Inputs are finite. Other missing/categorical semantics fail closed.
    if (decisions[i] != 0 && decisions[i] != 2) {
      throw std::invalid_argument("Unsupported decision_type (categorical/missing split)");
    }
    if (features[i] < 0 || features[i] >= static_cast<int>(kNumFeatures)) {
      throw std::invalid_argument("Split feature outside contract");
    }
    splits.push_back({static_cast<std::size_t>(features[i]), thresholds[i], left[i], right[i]});
  }
  return Tree(std::move(splits), std::move(scores));
}

void validate_header(const Fields& header) {
  if (require(header, "version") != "v4" ||
      integer(require(header, "num_class")) != 3 ||
      integer(require(header, "num_tree_per_iteration")) != 3 ||
      integer(require(header, "max_feature_idx")) != static_cast<int>(kNumFeatures - 1) ||
      require(header, "objective") != "multiclass num_class:3" ||
      require(header, "feature_names") != kFeatureNames) {
    throw std::invalid_argument("Model header does not match the feature/class contract");
  }
  if (header.contains("average_output")) {
    throw std::invalid_argument("Averaged ensembles are unsupported");
  }
}
}  // namespace

LightGBMModel::LightGBMModel(std::vector<Tree> trees) : trees_(std::move(trees)) {}

LightGBMModel LightGBMModel::load(const std::filesystem::path& path) {
  std::ifstream stream(path);
  if (!stream) throw std::runtime_error("Cannot open model: " + path.string());
  return parse(stream);
}

LightGBMModel LightGBMModel::parse(std::istream& input) {
  Fields header, fields;
  std::vector<Tree> trees;
  std::string line;
  std::size_t bytes = 0;
  bool magic = false, in_tree = false, ended = false;
  while (std::getline(input, line)) {
    bytes += line.size() + 1;
    if (bytes > kMaxModelBytes) throw std::invalid_argument("Model exceeds 64 MiB limit");
    line = trim(line);
    if (line.empty()) continue;
    if (!magic) {
      if (line != "tree") throw std::invalid_argument("Expected LightGBM tree file");
      magic = true;
      continue;
    }
    if (line == "end of trees") {
      if (!in_tree) throw std::invalid_argument("Model contains no trees");
      trees.push_back(parse_tree(fields));
      ended = true;
      break;  // Feature importance and training parameters are not inference data.
    }
    if (line == "average_output") throw std::invalid_argument("Averaged ensembles are unsupported");
    const auto separator = line.find('=');
    if (separator == std::string::npos) throw std::invalid_argument("Malformed model line");
    const std::string key = trim(line.substr(0, separator));
    const std::string value = trim(line.substr(separator + 1));
    if (key == "Tree") {
      if (in_tree) trees.push_back(parse_tree(fields));
      else validate_header(header);
      if (trees.size() >= kMaxTrees || integer(value) != static_cast<int>(trees.size())) {
        throw std::invalid_argument("Nonsequential or excessive tree count");
      }
      fields.clear();
      in_tree = true;
    } else {
      auto& target = in_tree ? fields : header;
      if (key.empty() || !target.emplace(key, value).second) {
        throw std::invalid_argument("Empty or duplicate model field: " + key);
      }
    }
  }
  if (input.bad()) throw std::runtime_error("Model read failed");
  if (!ended || trees.empty() || trees.size() % 3 != 0) {
    throw std::invalid_argument("Truncated model or incomplete three-tree iteration");
  }
  // We parse sequentially; tree_sizes is checked for count/validity rather than
  // used for byte seeking (which would depend on newline encoding).
  const auto sizes = array<int>(header, "tree_sizes", trees.size(), integer);
  for (int size : sizes) {
    if (size <= 0) throw std::invalid_argument("Invalid tree size");
  }
  return LightGBMModel(std::move(trees));
}

Prediction LightGBMModel::predict(const FeatureVector& features, double conviction_tau) const {
  if (trees_.empty()) throw std::logic_error("Cannot predict with an empty/moved model");
  if (!std::isfinite(conviction_tau) || conviction_tau < 0 || conviction_tau > 1) {
    throw std::invalid_argument("Conviction threshold must be in [0,1]");
  }
  for (double value : features) {
    if (!std::isfinite(value)) throw std::invalid_argument("Features must be finite");
  }
  Prediction result;
  for (std::size_t i = 0; i < trees_.size(); ++i) {
    result.raw_scores[i % 3] += trees_[i].evaluate(features);
  }
  const double maximum = *std::max_element(result.raw_scores.begin(), result.raw_scores.end());
  double sum = 0;
  for (std::size_t i = 0; i < 3; ++i) {
    if (!std::isfinite(result.raw_scores[i])) throw std::overflow_error("Nonfinite ensemble score");
    result.probabilities[i] = std::exp(result.raw_scores[i] - maximum);
    sum += result.probabilities[i];
  }
  for (double& probability : result.probabilities) probability /= sum;
  if (result.probabilities[2] > conviction_tau && result.probabilities[2] > result.probabilities[0]) {
    result.class_id = 2;
  } else if (result.probabilities[0] > conviction_tau && result.probabilities[0] > result.probabilities[2]) {
    result.class_id = 0;
  }
  return result;
}

}  // namespace lob
