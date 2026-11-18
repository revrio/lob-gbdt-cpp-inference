#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <locale>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>

#include "lob/csv_reader.hpp"
#include "lob/feature_engine.hpp"
#include "lob/model.hpp"

namespace {
using Clock = std::chrono::steady_clock;
double ns(Clock::duration duration) { return std::chrono::duration<double, std::nano>(duration).count(); }

struct Options {
  std::filesystem::path model = "contract/model.txt", input, output, summary;
  double tau = -1;
  bool no_output = false, profile = false;
};

Options options(int argc, char** argv) {
  Options o;
  std::set<std::string> seen;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    if (!seen.insert(key).second) throw std::invalid_argument("Repeated option: " + key);
    if (key == "--no-output") o.no_output = true;
    else if (key == "--profile") o.profile = true;
    else {
      if (i + 1 == argc) throw std::invalid_argument("Missing value for " + key);
      const std::string value = argv[++i];
      if (key == "--model") o.model = value;
      else if (key == "--input") o.input = value;
      else if (key == "--output") o.output = value;
      else if (key == "--summary") o.summary = value;
      else if (key == "--tau") {
        const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), o.tau);
        if (error != std::errc{} || end != value.data() + value.size()) throw std::invalid_argument("Invalid --tau");
      } else throw std::invalid_argument("Unknown option: " + key);
    }
  }
  if (o.input.empty()) throw std::invalid_argument("--input is required");
  if (!std::isfinite(o.tau) || o.tau < 0 || o.tau > 1) throw std::invalid_argument("Supply --tau in [0,1] from metadata.json");
  if (o.no_output == !o.output.empty()) throw std::invalid_argument("Choose exactly one of --output PATH or --no-output");
  return o;
}

// Results are published only after successful EOF and flush. Existing output
// paths are never replaced. The temporary directory belongs to this invocation.
class AtomicOutput {
 public:
  explicit AtomicOutput(const std::filesystem::path& destination) : destination_(destination) {
    if (std::filesystem::exists(destination_) || std::filesystem::is_symlink(destination_)) {
      throw std::runtime_error("Output already exists: " + destination_.string());
    }
    const auto parent = destination_.has_parent_path() ? destination_.parent_path() : std::filesystem::path(".");
    const auto token = Clock::now().time_since_epoch().count();
    for (int attempt = 0; attempt < 100; ++attempt) {
      directory_ = parent / (".lob-replay-" + std::to_string(token) + "-" + std::to_string(attempt));
      if (std::filesystem::create_directory(directory_)) {
        temporary_ = directory_ / "output";
        stream_.open(temporary_, std::ios::binary);
        if (!stream_) {
          std::filesystem::remove(directory_);
          throw std::runtime_error("Cannot create temporary output for " + destination_.string());
        }
        stream_.exceptions(std::ios::badbit | std::ios::failbit);
        stream_.imbue(std::locale::classic());
        return;
      }
    }
    throw std::runtime_error("Cannot create temporary output directory");
  }
  ~AtomicOutput() {
    // Destructor cleanup must not throw during error propagation.
    stream_.exceptions(std::ios::goodbit);
    stream_.close();
    std::error_code error;
    std::filesystem::remove(temporary_, error);
    std::filesystem::remove(directory_, error);
  }
  AtomicOutput(const AtomicOutput&) = delete;
  AtomicOutput& operator=(const AtomicOutput&) = delete;
  std::ostream& stream() { return stream_; }
  void flush() { stream_.flush(); }
  void publish() {
    stream_.close();
    // Same-filesystem hard-link publication fails if the destination appeared
    // during replay. A completed CSV never replaces an existing file.
    std::filesystem::create_hard_link(temporary_, destination_);
  }
 private:
  std::filesystem::path destination_, directory_, temporary_;
  std::ofstream stream_;
};

struct Statistics {
  std::uint64_t rows = 0, warmup = 0, predictions = 0, resets = 0;
  std::array<std::uint64_t, 3> classes{};
  std::int64_t first_timestamp = 0, last_timestamp = 0;
  double parse_ns = 0, features_ns = 0, inference_ns = 0, output_ns = 0, wall_ns = 0;
  double checksum = 0;
};

Statistics replay(std::istream& input, const std::string& source, const lob::LightGBMModel& model,
                  double tau, AtomicOutput* output, bool profile) {
  Statistics stats;
  const auto start = Clock::now();
  auto stage = profile ? Clock::now() : Clock::time_point{};
  lob::CsvSnapshotReader reader(input, source);
  if (profile) stats.parse_ns += ns(Clock::now() - stage);
  if (output) {
    stage = profile ? Clock::now() : Clock::time_point{};
    output->stream() << "row_id,timestamp_ms,last_update_id,raw_down,raw_flat,raw_up,p_down,p_flat,p_up,class_id\n";
    output->stream() << std::setprecision(17);
    if (profile) stats.output_ns += ns(Clock::now() - stage);
  }
  lob::FeatureEngine engine;
  lob::FeatureVector features{};
  lob::SnapshotRecord record;
  for (;;) {
    stage = profile ? Clock::now() : Clock::time_point{};
    const bool exists = reader.next(record);
    if (profile) stats.parse_ns += ns(Clock::now() - stage);
    if (!exists) break;
    if (!stats.rows) stats.first_timestamp = record.book.timestamp_ms;
    else if (record.book.timestamp_ms - stats.last_timestamp > lob::kMaxGapMs) ++stats.resets;
    stats.last_timestamp = record.book.timestamp_ms;
    ++stats.rows;
    stage = profile ? Clock::now() : Clock::time_point{};
    const bool ready = engine.update(record.book, features);
    if (profile) stats.features_ns += ns(Clock::now() - stage);
    if (!ready) { ++stats.warmup; continue; }
    stage = profile ? Clock::now() : Clock::time_point{};
    const auto prediction = model.predict(features, tau);
    if (profile) stats.inference_ns += ns(Clock::now() - stage);
    ++stats.predictions;
    ++stats.classes[static_cast<std::size_t>(prediction.class_id)];
    for (double value : prediction.raw_scores) stats.checksum += value;
    for (double value : prediction.probabilities) stats.checksum += value;
    stats.checksum += prediction.class_id;
    if (output) {
      stage = profile ? Clock::now() : Clock::time_point{};
      auto& stream = output->stream();
      stream << record.row_id << ',' << record.book.timestamp_ms << ',' << record.last_update_id;
      for (double score : prediction.raw_scores) stream << ',' << score;
      for (double probability : prediction.probabilities) stream << ',' << probability;
      stream << ',' << prediction.class_id << '\n';
      if (profile) stats.output_ns += ns(Clock::now() - stage);
    }
  }
  if (output) {
    stage = profile ? Clock::now() : Clock::time_point{};
    output->flush();
    if (profile) stats.output_ns += ns(Clock::now() - stage);
  }
  stats.wall_ns = ns(Clock::now() - start);
  if (!std::isfinite(stats.checksum)) throw std::overflow_error("Nonfinite prediction checksum");
  return stats;
}

std::string summary(const Statistics& s, const Options& o, std::size_t trees, double load_ns) {
  std::ostringstream out;
  out.imbue(std::locale::classic());
  out << std::setprecision(17)
      << "{\n  \"schema_version\": 1,\n  \"snapshots\": " << s.rows
      << ",\n  \"warmup_rows\": " << s.warmup
      << ",\n  \"predictions\": " << s.predictions
      << ",\n  \"gap_resets\": " << s.resets
      << ",\n  \"segments\": " << (s.rows ? s.resets + 1 : 0)
      << ",\n  \"first_timestamp_ms\": " << s.first_timestamp
      << ",\n  \"last_timestamp_ms\": " << s.last_timestamp
      << ",\n  \"classes_down_flat_up\": [" << s.classes[0] << ',' << s.classes[1] << ',' << s.classes[2] << ']'
      << ",\n  \"tau\": " << o.tau << ",\n  \"trees\": " << trees
      << ",\n  \"max_gap_ms\": " << lob::kMaxGapMs
      << ",\n  \"output_enabled\": " << (o.no_output ? "false" : "true")
      << ",\n  \"profiled\": " << (o.profile ? "true" : "false")
      << ",\n  \"model_load_ns\": " << load_ns
      << ",\n  \"replay_wall_ns\": " << s.wall_ns
      << ",\n  \"snapshots_per_second\": " << (s.wall_ns > 0 ? static_cast<double>(s.rows) * 1e9 / s.wall_ns : 0)
      << ",\n  \"predictions_per_second\": " << (s.wall_ns > 0 ? static_cast<double>(s.predictions) * 1e9 / s.wall_ns : 0)
      << ",\n  \"profile_stage_ns\": {\"parse_validate\":" << s.parse_ns
      << ",\"features\":" << s.features_ns << ",\"inference\":" << s.inference_ns
      << ",\"output\":" << s.output_ns << '}'
      << ",\n  \"checksum\": " << s.checksum << "\n}\n";
  return out.str();
}
}  // namespace

int main(int argc, char** argv) {
  if (argc == 2 && std::string(argv[1]) == "--help") {
    std::cout << "Usage: lob_replay --input recording.csv --tau 0.31 [--model contract/model.txt]\n"
                 "                  (--output predictions.csv | --no-output) [--summary result.json] [--profile]\n"
                 "Replay runs at maximum speed. Existing outputs are never overwritten.\n";
    return 0;
  }
  try {
    const auto o = options(argc, argv);
    // Existing paths (including input/model) are protected by AtomicOutput.
    if (!o.summary.empty() && !o.output.empty() &&
        std::filesystem::weakly_canonical(o.summary) == std::filesystem::weakly_canonical(o.output)) {
      throw std::invalid_argument("Summary and predictions must have different paths");
    }
    std::ifstream input(o.input, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot open input: " + o.input.string());
    const auto load_start = Clock::now();
    const auto model = lob::LightGBMModel::load(o.model);
    const double load_ns = ns(Clock::now() - load_start);
    std::unique_ptr<AtomicOutput> output, summary_file;
    if (!o.no_output) output = std::make_unique<AtomicOutput>(o.output);
    if (!o.summary.empty()) summary_file = std::make_unique<AtomicOutput>(o.summary);
    const auto stats = replay(input, o.input.string(), model, o.tau, output.get(), o.profile);
    const auto json = summary(stats, o, model.num_trees(), load_ns);
    if (summary_file) { summary_file->stream() << json; summary_file->flush(); }
    if (output) output->publish();
    if (summary_file) summary_file->publish();
    std::cout << json;
    std::cout.flush();
    if (!std::cout) throw std::runtime_error("Cannot write terminal summary");
  } catch (const std::exception& error) {
    std::cerr << "Replay error: " << error.what() << '\n';
    return 1;
  }
}
