#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "lob/feature_engine.hpp"
#include "lob/model.hpp"

namespace {
using Clock = std::chrono::steady_clock;
using Nanoseconds = std::chrono::duration<double, std::nano>;

struct Options {
  std::filesystem::path model = "contract/model.txt";
  std::filesystem::path input = "contract/golden_input_lob.csv";
  std::filesystem::path output;
  double tau = -1;
  std::size_t passes = 20;
  std::size_t latency_passes = 5;
};

std::size_t positive_integer(const std::string& s) {
  std::size_t used = 0;
  if (s.empty() || s.find_first_not_of("0123456789") != std::string::npos) {
    throw std::invalid_argument("Expected a positive integer");
  }
  const auto n = std::stoull(s, &used);
  if (used != s.size() || n == 0 || n > 10000) throw std::invalid_argument("Pass count must be 1..10000");
  return static_cast<std::size_t>(n);
}

double finite_number(const std::string& s) {
  std::size_t used = 0;
  double n = 0;
  try { n = std::stod(s, &used); }
  catch (const std::exception&) { throw std::invalid_argument("Expected finite number: " + s); }
  if (used != s.size() || !std::isfinite(n)) throw std::invalid_argument("Expected finite number");
  return n;
}

Options options(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    if (key == "--help") {
      std::cout << "Usage: lob_bench --tau 0.31 [--model PATH] [--input PATH]\n"
                   "                 [--passes 20] [--latency-passes 5] [--output PATH]\n"
                   "Runs an in-memory, warm-cache benchmark; input is the canonical golden CSV.\n";
      std::exit(0);
    }
    if (i + 1 == argc) throw std::invalid_argument("Missing value for " + key);
    const std::string value = argv[++i];
    if (key == "--model") o.model = value;
    else if (key == "--input") o.input = value;
    else if (key == "--output") o.output = value;
    else if (key == "--tau") o.tau = finite_number(value);
    else if (key == "--passes") o.passes = positive_integer(value);
    else if (key == "--latency-passes") o.latency_passes = positive_integer(value);
    else throw std::invalid_argument("Unknown option: " + key);
  }
  if (o.tau < 0 || o.tau > 1) throw std::invalid_argument("Supply --tau in [0,1] from contract/metadata.json");
  return o;
}

std::vector<std::string> cells(std::string line) {
  if (!line.empty() && line.back() == '\r') line.pop_back();
  std::vector<std::string> out;
  std::istringstream stream(line);
  std::string cell;
  while (std::getline(stream, cell, ',')) out.push_back(cell);
  return out;
}

std::int64_t nonnegative_integer(const std::string& s) {
  if (s.empty() || s.find_first_not_of("0123456789") != std::string::npos) {
    throw std::invalid_argument("Invalid CSV integer");
  }
  return std::stoll(s);
}

// Deliberately narrow reader: canonical unquoted fixture CSV, one contiguous
// segment. General replay input parsing is a separate milestone.
std::vector<lob::BookSnapshot> load_inputs(const std::filesystem::path& path) {
  std::ifstream stream(path);
  if (!stream) throw std::runtime_error("Cannot open inputs: " + path.string());
  std::ostringstream header;
  header << "row_id,timestamp_ms,last_update_id";
  for (std::size_t i = 1; i <= lob::kLevels; ++i) {
    header << ",bid_px_" << i << ",bid_sz_" << i << ",ask_px_" << i << ",ask_sz_" << i;
  }
  std::string line;
  if (!std::getline(stream, line) || cells(line) != cells(header.str())) {
    throw std::invalid_argument("Expected canonical golden_input_lob.csv header");
  }
  std::vector<lob::BookSnapshot> result;
  std::int64_t previous_id = -1;
  while (std::getline(stream, line)) {
    const auto c = cells(line);
    if (c.size() != 3 + 4 * lob::kLevels || nonnegative_integer(c[0]) != static_cast<std::int64_t>(result.size())) {
      throw std::invalid_argument("CSV shape or row_id mismatch");
    }
    lob::BookSnapshot s;
    s.timestamp_ms = nonnegative_integer(c[1]);
    const auto id = nonnegative_integer(c[2]);
    if (id <= previous_id) throw std::invalid_argument("Update IDs must increase");
    previous_id = id;
    if (!result.empty() && (s.timestamp_ms < result.back().timestamp_ms ||
        s.timestamp_ms - result.back().timestamp_ms > lob::kMaxGapMs)) {
      throw std::invalid_argument("Benchmark fixture must be a single ordered segment");
    }
    for (std::size_t i = 0; i < lob::kLevels; ++i) {
      s.bid_px[i] = finite_number(c[3 + 4 * i]);
      s.bid_sz[i] = finite_number(c[4 + 4 * i]);
      s.ask_px[i] = finite_number(c[5 + 4 * i]);
      s.ask_sz[i] = finite_number(c[6 + 4 * i]);
      if (s.bid_px[i] <= 0 || s.ask_px[i] <= 0 || s.bid_sz[i] <= 0 || s.ask_sz[i] <= 0 ||
          (i && (s.bid_px[i] >= s.bid_px[i - 1] || s.ask_px[i] <= s.ask_px[i - 1]))) {
        throw std::invalid_argument("Invalid book ladder, price or quantity");
      }
    }
    if (s.bid_px[0] >= s.ask_px[0]) throw std::invalid_argument("Crossed book");
    result.push_back(s);
  }
  if (stream.bad()) throw std::runtime_error("CSV read failed");
  if (result.size() <= lob::kWarmupTicks) throw std::invalid_argument("Not enough snapshots after warm-up");
  return result;
}

// Compiler barriers keep the work/result stores within each timestamp boundary.
// No CPU serialization instruction is needed for this steady_clock baseline.
Clock::time_point timestamp() {
  std::atomic_signal_fence(std::memory_order_seq_cst);
  const auto t = Clock::now();
  std::atomic_signal_fence(std::memory_order_seq_cst);
  return t;
}

double percentile(std::vector<double> sorted, double q) {
  std::sort(sorted.begin(), sorted.end());
  const auto rank = static_cast<std::size_t>(std::ceil(q * static_cast<double>(sorted.size())));
  return sorted[std::max<std::size_t>(1, rank) - 1];
}

std::string quote(const std::string& s) {
  std::ostringstream out;
  out << '"';
  for (unsigned char c : s) {
    if (c == '"' || c == '\\') out << '\\' << c;
    else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(c) << std::dec;
    else out << c;
  }
  out << '"';
  return out.str();
}

std::string cpu_name() {
  std::ifstream cpu("/proc/cpuinfo");
  std::string line;
  while (std::getline(cpu, line)) {
    if (line.starts_with("model name")) return line.substr(line.find(':') + 2);
  }
  return "unknown (record hardware separately)";
}

enum class Mode { Features, Inference, Combined };
const char* name(Mode m) {
  if (m == Mode::Features) return "features";
  if (m == Mode::Inference) return "inference";
  return "combined";
}

struct Measurement {
  std::vector<double> latency_ns;
  std::vector<double> pass_ns;
  double checksum = 0;
};

Measurement measure(Mode mode, const Options& o, const lob::LightGBMModel& model,
                    const std::vector<lob::BookSnapshot>& snapshots,
                    const std::vector<lob::FeatureVector>& prepared) {
  const auto n = prepared.size();
  // Allocate outside timing. Observable output stores prevent dead-code removal;
  // both throughput and latency timings include writing the returned result.
  std::vector<lob::FeatureVector> features(n);
  std::vector<lob::Prediction> predictions(n);
  Measurement result;
  result.pass_ns.reserve(o.passes);
  result.latency_ns.reserve(n * o.latency_passes);
  const auto consume = [&]() {
    double sum = 0;
    if (mode == Mode::Features) {
      for (const auto& row : features) for (double x : row) sum += x;
    } else {
      for (const auto& p : predictions) {
        for (double x : p.raw_scores) sum += x;
        for (double x : p.probabilities) sum += x;
        sum += p.class_id;
      }
    }
    result.checksum += sum;
  };
  const auto pass = [&](bool individual, bool record) {
    lob::FeatureEngine engine;
    lob::FeatureVector x{};
    if (mode != Mode::Inference) {
      for (std::size_t i = 0; i < lob::kWarmupTicks; ++i) engine.update(snapshots[i], x);
    }
    const auto operation = [&](std::size_t i) {
      if (mode == Mode::Inference) predictions[i] = model.predict(prepared[i], o.tau);
      else {
        const bool ready = engine.update(snapshots[i + lob::kWarmupTicks], x);
        if (!ready) throw std::logic_error("Unexpected reset/readiness in benchmark");
        if (mode == Mode::Features) features[i] = x;
        else predictions[i] = model.predict(x, o.tau);
      }
    };
    if (individual) {
      for (std::size_t i = 0; i < n; ++i) {
        const auto begin = timestamp();
        operation(i);
        const auto end = timestamp();
        result.latency_ns.push_back(Nanoseconds(end - begin).count());
      }
    } else {
      const auto begin = timestamp();
      for (std::size_t i = 0; i < n; ++i) operation(i);
      const auto end = timestamp();
      if (record) result.pass_ns.push_back(Nanoseconds(end - begin).count());
    }
    consume();  // Outside the timed region, uses every output field.
  };
  pass(false, false);
  pass(false, false);  // Warm model, code and output pages before measurement.
  for (std::size_t i = 0; i < o.passes; ++i) pass(false, true);
  for (std::size_t i = 0; i < o.latency_passes; ++i) pass(true, true);
  if (!std::isfinite(result.checksum)) throw std::runtime_error("Nonfinite output checksum");
  return result;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    const auto o = options(argc, argv);
    const auto same_file = [&](const std::filesystem::path& input) {
      return std::filesystem::absolute(o.output).lexically_normal() == std::filesystem::absolute(input).lexically_normal() ||
             (std::filesystem::exists(o.output) && std::filesystem::equivalent(o.output, input));
    };
    if (!o.output.empty() && (same_file(o.model) || same_file(o.input))) {
      throw std::invalid_argument("Output must not overwrite benchmark inputs");
    }
    const auto model = lob::LightGBMModel::load(o.model);
    const auto snapshots = load_inputs(o.input);
    std::vector<lob::FeatureVector> prepared;
    prepared.reserve(snapshots.size() - lob::kWarmupTicks);
    lob::FeatureEngine engine;
    lob::FeatureVector x{};
    for (const auto& s : snapshots) if (engine.update(s, x)) prepared.push_back(x);
    std::vector<double> overhead;
    overhead.reserve(10000);
    for (int i = 0; i < 10000; ++i) {
      const auto begin = timestamp();
      const auto end = timestamp();
      overhead.push_back(Nanoseconds(end - begin).count());
    }
    std::ostringstream json;
    json << std::setprecision(12);
    json << "{\n  \"benchmark\": \"warm in-memory engine baseline\",\n"
         << "  \"compiler\": " << quote(LOB_BENCH_COMPILER) << ",\n"
         << "  \"build_type\": " << quote(LOB_BENCH_BUILD_TYPE) << ",\n"
         << "  \"configured_cxx_flags\": " << quote(LOB_BENCH_FLAGS) << ",\n"
         << "  \"cpu\": " << quote(cpu_name()) << ",\n"
         << "  \"hardware_threads\": " << std::thread::hardware_concurrency() << ",\n"
         << "  \"model_path\": " << quote(std::filesystem::absolute(o.model).string()) << ",\n"
         << "  \"input_path\": " << quote(std::filesystem::absolute(o.input).string()) << ",\n"
         << "  \"trees\": " << model.num_trees() << ",\n"
         << "  \"tau\": " << o.tau << ",\n"
         << "  \"snapshots\": " << snapshots.size() << ",\n"
         << "  \"ready_updates_per_pass\": " << prepared.size() << ",\n"
         << "  \"warmup_updates_excluded_per_pass\": " << lob::kWarmupTicks << ",\n"
         << "  \"untimed_warming_passes_per_mode\": 2,\n"
         << "  \"throughput_passes\": " << o.passes << ",\n"
         << "  \"latency_passes\": " << o.latency_passes << ",\n"
         << "  \"clock_empty_interval_p50_ns\": " << percentile(overhead, .50) << ",\n"
         << "  \"clock_empty_interval_p99_ns\": " << percentile(overhead, .99) << ",\n"
         << "  \"cpu_pinned\": false,\n"
         << "  \"includes\": \"engine work and result stores; no clock-overhead subtraction\",\n"
         << "  \"excludes\": \"loading, parsing, checksum reduction, screen output, networking, per-segment warm-up\",\n"
         << "  \"results\": {\n";
    bool first = true;
    for (Mode mode : {Mode::Features, Mode::Inference, Mode::Combined}) {
      std::cerr << "Measuring " << name(mode) << "...\n";
      const auto m = measure(mode, o, model, snapshots, prepared);
      const double median = percentile(m.pass_ns, .50);
      if (!first) json << ",\n";
      first = false;
      json << "    " << quote(name(mode)) << ": {\n"
           << "      \"latency_samples\": " << m.latency_ns.size() << ",\n"
           << "      \"p50_ns\": " << percentile(m.latency_ns, .50) << ",\n"
           << "      \"p95_ns\": " << percentile(m.latency_ns, .95) << ",\n"
           << "      \"p99_ns\": " << percentile(m.latency_ns, .99) << ",\n"
           << "      \"max_ns\": " << *std::max_element(m.latency_ns.begin(), m.latency_ns.end()) << ",\n"
           << "      \"median_pass_ns\": " << median << ",\n"
           << "      \"ns_per_update_from_median_pass\": " << median / static_cast<double>(prepared.size()) << ",\n"
           << "      \"updates_per_second_from_median_pass\": " << static_cast<double>(prepared.size()) * 1e9 / median << ",\n"
           << "      \"checksum\": " << m.checksum << "\n    }";
    }
    json << "\n  }\n}\n";
    if (!o.output.empty()) {
      std::ofstream file(o.output);
      if (!file || !(file << json.str())) throw std::runtime_error("Cannot write benchmark output");
    }
    std::cout << json.str();
  } catch (const std::exception& error) {
    std::cerr << "Benchmark error: " << error.what() << '\n';
    return 1;
  }
}
