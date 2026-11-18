#include "lob/csv_reader.hpp"
#include <array>
#include <charconv>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace lob {
namespace {
template <typename T>
T parse(std::string_view text) {
  T value{};
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
  if (text.empty() || error != std::errc{} || end != text.data() + text.size()) {
    throw std::invalid_argument("Invalid numeric field");
  }
  return value;
}
}

std::string CsvSnapshotReader::header() {
  std::ostringstream out;
  out << "row_id,timestamp_ms,last_update_id";
  for (std::size_t i = 1; i <= kLevels; ++i) {
    out << ",bid_px_" << i << ",bid_sz_" << i << ",ask_px_" << i << ",ask_sz_" << i;
  }
  return out.str();
}

[[noreturn]] void CsvSnapshotReader::fail(const std::string& message) const {
  throw std::runtime_error(source_ + ":" + std::to_string(line_number_) + ": " + message);
}

bool CsvSnapshotReader::line(std::string& result) {
  // Bound malformed row memory before allocating. Canonical rows are ~1 KiB.
  std::array<char, 16384> buffer{};
  input_.getline(buffer.data(), static_cast<std::streamsize>(buffer.size()));
  const auto count = input_.gcount();
  if (input_.bad()) fail("Input read failed");
  if (count == 0 && input_.eof()) return false;
  ++line_number_;
  if (input_.fail() && !input_.eof()) fail("Row exceeds 16382 bytes or stream failed");
  const auto length = static_cast<std::size_t>(count) - (input_.eof() ? 0U : 1U);
  result.assign(buffer.data(), length);
  if (result.find('\0') != std::string::npos) fail("Embedded NUL byte");
  if (!result.empty() && result.back() == '\r') result.pop_back();
  return true;
}

CsvSnapshotReader::CsvSnapshotReader(std::istream& input, std::string source)
    : input_(input), source_(std::move(source)) {
  std::string text;
  if (!line(text)) fail("Missing CSV header");
  if (text != header()) fail("Unexpected CSV header; use export_replay.py");
}

bool CsvSnapshotReader::next(SnapshotRecord& record) {
  std::string text;
  if (!line(text)) return false;
  try {
    std::array<std::string_view, 3 + 4 * kLevels> cells;
    std::size_t start = 0;
    for (std::size_t i = 0; i < cells.size(); ++i) {
      const auto end = text.find(',', start);
      const bool last_column = i + 1 == cells.size();
      if ((!last_column && end == std::string::npos) ||
          (last_column && end != std::string::npos)) {
        throw std::invalid_argument("Wrong number of CSV columns");
      }
      const auto length = end == std::string::npos ? text.size() - start : end - start;
      cells[i] = std::string_view(text).substr(start, length);
      start = end == std::string::npos ? text.size() : end + 1;
    }
    SnapshotRecord next;
    next.row_id = parse<std::uint64_t>(cells[0]);
    next.book.timestamp_ms = parse<std::int64_t>(cells[1]);
    next.last_update_id = parse<std::uint64_t>(cells[2]);
    if (next.row_id != rows_) throw std::invalid_argument("row_id must start at zero and increase by one");
    if (next.book.timestamp_ms < 0 || (rows_ && next.book.timestamp_ms < previous_timestamp_)) {
      throw std::invalid_argument("Timestamps must be nonnegative and nondecreasing");
    }
    if (rows_ && next.last_update_id <= previous_id_) throw std::invalid_argument("Update IDs must strictly increase");
    auto& s = next.book;
    for (std::size_t i = 0; i < kLevels; ++i) {
      s.bid_px[i] = parse<double>(cells[3 + 4 * i]);
      s.bid_sz[i] = parse<double>(cells[4 + 4 * i]);
      s.ask_px[i] = parse<double>(cells[5 + 4 * i]);
      s.ask_sz[i] = parse<double>(cells[6 + 4 * i]);
      for (double value : {s.bid_px[i], s.bid_sz[i], s.ask_px[i], s.ask_sz[i]}) {
        if (!std::isfinite(value) || value <= 0) throw std::invalid_argument("Prices/quantities must be finite and positive");
      }
      if (i && (s.bid_px[i] >= s.bid_px[i - 1] || s.ask_px[i] <= s.ask_px[i - 1])) {
        throw std::invalid_argument("Unordered price ladder");
      }
    }
    if (s.bid_px[0] >= s.ask_px[0]) throw std::invalid_argument("Crossed or locked book");
    previous_timestamp_ = s.timestamp_ms;
    previous_id_ = next.last_update_id;
    ++rows_;
    record = next;
    return true;
  } catch (const std::invalid_argument& error) {
    fail(error.what());
  }
}
}  // namespace lob
