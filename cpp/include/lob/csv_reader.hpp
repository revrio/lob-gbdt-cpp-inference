#pragma once
#include <cstdint>
#include <istream>
#include <string>
#include "lob/types.hpp"

namespace lob {
struct SnapshotRecord {
  std::uint64_t row_id = 0;
  std::uint64_t last_update_id = 0;
  BookSnapshot book;
};

// Streaming reader for the canonical, unquoted raw snapshot CSV. Valid gaps
// pass through to FeatureEngine; invalid rows fail before modifying the output.
class CsvSnapshotReader {
 public:
  CsvSnapshotReader(std::istream& input, std::string source);
  bool next(SnapshotRecord& record);
  static std::string header();
 private:
  bool line(std::string& result);
  [[noreturn]] void fail(const std::string& message) const;
  std::istream& input_;
  std::string source_;
  std::size_t line_number_ = 0;
  std::uint64_t rows_ = 0;
  std::int64_t previous_timestamp_ = 0;
  std::uint64_t previous_id_ = 0;
};
}  // namespace lob
