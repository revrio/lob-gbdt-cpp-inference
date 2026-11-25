#pragma once
#include <chrono>
#include <cstdint>
#include <optional>
#include <ostream>
#include <string>

#include "lob/csv_reader.hpp"
#include "lob/model.hpp"

namespace lob::app {

struct ReplayViewState {
  SnapshotRecord record;
  std::optional<Prediction> prediction;
  std::uint64_t snapshots = 0, predictions = 0, resets = 0, skipped_gaps = 0;
  std::uint64_t segment_updates = 0;
  double engine_us = 0, lag_ms = 0;
};

// UI code belongs to the replay application, not the feature/model library.
// stderr carries the view; stdout remains a machine-readable final JSON summary.
class TerminalView {
 public:
  TerminalView(std::ostream& output, bool interactive, double speed, double tau, bool keep_gaps);
  ~TerminalView();
  TerminalView(const TerminalView&) = delete;
  TerminalView& operator=(const TerminalView&) = delete;
  bool update(const ReplayViewState& state, bool force = false);

 private:
  std::ostream& output_;
  bool interactive_, active_ = false;
  double speed_, tau_;
  bool keep_gaps_;
  std::chrono::steady_clock::time_point last_frame_{};
};

std::string playback_frame(const ReplayViewState& state, double speed, double tau,
                           bool keep_gaps, bool compact);

}  // namespace lob::app
