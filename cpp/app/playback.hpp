#pragma once
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "lob/types.hpp"

namespace lob::app {

// Only calculates deadlines. Keeping sleep out of this class lets tests use an
// artificial clock and check long recording gaps without actually waiting.
class PlaybackTimeline {
 public:
  using Clock = std::chrono::steady_clock;
  struct Step {
    Clock::time_point deadline;
    bool skipped_gap;
  };

  PlaybackTimeline(double speed, bool keep_gaps) : speed_(speed), keep_gaps_(keep_gaps) {
    if (!std::isfinite(speed) || speed < 0.01 || speed > 1000) {
      throw std::invalid_argument("Playback speed must be in [0.01,1000]");
    }
  }

  Step next(std::int64_t timestamp_ms, Clock::time_point now) {
    if (timestamp_ms < 0 || (started_ && timestamp_ms < previous_ms_)) {
      throw std::invalid_argument("Playback timestamps must be nonnegative and nondecreasing");
    }
    if (!started_) {
      started_ = true;
      anchor_ = now;
      previous_ms_ = timestamp_ms;
      return {anchor_, false};
    }
    const auto interval = timestamp_ms - previous_ms_;
    const bool skip = interval > kMaxGapMs && !keep_gaps_;
    // Accumulate source milliseconds, rather than rounding each scaled interval.
    // All deadlines use one anchor: processing time does not add playback drift.
    const auto elapsed = elapsed_ms_ + (skip ? 0 : interval);
    const auto ticks = std::chrono::duration<long double, Clock::period>(
        std::chrono::duration<long double, std::milli>(elapsed / static_cast<long double>(speed_))).count();
    const long double remaining = static_cast<long double>(Clock::time_point::max().time_since_epoch().count())
        - anchor_.time_since_epoch().count();
    if (ticks >= remaining || ticks >= static_cast<long double>(std::numeric_limits<Clock::duration::rep>::max())) {
      throw std::overflow_error("Recording duration exceeds the playback clock range");
    }
    elapsed_ms_ = elapsed;
    previous_ms_ = timestamp_ms;
    return {anchor_ + Clock::duration(static_cast<Clock::duration::rep>(ticks)), skip};
  }

 private:
  double speed_;
  bool keep_gaps_, started_ = false;
  std::int64_t previous_ms_ = 0, elapsed_ms_ = 0;
  Clock::time_point anchor_{};
};

}  // namespace lob::app
