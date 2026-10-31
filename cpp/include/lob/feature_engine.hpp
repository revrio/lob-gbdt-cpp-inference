#pragma once
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "lob/features.hpp"
#include "lob/ring_buffer.hpp"

namespace lob {

// Streaming implementation of the fixed feature contract. Book validation
// belongs to the input boundary; this engine requires finite, valid prices and
// quantities. All owned history has fixed capacity and is stored in the object.
class FeatureEngine {
 public:
  // Consumes one update. A false return means warm-up: output is untouched.
  // Invalid timestamps are rejected before any state or output is modified.
  bool update(const BookSnapshot& snapshot, FeatureVector& output) {
    if (snapshot.timestamp_ms < 0 ||
        (have_previous_ && snapshot.timestamp_ms < previous_timestamp_)) {
      throw std::invalid_argument("FeatureEngine requires nonnegative, nondecreasing timestamps");
    }
    // Both timestamps are now nonnegative and ordered, so this subtraction
    // cannot overflow an int64_t even at the maximum supported timestamp.
    if (have_previous_ && snapshot.timestamp_ms - previous_timestamp_ > kMaxGapMs) {
      reset();
    }

    const double mid = mid_price(snapshot);
    double ofi = 0.0;
    double dmid = 0.0;
    if (have_previous_) {
      double bid_flow;
      if (snapshot.bid_px[0] > previous_bid_px_) {
        bid_flow = snapshot.bid_sz[0];
      } else if (snapshot.bid_px[0] == previous_bid_px_) {
        bid_flow = snapshot.bid_sz[0] - previous_bid_sz_;
      } else {
        bid_flow = -previous_bid_sz_;
      }

      double ask_flow;
      if (snapshot.ask_px[0] < previous_ask_px_) {
        ask_flow = snapshot.ask_sz[0];
      } else if (snapshot.ask_px[0] == previous_ask_px_) {
        ask_flow = snapshot.ask_sz[0] - previous_ask_sz_;
      } else {
        ask_flow = -previous_ask_sz_;
      }
      ofi = bid_flow - ask_flow;
      dmid = mid - previous_mid_;
    }

    ofi_10_.push(ofi);
    ofi_50_.push(ofi);
    ofi_300_.push(ofi);
    dmid_50_.push(dmid);
    mid_history_.push(mid);

    previous_timestamp_ = snapshot.timestamp_ms;
    previous_bid_px_ = snapshot.bid_px[0];
    previous_bid_sz_ = snapshot.bid_sz[0];
    previous_ask_px_ = snapshot.ask_px[0];
    previous_ask_sz_ = snapshot.ask_sz[0];
    previous_mid_ = mid;
    have_previous_ = true;
    // Count 0..301 and then stop. Readiness needs no unbounded tick counter.
    if (ticks_seen_ <= kWarmupTicks) ++ticks_seen_;
    if (ticks_seen_ <= kWarmupTicks) return false;

    compute_stateless(snapshot, output);
    output[kOfi10] = ofi_10_.sum();
    output[kOfi50] = ofi_50_.sum();
    output[kOfi300] = ofi_300_.sum();
    output[kRet50] = mid - mid_history_.oldest();
    output[kVol50] = dmid_50_.std_pop();
    return true;
  }

  void reset() noexcept {
    ofi_10_.clear();
    ofi_50_.clear();
    ofi_300_.clear();
    dmid_50_.clear();
    mid_history_.clear();
    ticks_seen_ = 0;
    have_previous_ = false;
    previous_timestamp_ = 0;
    previous_bid_px_ = previous_bid_sz_ = 0.0;
    previous_ask_px_ = previous_ask_sz_ = previous_mid_ = 0.0;
  }

 private:
  RollingSum<10> ofi_10_;
  RollingSum<50> ofi_50_;
  RollingSum<300> ofi_300_;
  RollingSum<50> dmid_50_;
  RingBuffer<double, 51> mid_history_;

  std::size_t ticks_seen_ = 0;
  bool have_previous_ = false;
  std::int64_t previous_timestamp_ = 0;
  double previous_bid_px_ = 0.0, previous_bid_sz_ = 0.0;
  double previous_ask_px_ = 0.0, previous_ask_sz_ = 0.0;
  double previous_mid_ = 0.0;
};

}  // namespace lob
