#pragma once
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>

namespace lob {

// Fixed-capacity circular buffer. No heap allocation.
template <typename T, std::size_t N>
class RingBuffer {
 public:
  // Pushes x and returns the value it evicted (T{} if the buffer was not full yet).
  T push(T x) {
    T evicted{};
    if (size_ == N) evicted = buf_[head_];
    else ++size_;
    buf_[head_] = x;
    if (++head_ == N) head_ = 0;
    return evicted;
  }

  bool full() const { return size_ == N; }
  std::size_t size() const { return size_; }

  // k = 0 is the newest element, k = 1 the one before it, ...
  const T& back(std::size_t k) const { return buf_[(head_ + N - 1 - k) % N]; }

  // Oldest element currently stored (only valid when size() > 0).
  const T& oldest() const { return buf_[size_ == N ? head_ : 0]; }

  void clear() { head_ = 0; size_ = 0; }

 private:
  std::array<T, N> buf_{};
  std::size_t head_ = 0;
  std::size_t size_ = 0;
};

// Rolling sum and sum of squares over the last N values, O(1) per update.
// Used for ofi_10/50/300. std_pop is useful for simple checks; inference
// volatility uses RollingVariance to preserve Python's numerical behaviour.
template <std::size_t N>
class RollingSum {
 public:
  void push(double x) {
    const bool full = buf_.full();
    double old = buf_.push(x);  // 0.0 until the window is full
    // Match the Python reference's separate compensated deletion/addition.
    // Combining x-old changes rounding and can cross a learned tree threshold.
    if (full) accumulate(-old, remove_compensation_);
    accumulate(x, add_compensation_);
    if (x == previous_) {
      if (same_count_ < N) ++same_count_;
    } else {
      same_count_ = 1;
    }
    previous_ = x;
    sum_sq_ += x * x - old * old;
  }
  bool full() const { return buf_.full(); }
  double sum() const {
    return buf_.size() && same_count_ >= buf_.size()
        ? previous_ * static_cast<double>(buf_.size()) : sum_;
  }

  // Population std dev, as in the spec (ddof = 0).
  double std_pop() const {
    double mean = sum() / N;
    double var = sum_sq_ / N - mean * mean;
    return std::sqrt(var > 0.0 ? var : 0.0);
  }

  void clear() {
    buf_.clear(); sum_ = sum_sq_ = add_compensation_ = remove_compensation_ = previous_ = 0;
    same_count_ = 0;
  }

 private:
  void accumulate(double value, double& compensation) {
    const double corrected = value - compensation;
    const double next = sum_ + corrected;
    compensation = (next - sum_) - corrected;
    sum_ = next;
  }
  RingBuffer<double, N> buf_;
  double sum_ = 0.0;
  double sum_sq_ = 0.0;
  double add_compensation_ = 0.0, remove_compensation_ = 0.0, previous_ = 0.0;
  std::size_t same_count_ = 0;
};

// Compensated rolling Welford variance for finite inputs. The operation order
// follows pandas 3.1.0's rolling variance (our frozen reference), including
// separate add/remove compensation and recomputation on severe cancellation.
// See pandas/_libs/window/aggregations.pyx, tag v3.1.0 (BSD-3-Clause).
// Small positive residuals must be preserved: the frozen trees split near zero.
template <std::size_t N>
class RollingVariance {
  static_assert(N > 1);
 public:
  void push(double value) {
    bool unstable = false;
    if (buffer_.full()) update(buffer_.oldest(), false, unstable);
    buffer_.push(value);
    update(value, true, unstable);
    if (unstable) {
      count_ = 0;
      mean_ = m2_ = add_compensation_ = remove_compensation_ = 0;
      for (std::size_t i = buffer_.size(); i > 0; --i) {
        update(buffer_.back(i - 1), true, unstable);
      }
    }
  }

  double std_pop() const {
    return std::sqrt(m2_ > 0 && count_ > 0 ? m2_ / static_cast<double>(count_) : 0.0);
  }

  void clear() {
    buffer_.clear();
    count_ = 0;
    mean_ = m2_ = add_compensation_ = remove_compensation_ = 0;
  }

 private:
  void update(double value, bool adding, bool& unstable) {
    const double before = m2_;
    if (adding) ++count_;
    else --count_;
    if (!count_) {
      mean_ = m2_ = 0;
      unstable = false;
      return;
    }
    double& compensation = adding ? add_compensation_ : remove_compensation_;
    const double previous_mean = mean_ - compensation;
    const double corrected = value - compensation;
    const double delta = corrected - mean_;
    compensation = (delta + mean_) - corrected;
    mean_ += (adding ? delta : -delta) / static_cast<double>(count_);
    const double product = (value - previous_mean) * (value - mean_);
    m2_ += adding ? product : -product;
    if (before * (std::numeric_limits<double>::epsilon() * 1000) > m2_) unstable = true;
  }

  RingBuffer<double, N> buffer_;
  std::size_t count_ = 0;
  double mean_ = 0, m2_ = 0;
  double add_compensation_ = 0, remove_compensation_ = 0;
};

}  // namespace lob
