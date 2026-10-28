#pragma once
#include <array>
#include <cmath>
#include <cstddef>

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
// Used for ofi_10/50/300 (sum) and vol_50 (sum and sum of squares).
template <std::size_t N>
class RollingSum {
 public:
  void push(double x) {
    double old = buf_.push(x);  // 0.0 until the window is full
    sum_ += x - old;
    sum_sq_ += x * x - old * old;
  }
  bool full() const { return buf_.full(); }
  double sum() const { return sum_; }

  // Population std dev, as in the spec (ddof = 0).
  double std_pop() const {
    double mean = sum_ / N;
    double var = sum_sq_ / N - mean * mean;
    return std::sqrt(var > 0.0 ? var : 0.0);
  }

  void clear() { buf_.clear(); sum_ = 0.0; sum_sq_ = 0.0; }

 private:
  RingBuffer<double, N> buf_;
  double sum_ = 0.0;
  double sum_sq_ = 0.0;
};

}  // namespace lob