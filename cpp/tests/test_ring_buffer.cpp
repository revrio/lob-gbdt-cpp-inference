#include <gtest/gtest.h>
#include "lob/ring_buffer.hpp"

TEST(RingBuffer, FillsThenWraps) {
  lob::RingBuffer<int, 3> rb;
  rb.push(1);
  rb.push(2);
  EXPECT_FALSE(rb.full());
  rb.push(3);
  EXPECT_TRUE(rb.full());
  EXPECT_EQ(rb.oldest(), 1);
  EXPECT_EQ(rb.push(4), 1);  // evicts 1
  EXPECT_EQ(rb.oldest(), 2);
  EXPECT_EQ(rb.back(0), 4);
  EXPECT_EQ(rb.back(2), 2);
}

TEST(RollingSum, MatchesNaiveWindow) {
  lob::RollingSum<4> rs;
  rs.push(1.0);
  rs.push(2.0);
  EXPECT_DOUBLE_EQ(rs.sum(), 3.0);   // not full yet
  rs.push(3.0);
  rs.push(4.0);
  rs.push(5.0);
  rs.push(6.0);
  EXPECT_DOUBLE_EQ(rs.sum(), 18.0);  // 3 + 4 + 5 + 6
}

TEST(RollingSum, PopulationStd) {
  lob::RollingSum<4> rs;
  for (double x : {2.0, 4.0, 4.0, 4.0}) rs.push(x);
  // mean = 3.5, variance = (2.25 + 0.25*3)/4 = 0.75
  EXPECT_NEAR(rs.std_pop(), std::sqrt(0.75), 1e-12);
}

TEST(RollingVariance, PopulationVarianceAndReset) {
  lob::RollingVariance<3> variance;
  variance.push(1);
  variance.push(2);
  variance.push(3);
  EXPECT_NEAR(variance.std_pop(), std::sqrt(2.0 / 3.0), 1e-14);
  variance.push(4);
  EXPECT_NEAR(variance.std_pop(), std::sqrt(2.0 / 3.0), 1e-14);
  variance.clear();
  for (int i = 0; i < 10; ++i) variance.push(7);
  EXPECT_DOUBLE_EQ(variance.std_pop(), 0);
}

TEST(RollingSum, CompensatedRoundingAndConstantWindowMatchPython) {
  // pandas 3.1.0 rolling(3).sum(), including cancellation and eviction.
  const std::array<double, 15> values{.1,.2,-.3,5.5,-2.2,-3.3,0,0,0,1e10,1,-1e10,.1,.2,-.3};
  const std::array<double, 13> expected{0,5.4000000000000004,3,4.4408920985006262e-16,
      -5.5,-3.2999999999999998,0,1e10,10000000001.,1,-9999999998.8999996,-9999999999.7000008,0};
  lob::RollingSum<3> sum;
  for (std::size_t i = 0; i < values.size(); ++i) {
    sum.push(values[i]);
    if (i >= 2) {
      EXPECT_DOUBLE_EQ(sum.sum(), expected[i - 2]) << i;
    }
  }
  sum.clear();
  sum.push(1); sum.push(1); sum.push(1);
  EXPECT_DOUBLE_EQ(sum.sum(), 3);
}
