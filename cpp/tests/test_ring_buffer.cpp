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