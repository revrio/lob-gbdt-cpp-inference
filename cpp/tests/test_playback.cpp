#include <gtest/gtest.h>
#include <sstream>

#include "playback.hpp"
#include "terminal_view.hpp"

using lob::app::PlaybackTimeline;
using namespace std::chrono_literals;

TEST(Playback, ScalesIntervalsWithoutAddingProcessingDrift) {
  const auto start = PlaybackTimeline::Clock::time_point(10s);
  PlaybackTimeline timeline(10, false);
  EXPECT_EQ(timeline.next(1000, start).deadline, start);
  EXPECT_EQ(timeline.next(1100, start + 2ms).deadline, start + 10ms);
  // Even if processing is late, deadlines stay on the original schedule.
  EXPECT_EQ(timeline.next(1200, start + 90ms).deadline, start + 20ms);
  EXPECT_EQ(timeline.next(1200, start + 91ms).deadline, start + 20ms);
}

TEST(Playback, SkipsOnlyGapsGreaterThanCutoffOrPreservesThemOnRequest) {
  const auto start = PlaybackTimeline::Clock::time_point{};
  PlaybackTimeline skip(1, false);
  skip.next(0, start);
  EXPECT_FALSE(skip.next(1000, start).skipped_gap);
  const auto gap = skip.next(2001, start);
  EXPECT_TRUE(gap.skipped_gap);
  EXPECT_EQ(gap.deadline, start + 1000ms);
  EXPECT_EQ(skip.next(2101, start).deadline, start + 1100ms);

  PlaybackTimeline keep(2, true);
  keep.next(0, start);
  EXPECT_EQ(keep.next(100000, start).deadline, start + 50s);
}

TEST(Playback, RejectsInvalidSpeedOrderAndUnrepresentableDuration) {
  for (double speed : {0., -1., .001, 1001., std::numeric_limits<double>::infinity(),
                       std::numeric_limits<double>::quiet_NaN()}) {
    EXPECT_THROW(PlaybackTimeline(speed, false), std::invalid_argument);
  }
  const auto start = PlaybackTimeline::Clock::time_point{};
  PlaybackTimeline timeline(1, true);
  EXPECT_THROW(timeline.next(-1, start), std::invalid_argument);
  timeline.next(100, start);
  EXPECT_THROW(timeline.next(99, start), std::invalid_argument);
  EXPECT_THROW(timeline.next(std::numeric_limits<std::int64_t>::max(), start), std::overflow_error);
}

TEST(PlaybackView, WarmupHasNoStalePredictionAndReadyShowsThresholdDecision) {
  lob::app::ReplayViewState state;
  state.segment_updates = 12;
  state.record.book.bid_px[0] = 100;
  state.record.book.ask_px[0] = 101;
  auto frame = lob::app::playback_frame(state, 10, .31, false, false);
  EXPECT_NE(frame.find("WARMUP 12/300"), std::string::npos);
  EXPECT_EQ(frame.find("decision="), std::string::npos);
  EXPECT_NE(frame.find("Bid quantity"), std::string::npos);
  lob::Prediction p;
  p.probabilities = {.04, .60, .36};
  p.class_id = 2;
  state.prediction = p;
  frame = lob::app::playback_frame(state, 10, .31, false, true);
  EXPECT_NE(frame.find("READY"), std::string::npos);
  EXPECT_NE(frame.find("decision=UP"), std::string::npos);
  EXPECT_NE(frame.find("Flat 60.00%"), std::string::npos);
  EXPECT_EQ(frame.find('\033'), std::string::npos);
  state.prediction.reset();
  EXPECT_NE(lob::app::playback_frame(state, 10, .31, false, true).find("WARMUP"), std::string::npos);
}

TEST(PlaybackView, InteractiveScreenAndCursorRestoreOnException) {
  std::ostringstream output;
  try {
    lob::app::TerminalView view(output, true, 1, .31, false);
    view.update({});
    throw std::runtime_error("input failed later");
  } catch (const std::runtime_error&) {}
  EXPECT_NE(output.str().find("\033[?1049h"), std::string::npos);
  EXPECT_TRUE(output.str().ends_with("\033[?25h\033[?1049l"));
}
