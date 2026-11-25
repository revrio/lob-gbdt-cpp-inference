#include "terminal_view.hpp"
#include <algorithm>
#include <ctime>
#include <iomanip>
#include <locale>
#include <sstream>
#include <stdexcept>

namespace lob::app {
namespace {
const char* class_name(int id) {
  constexpr const char* names[] = {"DOWN", "FLAT", "UP"};
  return names[id];
}

std::string utc_time(std::int64_t timestamp_ms) {
  const std::time_t seconds = timestamp_ms / 1000;
  const auto* time = std::gmtime(&seconds);
  if (!time) return std::to_string(timestamp_ms) + "ms since epoch";
  std::ostringstream out;
  out << std::put_time(time, "%Y-%m-%d %H:%M:%S") << '.'
      << std::setfill('0') << std::setw(3) << timestamp_ms % 1000 << " UTC";
  return out.str();
}
}

std::string playback_frame(const ReplayViewState& s, double speed, double tau,
                           bool keep_gaps, bool compact) {
  std::ostringstream out;
  out.imbue(std::locale::classic());
  out << std::fixed << std::setprecision(2);
  if (!compact) {
    out << "RECORDED LOB REPLAY | " << speed << "x | "
        << (keep_gaps ? "original gap waits" : "gap waits >1000ms skipped") << '\n'
        << utc_time(s.record.book.timestamp_ms) << " | Ctrl+C to stop\n";
  }
  if (compact) {
    out << "row=" << s.record.row_id << " timestamp_ms=" << s.record.book.timestamp_ms
        << " snapshots=" << s.snapshots << " predictions=" << s.predictions
        << " resets=" << s.resets << " skipped_gaps=" << s.skipped_gaps << " | ";
  } else {
    out << "Row " << s.record.row_id << " | predictions " << s.predictions
        << " | resets " << s.resets << " | skipped gaps " << s.skipped_gaps << '\n';
  }
  if (s.prediction) {
    const auto& p = *s.prediction;
    out << "READY | Down " << p.probabilities[0] * 100 << "% | Flat "
        << p.probabilities[1] * 100 << "% | Up " << p.probabilities[2] * 100
        << "% | decision=" << class_name(p.class_id) << " (tau=" << tau << ')';
  } else {
    out << "WARMUP " << std::min<std::uint64_t>(s.segment_updates, kWarmupTicks)
        << '/' << kWarmupTicks << " | probabilities unavailable";
  }
  out << (compact ? " | " : "\n") << "engine_update_us=" << s.engine_us
      << " schedule_lag_ms=" << s.lag_ms;
  const auto& b = s.record.book;
  out << (compact ? " | " : "\n") << "mid=" << std::setprecision(3)
      << (b.bid_px[0] + b.ask_px[0]) / 2 << " spread=" << std::setprecision(2)
      << b.ask_px[0] - b.bid_px[0] << '\n';
  if (!compact) {
    out << "\n     Bid quantity      Bid price |      Ask price     Ask quantity\n";
    for (std::size_t i = 0; i < kLevels; ++i) {
      out << std::setw(17) << std::setprecision(5) << b.bid_sz[i]
          << std::setw(15) << std::setprecision(2) << b.bid_px[i] << " |"
          << std::setw(15) << b.ask_px[i]
          << std::setw(17) << std::setprecision(5) << b.ask_sz[i] << '\n';
    }
    out << "\nDecision uses tau, not necessarily the largest probability.\n"
        << "Engine timing excludes playback waits and display. No orders are sent.\n";
  }
  return out.str();
}

TerminalView::TerminalView(std::ostream& output, bool interactive, double speed,
                           double tau, bool keep_gaps)
    : output_(output), interactive_(interactive), speed_(speed), tau_(tau), keep_gaps_(keep_gaps) {}

TerminalView::~TerminalView() {
  if (active_ && interactive_) {
    // Restore the main screen and cursor during both normal return and exceptions.
    try { output_ << "\033[?25h\033[?1049l" << std::flush; } catch (...) {}
  }
}

bool TerminalView::update(const ReplayViewState& state, bool force) {
  const auto now = std::chrono::steady_clock::now();
  if (active_ && !force && now - last_frame_ < std::chrono::milliseconds(200)) return false;
  if (interactive_) {
    if (!active_) output_ << "\033[?1049h\033[?25l";
    output_ << "\033[H\033[2J";
  }
  // Set active before writing so an output failure still restores the terminal.
  active_ = true;
  output_ << playback_frame(state, speed_, tau_, keep_gaps_, !interactive_) << std::flush;
  if (!output_) throw std::runtime_error("Cannot write playback display");
  last_frame_ = now;
  return true;
}

}  // namespace lob::app
