#include <gtest/gtest.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include "lob/csv_reader.hpp"

namespace {
std::string row(int index, long long timestamp, unsigned long long id) {
  std::ostringstream out;
  out << index << ',' << timestamp << ',' << id;
  for (int i = 0; i < 10; ++i) out << ',' << 100-i << ",3," << 101+i << ",1";
  return out.str();
}
std::string replace(std::string text, const std::string& old, const std::string& value) {
  text.replace(text.find(old), old.size(), value); return text;
}
}

TEST(CsvReader, ReadsCRLFAndFinalLineWithoutNewline) {
  std::istringstream input(lob::CsvSnapshotReader::header()+"\r\n"+row(0,0,1)+"\r\n"+row(1,100,2));
  lob::CsvSnapshotReader reader(input, "sample.csv");
  lob::SnapshotRecord record;
  ASSERT_TRUE(reader.next(record));
  EXPECT_EQ(record.row_id, 0U);
  EXPECT_DOUBLE_EQ(record.book.bid_px[9], 91);
  ASSERT_TRUE(reader.next(record));
  EXPECT_EQ(record.book.timestamp_ms, 100);
  EXPECT_FALSE(reader.next(record));
}

TEST(CsvReader, AllowsEqualTimestampsAndLongGaps) {
  std::istringstream input(lob::CsvSnapshotReader::header()+"\n"+row(0,0,1)+"\n"+row(1,0,2)+"\n"+row(2,50000,3));
  lob::CsvSnapshotReader reader(input, "sample.csv");
  lob::SnapshotRecord record;
  EXPECT_TRUE(reader.next(record)); EXPECT_TRUE(reader.next(record)); EXPECT_TRUE(reader.next(record));
}

TEST(CsvReader, RejectsMissingOrIncorrectHeader) {
  std::istringstream empty;
  EXPECT_THROW((lob::CsvSnapshotReader(empty, "empty.csv")), std::runtime_error);
  std::istringstream wrong("timestamp_ms\n");
  EXPECT_THROW((lob::CsvSnapshotReader(wrong, "wrong.csv")), std::runtime_error);
}

TEST(CsvReader, MalformedRowHasSourceAndLineAndDoesNotModifyOutput) {
  std::istringstream input(lob::CsvSnapshotReader::header()+"\n"+row(0,0,1)+"\nwrong\n");
  lob::CsvSnapshotReader reader(input, "capture.csv");
  lob::SnapshotRecord record;
  ASSERT_TRUE(reader.next(record));
  try { reader.next(record); FAIL() << "Accepted malformed row"; }
  catch (const std::runtime_error& error) {
    EXPECT_NE(std::string(error.what()).find("capture.csv:3:"), std::string::npos);
  }
  EXPECT_EQ(record.row_id, 0U);
}

TEST(CsvReader, RejectsNumericShapesAndInvalidBooks) {
  for (const auto& value : {replace(row(0,0,1), ",100,3,101,1", ",nan,3,101,1"),
                            replace(row(0,0,1), ",100,3,101,1", ",100,0,101,1"),
                            replace(row(0,0,1), ",100,3,101,1", ",102,3,101,1"),
                            replace(row(0,0,1), ",99,3,102,1", ",100,3,102,1"),
                            replace(row(0,0,1), ",100,3,101,1", ",100x,3,101,1"),
                            row(0,0,1)+",extra", row(0,0,1)+",", std::string(""), row(0,-1,1)}) {
    std::istringstream input(lob::CsvSnapshotReader::header()+"\n"+value+"\n");
    lob::CsvSnapshotReader reader(input, "bad.csv");
    lob::SnapshotRecord record;
    EXPECT_THROW(reader.next(record), std::runtime_error) << value;
  }
}

TEST(CsvReader, RejectsOutOfOrderAndDuplicateIdentifiers) {
  for (const auto& value : {row(2,100,2), row(1,99,2), row(1,101,1)}) {
    std::istringstream input(lob::CsvSnapshotReader::header()+"\n"+row(0,100,1)+"\n"+value);
    lob::CsvSnapshotReader reader(input, "bad.csv"); lob::SnapshotRecord record;
    ASSERT_TRUE(reader.next(record));
    EXPECT_THROW(reader.next(record), std::runtime_error);
  }
}

TEST(CsvReader, RejectsOversizedRowsEmbeddedNulAndBrokenStreams) {
  for (const auto& value : {std::string(17000, '1'), row(0,0,1)+std::string(1, '\0')}) {
    std::istringstream input(lob::CsvSnapshotReader::header()+"\n"+value+"\n");
    lob::CsvSnapshotReader reader(input, "bad.csv"); lob::SnapshotRecord record;
    EXPECT_THROW(reader.next(record), std::runtime_error);
  }
  std::istringstream input(lob::CsvSnapshotReader::header()+"\n");
  lob::CsvSnapshotReader reader(input, "broken.csv"); lob::SnapshotRecord record;
  input.setstate(std::ios::badbit);
  EXPECT_THROW(reader.next(record), std::runtime_error);
}
