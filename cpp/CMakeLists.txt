cmake_minimum_required(VERSION 3.16)
project(lob_inference CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

include(FetchContent)
FetchContent_Declare(googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip)
FetchContent_MakeAvailable(googletest)

enable_testing()
add_library(lob STATIC src/tree.cpp src/model.cpp)
target_include_directories(lob PUBLIC include)

add_executable(tests
  tests/test_smoke.cpp
  tests/test_ring_buffer.cpp
  tests/test_features.cpp
  tests/test_feature_engine.cpp
  tests/test_feature_parity.cpp
  tests/test_tree.cpp
  tests/test_model.cpp)
target_link_libraries(tests lob GTest::gtest_main)
target_compile_definitions(tests PRIVATE LOB_CONTRACT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../contract")
include(GoogleTest)
gtest_discover_tests(tests)

# A separate executable keeps benchmark timing code out of the engine and tests.
add_executable(lob_bench bench/benchmark.cpp)
target_link_libraries(lob_bench PRIVATE lob)
string(TOUPPER "${CMAKE_BUILD_TYPE}" LOB_BENCH_CONFIG)
target_compile_definitions(lob_bench PRIVATE
  LOB_BENCH_COMPILER="${CMAKE_CXX_COMPILER_ID}-${CMAKE_CXX_COMPILER_VERSION}"
  LOB_BENCH_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
  LOB_BENCH_FLAGS="${CMAKE_CXX_FLAGS}\ ${CMAKE_CXX_FLAGS_${LOB_BENCH_CONFIG}}")
