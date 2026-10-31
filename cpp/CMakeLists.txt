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
add_library(lob INTERFACE)
target_include_directories(lob INTERFACE include)

add_executable(tests
  tests/test_smoke.cpp
  tests/test_ring_buffer.cpp
  tests/test_features.cpp
  tests/test_feature_engine.cpp
  tests/test_feature_parity.cpp)
target_link_libraries(tests lob GTest::gtest_main)
target_compile_definitions(tests PRIVATE LOB_CONTRACT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../contract")
include(GoogleTest)
gtest_discover_tests(tests)
