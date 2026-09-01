# SPDX-License-Identifier: MPL-2.0
# Minimal external consumer example
#
#   cmake -S examples/external_consumer -B /tmp/consumer-build
#   cmake --build /tmp/consumer-build -j
#   /tmp/consumer-build/consumer <hifigan.gguf> <mel.bin> <f0.bin> <out.wav>
cmake_minimum_required(VERSION 3.18)
project(pc_nsf_hifigan_consumer CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Consume the library via add_subdirectory (also the only cross-repo pattern
# available w/o install; install/export works identically).
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../..
                  ${CMAKE_BINARY_DIR}/pc_nsf_hifigan)

add_executable(consumer main.cpp)
target_link_libraries(consumer PRIVATE pc_nsf_hifigan::pc_nsf_hifigan)
