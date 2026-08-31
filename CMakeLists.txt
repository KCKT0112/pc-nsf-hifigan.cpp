cmake_minimum_required(VERSION 3.18)
project(pc_nsf_hifigan
    VERSION 0.1.0
    DESCRIPTION "NSF-HiFiGAN (mini_nsf) native ggml inference engine"
    LANGUAGES C CXX)

# -----------------------------------------------------------------------------
# Options
# -----------------------------------------------------------------------------
option(PCNSF_CUDA  "Enable ggml CUDA backend"  OFF)
option(PCNSF_VULKAN "Enable ggml Vulkan backend" OFF)
option(PCNSF_BUILD_CLI    "Build hifigan_cli executable" ON)
option(PCNSF_BUILD_TESTS  "Build C++ tests"              ON)
option(PCNSF_BUILD_EXAMPLES "Build external consumer example" OFF)
option(PCNSF_BUILD_BENCHMARK "Build pcnsf_benchmark executable" OFF)

# On Apple, default to Metal backend on. Use PCNSF_METAL to override.
if(APPLE)
    option(PCNSF_METAL "Enable ggml Metal backend" ON)
else()
    option(PCNSF_METAL "Enable ggml Metal backend" OFF)
endif()

# -----------------------------------------------------------------------------
# Global settings
# -----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
if(MSVC)
    foreach(_cfg IN ITEMS Debug Release RelWithDebInfo MinSizeRel)
        string(TOUPPER "${_cfg}" _CFG)
        set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${_CFG} "${CMAKE_BINARY_DIR}/bin")
        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${_CFG} "${CMAKE_BINARY_DIR}/bin")
        set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${_CFG} "${CMAKE_BINARY_DIR}/lib")
    endforeach()
    # MSVC-only flag; must NOT reach NVCC (a bare /utf-8 token makes nvcc
    # treat it as an extra input file -> "A single input file is required").
    add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/utf-8>)
endif()

if(MSVC)
    set(PCNSF_WARN_FLAGS /W3 /wd4244 /wd4267 /wd4996)
else()
    set(PCNSF_WARN_FLAGS -Wall -Wextra -Wno-unused-parameter)
endif()

# -----------------------------------------------------------------------------
# Dependencies (all via FetchContent)
# -----------------------------------------------------------------------------
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Dependencies.cmake)

# -----------------------------------------------------------------------------
# Core library: pc_nsf_hifigan
# -----------------------------------------------------------------------------
set(PCNSF_CORE_SOURCES
    src/gguf_model.cpp
    src/mel.cpp
    src/hifigan.cpp
)

add_library(pc_nsf_hifigan STATIC ${PCNSF_CORE_SOURCES})
add_library(pc_nsf_hifigan::pc_nsf_hifigan ALIAS pc_nsf_hifigan)

target_include_directories(pc_nsf_hifigan
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(pc_nsf_hifigan
    PUBLIC  ggml mininsf::mininsf
)
target_link_libraries(pc_nsf_hifigan PRIVATE pocketfft)
if(MINGW AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
    # gcc<9 needs the separate std::filesystem runtime on MinGW
    target_link_libraries(pc_nsf_hifigan PUBLIC stdc++fs)
endif()

target_compile_options(pc_nsf_hifigan PRIVATE ${PCNSF_WARN_FLAGS})
target_compile_definitions(pc_nsf_hifigan PUBLIC
    PCNSF_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
    PCNSF_VERSION_MINOR=${PROJECT_VERSION_MINOR}
    PCNSF_VERSION_PATCH=${PROJECT_VERSION_PATCH}
)

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
if(PCNSF_BUILD_CLI)
    add_executable(hifigan_cli tools/hifigan_cli.cpp)
    target_link_libraries(hifigan_cli PRIVATE pc_nsf_hifigan::pc_nsf_hifigan dr_wav)
    target_compile_options(hifigan_cli PRIVATE ${PCNSF_WARN_FLAGS})
    add_executable(test_conv_direct tools/test_conv_direct.cpp)
    target_link_libraries(test_conv_direct PRIVATE pc_nsf_hifigan::pc_nsf_hifigan)
    target_compile_options(test_conv_direct PRIVATE ${PCNSF_WARN_FLAGS})
endif()

# -----------------------------------------------------------------------------
# Benchmark (ggml vs ONNX Runtime CPU)
# -----------------------------------------------------------------------------
if(PCNSF_BUILD_BENCHMARK)
    if(NOT ONNX_RUNTIME_ROOT)
        set(ONNX_RUNTIME_ROOT "$ENV{ONNX_RUNTIME_ROOT}" CACHE PATH "ONNX Runtime root")
    endif()
    if(NOT ONNX_RUNTIME_ROOT)
        message(FATAL_ERROR "PCNSF_BUILD_BENCHMARK=ON requires ONNX_RUNTIME_ROOT to be set")
    endif()

    set(ORT_INC "${ONNX_RUNTIME_ROOT}/include")
    if(WIN32)
        set(ORT_LIB "${ONNX_RUNTIME_ROOT}/lib/onnxruntime.lib")
        if(NOT EXISTS "${ORT_LIB}")
            set(ORT_LIB "${ONNX_RUNTIME_ROOT}/onnxruntime.lib")
        endif()
        if(NOT EXISTS "${ORT_LIB}")
            set(ORT_LIB "${CMAKE_CURRENT_SOURCE_DIR}/tools/benchmark/onnxruntime.lib")
        endif()
        set(ORT_DLL "${ONNX_RUNTIME_ROOT}/bin/onnxruntime.dll")
        if(NOT EXISTS "${ORT_DLL}")
            set(ORT_DLL "${ONNX_RUNTIME_ROOT}/onnxruntime.dll")
        endif()
        set(ORT_SHARED_DLL "${ONNX_RUNTIME_ROOT}/bin/onnxruntime_providers_shared.dll")
        if(NOT EXISTS "${ORT_SHARED_DLL}")
            set(ORT_SHARED_DLL "${ONNX_RUNTIME_ROOT}/onnxruntime_providers_shared.dll")
        endif()
    elseif(APPLE)
        set(ORT_LIB "${ONNX_RUNTIME_ROOT}/lib/libonnxruntime.dylib")
    else()
        set(ORT_LIB "${ONNX_RUNTIME_ROOT}/lib/libonnxruntime.so")
    endif()

    add_executable(pcnsf_benchmark
        tools/benchmark/main.cpp
        tools/benchmark/cpu_info.cpp
        tools/benchmark/timer.cpp
        tools/benchmark/ggml_runner.cpp
        tools/benchmark/ort_runner.cpp
        tools/benchmark/accuracy.cpp
        tools/benchmark/report.cpp
    )
    target_include_directories(pcnsf_benchmark PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/tools/benchmark
        ${ORT_INC}
    )
    target_link_libraries(pcnsf_benchmark PRIVATE
        pc_nsf_hifigan::pc_nsf_hifigan
        ${ORT_LIB}
    )
    target_compile_options(pcnsf_benchmark PRIVATE ${PCNSF_WARN_FLAGS})

    # Copy ONNX Runtime shared libs / DLLs next to the executable
    if(WIN32 AND EXISTS "${ORT_DLL}")
        add_custom_COMMAND(TARGET pcnsf_benchmark POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy ${ORT_DLL} $<TARGET_FILE_DIR:pcnsf_benchmark>)
    endif()
    if(WIN32 AND EXISTS "${ORT_SHARED_DLL}")
        add_custom_COMMAND(TARGET pcnsf_benchmark POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy ${ORT_SHARED_DLL} $<TARGET_FILE_DIR:pcnsf_benchmark>)
    endif()
    if(APPLE AND EXISTS "${ORT_LIB}")
        add_custom_COMMAND(TARGET pcnsf_benchmark POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy ${ORT_LIB} $<TARGET_FILE_DIR:pcnsf_benchmark>)
    endif()
    if(UNIX AND NOT APPLE AND EXISTS "${ORT_LIB}")
        add_custom_COMMAND(TARGET pcnsf_benchmark POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy ${ORT_LIB} $<TARGET_FILE_DIR:pcnsf_benchmark>)
    endif()

    # Copy models next to executable if present
    set(PCNSF_BENCH_MODEL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/tools/benchmark/models")
    foreach(m IN ITEMS hifigan_f32.gguf nsf_hifigan.onnx)
        if(EXISTS "${PCNSF_BENCH_MODEL_DIR}/${m}")
            add_custom_COMMAND(TARGET pcnsf_benchmark POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy
                    "${PCNSF_BENCH_MODEL_DIR}/${m}"
                    $<TARGET_FILE_DIR:pcnsf_benchmark>)
        endif()
    endforeach()
endif()

# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
if(PCNSF_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# -----------------------------------------------------------------------------
# Examples
# -----------------------------------------------------------------------------
if(PCNSF_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# -----------------------------------------------------------------------------
# Install
# -----------------------------------------------------------------------------
install(DIRECTORY include/pc_nsf_hifigan
        DESTINATION include
        FILES_MATCHING PATTERN "*.h")

install(TARGETS pc_nsf_hifigan
        EXPORT pc_nsf_hifigan-targets
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)

message(STATUS "")
message(STATUS "pc_nsf_hifigan ${PROJECT_VERSION} configuration:")
message(STATUS "  Metal .............. ${PCNSF_METAL}")
message(STATUS "  CUDA ............... ${PCNSF_CUDA}")
message(STATUS "  Vulkan ............. ${PCNSF_VULKAN}")
message(STATUS "  Build CLI .......... ${PCNSF_BUILD_CLI}")
message(STATUS "  Build Tests ........ ${PCNSF_BUILD_TESTS}")
message(STATUS "  Benchmark .......... ${PCNSF_BUILD_BENCHMARK}")
message(STATUS "")
