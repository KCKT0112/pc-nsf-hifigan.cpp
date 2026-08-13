# ----------------------------------------------------------------------------
# External dependencies — all via FetchContent, nothing vendored
# (except pocketfft_hdronly.h which the original audio_ggml vendored as a
# single header and is kept for reproducibility).
# ----------------------------------------------------------------------------
include(FetchContent)

# Don't let child projects surprise us with their own versions.
set(FETCHCONTENT_UPDATES_DISCONNECTED ON CACHE BOOL "" FORCE)

# How libmininsf gets consumed:
#   D1 decision: pc-nsf-hifigan.cpp references KakaruHayate/libmininsf for the
#   mini-nsf sine source; the ggml vocoder body is this repo.

# Propagate backend toggles as ggml's own option names *before* add_subdirectory.
set(GGML_BUILD_TESTS     OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES  OFF CACHE BOOL "" FORCE)
set(GGML_METAL           ${PCNSF_METAL}  CACHE BOOL "ggml: enable Metal"  FORCE)
set(GGML_CUDA            ${PCNSF_CUDA}   CACHE BOOL "ggml: enable CUDA"   FORCE)
set(GGML_VULKAN          ${PCNSF_VULKAN} CACHE BOOL "ggml: enable Vulkan" FORCE)

if(APPLE AND PCNSF_METAL)
    set(GGML_METAL_EMBED_LIBRARY OFF CACHE BOOL "ggml: embed Metal library" FORCE)
endif()

# ggml (MIT) — tensor engine.  Pinned to v0.11.0 (matches game_ggml_cli).
# Per D2 we do NOT maintain a fork; we track ggml main and only carry .patch
# files in KakaruHayate/ggml-patch.  No patches are applied here (CPU/F16
# path needs none; the CUDA conv_transpose_1d local window / im2col wide OW
# patches live in ggml-patch and are applied by consumers that enable CUDA).
FetchContent_Declare(
    ggml
    GIT_REPOSITORY https://github.com/ggerganov/ggml.git
    GIT_TAG        v0.11.0
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(ggml)

# libmininsf (MIT) — mini-nsf sine source generator.
FetchContent_Declare(
    mininsf
    GIT_REPOSITORY https://github.com/KakaruHayate/libmininsf.git
    GIT_TAG        main
    GIT_SHALLOW    TRUE
)
set(MININSF_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(MININSF_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(mininsf)

# pocketfft (BSD-3-Clause) — header-only STFT r2c FFT used by mel.cpp.
# Idempotent: when this repo is aggregated (hifisampler/hachitune), the
# parent may already define the same helper target.
FetchContent_Declare(
    pocketfft
    GIT_REPOSITORY https://github.com/mreineck/pocketfft.git
    GIT_TAG        32424d2067c2e8043dc646a4e49754b2b40cc549   # cpp @ 2025-10
)
FetchContent_MakeAvailable(pocketfft)
if(NOT TARGET pocketfft)
    add_library(pocketfft INTERFACE)
    target_include_directories(pocketfft SYSTEM INTERFACE "${pocketfft_SOURCE_DIR}")
endif()

# dr_libs (Public Domain/MIT-0) — single-header WAV writer used by the CLI.
FetchContent_Declare(
    dr_libs
    GIT_REPOSITORY https://github.com/mackron/dr_libs.git
    GIT_TAG        243e26ffa08a24dc8ae2e7a8c57123d9e504690c   # master @ 2025-10
)
FetchContent_MakeAvailable(dr_libs)
if(NOT TARGET dr_wav)
    add_library(dr_wav INTERFACE)
    target_include_directories(dr_wav SYSTEM INTERFACE "${dr_libs_SOURCE_DIR}")
endif()

message(STATUS "Third-party fetched:")
message(STATUS "  ggml       ${ggml_SOURCE_DIR}")
message(STATUS "  mininsf    ${mininsf_SOURCE_DIR}")
message(STATUS "  pocketfft  ${pocketfft_SOURCE_DIR}")
message(STATUS "  dr_libs    ${dr_libs_SOURCE_DIR}")
