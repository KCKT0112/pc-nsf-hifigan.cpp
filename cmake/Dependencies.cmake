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

# ggml (MIT) — tensor engine.  Pinned to v0.19.0 (matches game_ggml_cli).
# D2-revised (2026-08-30): this repo's vocoder body now *consumes* APIs added
# by KakaruHayate/ggml-audio-patch (ggml_conv_direct_1d / *_fused /
# ggml_add_leaky_relu).  "No patches applied here" was true when hifigan.cpp
# was I/O-only; it is false now.  We vendor a byte-identical snapshot of the
# 4 shipped patches into ./patches/ and have FetchContent apply them
# idempotently on first populate.  Keeping them as files (not a ggml fork)
# preserves the D2 intent: ggml remains stock upstream, the diff lives here.
#
# Idempotency: FetchContent PATCH_COMMAND runs only on first populate; second
# configure of the same build dir skips it (ggml-subbuild stamp).  The
# `git apply --check || git apply -R --check` pattern tolerates the corner
# case where the build dir keeps its _deps but the stamp was wiped.
set(_pcnsf_ggml_patch_dir "${CMAKE_CURRENT_SOURCE_DIR}/patches")
set(_pcnsf_ggml_patch_1 "${_pcnsf_ggml_patch_dir}/learned-ops-ggml0190.patch")
set(_pcnsf_ggml_patch_2 "${_pcnsf_ggml_patch_dir}/qvac-ops-ggml0190.patch")
set(_pcnsf_ggml_patch_3 "${_pcnsf_ggml_patch_dir}/metal-ops-ggml0190.patch")
set(_pcnsf_ggml_patch_4 "${_pcnsf_ggml_patch_dir}/vulkan-conv-direct-1d-ggml0190.patch")
foreach(_p IN ITEMS "${_pcnsf_ggml_patch_1}" "${_pcnsf_ggml_patch_2}" "${_pcnsf_ggml_patch_3}" "${_pcnsf_ggml_patch_4}")
    if(NOT EXISTS "${_p}")
        message(FATAL_ERROR "ggml patch snapshot missing: ${_p} — regenerate from KakaruHayate/ggml-audio-patch @ 55c9389")
    endif()
endforeach()

FetchContent_Declare(
    ggml
    GIT_REPOSITORY https://github.com/ggerganov/ggml.git
    GIT_TAG        v0.19.0
    GIT_SHALLOW    TRUE
    # NOTE: pass each patch as its own -D.  A ;-separated list inside one -D
    # would be re-split when ExternalProject materialises PATCH_COMMAND into
    # its subbuild script, silently dropping patches 2..4.  Enumerated vars
    # are immune to that.
    PATCH_COMMAND  ${CMAKE_COMMAND}
                   -DGGML_SOURCE_DIR=<SOURCE_DIR>
                   "-DGGML_PATCH_1=${_pcnsf_ggml_patch_1}"
                   "-DGGML_PATCH_2=${_pcnsf_ggml_patch_2}"
                   "-DGGML_PATCH_3=${_pcnsf_ggml_patch_3}"
                   "-DGGML_PATCH_4=${_pcnsf_ggml_patch_4}"
                   -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/ApplyGgmlPatches.cmake"
)
FetchContent_MakeAvailable(ggml)
unset(_pcnsf_ggml_patch_dir)
unset(_pcnsf_ggml_patch_1)
unset(_pcnsf_ggml_patch_2)
unset(_pcnsf_ggml_patch_3)
unset(_pcnsf_ggml_patch_4)

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
