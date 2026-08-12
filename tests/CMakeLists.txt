# -----------------------------------------------------------------------------
# Tests: golden-file comparison.  gen_golden.py (numpy reference) produces
# goldens under ${CMAKE_BINARY_DIR}/tests/golden once at CTest time, then the
# C++ tests compare against them.  No torch / model asset required.
# -----------------------------------------------------------------------------
find_package(Python3 COMPONENTS Interpreter)

if(NOT Python3_Interpreter_FOUND)
    message(WARNING "Python3 not found — golden generation disabled; tests will be skipped")
    return()
endif()

set(PCNSF_GOLDEN_DIR "${CMAKE_BINARY_DIR}/tests/golden")
file(MAKE_DIRECTORY "${PCNSF_GOLDEN_DIR}")

# --- golden generator fixture ----------------------------------------------
add_test(
    NAME gen_golden
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/gen_golden.py
            --out ${PCNSF_GOLDEN_DIR}
)
set_tests_properties(gen_golden PROPERTIES
    FIXTURES_SETUP golden
    LABELS "gen"
)

# --- C++ tests (each fixture-required golden) --------------------------------
function(pcnsf_add_test name binary runner)
    add_executable(${binary} ${runner})
    target_link_libraries(${binary} PRIVATE pc_nsf_hifigan::pc_nsf_hifigan mininsf::mininsf)
    target_compile_options(${binary} PRIVATE ${PCNSF_WARN_FLAGS})
    target_compile_definitions(${binary} PRIVATE PCNSF_GOLDEN_DIR="${PCNSF_GOLDEN_DIR}")
    add_test(NAME ${name} COMMAND ${binary})
    set_tests_properties(${name} PROPERTIES FIXTURES_REQUIRED golden LABELS "core")
endfunction()

pcnsf_add_test(t01_mel_ng      test_mel_ng        test_mel.cpp)
pcnsf_add_test(t02_mel_nvstft  test_mel_nvstft    test_mel_nvstft.cpp)
pcnsf_add_test(t03_source      test_source        test_source.cpp)
pcnsf_add_test(t04_gguf_meta   test_gguf_meta     test_gguf_meta.cpp)

# optional end-to-end vocode test driven by PCNSF_MODEL_GGUF
if(DEFINED ENV{PCNSF_MODEL_GGUF})
    add_executable(test_vocode tests/test_vocode.cpp)
    target_link_libraries(test_vocode PRIVATE pc_nsf_hifigan::pc_nsf_hifigan)
    target_compile_definitions(test_vocode PRIVATE
        PCNSF_MODEL_GGUF="${PCNSF_MODEL_GGUF}" PCNSF_GOLDEN_DIR="${PCNSF_GOLDEN_DIR}")
    add_test(NAME t05_vocode COMMAND test_vocode)
    set_tests_properties(t05_vocode PROPERTIES FIXTURES_REQUIRED golden LABELS "e2e")
endif()
