// SPDX-License-Identifier: MPL-2.0
// Minimal third-party integration example for pc-nsf-hifigan.cpp.
//
// Loads a hifigan GGUF, reads mel + f0 .bin files (mel row-major [T,128],
// f0 [T] in Hz), vocodes and writes a 44.1k mono WAV via the library.
//
// This mirrors exactly what a downstream vocoder bridge (e.g. DiffSinger-
// style C++ pipeline) would do; keeps WAV I/O trivial to stay dependency-free.

#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <pc_nsf_hifigan/hifigan.h>

static std::vector<float> read_raw(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    f.seekg(0, std::ios::end);
    long sz = (long) f.tellg();
    f.seekg(0, std::ios::beg);
    if (sz <= 0 || sz % 4) throw std::runtime_error("expected float32 raw: " + path);
    std::vector<float> out(sz / 4);
    f.read(reinterpret_cast<char*>(out.data()), sz);
    return out;
}

static void write_wav16(const std::string & path, const std::vector<float> & wav,
                        int sr) {
    // Minimal PCM-16 WAV writer (RIFF). Good enough for the demo.
    FILE * w = std::fopen(path.c_str(), "wb");
    if (!w) throw std::runtime_error("cannot open " + path);
    auto u32 = [&](uint32_t v) { std::fwrite(&v, 4, 1, w); };
    auto u16 = [&](uint16_t v) { std::fwrite(&v, 2, 1, w); };
    std::fwrite("RIFF", 1, 4, w);
    u32((uint32_t) (36 + wav.size() * 2));
    std::fwrite("WAVE", 1, 4, w);
    std::fwrite("fmt ", 1, 4, w);
    u32(16); u16(1); u16(1); u32((uint32_t) sr); u32((uint32_t) sr * 2);
    u16(2); u16(16);
    std::fwrite("data", 1, 4, w);
    u32((uint32_t) (wav.size() * 2));
    for (float v : wav) { int16_t s = (int16_t)(v * 32767.0f); std::fwrite(&s, 2, 1, w); }
    std::fclose(w);
}

int main(int argc, char ** argv) {
    if (argc < 5) {
        std::fprintf(stderr, "usage: %s <hifigan.gguf> <mel.bin> <f0.bin> <out.wav>\n", argv[0]);
        return 1;
    }
    try {
        pc_nsf_hifigan::HifiganModel m(argv[1], 4);
        auto mel = read_raw(argv[2]);
        auto f0  = read_raw(argv[3]);
        if ((int) f0.size() == 0 || (int) (mel.size() / (size_t) m.num_mels) != (int) f0.size())
            throw std::runtime_error("T mismatch: mel vs f0");
        std::vector<float> wav;
        pc_nsf_hifigan::hifigan_run(m, mel.data(), f0.data(), (int) f0.size(), wav);
        write_wav16(argv[4], wav, m.sampling_rate);
        std::printf("wrote %s (%d samples, %.2fs @ %d Hz)\n", argv[4],
                    (int) wav.size(), (double) wav.size() / m.sampling_rate,
                    m.sampling_rate);
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
}
