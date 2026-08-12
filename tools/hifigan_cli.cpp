// hifigan_cli: vocode mel + f0 into a mono waveform.
//
//   hifigan_cli <gguf> <mel.bin> <f0.bin> <out.wav>
//   hifigan_cli --batch <list.txt> <outdir> [--warmup N]   (per-line: gguf mel f0)
//
// mel.bin: row-major float32 [T, num_mels];  f0.bin: float32 [T] in Hz.
// Two frames per line for --batch: mel then f0.

#include "pc_nsf_hifigan/hifigan.h"
#include "pc_nsf_hifigan/mel.h"

#define DR_WAV_IMPLEMENTATION
#include <dr_wav.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static bool read_raw(const std::string & path, std::vector<float> & out) {
    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END);
    long sz = std::ftell(f);
    if (sz <= 0 || sz % 4) return false;
    std::fseek(f, 0, SEEK_SET);
    out.resize(sz / 4);
    size_t got = std::fread(out.data(), 4, out.size(), f);
    std::fclose(f);
    return got == out.size();
}

static bool write_wav(const std::string & path, const float * data, size_t n,
                      int sample_rate) {
    drwav w;
    if (!drwav_init_file_write(&w, path.c_str(), nullptr, nullptr)) return false;
    drwav_uint64 written = drwav_write_pcm_frames(&w, (drwav_uint64) n, data);
    drwav_uninit(&w);
    return written == n;
}

static int vocode(const char * gguf, const char * melp, const char * f0p, const char * outp) {
    pc_nsf_hifigan::HifiganModel m(gguf, 4);
    std::vector<float> mel, f0;
    if (!read_raw(melp, mel) || !read_raw(f0p, f0)) return 1;
    const int T = (int) f0.size();
    if ((int) (mel.size() / (size_t) m.num_mels) != T) {
        std::fprintf(stderr, "mel frames %zu != f0 frames %d\n",
                     mel.size() / (size_t) m.num_mels, T);
        return 1;
    }
    std::vector<float> wav;
    pc_nsf_hifigan::hifigan_run(m, mel.data(), f0.data(), T, wav);
    write_wav(outp, wav.data(), wav.size(), m.sampling_rate);
    std::printf("wrote %s : %zu samples (%.2f s @ %d Hz)\n", outp,
                wav.size(), (double) wav.size() / m.sampling_rate, m.sampling_rate);
    return 0;
}

static int run_batch(const char * gguf, const char * list, const char * outdir,
                     int warmup) {
    pc_nsf_hifigan::HifiganModel m(gguf, 4);
    std::ifstream in(list);
    std::string line;
    int idx = 0;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::string melp, f0p, name;
        char a[4096], b[4096], c[4096];
        if (std::sscanf(line.c_str(), "%4095s %4095s %4095s", a, b, c) != 3)
            continue;
        melp = a; f0p = b; name = c;
        std::vector<float> mel, f0;
        if (!read_raw(melp, mel) || !read_raw(f0p, f0)) {
            std::fprintf(stderr, "skip %s (cannot read inputs)\n", name.c_str());
            continue;
        }
        const int T = (int) f0.size();
        std::vector<float> wav;
        pc_nsf_hifigan::hifigan_run(m, mel.data(), f0.data(), T, wav);
        write_wav(outdir + ("/" + name + ".wav"), wav.data(), wav.size(), m.sampling_rate);
        ++idx;
    }
    std::printf("batch done: %d clips -> %s\n", idx, outdir);
    return 0;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
            "usage: hifigan_cli <gguf> <mel.bin> <f0.bin> <out.wav>\n"
            "   or  hifigan_cli --batch <gguf> <list.txt> <outdir> [--warmup N]\n");
        return 1;
    }
    if (std::strcmp(argv[1], "--batch") == 0) {
        int warmup = 0;
        const char * gguf = argv[2], * list = argv[3], * outdir = argv[4];
        if (argc >= 7 && std::strcmp(argv[5], "--warmup") == 0) warmup = std::atoi(argv[6]);
        while (warmup-- > 0) { std::ifstream f(list); std::string l; std::getline(f, l); }
        return run_batch(gguf, list, outdir, warmup);
    }
    return vocode(argv[1], argv[2], argv[3], argv[4]);
}
