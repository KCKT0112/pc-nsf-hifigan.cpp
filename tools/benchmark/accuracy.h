#pragma once

#include <string>
#include <vector>

namespace bench {

struct AccuracyReport {
    size_t length_a = 0;
    size_t length_b = 0;
    double corr = 0.0;
    double maxabs = 0.0;
    double rms = 0.0;
    double snr_db = 0.0;
};

AccuracyReport compare_audio(const std::vector<float>& a,
                             const std::vector<float>& b,
                             const std::string& name_a,
                             const std::string& name_b);

std::string accuracy_to_string(const AccuracyReport& r);

} // namespace bench
