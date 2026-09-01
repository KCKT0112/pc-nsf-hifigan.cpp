// SPDX-License-Identifier: MPL-2.0
#include "accuracy.h"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace bench {

AccuracyReport compare_audio(const std::vector<float>& a,
                             const std::vector<float>& b,
                             const std::string& name_a,
                             const std::string& name_b) {
    AccuracyReport r{};
    r.length_a = a.size();
    r.length_b = b.size();

    size_t n = std::min(a.size(), b.size());
    if (n == 0) return r;

    double sum_a = 0.0;
    double sum_b = 0.0;
    double sum_aa = 0.0;
    double sum_bb = 0.0;
    double sum_ab = 0.0;
    double maxabs = 0.0;
    double sum_sq = 0.0;

    for (size_t i = 0; i < n; ++i) {
        double da = a[i];
        double db = b[i];
        double diff = da - db;
        double adiff = std::fabs(diff);
        if (adiff > maxabs) maxabs = adiff;
        sum_sq += diff * diff;

        sum_a += da;
        sum_b += db;
        sum_aa += da * da;
        sum_bb += db * db;
        sum_ab += da * db;
    }

    double mean_a = sum_a / n;
    double mean_b = sum_b / n;
    double cov_ab = sum_ab / n - mean_a * mean_b;
    double var_a = sum_aa / n - mean_a * mean_a;
    double var_b = sum_bb / n - mean_b * mean_b;

    double denom = std::sqrt(var_a * var_b);
    if (denom > 1e-20) {
        r.corr = cov_ab / denom;
    } else {
        r.corr = 1.0;
    }

    r.maxabs = maxabs;
    r.rms = std::sqrt(sum_sq / n);

    double signal = var_a + var_b;
    if (r.rms > 1e-20) {
        r.snr_db = 10.0 * std::log10(std::max(1e-30, signal / 2.0) / (r.rms * r.rms));
    }
    return r;
}

std::string accuracy_to_string(const AccuracyReport& r) {
    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss << "Lengths: " << r.length_a << " / " << r.length_b << "\n";
    oss << "Correlation: " << std::setprecision(9) << r.corr << "\n";
    oss << "Max |Δ|:     " << std::setprecision(9) << r.maxabs << "\n";
    oss << "RMS Δ:      " << std::setprecision(9) << r.rms << "\n";
    oss << "SNR (dB):   " << std::setprecision(3) << r.snr_db << "\n";
    return oss.str();
}

} // namespace bench
