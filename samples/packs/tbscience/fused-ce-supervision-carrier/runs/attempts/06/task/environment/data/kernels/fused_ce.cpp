// Fused single-pass smoothed token cross-entropy.
//
// A per-position term is formed only at the non-ignored positions: the
// label-smoothed log-likelihood of the target gathered at that position.
// Under the mean reduction the sum of those terms is divided by their count,
// so a record with no non-ignored position reduces 0/0 and comes back
// non-finite.
//
// Built with -O2 -fno-fast-math: the double arithmetic is plain IEEE-754.

#include <cmath>
#include <cstddef>
#include <cstdint>

extern "C" double fused_cross_entropy(const double *logits,
                                      const std::int64_t *targets,
                                      int n, int v,
                                      double label_smoothing,
                                      int reduction /* 0 = mean, 1 = sum */) {
    double total = 0.0;
    long long count = 0;

    for (int i = 0; i < n; ++i) {
        const std::int64_t y = targets[i];
        if (y < 0) continue;  // ignore_index

        const double *row = logits + static_cast<std::size_t>(i) * static_cast<std::size_t>(v);

        double m = row[0];
        for (int k = 1; k < v; ++k) {
            if (row[k] > m) m = row[k];
        }
        double s = 0.0;
        for (int k = 0; k < v; ++k) {
            s += std::exp(row[k] - m);
        }
        const double lse = m + std::log(s);

        double term = 0.0;
        for (int k = 0; k < v; ++k) {
            term -= label_smoothing * (row[k] - lse) / static_cast<double>(v);
        }
        term -= (1.0 - label_smoothing) * (row[y] - lse);

        total += term;
        ++count;
    }

    if (reduction == 1) return total;
    return total / static_cast<double>(count);
}
