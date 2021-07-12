#include "compare.hpp"
#include "../MockCSVM.hpp"
#include "plssvm/typedef.hpp"

#include <string>
#include <vector>

std::vector<plssvm::real_t> generate_q(const std::string path) {
    std::vector<plssvm::real_t> q;
    MockCSVM csvm;
    csvm.libsvmParser(path);

    q.reserve(csvm.data.size());
    for (int i = 0; i < csvm.data.size() - 1; ++i) {
        q.emplace_back(csvm.kernel_function(csvm.data.back(), csvm.data[i]));
    }
    return q;
}

real_t linear_kernel(const std::vector<plssvm::real_t> &x1, const std::vector<plssvm::real_t> &x2) {
    assert(x1.size() == x2.size());
    plssvm::real_t result = static_cast<plssvm::real_t>(0.0);
    for (size_t i = 0; i < x1.size(); ++i) {
        result += x1[i] * x2[i];
    }
    return result;
}

std::vector<plssvm::real_t> kernel_linear_function(const std::vector<std::vector<plssvm::real_t>> &data, std::vector<plssvm::real_t> &x, const std::vector<plssvm::real_t> &q, const plssvm::real_t sgn, const plssvm::real_t QA_cost, const plssvm::real_t cost) {
    assert(x.size() == q.size());
    assert(x.size() == data.size() - 1);

    const size_t dept = x.size();

    std::vector<plssvm::real_t> r(dept, 0.0);

    for (int i = 0; i < dept; ++i) {
        for (int j = 0; j < dept; ++j) {
            if (i >= j) {
                real_t temp = linear_kernel(data[i], data[j]) - q[i] - q[j] + QA_cost;
                if (i == j) {
                    r[i] += (temp + 1 / cost) * x[i] * sgn;
                } else {
                    r[i] += (temp) *x[j] * sgn;
                    r[j] += (temp) *x[i] * sgn;
                }
            }
        }
    }
    return r;
}