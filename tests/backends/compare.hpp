#ifndef TESTS_BACKENDS_COMPARE
#define TESTS_BACKENDS_COMPARE
#include "plssvm/kernel_types.hpp"
#include "plssvm/typedef.hpp"

#include <string>
#include <vector>

std::vector<plssvm::real_t> generate_q(const std::string path);
plssvm::real_t linear_kernel(const std::vector<plssvm::real_t> &x1, const std::vector<plssvm::real_t> &x2);

template <plssvm::kernel_type kernel_type>
std::vector<plssvm::real_t> q(const std::vector<std::vector<plssvm::real_t>> &data) {
    std::vector<plssvm::real_t> result;

    result.reserve(data.size());
    for (int i = 0; i < data.size() - 1; ++i) {
        if (kernel_type == plssvm::kernel_type::linear) {
            result.emplace_back(linear_kernel(data.back(), data[i]));
        } else {
            //TODO: Other Kernels
        }
    }
    return result;
}

std::vector<plssvm::real_t> kernel_linear_function(const std::vector<std::vector<plssvm::real_t>> &data, std::vector<plssvm::real_t> &x, const std::vector<plssvm::real_t> &q, const plssvm::real_t sgn, const plssvm::real_t QA_cost, const plssvm::real_t cost);
#endif /* TESTS_BACKENDS_COMPARE */
