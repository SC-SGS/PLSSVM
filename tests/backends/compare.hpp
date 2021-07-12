#ifndef TESTS_BACKENDS_COMPARE
#define TESTS_BACKENDS_COMPARE
#include "plssvm/kernel_types.hpp"
#include "plssvm/typedef.hpp"

#include <string>
#include <vector>

template <typename real_type>
std::vector<real_type> generate_q(const std::string &path);
template <typename real_type>
real_type linear_kernel(const std::vector<real_type> &x1, const std::vector<real_type> &x2);

template <plssvm::kernel_type kernel_type, typename real_type>
std::vector<real_type> q(const std::vector<std::vector<real_type>> &data) {
    std::vector<real_type> result;

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

template <typename real_type>
std::vector<real_type> kernel_linear_function(const std::vector<std::vector<real_type>> &data, std::vector<real_type> &x, const std::vector<real_type> &q, int sgn, real_type QA_cost, real_type cost);

#endif /* TESTS_BACKENDS_COMPARE */
