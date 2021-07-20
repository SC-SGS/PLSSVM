#ifndef TESTS_BACKENDS_COMPARE
#define TESTS_BACKENDS_COMPARE

#include "plssvm/detail/utility.hpp"  // plssvm::detail::always_false_v
#include "plssvm/kernel_types.hpp"    // plssvm::kernel_type

#include <string>  // std::string
#include <vector>  // std::vector

template <typename real_type>
real_type linear_kernel(const std::vector<real_type> &x1, const std::vector<real_type> &x2);
template <typename real_type>
real_type poly_kernel(const std::vector<real_type> &x1, const std::vector<real_type> &x2, real_type degree, real_type gamma, real_type coef0);
template <typename real_type>
real_type radial_kernel(const std::vector<real_type> &x1, const std::vector<real_type> &x2, real_type gamma);

template <typename real_type>
std::vector<real_type> generate_q(const std::string &path);

template <plssvm::kernel_type kernel_type, typename real_type>
std::vector<real_type> q(const std::vector<std::vector<real_type>> &data) {
    std::vector<real_type> result;

    result.reserve(data.size());
    for (std::size_t i = 0; i < data.size() - 1; ++i) {
        // TODO: Other Kernels
        if constexpr (kernel_type == plssvm::kernel_type::linear) {
            result.emplace_back(linear_kernel(data.back(), data[i]));
        } else if constexpr (kernel_type == plssvm::kernel_type::polynomial) {
            static_assert(plssvm::detail::always_false_v<real_type>, "Tests for the polynomial kernel currently not available!");
        } else if constexpr (kernel_type == plssvm::kernel_type::rbf) {
            static_assert(plssvm::detail::always_false_v<real_type>, "Tests for the radial basis function kernel currently not available!");
        } else {
            static_assert(plssvm::detail::always_false_v<real_type>, "Unknown kernel type!");
        }
    }
    return result;
}

template <typename real_type>
std::vector<real_type> kernel_linear_function(const std::vector<std::vector<real_type>> &data, std::vector<real_type> &x, const std::vector<real_type> &q, real_type sgn, real_type QA_cost, real_type cost);

#endif /* TESTS_BACKENDS_COMPARE */
