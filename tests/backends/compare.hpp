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

template <plssvm::kernel_type kernel_type, typename real_type, typename... Args>
std::vector<real_type> generate_q(const std::vector<std::vector<real_type>> &data, Args &&...args) {
    std::vector<real_type> result;
    result.reserve(data.size());

    for (std::size_t i = 0; i < data.size() - 1; ++i) {
        if constexpr (kernel_type == plssvm::kernel_type::linear) {
            result.template emplace_back(linear_kernel(data.back(), data[i], std::forward<Args>(args)...));
        } else if constexpr (kernel_type == plssvm::kernel_type::polynomial) {
            result.template emplace_back(poly_kernel(data.back(), data[i], std::forward<Args>(args)...));
        } else if constexpr (kernel_type == plssvm::kernel_type::rbf) {
            result.template emplace_back(radial_kernel(data.back(), data[i], std::forward<Args>(args)...));
        } else {
            static_assert(plssvm::detail::always_false_v<real_type>, "Unknown kernel type!");
        }
    }
    return result;
}

template <typename real_type>
std::vector<real_type> kernel_linear_function(const std::vector<std::vector<real_type>> &data, std::vector<real_type> &x, const std::vector<real_type> &q, real_type QA_cost, real_type cost, int sgn);
template <typename real_type>
std::vector<real_type> kernel_polynomial_function(const std::vector<std::vector<real_type>> &data, std::vector<real_type> &x, const std::vector<real_type> &q, real_type QA_cost, real_type cost, int sgn, real_type degree, real_type gamma, real_type coef0);
template <typename real_type>
std::vector<real_type> kernel_radial_function(const std::vector<std::vector<real_type>> &data, std::vector<real_type> &x, const std::vector<real_type> &q, real_type QA_cost, real_type cost, int sgn, real_type gamma);

#endif /* TESTS_BACKENDS_COMPARE */
