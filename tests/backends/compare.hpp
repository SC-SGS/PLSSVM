/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright
 *
 * @brief Functions used for testing the correctness of the PLSSVM implementation.
 */

#pragma once

#include "plssvm/detail/utility.hpp"  // plssvm::detail::always_false_v
#include "plssvm/kernel_types.hpp"    // plssvm::kernel_type

#include <cstddef>  // std::size_t
#include <utility>  // std::forward
#include <vector>   // std::vector

namespace compare {
namespace detail {

template <typename real_type>
real_type linear_kernel(const std::vector<real_type> &x1, const std::vector<real_type> &x2);
template <typename real_type>
real_type poly_kernel(const std::vector<real_type> &x1, const std::vector<real_type> &x2, real_type degree, real_type gamma, real_type coef0);
template <typename real_type>
real_type radial_kernel(const std::vector<real_type> &x1, const std::vector<real_type> &x2, real_type gamma);

}  // namespace detail

template <plssvm::kernel_type kernel, typename real_type, typename... Args>
real_type kernel_function(const std::vector<real_type> &x1, const std::vector<real_type> &x2, Args &&...args) {
    assert((x1.size() == x2.size()) && "Size mismatch!: x1.size() != x2.size()");

    if constexpr (kernel == plssvm::kernel_type::linear) {
        return detail::linear_kernel(x1, x2, std::forward<Args>(args)...);
    } else if constexpr (kernel == plssvm::kernel_type::polynomial) {
        return detail::poly_kernel(x1, x2, std::forward<Args>(args)...);
    } else if constexpr (kernel == plssvm::kernel_type::rbf) {
        return detail::radial_kernel(x1, x2, std::forward<Args>(args)...);
    } else {
        static_assert(plssvm::detail::always_false_v<real_type>, "Unknown kernel type!");
    }
}

template <plssvm::kernel_type kernel, typename real_type, typename... Args>
std::vector<real_type> generate_q(const std::vector<std::vector<real_type>> &data, Args &&...args) {
    std::vector<real_type> result;
    result.reserve(data.size());

    for (std::size_t i = 0; i < data.size() - 1; ++i) {
        result.template emplace_back(kernel_function<kernel>(data.back(), data[i], args...));
    }
    return result;
}

template <plssvm::kernel_type kernel, typename real_type, typename... Args>
std::vector<real_type> device_kernel_function(const std::vector<std::vector<real_type>> &data, std::vector<real_type> &x, const std::vector<real_type> &q, real_type QA_cost, real_type cost, int sgn, Args &&...args) {
    assert((x.size() == q.size()) && "Sizes mismatch!");
    assert((x.size() == data.size() - 1) && "Sizes mismatch!");

    const std::size_t dept = x.size();

    std::vector<real_type> r(dept, 0.0);
    for (std::size_t i = 0; i < dept; ++i) {
        for (std::size_t j = 0; j < dept; ++j) {
            if (i >= j) {
                const real_type temp = kernel_function<kernel>(data[i], data[j], args...) + QA_cost - q[i] - q[j];
                if (i == j) {
                    r[i] += (temp + 1.0 / cost) * x[i] * sgn;
                } else {
                    r[i] += temp * x[j] * sgn;
                    r[j] += temp * x[i] * sgn;
                }
            }
        }
    }
    return r;
}

}  // namespace compare
