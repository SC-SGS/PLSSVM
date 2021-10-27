/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "compare.hpp"

#include "plssvm/detail/assert.hpp"  // PLSSVM_ASSERT

#include <cmath>   // std::pow, std::exp, std::fma
#include <vector>  // std::vector

namespace compare::detail {

template <typename real_type>
real_type linear_kernel(const std::vector<real_type> &x, const std::vector<real_type> &y) {
    PLSSVM_ASSERT(x.size() == y.size(), "Sizes mismatch!: {} != {}", x.size(), y.size());

    real_type result{ 0.0 };
    for (typename std::vector<real_type>::size_type i = 0; i < x.size(); ++i) {
        result = std::fma(x[i], y[i], result);
    }
    return result;
}
template float linear_kernel(const std::vector<float> &, const std::vector<float> &);
template double linear_kernel(const std::vector<double> &, const std::vector<double> &);

template <typename real_type>
real_type poly_kernel(const std::vector<real_type> &x, const std::vector<real_type> &y, const int degree, const real_type gamma, const real_type coef0) {
    PLSSVM_ASSERT(x.size() == y.size(), "Sizes mismatch!: {} != {}", x.size(), y.size());

    real_type result{ 0.0 };
    for (typename std::vector<real_type>::size_type i = 0; i < x.size(); ++i) {
        result = std::fma(x[i], y[i], result);
    }
    return std::pow(std::fma(gamma, result, coef0), static_cast<real_type>(degree));
}
template float poly_kernel(const std::vector<float> &, const std::vector<float> &, const int, const float, const float);
template double poly_kernel(const std::vector<double> &, const std::vector<double> &, const int, const double, const double);

template <typename real_type>
real_type radial_kernel(const std::vector<real_type> &x, const std::vector<real_type> &y, const real_type gamma) {
    PLSSVM_ASSERT(x.size() == y.size(), "Sizes mismatch!: {} != {}", x.size(), y.size());

    real_type result{ 0.0 };
    for (typename std::vector<real_type>::size_type i = 0; i < x.size(); ++i) {
        const real_type diff = x[i] - y[i];
        result = std::fma(diff, diff, result);
    }
    return std::exp(-gamma * result);
}
template float radial_kernel(const std::vector<float> &, const std::vector<float> &, const float);
template double radial_kernel(const std::vector<double> &, const std::vector<double> &, const double);

}  // namespace compare::detail
