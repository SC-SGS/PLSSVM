/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright
 */

#include "compare.hpp"

#include "plssvm/detail/assert.hpp"  // PLSSVM_ASSERT

#include <cmath>   // std::pow, std::exp, std::fma
#include <vector>  // std::vector

namespace compare::detail {

template <typename real_type>
real_type linear_kernel(const std::vector<real_type> &x1, const std::vector<real_type> &x2) {
    PLSSVM_ASSERT(x1.size() == x2.size(), "Sizes mismatch!: {} != {}", x1.size(), x2.size());

    real_type result{ 0.0 };
    for (std::size_t i = 0; i < x1.size(); ++i) {
        result = std::fma(x1[i], x2[i], result);  // TODO: enable auto fma
    }
    return result;
}
template float linear_kernel(const std::vector<float> &, const std::vector<float> &);
template double linear_kernel(const std::vector<double> &, const std::vector<double> &);

template <typename real_type>
real_type poly_kernel(const std::vector<real_type> &x1, const std::vector<real_type> &x2, const int degree, const real_type gamma, const real_type coef0) {
    PLSSVM_ASSERT(x1.size() == x2.size(), "Sizes mismatch!: {} != {}", x1.size(), x2.size());

    real_type result{ 0.0 };
    for (std::size_t i = 0; i < x1.size(); ++i) {
        // result += x1[i] * x2[i];
        result = std::fma(x1[i], x2[i], result);  // TODO: enable auto fma
    }
    return std::pow(std::fma(gamma, result, coef0), static_cast<real_type>(degree));  // TODO: enable auto fma
}
template float poly_kernel(const std::vector<float> &, const std::vector<float> &, const int, const float, const float);
template double poly_kernel(const std::vector<double> &, const std::vector<double> &, const int, const double, const double);

template <typename real_type>
real_type radial_kernel(const std::vector<real_type> &x1, const std::vector<real_type> &x2, const real_type gamma) {
    PLSSVM_ASSERT(x1.size() == x2.size(), "Sizes mismatch!: {} != {}", x1.size(), x2.size());

    real_type result{ 0.0 };
    for (std::size_t i = 0; i < x1.size(); ++i) {
        result += (x1[i] - x2[i]) * (x1[i] - x2[i]);
    }
    return std::exp(-gamma * result);
}
template float radial_kernel(const std::vector<float> &, const std::vector<float> &, const float);
template double radial_kernel(const std::vector<double> &, const std::vector<double> &, const double);

}  // namespace compare::detail
