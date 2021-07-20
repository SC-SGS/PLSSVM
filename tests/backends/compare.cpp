/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright
 */

#include "compare.hpp"

#include <cassert>  // assert
#include <cmath>    // std::pow, std::exp
#include <vector>   // std::vector

namespace compare::detail {

template <typename real_type>
real_type linear_kernel(const std::vector<real_type> &x1, const std::vector<real_type> &x2) {
    assert((x1.size() == x2.size()) && "Sizes mismatch!: x1.size() != x2.size()");

    real_type result{ 0.0 };
    for (std::size_t i = 0; i < x1.size(); ++i) {
        result += x1[i] * x2[i];
    }
    return result;
}
template float linear_kernel(const std::vector<float> &, const std::vector<float> &);
template double linear_kernel(const std::vector<double> &, const std::vector<double> &);

template <typename real_type>
real_type poly_kernel(const std::vector<real_type> &x1, const std::vector<real_type> &x2, const real_type degree, const real_type gamma, const real_type coef0) {
    assert((x1.size() == x2.size()) && "Sizes mismatch!: x1.size() != x2.size()");

    real_type result{ 0.0 };
    for (std::size_t i = 0; i < x1.size(); ++i) {
        result += x1[i] * x2[i];
    }
    return std::pow(gamma * result + coef0, degree);
}
template float poly_kernel(const std::vector<float> &, const std::vector<float> &, const float, const float, const float);
template double poly_kernel(const std::vector<double> &, const std::vector<double> &, const double, const double, const double);

template <typename real_type>
real_type radial_kernel(const std::vector<real_type> &x1, const std::vector<real_type> &x2, const real_type gamma) {
    assert((x1.size() == x2.size()) && "Sizes mismatch!: x1.size() != x2.size()");

    real_type result{ 0.0 };
    for (std::size_t i = 0; i < x1.size(); ++i) {
        result += (x1[i] - x2[i]) * (x1[i] - x2[i]);
    }
    return std::exp(-gamma * result);
}
template float radial_kernel(const std::vector<float> &, const std::vector<float> &, const float);
template double radial_kernel(const std::vector<double> &, const std::vector<double> &, const double);

}  // namespace compare::detail
