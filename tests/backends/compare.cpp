#include "compare.hpp"

#include "../MockCSVM.hpp"  // MockCSVM

#include <cassert>  // assert
#include <cmath>    // std::pow, std::exp
#include <string>   // std::string
#include <vector>   // std::vector

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

template <typename real_type>
std::vector<real_type> kernel_linear_function(const std::vector<std::vector<real_type>> &data, std::vector<real_type> &x, const std::vector<real_type> &q, const real_type QA_cost, const real_type cost, const int sgn) {
    assert((x.size() == q.size()) && "Sizes mismatch!");
    assert((x.size() == data.size() - 1) && "Sizes mismatch!");

    const std::size_t dept = x.size();

    std::vector<real_type> r(dept, 0.0);
    for (std::size_t i = 0; i < dept; ++i) {
        for (std::size_t j = 0; j < dept; ++j) {
            if (i >= j) {
                const real_type temp = linear_kernel(data[i], data[j]) + QA_cost - q[i] - q[j];
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
template std::vector<float> kernel_linear_function(const std::vector<std::vector<float>> &, std::vector<float> &, const std::vector<float> &, const float, const float, const int);
template std::vector<double> kernel_linear_function(const std::vector<std::vector<double>> &, std::vector<double> &, const std::vector<double> &, const double, const double, const int);

template <typename real_type>
std::vector<real_type> kernel_polynomial_function(const std::vector<std::vector<real_type>> &data, std::vector<real_type> &x, const std::vector<real_type> &q, const real_type QA_cost, const real_type cost, const int sgn, const real_type degree, const real_type gamma, const real_type coef0) {
    assert((x.size() == q.size()) && "Sizes mismatch!");
    assert((x.size() == data.size() - 1) && "Sizes mismatch!");

    const std::size_t dept = x.size();

    std::vector<real_type> r(dept, 0.0);
    for (std::size_t i = 0; i < dept; ++i) {
        for (std::size_t j = 0; j < dept; ++j) {
            if (i >= j) {
                const real_type temp = poly_kernel(data[i], data[j], degree, gamma, coef0) + QA_cost - q[i] - q[j];
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
template std::vector<float> kernel_polynomial_function(const std::vector<std::vector<float>> &, std::vector<float> &, const std::vector<float> &, const float, const float, const int, const float, const float, const float);
template std::vector<double> kernel_polynomial_function(const std::vector<std::vector<double>> &, std::vector<double> &, const std::vector<double> &, const double, const double, const int, const double, const double, const double);

template <typename real_type>
std::vector<real_type> kernel_radial_function(const std::vector<std::vector<real_type>> &data, std::vector<real_type> &x, const std::vector<real_type> &q, const real_type QA_cost, const real_type cost, const int sgn, const real_type gamma) {
    assert((x.size() == q.size()) && "Sizes mismatch!");
    assert((x.size() == data.size() - 1) && "Sizes mismatch!");

    const std::size_t dept = x.size();

    std::vector<real_type> r(dept, 0.0);
    for (std::size_t i = 0; i < dept; ++i) {
        for (std::size_t j = 0; j < dept; ++j) {
            if (i >= j) {
                const real_type temp = radial_kernel(data[i], data[j], gamma) + QA_cost - q[i] - q[j];
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
template std::vector<float> kernel_radial_function(const std::vector<std::vector<float>> &, std::vector<float> &, const std::vector<float> &, const float, const float, const int, const float);
template std::vector<double> kernel_radial_function(const std::vector<std::vector<double>> &, std::vector<double> &, const std::vector<double> &, const double, const double, const int, const double);
