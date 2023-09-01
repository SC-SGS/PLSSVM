/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "compare.hpp"

#include "plssvm/detail/assert.hpp"          // PLSSVM_ASSERT
#include "plssvm/kernel_function_types.hpp"  // plssvm::kernel_function_type
#include "plssvm/matrix.hpp"                 // plssvm::aos_matrix
#include "plssvm/parameter.hpp"              // plssvm::detail::parameter

#include <algorithm>                         // std::min
#include <cmath>                             // std::pow, std::exp, std::fma
#include <cstddef>                           // std::size_t
#include <vector>                            // std::vector

namespace compare {

namespace detail {

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
real_type linear_kernel(const std::vector<real_type> &x, const std::vector<real_type> &y, const std::size_t num_devices) {
    PLSSVM_ASSERT(x.size() == y.size(), "Sizes mismatch!: {} != {}", x.size(), y.size());
    PLSSVM_ASSERT(num_devices > 0, "At least one device must be available!");

    const std::size_t block_size = x.size() / num_devices;
    real_type result{ 0.0 };
    for (std::size_t d = 0; d < num_devices; ++d) {
        real_type tmp{ 0.0 };
        for (std::size_t i = d * block_size; i < std::min(x.size(), (d + 1) * block_size); ++i) {
            tmp = std::fma(x[i], y[i], tmp);
        }
        result += tmp;
    }
    return result;
}
template float linear_kernel(const std::vector<float> &, const std::vector<float> &, const std::size_t);
template double linear_kernel(const std::vector<double> &, const std::vector<double> &, const std::size_t);

template <typename real_type>
real_type polynomial_kernel(const std::vector<real_type> &x, const std::vector<real_type> &y, const int degree, const real_type gamma, const real_type coef0) {
    PLSSVM_ASSERT(x.size() == y.size(), "Sizes mismatch!: {} != {}", x.size(), y.size());

    real_type result{ 0.0 };
    for (typename std::vector<real_type>::size_type i = 0; i < x.size(); ++i) {
        result = std::fma(x[i], y[i], result);
    }
    return std::pow(std::fma(gamma, result, coef0), static_cast<real_type>(degree));
}
template float polynomial_kernel(const std::vector<float> &, const std::vector<float> &, int, float, float);
template double polynomial_kernel(const std::vector<double> &, const std::vector<double> &, int, double, double);

template <typename real_type>
real_type rbf_kernel(const std::vector<real_type> &x, const std::vector<real_type> &y, const real_type gamma) {
    PLSSVM_ASSERT(x.size() == y.size(), "Sizes mismatch!: {} != {}", x.size(), y.size());

    real_type result{ 0.0 };
    for (typename std::vector<real_type>::size_type i = 0; i < x.size(); ++i) {
        const real_type diff = x[i] - y[i];
        result = std::fma(diff, diff, result);
    }
    return std::exp(-gamma * result);
}
template float rbf_kernel(const std::vector<float> &, const std::vector<float> &, float);
template double rbf_kernel(const std::vector<double> &, const std::vector<double> &, double);


template <typename real_type, plssvm::layout_type layout>
real_type linear_kernel(const plssvm::matrix<real_type, layout> &X, const std::size_t i, const plssvm::matrix<real_type, layout> &Y, const std::size_t j, const std::size_t num_devices) {
    PLSSVM_ASSERT(X.num_cols() == Y.num_cols(), "Sizes mismatch!: {} != {}", X.num_cols(), Y.num_cols());
    PLSSVM_ASSERT(i < X.num_rows(), "Out-of-bounce access!: {} < {}", i, X.num_rows());
    PLSSVM_ASSERT(j < Y.num_rows(), "Out-of-bounce access!: {} < {}", j, Y.num_rows());
    PLSSVM_ASSERT(num_devices > 0, "At least one device must be available!");

    const std::size_t block_size = X.num_cols() / num_devices;
    real_type result{ 0.0 };
    for (std::size_t d = 0; d < num_devices; ++d) {
        real_type tmp{ 0.0 };
        for (std::size_t bd = d * block_size; bd < std::min(X.num_cols(), (d + 1) * block_size); ++bd) {
            tmp = std::fma(X(i, bd), Y(j, bd), tmp);
        }
        result += tmp;
    }
    return result;
}
template float linear_kernel(const plssvm::matrix<float, plssvm::layout_type::aos> &, const std::size_t, const plssvm::matrix<float, plssvm::layout_type::aos> &, const std::size_t, const std::size_t);
template float linear_kernel(const plssvm::matrix<float, plssvm::layout_type::soa> &, const std::size_t, const plssvm::matrix<float, plssvm::layout_type::soa> &, const std::size_t, const std::size_t);
template double linear_kernel(const plssvm::matrix<double, plssvm::layout_type::aos> &, const std::size_t, const plssvm::matrix<double, plssvm::layout_type::aos> &, const std::size_t, const std::size_t);
template double linear_kernel(const plssvm::matrix<double, plssvm::layout_type::soa> &, const std::size_t, const plssvm::matrix<double, plssvm::layout_type::soa> &, const std::size_t, const std::size_t);

template <typename real_type, plssvm::layout_type layout>
real_type polynomial_kernel(const plssvm::matrix<real_type, layout> &X, const std::size_t i, const plssvm::matrix<real_type, layout> &Y, const std::size_t j, const int degree, const real_type gamma, const real_type coef0) {
    PLSSVM_ASSERT(X.num_cols() == Y.num_cols(), "Sizes mismatch!: {} != {}", X.num_cols(), Y.num_cols());
    PLSSVM_ASSERT(i < X.num_rows(), "Out-of-bounce access!: {} < {}", i, X.num_rows());
    PLSSVM_ASSERT(j < Y.num_rows(), "Out-of-bounce access!: {} < {}", j, Y.num_rows());

    real_type result{ 0.0 };
    for (typename std::vector<real_type>::size_type dim = 0; dim < X.num_cols(); ++dim) {
        const real_type diff = X(i, dim) - Y(j, dim);
        result = std::fma(diff, diff, result);
    }
    return std::pow(std::fma(gamma, result, coef0), static_cast<real_type>(degree));
}
template float polynomial_kernel(const plssvm::matrix<float, plssvm::layout_type::aos> &, const std::size_t, const plssvm::matrix<float, plssvm::layout_type::aos> &, const std::size_t, const int, const float, const float);
template float polynomial_kernel(const plssvm::matrix<float, plssvm::layout_type::soa> &, const std::size_t, const plssvm::matrix<float, plssvm::layout_type::soa> &, const std::size_t, const int, const float, const float);
template double polynomial_kernel(const plssvm::matrix<double, plssvm::layout_type::aos> &, const std::size_t, const plssvm::matrix<double, plssvm::layout_type::aos> &, const std::size_t, const int, const double, const double);
template double polynomial_kernel(const plssvm::matrix<double, plssvm::layout_type::soa> &, const std::size_t, const plssvm::matrix<double, plssvm::layout_type::soa> &, const std::size_t, const int, const double, const double);


template <typename real_type, plssvm::layout_type layout>
real_type rbf_kernel(const plssvm::matrix<real_type, layout> &X, const std::size_t i, const plssvm::matrix<real_type, layout> &Y, const std::size_t j, const real_type gamma) {
    PLSSVM_ASSERT(X.num_cols() == Y.num_cols(), "Sizes mismatch!: {} != {}", X.num_cols(), Y.num_cols());
    PLSSVM_ASSERT(i < X.num_rows(), "Out-of-bounce access!: {} < {}", i, X.num_rows());
    PLSSVM_ASSERT(j < Y.num_rows(), "Out-of-bounce access!: {} < {}", j, Y.num_rows());

    real_type result{ 0.0 };
    for (typename std::vector<real_type>::size_type dim = 0; dim < X.num_cols(); ++dim) {
        const real_type diff = X(i, dim) - Y(j, dim);
        result = std::fma(diff, diff, result);
    }
    return std::exp(-gamma * result);
}
template float rbf_kernel(const plssvm::matrix<float, plssvm::layout_type::aos> &, const std::size_t, const plssvm::matrix<float, plssvm::layout_type::aos> &, const std::size_t, const float);
template float rbf_kernel(const plssvm::matrix<float, plssvm::layout_type::soa> &, const std::size_t, const plssvm::matrix<float, plssvm::layout_type::soa> &, const std::size_t, const float);
template double rbf_kernel(const plssvm::matrix<double, plssvm::layout_type::aos> &, const std::size_t, const plssvm::matrix<double, plssvm::layout_type::aos> &, const std::size_t, const double);
template double rbf_kernel(const plssvm::matrix<double, plssvm::layout_type::soa> &, const std::size_t, const plssvm::matrix<double, plssvm::layout_type::soa> &, const std::size_t, const double);


}  // namespace detail

template <typename real_type>
real_type kernel_function(const plssvm::parameter &params, const std::vector<real_type> &x, const std::vector<real_type> &y, [[maybe_unused]] const std::size_t num_devices) {
    PLSSVM_ASSERT(x.size() == y.size(), "Sizes mismatch!: {} != {}", x.size(), y.size());

    switch (params.kernel_type) {
        case plssvm::kernel_function_type::linear:
            return detail::linear_kernel(x, y, num_devices);
        case plssvm::kernel_function_type::polynomial:
            return detail::polynomial_kernel(x, y, params.degree.value(), static_cast<real_type>(params.gamma.value()), static_cast<real_type>(params.coef0.value()));
        case plssvm::kernel_function_type::rbf:
            return detail::rbf_kernel(x, y, static_cast<real_type>(params.gamma.value()));
    }
    // unreachable
    return real_type{};
}
template float kernel_function(const plssvm::parameter &, const std::vector<float> &, const std::vector<float> &, std::size_t);
template double kernel_function(const plssvm::parameter &, const std::vector<double> &, const std::vector<double> &, std::size_t);

template <typename real_type, plssvm::layout_type layout>
real_type kernel_function(const plssvm::parameter &params, const plssvm::matrix<real_type, layout> &X, const std::size_t i, const plssvm::matrix<real_type, layout> &Y, const std::size_t j, [[maybe_unused]] const std::size_t num_devices) {
    PLSSVM_ASSERT(X.num_cols() == Y.num_cols(), "Sizes mismatch!: {} != {}", X.num_cols(), Y.num_cols());
    PLSSVM_ASSERT(i < X.num_rows(), "Out-of-bounce access!: {} < {}", i, X.num_rows());
    PLSSVM_ASSERT(j < Y.num_rows(), "Out-of-bounce access!: {} < {}", j, Y.num_rows());

    switch (params.kernel_type) {
        case plssvm::kernel_function_type::linear:
            return detail::linear_kernel(X, i, Y, j, num_devices);
            break;
        case plssvm::kernel_function_type::polynomial:
            return detail::polynomial_kernel(X, i, Y, j, params.degree.value(), static_cast<real_type>(params.gamma.value()), static_cast<real_type>(params.coef0.value()));
            break;
        case plssvm::kernel_function_type::rbf:
            return detail::rbf_kernel(X, i, Y, j, static_cast<real_type>(params.gamma.value()));
            break;
    }
    // unreachable
    return real_type{};
}

template float kernel_function(const plssvm::parameter &, const plssvm::matrix<float, plssvm::layout_type::aos> &, const std::size_t, const plssvm::matrix<float, plssvm::layout_type::aos> &, const std::size_t, const std::size_t);
template float kernel_function(const plssvm::parameter &, const plssvm::matrix<float, plssvm::layout_type::soa> &, const std::size_t, const plssvm::matrix<float, plssvm::layout_type::soa> &, const std::size_t, const std::size_t);
template double kernel_function(const plssvm::parameter &, const plssvm::matrix<double, plssvm::layout_type::aos> &, const std::size_t, const plssvm::matrix<double, plssvm::layout_type::aos> &, const std::size_t, const std::size_t);
template double kernel_function(const plssvm::parameter &, const plssvm::matrix<double, plssvm::layout_type::soa> &, const std::size_t, const plssvm::matrix<double, plssvm::layout_type::soa> &, const std::size_t, const std::size_t);

template <typename real_type>
std::vector<real_type> generate_q(const plssvm::parameter &params, const plssvm::aos_matrix<real_type> &data, [[maybe_unused]] const std::size_t num_devices) {
    std::vector<real_type> result(data.num_rows() - 1);
    for (typename std::vector<std::vector<real_type>>::size_type i = 0; i < result.size(); ++i) {
        result[i] = kernel_function(params, data, data.num_rows() - 1, data, i, num_devices);
    }
    return result;
}
template std::vector<float> generate_q(const plssvm::parameter &, const plssvm::aos_matrix<float> &, std::size_t);
template std::vector<double> generate_q(const plssvm::parameter &, const plssvm::aos_matrix<double> &, std::size_t);

template <typename real_type>
[[nodiscard]] std::vector<std::vector<real_type>> assemble_kernel_matrix(const plssvm::parameter &params, const std::vector<std::vector<real_type>> &data, const std::vector<real_type> &q, const real_type QA_cost, const std::size_t num_devices) {
    std::vector<std::vector<real_type>> result(data.size() - 1, std::vector<real_type>(data.size() - 1, real_type{ 0.0 }));
    for (typename std::vector<std::vector<real_type>>::size_type row = 0; row < data.size() - 1; ++row) {
        for (typename std::vector<std::vector<real_type>>::size_type col = 0; col < data.size() - 1; ++col) {
            result[row][col] = kernel_function(params, data[row], data[col], num_devices) + QA_cost - q[row] - q[col];
            if (row == col) {
                result[row][col] += real_type{ 1.0 } / static_cast<real_type>(params.cost);
            }
        }
    }
    return result;
}
template std::vector<std::vector<float>> assemble_kernel_matrix(const plssvm::parameter &, const std::vector<std::vector<float>> &, const std::vector<float> &, const float, const std::size_t);
template std::vector<std::vector<double>> assemble_kernel_matrix(const plssvm::parameter &, const std::vector<std::vector<double>> &, const std::vector<double> &, const double, const std::size_t);

template <typename real_type, plssvm::layout_type layout>
std::vector<real_type> calculate_w(const plssvm::matrix<real_type, layout> &support_vectors, const std::vector<real_type> &weights) {
    PLSSVM_ASSERT(support_vectors.num_rows() == weights.size(), "Sizes mismatch!: {} != {}", support_vectors.num_rows(), weights.size());

    std::vector<real_type> result(support_vectors.num_cols());
    for (typename std::vector<real_type>::size_type i = 0; i < result.size(); ++i) {
        for (typename std::vector<real_type>::size_type j = 0; j < weights.size(); ++j) {
            result[i] = std::fma(weights[j], support_vectors(j, i), result[i]);
        }
    }
    return result;
}
template std::vector<float> calculate_w(const plssvm::matrix<float, plssvm::layout_type::aos> &, const std::vector<float> &);
template std::vector<float> calculate_w(const plssvm::matrix<float, plssvm::layout_type::soa> &, const std::vector<float> &);
template std::vector<double> calculate_w(const plssvm::matrix<double, plssvm::layout_type::aos> &, const std::vector<double> &);
template std::vector<double> calculate_w(const plssvm::matrix<double, plssvm::layout_type::soa> &, const std::vector<double> &);

template <typename real_type>
std::vector<real_type> device_kernel_function(const plssvm::parameter &params, const std::vector<std::vector<real_type>> &data, const std::vector<real_type> &rhs, const std::vector<real_type> &q, const real_type QA_cost, const real_type add) {
    PLSSVM_ASSERT(rhs.size() == q.size(), "Sizes mismatch!: {} != {}", rhs.size(), q.size());
    PLSSVM_ASSERT(rhs.size() == data.size() - 1, "Sizes mismatch!: {} != {}", rhs.size(), data.size() - 1);
    PLSSVM_ASSERT(add == real_type{ -1.0 } || add == real_type{ 1.0 }, "add must either be -1.0 or 1.0, but is {}!", add);

    using size_type = typename std::vector<real_type>::size_type;

    const size_type dept = rhs.size();

    std::vector<real_type> result(dept, 0.0);
    for (size_type i = 0; i < dept; ++i) {
        for (size_type j = 0; j < dept; ++j) {
            if (i >= j) {
                const real_type temp = kernel_function(params, data[i], data[j]) + QA_cost - q[i] - q[j];
                if (i == j) {
                    result[i] += (temp + real_type{ 1.0 } / static_cast<real_type>(params.cost)) * rhs[i] * add;
                } else {
                    result[i] += temp * rhs[j] * add;
                    result[j] += temp * rhs[i] * add;
                }
            }
        }
    }
    return result;
}
template std::vector<float> device_kernel_function(const plssvm::parameter &, const std::vector<std::vector<float>> &, const std::vector<float> &, const std::vector<float> &, float, float);
template std::vector<double> device_kernel_function(const plssvm::parameter &, const std::vector<std::vector<double>> &, const std::vector<double> &, const std::vector<double> &, double, double);

}  // namespace compare
