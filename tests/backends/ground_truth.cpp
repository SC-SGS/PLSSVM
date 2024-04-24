/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "tests/backends/ground_truth.hpp"

#include "plssvm/constants.hpp"                 // plssvm::PADDING_SIZE
#include "plssvm/detail/assert.hpp"             // PLSSVM_ASSERT
#include "plssvm/detail/data_distribution.hpp"  // plssvm::detail::triangular_data_distribution
#include "plssvm/kernel_function_types.hpp"     // plssvm::kernel_function_type
#include "plssvm/matrix.hpp"                    // plssvm::matrix, plssvm::layout_type
#include "plssvm/parameter.hpp"                 // plssvm::parameter
#include "plssvm/shape.hpp"                     // plssvm::shape

#include <cmath>      // std::pow, std::exp, std::fma
#include <cstddef>    // std::size_t
#include <utility>    // std::pair, std::make_pair, std::move
#include <vector>     // std::vector

namespace ground_truth {

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
real_type polynomial_kernel(const std::vector<real_type> &x, const std::vector<real_type> &y, const int degree, const real_type gamma, const real_type coef0) {
    PLSSVM_ASSERT(x.size() == y.size(), "Sizes mismatch!: {} != {}", x.size(), y.size());

    real_type result{ 0.0 };
    for (typename std::vector<real_type>::size_type i = 0; i < x.size(); ++i) {
        result = std::fma(x[i], y[i], result);
    }
    return std::pow(std::fma(gamma, result, coef0), static_cast<real_type>(degree));
}

template float polynomial_kernel(const std::vector<float> &, const std::vector<float> &, const int, const float, const float);
template double polynomial_kernel(const std::vector<double> &, const std::vector<double> &, const int, const double, const double);

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

template float rbf_kernel(const std::vector<float> &, const std::vector<float> &, const float);
template double rbf_kernel(const std::vector<double> &, const std::vector<double> &, const double);

template <typename real_type>
real_type sigmoid_kernel(const std::vector<real_type> &x, const std::vector<real_type> &y, const real_type gamma, const real_type coef0) {
    PLSSVM_ASSERT(x.size() == y.size(), "Sizes mismatch!: {} != {}", x.size(), y.size());

    real_type result{ 0.0 };
    for (typename std::vector<real_type>::size_type i = 0; i < x.size(); ++i) {
        result = std::fma(x[i], y[i], result);
    }
    return std::tanh(std::fma(gamma, result, coef0));
}

template float sigmoid_kernel(const std::vector<float> &, const std::vector<float> &, const float, const float);
template double sigmoid_kernel(const std::vector<double> &, const std::vector<double> &, const double, const double);

template <typename real_type>
real_type laplacian_kernel(const std::vector<real_type> &x, const std::vector<real_type> &y, const real_type gamma) {
    PLSSVM_ASSERT(x.size() == y.size(), "Sizes mismatch!: {} != {}", x.size(), y.size());

    real_type result{ 0.0 };
    for (typename std::vector<real_type>::size_type i = 0; i < x.size(); ++i) {
        result += std::abs(x[i] - y[i]);
    }
    return std::exp(-gamma * result);
}

template float laplacian_kernel(const std::vector<float> &, const std::vector<float> &, const float);
template double laplacian_kernel(const std::vector<double> &, const std::vector<double> &, const double);

template <typename real_type>
real_type chi_squared_kernel(const std::vector<real_type> &x, const std::vector<real_type> &y, const real_type gamma) {
    PLSSVM_ASSERT(x.size() == y.size(), "Sizes mismatch!: {} != {}", x.size(), y.size());

    real_type result{ 0.0 };
    for (typename std::vector<real_type>::size_type i = 0; i < x.size(); ++i) {
        const real_type sum = x[i] + y[i];
        if (sum != real_type{ 0.0 }) {
            const real_type temp = x[i] - y[i];
            result += (temp * temp) / sum;
        }
    }
    return std::exp(-gamma * result);
}

template float chi_squared_kernel(const std::vector<float> &, const std::vector<float> &, const float);
template double chi_squared_kernel(const std::vector<double> &, const std::vector<double> &, const double);


template <typename real_type, plssvm::layout_type layout>
real_type linear_kernel(const plssvm::matrix<real_type, layout> &X, const std::size_t i, const plssvm::matrix<real_type, layout> &Y, const std::size_t j) {
    PLSSVM_ASSERT(X.num_cols() == Y.num_cols(), "Sizes mismatch!: {} != {}", X.num_cols(), Y.num_cols());
    PLSSVM_ASSERT(i < X.num_rows(), "Out-of-bounce access!: {} < {}", i, X.num_rows());
    PLSSVM_ASSERT(j < Y.num_rows(), "Out-of-bounce access!: {} < {}", j, Y.num_rows());

    real_type result{ 0.0 };
    for (typename std::vector<real_type>::size_type dim = 0; dim < X.num_cols(); ++dim) {
        result = std::fma(X(i, dim), Y(j, dim), result);
    }
    return result;
}

template float linear_kernel(const plssvm::matrix<float, plssvm::layout_type::aos> &, const std::size_t, const plssvm::matrix<float, plssvm::layout_type::aos> &, const std::size_t);
template float linear_kernel(const plssvm::matrix<float, plssvm::layout_type::soa> &, const std::size_t, const plssvm::matrix<float, plssvm::layout_type::soa> &, const std::size_t);
template double linear_kernel(const plssvm::matrix<double, plssvm::layout_type::aos> &, const std::size_t, const plssvm::matrix<double, plssvm::layout_type::aos> &, const std::size_t);
template double linear_kernel(const plssvm::matrix<double, plssvm::layout_type::soa> &, const std::size_t, const plssvm::matrix<double, plssvm::layout_type::soa> &, const std::size_t);

template <typename real_type, plssvm::layout_type layout>
real_type polynomial_kernel(const plssvm::matrix<real_type, layout> &X, const std::size_t i, const plssvm::matrix<real_type, layout> &Y, const std::size_t j, const int degree, const real_type gamma, const real_type coef0) {
    PLSSVM_ASSERT(X.num_cols() == Y.num_cols(), "Sizes mismatch!: {} != {}", X.num_cols(), Y.num_cols());
    PLSSVM_ASSERT(i < X.num_rows(), "Out-of-bounce access!: {} < {}", i, X.num_rows());
    PLSSVM_ASSERT(j < Y.num_rows(), "Out-of-bounce access!: {} < {}", j, Y.num_rows());

    real_type result{ 0.0 };
    for (typename std::vector<real_type>::size_type dim = 0; dim < X.num_cols(); ++dim) {
        result = std::fma(X(i, dim), Y(j, dim), result);
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

template <typename real_type, plssvm::layout_type layout>
real_type sigmoid_kernel(const plssvm::matrix<real_type, layout> &X, const std::size_t i, const plssvm::matrix<real_type, layout> &Y, const std::size_t j, const real_type gamma, const real_type coef0) {
    PLSSVM_ASSERT(X.num_cols() == Y.num_cols(), "Sizes mismatch!: {} != {}", X.num_cols(), Y.num_cols());
    PLSSVM_ASSERT(i < X.num_rows(), "Out-of-bounce access!: {} < {}", i, X.num_rows());
    PLSSVM_ASSERT(j < Y.num_rows(), "Out-of-bounce access!: {} < {}", j, Y.num_rows());

    real_type result{ 0.0 };
    for (typename std::vector<real_type>::size_type dim = 0; dim < X.num_cols(); ++dim) {
        result = std::fma(X(i, dim), Y(j, dim), result);
    }
    return std::tanh(std::fma(gamma, result, coef0));
}

template float sigmoid_kernel(const plssvm::matrix<float, plssvm::layout_type::aos> &, const std::size_t, const plssvm::matrix<float, plssvm::layout_type::aos> &, const std::size_t, const float, const float);
template float sigmoid_kernel(const plssvm::matrix<float, plssvm::layout_type::soa> &, const std::size_t, const plssvm::matrix<float, plssvm::layout_type::soa> &, const std::size_t, const float, const float);
template double sigmoid_kernel(const plssvm::matrix<double, plssvm::layout_type::aos> &, const std::size_t, const plssvm::matrix<double, plssvm::layout_type::aos> &, const std::size_t, const double, const double);
template double sigmoid_kernel(const plssvm::matrix<double, plssvm::layout_type::soa> &, const std::size_t, const plssvm::matrix<double, plssvm::layout_type::soa> &, const std::size_t, const double, const double);

template <typename real_type, plssvm::layout_type layout>
real_type laplacian_kernel(const plssvm::matrix<real_type, layout> &X, const std::size_t i, const plssvm::matrix<real_type, layout> &Y, const std::size_t j, const real_type gamma) {
    PLSSVM_ASSERT(X.num_cols() == Y.num_cols(), "Sizes mismatch!: {} != {}", X.num_cols(), Y.num_cols());
    PLSSVM_ASSERT(i < X.num_rows(), "Out-of-bounce access!: {} < {}", i, X.num_rows());
    PLSSVM_ASSERT(j < Y.num_rows(), "Out-of-bounce access!: {} < {}", j, Y.num_rows());

    real_type result{ 0.0 };
    for (typename std::vector<real_type>::size_type dim = 0; dim < X.num_cols(); ++dim) {
        result += std::abs(X(i, dim) - Y(j, dim));
    }
    return std::exp(-gamma * result);
}

template float laplacian_kernel(const plssvm::matrix<float, plssvm::layout_type::aos> &, const std::size_t, const plssvm::matrix<float, plssvm::layout_type::aos> &, const std::size_t, const float);
template float laplacian_kernel(const plssvm::matrix<float, plssvm::layout_type::soa> &, const std::size_t, const plssvm::matrix<float, plssvm::layout_type::soa> &, const std::size_t, const float);
template double laplacian_kernel(const plssvm::matrix<double, plssvm::layout_type::aos> &, const std::size_t, const plssvm::matrix<double, plssvm::layout_type::aos> &, const std::size_t, const double);
template double laplacian_kernel(const plssvm::matrix<double, plssvm::layout_type::soa> &, const std::size_t, const plssvm::matrix<double, plssvm::layout_type::soa> &, const std::size_t, const double);

template <typename real_type, plssvm::layout_type layout>
real_type chi_squared_kernel(const plssvm::matrix<real_type, layout> &X, const std::size_t i, const plssvm::matrix<real_type, layout> &Y, const std::size_t j, const real_type gamma) {
    PLSSVM_ASSERT(X.num_cols() == Y.num_cols(), "Sizes mismatch!: {} != {}", X.num_cols(), Y.num_cols());
    PLSSVM_ASSERT(i < X.num_rows(), "Out-of-bounce access!: {} < {}", i, X.num_rows());
    PLSSVM_ASSERT(j < Y.num_rows(), "Out-of-bounce access!: {} < {}", j, Y.num_rows());

    real_type result{ 0.0 };
    for (typename std::vector<real_type>::size_type dim = 0; dim < X.num_cols(); ++dim) {
        const real_type sum = X(i, dim) + Y(j, dim);
        if (sum != real_type{ 0.0 }) {
            const real_type temp = X(i, dim) - Y(j, dim);
            result += (temp * temp) / sum;
        }
    }
    return std::exp(-gamma * result);
}

template float chi_squared_kernel(const plssvm::matrix<float, plssvm::layout_type::aos> &, const std::size_t, const plssvm::matrix<float, plssvm::layout_type::aos> &, const std::size_t, const float);
template float chi_squared_kernel(const plssvm::matrix<float, plssvm::layout_type::soa> &, const std::size_t, const plssvm::matrix<float, plssvm::layout_type::soa> &, const std::size_t, const float);
template double chi_squared_kernel(const plssvm::matrix<double, plssvm::layout_type::aos> &, const std::size_t, const plssvm::matrix<double, plssvm::layout_type::aos> &, const std::size_t, const double);
template double chi_squared_kernel(const plssvm::matrix<double, plssvm::layout_type::soa> &, const std::size_t, const plssvm::matrix<double, plssvm::layout_type::soa> &, const std::size_t, const double);

template <typename real_type>
plssvm::aos_matrix<real_type> predict_values(const plssvm::parameter &params, const plssvm::soa_matrix<real_type> &w, const plssvm::aos_matrix<real_type> &weights, const std::vector<real_type> &rho, const plssvm::soa_matrix<real_type> &support_vectors, const plssvm::soa_matrix<real_type> &predict_points, const std::size_t row_offset, const std::size_t device_specific_num_rows) {
    PLSSVM_ASSERT(w.empty() || w.num_rows() == weights.num_rows(), "Sizes mismatch!: {} != {}", w.num_rows(), weights.num_rows());
    PLSSVM_ASSERT(w.empty() || w.num_cols() == support_vectors.num_cols(), "Sizes mismatch!: {} != {}", w.num_cols(), support_vectors.num_cols());
    PLSSVM_ASSERT(weights.num_rows() == rho.size(), "Sizes mismatch!: {} != {}", weights.num_rows(), rho.size());
    PLSSVM_ASSERT(weights.num_cols() == support_vectors.num_rows(), "Sizes mismatch!: {} != {}", weights.num_cols(), support_vectors.num_rows());
    PLSSVM_ASSERT(support_vectors.num_cols() == predict_points.num_cols(), "Sizes mismatch!: {} != {}", support_vectors.num_cols(), predict_points.num_cols());

    const std::size_t num_classes = weights.num_rows();
    const std::size_t num_sv = support_vectors.num_rows();
    const std::size_t num_features = predict_points.num_cols();

    plssvm::aos_matrix<real_type> result{ plssvm::shape{ device_specific_num_rows, num_classes }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE } };

    switch (params.kernel_type) {
        case plssvm::kernel_function_type::linear:
            {
                for (std::size_t c = 0; c < num_classes; ++c) {
                    for (std::size_t i = row_offset; i < row_offset + device_specific_num_rows; ++i) {
                        real_type temp{ 0.0 };
                        for (std::size_t f = 0; f < num_features; ++f) {
                            temp = std::fma(w(c, f), predict_points(i, f), temp);
                        }
                        result(i - row_offset, c) = temp - rho[c];
                    }
                }
            }
            break;
        case plssvm::kernel_function_type::polynomial:
            {
                for (std::size_t c = 0; c < num_classes; ++c) {
                    for (std::size_t i = row_offset; i < row_offset + device_specific_num_rows; ++i) {
                        for (std::size_t j = 0; j < num_sv; ++j) {
                            real_type temp{ 0.0 };
                            for (std::size_t f = 0; f < num_features; ++f) {
                                temp = std::fma(support_vectors(j, f), predict_points(i, f), temp);
                            }
                            temp = std::fma(static_cast<real_type>(plssvm::get_gamma_value(params.gamma)), temp, static_cast<real_type>(params.coef0));
                            temp = weights(c, j) * static_cast<real_type>(std::pow(temp, params.degree));
                            if (j == 0) {
                                temp -= rho[c];
                            }
                            result(i - row_offset, c) += temp;
                        }
                    }
                }
            }
            break;
        case plssvm::kernel_function_type::rbf:
            {
                for (std::size_t c = 0; c < num_classes; ++c) {
                    for (std::size_t i = row_offset; i < row_offset + device_specific_num_rows; ++i) {
                        for (std::size_t j = 0; j < num_sv; ++j) {
                            real_type temp{ 0.0 };
                            for (std::size_t f = 0; f < num_features; ++f) {
                                const real_type d = support_vectors(j, f) - predict_points(i, f);
                                temp = std::fma(d, d, temp);
                            }
                            temp = weights(c, j) * static_cast<real_type>(std::exp(static_cast<real_type>(-plssvm::get_gamma_value(params.gamma)) * temp));
                            if (j == 0) {
                                temp -= rho[c];
                            }
                            result(i - row_offset, c) += temp;
                        }
                    }
                }
            }
            break;
        case plssvm::kernel_function_type::sigmoid:
            {
                for (std::size_t c = 0; c < num_classes; ++c) {
                    for (std::size_t i = row_offset; i < row_offset + device_specific_num_rows; ++i) {
                        for (std::size_t j = 0; j < num_sv; ++j) {
                            real_type temp{ 0.0 };
                            for (std::size_t f = 0; f < num_features; ++f) {
                                temp = std::fma(support_vectors(j, f), predict_points(i, f), temp);
                            }
                            temp = weights(c, j) * static_cast<real_type>(std::tanh(static_cast<real_type>(plssvm::get_gamma_value(params.gamma)) * temp + static_cast<real_type>(params.coef0)));
                            if (j == 0) {
                                temp -= rho[c];
                            }
                            result(i - row_offset, c) += temp;
                        }
                    }
                }
            }
            break;
        case plssvm::kernel_function_type::laplacian:
            {
                for (std::size_t c = 0; c < num_classes; ++c) {
                    for (std::size_t i = row_offset; i < row_offset + device_specific_num_rows; ++i) {
                        for (std::size_t j = 0; j < num_sv; ++j) {
                            real_type temp{ 0.0 };
                            for (std::size_t f = 0; f < num_features; ++f) {
                                temp += std::abs(support_vectors(j, f) - predict_points(i, f));
                            }
                            temp = weights(c, j) * static_cast<real_type>(std::exp(static_cast<real_type>(-plssvm::get_gamma_value(params.gamma)) * temp));
                            if (j == 0) {
                                temp -= rho[c];
                            }
                            result(i - row_offset, c) += temp;
                        }
                    }
                }
            }
            break;
        case plssvm::kernel_function_type::chi_squared:
            {
                for (std::size_t c = 0; c < num_classes; ++c) {
                    for (std::size_t i = row_offset; i < row_offset + device_specific_num_rows; ++i) {
                        for (std::size_t j = 0; j < num_sv; ++j) {
                            real_type temp{ 0.0 };
                            for (std::size_t f = 0; f < num_features; ++f) {
                                const real_type diff = support_vectors(j, f) - predict_points(i, f);
                                temp += (diff * diff) / (support_vectors(j, f) + predict_points(i, f));
                            }
                            temp = weights(c, j) * static_cast<real_type>(std::exp(static_cast<real_type>(-plssvm::get_gamma_value(params.gamma)) * temp));
                            if (j == 0) {
                                temp -= rho[c];
                            }
                            result(i - row_offset, c) += temp;
                        }
                    }
                }
            }
            break;
    }

    return result;
}

template plssvm::aos_matrix<float> predict_values(const plssvm::parameter &, const plssvm::soa_matrix<float> &, const plssvm::aos_matrix<float> &, const std::vector<float> &, const plssvm::soa_matrix<float> &, const plssvm::soa_matrix<float> &, const std::size_t, const std::size_t);
template plssvm::aos_matrix<double> predict_values(const plssvm::parameter &, const plssvm::soa_matrix<double> &, const plssvm::aos_matrix<double> &, const std::vector<double> &, const plssvm::soa_matrix<double> &, const plssvm::soa_matrix<double> &, const std::size_t, const std::size_t);


}  // namespace detail

template <typename real_type>
real_type kernel_function(const plssvm::parameter &params, const std::vector<real_type> &x, const std::vector<real_type> &y) {
    PLSSVM_ASSERT(x.size() == y.size(), "Sizes mismatch!: {} != {}", x.size(), y.size());

    switch (params.kernel_type) {
        case plssvm::kernel_function_type::linear:
            return detail::linear_kernel(x, y);
        case plssvm::kernel_function_type::polynomial:
            return detail::polynomial_kernel(x, y, params.degree, static_cast<real_type>(plssvm::get_gamma_value(params.gamma)), static_cast<real_type>(params.coef0));
        case plssvm::kernel_function_type::rbf:
            return detail::rbf_kernel(x, y, static_cast<real_type>(plssvm::get_gamma_value(params.gamma)));
        case plssvm::kernel_function_type::sigmoid:
            return detail::sigmoid_kernel(x, y, static_cast<real_type>(plssvm::get_gamma_value(params.gamma)), static_cast<real_type>(params.coef0));
        case plssvm::kernel_function_type::laplacian:
            return detail::laplacian_kernel(x, y, static_cast<real_type>(plssvm::get_gamma_value(params.gamma)));
        case plssvm::kernel_function_type::chi_squared:
            return detail::chi_squared_kernel(x, y, static_cast<real_type>(plssvm::get_gamma_value(params.gamma)));
    }
    // unreachable
    return real_type{};
}

template float kernel_function(const plssvm::parameter &, const std::vector<float> &, const std::vector<float> &);
template double kernel_function(const plssvm::parameter &, const std::vector<double> &, const std::vector<double> &);

template <typename real_type, plssvm::layout_type layout>
real_type kernel_function(const plssvm::parameter &params, const plssvm::matrix<real_type, layout> &X, const std::size_t i, const plssvm::matrix<real_type, layout> &Y, const std::size_t j) {
    PLSSVM_ASSERT(X.num_cols() == Y.num_cols(), "Sizes mismatch!: {} != {}", X.num_cols(), Y.num_cols());
    PLSSVM_ASSERT(i < X.num_rows(), "Out-of-bounce access!: {} < {}", i, X.num_rows());
    PLSSVM_ASSERT(j < Y.num_rows(), "Out-of-bounce access!: {} < {}", j, Y.num_rows());

    switch (params.kernel_type) {
        case plssvm::kernel_function_type::linear:
            return detail::linear_kernel(X, i, Y, j);
        case plssvm::kernel_function_type::polynomial:
            return detail::polynomial_kernel(X, i, Y, j, params.degree, static_cast<real_type>(plssvm::get_gamma_value(params.gamma)), static_cast<real_type>(params.coef0));
        case plssvm::kernel_function_type::rbf:
            return detail::rbf_kernel(X, i, Y, j, static_cast<real_type>(plssvm::get_gamma_value(params.gamma)));
        case plssvm::kernel_function_type::sigmoid:
            return detail::sigmoid_kernel(X, i, Y, j, static_cast<real_type>(plssvm::get_gamma_value(params.gamma)), static_cast<real_type>(params.coef0));
        case plssvm::kernel_function_type::laplacian:
            return detail::laplacian_kernel(X, i, Y, j, static_cast<real_type>(plssvm::get_gamma_value(params.gamma)));
        case plssvm::kernel_function_type::chi_squared:
            return detail::chi_squared_kernel(X, i, Y, j, static_cast<real_type>(plssvm::get_gamma_value(params.gamma)));
    }
    // unreachable
    return real_type{};
}

template float kernel_function(const plssvm::parameter &, const plssvm::matrix<float, plssvm::layout_type::aos> &, const std::size_t, const plssvm::matrix<float, plssvm::layout_type::aos> &, const std::size_t);
template float kernel_function(const plssvm::parameter &, const plssvm::matrix<float, plssvm::layout_type::soa> &, const std::size_t, const plssvm::matrix<float, plssvm::layout_type::soa> &, const std::size_t);
template double kernel_function(const plssvm::parameter &, const plssvm::matrix<double, plssvm::layout_type::aos> &, const std::size_t, const plssvm::matrix<double, plssvm::layout_type::aos> &, const std::size_t);
template double kernel_function(const plssvm::parameter &, const plssvm::matrix<double, plssvm::layout_type::soa> &, const std::size_t, const plssvm::matrix<double, plssvm::layout_type::soa> &, const std::size_t);

template <typename real_type>
std::pair<std::vector<real_type>, real_type> perform_dimensional_reduction(const plssvm::parameter &params, const plssvm::soa_matrix<real_type> &data) {
    std::vector<real_type> result(data.num_rows() - 1);
    for (typename std::vector<std::vector<real_type>>::size_type i = 0; i < result.size(); ++i) {
        result[i] = kernel_function(params, data, data.num_rows() - 1, data, i);
    }
    const real_type QA_cost = kernel_function(params, data, data.num_rows() - 1, data, data.num_rows() - 1) + real_type{ 1.0 } / static_cast<real_type>(params.cost);
    return std::make_pair(std::move(result), QA_cost);
}

template std::pair<std::vector<float>, float> perform_dimensional_reduction(const plssvm::parameter &, const plssvm::soa_matrix<float> &);
template std::pair<std::vector<double>, double> perform_dimensional_reduction(const plssvm::parameter &, const plssvm::soa_matrix<double> &);

template <typename real_type>
std::vector<real_type> assemble_device_specific_kernel_matrix(const plssvm::parameter &params, const plssvm::soa_matrix<real_type> &data, const std::vector<real_type> &q, const real_type QA_cost, const plssvm::detail::data_distribution &dist, const std::size_t device_id) {
    const auto &tri_dist = dynamic_cast<const plssvm::detail::triangular_data_distribution &>(dist);
    std::vector<real_type> result{};
    result.reserve(tri_dist.calculate_explicit_kernel_matrix_num_entries_padded(device_id));
    const std::size_t num_rows_reduced = data.num_rows() - 1;

    for (std::size_t row = tri_dist.place_row_offset(device_id); row < tri_dist.place_row_offset(device_id) + tri_dist.place_specific_num_rows(device_id); ++row) {
        for (std::size_t col = row; col < num_rows_reduced; ++col) {
            result.push_back(kernel_function(params, data, row, data, col) + QA_cost - q[row] - q[col]);
            if (row == col) {
                result.back() += real_type{ 1.0 } / static_cast<real_type>(params.cost);
            }
        }
        result.insert(result.cend(), plssvm::PADDING_SIZE, real_type{ 0.0 });
    }
    const std::size_t remaining_rows = num_rows_reduced - (tri_dist.place_row_offset(device_id) + tri_dist.place_specific_num_rows(device_id));
    const std::size_t remaining_rows_without_padding = remaining_rows - plssvm::PADDING_SIZE;
    const std::size_t num_padding_entries = (remaining_rows * (remaining_rows + 1) / 2) - (remaining_rows_without_padding * (remaining_rows_without_padding + 1) / 2);
    result.insert(result.cend(), num_padding_entries + static_cast<std::size_t>(plssvm::PADDING_SIZE * plssvm::PADDING_SIZE), real_type{ 0.0 });

    return result;
}

template std::vector<float> assemble_device_specific_kernel_matrix(const plssvm::parameter &, const plssvm::soa_matrix<float> &, const std::vector<float> &, const float, const plssvm::detail::data_distribution &, const std::size_t);
template std::vector<double> assemble_device_specific_kernel_matrix(const plssvm::parameter &, const plssvm::soa_matrix<double> &, const std::vector<double> &, const double, const plssvm::detail::data_distribution &, const std::size_t);

template <typename real_type>
plssvm::aos_matrix<real_type> assemble_full_kernel_matrix(const plssvm::parameter &params, const plssvm::soa_matrix<real_type> &data, const std::vector<real_type> &q, const real_type QA_cost) {
    PLSSVM_ASSERT(data.num_rows() - 1 == q.size(), "Sizes mismatch!: {} != {}", data.num_rows() - 1, q.size());

    const std::size_t num_rows_reduced = data.num_rows() - 1;
    plssvm::aos_matrix<real_type> result{ plssvm::shape{ num_rows_reduced, num_rows_reduced }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE } };

    for (std::size_t row = 0; row < num_rows_reduced; ++row) {
        for (std::size_t col = row; col < num_rows_reduced; ++col) {
            result(row, col) = kernel_function(params, data, row, data, col) + QA_cost - q[row] - q[col];
            result(col, row) = result(row, col);
            if (row == col) {
                result(row, col) += real_type{ 1.0 } / static_cast<real_type>(params.cost);
            }
        }
    }
    return result;
}

template plssvm::aos_matrix<float> assemble_full_kernel_matrix(const plssvm::parameter &, const plssvm::soa_matrix<float> &, const std::vector<float> &, const float);
template plssvm::aos_matrix<double> assemble_full_kernel_matrix(const plssvm::parameter &, const plssvm::soa_matrix<double> &, const std::vector<double> &, const double);

template <typename real_type>
void gemm(const real_type alpha, const plssvm::aos_matrix<real_type> &A, const plssvm::soa_matrix<real_type> &B, const real_type beta, plssvm::soa_matrix<real_type> &C) {
    PLSSVM_ASSERT(A.shape() == (plssvm::shape{ B.num_cols(), B.num_cols() }), "Shapes mismatch!: {} != {}", A.shape(), (plssvm::shape{ B.num_cols(), B.num_cols() }));
    PLSSVM_ASSERT(B.shape() == C.shape(), "Shapes mismatch!: {} != {}", B.shape(), C.shape());
    // A: #data_points - 1 x #data_points - 1
    // B: #classes x #data_points - 1
    // C: #classes x #data_points - 1

    for (std::size_t row = 0; row < C.num_rows(); ++row) {
        for (std::size_t col = 0; col < C.num_cols(); ++col) {
            real_type temp{ 0.0 };
            for (std::size_t k = 0; k < B.num_cols(); ++k) {
                temp = std::fma(A(col, k), B(row, k), temp);
            }
            C(row, col) = alpha * temp + beta * C(row, col);
        }
    }
}

template void gemm(const float, const plssvm::aos_matrix<float> &, const plssvm::soa_matrix<float> &, const float, plssvm::soa_matrix<float> &);
template void gemm(const double, const plssvm::aos_matrix<double> &, const plssvm::soa_matrix<double> &, const double, plssvm::soa_matrix<double> &);

template <typename real_type>
void device_specific_gemm(const real_type alpha, const plssvm::aos_matrix<real_type> &A, const plssvm::soa_matrix<real_type> &B, plssvm::soa_matrix<real_type> &C, const plssvm::detail::data_distribution &dist, std::size_t device_id) {
    PLSSVM_ASSERT(A.shape() == (plssvm::shape{ B.num_cols(), B.num_cols() }), "Shapes mismatch!: {} != {}", A.shape(), (plssvm::shape{ B.num_cols(), B.num_cols() }));
    PLSSVM_ASSERT(B.shape() == C.shape(), "Shapes mismatch!: {} != {}", B.shape(), C.shape());
    // A: #data_points - 1 x #data_points - 1 -> memory optimized to only hold the upper triangular matrix!
    // B: #classes x #data_points - 1
    // C: #classes x #data_points - 1

    const auto &tri_dist = dynamic_cast<const plssvm::detail::triangular_data_distribution &>(dist);
    const std::size_t num_rows_reduced = B.num_cols();
    const std::size_t num_classes = B.num_rows();

    // normal access
    for (std::size_t row = 0; row < num_classes; ++row) {
        for (std::size_t col = tri_dist.place_row_offset(device_id); col < tri_dist.place_row_offset(device_id) + tri_dist.place_specific_num_rows(device_id); ++col) {
            real_type temp{ 0.0 };
            for (std::size_t k = tri_dist.place_row_offset(device_id); k < num_rows_reduced; ++k) {
                temp = std::fma(A(col, k), B(row, k), temp);
            }
            C(row, col) += alpha * temp;
        }
    }

    // "mirror" access
    for (std::size_t row = 0; row < num_classes; ++row) {
        for (std::size_t col = tri_dist.place_row_offset(device_id) + tri_dist.place_specific_num_rows(device_id); col < num_rows_reduced; ++col) {
            real_type temp{ 0.0 };
            for (std::size_t k = tri_dist.place_row_offset(device_id); k < tri_dist.place_row_offset(device_id) + tri_dist.place_specific_num_rows(device_id); ++k) {
                temp = std::fma(A(col, k), B(row, k), temp);
            }
            C(row, col) += alpha * temp;
        }
    }
}

template void device_specific_gemm(const float, const plssvm::aos_matrix<float> &, const plssvm::soa_matrix<float> &, plssvm::soa_matrix<float> &, const plssvm::detail::data_distribution &, const std::size_t);
template void device_specific_gemm(const double, const plssvm::aos_matrix<double> &, const plssvm::soa_matrix<double> &, plssvm::soa_matrix<double> &, const plssvm::detail::data_distribution &, const std::size_t);

template <typename real_type>
plssvm::soa_matrix<real_type> calculate_w(const plssvm::aos_matrix<real_type> &weights, const plssvm::soa_matrix<real_type> &support_vectors) {
    PLSSVM_ASSERT(support_vectors.num_rows() == weights.num_cols(), "Sizes mismatch!: {} != {}", support_vectors.num_rows(), weights.num_cols());

    plssvm::soa_matrix<real_type> result{ plssvm::shape{ weights.num_rows(), support_vectors.num_cols() }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE } };
    for (std::size_t c = 0; c < weights.num_rows(); ++c) {
        for (std::size_t i = 0; i < support_vectors.num_cols(); ++i) {
            for (std::size_t j = 0; j < weights.num_cols(); ++j) {
                result(c, i) = std::fma(weights(c, j), support_vectors(j, i), result(c, i));
            }
        }
    }
    return result;
}

template plssvm::soa_matrix<float> calculate_w(const plssvm::aos_matrix<float> &, const plssvm::soa_matrix<float> &);
template plssvm::soa_matrix<double> calculate_w(const plssvm::aos_matrix<double> &, const plssvm::soa_matrix<double> &);

template <typename real_type>
plssvm::soa_matrix<real_type> calculate_device_specific_w(const plssvm::aos_matrix<real_type> &weights, const plssvm::soa_matrix<real_type> &support_vectors, const plssvm::detail::data_distribution &dist, const std::size_t device_id) {
    PLSSVM_ASSERT(support_vectors.num_rows() == weights.num_cols(), "Sizes mismatch!: {} != {}", support_vectors.num_rows(), weights.num_cols());
    // weights:         #num_classes x #num_data_points - 1
    // support_vectors: #num_data_points - 1 x #num_features
    // result:          #num_classes x #num_features

    const auto &rect_dist = dynamic_cast<const plssvm::detail::rectangular_data_distribution &>(dist);

    plssvm::soa_matrix<real_type> result{ plssvm::shape{ weights.num_rows(), support_vectors.num_cols() }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE } };
    for (std::size_t c = 0; c < weights.num_rows(); ++c) {
        for (std::size_t i = 0; i < support_vectors.num_cols(); ++i) {
            for (std::size_t j = rect_dist.place_row_offset(device_id); j < rect_dist.place_row_offset(device_id) + rect_dist.place_specific_num_rows(device_id); ++j) {
                result(c, i) = std::fma(weights(c, j), support_vectors(j, i), result(c, i));
            }
        }
    }
    return result;
}

template plssvm::soa_matrix<float> calculate_device_specific_w(const plssvm::aos_matrix<float> &, const plssvm::soa_matrix<float> &, const plssvm::detail::data_distribution &, const std::size_t);
template plssvm::soa_matrix<double> calculate_device_specific_w(const plssvm::aos_matrix<double> &, const plssvm::soa_matrix<double> &, const plssvm::detail::data_distribution &, const std::size_t);

template <typename real_type>
plssvm::aos_matrix<real_type> predict_values(const plssvm::parameter &params, const plssvm::soa_matrix<real_type> &w, const plssvm::aos_matrix<real_type> &weights, const std::vector<real_type> &rho, const plssvm::soa_matrix<real_type> &support_vectors, const plssvm::soa_matrix<real_type> &predict_points) {
    PLSSVM_ASSERT(w.empty() || w.num_rows() == weights.num_rows(), "Sizes mismatch!: {} != {}", w.num_rows(), weights.num_rows());
    PLSSVM_ASSERT(w.empty() || w.num_cols() == support_vectors.num_cols(), "Sizes mismatch!: {} != {}", w.num_cols(), support_vectors.num_cols());
    PLSSVM_ASSERT(weights.num_rows() == rho.size(), "Sizes mismatch!: {} != {}", weights.num_rows(), rho.size());
    PLSSVM_ASSERT(weights.num_cols() == support_vectors.num_rows(), "Sizes mismatch!: {} != {}", weights.num_cols(), support_vectors.num_rows());
    PLSSVM_ASSERT(support_vectors.num_cols() == predict_points.num_cols(), "Sizes mismatch!: {} != {}", support_vectors.num_cols(), predict_points.num_cols());

    const std::size_t num_predict_points = predict_points.num_rows();

    return detail::predict_values(params, w, weights, rho, support_vectors, predict_points, 0, num_predict_points);
}

template plssvm::aos_matrix<float> predict_values(const plssvm::parameter &, const plssvm::soa_matrix<float> &, const plssvm::aos_matrix<float> &, const std::vector<float> &, const plssvm::soa_matrix<float> &, const plssvm::soa_matrix<float> &);
template plssvm::aos_matrix<double> predict_values(const plssvm::parameter &, const plssvm::soa_matrix<double> &, const plssvm::aos_matrix<double> &, const std::vector<double> &, const plssvm::soa_matrix<double> &, const plssvm::soa_matrix<double> &);

template <typename real_type>
[[nodiscard]] plssvm::aos_matrix<real_type> predict_device_specific_values(const plssvm::parameter &params, const plssvm::soa_matrix<real_type> &w, const plssvm::aos_matrix<real_type> &weights, const std::vector<real_type> &rho, const plssvm::soa_matrix<real_type> &support_vectors, const plssvm::soa_matrix<real_type> &predict_points, const plssvm::detail::data_distribution &dist, const std::size_t device_id) {
    PLSSVM_ASSERT(w.empty() || w.num_rows() == weights.num_rows(), "Sizes mismatch!: {} != {}", w.num_rows(), weights.num_rows());
    PLSSVM_ASSERT(w.empty() || w.num_cols() == support_vectors.num_cols(), "Sizes mismatch!: {} != {}", w.num_cols(), support_vectors.num_cols());
    PLSSVM_ASSERT(weights.num_rows() == rho.size(), "Sizes mismatch!: {} != {}", weights.num_rows(), rho.size());
    PLSSVM_ASSERT(weights.num_cols() == support_vectors.num_rows(), "Sizes mismatch!: {} != {}", weights.num_cols(), support_vectors.num_rows());
    PLSSVM_ASSERT(support_vectors.num_cols() == predict_points.num_cols(), "Sizes mismatch!: {} != {}", support_vectors.num_cols(), predict_points.num_cols());

    const auto &rect_dist = dynamic_cast<const plssvm::detail::rectangular_data_distribution &>(dist);
    const std::size_t device_specific_num_predict_points = rect_dist.place_specific_num_rows(device_id);
    const std::size_t row_offset = rect_dist.place_row_offset(device_id);

    return detail::predict_values(params, w, weights, rho, support_vectors, predict_points, row_offset, device_specific_num_predict_points);
}

template plssvm::aos_matrix<float> predict_device_specific_values(const plssvm::parameter &, const plssvm::soa_matrix<float> &, const plssvm::aos_matrix<float> &, const std::vector<float> &, const plssvm::soa_matrix<float> &, const plssvm::soa_matrix<float> &, const plssvm::detail::data_distribution &, const std::size_t);
template plssvm::aos_matrix<double> predict_device_specific_values(const plssvm::parameter &, const plssvm::soa_matrix<double> &, const plssvm::aos_matrix<double> &, const std::vector<double> &, const plssvm::soa_matrix<double> &, const plssvm::soa_matrix<double> &, const plssvm::detail::data_distribution &, const std::size_t);

}  // namespace ground_truth
