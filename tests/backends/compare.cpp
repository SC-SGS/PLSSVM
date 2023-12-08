/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "backends/compare.hpp"

#include "plssvm/constants.hpp"              // plssvm::PADDING_SIZE
#include "plssvm/detail/assert.hpp"          // PLSSVM_ASSERT
#include "plssvm/kernel_function_types.hpp"  // plssvm::kernel_function_type
#include "plssvm/matrix.hpp"                 // plssvm::matrix, plssvm::layout_type
#include "plssvm/parameter.hpp"              // plssvm::parameter

#include <algorithm>  // std::min
#include <cmath>      // std::pow, std::exp, std::fma
#include <cstddef>    // std::size_t
#include <vector>     // std::vector

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
        case plssvm::kernel_function_type::polynomial:
            return detail::polynomial_kernel(X, i, Y, j, params.degree.value(), static_cast<real_type>(params.gamma.value()), static_cast<real_type>(params.coef0.value()));
        case plssvm::kernel_function_type::rbf:
            return detail::rbf_kernel(X, i, Y, j, static_cast<real_type>(params.gamma.value()));
    }
    // unreachable
    return real_type{};
}

template float kernel_function(const plssvm::parameter &, const plssvm::matrix<float, plssvm::layout_type::aos> &, const std::size_t, const plssvm::matrix<float, plssvm::layout_type::aos> &, const std::size_t, const std::size_t);
template float kernel_function(const plssvm::parameter &, const plssvm::matrix<float, plssvm::layout_type::soa> &, const std::size_t, const plssvm::matrix<float, plssvm::layout_type::soa> &, const std::size_t, const std::size_t);
template double kernel_function(const plssvm::parameter &, const plssvm::matrix<double, plssvm::layout_type::aos> &, const std::size_t, const plssvm::matrix<double, plssvm::layout_type::aos> &, const std::size_t, const std::size_t);
template double kernel_function(const plssvm::parameter &, const plssvm::matrix<double, plssvm::layout_type::soa> &, const std::size_t, const plssvm::matrix<double, plssvm::layout_type::soa> &, const std::size_t, const std::size_t);

template <typename real_type>
std::vector<real_type> perform_dimensional_reduction(const plssvm::parameter &params, const plssvm::soa_matrix<real_type> &data, [[maybe_unused]] const std::size_t num_devices) {
    std::vector<real_type> result(data.num_rows() - 1);
    for (typename std::vector<std::vector<real_type>>::size_type i = 0; i < result.size(); ++i) {
        result[i] = kernel_function(params, data, data.num_rows() - 1, data, i, num_devices);
    }
    return result;
}
template std::vector<float> perform_dimensional_reduction(const plssvm::parameter &, const plssvm::soa_matrix<float> &, std::size_t);
template std::vector<double> perform_dimensional_reduction(const plssvm::parameter &, const plssvm::soa_matrix<double> &, std::size_t);

template <typename real_type>
[[nodiscard]] std::vector<real_type> assemble_kernel_matrix_symm(const plssvm::parameter &params, const plssvm::soa_matrix<real_type> &data, const std::vector<real_type> &q, const real_type QA_cost, const std::size_t padding, const std::size_t num_devices) {
    const std::size_t num_rows_reduced = data.num_rows() - 1;
    std::vector<real_type> result{};
    result.reserve((num_rows_reduced + padding) * (num_rows_reduced + padding + 1) / 2);

    for (std::size_t row = 0; row < num_rows_reduced; ++row) {
        for (std::size_t col = row; col < num_rows_reduced; ++col) {
            result.push_back(kernel_function(params, data, row, data, col, num_devices) + QA_cost - q[row] - q[col]);
            if (row == col) {
                result.back() += real_type{ 1.0 } / static_cast<real_type>(params.cost);
            }
        }
        result.insert(result.cend(), padding, real_type{ 0.0 });
    }
    result.insert(result.cend(), padding * (padding + 1) / 2, real_type{ 0.0 });
    return result;
}
template std::vector<float> assemble_kernel_matrix_symm(const plssvm::parameter &, const plssvm::soa_matrix<float> &, const std::vector<float> &, const float, const std::size_t, const std::size_t);
template std::vector<double> assemble_kernel_matrix_symm(const plssvm::parameter &, const plssvm::soa_matrix<double> &, const std::vector<double> &, const double, const std::size_t, const std::size_t);

template <typename real_type>
[[nodiscard]] std::vector<real_type> assemble_kernel_matrix_gemm(const plssvm::parameter &params, const plssvm::soa_matrix<real_type> &data, const std::vector<real_type> &q, const real_type QA_cost, const std::size_t padding, const std::size_t num_devices) {
    PLSSVM_ASSERT(data.num_rows() - 1 == q.size(), "Sizes mismatch!: {} != {}", data.num_rows() - 1, q.size());

    const std::size_t num_rows_reduced = data.num_rows() - 1;
    std::vector<real_type> result{};
    result.reserve((num_rows_reduced + padding) * (num_rows_reduced + padding));

    for (std::size_t row = 0; row < num_rows_reduced; ++row) {
        for (std::size_t col = 0; col < num_rows_reduced; ++col) {
            result.push_back(kernel_function(params, data, row, data, col, num_devices) + QA_cost - q[row] - q[col]);
            if (row == col) {
                result.back() += real_type{ 1.0 } / static_cast<real_type>(params.cost);
            }
        }
        result.insert(result.cend(), padding, real_type{ 0.0 });
    }
    result.insert(result.cend(), padding * (num_rows_reduced + padding), real_type{ 0.0 });
    return result;
}
template std::vector<float> assemble_kernel_matrix_gemm(const plssvm::parameter &, const plssvm::soa_matrix<float> &, const std::vector<float> &, const float, const std::size_t, const std::size_t);
template std::vector<double> assemble_kernel_matrix_gemm(const plssvm::parameter &, const plssvm::soa_matrix<double> &, const std::vector<double> &, const double, const std::size_t, const std::size_t);

template <typename real_type>
void gemm(const real_type alpha, const std::vector<real_type> &A, const plssvm::soa_matrix<real_type> &B, const real_type beta, plssvm::soa_matrix<real_type> &C) {
    PLSSVM_ASSERT(A.size() == (B.num_cols() + plssvm::PADDING_SIZE) * (B.num_cols() + plssvm::PADDING_SIZE), "Sizes mismatch!: {} != {}", A.size(), (B.num_cols() + plssvm::PADDING_SIZE) * (B.num_cols() + plssvm::PADDING_SIZE));
    PLSSVM_ASSERT(B.shape() == C.shape(), "Shapes mismatch!: [{}] != [{}]", fmt::join(B.shape(), ", "), fmt::join(C.shape(), ", "));
    // A: #data_points - 1 x #data_points - 1
    // B: #classes x #data_points - 1
    // C: #classes x #data_points - 1

    for (std::size_t row = 0; row < C.num_rows(); ++row) {
        for (std::size_t col = 0; col < C.num_cols(); ++col) {
            real_type temp{ 0.0 };
            for (std::size_t k = 0; k < B.num_cols(); ++k) {
                temp = std::fma(A[col * C.num_cols_padded() + k], B(row, k), temp);
            }
            C(row, col) = alpha * temp + beta * C(row, col);
        }
    }
}
template void gemm(const float, const std::vector<float> &, const plssvm::soa_matrix<float> &, const float, plssvm::soa_matrix<float> &);
template void gemm(const double, const std::vector<double> &, const plssvm::soa_matrix<double> &, const double, plssvm::soa_matrix<double> &);

template <typename real_type>
plssvm::soa_matrix<real_type> calculate_w(const plssvm::aos_matrix<real_type> &weights, const plssvm::soa_matrix<real_type> &support_vectors) {
    PLSSVM_ASSERT(support_vectors.num_rows() == weights.num_cols(), "Sizes mismatch!: {} != {}", support_vectors.num_rows(), weights.num_cols());

    plssvm::soa_matrix<real_type> result{ weights.num_rows(), support_vectors.num_cols(), plssvm::PADDING_SIZE, plssvm::PADDING_SIZE };
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
[[nodiscard]] plssvm::aos_matrix<real_type> predict_values(const plssvm::parameter &params, const plssvm::soa_matrix<real_type> &w, const plssvm::aos_matrix<real_type> &weights, const std::vector<real_type> &rho, const plssvm::soa_matrix<real_type> &support_vectors, const plssvm::soa_matrix<real_type> &predict_points) {
    PLSSVM_ASSERT(w.empty() || w.num_rows() == weights.num_rows(), "Sizes mismatch!: {} != {}", w.num_rows(), weights.num_rows());
    PLSSVM_ASSERT(w.empty() || w.num_cols() == support_vectors.num_cols(), "Sizes mismatch!: {} != {}", w.num_cols(), support_vectors.num_cols());
    PLSSVM_ASSERT(weights.num_rows() == rho.size(), "Sizes mismatch!: {} != {}", weights.num_rows(), rho.size());
    PLSSVM_ASSERT(weights.num_cols() == support_vectors.num_rows(), "Sizes mismatch!: {} != {}", weights.num_cols(), support_vectors.num_rows());
    PLSSVM_ASSERT(support_vectors.num_cols() == predict_points.num_cols(), "Sizes mismatch!: {} != {}", support_vectors.num_cols(), predict_points.num_cols());

    const std::size_t num_classes = weights.num_rows();
    const std::size_t num_predict_points = predict_points.num_rows();
    const std::size_t num_sv = support_vectors.num_rows();
    const std::size_t num_features = predict_points.num_cols();

    plssvm::aos_matrix<real_type> result{ num_predict_points, num_classes, plssvm::PADDING_SIZE, plssvm::PADDING_SIZE };

    switch (params.kernel_type) {
        case plssvm::kernel_function_type::linear: {
            for (std::size_t c = 0; c < num_classes; ++c) {
                for (std::size_t i = 0; i < num_predict_points; ++i) {
                    real_type temp{ 0.0 };
                    for (std::size_t f = 0; f < num_features; ++f) {
                        temp = std::fma(w(c, f), predict_points(i, f), temp);
                    }
                    result(i, c) = temp - rho[c];
                }
            }
        } break;
        case plssvm::kernel_function_type::polynomial: {
            for (std::size_t c = 0; c < num_classes; ++c) {
                for (std::size_t i = 0; i < num_predict_points; ++i) {
                    for (std::size_t j = 0; j < num_sv; ++j) {
                        real_type temp{ 0.0 };
                        for (std::size_t f = 0; f < num_features; ++f) {
                            temp = std::fma(support_vectors(j, f), predict_points(i, f), temp);
                        }
                        temp = std::fma(static_cast<real_type>(params.gamma.value()), temp, static_cast<real_type>(params.coef0.value()));
                        temp = weights(c, j) * static_cast<real_type>(std::pow(temp, params.degree.value()));
                        if (j == 0) {
                            temp -= rho[c];
                        }
                        result(i, c) += temp;
                    }
                }
            }
        } break;
        case plssvm::kernel_function_type::rbf: {
            for (std::size_t c = 0; c < num_classes; ++c) {
                for (std::size_t i = 0; i < num_predict_points; ++i) {
                    for (std::size_t j = 0; j < num_sv; ++j) {
                        real_type temp{ 0.0 };
                        for (std::size_t f = 0; f < num_features; ++f) {
                            const real_type d = support_vectors(j, f) - predict_points(i, f);
                            temp = std::fma(d, d, temp);
                        }
                        temp = weights(c, j) * static_cast<real_type>(std::exp(static_cast<real_type>(-params.gamma.value()) * temp));
                        if (j == 0) {
                            temp -= rho[c];
                        }
                        result(i, c) += temp;
                    }
                }
            }
        } break;
    }

    return result;
}

template plssvm::aos_matrix<float> predict_values(const plssvm::parameter &, const plssvm::soa_matrix<float> &, const plssvm::aos_matrix<float> &, const std::vector<float> &, const plssvm::soa_matrix<float> &, const plssvm::soa_matrix<float> &);
template plssvm::aos_matrix<double> predict_values(const plssvm::parameter &, const plssvm::soa_matrix<double> &, const plssvm::aos_matrix<double> &, const std::vector<double> &, const plssvm::soa_matrix<double> &, const plssvm::soa_matrix<double> &);

}  // namespace compare
