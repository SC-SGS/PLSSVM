/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/OpenMP/kernel_matrix_assembly.hpp"

#include "plssvm/constants.hpp"              // plssvm::kernel_index_type, plssvm::OPENMP_BLOCK_SIZE
#include "plssvm/detail/assert.hpp"          // PLSSVM_ASSERT
#include "plssvm/kernel_function_types.hpp"  // plssvm::kernel_function_type, plssvm::kernel_function

#include <utility>                           // std::forward
#include <vector>                            // std::vector

namespace plssvm::openmp {

namespace detail {

template <kernel_function_type kernel, typename real_type, typename... Args>
void kernel_matrix_assembly(const std::vector<real_type> &q, std::vector<std::vector<real_type>> &ret, const std::vector<std::vector<real_type>> &data, const real_type QA_cost, const real_type cost, Args &&...args) {
    PLSSVM_ASSERT(q.size() == data.size() - 1, "Sizes mismatch!: {} != {}", q.size(), data.size() - 1);
    PLSSVM_ASSERT(q.size() == ret.size(), "Sizes mismatch!: {} != {}", q.size(), ret.size());
    PLSSVM_ASSERT(cost != real_type{ 0.0 }, "cost must not be 0.0 since it is 1 / plssvm::cost!");

    const auto dept = static_cast<kernel_index_type>(q.size());
    const auto num_features = static_cast<kernel_index_type>(data.front().size());

    #pragma omp parallel for collapse(2)
    for (kernel_index_type row = 0; row < dept; row += OPENMP_BLOCK_SIZE) {
        for (kernel_index_type col = 0; col < dept; col += OPENMP_BLOCK_SIZE) {
            for (kernel_index_type row_block = 0; row_block < OPENMP_BLOCK_SIZE && row + row_block < dept; ++row_block) {
                for (kernel_index_type col_block = 0; col_block < OPENMP_BLOCK_SIZE && col + col_block < dept; ++col_block) {

                    real_type temp{ 0.0 };
                    #pragma omp simd reduction(+ : temp)
                    for (kernel_index_type dim = 0; dim < num_features; ++dim) {
                        if constexpr (kernel == plssvm::kernel_function_type::rbf) {
                            const auto diff = data[row + row_block][dim] - data[col + col_block][dim];
                            temp += diff * diff;
                        } else {
                            temp += data[row + row_block][dim] * data[col + col_block][dim];
                        }
                    }
                    if constexpr (kernel == plssvm::kernel_function_type::linear) {
                        static_assert(sizeof...(args) == 0, "Illegal number of additional parameters! Must be 0.");
                        ret[row + row_block][col + col_block] = temp + QA_cost - q[row + row_block] - q[col + col_block];
                    } else if constexpr (kernel == plssvm::kernel_function_type::polynomial) {
                        static_assert(sizeof...(args) == 3, "Illegal number of additional parameters! Must be 3.");
                        const auto degree = static_cast<real_type>(::plssvm::detail::get<0>(args...));
                        const auto gamma = static_cast<real_type>(::plssvm::detail::get<1>(args...));
                        const auto coef0 = static_cast<real_type>(::plssvm::detail::get<2>(args...));
                        ret[row + row_block][col + col_block] = std::pow(gamma * temp + coef0, degree) + QA_cost - q[row + row_block] - q[col + col_block];
                    } else if constexpr (kernel == kernel_function_type::rbf) {
                        static_assert(sizeof...(args) == 1, "Illegal number of additional parameters! Must be 1.");
                        const auto gamma = static_cast<real_type>(::plssvm::detail::get<0>(args...));
                        ret[row + row_block][col + col_block] = std::exp(-gamma * temp) + QA_cost - q[row + row_block] - q[col + col_block];
                    } else {
                        static_assert(::plssvm::detail::always_false_v<real_type>, "Unknown kernel type!");
                    }

                }
            }
        }
    }

    // apply cost to diagonal
    #pragma omp parallel for
    for (kernel_index_type i = 0; i < dept; ++i) {
        ret[i][i] += cost;
    }
}

}  // namespace detail

template <typename real_type>
void linear_kernel_matrix_assembly(const std::vector<real_type> &q, std::vector<std::vector<real_type>> &ret, const std::vector<std::vector<real_type>> &data, const real_type QA_cost, const real_type cost) {
    detail::kernel_matrix_assembly<kernel_function_type::linear>(q, ret, data, QA_cost, cost);
}
template void linear_kernel_matrix_assembly(const std::vector<float> &, std::vector<std::vector<float>> &, const std::vector<std::vector<float>> &, float, float);
template void linear_kernel_matrix_assembly(const std::vector<double> &, std::vector<std::vector<double>> &, const std::vector<std::vector<double>> &, double, double);

template <typename real_type>
void polynomial_kernel_matrix_assembly(const std::vector<real_type> &q, std::vector<std::vector<real_type>> &ret, const std::vector<std::vector<real_type>> &data, const real_type QA_cost, const real_type cost, const int degree, const real_type gamma, const real_type coef0) {
    PLSSVM_ASSERT(gamma > real_type{ 0.0 }, "gamma must be greater than 0, but is {}!", gamma);

    detail::kernel_matrix_assembly<kernel_function_type::polynomial>(q, ret, data, QA_cost, cost, degree, gamma, coef0);
}
template void polynomial_kernel_matrix_assembly(const std::vector<float> &, std::vector<std::vector<float>> &, const std::vector<std::vector<float>> &, float, float, int, float, float);
template void polynomial_kernel_matrix_assembly(const std::vector<double> &, std::vector<std::vector<double>> &, const std::vector<std::vector<double>> &, double, double, int, double, double);

template <typename real_type>
void rbf_kernel_matrix_assembly(const std::vector<real_type> &q, std::vector<std::vector<real_type>> &ret, const std::vector<std::vector<real_type>> &data, const real_type QA_cost, const real_type cost, const real_type gamma) {
    PLSSVM_ASSERT(gamma > real_type{ 0.0 }, "gamma must be greater than 0, but is {}!", gamma);

    detail::kernel_matrix_assembly<kernel_function_type::rbf>(q, ret, data, QA_cost, cost, gamma);
}
template void rbf_kernel_matrix_assembly(const std::vector<float> &, std::vector<std::vector<float>> &, const std::vector<std::vector<float>> &, float, float, float);
template void rbf_kernel_matrix_assembly(const std::vector<double> &, std::vector<std::vector<double>> &, const std::vector<std::vector<double>> &, double, double, double);

}  // namespace plssvm::openmp
