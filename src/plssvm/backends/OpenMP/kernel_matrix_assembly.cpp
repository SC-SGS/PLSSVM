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

    // can't use default(none) due to the parameter pack Args (args)
    #pragma omp parallel for collapse(2)
    for (kernel_index_type row = 0; row < dept; ++row) {
        for (kernel_index_type col = 0; col < dept; ++col) {
            ret[row][col] = kernel_function<kernel>(data[row], data[col], std::forward<Args>(args)...) + QA_cost - q[row] - q[col];
            if (row == col) {
                ret[row][row] += cost;
            }
        }
    }

    //    #pragma omp parallel for collapse(2) schedule(dynamic)
    //    for (kernel_index_type i = 0; i < dept; i += OPENMP_BLOCK_SIZE) {
    //        for (kernel_index_type j = 0; j < dept; j += OPENMP_BLOCK_SIZE) {
    //            for (kernel_index_type ii = 0; ii < OPENMP_BLOCK_SIZE && ii + i < dept; ++ii) {
    //                real_type ret_iii = 0.0;
    //                for (kernel_index_type jj = 0; jj < OPENMP_BLOCK_SIZE && jj + j < dept; ++jj) {
    //                    if (ii + i >= jj + j) {
    //                        const real_type temp = kernel_function<kernel>(data[ii + i], data[jj + j], std::forward<Args>(args)...) + QA_cost - q[ii + i] - q[jj + j];
    //                        if (ii + i == jj + j) {
    //                            ret_iii += temp + cost;
    //                        } else {
    //                            ret_iii += temp;
    //                            #pragma omp atomic
    //                            ret[ii + i][jj + j] += temp;
    //                        }
    //                    }
    //                }
    //                #pragma omp atomic
    //                ret[ii + i][jj + j] += ret_iii;
    //            }
    //        }
    //    }
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
