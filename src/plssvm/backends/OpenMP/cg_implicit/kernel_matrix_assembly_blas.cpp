/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/OpenMP/cg_implicit/kernel_matrix_assembly_blas.hpp"

#include "plssvm/constants.hpp"              // plssvm::real_type
#include "plssvm/detail/assert.hpp"          // PLSSVM_ASSERT
#include "plssvm/detail/operators.hpp"       // overloaded arithmetic operations for a plssvm::matrix
#include "plssvm/kernel_function_types.hpp"  // plssvm::kernel_function_type, plssvm::kernel_function
#include "plssvm/matrix.hpp"                 // aos_matrix

#include <cstddef>  // std::size_t
#include <vector>   // std::vector

namespace plssvm::openmp {

namespace detail {

template <kernel_function_type kernel, typename... Args>
void device_kernel_assembly_symm(real_type alpha, const std::vector<real_type> &q, const aos_matrix<real_type> &data, real_type QA_cost, real_type cost, const aos_matrix<real_type> &B, real_type beta, aos_matrix<real_type> &C, Args... args) {
    PLSSVM_ASSERT(q.size() == data.num_rows() - 1, "Sizes mismatch!: {} != {}", q.size(), data.num_rows() - 1);
    PLSSVM_ASSERT(cost != real_type{ 0.0 }, "cost must not be 0.0 since it is 1 / plssvm::cost!");
    PLSSVM_ASSERT(B.shape() == C.shape(), "The matrices B and C must have the same shape!");

    using namespace operators;

    const std::size_t dept = q.size();

    // alpha * A * B + beta * C
    C *= beta;

    // loop over all rows in the IMPLICIT kernel matrix
    #pragma omp parallel for schedule(dynamic)
    for (std::size_t km_row = 0; km_row < dept; ++km_row) {
        // loop over all columns in the IMPLICIT kernel matrix
        // half number of computations by exploiting symmetry
        // NOTE: diagonal NOT included!
        for (std::size_t km_col = km_row + 1; km_col < dept; ++km_col) {
            // calculate kernel matrix entry
            const real_type temp = kernel_function<kernel>(data, km_row, data, km_col, args...) + QA_cost - q[km_row] - q[km_col];

            // calculate the values of alpha * A * B
            for (std::size_t row = 0; row < B.num_rows(); ++row) {
                #pragma omp atomic
                C(row, km_row) += alpha * temp * B(row, km_col);
                // symmetry
                #pragma omp atomic
                C(row, km_col) += alpha * temp * B(row, km_row);
            }
        }
    }
    // handle diagonal
    #pragma omp parallel for
    for (std::size_t i = 0; i < dept; ++i) {
        const real_type temp = kernel_function<kernel>(data, i, data, i, args...) + cost + QA_cost - q[i] - q[i];

        // calculate the values of alpha * A * B
        for (std::size_t row = 0; row < B.num_rows(); ++row) {
            C(row, i) += alpha * temp * B(row, i);
        }
    }
}

}  // namespace detail

void device_kernel_assembly_linear_symm(real_type alpha, const std::vector<real_type> &q, const aos_matrix<real_type> &data, real_type QA_cost, real_type cost, const aos_matrix<real_type> &B, real_type beta, aos_matrix<real_type> &C) {
    detail::device_kernel_assembly_symm<kernel_function_type::linear>(alpha, q, data, QA_cost, cost, B, beta, C);
}

void device_kernel_assembly_polynomial_symm(real_type alpha, const std::vector<real_type> &q, const aos_matrix<real_type> &data, real_type QA_cost, real_type cost, const int degree, const real_type gamma, const real_type coef0, const aos_matrix<real_type> &B, real_type beta, aos_matrix<real_type> &C) {
    PLSSVM_ASSERT(gamma > real_type{ 0.0 }, "gamma must be greater than 0, but is {}!", gamma);

    detail::device_kernel_assembly_symm<kernel_function_type::polynomial>(alpha, q, data, QA_cost, cost, B, beta, C, degree, gamma, coef0);
}

void device_kernel_assembly_rbf_symm(real_type alpha, const std::vector<real_type> &q, const aos_matrix<real_type> &data, real_type QA_cost, real_type cost, const real_type gamma, const aos_matrix<real_type> &B, real_type beta, aos_matrix<real_type> &C) {
    PLSSVM_ASSERT(gamma > real_type{ 0.0 }, "gamma must be greater than 0, but is {}!", gamma);

    detail::device_kernel_assembly_symm<kernel_function_type::rbf>(alpha, q, data, QA_cost, cost, B, beta, C, gamma);
}

}  // namespace plssvm::openmp