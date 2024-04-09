/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Functions for explicitly performing a BLAS GEMM like matrix-matrix multiplication using the stdpar backend.
 */

#ifndef PLSSVM_BACKENDS_STDPAR_KERNEL_CG_EXPLICIT_BLAS_HPP_
#define PLSSVM_BACKENDS_STDPAR_KERNEL_CG_EXPLICIT_BLAS_HPP_
#pragma once

#include "plssvm/constants.hpp"      // plssvm::{real_type, INTERNAL_BLOCK_SIZE, PADDING_SIZE}
#include "plssvm/detail/assert.hpp"  // PLSSVM_ASSERT
#include "plssvm/matrix.hpp"         // plssvm::soa_matrix
#include "plssvm/shape.hpp"          // plssvm::shape

#include <array>      // std::array
#include <cstddef>    // std::size_t
#include <execution>  // std::execution::par_unseq
#include <vector>     // std::vector

namespace plssvm::stdpar::detail {

/**
 * @brief Perform an explicit BLAS SYMM operation: `C = alpha * A * B + beta * C` where @p A is a symmetric matrix (memory optimized), @p B and @p C are matrices, and @p alpha and @p beta are scalars.
 * @param[in] num_rows the number of rows in @p A and @p C
 * @param[in] num_rhs the number of columns in @p B and @p C
 * @param[in] alpha the scalar alpha value
 * @param[in] A the matrix @p A
 * @param[in] B the matrix @p B
 * @param[in] beta the scalar beta value
 * @param[in,out] C the matrix @p C, also used as result matrix
 */
inline void device_kernel_symm(const unsigned long long num_rows, const unsigned long long num_rhs, const real_type alpha, const std::vector<real_type> &A, const soa_matrix<real_type> &B, const real_type beta, soa_matrix<real_type> &C) {
    PLSSVM_ASSERT(A.size() == (num_rows + PADDING_SIZE) * (num_rows + PADDING_SIZE + 1) / 2, "A matrix sizes mismatch!: {} != {}", A.size(), (num_rows + PADDING_SIZE) * (num_rows + PADDING_SIZE + 1) / 2);
    PLSSVM_ASSERT(B.shape() == (plssvm::shape{ num_rows, num_rhs }), "B matrix sizes mismatch!: {} != [{}, {}]", B.shape(), num_rows, num_rhs);
    PLSSVM_ASSERT(C.shape() == (plssvm::shape{ num_rows, num_rhs }), "C matrix sizes mismatch!: {} != [{}, {}]", C.shape(), num_rows, num_rhs);

    const std::size_t blocked_num_rhs = (num_rhs + PADDING_SIZE) / INTERNAL_BLOCK_SIZE;
    const std::size_t blocked_num_rows = (num_rows + PADDING_SIZE) / INTERNAL_BLOCK_SIZE;

    std::vector<std::pair<std::size_t, std::size_t>> range(blocked_num_rhs * blocked_num_rows);
#pragma omp parallel for
    for (std::size_t i = 0; i < range.size(); ++i) {
        range[i] = std::make_pair(i / blocked_num_rows, i % blocked_num_rows);
    }

    std::for_each(std::execution::par_unseq, range.cbegin(), range.cend(), [=, A_ptr = A.data(), B_ptr = B.data(), C_ptr = C.data()](const std::pair<std::size_t, std::size_t> idx_2d) {
        const auto [i, j] = idx_2d;

        real_type temp[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE] = { { 0.0 } };

        for (unsigned long long dim = 0; dim < num_rows; ++dim) {
            // calculation
            for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                    const unsigned long long global_i = i * INTERNAL_BLOCK_SIZE + internal_i;
                    const unsigned long long global_j = j * INTERNAL_BLOCK_SIZE + internal_j;

                    real_type A_val = 0.0;
                    if (dim < global_j) {
                        A_val = A_ptr[dim * (num_rows + PADDING_SIZE) + global_j - dim * (dim + 1) / 2];
                    } else {
                        A_val = A_ptr[global_j * (num_rows + PADDING_SIZE) + dim - global_j * (global_j + 1) / 2];
                    }
                    temp[internal_i][internal_j] += A_val * B_ptr[dim * (num_rhs + PADDING_SIZE) + global_i];
                }
            }
        }

        for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
            for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                const unsigned long long global_i = i * INTERNAL_BLOCK_SIZE + internal_i;
                const unsigned long long global_j = j * INTERNAL_BLOCK_SIZE + internal_j;

                if (global_i < num_rhs && global_j < num_rows) {
                    C_ptr[global_j * (num_rhs + PADDING_SIZE) + global_i] = alpha * temp[internal_i][internal_j] + beta * C_ptr[global_j * (num_rhs + PADDING_SIZE) + global_i];
                }
            }
        }
    });
}

}  // namespace plssvm::stdpar::detail

#endif  // PLSSVM_BACKENDS_STDPAR_KERNEL_CG_EXPLICIT_BLAS_HPP_
