/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Functions for explicitly performing a BLAS GEMM like matrix-matrix multiplication using the CUDA backend.
 */

#ifndef PLSSVM_BACKENDS_OPENMP_KERNEL_CG_EXPLICIT_BLAS_HPP_
#define PLSSVM_BACKENDS_OPENMP_KERNEL_CG_EXPLICIT_BLAS_HPP_
#pragma once

#include "plssvm/constants.hpp"      // plssvm::real_type
#include "plssvm/detail/assert.hpp"  // PLSSVM_ASSERT
#include "plssvm/matrix.hpp"         // plssvm::aos_matrix
#include "plssvm/shape.hpp"          // plssvm::shape

#include <array>    // std::array
#include <cmath>    // std::ceil
#include <cstddef>  // std::size_t
#include <vector>   // std::vector

namespace plssvm::openmp::detail {

/**
 * @brief Perform an explicit BLAS SYMM operation: `C = alpha * A * B + beta * C` where @p A is a `m x k` symmetric matrix (memory optimized), @p B is a `k x n` matrix, @p C is a `m x n` matrix, and @p alpha and @p beta are scalars.
 * @param[in] num_rows the number of rows and columns in @p A
 * @param[in] num_rhs the number of rows in @p B and @p C
 * @param[in] alpha the scalar alpha value
 * @param[in] A the matrix @p A
 * @param[in] B the matrix @p B
 * @param[in] beta the scalar beta value
 * @param[in,out] C the matrix @p C, also used as result matrix
 */
inline void device_kernel_symm(const std::size_t num_rows, const std::size_t num_rhs, const real_type alpha, const std::vector<real_type> &A, const soa_matrix<real_type> &B, const real_type beta, soa_matrix<real_type> &C) {
    // compute: C = alpha * A * B + beta * C with A in m x k, B in n x k, and C in n x m, alpha, beta as scalar
    PLSSVM_ASSERT(A.size() == (num_rows + PADDING_SIZE) * (num_rows + PADDING_SIZE + 1) / 2, "A matrix sizes mismatch!: {} != {}", A.size(), (num_rows + PADDING_SIZE) * (num_rows + PADDING_SIZE + 1) / 2);
    PLSSVM_ASSERT(B.shape() == (plssvm::shape{ num_rhs, num_rows }), "B matrix sizes mismatch!: {} != [{}, {}]", B.shape(), num_rhs, num_rows);
    PLSSVM_ASSERT(C.shape() == (plssvm::shape{ num_rhs, num_rows }), "C matrix sizes mismatch!: {} != [{}, {}]", C.shape(), num_rhs, num_rows);

    // calculate constants
    const auto blocked_num_rhs = static_cast<std::size_t>(std::ceil(static_cast<real_type>(num_rhs) / INTERNAL_BLOCK_SIZE));
    const auto blocked_num_rows = static_cast<std::size_t>(std::ceil(static_cast<real_type>(num_rows) / INTERNAL_BLOCK_SIZE));

    // cast all values to 64-bit unsigned long long to prevent potential 32-bit overflows
    const auto INTERNAL_BLOCK_SIZE_uz = static_cast<std::size_t>(INTERNAL_BLOCK_SIZE);
    const auto THREAD_BLOCK_SIZE_uz = static_cast<std::size_t>(THREAD_BLOCK_SIZE);
    const auto PADDING_SIZE_uz = static_cast<std::size_t>(PADDING_SIZE);

#pragma omp parallel for collapse(2)
    for (std::size_t rhs = 0; rhs < blocked_num_rhs; rhs += THREAD_BLOCK_SIZE_uz) {
        for (std::size_t row = 0; row < blocked_num_rows; row += THREAD_BLOCK_SIZE_uz) {
            // perform operations on the current block
            for (std::size_t rhs_block = 0; rhs_block < THREAD_BLOCK_SIZE_uz; ++rhs_block) {
                for (std::size_t row_block = 0; row_block < THREAD_BLOCK_SIZE_uz; ++row_block) {
                    // calculate the indices used in the current thread
                    const std::size_t rhs_idx = (rhs + rhs_block) * INTERNAL_BLOCK_SIZE_uz;
                    const std::size_t row_idx = (row + row_block) * INTERNAL_BLOCK_SIZE_uz;

                    // create a thread private array used for internal caching
                    std::array<std::array<real_type, INTERNAL_BLOCK_SIZE>, INTERNAL_BLOCK_SIZE> temp{};

                    // iterate over all features
                    for (std::size_t dim = 0; dim < num_rows; ++dim) {
                        // perform the dot product calculation
                        for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                            for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                                const std::size_t global_i = rhs_idx + static_cast<std::size_t>(internal_i);
                                const std::size_t global_j = row_idx + static_cast<std::size_t>(internal_j);

                                real_type A_val = 0.0;
                                // determine on which side of the diagonal we are located
                                if (dim < global_j) {
                                    A_val = A[dim * (num_rows + PADDING_SIZE_uz) + global_j - dim * (dim + std::size_t{ 1 }) / std::size_t{ 2 }];
                                } else {
                                    A_val = A[global_j * (num_rows + PADDING_SIZE_uz) + dim - global_j * (global_j + std::size_t{ 1 }) / std::size_t{ 2 }];
                                }
                                temp[internal_i][internal_j] += A_val * B(global_i, dim);
                            }
                        }
                    }

                    // apply the (partial) BLAS operation and update C
                    for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                        for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                            const std::size_t global_i = rhs_idx + static_cast<std::size_t>(internal_i);
                            const std::size_t global_j = row_idx + static_cast<std::size_t>(internal_j);

                            // be sure to not perform out of bounds accesses
                            if (global_i < num_rhs && global_j < num_rows) {
                                C(global_i, global_j) = alpha * temp[internal_i][internal_j] + beta * C(global_i, global_j);
                            }
                        }
                    }
                }
            }
        }
    }
}

}  // namespace plssvm::openmp::detail

#endif  // PLSSVM_BACKENDS_OPENMP_KERNEL_CG_EXPLICIT_BLAS_HPP_
