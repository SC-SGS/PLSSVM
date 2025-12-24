/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Functions for explicitly performing a BLAS GEMM like matrix-matrix multiplication using the OpenMP backend.
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

namespace plssvm::openmp::detail {

/**
 * @brief Perform an explicit BLAS SYMM operation: `C = alpha * A * B + beta * C` where @p A is a `m x k` symmetric matrix (memory optimized), @p B is a `k x n` matrix, @p C is a `m x n` matrix, and @p alpha and @p beta are scalars.
 * @param[in] num_rows the number of rows and columns in @p A
 * @param[in] num_rhs the number of rows in @p B and @p C
 * @param[in] device_num_rows the number of rows the current device is responsible for
 * @param[in] device_row_offset the first row in @p data the current device is responsible for
 * @param[in] alpha the scalar alpha value
 * @param[in] A the matrix @p A
 * @param[in] B the matrix @p B
 * @param[in] beta the scalar beta value
 * @param[in,out] C the matrix @p C, also used as result matrix
 */
inline void device_kernel_symm(const std::size_t num_rows, const std::size_t num_rhs, const std::size_t device_num_rows, const std::size_t device_row_offset, const real_type alpha, const real_type *A, const soa_matrix<real_type> &B, const real_type beta, soa_matrix<real_type> &C) {
    // compute: C = alpha * A * B + beta * C with A in m x k, B in n x k, and C in n x m, alpha, beta as scalar
    PLSSVM_ASSERT(A != nullptr, "The A matrix result pointer must be valid!");
    PLSSVM_ASSERT(B.shape() == (plssvm::shape{ num_rhs, num_rows }), "B matrix sizes mismatch!: {} != [{}, {}]", B.shape(), num_rhs, num_rows);
    PLSSVM_ASSERT(C.shape() == (plssvm::shape{ num_rhs, num_rows }), "C matrix sizes mismatch!: {} != [{}, {}]", C.shape(), num_rhs, num_rows);
    PLSSVM_ASSERT(num_rows >= device_num_rows, "The number of place specific rows ({}) cannot be greater the the total number of rows ({})!", device_num_rows, num_rows);
    PLSSVM_ASSERT(num_rows >= device_row_offset, "The row offset ({}) cannot be greater the the total number of rows ({})!", device_row_offset, num_rows);

    // calculate constants
    const auto blocked_num_rhs = static_cast<std::size_t>(std::ceil(static_cast<real_type>(num_rhs) / INTERNAL_BLOCK_SIZE));
    const auto blocked_device_num_rows = static_cast<std::size_t>(std::ceil(static_cast<real_type>(device_num_rows) / INTERNAL_BLOCK_SIZE));

    // cast all values to 64-bit unsigned long long to prevent potential 32-bit overflows
    constexpr auto INTERNAL_BLOCK_SIZE_uz = static_cast<std::size_t>(INTERNAL_BLOCK_SIZE);
    constexpr auto THREAD_BLOCK_SIZE_uz = static_cast<std::size_t>(THREAD_BLOCK_SIZE);
    constexpr auto PADDING_SIZE_uz = static_cast<std::size_t>(PADDING_SIZE);

#pragma omp parallel for collapse(2)
    for (std::size_t rhs_block = 0; rhs_block < blocked_num_rhs; rhs_block += THREAD_BLOCK_SIZE_uz) {
        for (std::size_t row_block = 0; row_block < blocked_device_num_rows; row_block += THREAD_BLOCK_SIZE_uz) {
            // perform operations on the current block
            for (std::size_t rhs_thread = 0; rhs_thread < THREAD_BLOCK_SIZE_uz; ++rhs_thread) {
                for (std::size_t row_thread = 0; row_thread < THREAD_BLOCK_SIZE_uz; ++row_thread) {
                    // calculate the indices used in the current thread
                    const std::size_t i_idx = (rhs_block + rhs_thread) * INTERNAL_BLOCK_SIZE_uz;  // num_rhs
                    const std::size_t j_idx = (row_block + row_thread) * INTERNAL_BLOCK_SIZE_uz;  // device_num_rows

                    // create a thread private array used for internal caching
                    std::array<std::array<real_type, INTERNAL_BLOCK_SIZE>, INTERNAL_BLOCK_SIZE> temp{};

                    // iterate over all values using blocking
                    for (std::size_t dim_block = 0; dim_block < (num_rows - device_row_offset); dim_block += THREAD_BLOCK_SIZE_uz) {
                        // perform the dot product calculation
                        for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                            for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                                // calculate the indices to access the global data
                                const auto global_i_idx = i_idx + static_cast<std::size_t>(internal_i);
                                const auto global_j_idx = j_idx + static_cast<std::size_t>(internal_j);

                                real_type sum{ 0.0 };
                                for (std::size_t dim = 0; dim < THREAD_BLOCK_SIZE_uz; ++dim) {
                                    real_type A_cache = 0.0;
                                    // determine on which side of the diagonal we are located
                                    if (dim_block + dim < global_j_idx) {
                                        A_cache = A[(dim_block + dim) * (num_rows - device_row_offset + PADDING_SIZE_uz) + global_j_idx - (dim_block + dim) * (dim_block + dim + std::size_t{ 1 }) / std::size_t{ 2 }];  // SoA, upper triangular matrix only
                                    } else {
                                        A_cache = A[global_j_idx * (num_rows - device_row_offset + PADDING_SIZE_uz) + dim_block + dim - global_j_idx * (global_j_idx + std::size_t{ 1 }) / std::size_t{ 2 }];  // SoA, upper triangular matrix only
                                    }
                                    sum += A_cache * B(global_i_idx, dim_block + dim + device_row_offset);
                                }
                                temp[internal_j][internal_i] += sum;
                            }
                        }
                    }

                    // apply the (partial) BLAS operation and update C
                    for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                        for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                            // calculate the indices to access the global data and the data with respect to the current device
                            const auto global_i_idx = i_idx + static_cast<std::size_t>(internal_i);
                            const auto device_global_j_idx = j_idx + static_cast<std::size_t>(internal_j);
                            const auto global_j_idx = device_row_offset + device_global_j_idx;

                            // be sure to not perform out-of-bounds accesses
                            if (global_i_idx < num_rhs && device_global_j_idx < device_num_rows) {
                                C(global_i_idx, global_j_idx) = alpha * temp[internal_j][internal_i] + beta * C(global_i_idx, global_j_idx);  // SoA
                            }
                        }
                    }
                }
            }
        }
    }
}

/**
 * @brief Perform an explicit BLAS SYMM operation: `C = alpha * A * B + beta * C` where @p A is a `m x k` symmetric matrix (memory optimized), @p B is a `k x n` matrix, @p C is a `m x n` matrix, and @p alpha and @p beta are scalars.
 * @param[in] num_rows the number of rows in @p A and @p C
 * @param[in] num_rhs the number of columns in @p B and @p C
 * @param[in] num_mirror_rows the number of rows to mirror down
 * @param[in] device_num_rows the number of rows in @p A and number of rows in @p B; thr rows in @p A are potentially distributed across multiple devices
 * @param[in] device_row_offset the first row this device is responsible for
 * @param[in] alpha the scalar alpha value
 * @param[in] A the matrix @p A
 * @param[in] B the matrix @p B
 * @param[in] beta the scalar beta value
 * @param[in,out] C the matrix @p C, also used as result matrix
 */
inline void device_kernel_symm_mirror(const std::size_t num_rows, const std::size_t num_rhs, const std::size_t num_mirror_rows, const std::size_t device_num_rows, const std::size_t device_row_offset, const real_type alpha, const real_type *A, const soa_matrix<real_type> &B, const real_type beta, soa_matrix<real_type> &C) {
    // compute: C = alpha * A * B + beta * C with A in m x k, B in n x k, and C in n x m, alpha, beta as scalar
    PLSSVM_ASSERT(A != nullptr, "The A matrix result pointer must be valid!");
    PLSSVM_ASSERT(B.shape() == (plssvm::shape{ num_rhs, num_rows }), "B matrix sizes mismatch!: {} != [{}, {}]", B.shape(), num_rhs, num_rows);
    PLSSVM_ASSERT(C.shape() == (plssvm::shape{ num_rhs, num_rows }), "C matrix sizes mismatch!: {} != [{}, {}]", C.shape(), num_rhs, num_rows);
    PLSSVM_ASSERT(num_rows >= device_num_rows, "The number of place specific rows ({}) cannot be greater the the total number of rows ({})!", device_num_rows, num_rows);
    PLSSVM_ASSERT(num_rows >= num_mirror_rows, "The number of mirror rows ({}) cannot be greater the the total number of rows ({})!", num_mirror_rows, num_rows);
    PLSSVM_ASSERT(num_rows >= device_row_offset, "The row offset ({}) cannot be greater the the total number of rows ({})!", device_row_offset, num_rows);

    // calculate constants
    const auto blocked_num_rhs = static_cast<std::size_t>(std::ceil(static_cast<real_type>(num_rhs) / INTERNAL_BLOCK_SIZE));
    const auto blocked_num_mirror_rows = static_cast<std::size_t>(std::ceil(static_cast<real_type>(num_mirror_rows) / INTERNAL_BLOCK_SIZE));

    // cast all values to 64-bit unsigned long long to prevent potential 32-bit overflows
    constexpr auto INTERNAL_BLOCK_SIZE_uz = static_cast<std::size_t>(INTERNAL_BLOCK_SIZE);
    constexpr auto THREAD_BLOCK_SIZE_uz = static_cast<std::size_t>(THREAD_BLOCK_SIZE);
    constexpr auto PADDING_SIZE_uz = static_cast<std::size_t>(PADDING_SIZE);

#pragma omp parallel for collapse(2)
    for (std::size_t rhs_block = 0; rhs_block < blocked_num_rhs; rhs_block += THREAD_BLOCK_SIZE_uz) {
        for (std::size_t row_block = 0; row_block < blocked_num_mirror_rows; row_block += THREAD_BLOCK_SIZE_uz) {
            // perform operations on the current block
            for (std::size_t rhs_thread = 0; rhs_thread < THREAD_BLOCK_SIZE_uz; ++rhs_thread) {
                for (std::size_t row_thread = 0; row_thread < THREAD_BLOCK_SIZE_uz; ++row_thread) {
                    // calculate the indices used in the current thread
                    const std::size_t i_idx = (rhs_block + rhs_thread) * INTERNAL_BLOCK_SIZE_uz;  // num_rhs
                    const std::size_t j_idx = (row_block + row_thread) * INTERNAL_BLOCK_SIZE_uz;  // num_mirror_rows

                    // create a thread private array used for internal caching
                    std::array<std::array<real_type, INTERNAL_BLOCK_SIZE>, INTERNAL_BLOCK_SIZE> temp{};

                    // iterate over the remaining values using blocking
                    for (std::size_t dim_block = 0; dim_block < device_num_rows; dim_block += THREAD_BLOCK_SIZE_uz) {
                        // perform the dot product calculation
                        for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                            for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                                // calculate the indices to access the global data
                                const auto global_i_idx = i_idx + static_cast<std::size_t>(internal_i);
                                const auto global_j_idx = j_idx + static_cast<std::size_t>(internal_j);

                                real_type sum{ 0.0 };
                                for (std::size_t dim = 0; dim < THREAD_BLOCK_SIZE_uz; ++dim) {
                                    sum += A[(dim_block + dim) * (num_rows - device_row_offset + PADDING_SIZE_uz) - (dim_block + dim - std::size_t{ 1 }) * (dim_block + dim) / std::size_t{ 2 } + device_num_rows - (dim_block + dim) + global_j_idx] *  // SoA, upper triangular matrix only
                                           B(global_i_idx, dim_block + dim + device_row_offset);                                                                                                                                                         // SoA
                                }
                                temp[internal_j][internal_i] += sum;
                            }
                        }
                    }

                    // apply the (remaining) BLAS operation and update C
                    for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                        for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                            // calculate the indices to access the global data and the data with respect to the current device
                            const auto global_i_idx = i_idx + static_cast<std::size_t>(internal_i);
                            const auto partial_global_j_idx = j_idx + static_cast<std::size_t>(internal_j);
                            const auto global_j_idx = device_row_offset + device_num_rows + partial_global_j_idx;

                            // be sure to not perform out-of-bounds accesses
                            if (global_i_idx < num_rhs && partial_global_j_idx < num_mirror_rows) {
                                C(global_i_idx, global_j_idx) = alpha * temp[internal_j][internal_i] + beta * C(global_i_idx, global_j_idx);  // SoA
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
