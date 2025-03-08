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

#include <algorithm>  // std::for_each
#include <array>      // std::array
#include <cmath>      // std::ceil
#include <cstddef>    // std::size_t
#include <execution>  // std::execution::par_unseq
#include <numeric>    // std::iota
#include <vector>     // std::vector

namespace plssvm::stdpar::detail {

/**
 * @brief Perform an explicit BLAS SYMM operation: `C = alpha * A * B + beta * C` where @p A is a symmetric matrix (memory optimized), @p B and @p C are matrices, and @p alpha and @p beta are scalars.
 * @param[in] num_rows the number of rows in @p A and @p C
 * @param[in] num_rhs the number of columns in @p B and @p C
 * @param[in] device_specific_num_rows the number of rows the current device is responsible for
 * @param[in] row_offset the first row in @p data the current device is responsible for
 * @param[in] alpha the scalar alpha value
 * @param[in] A the matrix @p A
 * @param[in] B the matrix @p B
 * @param[in] beta the scalar beta value
 * @param[in,out] C the matrix @p C, also used as result matrix
 */
inline void device_kernel_symm(const std::size_t num_rows, const std::size_t num_rhs, const std::size_t device_specific_num_rows, const std::size_t row_offset, const real_type alpha, const std::vector<real_type> &A, const soa_matrix<real_type> &B, const real_type beta, soa_matrix<real_type> &C) {
    PLSSVM_ASSERT(!A.empty(), "A matrix may not be empty!");
    PLSSVM_ASSERT(B.shape() == (plssvm::shape{ num_rhs, num_rows }), "B matrix sizes mismatch!: {} != [{}, {}]", B.shape(), num_rhs, num_rows);
    PLSSVM_ASSERT(C.shape() == (plssvm::shape{ num_rhs, num_rows }), "C matrix sizes mismatch!: {} != [{}, {}]", C.shape(), num_rhs, num_rows);
    PLSSVM_ASSERT(num_rows >= device_specific_num_rows, "The number of place specific rows ({}) cannot be greater the the total number of rows ({})!", device_specific_num_rows, num_rows);
    PLSSVM_ASSERT(num_rows >= row_offset, "The row offset ({}) cannot be greater the the total number of rows ({})!", row_offset, num_rows);

    // calculate constants
    const auto blocked_num_rhs = static_cast<std::size_t>(std::ceil(static_cast<real_type>(num_rhs) / INTERNAL_BLOCK_SIZE));
    const auto blocked_device_specific_num_rows = static_cast<std::size_t>(std::ceil(static_cast<real_type>(device_specific_num_rows) / INTERNAL_BLOCK_SIZE));

    // cast all values to 64-bit unsigned long long to prevent potential 32-bit overflows
    const auto INTERNAL_BLOCK_SIZE_uz = static_cast<std::size_t>(INTERNAL_BLOCK_SIZE);
    const auto PADDING_SIZE_uz = static_cast<std::size_t>(PADDING_SIZE);

    // define range over which should be iterated
    std::vector<std::size_t> range(blocked_num_rhs * blocked_device_specific_num_rows);
    std::iota(range.begin(), range.end(), 0);

    std::for_each(std::execution::par_unseq, range.begin(), range.end(), [=, A_ptr = A.data(), B_ptr = B.data(), C_ptr = C.data()](const std::size_t idx) {
        // calculate the indices used in the current thread
        const std::size_t rhs = idx / blocked_device_specific_num_rows;
        const std::size_t row = idx % blocked_device_specific_num_rows;

        const std::size_t rhs_idx = rhs * INTERNAL_BLOCK_SIZE_uz;
        const std::size_t row_idx = row * INTERNAL_BLOCK_SIZE_uz;

        // create a thread private array used for internal caching
        std::array<std::array<real_type, INTERNAL_BLOCK_SIZE>, INTERNAL_BLOCK_SIZE> temp{};

        // iterate over all features
        for (std::size_t dim = 0; dim < (num_rows - row_offset); ++dim) {
            // perform the dot product calculation
            for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                    const std::size_t global_rhs = rhs_idx + static_cast<std::size_t>(internal_i);
                    const std::size_t global_row = row_idx + static_cast<std::size_t>(internal_j);

                    real_type A_val = 0.0;
                    // determine on which side of the diagonal we are located
                    if (dim < global_row) {
                        A_val = A_ptr[dim * (num_rows - row_offset + PADDING_SIZE_uz) + global_row - dim * (dim + std::size_t{ 1 }) / std::size_t{ 2 }];
                    } else {
                        A_val = A_ptr[global_row * (num_rows - row_offset + PADDING_SIZE_uz) + dim - global_row * (global_row + std::size_t{ 1 }) / std::size_t{ 2 }];
                    }
                    temp[internal_i][internal_j] += A_val * B_ptr[(dim + row_offset) * (num_rhs + PADDING_SIZE_uz) + global_rhs];
                }
            }
        }

        // apply the (partial) BLAS operation and update C
        for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
            for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                const std::size_t global_rhs = rhs_idx + static_cast<std::size_t>(internal_i);
                const std::size_t device_global_row = row_idx + static_cast<std::size_t>(internal_j);
                const std::size_t global_row = row_offset + row_idx + static_cast<std::size_t>(internal_j);

                // be sure to not perform out of bounds accesses
                if (global_rhs < num_rhs && device_global_row < device_specific_num_rows) {
                    C_ptr[global_row * (num_rhs + PADDING_SIZE_uz) + global_rhs] = alpha * temp[internal_i][internal_j] + beta * C_ptr[global_row * (num_rhs + PADDING_SIZE_uz) + global_rhs];
                }
            }
        }
    });
}

/**
 * @brief Perform an explicit BLAS SYMM operation: `C = alpha * A * B + beta * C` where @p A is a `m x k` symmetric matrix (memory optimized), @p B is a `k x n` matrix, @p C is a `m x n` matrix, and @p alpha and @p beta are scalars.
 * @param[in] num_rows the number of rows in @p A and @p C
 * @param[in] num_rhs the number of columns in @p B and @p C
 * @param[in] num_mirror_rows the number of rows to mirror down
 * @param[in] device_specific_num_rows the number of rows in @p A and number of rows in @p B; thr rows in @p A are potentially distributed across multiple devices
 * @param[in] row_offset the first row this device is responsible for
 * @param[in] alpha the scalar alpha value
 * @param[in] A the matrix @p A
 * @param[in] B the matrix @p B
 * @param[in] beta the scalar beta value
 * @param[in,out] C the matrix @p C, also used as result matrix
 */
inline void device_kernel_symm_mirror(const std::size_t num_rows, const std::size_t num_rhs, const std::size_t num_mirror_rows, const std::size_t device_specific_num_rows, const std::size_t row_offset, const real_type alpha, const std::vector<real_type> &A, const soa_matrix<real_type> &B, const real_type beta, soa_matrix<real_type> &C) {
    // compute: C = alpha * A * B + beta * C with A in m x k, B in n x k, and C in n x m, alpha, beta as scalar
    PLSSVM_ASSERT(!A.empty(), "A matrix may not be empty!");
    PLSSVM_ASSERT(B.shape() == (plssvm::shape{ num_rhs, num_rows }), "B matrix sizes mismatch!: {} != [{}, {}]", B.shape(), num_rhs, num_rows);
    PLSSVM_ASSERT(C.shape() == (plssvm::shape{ num_rhs, num_rows }), "C matrix sizes mismatch!: {} != [{}, {}]", C.shape(), num_rhs, num_rows);
    PLSSVM_ASSERT(num_rows >= device_specific_num_rows, "The number of place specific rows ({}) cannot be greater the the total number of rows ({})!", device_specific_num_rows, num_rows);
    PLSSVM_ASSERT(num_rows >= num_mirror_rows, "The number of mirror rows ({}) cannot be greater the the total number of rows ({})!", num_mirror_rows, num_rows);
    PLSSVM_ASSERT(num_rows >= row_offset, "The row offset ({}) cannot be greater the the total number of rows ({})!", row_offset, num_rows);

    // calculate constants
    const auto blocked_num_rhs = static_cast<std::size_t>(std::ceil(static_cast<real_type>(num_rhs) / INTERNAL_BLOCK_SIZE));
    const auto blocked_num_mirror_rows = static_cast<std::size_t>(std::ceil(static_cast<real_type>(num_mirror_rows) / INTERNAL_BLOCK_SIZE));

    // cast all values to 64-bit unsigned long long to prevent potential 32-bit overflows
    const auto INTERNAL_BLOCK_SIZE_uz = static_cast<std::size_t>(INTERNAL_BLOCK_SIZE);
    const auto PADDING_SIZE_uz = static_cast<std::size_t>(PADDING_SIZE);

    // define range over which should be iterated
    std::vector<std::size_t> range(blocked_num_rhs * blocked_num_mirror_rows);  // define range over which should be iterated
    std::iota(range.begin(), range.end(), 0);

    std::for_each(std::execution::par_unseq, range.begin(), range.end(), [=, A_ptr = A.data(), B_ptr = B.data(), C_ptr = C.data()](const std::size_t idx) {
        // calculate the indices used in the current thread
        const std::size_t rhs = idx / blocked_num_mirror_rows;
        const std::size_t row = idx % blocked_num_mirror_rows;

        const std::size_t rhs_idx = rhs * INTERNAL_BLOCK_SIZE_uz;
        const std::size_t row_idx = row * INTERNAL_BLOCK_SIZE_uz;

        // create a thread private array used for internal caching
        std::array<std::array<real_type, INTERNAL_BLOCK_SIZE>, INTERNAL_BLOCK_SIZE> temp{};

        // iterate over all features
        for (std::size_t dim = 0; dim < device_specific_num_rows; ++dim) {
            // perform the dot product calculation
            for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                    const std::size_t global_rhs = rhs_idx + static_cast<std::size_t>(internal_i);
                    const std::size_t global_row = row_idx + static_cast<std::size_t>(internal_j);

                    const real_type A_val = A_ptr[dim * (num_rows - row_offset + PADDING_SIZE_uz) - (dim - std::size_t{ 1 }) * dim / std::size_t{ 2 } + device_specific_num_rows - dim + global_row];
                    temp[internal_i][internal_j] += A_val * B_ptr[(dim + row_offset) * (num_rhs + PADDING_SIZE_uz) + global_rhs];
                }
            }
        }

        // apply the (partial) BLAS operation and update C
        for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
            for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                const std::size_t global_rhs = rhs_idx + static_cast<std::size_t>(internal_i);
                const std::size_t partial_global_row = row_idx + static_cast<std::size_t>(internal_j);
                const std::size_t global_row = row_offset + device_specific_num_rows + row_idx + static_cast<std::size_t>(internal_j);

                // be sure to not perform out of bounds accesses
                if (global_rhs < num_rhs && partial_global_row < num_mirror_rows) {
                    C_ptr[global_row * (num_rhs + PADDING_SIZE_uz) + global_rhs] = alpha * temp[internal_i][internal_j] + beta * C[global_row * (num_rhs + PADDING_SIZE_uz) + global_rhs];
                }
            }
        }
    });
}

}  // namespace plssvm::stdpar::detail

#endif  // PLSSVM_BACKENDS_STDPAR_KERNEL_CG_EXPLICIT_BLAS_HPP_
