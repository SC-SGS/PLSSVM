/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Functions for explicitly assembling the kernel matrix using the OpenMP backend.
 */

#ifndef PLSSVM_BACKENDS_OPENMP_KERNEL_CG_EXPLICIT_KERNEL_MATRIX_ASSEMBLY_HPP_
#define PLSSVM_BACKENDS_OPENMP_KERNEL_CG_EXPLICIT_KERNEL_MATRIX_ASSEMBLY_HPP_
#pragma once

#include "plssvm/backends/OpenMP/kernel/kernel_functions.hpp"  // plssvm::openmp::detail::{feature_reduce, apply_kernel_function}
#include "plssvm/constants.hpp"                                // plssvm::real_type, plssvm::THREAD_BLOCK_SIZE, plssvm::INTERNAL_BLOCK_SIZE, plssvm::PADDING_SIZE
#include "plssvm/detail/assert.hpp"                            // PLSSVM_ASSERT
#include "plssvm/kernel_function_types.hpp"                    // plssvm::kernel_function_type
#include "plssvm/matrix.hpp"                                   // plssvm::aos_matrix

#include <array>    // std::array
#include <cmath>    // std::ceil
#include <cstddef>  // std::size_t
#include <vector>   // std::vector

namespace plssvm::openmp::detail {

/**
 * @brief Assemble the kernel matrix using the @p kernel function.
 * @tparam kernel the compile-time kernel function to use
 * @tparam Args the types of the potential additional arguments for the @p kernel function
 * @param[in] q the `q` vector
 * @param[out] kernel_matrix the resulting kernel matrix
 * @param[in] data the data matrix
 * @param[in] QA_cost he bottom right matrix entry multiplied by cost
 * @param[in] cost 1 / the cost parameter in the C-SVM
 * @param[in] kernel_function_parameter the potential additional arguments for the @p kernel function
 */
template <kernel_function_type kernel, typename... Args>
void device_kernel_assembly(const std::vector<real_type> &q, std::vector<real_type> &kernel_matrix, const soa_matrix<real_type> &data, const real_type QA_cost, const real_type cost, Args... kernel_function_parameter) {
    PLSSVM_ASSERT(q.size() == data.num_rows() - 1, "Sizes mismatch!: {} != {}", q.size(), data.num_rows() - 1);
    PLSSVM_ASSERT(kernel_matrix.size() == (q.size() + PADDING_SIZE) * (q.size() + PADDING_SIZE + 1) / 2, "Sizes mismatch (SYMM)!: {} != {}", kernel_matrix.size(), (q.size() + PADDING_SIZE) * (q.size() + PADDING_SIZE + 1) / 2);
    PLSSVM_ASSERT(cost != real_type{ 0.0 }, "cost must not be 0.0 since it is 1 / plssvm::cost!");

    // calculate constants
    const std::size_t dept = q.size();
    const auto blocked_dept = static_cast<std::size_t>(std::ceil(static_cast<real_type>(dept) / INTERNAL_BLOCK_SIZE));
    const std::size_t num_features = data.num_cols();

    // cast all values to 64-bit unsigned long long to prevent potential 32-bit overflows
    const auto INTERNAL_BLOCK_SIZE_uz = static_cast<std::size_t>(INTERNAL_BLOCK_SIZE);
    const auto THREAD_BLOCK_SIZE_uz = static_cast<std::size_t>(THREAD_BLOCK_SIZE);
    const auto PADDING_SIZE_uz = static_cast<std::size_t>(PADDING_SIZE);

#pragma omp parallel for collapse(2) schedule(dynamic)
    for (std::size_t row = 0; row < blocked_dept; row += THREAD_BLOCK_SIZE_uz) {
        for (std::size_t col = 0; col < blocked_dept; col += THREAD_BLOCK_SIZE_uz) {
            // perform operations on the current block
            for (std::size_t row_block = 0; row_block < THREAD_BLOCK_SIZE_uz; ++row_block) {
                for (std::size_t col_block = 0; col_block < THREAD_BLOCK_SIZE_uz; ++col_block) {
                    // calculate the indices used in the current thread
                    const std::size_t row_idx = (row + row_block) * INTERNAL_BLOCK_SIZE_uz;
                    const std::size_t col_idx = (col + col_block) * INTERNAL_BLOCK_SIZE_uz;

                    // only calculate the upper triangular matrix
                    if (row_idx >= col_idx) {
                        // create a thread private array used for internal caching
                        std::array<std::array<real_type, INTERNAL_BLOCK_SIZE>, INTERNAL_BLOCK_SIZE> temp{};

                        // iterate over all features
                        for (std::size_t dim = 0; dim < num_features; ++dim) {
                            // perform the feature reduction calculation
                            for (unsigned internal_row = 0; internal_row < INTERNAL_BLOCK_SIZE; ++internal_row) {
                                for (unsigned internal_col = 0; internal_col < INTERNAL_BLOCK_SIZE; ++internal_col) {
                                    const std::size_t global_row = row_idx + static_cast<std::size_t>(internal_row);
                                    const std::size_t global_col = col_idx + static_cast<std::size_t>(internal_col);

                                    temp[internal_row][internal_col] += detail::feature_reduce<kernel>(data(global_row, dim), data(global_col, dim));
                                }
                            }
                        }

                        // apply the remaining part of the kernel function and store the value in the output kernel matrix
                        for (unsigned internal_row = 0; internal_row < INTERNAL_BLOCK_SIZE; ++internal_row) {
                            for (unsigned internal_col = 0; internal_col < INTERNAL_BLOCK_SIZE; ++internal_col) {
                                // calculate the indices to access the kernel matrix (the part stored on the current device)
                                const std::size_t global_row = row_idx + static_cast<std::size_t>(internal_row);
                                const std::size_t global_col = col_idx + static_cast<std::size_t>(internal_col);

                                // be sure to not perform out of bounds accesses for the kernel matrix (only using the upper triangular matrix)
                                if (global_row < dept && global_col < dept && global_row >= global_col) {
                                    real_type temp_ij = temp[internal_row][internal_col];
                                    temp_ij = detail::apply_kernel_function<kernel>(temp_ij, kernel_function_parameter...) + QA_cost - q[global_row] - q[global_col];
                                    // apply the cost on the diagonal
                                    if (global_row == global_col) {
                                        temp_ij += cost;
                                    }
                                    // update the kernel matrix
                                    kernel_matrix[global_col * (dept + PADDING_SIZE_uz) + global_row - global_col * (global_col + std::size_t{ 1 }) / std::size_t{ 2 }] = temp_ij;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

}  // namespace plssvm::openmp::detail

#endif  // PLSSVM_BACKENDS_OPENMP_KERNEL_CG_EXPLICIT_KERNEL_MATRIX_ASSEMBLY_HPP_
