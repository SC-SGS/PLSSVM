/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Functions for performing a matrix-matrix multiplication using an implicit kernel matrix.
 */

#ifndef PLSSVM_BACKENDS_OPENMP_KERNEL_CG_IMPLICIT_KERNEL_MATRIX_ASSEMBLY_BLAS_HPP_
#define PLSSVM_BACKENDS_OPENMP_KERNEL_CG_IMPLICIT_KERNEL_MATRIX_ASSEMBLY_BLAS_HPP_
#pragma once

#include "plssvm/constants.hpp"              // plssvm::real_type, plssvm::OPENMP_BLOCK_SIZE
#include "plssvm/detail/assert.hpp"          // PLSSVM_ASSERT
#include "plssvm/detail/operators.hpp"       // overloaded arithmetic operations for a plssvm::matrix
#include "plssvm/kernel_function_types.hpp"  // plssvm::kernel_function_type
#include "plssvm/matrix.hpp"                 // aos_matrix

#include <array>    // std::array
#include <cstddef>  // std::size_t
#include <vector>   // std::vector

namespace plssvm::openmp::detail {

/**
 * @brief Perform an implicit BLAS SYMM-like operation: `C = alpha * A * B + C` where `A` is the implicitly calculated kernel matrix using the @p kernel function (never actually stored, reducing the amount of needed global memory), @p B and @p C are matrices, and @p alpha is a scalar.
 * @tparam kernel the compile-time kernel function to use
 * @tparam Args the types of the potential additional arguments for the @p kernel function
 * @param[in] alpha the scalar alpha value
 * @param[in] q the `q` vector
 * @param[in] data the data matrix
 * @param[in] QA_cost he bottom right matrix entry multiplied by cost
 * @param[in] cost 1 / the cost parameter in the C-SVM
 * @param[in] B the matrix @p B
 * @param[in] beta the beta alpha value
 * @param[in,out] C the matrix @p C
 * @param[in] kernel_function_parameter the potential additional arguments for the @p kernel function
 */
template <kernel_function_type kernel, typename... Args>
inline void device_kernel_assembly_symm(const real_type alpha, const std::vector<real_type> &q, const soa_matrix<real_type> &data, const real_type QA_cost, const real_type cost, const soa_matrix<real_type> &B, const real_type beta, soa_matrix<real_type> &C, Args... kernel_function_parameter) {
    PLSSVM_ASSERT(q.size() == data.num_rows() - 1, "Sizes mismatch!: {} != {}", q.size(), data.num_rows() - 1);
    PLSSVM_ASSERT(cost != real_type{ 0.0 }, "cost must not be 0.0 since it is 1 / plssvm::cost!");
    PLSSVM_ASSERT(B.shape() == C.shape(), "The matrices B and C must have the same shape!");
    PLSSVM_ASSERT(B.num_cols() == q.size(), "The number of columns in B ({}) must be the same as the values in q ({})!", B.num_cols(), q.size());

    using namespace operators;

    const std::size_t dept = q.size();
    const std::size_t blocked_dept = (dept + PADDING_SIZE) / INTERNAL_BLOCK_SIZE;
    const std::size_t num_features = data.num_cols();
    const std::size_t num_classes = B.num_rows();

    // alpha * A * B + beta * C
    C *= beta;

#pragma omp parallel for collapse(2) schedule(dynamic)
    for (std::size_t row = 0; row < blocked_dept; row += THREAD_BLOCK_SIZE) {
        for (std::size_t col = 0; col < blocked_dept; col += THREAD_BLOCK_SIZE) {
            // perform operations on the current block
            for (std::size_t row_block = 0; row_block < THREAD_BLOCK_SIZE; ++row_block) {
                for (std::size_t col_block = 0; col_block < THREAD_BLOCK_SIZE; ++col_block) {
                    const std::size_t row_idx = (row + row_block) * INTERNAL_BLOCK_SIZE;
                    const std::size_t col_idx = (col + col_block) * INTERNAL_BLOCK_SIZE;

                    // use symmetry and only calculate upper triangular matrix
                    if (row_idx >= col_idx) {
                        std::array<std::array<real_type, INTERNAL_BLOCK_SIZE>, INTERNAL_BLOCK_SIZE> temp{};

                        for (std::size_t dim = 0; dim < num_features; ++dim) {
                            for (unsigned internal_row = 0; internal_row < INTERNAL_BLOCK_SIZE; ++internal_row) {
                                for (unsigned internal_col = 0; internal_col < INTERNAL_BLOCK_SIZE; ++internal_col) {
                                    temp[internal_row][internal_col] += detail::feature_reduce<kernel>(data(row_idx + internal_row, dim), data(col_idx + internal_col, dim));
                                }
                            }
                        }

                        for (unsigned internal_row = 0; internal_row < INTERNAL_BLOCK_SIZE; ++internal_row) {
                            for (unsigned internal_col = 0; internal_col < INTERNAL_BLOCK_SIZE; ++internal_col) {
                                const std::size_t global_row = row_idx + internal_row;
                                const std::size_t global_col = col_idx + internal_col;

                                if (global_row < dept && global_col < dept && global_row >= global_col) {
                                    real_type temp_ij = temp[internal_row][internal_col];
                                    temp_ij = detail::apply_kernel_function<kernel>(temp_ij, kernel_function_parameter...) + QA_cost - q[global_row] - q[global_col];
                                    if (global_row == global_col) {
                                        temp_ij += cost;
                                        // calculate the values of alpha * A * B
                                        for (std::size_t class_idx = 0; class_idx < num_classes; ++class_idx) {
#pragma omp atomic
                                            C(class_idx, global_row) += alpha * temp_ij * B(class_idx, global_row);
                                        }
                                    } else {
                                        // calculate the values of alpha * A * B
                                        for (std::size_t class_idx = 0; class_idx < num_classes; ++class_idx) {
#pragma omp atomic
                                            C(class_idx, global_row) += alpha * temp_ij * B(class_idx, global_col);
// symmetry
#pragma omp atomic
                                            C(class_idx, global_col) += alpha * temp_ij * B(class_idx, global_row);
                                        }
                                    }
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

#endif  // PLSSVM_BACKENDS_OPENMP_KERNEL_CG_IMPLICIT_KERNEL_MATRIX_ASSEMBLY_BLAS_HPP_
