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

#ifndef PLSSVM_BACKENDS_STDPAR_KERNEL_CG_IMPLICIT_KERNEL_MATRIX_ASSEMBLY_BLAS_HPP_
#define PLSSVM_BACKENDS_STDPAR_KERNEL_CG_IMPLICIT_KERNEL_MATRIX_ASSEMBLY_BLAS_HPP_
#pragma once

#include "plssvm/backends/stdpar/detail/utility.hpp"           // plssvm::stdpar::detail::atomic_ref
#include "plssvm/backends/stdpar/kernel/kernel_functions.hpp"  // plssvm::stdpar::detail::{feature_reduce, apply_kernel_function}
#include "plssvm/constants.hpp"                                // plssvm::real_type
#include "plssvm/detail/assert.hpp"                            // PLSSVM_ASSERT
#include "plssvm/detail/operators.hpp"                         // overloaded arithmetic operations for a plssvm::matrix
#include "plssvm/kernel_function_types.hpp"                    // plssvm::kernel_function_type
#include "plssvm/kernel_functions.hpp"                         // plssvm::kernel_function
#include "plssvm/matrix.hpp"                                   // aos_matrix

#include <algorithm>  // std::for_each
#include <array>      // std::array
#include <cmath>      // std::ceil
#include <cstddef>    // std::size_t
#include <execution>  // std::execution::par_unseq
#include <utility>    // std::pair, std::make_pair
#include <vector>     // std::vector

namespace plssvm::stdpar::detail {

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

    // alpha * A * B + beta * C
    C *= beta;

    // calculate constants
    const std::size_t dept = q.size();
    const auto blocked_dept = static_cast<std::size_t>(std::ceil(static_cast<real_type>(dept) / INTERNAL_BLOCK_SIZE));
    const std::size_t num_features = data.num_cols();
    const std::size_t num_classes = B.num_rows();

    // cast all values to 64-bit unsigned long long to prevent potential 32-bit overflows
    const auto INTERNAL_BLOCK_SIZE_uz = static_cast<std::size_t>(INTERNAL_BLOCK_SIZE);
    const auto PADDING_SIZE_uz = static_cast<std::size_t>(PADDING_SIZE);

    // calculate indices over which we parallelize
    std::vector<std::pair<std::size_t, std::size_t>> range(blocked_dept * (blocked_dept + 1) / 2);
#pragma omp parallel for
    for (std::size_t i = 0; i < blocked_dept * blocked_dept; ++i) {
        const std::size_t row = i / blocked_dept;
        const std::size_t col = i % blocked_dept;
        // only create valid row <-> col index pairs
        if (row >= col) {
            range[col * blocked_dept + row - col * (col + 1) / 2] = std::make_pair(row, col);
        }
    }

    std::for_each(std::execution::par_unseq, range.begin(), range.end(), [=, q_ptr = q.data(), data_ptr = data.data(), B_ptr = B.data(), C_ptr = C.data()](const std::pair<std::size_t, std::size_t> idx) {
        // calculate the indices used in the current thread
        const auto [row, col] = idx;
        const std::size_t row_idx = row * INTERNAL_BLOCK_SIZE_uz;
        const std::size_t col_idx = col * INTERNAL_BLOCK_SIZE_uz;

        // only calculate the upper triangular matrix -> done be only iterating over valid row <-> col pairs
        // create a thread private array used for internal caching
        std::array<std::array<real_type, INTERNAL_BLOCK_SIZE>, INTERNAL_BLOCK_SIZE> temp{};

        // iterate over all features
        for (std::size_t dim = 0; dim < num_features; ++dim) {
            for (unsigned internal_row = 0; internal_row < INTERNAL_BLOCK_SIZE; ++internal_row) {
                for (unsigned internal_col = 0; internal_col < INTERNAL_BLOCK_SIZE; ++internal_col) {
                    const std::size_t global_row = row_idx + static_cast<std::size_t>(internal_row);
                    const std::size_t global_col = col_idx + static_cast<std::size_t>(internal_col);

                    temp[internal_row][internal_col] += detail::feature_reduce<kernel>(data_ptr[dim * (dept + 1 + PADDING_SIZE_uz) + global_row], data_ptr[dim * (dept + 1 + PADDING_SIZE_uz) + global_col]);
                }
            }
        }

        // apply the remaining part of the kernel function and store the value in the output kernel matrix
        for (unsigned internal_row = 0; internal_row < INTERNAL_BLOCK_SIZE; ++internal_row) {
            for (unsigned internal_col = 0; internal_col < INTERNAL_BLOCK_SIZE; ++internal_col) {
                const std::size_t global_row = row_idx + static_cast<std::size_t>(internal_row);
                const std::size_t global_col = col_idx + static_cast<std::size_t>(internal_col);

                // be sure to not perform out of bounds accesses for the kernel matrix (only using the upper triangular matrix)
                if (global_row < dept && global_col < dept && global_row >= global_col) {
                    real_type temp_ij = temp[internal_row][internal_col];
                    temp_ij = detail::apply_kernel_function<kernel>(temp_ij, kernel_function_parameter...) + QA_cost - q_ptr[global_row] - q_ptr[global_col];
                    // apply the cost on the diagonal
                    if (global_row == global_col) {
                        temp_ij += cost;
                        // calculate the values of alpha * A * B
                        for (std::size_t class_idx = 0; class_idx < num_classes; ++class_idx) {
                            atomic_ref<real_type>{ C_ptr[global_row * (num_classes + PADDING_SIZE_uz) + class_idx] } += alpha * temp_ij * B_ptr[global_row * (num_classes + PADDING_SIZE_uz) + class_idx];
                        }
                    } else {
                        // calculate the values of alpha * A * B
                        for (std::size_t class_idx = 0; class_idx < num_classes; ++class_idx) {
                            atomic_ref<real_type>{ C_ptr[global_row * (num_classes + PADDING_SIZE_uz) + class_idx] } += alpha * temp_ij * B_ptr[global_col * (num_classes + PADDING_SIZE_uz) + class_idx];
                            // symmetry
                            atomic_ref<real_type>{ C_ptr[global_col * (num_classes + PADDING_SIZE_uz) + class_idx] } += alpha * temp_ij * B_ptr[global_row * (num_classes + PADDING_SIZE_uz) + class_idx];
                        }
                    }
                }
            }
        }
    });
}

}  // namespace plssvm::stdpar::detail

#endif  // PLSSVM_BACKENDS_STDPAR_KERNEL_CG_IMPLICIT_KERNEL_MATRIX_ASSEMBLY_BLAS_HPP_
