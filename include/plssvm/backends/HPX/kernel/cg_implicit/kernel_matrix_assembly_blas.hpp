/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @author Alexander Strack
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Functions for performing a matrix-matrix multiplication using an implicit kernel matrix.
 */

#ifndef PLSSVM_BACKENDS_HPX_KERNEL_CG_IMPLICIT_KERNEL_MATRIX_ASSEMBLY_BLAS_HPP_
#define PLSSVM_BACKENDS_HPX_KERNEL_CG_IMPLICIT_KERNEL_MATRIX_ASSEMBLY_BLAS_HPP_
#pragma once

#include "plssvm/backends/HPX/detail/utility.hpp"           // plssvm::hpx::detail::atomic_ref
#include "plssvm/backends/HPX/kernel/kernel_functions.hpp"  // plssvm::hpx::detail::{feature_reduce, apply_kernel_function}
#include "plssvm/constants.hpp"                             // plssvm::real_type
#include "plssvm/detail/assert.hpp"                         // PLSSVM_ASSERT
#include "plssvm/kernel_function_types.hpp"                 // plssvm::kernel_function_type
#include "plssvm/kernel_functions.hpp"                      // plssvm::kernel_function
#include "plssvm/matrix.hpp"                                // aos_matrix

#include "hpx/execution.hpp"                               // hpx::execution::par_unseq
#include "hpx/parallel/segmented_algorithms/for_each.hpp"  // hpx::for_each

#include <array>    // std::array
#include <cmath>    // std::ceil
#include <cstddef>  // std::size_t, std::sqrt
#include <numeric>  // std::iota
#include <vector>   // std::vector

namespace plssvm::hpx::detail {

/**
 * @brief Perform an implicit BLAS SYMM-like operation: `C = alpha * A * B + C` where `A` is the implicitly calculated kernel matrix using the @p kernel_function (never actually stored, reducing the amount of needed global memory), @p B and @p C are matrices, and @p alpha is a scalar.
 * @tparam kernel_function the compile-time kernel function to use
 * @tparam Args the types of the potential additional arguments for the @p kernel_function function
 * @param[in] alpha the scalar alpha value
 * @param[in] q the `q` vector
 * @param[in] data the data matrix
 * @param[in] device_num_rows the number of rows the current device is responsible for
 * @param[in] device_row_offset the first row in @p data the current device is responsible for
 * @param[in] QA_cost he bottom right matrix entry multiplied by cost
 * @param[in] cost 1 / the cost parameter in the C-SVM
 * @param[in] B the matrix @p B
 * @param[in,out] C the matrix @p C
 * @param[in] kernel_function_parameter the potential additional arguments for the @p kernel_function function
 */
template <kernel_function_type kernel_function, typename... Args>
inline void device_kernel_assembly_symm(const real_type alpha, const std::vector<real_type> &q, const soa_matrix<real_type> &data, const std::size_t device_num_rows, const std::size_t device_row_offset, const real_type QA_cost, const real_type cost, const soa_matrix<real_type> &B, soa_matrix<real_type> &C, Args... kernel_function_parameter) {
    PLSSVM_ASSERT(q.size() == data.num_rows() - 1, "Sizes mismatch!: {} != {}", q.size(), data.num_rows() - 1);
    PLSSVM_ASSERT(q.size() >= device_num_rows, "The number of place specific rows ({}) cannot be greater the the total number of rows ({})!", device_num_rows, q.size());
    PLSSVM_ASSERT(q.size() >= device_row_offset, "The row offset ({}) cannot be greater the the total number of rows ({})!", device_row_offset, q.size());
    PLSSVM_ASSERT(cost != real_type{ 0.0 }, "cost must not be 0.0 since it is 1 / plssvm::cost!");
    PLSSVM_ASSERT(B.shape() == C.shape(), "The matrices B and C must have the same shape!");
    PLSSVM_ASSERT(B.num_cols() == q.size(), "The number of columns in B ({}) must be the same as the values in q ({})!", B.num_cols(), q.size());

    // calculate constants
    const std::size_t num_rows = data.num_rows() - 1;
    const std::size_t num_features = data.num_cols();
    const std::size_t num_classes = B.num_rows();
    const auto blocked_row_range = static_cast<std::size_t>(std::ceil(static_cast<real_type>(num_rows - device_row_offset) / INTERNAL_BLOCK_SIZE));
    const auto blocked_device_num_rows = static_cast<std::size_t>(std::ceil(static_cast<real_type>(device_num_rows) / INTERNAL_BLOCK_SIZE));

    // cast all values to 64-bit unsigned long long to prevent potential 32-bit overflows
    constexpr auto INTERNAL_BLOCK_SIZE_uz = static_cast<std::size_t>(INTERNAL_BLOCK_SIZE);
    constexpr auto THREAD_BLOCK_SIZE_uz = static_cast<std::size_t>(THREAD_BLOCK_SIZE);

    // define the range over which should be iterated
    std::vector<std::size_t> indices(blocked_row_range * blocked_device_num_rows);
    std::iota(indices.begin(), indices.end(), 0);

    ::hpx::for_each(::hpx::execution::par_unseq, indices.cbegin(), indices.cend(), [&](const std::size_t idx) {
        // calculate the indices used in the current thread
        const std::size_t i_idx = (idx % blocked_device_num_rows) * INTERNAL_BLOCK_SIZE_uz;
        const std::size_t j_idx = (idx / blocked_device_num_rows) * INTERNAL_BLOCK_SIZE_uz;

        // only calculate the upper triangular matrix
        if (i_idx >= j_idx) {
            // create a thread private array used for internal caching
            std::array<std::array<real_type, INTERNAL_BLOCK_SIZE>, INTERNAL_BLOCK_SIZE> temp{};

            //*************************************************************************//
            //                   inplace kernel matrix construction                    //
            //*************************************************************************//
            // iterate over all features using blocking
            for (std::size_t feature_block = 0; feature_block < num_features; feature_block += THREAD_BLOCK_SIZE_uz) {
                for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                    for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                        // calculate the indices to access the global data
                        const auto global_i_idx = device_row_offset + i_idx + static_cast<std::size_t>(internal_i);
                        const auto global_j_idx = device_row_offset + j_idx + static_cast<std::size_t>(internal_j);

                        real_type sum{ 0.0 };
                        for (std::size_t feature = 0; feature < THREAD_BLOCK_SIZE_uz; ++feature) {
                            sum += detail::feature_reduce<kernel_function>(data(global_i_idx, feature_block + feature), data(global_j_idx, feature_block + feature));
                        }
                        temp[internal_j][internal_i] += sum;
                    }
                }
            }

            // apply the remaining part of the kernel function and store the value in the output kernel matrix
            for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                    // calculate the indices to access the global data and the data with respect to the current device
                    const auto device_global_i_idx = i_idx + static_cast<std::size_t>(internal_i);
                    const auto global_i_idx = device_row_offset + device_global_i_idx;
                    const auto device_global_j_idx = j_idx + static_cast<std::size_t>(internal_j);
                    const auto global_j_idx = device_row_offset + device_global_j_idx;

                    // be sure to not perform out-of-bounds accesses (only using the upper triangular matrix)
                    if (device_global_i_idx < (num_rows - device_row_offset) && device_global_j_idx < device_num_rows && global_i_idx >= global_j_idx) {
                        // apply the final kernel function
                        temp[internal_j][internal_i] = detail::apply_kernel_function<kernel_function>(temp[internal_j][internal_i], kernel_function_parameter...) + QA_cost - q[global_i_idx] - q[global_j_idx];
                        // apply the cost on the diagonal
                        if (global_i_idx == global_j_idx) {
                            temp[internal_j][internal_i] += cost;
                        }
                    } else {
                        // be sure to set the value to zero otherwise
                        temp[internal_j][internal_i] = real_type{ 0.0 };
                    }
                }
            }

            //*************************************************************************//
            //                     calculate C += alpha * temp * B                     //
            //*************************************************************************//
            for (std::size_t class_block = 0; class_block < num_classes; class_block += THREAD_BLOCK_SIZE_uz) {
                for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                    for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                        // calculate the indices to access the global data
                        const auto global_i_idx = device_row_offset + i_idx + static_cast<std::size_t>(internal_i);
                        const auto global_j_idx = device_row_offset + j_idx + static_cast<std::size_t>(internal_j);

                        if (global_i_idx == global_j_idx) {
                            // only apply once to the diagonal
                            for (std::size_t class_idx = 0; class_idx < THREAD_BLOCK_SIZE_uz; ++class_idx) {
                                atomic_ref<real_type>{ C(class_block + class_idx, global_i_idx) } += alpha * temp[internal_j][internal_i] * B(class_block + class_idx, global_i_idx);
                            }
                        } else {
                            // apply it for the upper and lower triangular matrix
                            for (std::size_t class_idx = 0; class_idx < THREAD_BLOCK_SIZE_uz; ++class_idx) {
                                atomic_ref<real_type>{ C(class_block + class_idx, global_i_idx) } += alpha * temp[internal_j][internal_i] * B(class_block + class_idx, global_j_idx);
                                // symmetry
                                atomic_ref<real_type>{ C(class_block + class_idx, global_j_idx) } += alpha * temp[internal_j][internal_i] * B(class_block + class_idx, global_i_idx);
                            }
                        }
                    }
                }
            }
        }
    });
}

}  // namespace plssvm::hpx::detail

#endif  // PLSSVM_BACKENDS_HPX_KERNEL_CG_IMPLICIT_KERNEL_MATRIX_ASSEMBLY_BLAS_HPP_
