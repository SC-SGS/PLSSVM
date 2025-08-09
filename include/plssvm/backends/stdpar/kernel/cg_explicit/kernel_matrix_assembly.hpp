/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Functions for explicitly assembling the kernel matrix using the stdpar backend.
 */

#ifndef PLSSVM_BACKENDS_STDPAR_KERNEL_CG_EXPLICIT_KERNEL_MATRIX_ASSEMBLY_HPP_
#define PLSSVM_BACKENDS_STDPAR_KERNEL_CG_EXPLICIT_KERNEL_MATRIX_ASSEMBLY_HPP_
#pragma once

#include "plssvm/backends/stdpar/kernel/kernel_functions.hpp"  // plssvm::stdpar::detail::{feature_reduce, apply_kernel_function}
#include "plssvm/constants.hpp"                                // plssvm::{real_type, INTERNAL_BLOCK_SIZE, PADDING_SIZE}
#include "plssvm/detail/assert.hpp"                            // PLSSVM_ASSERT
#include "plssvm/kernel_function_types.hpp"                    // plssvm::kernel_function_type
#include "plssvm/matrix.hpp"                                   // plssvm::soa_matrix
#include "plssvm/target_platforms.hpp"                         // plssvm::target_platform

#include <algorithm>  // std::for_each
#include <array>      // std::array
#include <cmath>      // std::ceil
#include <cstddef>    // std::size_t
#include <execution>  // std::execution::par_unseq
#include <numeric>    // std::iota
#include <vector>     // std::vector

namespace plssvm::stdpar::detail {

/**
 * @brief Create the explicit kernel matrix using the @p kernel_function.
 * @tparam target the target platform
 * @tparam kernel_function the type of the used kernel function
 * @tparam Args the types of the parameters necessary for the specific kernel function
 */
template <target_platform target, kernel_function_type kernel_function, typename... Args>
struct device_kernel_assembly {
    /**
     * @brief Assemble the kernel matrix using the specified kernel function.
     * @param[out] kernel_matrix the resulting kernel matrix
     * @param[in] data the data matrix
     * @param[in] device_num_rows the number of rows the current device is responsible for
     * @param[in] device_row_offset the first row in @p data the current device is responsible for
     * @param[in] q the `q` vector
     * @param[in] QA_cost he bottom right matrix entry multiplied by cost
     * @param[in] cost 1 / the cost parameter in the C-SVM
     * @param[in] kernel_function_parameter the potential additional arguments for the kernel function
     */
    void operator()(real_type *kernel_matrix, const soa_matrix<real_type> &data, const std::size_t device_num_rows, const std::size_t device_row_offset, const std::vector<real_type> &q, const real_type QA_cost, const real_type cost, Args... kernel_function_parameter) {
        PLSSVM_ASSERT(kernel_matrix != nullptr, "The kernel matrix result pointer must be valid!");
        PLSSVM_ASSERT(q.size() == data.num_rows() - 1, "Sizes mismatch!: {} != {}", q.size(), data.num_rows() - 1);
        PLSSVM_ASSERT(q.size() >= device_num_rows, "The number of place specific rows ({}) cannot be greater the the total number of rows ({})!", device_num_rows, q.size());
        PLSSVM_ASSERT(q.size() >= device_row_offset, "The row offset ({}) cannot be greater the the total number of rows ({})!", device_row_offset, q.size());
        PLSSVM_ASSERT(cost != real_type{ 0.0 }, "cost must not be 0.0 since it is 1 / plssvm::cost!");

        // calculate constants
        const std::size_t num_rows = data.num_rows() - 1;
        const std::size_t num_features = data.num_cols();
        const auto blocked_row_range = static_cast<std::size_t>(std::ceil(static_cast<real_type>(num_rows - device_row_offset) / INTERNAL_BLOCK_SIZE));
        const auto blocked_device_num_rows = static_cast<std::size_t>(std::ceil(static_cast<real_type>(device_num_rows) / INTERNAL_BLOCK_SIZE));

        // cast all values to 64-bit unsigned long long to prevent potential 32-bit overflows
        constexpr auto INTERNAL_BLOCK_SIZE_uz = static_cast<std::size_t>(INTERNAL_BLOCK_SIZE);
        constexpr auto THREAD_BLOCK_SIZE_uz = static_cast<std::size_t>(THREAD_BLOCK_SIZE);
        constexpr auto PADDING_SIZE_uz = static_cast<std::size_t>(PADDING_SIZE);

        // define the range over which should be iterated
        std::vector<std::size_t> indices(blocked_row_range * blocked_device_num_rows);
        std::iota(indices.begin(), indices.end(), 0);

        std::for_each(std::execution::par_unseq, indices.begin(), indices.end(), [=, q_ptr = q.data(), data_ptr = data.data(), kernel_matrix_ptr = kernel_matrix](const std::size_t idx) {
            // calculate the indices used in the current thread
            const std::size_t i_idx = (idx / blocked_device_num_rows) * INTERNAL_BLOCK_SIZE_uz;
            const std::size_t j_idx = (idx % blocked_device_num_rows) * INTERNAL_BLOCK_SIZE_uz;

            // only calculate the upper triangular matrix
            if (i_idx >= j_idx) {
                // create a thread private array used for internal caching
                std::array<std::array<real_type, INTERNAL_BLOCK_SIZE>, INTERNAL_BLOCK_SIZE> temp{};

                // iterate over all features
                for (std::size_t feature_block = 0; feature_block < num_features; feature_block += THREAD_BLOCK_SIZE_uz) {
                    if constexpr (target == target_platform::cpu) {
                        // perform the feature reduction calculation, the feature is the fastest moving index
                        for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                            for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                                // calculate the indices to access the global data
                                const auto global_i_idx = device_row_offset + i_idx + static_cast<std::size_t>(internal_i);
                                const auto global_j_idx = device_row_offset + j_idx + static_cast<std::size_t>(internal_j);

                                real_type sum{ 0.0 };
                                for (std::size_t feature = 0; feature < THREAD_BLOCK_SIZE_uz; ++feature) {
                                    sum += detail::feature_reduce<kernel_function>(data_ptr[(feature_block + feature) * (num_rows + std::size_t{ 1 } + PADDING_SIZE_uz) + global_i_idx],   // SoA
                                                                                   data_ptr[(feature_block + feature) * (num_rows + std::size_t{ 1 } + PADDING_SIZE_uz) + global_j_idx]);  // SoA
                                }
                                temp[internal_i][internal_j] += sum;
                            }
                        }
                    } else {
                        // perform the feature reduction calculation, the feature is the slowest moving index
                        for (std::size_t feature = 0; feature < THREAD_BLOCK_SIZE_uz; ++feature) {
                            for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                                for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                                    // calculate the indices to access the global data
                                    const auto global_i_idx = device_row_offset + i_idx + static_cast<std::size_t>(internal_i);
                                    const auto global_j_idx = device_row_offset + j_idx + static_cast<std::size_t>(internal_j);

                                    temp[internal_i][internal_j] += detail::feature_reduce<kernel_function>(data_ptr[(feature_block + feature) * (num_rows + std::size_t{ 1 } + PADDING_SIZE_uz) + global_i_idx],   // SoA
                                                                                                            data_ptr[(feature_block + feature) * (num_rows + std::size_t{ 1 } + PADDING_SIZE_uz) + global_j_idx]);  // SoA
                                }
                            }
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
                            real_type temp_ij = temp[internal_i][internal_j];
                            // apply the final kernel function
                            temp_ij = detail::apply_kernel_function<kernel_function>(temp_ij, kernel_function_parameter...) + QA_cost - q_ptr[global_i_idx] - q_ptr[global_j_idx];
                            // apply the cost on the diagonal
                            if (global_i_idx == global_j_idx) {
                                temp_ij += cost;
                            }
                            // update the upper triangular kernel matrix
                            kernel_matrix_ptr[device_global_j_idx * (num_rows - device_row_offset + PADDING_SIZE_uz) - device_global_j_idx * (device_global_j_idx + std::size_t{ 1 }) / std::size_t{ 2 } + device_global_i_idx] = temp_ij;
                        }
                    }
                }
            }
        });
    }
};

}  // namespace plssvm::stdpar::detail

#endif  // PLSSVM_BACKENDS_STDPAR_KERNEL_CG_EXPLICIT_KERNEL_MATRIX_ASSEMBLY_HPP_
