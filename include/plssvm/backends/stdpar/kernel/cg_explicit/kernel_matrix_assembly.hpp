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

#include "plssvm/constants.hpp"              // plssvm::real_type
#include "plssvm/detail/assert.hpp"          // PLSSVM_ASSERT
#include "plssvm/kernel_function_types.hpp"  // plssvm::kernel_function_type
#include "plssvm/kernel_functions.hpp"       // plssvm::kernel_function
#include "plssvm/matrix.hpp"                 // plssvm::aos_matrix

#include <cstddef>  // std::size_t
#include <execution>
#include <vector>  // std::vector

namespace plssvm::stdpar::detail {

/**
 * @brief Assemble the kernel matrix using the @p kernel function.
 * @tparam kernel the compile-time kernel function to use
 * @tparam layout the compile-time layout type for the data matrix
 * @tparam Args the types of the potential additional arguments for the @p kernel function
 * @param[in] q the `q` vector
 * @param[out] ret the resulting kernel matrix
 * @param[in] data the data matrix
 * @param[in] QA_cost he bottom right matrix entry multiplied by cost
 * @param[in] cost 1 / the cost parameter in the C-SVM
 * @param[in] args the potential additional arguments for the @p kernel function
 */
template <kernel_function_type kernel, layout_type layout, typename... Args>
void device_kernel_assembly(const std::vector<real_type> &q, std::vector<real_type> &ret, const matrix<real_type, layout> &data, const real_type QA_cost, const real_type cost, Args... args) {
    PLSSVM_ASSERT(q.size() == data.num_rows() - 1, "Sizes mismatch!: {} != {}", q.size(), data.num_rows() - 1);
    PLSSVM_ASSERT(ret.size() == (q.size() + PADDING_SIZE) * (q.size() + PADDING_SIZE + 1) / 2, "Sizes mismatch (SYMM)!: {} != {}", ret.size(), (q.size() + PADDING_SIZE) * (q.size() + PADDING_SIZE + 1) / 2);
    PLSSVM_ASSERT(cost != real_type{ 0.0 }, "cost must not be 0.0 since it is 1 / plssvm::cost!");

    const std::size_t dept = q.size();
    const std::size_t num_features = data.num_cols();

    const std::size_t blocked_dept = (dept + PADDING_SIZE) / INTERNAL_BLOCK_SIZE;

    std::vector<std::pair<std::size_t, std::size_t>> range(blocked_dept * blocked_dept);
#pragma omp parallel for
    for (std::size_t i = 0; i < range.size(); ++i) {
        range[i] = std::make_pair(i / blocked_dept, i % blocked_dept);
    }

    std::for_each(std::execution::par_unseq, range.cbegin(), range.cend(), [=, q_ptr = q.data(), data_ptr = data.data(), ret_ptr = ret.data()](const std::pair<std::size_t, std::size_t> i) {
        const auto [row, col] = i;

        if (row >= col) {
            real_type temp[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE] = { { 0.0 } };

            for (unsigned long long dim = 0; dim < num_features; dim += FEATURE_BLOCK_SIZE) {
                // calculation
                for (unsigned block_dim = 0; block_dim < FEATURE_BLOCK_SIZE; ++block_dim) {
                    for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                        const real_type internal_i_temp = data_ptr[(dim + block_dim) * (dept + 1 + PADDING_SIZE) + row + internal_i];
                        for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                            temp[internal_i][internal_j] += internal_i_temp * data_ptr[(dim + block_dim) * (dept + 1 + PADDING_SIZE) + col + internal_j];
                        }
                    }
                }
            }

            for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                    const unsigned long long global_i = row * INTERNAL_BLOCK_SIZE + internal_i;
                    const unsigned long long global_j = col * INTERNAL_BLOCK_SIZE + internal_j;

                    if (global_i < dept && global_j < dept && global_i >= global_j) {
                        real_type temp_ij = temp[internal_i][internal_j];
                        temp_ij = temp_ij + QA_cost - q_ptr[global_i] - q_ptr[global_j];
                        if (global_i == global_j) {
                            temp_ij += cost;
                        }
                        ret_ptr[global_j * (dept + PADDING_SIZE) + global_i - global_j * (global_j + 1) / 2] = temp_ij;
                    }
                }
            }
        }
    });
}

}  // namespace plssvm::stdpar::detail

#endif  // PLSSVM_BACKENDS_STDPAR_KERNEL_CG_EXPLICIT_KERNEL_MATRIX_ASSEMBLY_HPP_
