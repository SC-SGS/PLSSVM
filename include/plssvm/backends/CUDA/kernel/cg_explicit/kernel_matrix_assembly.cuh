/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Functions for explicitly assembling the kernel matrix using the CUDA backend.
 */

#ifndef PLSSVM_BACKENDS_CUDA_KERNEL_CG_EXPLICIT_KERNEL_MATRIX_ASSEMBLY_CUH_
#define PLSSVM_BACKENDS_CUDA_KERNEL_CG_EXPLICIT_KERNEL_MATRIX_ASSEMBLY_CUH_
#pragma once

#include "plssvm/backends/CUDA/kernel/kernel_functions.cuh"  // plssvm::cuda::detail::{feature_reduce, apply_kernel_function}
#include "plssvm/constants.hpp"                              // plssvm::{real_type, THREAD_BLOCK_SIZE, INTERNAL_BLOCK_SIZE, PADDING_SIZE}
#include "plssvm/kernel_function_types.hpp"                  // plssvm::kernel_function_type

#include <cstddef>  // std::size_t

namespace plssvm::cuda::detail {

/**
 * @brief Create the explicit kernel matrix using the @p kernel_function.
 * @tparam kernel_function the type of the used kernel function
 * @tparam Args the types of the parameters necessary for the specific kernel function
 * @param[out] kernel_matrix the calculated kernel matrix
 * @param[in] data the data points to calculate the kernel matrix from
 * @param[in] num_rows the total number of data points (= total number of rows)
 * @param[in] device_num_rows the number of rows the current device is responsible for
 * @param[in] device_row_offset the first row in @p data_d the current device is responsible for
 * @param[in] num_features the number of features per data point
 * @param[in] q the vector used in the dimensional reduction
 * @param[in] QA_cost the scalar used in the dimensional reduction
 * @param[in] cost the cost factor the diagonal is scaled with
 * @param[in] grid_x_offset the offset in x-dimension into the data points if more than one execution grid has to be used
 * @param[in] grid_y_offset the offset in y-dimension into the data points if more than one execution grid has to be used
 * @param[in] kernel_function_parameter the parameters necessary to apply the @p kernel_function
 */
template <kernel_function_type kernel_function, typename... Args>
__global__ void device_kernel_assembly(real_type *kernel_matrix, const real_type *data, const std::size_t num_rows, const std::size_t device_num_rows, const std::size_t device_row_offset, const std::size_t num_features, const real_type *q, const real_type QA_cost, const real_type cost, const std::size_t grid_x_offset, const std::size_t grid_y_offset, Args... kernel_function_parameter) {
    // cast all values to 64-bit std::size_t to prevent potential 32-bit overflows
    constexpr auto INTERNAL_BLOCK_SIZE_uz = static_cast<std::size_t>(INTERNAL_BLOCK_SIZE);
    constexpr auto THREAD_BLOCK_SIZE_uz = static_cast<std::size_t>(THREAD_BLOCK_SIZE);
    constexpr auto PADDING_SIZE_uz = static_cast<std::size_t>(PADDING_SIZE);

    const auto threadIdx_x = static_cast<std::size_t>(threadIdx.x);                // current thread in block x-dimension
    const auto threadIdx_y = static_cast<std::size_t>(threadIdx.y);                // current thread in block y-dimension
    const auto blockDim_x = static_cast<std::size_t>(blockDim.x);                  // number of threads in block x-dimension
    const auto blockDim_y = static_cast<std::size_t>(blockDim.y);                  // number of threads in block y-dimension
    const auto blockIdx_x = static_cast<std::size_t>(blockIdx.x) + grid_x_offset;  // current block in grid x-dimension + offsets if the grid size is too large
    const auto blockIdx_y = static_cast<std::size_t>(blockIdx.y) + grid_y_offset;  // current block in grid y-dimension + offsets if the grid size is too large

    // create two shared memory arrays used for caching
    __shared__ real_type data_i_cache[THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];
    __shared__ real_type data_j_cache[THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];

    // only calculate the upper triangular matrix -> can't use threadIdx since all threads in a warp must progress further
    if (blockIdx_x >= blockIdx_y) {
        // create a thread private array used for internal caching
        real_type temp[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE]{};

        {
            // calculate the indices used in the current thread, pays attention to coalesced memory accesses
            const auto i_idx_linear = blockIdx_x * blockDim_x * INTERNAL_BLOCK_SIZE_uz + threadIdx_x;  // num_rows - device_row_offset
            const auto j_idx_linear = blockIdx_y * blockDim_y * INTERNAL_BLOCK_SIZE_uz + threadIdx_x;  // device_num_rows

            // iterate over all features using blocking to be able to cache them for faster memory accesses
            for (std::size_t dim = 0; dim < num_features; dim += THREAD_BLOCK_SIZE_uz) {
                // load data into shared memory
                for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                    // calculate the indices to access the global data, pays attention to coalesced memory accesses
                    const auto global_i_idx_linear = device_row_offset + i_idx_linear + static_cast<std::size_t>(internal) * THREAD_BLOCK_SIZE_uz;
                    const auto global_j_idx_linear = device_row_offset + j_idx_linear + static_cast<std::size_t>(internal) * THREAD_BLOCK_SIZE_uz;

                    // store the values in the shared memory
                    data_i_cache[threadIdx.y][internal * THREAD_BLOCK_SIZE + threadIdx.x] = data[(dim + threadIdx_y) * (num_rows + std::size_t{ 1 } + PADDING_SIZE_uz) + global_i_idx_linear];  // SoA
                    data_j_cache[threadIdx.y][internal * THREAD_BLOCK_SIZE + threadIdx.x] = data[(dim + threadIdx_y) * (num_rows + std::size_t{ 1 } + PADDING_SIZE_uz) + global_j_idx_linear];  // SoA
                }
                __syncthreads();  // wait until all threads loaded their part of the data

                // perform the feature reduction calculation
                for (unsigned block_dim = 0; block_dim < THREAD_BLOCK_SIZE; ++block_dim) {
                    for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                        for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                            temp[internal_i][internal_j] += detail::feature_reduce<kernel_function>(data_i_cache[block_dim][threadIdx.x * INTERNAL_BLOCK_SIZE + internal_i],
                                                                                                    data_j_cache[block_dim][threadIdx.y * INTERNAL_BLOCK_SIZE + internal_j]);
                        }
                    }
                }
                __syncthreads();  // wait until all threads performed their part of the calculations
            }
        }

        // calculate the indices used in the current thread
        const auto i_idx = (blockIdx_x * blockDim_x + threadIdx_x) * INTERNAL_BLOCK_SIZE_uz;  // num_rows - device_row_offset
        const auto j_idx = (blockIdx_y * blockDim_y + threadIdx_y) * INTERNAL_BLOCK_SIZE_uz;  // device_num_rows

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
                    temp_ij = detail::apply_kernel_function<kernel_function>(temp_ij, kernel_function_parameter...) + QA_cost - q[global_i_idx] - q[global_j_idx];
                    // apply the cost on the diagonal
                    if (global_i_idx == global_j_idx) {
                        temp_ij += cost;
                    }
                    // update the upper triangular kernel matrix
                    kernel_matrix[device_global_j_idx * (num_rows - device_row_offset + PADDING_SIZE_uz) - device_global_j_idx * (device_global_j_idx + std::size_t{ 1 }) / std::size_t{ 2 } + device_global_i_idx] = temp_ij;
                }
            }
        }
    }
}

}  // namespace plssvm::cuda::detail

#endif  // PLSSVM_BACKENDS_CUDA_KERNEL_CG_EXPLICIT_KERNEL_MATRIX_ASSEMBLY_CUH_
