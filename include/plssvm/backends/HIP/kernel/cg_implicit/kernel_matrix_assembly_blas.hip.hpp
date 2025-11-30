/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Functions for explicitly assembling the kernel matrix using the HIP backend.
 */

#ifndef PLSSVM_BACKENDS_HIP_KERNEL_CG_IMPLICIT_KERNEL_MATRIX_ASSEMBLY_BLAS_HIP_HPP_
#define PLSSVM_BACKENDS_HIP_KERNEL_CG_IMPLICIT_KERNEL_MATRIX_ASSEMBLY_BLAS_HIP_HPP_
#pragma once

#include "plssvm/backends/HIP/kernel/detail/reinterpret_array.hip.hpp"  // plssvm::hip::detail::reinterpret_array
#include "plssvm/backends/HIP/kernel/kernel_functions.hip.hpp"          // plssvm::hip::detail::{feature_reduce, apply_kernel_function}
#include "plssvm/constants.hpp"                                         // plssvm::{real_type, THREAD_BLOCK_SIZE, INTERNAL_BLOCK_SIZE, PADDING_SIZE}
#include "plssvm/kernel_function_types.hpp"                             // plssvm::kernel_function_type

#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"

#include <cstddef>  // std::size_t

namespace plssvm::hip::detail {

/**
 * @brief Perform an implicit BLAS SYMM-like operation: `C = alpha * A * B + C` where `A` is the implicitly calculated kernel matrix using the @p kernel_function (never actually stored, reducing the amount of needed global memory), @p B and @p C are matrices, and @p alpha is a scalar.
 * @tparam kernel_function the type of the used kernel function
 * @tparam Args the types of the parameters necessary for the specific kernel function
 * @param[in] alpha the scalar alpha value
 * @param[in] q the vector used in the dimensional reduction
 * @param[in] data the data points to calculate the implicit kernel matrix from
 * @param[in] num_rows the total number of data points (= total number of rows)
 * @param[in] device_num_rows the number of rows the current device is responsible for
 * @param[in] device_row_offset the first row in @p data the current device is responsible for
 * @param[in] num_features the number of features per data point
 * @param[in] QA_cost the scalar used in the dimensional reduction
 * @param[in] cost the cost factor the diagonal is scaled with
 * @param[in] B the matrix @p B
 * @param[in,out] C the matrix @p C
 * @param[in] num_classes the number of classes in the data set
 * @param[in] grid_x_offset the offset in x-dimension into the data points if more than one execution grid has to be used
 * @param[in] grid_y_offset the offset in y-dimension into the data points if more than one execution grid has to be used
 * @param[in] kernel_function_parameter the parameters necessary to apply the @p kernel_function
 */
template <kernel_function_type kernel_function, typename... Args>
__global__ void device_kernel_assembly_symm(const real_type alpha, const real_type *q, const real_type *data, const std::size_t num_rows, const std::size_t device_num_rows, const std::size_t device_row_offset, const std::size_t num_features, const real_type QA_cost, const real_type cost, const real_type *B, real_type *C, const std::size_t num_classes, const std::size_t grid_x_offset, const std::size_t grid_y_offset, Args... kernel_function_parameter) {
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

    // calculate the indices used in the current thread
    const auto i_idx = (blockIdx_x * blockDim_x + threadIdx_x) * INTERNAL_BLOCK_SIZE_uz;  // num_rows - device_row_offset
    const auto j_idx = (blockIdx_y * blockDim_y + threadIdx_y) * INTERNAL_BLOCK_SIZE_uz;  // device_num_rows

    // calculate the indices used in the current thread, pays attention to coalesced memory accesses
    const auto i_idx_linear = blockIdx_x * blockDim_x * INTERNAL_BLOCK_SIZE_uz + threadIdx_x;  // num_rows - device_row_offset
    const auto j_idx_linear = blockIdx_y * blockDim_y * INTERNAL_BLOCK_SIZE_uz + threadIdx_x;  // device_num_rows

    // create two shared memory arrays used for caching
    __shared__ real_type cache_one[THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];
    __shared__ real_type cache_two[THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];

    // only calculate the upper triangular matrix -> can't use threadIdx since all threads in a wavefront must progress further
    if (blockIdx_x >= blockIdx_y) {
        // create a thread private array used for internal caching
        real_type temp[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE]{};

        //*************************************************************************//
        //                   inplace kernel matrix construction                    //
        //*************************************************************************//
        {
            // reinterpret the shared memory arrays to be of shape [THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE]
            auto *data_i_cache = reinterpret_array<INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE>(cache_one);
            auto *data_j_cache = reinterpret_array<INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE>(cache_two);

            // iterate over all features using blocking to be able to cache them for faster memory accesses
            for (std::size_t feature_block = 0; feature_block < num_features; feature_block += THREAD_BLOCK_SIZE_uz) {
                // load data into shared memory
                for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                    // calculate the indices to access the global data, pays attention to coalesced memory accesses
                    const auto global_i_idx_linear = device_row_offset + i_idx_linear + static_cast<std::size_t>(internal) * THREAD_BLOCK_SIZE_uz;
                    const auto global_j_idx_linear = device_row_offset + j_idx_linear + static_cast<std::size_t>(internal) * THREAD_BLOCK_SIZE_uz;

                    // store the values in the shared memory
                    data_i_cache[threadIdx.y][internal * THREAD_BLOCK_SIZE + threadIdx.x] = data[(feature_block + threadIdx_y) * (num_rows + std::size_t{ 1 } + PADDING_SIZE_uz) + global_i_idx_linear];  // SoA
                    data_j_cache[threadIdx.y][internal * THREAD_BLOCK_SIZE + threadIdx.x] = data[(feature_block + threadIdx_y) * (num_rows + std::size_t{ 1 } + PADDING_SIZE_uz) + global_j_idx_linear];  // SoA
                }
                __syncthreads();  // wait until all threads loaded their part of the data

                // perform the feature reduction calculation
                for (unsigned feature = 0; feature < THREAD_BLOCK_SIZE; ++feature) {
                    for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                        for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                            temp[internal_i][internal_j] += detail::feature_reduce<kernel_function>(data_i_cache[feature][threadIdx.x * INTERNAL_BLOCK_SIZE + internal_i],
                                                                                                    data_j_cache[feature][threadIdx.y * INTERNAL_BLOCK_SIZE + internal_j]);
                        }
                    }
                }
                __syncthreads();  // wait until all threads performed their part of the calculations
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
                    temp[internal_i][internal_j] = detail::apply_kernel_function<kernel_function>(temp[internal_i][internal_j], kernel_function_parameter...) + QA_cost - q[global_i_idx] - q[global_j_idx];
                    // apply the cost on the diagonal
                    if (global_i_idx == global_j_idx) {
                        temp[internal_i][internal_j] += cost;
                    }
                } else {
                    // be sure to set the value to zero otherwise
                    temp[internal_i][internal_j] = real_type{ 0.0 };
                }
            }
        }

        //*************************************************************************//
        //     calculate C += alpha * temp * B for the UPPER triangular matrix     //
        //*************************************************************************//
        {
            // reinterpret the shared memory arrays to be of shape [INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE][THREAD_BLOCK_SIZE]
            auto *B_cache = reinterpret_array<THREAD_BLOCK_SIZE>(cache_one);
            auto *C_out_cache = reinterpret_array<THREAD_BLOCK_SIZE>(cache_two);

            // iterate over all classes using blocking to be able to cache them for faster memory accesses
            for (std::size_t class_block = 0; class_block < num_classes; class_block += THREAD_BLOCK_SIZE_uz) {
                // load data into shared memory
                for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                    // calculate the indices to access the global data, pays attention to coalesced memory accesses
                    const auto global_i_idx_linear = device_row_offset + i_idx_linear + static_cast<std::size_t>(internal) * THREAD_BLOCK_SIZE_uz;

                    // store the values in the shared memory
                    B_cache[internal * THREAD_BLOCK_SIZE + threadIdx.x][threadIdx.y] = alpha * B[global_i_idx_linear * (num_classes + PADDING_SIZE_uz) + class_block + threadIdx_y];  // SoA
                    C_out_cache[internal * THREAD_BLOCK_SIZE + threadIdx.x][threadIdx.y] = real_type{ 0.0 };                                                                          // SoA
                }
                __syncthreads();  // wait until all threads loaded their part of the data

                // calculate intermediate results and store them in shared memory
                for (unsigned class_idx = 0; class_idx < THREAD_BLOCK_SIZE; ++class_idx) {
                    for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                        for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                            C_out_cache[threadIdx.y * INTERNAL_BLOCK_SIZE + internal_j][(class_idx + threadIdx.x) % THREAD_BLOCK_SIZE] +=
                                temp[internal_i][internal_j] * B_cache[threadIdx.x * INTERNAL_BLOCK_SIZE + internal_i][(class_idx + threadIdx.x) % THREAD_BLOCK_SIZE];
                        }
                    }
                    __syncthreads();  // wait until all threads performed their part of the calculations
                }

                // atomically add the intermediate cached results to the C matrix
                for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                    // calculate the indices to access the global data
                    const auto global_j_idx = device_row_offset + j_idx + static_cast<std::size_t>(internal);

                    atomicAdd(&C[global_j_idx * (num_classes + PADDING_SIZE_uz) + class_block + threadIdx_x], C_out_cache[threadIdx.y * INTERNAL_BLOCK_SIZE + internal][threadIdx.x]);  // SoA
                }
                __syncthreads();  // wait until all threads updated C with their values
            }
        }

        // set potential diagonal entries in temp to 0.0 such that we don't apply the main diagonal twice to C
        for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
            for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                // calculate the indices to access the global data
                const auto global_i_idx = device_row_offset + i_idx + static_cast<std::size_t>(internal_i);
                const auto global_j_idx = device_row_offset + j_idx + static_cast<std::size_t>(internal_j);

                // update the diagonal
                if (global_i_idx == global_j_idx) {
                    temp[internal_i][internal_j] = real_type{ 0.0 };
                }
            }
        }

        //*************************************************************************//
        //     calculate C += alpha * temp * B for the LOWER triangular matrix     //
        //*************************************************************************//
        {
            // reinterpret the shared memory arrays to be of shape [THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE]
            auto *B_cache = reinterpret_array<INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE>(cache_one);
            auto *C_out_cache = reinterpret_array<INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE>(cache_two);

            // iterate over all classes using blocking to be able to cache them for faster memory accesses
            for (std::size_t class_block = 0; class_block < num_classes; class_block += THREAD_BLOCK_SIZE_uz) {
                // load data into shared memory
                for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                    // calculate the indices to access the global data, pays attention to coalesced memory accesses
                    const auto global_j_idx_linear = device_row_offset + j_idx_linear + static_cast<std::size_t>(internal) * THREAD_BLOCK_SIZE_uz;

                    // store the values in the shared memory
                    B_cache[threadIdx.y][internal * THREAD_BLOCK_SIZE + threadIdx.x] = alpha * B[global_j_idx_linear * (num_classes + PADDING_SIZE_uz) + class_block + threadIdx_y];  // SoA
                    C_out_cache[threadIdx.y][internal * THREAD_BLOCK_SIZE + threadIdx.x] = real_type{ 0.0 };
                }
                __syncthreads();  // wait until all threads loaded their part of the data

                // calculate intermediate results and store them in shared memory
                for (unsigned class_idx = 0; class_idx < THREAD_BLOCK_SIZE; ++class_idx) {
                    for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                        for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                            C_out_cache[(class_idx + threadIdx.y) % THREAD_BLOCK_SIZE][internal_i * THREAD_BLOCK_SIZE + threadIdx.x] +=
                                temp[internal_i][internal_j] * B_cache[(class_idx + threadIdx.y) % THREAD_BLOCK_SIZE][threadIdx.y * INTERNAL_BLOCK_SIZE + internal_j];
                        }
                    }
                    __syncthreads();  // wait until all threads performed their part of the calculations
                }

                // atomically add the intermediate cached results to the C matrix
                for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                    // calculate the indices to access the global data
                    const auto global_i_idx = device_row_offset + i_idx + static_cast<std::size_t>(internal);

                    atomicAdd(&C[global_i_idx * (num_classes + PADDING_SIZE_uz) + class_block + threadIdx_y], C_out_cache[threadIdx.y][internal * THREAD_BLOCK_SIZE + threadIdx.x]);  // SoA
                }
                __syncthreads();  // wait until all threads updated C with their values
            }
        }
    }
}

}  // namespace plssvm::hip::detail

#endif  // PLSSVM_BACKENDS_HIP_KERNEL_CG_IMPLICIT_KERNEL_MATRIX_ASSEMBLY_BLAS_HIP_HPP_
