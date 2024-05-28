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

#ifndef PLSSVM_BACKENDS_HIP_CG_IMPLICIT_KERNEL_MATRIX_ASSEMBLY_BLAS_HIP_HPP_
#define PLSSVM_BACKENDS_HIP_CG_IMPLICIT_KERNEL_MATRIX_ASSEMBLY_BLAS_HIP_HPP_
#pragma once

#include "plssvm/backends/HIP/kernel/kernel_functions.hip.hpp"  // plssvm::hip::detail::{feature_reduce, apply_kernel_function}
#include "plssvm/constants.hpp"                                 // plssvm::{real_type, THREAD_BLOCK_SIZE, INTERNAL_BLOCK_SIZE, FEATURE_BLOCK_SIZE, PADDING_SIZE}
#include "plssvm/kernel_function_types.hpp"                     // plssvm::kernel_function_type

#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"

namespace plssvm::hip::detail {

/**
 * @brief Perform an implicit BLAS SYMM-like operation: `C = alpha * A * B + C` where `A` is the implicitly calculated kernel matrix using the @p kernel_function (never actually stored, reducing the amount of needed global memory), @p B and @p C are matrices, and @p alpha is a scalar.
 * @tparam kernel_function the type of the used kernel function
 * @tparam Args the types of the parameters necessary for the specific kernel function
 * @param[in] alpha the scalar alpha value
 * @param[in] q the vector used in the dimensional reduction
 * @param[in] data_d the data points to calculate the implicit kernel matrix from
 * @param[in] num_rows the total number of data points (= total number of rows)
 * @param[in] device_num_rows the number of rows the current device is responsible for
 * @param[in] row_offset the first row in @p data_d the current device is responsible for
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
__global__ void device_kernel_assembly_symm(const real_type alpha, const real_type *q, const real_type *data_d, const unsigned long long num_rows, const unsigned long long device_num_rows, const unsigned long long row_offset, const unsigned long long num_features, const real_type QA_cost, const real_type cost, const real_type *B, real_type *C, const unsigned long long num_classes, const unsigned long long grid_x_offset, const unsigned long long grid_y_offset, Args... kernel_function_parameter) {
    // cast all values to 64-bit unsigned long long to prevent potential 32-bit overflows
    const auto threadIdx_x = static_cast<unsigned long long>(threadIdx.x);                // current thread in block x-dimension
    const auto threadIdx_y = static_cast<unsigned long long>(threadIdx.y);                // current thread in block y-dimension
    const auto blockDim_x = static_cast<unsigned long long>(blockDim.x);                  // number of threads in block x-dimension
    const auto blockDim_y = static_cast<unsigned long long>(blockDim.y);                  // number of threads in block y-dimension
    const auto blockIdx_x = static_cast<unsigned long long>(blockIdx.x) + grid_x_offset;  // current block in grid x-dimension + offsets if the grid size would be too large
    const auto blockIdx_y = static_cast<unsigned long long>(blockIdx.y) + grid_y_offset;  // current block in grid y-dimension + offsets if the grid size would be too large
    const auto INTERNAL_BLOCK_SIZE_ull = static_cast<unsigned long long>(INTERNAL_BLOCK_SIZE);
    const auto THREAD_BLOCK_SIZE_ull = static_cast<unsigned long long>(THREAD_BLOCK_SIZE);
    const auto FEATURE_BLOCK_SIZE_ull = static_cast<unsigned long long>(FEATURE_BLOCK_SIZE);
    const auto PADDING_SIZE_ull = static_cast<unsigned long long>(PADDING_SIZE);

    // calculate the indices used in the current thread
    const auto i = (blockIdx_x * blockDim_x + threadIdx_x) * INTERNAL_BLOCK_SIZE_ull;
    const auto i_linear = blockIdx_x * blockDim_x * INTERNAL_BLOCK_SIZE_ull + threadIdx_x;
    const auto j = (blockIdx_y * blockDim_y + threadIdx_y) * INTERNAL_BLOCK_SIZE_ull;
    const auto j_linear = blockIdx_y * blockDim_y * INTERNAL_BLOCK_SIZE_ull + threadIdx_x;

    // only calculate the upper triangular matrix -> can't use threadIdx since all threads in a wavefront must progress further
    if (blockIdx_x >= blockIdx_y) {
        // create a thread private array used for internal caching
        real_type temp[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE]{};

        {
            // create the shared memory arrays used for caching data point features
            __shared__ real_type data_cache_i[FEATURE_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];
            __shared__ real_type data_cache_j[FEATURE_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];

            // iterate over all features using blocking to be able to cache them for faster memory accesses
            for (unsigned long long dim = 0; dim < num_features; dim += FEATURE_BLOCK_SIZE_ull) {
                // load data into shared memory
                for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                    const auto global_i = row_offset + i_linear + static_cast<unsigned long long>(internal) * THREAD_BLOCK_SIZE_ull;
                    const auto global_j = row_offset + j_linear + static_cast<unsigned long long>(internal) * THREAD_BLOCK_SIZE_ull;

                    // FEATURE_BLOCK_SIZE = 2 * THREAD_BLOCK_SIZE -> store twice as many values in the shared memory
                    data_cache_i[threadIdx.y][internal * THREAD_BLOCK_SIZE + threadIdx.x] = data_d[(dim + threadIdx_y) * (num_rows + 1ull + PADDING_SIZE_ull) + global_i];
                    data_cache_i[threadIdx.y + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + threadIdx.x] = data_d[(dim + threadIdx_y + THREAD_BLOCK_SIZE_ull) * (num_rows + 1ull + PADDING_SIZE_ull) + global_i];
                    data_cache_j[threadIdx.y][internal * THREAD_BLOCK_SIZE + threadIdx.x] = data_d[(dim + threadIdx_y) * (num_rows + 1ull + PADDING_SIZE_ull) + global_j];
                    data_cache_j[threadIdx.y + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + threadIdx.x] = data_d[(dim + threadIdx_y + THREAD_BLOCK_SIZE_ull) * (num_rows + 1ull + PADDING_SIZE_ull) + global_j];
                }
                __syncthreads();  // wait until all threads loaded their part of the data

                // perform the feature reduction calculation
                for (unsigned block_dim = 0; block_dim < FEATURE_BLOCK_SIZE; ++block_dim) {
                    for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                        for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                            temp[internal_i][internal_j] += detail::feature_reduce<kernel_function>(data_cache_i[block_dim][threadIdx.x * INTERNAL_BLOCK_SIZE + internal_i],
                                                                                                    data_cache_j[block_dim][threadIdx.y * INTERNAL_BLOCK_SIZE + internal_j]);
                        }
                    }
                }
                __syncthreads();  // wait until all threads performed their part of the calculations
            }
        }

        // apply the remaining part of the kernel function and store the value in the output kernel matrix
        for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
            for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                const auto global_i = row_offset + i + static_cast<unsigned long long>(internal_i);
                const auto device_global_i = i + static_cast<unsigned long long>(internal_i);
                const auto global_j = row_offset + j + static_cast<unsigned long long>(internal_j);
                const auto device_global_j = j + static_cast<unsigned long long>(internal_j);

                // be sure to not perform out of bounds accesses for the kernel matrix (only using the upper triangular matrix)
                if ((device_global_i < (num_rows - row_offset) && device_global_j < device_num_rows && global_i >= global_j)) {
                    temp[internal_i][internal_j] = detail::apply_kernel_function<kernel_function>(temp[internal_i][internal_j], kernel_function_parameter...) + QA_cost - q[global_i] - q[global_j];
                    // apply the cost on the diagonal
                    if (global_i == global_j) {
                        temp[internal_i][internal_j] += cost;
                    }
                } else {
                    // be sure to set the value to zero otherwise
                    temp[internal_i][internal_j] = real_type{ 0.0 };
                }
            }
        }

        // calculate C += alpha * temp * B for the UPPER triangular matrix
        {
            // same shared memory size but with different dimensions
            __shared__ real_type B_cache[INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE][FEATURE_BLOCK_SIZE];
            __shared__ real_type C_out_cache[INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE][FEATURE_BLOCK_SIZE];

            // iterate over all classes using blocking to be able to cache them for faster memory accesses
            for (unsigned long long dim = 0; dim < num_classes; dim += FEATURE_BLOCK_SIZE_ull) {
                // load data into shared memory
                for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                    const auto global_i = row_offset + i_linear + static_cast<unsigned long long>(internal) * THREAD_BLOCK_SIZE_ull;

                    // FEATURE_BLOCK_SIZE = 2 * THREAD_BLOCK_SIZE -> store twice as many values in the shared memory
                    B_cache[internal * THREAD_BLOCK_SIZE + threadIdx.x][threadIdx.y] = alpha * B[global_i * (num_classes + PADDING_SIZE_ull) + dim + threadIdx_y];
                    B_cache[internal * THREAD_BLOCK_SIZE + threadIdx.x][threadIdx.y + THREAD_BLOCK_SIZE] = alpha * B[global_i * (num_classes + PADDING_SIZE_ull) + dim + threadIdx_y + THREAD_BLOCK_SIZE_ull];
                    C_out_cache[internal * THREAD_BLOCK_SIZE + threadIdx.x][threadIdx.y] = real_type{ 0.0 };
                    C_out_cache[internal * THREAD_BLOCK_SIZE + threadIdx.x][threadIdx.y + THREAD_BLOCK_SIZE] = real_type{ 0.0 };
                }
                __syncthreads();  // wait until all threads loaded their part of the data

                // calculate intermediate results and store them in shared memory
                for (unsigned class_idx = 0; class_idx < FEATURE_BLOCK_SIZE; ++class_idx) {
                    for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                        for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                            C_out_cache[threadIdx.y * INTERNAL_BLOCK_SIZE + internal_j][(class_idx + threadIdx.x) % FEATURE_BLOCK_SIZE] +=
                                temp[internal_i][internal_j] * B_cache[threadIdx.x * INTERNAL_BLOCK_SIZE + internal_i][(class_idx + threadIdx.x) % FEATURE_BLOCK_SIZE];
                        }
                    }
                    __syncthreads();  // wait until all threads performed their part of the calculations
                }

                // add intermediate cached results to C
                for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                    const auto global_j = row_offset + j + static_cast<unsigned long long>(internal);
                    atomicAdd(&C[global_j * (num_classes + PADDING_SIZE_ull) + dim + threadIdx_x], C_out_cache[threadIdx.y * INTERNAL_BLOCK_SIZE + internal][threadIdx.x]);
                    atomicAdd(&C[global_j * (num_classes + PADDING_SIZE_ull) + dim + threadIdx_x + THREAD_BLOCK_SIZE_ull], C_out_cache[threadIdx.y * INTERNAL_BLOCK_SIZE + internal][threadIdx.x + THREAD_BLOCK_SIZE]);
                }
                __syncthreads();  // wai until all threads updated C with their values
            }
        }

        // set potential diagonal entries in temp to 0.0 such that we don't apply the main diagonal twice to C
        for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
            for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                const auto global_i = row_offset + i + static_cast<unsigned long long>(internal_i);
                const auto global_j = row_offset + j + static_cast<unsigned long long>(internal_j);

                if (global_i == global_j) {
                    temp[internal_i][internal_j] = real_type{ 0.0 };
                }
            }
        }

        // calculate C += alpha * temp * B for the LOWER triangular matrix
        {
            // same shared memory size but with different dimensions
            __shared__ real_type B_cache[FEATURE_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];
            __shared__ real_type C_out_cache[FEATURE_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];

            // iterate over all classes using blocking to be able to cache them for faster memory accesses
            for (unsigned long long dim = 0; dim < num_classes; dim += FEATURE_BLOCK_SIZE_ull) {
                // load data into shared memory
                for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                    const auto global_j = row_offset + j_linear + static_cast<unsigned long long>(internal) * THREAD_BLOCK_SIZE_ull;

                    // FEATURE_BLOCK_SIZE = 2 * THREAD_BLOCK_SIZE -> store twice as many values in the shared memory
                    B_cache[threadIdx.y][internal * THREAD_BLOCK_SIZE + threadIdx.x] = alpha * B[global_j * (num_classes + PADDING_SIZE_ull) + dim + threadIdx_y];
                    B_cache[threadIdx.y + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + threadIdx.x] = alpha * B[global_j * (num_classes + PADDING_SIZE_ull) + dim + threadIdx_y + THREAD_BLOCK_SIZE_ull];
                    C_out_cache[threadIdx.y][internal * THREAD_BLOCK_SIZE + threadIdx.x] = real_type{ 0.0 };
                    C_out_cache[threadIdx.y + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + threadIdx.x] = real_type{ 0.0 };
                }
                __syncthreads();  // wait until all threads loaded their part of the data

                // calculate intermediate results and store them in shared memory
                for (unsigned class_idx = 0; class_idx < FEATURE_BLOCK_SIZE; ++class_idx) {
                    for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                        for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                            C_out_cache[(class_idx + threadIdx.y) % FEATURE_BLOCK_SIZE][internal_i * THREAD_BLOCK_SIZE + threadIdx.x] +=
                                temp[internal_i][internal_j] * B_cache[(class_idx + threadIdx.y) % FEATURE_BLOCK_SIZE][threadIdx.y * INTERNAL_BLOCK_SIZE + internal_j];
                        }
                    }
                    __syncthreads();  // wait until all threads performed their part of the calculations
                }

                // add intermediate cached results to C
                for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                    const auto global_i = row_offset + i + static_cast<unsigned long long>(internal);
                    atomicAdd(&C[global_i * (num_classes + PADDING_SIZE_ull) + dim + threadIdx_y], C_out_cache[threadIdx.y][internal * THREAD_BLOCK_SIZE + threadIdx.x]);
                    atomicAdd(&C[global_i * (num_classes + PADDING_SIZE_ull) + dim + threadIdx_y + THREAD_BLOCK_SIZE_ull], C_out_cache[threadIdx.y + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + threadIdx.x]);
                }
                __syncthreads();  // wait until all threads updated C with their values
            }
        }
    }
}

}  // namespace plssvm::hip::detail

#endif  // PLSSVM_BACKENDS_HIP_CG_IMPLICIT_KERNEL_MATRIX_ASSEMBLY_BLAS_HIP_HPP_
