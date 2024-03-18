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
 * @param[in] kernel_function_parameter the parameters necessary to apply the @p kernel_function
 */
template <kernel_function_type kernel_function, typename... Args>
__global__ void device_kernel_assembly_symm(const real_type alpha, const real_type *q, const real_type *data_d, const unsigned long long num_rows, const unsigned long long device_num_rows, const unsigned long long row_offset, const unsigned long long num_features, const real_type QA_cost, const real_type cost, const real_type *B, real_type *C, const unsigned long long num_classes, Args... kernel_function_parameter) {
    const unsigned long long i = (blockIdx.x * blockDim.x + threadIdx.x) * INTERNAL_BLOCK_SIZE;
    const unsigned long long i_linear = blockIdx.x * blockDim.x * INTERNAL_BLOCK_SIZE + threadIdx.x;
    const unsigned long long j = (blockIdx.y * blockDim.y + threadIdx.y) * INTERNAL_BLOCK_SIZE;
    const unsigned long long j_cached_idx_linear = blockIdx.y * blockDim.y * INTERNAL_BLOCK_SIZE + threadIdx.x;

    if (blockIdx.x >= blockIdx.y) {
        real_type temp[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE] = { { 0.0 } };

        {
            __shared__ real_type data_cache_i[FEATURE_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];
            __shared__ real_type data_cache_j[FEATURE_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];

            for (unsigned long long dim = 0; dim < num_features; dim += FEATURE_BLOCK_SIZE) {
                // load data into shared memory
                for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                    const unsigned long long global_i = row_offset + i_linear + internal * THREAD_BLOCK_SIZE;
                    const unsigned long long global_j = row_offset + j_cached_idx_linear + internal * THREAD_BLOCK_SIZE;

                    data_cache_i[threadIdx.y][internal * THREAD_BLOCK_SIZE + threadIdx.x] = data_d[(dim + threadIdx.y) * (num_rows + 1 + PADDING_SIZE) + global_i];
                    data_cache_i[threadIdx.y + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + threadIdx.x] = data_d[(dim + threadIdx.y + THREAD_BLOCK_SIZE) * (num_rows + 1 + PADDING_SIZE) + global_i];
                    data_cache_j[threadIdx.y][internal * THREAD_BLOCK_SIZE + threadIdx.x] = data_d[(dim + threadIdx.y) * (num_rows + 1 + PADDING_SIZE) + global_j];
                    data_cache_j[threadIdx.y + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + threadIdx.x] = data_d[(dim + threadIdx.y + THREAD_BLOCK_SIZE) * (num_rows + 1 + PADDING_SIZE) + global_j];
                }
                __syncthreads();

                // calculation
                for (unsigned block_dim = 0; block_dim < FEATURE_BLOCK_SIZE; ++block_dim) {
                    for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                        for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                            temp[internal_i][internal_j] += detail::feature_reduce<kernel_function>(data_cache_i[block_dim][threadIdx.x * INTERNAL_BLOCK_SIZE + internal_i],
                                                                                                    data_cache_j[block_dim][threadIdx.y * INTERNAL_BLOCK_SIZE + internal_j]);
                        }
                    }
                }
                __syncthreads();
            }
        }

        // update temp using the rbf kernel function taking the dimensional reduction into account and apply the cost to the diagonal
        for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
            for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                const unsigned long long global_i = row_offset + i + internal_i;
                const unsigned long long device_global_i = i + internal_i;
                const unsigned long long global_j = row_offset + j + internal_j;
                const unsigned long long device_global_j = j + internal_j;

                if ((device_global_i < (num_rows - row_offset) && device_global_j < device_num_rows && global_i >= global_j)) {
                    temp[internal_i][internal_j] = detail::apply_kernel_function<kernel_function>(temp[internal_i][internal_j], kernel_function_parameter...) + QA_cost - q[global_i] - q[global_j];
                    if (global_i == global_j) {
                        temp[internal_i][internal_j] += cost;
                    }
                } else {
                    temp[internal_i][internal_j] = 0.0;
                }
            }
        }

        // calculate C += alpha * temp * B for the UPPER triangular matrix
        {
            __shared__ real_type B_cache[INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE][FEATURE_BLOCK_SIZE];
            __shared__ real_type C_out_cache[INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE][FEATURE_BLOCK_SIZE];

            for (unsigned long long dim = 0; dim < num_classes; dim += FEATURE_BLOCK_SIZE) {
                // load data into shared memory
                for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                    const unsigned long long global_i = row_offset + i_linear + internal * THREAD_BLOCK_SIZE;

                    B_cache[internal * THREAD_BLOCK_SIZE + threadIdx.x][threadIdx.y] = alpha * B[global_i * (num_classes + PADDING_SIZE) + dim + threadIdx.y];
                    B_cache[internal * THREAD_BLOCK_SIZE + threadIdx.x][threadIdx.y + THREAD_BLOCK_SIZE] = alpha * B[global_i * (num_classes + PADDING_SIZE) + dim + threadIdx.y + THREAD_BLOCK_SIZE];

                    C_out_cache[internal * THREAD_BLOCK_SIZE + threadIdx.x][threadIdx.y] = 0.0;
                    C_out_cache[internal * THREAD_BLOCK_SIZE + threadIdx.x][threadIdx.y + THREAD_BLOCK_SIZE] = 0.0;
                }
                __syncthreads();

                // calculate intermediate results and store them in shared memory
                for (unsigned class_idx = 0; class_idx < FEATURE_BLOCK_SIZE; ++class_idx) {
                    for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                        for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                            C_out_cache[threadIdx.y * INTERNAL_BLOCK_SIZE + internal_j][(class_idx + threadIdx.x) % FEATURE_BLOCK_SIZE] +=
                                temp[internal_i][internal_j] * B_cache[threadIdx.x * INTERNAL_BLOCK_SIZE + internal_i][(class_idx + threadIdx.x) % FEATURE_BLOCK_SIZE];
                        }
                    }
                    __syncthreads();
                }

                // add intermediate cached results to C
                for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                    const unsigned long long global_j = row_offset + j + internal;
                    atomicAdd(&C[global_j * (num_classes + PADDING_SIZE) + dim + threadIdx.x], C_out_cache[threadIdx.y * INTERNAL_BLOCK_SIZE + internal][threadIdx.x]);
                    atomicAdd(&C[global_j * (num_classes + PADDING_SIZE) + dim + threadIdx.x + THREAD_BLOCK_SIZE], C_out_cache[threadIdx.y * INTERNAL_BLOCK_SIZE + internal][threadIdx.x + THREAD_BLOCK_SIZE]);
                }
                __syncthreads();
            }
        }

        // set potential diagonal entries in temp to 0.0 such that we don't apply the main diagonal twice to C
        for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
            for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                const unsigned long long global_i = row_offset + i + internal_i;
                const unsigned long long global_j = row_offset + j + internal_j;

                if (global_i == global_j) {
                    temp[internal_i][internal_j] = 0.0;
                }
            }
        }

        // calculate C += alpha * temp * B for the LOWER triangular matrix
        {
            __shared__ real_type B_cache[FEATURE_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];
            __shared__ real_type C_out_cache[FEATURE_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];

            for (unsigned long long dim = 0; dim < num_classes; dim += FEATURE_BLOCK_SIZE) {
                // load data into shared memory
                for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                    const unsigned long long global_j = row_offset + j_cached_idx_linear + internal * THREAD_BLOCK_SIZE;

                    B_cache[threadIdx.y][internal * THREAD_BLOCK_SIZE + threadIdx.x] = alpha * B[global_j * (num_classes + PADDING_SIZE) + dim + threadIdx.y];
                    B_cache[threadIdx.y + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + threadIdx.x] = alpha * B[global_j * (num_classes + PADDING_SIZE) + dim + threadIdx.y + THREAD_BLOCK_SIZE];

                    C_out_cache[threadIdx.y][internal * THREAD_BLOCK_SIZE + threadIdx.x] = 0.0;
                    C_out_cache[threadIdx.y + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + threadIdx.x] = 0.0;
                }
                __syncthreads();

                // calculate intermediate results and store them in shared memory
                for (unsigned class_idx = 0; class_idx < FEATURE_BLOCK_SIZE; ++class_idx) {
                    for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                        for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                            C_out_cache[(class_idx + threadIdx.y) % FEATURE_BLOCK_SIZE][internal_i * THREAD_BLOCK_SIZE + threadIdx.x] +=
                                temp[internal_i][internal_j] * B_cache[(class_idx + threadIdx.y) % FEATURE_BLOCK_SIZE][threadIdx.y * INTERNAL_BLOCK_SIZE + internal_j];
                        }
                    }
                    __syncthreads();
                }

                // add intermediate cached results to C
                for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                    const unsigned long long global_i = row_offset + i + internal;
                    atomicAdd(&C[global_i * (num_classes + PADDING_SIZE) + dim + threadIdx.y], C_out_cache[threadIdx.y][internal * THREAD_BLOCK_SIZE + threadIdx.x]);
                    atomicAdd(&C[global_i * (num_classes + PADDING_SIZE) + dim + threadIdx.y + THREAD_BLOCK_SIZE], C_out_cache[threadIdx.y + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + threadIdx.x]);
                }
                __syncthreads();
            }
        }
    }
}

}  // namespace plssvm::hip::detail

#endif  // PLSSVM_BACKENDS_HIP_CG_IMPLICIT_KERNEL_MATRIX_ASSEMBLY_BLAS_HIP_HPP_
