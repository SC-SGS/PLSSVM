/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Functions for explicitly assemblying the kernel matrix using the HIP backend.
 */

#ifndef PLSSVM_BACKENDS_HIP_CG_EXPLICIT_KERNEL_MATRIX_ASSEMBLY_HIP_HPP_
#define PLSSVM_BACKENDS_HIP_CG_EXPLICIT_KERNEL_MATRIX_ASSEMBLY_HIP_HPP_
#pragma once

#include "plssvm/constants.hpp"  // plssvm::{real_type, THREAD_BLOCK_SIZE, INTERNAL_BLOCK_SIZE, FEATURE_BLOCK_SIZE, PADDING_SIZE}

#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"

namespace plssvm::hip {

/**
 * @brief Create the explicit kernel matrix using the linear kernel function (\f$\vec{u}^T \cdot \vec{v}\f$).
 * @param[out] ret the calculated kernel matrix
 * @param[in] data_d the data points to calculate the kernel matrix from
 * @param[in] num_rows the number of data points
 * @param[in] num_features the number of features per data point
 * @param[in] q the vector used in the dimensional reduction
 * @param[in] QA_cost the scalar used in the dimensional reduction
 * @param[in] cost the cost factor the diagonal is scaled with
 */
__global__ void device_kernel_assembly_linear(real_type *ret, const real_type *data_d, const unsigned long long num_rows, const unsigned long long num_features, const real_type *q, const real_type QA_cost, const real_type cost) {
    const unsigned long long i = (blockIdx.x * blockDim.x + threadIdx.x) * INTERNAL_BLOCK_SIZE;
    const unsigned long long i_linear = blockIdx.x * blockDim.x * INTERNAL_BLOCK_SIZE + threadIdx.x;
    const unsigned long long j = (blockIdx.y * blockDim.y + threadIdx.y) * INTERNAL_BLOCK_SIZE;
    const unsigned long long j_cached_idx_linear = blockIdx.y * blockDim.y * INTERNAL_BLOCK_SIZE + threadIdx.x;

    __shared__ real_type data_cache_i[FEATURE_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];
    __shared__ real_type data_cache_j[FEATURE_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];

    if (blockIdx.x >= blockIdx.y) {
        real_type temp[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE] = { { 0.0 } };

        for (unsigned long long dim = 0; dim < num_features; dim += FEATURE_BLOCK_SIZE) {
            // load data into shared memory
            for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                const unsigned long long global_i = i_linear + internal * THREAD_BLOCK_SIZE;
                const unsigned long long global_j = j_cached_idx_linear + internal * THREAD_BLOCK_SIZE;

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
                        temp[internal_i][internal_j] += data_cache_i[block_dim][threadIdx.x * INTERNAL_BLOCK_SIZE + internal_i] * data_cache_j[block_dim][threadIdx.y * INTERNAL_BLOCK_SIZE + internal_j];
                    }
                }
            }
            __syncthreads();
        }

        for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
            for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                const unsigned long long global_i = i + internal_i;
                const unsigned long long global_j = j + internal_j;

                if (global_i < num_rows && global_j < num_rows && global_i >= global_j) {
                    real_type temp_ij = temp[internal_i][internal_j];
                    temp_ij = temp_ij + QA_cost - q[global_i] - q[global_j];
                    if (global_i == global_j) {
                        temp_ij += cost;
                    }

#if defined(PLSSVM_USE_GEMM)
                    ret[global_j * (num_rows + PADDING_SIZE) + global_i] = temp_ij;
                    ret[global_i * (num_rows + PADDING_SIZE) + global_j] = temp_ij;
#else
                    ret[global_j * (num_rows + PADDING_SIZE) + global_i - global_j * (global_j + 1) / 2] = temp_ij;
#endif
                }
            }
        }
    }
}

/**
 * @brief Create the explicit kernel matrix using the polynomial kernel function (\f$(gamma \cdot \vec{u}^T \cdot \vec{v} + coef0)^{degree}\f$).
 * @param[out] ret the calculated kernel matrix
 * @param[in] data_d the data points to calculate the kernel matrix from
 * @param[in] num_rows the number of data points
 * @param[in] num_features the number of features per data point
 * @param[in] q the vector used in the dimensional reduction
 * @param[in] QA_cost the scalar used in the dimensional reduction
 * @param[in] cost the cost factor the diagonal is scaled with
 * @param[in] degree parameter used in the polynomial kernel function
 * @param[in] gamma parameter used in the polynomial kernel function
 * @param[in] coef0 parameter used in the polynomial kernel function
 */
__global__ void device_kernel_assembly_polynomial(real_type *ret, const real_type *data_d, const unsigned long long num_rows, const unsigned long long num_features, const real_type *q, const real_type QA_cost, const real_type cost, const int degree, const real_type gamma, const real_type coef0) {
    const unsigned long long i = (blockIdx.x * blockDim.x + threadIdx.x) * INTERNAL_BLOCK_SIZE;
    const unsigned long long i_linear = blockIdx.x * blockDim.x * INTERNAL_BLOCK_SIZE + threadIdx.x;
    const unsigned long long j = (blockIdx.y * blockDim.y + threadIdx.y) * INTERNAL_BLOCK_SIZE;
    const unsigned long long j_cached_idx_linear = blockIdx.y * blockDim.y * INTERNAL_BLOCK_SIZE + threadIdx.x;

    __shared__ real_type data_cache_i[FEATURE_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];
    __shared__ real_type data_cache_j[FEATURE_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];

    if (blockIdx.x >= blockIdx.y) {
        real_type temp[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE] = { { 0.0 } };

        for (unsigned long long dim = 0; dim < num_features; dim += FEATURE_BLOCK_SIZE) {
            // load data into shared memory
            for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                const unsigned long long global_i = i_linear + internal * THREAD_BLOCK_SIZE;
                const unsigned long long global_j = j_cached_idx_linear + internal * THREAD_BLOCK_SIZE;

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
                        temp[internal_i][internal_j] += data_cache_i[block_dim][threadIdx.x * INTERNAL_BLOCK_SIZE + internal_i] * data_cache_j[block_dim][threadIdx.y * INTERNAL_BLOCK_SIZE + internal_j];
                    }
                }
            }
            __syncthreads();
        }

        for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
            for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                const unsigned long long global_i = i + internal_i;
                const unsigned long long global_j = j + internal_j;

                if (global_i < num_rows && global_j < num_rows && global_i >= global_j) {
                    real_type temp_ij = temp[internal_i][internal_j];
                    temp_ij = pow(gamma * temp_ij + coef0, (double) degree) + QA_cost - q[global_i] - q[global_j];
                    if (global_i == global_j) {
                        temp_ij += cost;
                    }

#if defined(PLSSVM_USE_GEMM)
                    ret[global_j * (num_rows + PADDING_SIZE) + global_i] = temp_ij;
                    ret[global_i * (num_rows + PADDING_SIZE) + global_j] = temp_ij;
#else
                    ret[global_j * (num_rows + PADDING_SIZE) + global_i - global_j * (global_j + 1) / 2] = temp_ij;
#endif
                }
            }
        }
    }
}

/**
 * @brief Create the explicit kernel matrix using the rbf kernel function (\f$e^{(-gamma \cdot |\vec{u} - \vec{v}|^2)}\f$).
 * @param[out] ret the calculated kernel matrix
 * @param[in] data_d the data points to calculate the kernel matrix from
 * @param[in] num_rows the number of data points
 * @param[in] num_features the number of features per data point
 * @param[in] q the vector used in the dimensional reduction
 * @param[in] QA_cost the scalar used in the dimensional reduction
 * @param[in] cost the cost factor the diagonal is scaled with
 * @param[in] gamma parameter used in the rbf kernel function
 */
__global__ void device_kernel_assembly_rbf(real_type *ret, const real_type *data_d, const unsigned long long num_rows, const unsigned long long num_features, const real_type *q, const real_type QA_cost, const real_type cost, const real_type gamma) {
    const unsigned long long i = (blockIdx.x * blockDim.x + threadIdx.x) * INTERNAL_BLOCK_SIZE;
    const unsigned long long i_linear = blockIdx.x * blockDim.x * INTERNAL_BLOCK_SIZE + threadIdx.x;
    const unsigned long long j = (blockIdx.y * blockDim.y + threadIdx.y) * INTERNAL_BLOCK_SIZE;
    const unsigned long long j_cached_idx_linear = blockIdx.y * blockDim.y * INTERNAL_BLOCK_SIZE + threadIdx.x;

    __shared__ real_type data_cache_i[FEATURE_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];
    __shared__ real_type data_cache_j[FEATURE_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];

    if (blockIdx.x >= blockIdx.y) {
        real_type temp[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE] = { { 0.0 } };

        for (unsigned long long dim = 0; dim < num_features; dim += FEATURE_BLOCK_SIZE) {
            // load data into shared memory
            for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                const unsigned long long global_i = i_linear + internal * THREAD_BLOCK_SIZE;
                const unsigned long long global_j = j_cached_idx_linear + internal * THREAD_BLOCK_SIZE;

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
                        const real_type d = data_cache_i[block_dim][threadIdx.x * INTERNAL_BLOCK_SIZE + internal_i] - data_cache_j[block_dim][threadIdx.y * INTERNAL_BLOCK_SIZE + internal_j];
                        temp[internal_i][internal_j] += d * d;
                    }
                }
            }
            __syncthreads();
        }

        for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
            for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                const unsigned long long global_i = i + internal_i;
                const unsigned long long global_j = j + internal_j;

                if (global_i < num_rows && global_j < num_rows && global_i >= global_j) {
                    real_type temp_ij = temp[internal_i][internal_j];
                    temp_ij = exp(-gamma * temp_ij) + QA_cost - q[global_i] - q[global_j];
                    if (global_i == global_j) {
                        temp_ij += cost;
                    }

#if defined(PLSSVM_USE_GEMM)
                    ret[global_j * (num_rows + PADDING_SIZE) + global_i] = temp_ij;
                    ret[global_i * (num_rows + PADDING_SIZE) + global_j] = temp_ij;
#else
                    ret[global_j * (num_rows + PADDING_SIZE) + global_i - global_j * (global_j + 1) / 2] = temp_ij;
#endif
                }
            }
        }
    }
}

}

#endif  // PLSSVM_BACKENDS_HIP_CG_EXPLICIT_KERNEL_MATRIX_ASSEMBLY_HIP_HPP_
