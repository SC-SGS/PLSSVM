/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Functions for explicitly performing a BLAS GEMM like matrix-matrix multiplication using the HIP backend.
 */

#ifndef PLSSVM_BACKENDS_HIP_CG_EXPLICIT_BLAS_HIP_HPP_
#define PLSSVM_BACKENDS_HIP_CG_EXPLICIT_BLAS_HIP_HPP_
#pragma once

#include "plssvm/constants.hpp"  // plssvm::{real_type, THREAD_BLOCK_SIZE, INTERNAL_BLOCK_SIZE, FEATURE_BLOCK_SIZE, PADDING_SIZE}

#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"

namespace plssvm::hip {

/**
 * @brief Perform an explicit BLAS GEMM operation: `C = alpha * A * B + beta * C` where @p A is a `m x k` matrix, @p B is a `k x n` matrix, @p C is a `m x n` matrix, and @p alpha and @p beta are scalars.
 * @param[in] m the number of rows in @p A and @p C
 * @param[in] n the number of columns in @p B and @p C
 * @param[in] k the number of rows in @p A and number of columns in @p B
 * @param[in] alpha the scalar alpha value
 * @param[in] A the matrix @p A
 * @param[in] B the matrix @p B
 * @param[in] beta the scalar beta value
 * @param[in,out] C the matrix @p C, also used as result matrix
 */
__global__ void device_kernel_gemm([[maybe_unused]] const unsigned long long m, const unsigned long long n, const unsigned long long k, const real_type alpha, const real_type *A, const real_type *B, const real_type beta, real_type *C) {
    // compute: C = alpha * A * B + beta * C with A in m x k, B in n x k, and C in n x m, alpha, beta as scalar
    const unsigned long long i = (blockIdx.x * blockDim.x + threadIdx.x) * INTERNAL_BLOCK_SIZE;  // # rhs
    const unsigned long long i_linear = blockIdx.x * blockDim.x * INTERNAL_BLOCK_SIZE + threadIdx.x;
    const unsigned long long j = (blockIdx.y * blockDim.y + threadIdx.y) * INTERNAL_BLOCK_SIZE;  // # rows
    const unsigned long long j_cached_idx_linear = blockIdx.y * blockDim.y * INTERNAL_BLOCK_SIZE + threadIdx.x;

    __shared__ real_type A_cache[FEATURE_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];
    __shared__ real_type B_cache[FEATURE_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];

    real_type temp[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE] = { { 0.0 } };

    for (unsigned long long dim = 0; dim < k; dim += FEATURE_BLOCK_SIZE) {
        // load data into shared memory
        for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
            const unsigned long long global_i = i_linear + internal * THREAD_BLOCK_SIZE;
            const unsigned long long global_j = j_cached_idx_linear + internal * THREAD_BLOCK_SIZE;

            A_cache[threadIdx.y][internal * THREAD_BLOCK_SIZE + threadIdx.x] = A[(dim + threadIdx.y) * (k + PADDING_SIZE) + global_j];
            A_cache[threadIdx.y + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + threadIdx.x] = A[(dim + threadIdx.y + THREAD_BLOCK_SIZE) * (k + PADDING_SIZE) + global_j];

            B_cache[threadIdx.y][internal * THREAD_BLOCK_SIZE + threadIdx.x] = B[(dim + threadIdx.y) * (n + PADDING_SIZE) + global_i];
            B_cache[threadIdx.y + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + threadIdx.x] = B[(dim + threadIdx.y + THREAD_BLOCK_SIZE) * (n + PADDING_SIZE) + global_i];
        }
        __syncthreads();

        // calculation
        for (unsigned block_dim = 0; block_dim < FEATURE_BLOCK_SIZE; ++block_dim) {
            for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                    temp[internal_i][internal_j] += A_cache[block_dim][threadIdx.y * INTERNAL_BLOCK_SIZE + internal_j] * B_cache[block_dim][threadIdx.x * INTERNAL_BLOCK_SIZE + internal_i];
                }
            }
        }
        __syncthreads();
    }

    for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
        for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
            const unsigned long long global_i = i + internal_i;
            const unsigned long long global_j = j + internal_j;

            C[global_j * (n + PADDING_SIZE) + global_i] = alpha * temp[internal_i][internal_j] + beta * C[global_j * (n + PADDING_SIZE) + global_i];
        }
    }
}

/**
 * @brief Perform an explicit BLAS SYMM operation: `C = alpha * A * B + beta * C` where @p A is a `m x k` symmetric matrix (memory optimized), @p B is a `k x n` matrix, @p C is a `m x n` matrix, and @p alpha and @p beta are scalars.
 * @param[in] m the number of rows in @p A and @p C
 * @param[in] n the number of columns in @p B and @p C
 * @param[in] k the number of rows in @p A and number of columns in @p B
 * @param[in] alpha the scalar alpha value
 * @param[in] A the matrix @p A
 * @param[in] B the matrix @p B
 * @param[in] beta the scalar beta value
 * @param[in,out] C the matrix @p C, also used as result matrix
 */
__global__ void device_kernel_symm([[maybe_unused]] const unsigned long long m, const unsigned long long n, const unsigned long long k, const real_type alpha, const real_type *A, const real_type *B, const real_type beta, real_type *C) {
    // compute: C = alpha * A * B + beta * C with A in m x k, B in n x k, and C in n x m, alpha, beta as scalar
    const unsigned long long i = (blockIdx.x * blockDim.x + threadIdx.x) * INTERNAL_BLOCK_SIZE;  // # rhs
    const unsigned long long i_linear = blockIdx.x * blockDim.x * INTERNAL_BLOCK_SIZE + threadIdx.x;
    const unsigned long long j = (blockIdx.y * blockDim.y + threadIdx.y) * INTERNAL_BLOCK_SIZE;  // # rows
    const unsigned long long j_cached_idx_linear = blockIdx.y * blockDim.y * INTERNAL_BLOCK_SIZE + threadIdx.x;

    __shared__ real_type A_cache[FEATURE_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];
    __shared__ real_type B_cache[FEATURE_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];

    real_type temp[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE] = { { 0.0 } };

    for (unsigned long long dim = 0; dim < k; dim += FEATURE_BLOCK_SIZE) {
        // load data into shared memory
        for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
            const unsigned long long global_i = i_linear + internal * THREAD_BLOCK_SIZE;
            const unsigned long long global_j = j_cached_idx_linear + internal * THREAD_BLOCK_SIZE;

            // determine on which side of the diagonal we are located
            if (dim + threadIdx.y < global_j) {
                A_cache[threadIdx.y][internal * THREAD_BLOCK_SIZE + threadIdx.x] = A[(dim + threadIdx.y) * (k + PADDING_SIZE) + global_j - (dim + threadIdx.y) * (dim + threadIdx.y + 1) / 2];
            } else {
                A_cache[threadIdx.y][internal * THREAD_BLOCK_SIZE + threadIdx.x] = A[global_j * (k + PADDING_SIZE) + dim + threadIdx.y - global_j * (global_j + 1) / 2];
            }
            // determine on which side of the diagonal we are located
            if (dim + threadIdx.y + THREAD_BLOCK_SIZE < global_j) {
                A_cache[threadIdx.y + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + threadIdx.x] = A[(dim + threadIdx.y + THREAD_BLOCK_SIZE) * (k + PADDING_SIZE) + global_j - (dim + threadIdx.y + THREAD_BLOCK_SIZE) * (dim + threadIdx.y + THREAD_BLOCK_SIZE + 1) / 2];
            } else {
                A_cache[threadIdx.y + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + threadIdx.x] = A[global_j * (k + PADDING_SIZE) + dim + threadIdx.y + THREAD_BLOCK_SIZE - global_j * (global_j + 1) / 2];
            }

            B_cache[threadIdx.y][internal * THREAD_BLOCK_SIZE + threadIdx.x] = B[(dim + threadIdx.y) * (n + PADDING_SIZE) + global_i];
            B_cache[threadIdx.y + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + threadIdx.x] = B[(dim + threadIdx.y + THREAD_BLOCK_SIZE) * (n + PADDING_SIZE) + global_i];
        }
        __syncthreads();

        // calculation
        for (unsigned block_dim = 0; block_dim < FEATURE_BLOCK_SIZE; ++block_dim) {
            for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                    temp[internal_i][internal_j] += A_cache[block_dim][threadIdx.y * INTERNAL_BLOCK_SIZE + internal_j] * B_cache[block_dim][threadIdx.x * INTERNAL_BLOCK_SIZE + internal_i];
                }
            }
        }
        __syncthreads();
    }

    for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
        for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
            const unsigned long long global_i = i + internal_i;
            const unsigned long long global_j = j + internal_j;

            C[global_j * (n + PADDING_SIZE) + global_i] = alpha * temp[internal_i][internal_j] + beta * C[global_j * (n + PADDING_SIZE) + global_i];
        }
    }
}

}  // namespace plssvm::hip

#endif  // PLSSVM_BACKENDS_HIP_CG_EXPLICIT_BLAS_HIP_HPP_
