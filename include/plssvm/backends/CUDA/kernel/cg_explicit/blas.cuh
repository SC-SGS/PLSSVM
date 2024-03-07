/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Functions for explicitly performing a BLAS GEMM like matrix-matrix multiplication using the CUDA backend.
 */

#ifndef PLSSVM_BACKENDS_CUDA_KERNEL_CG_EXPLICIT_BLAS_CUH_
#define PLSSVM_BACKENDS_CUDA_KERNEL_CG_EXPLICIT_BLAS_CUH_
#pragma once

#include "plssvm/constants.hpp"  // plssvm::{real_type, THREAD_BLOCK_SIZE, INTERNAL_BLOCK_SIZE, FEATURE_BLOCK_SIZE, PADDING_SIZE}

namespace plssvm::cuda {

/**
 * @brief Perform an explicit BLAS SYMM operation: `C = alpha * A * B + beta * C` where @p A is a `m x k` symmetric matrix (memory optimized), @p B is a `k x n` matrix, @p C is a `m x n` matrix, and @p alpha and @p beta are scalars.
 * @details In a multi-GPU setting, this function is only responsible for the rows this device is responsible for!
 * @param[in] num_rows the number of rows in @p A and @p C
 * @param[in] num_rhs the number of columns in @p B and @p C
 * @param[in] device_specific_num_rows the number of rows in @p A and number of rows in @p B; thr rows in @p A are potentially distributed across multiple devices
 * @param[in] row_offset the first row this device is responsible for
 * @param[in] alpha the scalar alpha value
 * @param[in] A the matrix @p A
 * @param[in] B the matrix @p B
 * @param[in] beta the scalar beta value
 * @param[in,out] C the matrix @p C, also used as result matrix
 */
__global__ void device_kernel_symm(const unsigned long long num_rows, const unsigned long long num_rhs, const unsigned long long device_specific_num_rows, const unsigned long long row_offset, const real_type alpha, const real_type *A, const real_type *B, const real_type beta, real_type *C) {
    // compute: C = alpha * A * B + beta * C with A in m x k, B in n x k, and C in n x m, alpha, beta as scalar
    const unsigned long long i = (blockIdx.x * blockDim.x + threadIdx.x) * INTERNAL_BLOCK_SIZE;  // # rhs -> num_rhs
    const unsigned long long i_linear = blockIdx.x * blockDim.x * INTERNAL_BLOCK_SIZE + threadIdx.x;
    const unsigned long long j = (blockIdx.y * blockDim.y + threadIdx.y) * INTERNAL_BLOCK_SIZE;  // # rows -> device_specific_num_rows
    const unsigned long long j_cached_idx_linear = blockIdx.y * blockDim.y * INTERNAL_BLOCK_SIZE + threadIdx.x;

    __shared__ real_type A_cache[FEATURE_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];
    __shared__ real_type B_cache[FEATURE_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];

    real_type temp[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE] = { 0.0 };

    for (unsigned long long dim = 0; dim < (num_rows - row_offset); dim += FEATURE_BLOCK_SIZE) {
        // load data into shared memory
        for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
            const unsigned long long global_i = i_linear + internal * THREAD_BLOCK_SIZE;
            const unsigned long long global_j = j_cached_idx_linear + internal * THREAD_BLOCK_SIZE;

            // determine on which side of the diagonal we are located
            if (dim + threadIdx.y < global_j) {
                A_cache[threadIdx.y][internal * THREAD_BLOCK_SIZE + threadIdx.x] = A[(dim + threadIdx.y) * (num_rows - row_offset + PADDING_SIZE) + global_j - (dim + threadIdx.y) * (dim + threadIdx.y + 1) / 2];
            } else {
                A_cache[threadIdx.y][internal * THREAD_BLOCK_SIZE + threadIdx.x] = A[global_j * (num_rows - row_offset + PADDING_SIZE) + dim + threadIdx.y - global_j * (global_j + 1) / 2];
            }
            // determine on which side of the diagonal we are located
            if (dim + threadIdx.y + THREAD_BLOCK_SIZE < global_j) {
                A_cache[threadIdx.y + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + threadIdx.x] = A[(dim + threadIdx.y + THREAD_BLOCK_SIZE) * (num_rows - row_offset + PADDING_SIZE) + global_j - (dim + threadIdx.y + THREAD_BLOCK_SIZE) * (dim + threadIdx.y + THREAD_BLOCK_SIZE + 1) / 2];
            } else {
                A_cache[threadIdx.y + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + threadIdx.x] = A[global_j * (num_rows - row_offset + PADDING_SIZE) + dim + threadIdx.y + THREAD_BLOCK_SIZE - global_j * (global_j + 1) / 2];
            }

            B_cache[threadIdx.y][internal * THREAD_BLOCK_SIZE + threadIdx.x] = B[(dim + row_offset + threadIdx.y) * (num_rhs + PADDING_SIZE) + global_i];
            B_cache[threadIdx.y + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + threadIdx.x] = B[(dim + row_offset + threadIdx.y + THREAD_BLOCK_SIZE) * (num_rhs + PADDING_SIZE) + global_i];
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
            const unsigned long long device_global_j = j + internal_j;
            const unsigned long long global_j = row_offset + j + internal_j;

            if (global_i < num_rhs && device_global_j < device_specific_num_rows) {
                C[global_j * (num_rhs + PADDING_SIZE) + global_i] = alpha * temp[internal_i][internal_j] + beta * C[global_j * (num_rhs + PADDING_SIZE) + global_i];
            }
        }
    }
}

/**
 * @brief Perform an explicit BLAS SYMM operation: `C = alpha * A * B + beta * C` where @p A is a `m x k` symmetric matrix (memory optimized), @p B is a `k x n` matrix, @p C is a `m x n` matrix, and @p alpha and @p beta are scalars.
 * @details In a multi-GPU setting, this function is responsible for mirroring down the columns this device is responsible for!
 * @param[in] num_rows the number of rows in @p A and @p C
 * @param[in] num_rhs the number of columns in @p B and @p C
 * @param[in] device_specific_num_rows the number of rows in @p A and number of rows in @p B; thr rows in @p A are potentially distributed across multiple devices
 * @param[in] row_offset the first row this device is responsible for
 * @param[in] alpha the scalar alpha value
 * @param[in] A the matrix @p A
 * @param[in] B the matrix @p B
 * @param[in] beta the scalar beta value
 * @param[in,out] C the matrix @p C, also used as result matrix
 */
__global__ void device_kernel_symm_mirror(const unsigned long long num_rows, const unsigned long long num_rhs, const unsigned long long num_mirror_rows, const unsigned long long device_specific_num_rows, const unsigned long long row_offset, const real_type alpha, const real_type *A, const real_type *B, const real_type beta, real_type *C) {
    const unsigned long long i = (blockIdx.x * blockDim.x + threadIdx.x) * INTERNAL_BLOCK_SIZE;  // # rhs -> num_rhs
    const unsigned long long i_linear = blockIdx.x * blockDim.x * INTERNAL_BLOCK_SIZE + threadIdx.x;
    const unsigned long long j = (blockIdx.y * blockDim.y + threadIdx.y) * INTERNAL_BLOCK_SIZE;  // # rows -> num_mirror_rows
    const unsigned long long j_cached_idx_linear = blockIdx.y * blockDim.y * INTERNAL_BLOCK_SIZE + threadIdx.x;

    __shared__ real_type A_cache[FEATURE_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];
    __shared__ real_type B_cache[FEATURE_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];

    real_type temp[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE] = { 0.0 };

    for (unsigned long long dim = 0; dim < device_specific_num_rows; dim += FEATURE_BLOCK_SIZE) {
        // load data into shared memory
        for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
            const unsigned long long global_i = i_linear + internal * THREAD_BLOCK_SIZE;
            const unsigned long long global_j = j_cached_idx_linear + internal * THREAD_BLOCK_SIZE;

            A_cache[threadIdx.y][internal * THREAD_BLOCK_SIZE + threadIdx.x] = A[(dim + threadIdx.y) * (num_rows - row_offset + PADDING_SIZE) - (dim + threadIdx.y - 1) * (dim + threadIdx.y) / 2 + device_specific_num_rows - (dim + threadIdx.y) + global_j];
            A_cache[threadIdx.y + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + threadIdx.x] = A[(dim + threadIdx.y + THREAD_BLOCK_SIZE) * (num_rows - row_offset + PADDING_SIZE) - (dim + threadIdx.y + THREAD_BLOCK_SIZE - 1) * (dim + threadIdx.y + THREAD_BLOCK_SIZE) / 2 + device_specific_num_rows - (dim + threadIdx.y + THREAD_BLOCK_SIZE) + global_j];

            B_cache[threadIdx.y][internal * THREAD_BLOCK_SIZE + threadIdx.x] = B[(row_offset + dim + threadIdx.y) * (num_rhs + PADDING_SIZE) + global_i];
            B_cache[threadIdx.y + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + threadIdx.x] = B[(row_offset + dim + threadIdx.y + THREAD_BLOCK_SIZE) * (num_rhs + PADDING_SIZE) + global_i];
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
            const unsigned long long partial_global_j = j + internal_j;
            const unsigned long long global_j = row_offset + device_specific_num_rows + j + internal_j;

            if (global_i < num_rhs && partial_global_j < num_mirror_rows) {
                C[global_j * (num_rhs + PADDING_SIZE) + global_i] = alpha * temp[internal_i][internal_j] + beta * C[global_j * (num_rhs + PADDING_SIZE) + global_i];
            }
        }
    }
}

/**
 * @brief Perform a simple inplace matrix addition: lhs += rhs.
 * @param[in] num_rows the number of rows in both matrices
 * @param[in] num_cols the number of columns in both matrices
 * @param[in,out] lhs the first matrix (updated inplace)
 * @param[in] rhs the second matrix
 */
__global__ void device_kernel_inplace_matrix_add(const unsigned long long num_rows, const unsigned long long num_cols, real_type *lhs, const real_type *rhs) {
    const unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;  // # rhs
    const unsigned long long j = blockIdx.y * blockDim.y + threadIdx.y;  // # num_rows
    // TODO: optimize?

    if (i < num_cols && j < num_rows) {
        lhs[j * (num_cols + PADDING_SIZE) + i] += rhs[j * (num_cols + PADDING_SIZE) + i];
    }
}

/**
 * @brief Perform a simple inplace matrix scale: lhs *= scalar.
 * @param[in] num_rows the number of rows in the matrix
 * @param[in] num_cols the number of columns in the matrix
 * @param[in,out] lhs the matrix (updated inplace)
 * @param[in] scale the value to scale
 */
__global__ void device_kernel_inplace_matrix_scale(const unsigned long long num_rows, const unsigned long long num_cols, real_type *lhs, const real_type scale) {
    const unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;  // # rhs
    const unsigned long long j = blockIdx.y * blockDim.y + threadIdx.y;  // # num_rows
    // TODO: optimize?

    if (i < num_cols && j < num_rows) {
        lhs[j * (num_cols + PADDING_SIZE) + i] *= scale;
    }
}

}  // namespace plssvm::cuda

#endif  // PLSSVM_BACKENDS_CUDA_KERNEL_CG_EXPLICIT_BLAS_CUH_
