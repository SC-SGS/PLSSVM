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

namespace plssvm::hip::detail {

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
__global__ void device_kernel_symm(const unsigned long long num_rows, const unsigned long long num_rhs, const unsigned long long device_specific_num_rows, const unsigned long long row_offset, const real_type alpha, const real_type *A, const real_type *B, const real_type beta, real_type *C, const unsigned long long grid_x_offset, const unsigned long long grid_y_offset) {
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
    const auto i = (blockIdx_x * blockDim_x + threadIdx_x) * INTERNAL_BLOCK_SIZE_ull;  // # rhs -> num_rhs
    const auto i_linear = blockIdx_x * blockDim_x * INTERNAL_BLOCK_SIZE_ull + threadIdx_x;
    const auto j = (blockIdx_y * blockDim_y + threadIdx_y) * INTERNAL_BLOCK_SIZE_ull;  // # rows -> device_specific_num_rows
    const auto j_linear = blockIdx_y * blockDim_y * INTERNAL_BLOCK_SIZE_ull + threadIdx_x;

    // create the shared memory arrays used for caching data point features
    __shared__ real_type A_cache[FEATURE_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];
    __shared__ real_type B_cache[FEATURE_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];

    // create a thread private array used for internal caching
    real_type temp[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE]{};

    // iterate over all features using blocking to be able to cache them for faster memory accesses
    for (unsigned long long dim = 0; dim < (num_rows - row_offset); dim += FEATURE_BLOCK_SIZE_ull) {
        // load data into shared memory
        for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
            const auto global_i = i_linear + static_cast<unsigned long long>(internal) * THREAD_BLOCK_SIZE_ull;
            const auto global_j = j_linear + static_cast<unsigned long long>(internal) * THREAD_BLOCK_SIZE_ull;

            // determine on which side of the diagonal we are located
            if (dim + threadIdx_y < global_j) {
                A_cache[threadIdx.y][internal * THREAD_BLOCK_SIZE + threadIdx.x] = A[(dim + threadIdx_y) * (num_rows - row_offset + PADDING_SIZE_ull) + global_j - (dim + threadIdx_y) * (dim + threadIdx_y + 1ull) / 2ull];
            } else {
                A_cache[threadIdx.y][internal * THREAD_BLOCK_SIZE + threadIdx.x] = A[global_j * (num_rows - row_offset + PADDING_SIZE_ull) + dim + threadIdx_y - global_j * (global_j + 1ull) / 2ull];
            }
            // determine on which side of the diagonal we are located
            if (dim + threadIdx.y + THREAD_BLOCK_SIZE < global_j) {
                A_cache[threadIdx.y + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + threadIdx.x] = A[(dim + threadIdx_y + THREAD_BLOCK_SIZE_ull) * (num_rows - row_offset + PADDING_SIZE_ull) + global_j - (dim + threadIdx_y + THREAD_BLOCK_SIZE_ull) * (dim + threadIdx_y + THREAD_BLOCK_SIZE_ull + 1ull) / 2ull];
            } else {
                A_cache[threadIdx.y + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + threadIdx.x] = A[global_j * (num_rows - row_offset + PADDING_SIZE_ull) + dim + threadIdx_y + THREAD_BLOCK_SIZE_ull - global_j * (global_j + 1ull) / 2ull];
            }

            B_cache[threadIdx.y][internal * THREAD_BLOCK_SIZE + threadIdx.x] = B[(dim + row_offset + threadIdx_y) * (num_rhs + PADDING_SIZE_ull) + global_i];
            B_cache[threadIdx.y + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + threadIdx.x] = B[(dim + row_offset + threadIdx_y + THREAD_BLOCK_SIZE_ull) * (num_rhs + PADDING_SIZE_ull) + global_i];
        }
        __syncthreads();  // wait until all threads loaded their part of the data

        // perform the dot product calculation
        for (unsigned block_dim = 0; block_dim < FEATURE_BLOCK_SIZE; ++block_dim) {
            for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                    temp[internal_i][internal_j] += A_cache[block_dim][threadIdx.y * INTERNAL_BLOCK_SIZE + internal_j] * B_cache[block_dim][threadIdx.x * INTERNAL_BLOCK_SIZE + internal_i];
                }
            }
        }
        __syncthreads();  // wait until all threads performed their part of the calculations
    }

    // apply the (partial) BLAS operation and update C
    for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
        for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
            const auto global_i = i + static_cast<unsigned long long>(internal_i);
            const auto device_global_j = j + static_cast<unsigned long long>(internal_j);
            const auto global_j = row_offset + j + static_cast<unsigned long long>(internal_j);

            // be sure to not perform out of bounds accesses
            if (global_i < num_rhs && device_global_j < device_specific_num_rows) {
                C[global_j * (num_rhs + PADDING_SIZE_ull) + global_i] = alpha * temp[internal_i][internal_j] + beta * C[global_j * (num_rhs + PADDING_SIZE_ull) + global_i];
            }
        }
    }
}

/**
 * @brief Perform an explicit BLAS SYMM operation: `C = alpha * A * B + beta * C` where @p A is a `m x k` symmetric matrix (memory optimized), @p B is a `k x n` matrix, @p C is a `m x n` matrix, and @p alpha and @p beta are scalars.
 * @details In a multi-GPU setting, this function is responsible for mirroring down the columns this device is responsible for!
 * @param[in] num_rows the number of rows in @p A and @p C
 * @param[in] num_rhs the number of columns in @p B and @p C
 * @param[in] num_mirror_rows the number of rows to mirror down
 * @param[in] device_specific_num_rows the number of rows in @p A and number of rows in @p B; thr rows in @p A are potentially distributed across multiple devices
 * @param[in] row_offset the first row this device is responsible for
 * @param[in] alpha the scalar alpha value
 * @param[in] A the matrix @p A
 * @param[in] B the matrix @p B
 * @param[in] beta the scalar beta value
 * @param[in,out] C the matrix @p C, also used as result matrix
 */
__global__ void device_kernel_symm_mirror(const unsigned long long num_rows, const unsigned long long num_rhs, const unsigned long long num_mirror_rows, const unsigned long long device_specific_num_rows, const unsigned long long row_offset, const real_type alpha, const real_type *A, const real_type *B, const real_type beta, real_type *C, const unsigned long long grid_x_offset, const unsigned long long grid_y_offset) {
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
    const auto i = (blockIdx_x * blockDim_x + threadIdx_x) * INTERNAL_BLOCK_SIZE_ull;  // # rhs -> num_rhs
    const auto i_linear = blockIdx_x * blockDim_x * INTERNAL_BLOCK_SIZE_ull + threadIdx_x;
    const auto j = (blockIdx_y * blockDim_y + threadIdx_y) * INTERNAL_BLOCK_SIZE_ull;  // # rows -> num_mirror_rows
    const auto j_linear = blockIdx_y * blockDim_y * INTERNAL_BLOCK_SIZE_ull + threadIdx_x;

    // create the shared memory arrays used for caching data point features
    __shared__ real_type A_cache[FEATURE_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];
    __shared__ real_type B_cache[FEATURE_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];

    // create a thread private array used for internal caching
    real_type temp[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE]{};

    // iterate over the remaining features using blocking to be able to cache them for faster memory accesses
    for (unsigned long long dim = 0; dim < device_specific_num_rows; dim += FEATURE_BLOCK_SIZE_ull) {
        // load data into shared memory
        for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
            const auto global_i = i_linear + static_cast<unsigned long long>(internal) * THREAD_BLOCK_SIZE_ull;
            const auto global_j = j_linear + static_cast<unsigned long long>(internal) * THREAD_BLOCK_SIZE_ull;

            // FEATURE_BLOCK_SIZE = 2 * THREAD_BLOCK_SIZE -> store twice as many values in the shared memory
            A_cache[threadIdx.y][internal * THREAD_BLOCK_SIZE + threadIdx.x] = A[(dim + threadIdx_y) * (num_rows - row_offset + PADDING_SIZE_ull) - (dim + threadIdx_y - 1ull) * (dim + threadIdx_y) / 2ull + device_specific_num_rows - (dim + threadIdx_y) + global_j];
            A_cache[threadIdx.y + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + threadIdx.x] = A[(dim + threadIdx_y + THREAD_BLOCK_SIZE_ull) * (num_rows - row_offset + PADDING_SIZE_ull) - (dim + threadIdx_y + THREAD_BLOCK_SIZE_ull - 1ull) * (dim + threadIdx_y + THREAD_BLOCK_SIZE_ull) / 2ull + device_specific_num_rows - (dim + threadIdx_y + THREAD_BLOCK_SIZE_ull) + global_j];
            B_cache[threadIdx.y][internal * THREAD_BLOCK_SIZE + threadIdx.x] = B[(row_offset + dim + threadIdx_y) * (num_rhs + PADDING_SIZE_ull) + global_i];
            B_cache[threadIdx.y + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + threadIdx.x] = B[(row_offset + dim + threadIdx_y + THREAD_BLOCK_SIZE_ull) * (num_rhs + PADDING_SIZE_ull) + global_i];
        }
        __syncthreads();  // wait until all threads loaded their part of the data

        // perform the feature reduction calculation
        for (unsigned block_dim = 0; block_dim < FEATURE_BLOCK_SIZE; ++block_dim) {
            for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                    temp[internal_i][internal_j] += A_cache[block_dim][threadIdx.y * INTERNAL_BLOCK_SIZE + internal_j] * B_cache[block_dim][threadIdx.x * INTERNAL_BLOCK_SIZE + internal_i];
                }
            }
        }
        __syncthreads();  // wait until all threads performed their part of the calculations
    }

    // apply the (remaining) BLAS operation and update C
    for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
        for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
            const auto global_i = i + static_cast<unsigned long long>(internal_i);
            const auto partial_global_j = j + static_cast<unsigned long long>(internal_j);
            const auto global_j = row_offset + device_specific_num_rows + j + static_cast<unsigned long long>(internal_j);

            // be sure to not perform out of bounds accesses
            if (global_i < num_rhs && partial_global_j < num_mirror_rows) {
                C[global_j * (num_rhs + PADDING_SIZE_ull) + global_i] = alpha * temp[internal_i][internal_j] + beta * C[global_j * (num_rhs + PADDING_SIZE_ull) + global_i];
            }
        }
    }
}

/**
 * @brief Perform a simple inplace matrix addition: lhs += rhs.
 * @param[in] num_cols the number of columns in both matrices
 * @param[in,out] lhs the first matrix (updated inplace)
 * @param[in] rhs the second matrix
 */
__global__ void device_kernel_inplace_matrix_add(const unsigned long long num_cols, real_type *lhs, const real_type *rhs, const unsigned long long grid_x_offset, const unsigned long long grid_y_offset) {
    // cast all values to 64-bit unsigned long long to prevent potential 32-bit overflows
    const auto threadIdx_x = static_cast<unsigned long long>(threadIdx.x);                // current thread in block x-dimension
    const auto threadIdx_y = static_cast<unsigned long long>(threadIdx.y);                // current thread in block y-dimension
    const auto blockDim_x = static_cast<unsigned long long>(blockDim.x);                  // number of threads in block x-dimension
    const auto blockDim_y = static_cast<unsigned long long>(blockDim.y);                  // number of threads in block y-dimension
    const auto blockIdx_x = static_cast<unsigned long long>(blockIdx.x) + grid_x_offset;  // current block in grid x-dimension + offsets if the grid size would be too large
    const auto blockIdx_y = static_cast<unsigned long long>(blockIdx.y) + grid_y_offset;  // current block in grid y-dimension + offsets if the grid size would be too large
    const auto INTERNAL_BLOCK_SIZE_ull = static_cast<unsigned long long>(INTERNAL_BLOCK_SIZE);
    const auto PADDING_SIZE_ull = static_cast<unsigned long long>(PADDING_SIZE);

    // calculate the indices used in the current thread
    const auto i = (blockIdx_x * blockDim_x + threadIdx_x) * INTERNAL_BLOCK_SIZE_ull;  // # num_rows
    const auto j = (blockIdx_y * blockDim_y + threadIdx_y) * INTERNAL_BLOCK_SIZE_ull;  // # num_rhs

    for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
        for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
            const auto global_i = i + static_cast<unsigned long long>(internal_i);
            const auto global_j = j + static_cast<unsigned long long>(internal_j);

            lhs[global_i * (num_cols + PADDING_SIZE_ull) + global_j] += rhs[global_i * (num_cols + PADDING_SIZE_ull) + global_j];
        }
    }
}

/**
 * @brief Perform a simple inplace matrix scale: lhs *= scalar.
 * @param[in] num_cols the number of columns in the matrix
 * @param[in,out] lhs the matrix (updated inplace)
 * @param[in] scale the value to scale
 */
__global__ void device_kernel_inplace_matrix_scale(const unsigned long long num_cols, real_type *lhs, const real_type scale, const unsigned long long grid_x_offset, const unsigned long long grid_y_offset) {
    // cast all values to 64-bit unsigned long long to prevent potential 32-bit overflows
    const auto threadIdx_x = static_cast<unsigned long long>(threadIdx.x);                // current thread in block x-dimension
    const auto threadIdx_y = static_cast<unsigned long long>(threadIdx.y);                // current thread in block y-dimension
    const auto blockDim_x = static_cast<unsigned long long>(blockDim.x);                  // number of threads in block x-dimension
    const auto blockDim_y = static_cast<unsigned long long>(blockDim.y);                  // number of threads in block y-dimension
    const auto blockIdx_x = static_cast<unsigned long long>(blockIdx.x) + grid_x_offset;  // current block in grid x-dimension + offsets if the grid size would be too large
    const auto blockIdx_y = static_cast<unsigned long long>(blockIdx.y) + grid_y_offset;  // current block in grid y-dimension + offsets if the grid size would be too large
    const auto INTERNAL_BLOCK_SIZE_ull = static_cast<unsigned long long>(INTERNAL_BLOCK_SIZE);
    const auto PADDING_SIZE_ull = static_cast<unsigned long long>(PADDING_SIZE);

    // calculate the indices used in the current thread
    const auto i = (blockIdx_x * blockDim_x + threadIdx_x) * INTERNAL_BLOCK_SIZE_ull;  // # num_rows
    const auto j = (blockIdx_y * blockDim_y + threadIdx_y) * INTERNAL_BLOCK_SIZE_ull;  // # num_rhs

    for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
        for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
            const auto global_i = i + static_cast<unsigned long long>(internal_i);
            const auto global_j = j + static_cast<unsigned long long>(internal_j);

            lhs[global_i * (num_cols + PADDING_SIZE_ull) + global_j] *= scale;
        }
    }
}

}  // namespace plssvm::hip::detail

#endif  // PLSSVM_BACKENDS_HIP_CG_EXPLICIT_BLAS_HIP_HPP_
