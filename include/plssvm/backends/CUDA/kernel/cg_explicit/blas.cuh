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

#include "plssvm/constants.hpp"  // plssvm::{real_type, THREAD_BLOCK_SIZE, INTERNAL_BLOCK_SIZE, PADDING_SIZE}

#include <cstddef>  // std::size_t

namespace plssvm::cuda::detail {

/**
 * @brief Perform an explicit BLAS SYMM operation: `C = alpha * A * B + beta * C` where @p A is a `m x k` symmetric matrix (memory optimized), @p B is a `k x n` matrix, @p C is a `m x n` matrix, and @p alpha and @p beta are scalars.
 * @details In a multi-GPU setting, this function is only responsible for the rows this device is responsible for!
 * @param[in] num_rows the number of rows in @p A and @p C
 * @param[in] num_rhs the number of columns in @p B and @p C
 * @param[in] device_num_rows the number of rows in @p A and number of rows in @p B; thr rows in @p A are potentially distributed across multiple devices
 * @param[in] device_row_offset the first row this device is responsible for
 * @param[in] alpha the scalar alpha value
 * @param[in] A the matrix @p A
 * @param[in] B the matrix @p B
 * @param[in] beta the scalar beta value
 * @param[in,out] C the matrix @p C, also used as result matrix
 * @param[in] grid_x_offset the offset in x-dimension into the data points if more than one execution grid has to be used
 * @param[in] grid_y_offset the offset in y-dimension into the data points if more than one execution grid has to be used
 */
__global__ void device_kernel_symm(const std::size_t num_rows, const std::size_t num_rhs, const std::size_t device_num_rows, const std::size_t device_row_offset, const real_type alpha, const real_type *A, const real_type *B, const real_type beta, real_type *C, const std::size_t grid_x_offset, const std::size_t grid_y_offset) {
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
    __shared__ real_type A_cache[THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];
    __shared__ real_type B_cache[THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];

    // create a thread private array used for internal caching
    real_type temp[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE]{};

    {
        // calculate the indices used in the current thread, pays attention to coalesced memory accesses
        const auto i_idx_linear = blockIdx_x * blockDim_x * INTERNAL_BLOCK_SIZE_uz + threadIdx_x;  // num_rhs
        const auto j_idx_linear = blockIdx_y * blockDim_y * INTERNAL_BLOCK_SIZE_uz + threadIdx_x;  // device_num_rows

        // iterate over all values using blocking to be able to cache them for faster memory accesses
        for (std::size_t dim_block = 0; dim_block < (num_rows - device_row_offset); dim_block += THREAD_BLOCK_SIZE_uz) {
            // load data into shared memory
            for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                // calculate the indices to access the global data, pays attention to coalesced memory accesses
                const auto global_i_idx_linear = i_idx_linear + static_cast<std::size_t>(internal) * THREAD_BLOCK_SIZE_uz;
                const auto global_j_idx_linear = j_idx_linear + static_cast<std::size_t>(internal) * THREAD_BLOCK_SIZE_uz;

                // store the values in the shared memory
                // determine on which side of the diagonal we are located
                if (dim_block + threadIdx_y < global_j_idx_linear) {
                    A_cache[threadIdx.y][internal * THREAD_BLOCK_SIZE + threadIdx.x] = A[(dim_block + threadIdx_y) * (num_rows - device_row_offset + PADDING_SIZE_uz) + global_j_idx_linear - (dim_block + threadIdx_y) * (dim_block + threadIdx_y + std::size_t{ 1 }) / std::size_t{ 2 }];  // SoA, upper triangular matrix only
                } else {
                    A_cache[threadIdx.y][internal * THREAD_BLOCK_SIZE + threadIdx.x] = A[global_j_idx_linear * (num_rows - device_row_offset + PADDING_SIZE_uz) + dim_block + threadIdx_y - global_j_idx_linear * (global_j_idx_linear + std::size_t{ 1 }) / std::size_t{ 2 }];  // SoA, upper triangular matrix only
                }
                B_cache[threadIdx.y][internal * THREAD_BLOCK_SIZE + threadIdx.x] = B[(dim_block + device_row_offset + threadIdx_y) * (num_rhs + PADDING_SIZE_uz) + global_i_idx_linear];  // SoA
            }
            __syncthreads();  // wait until all threads loaded their part of the data

            // perform the dot product calculation
            for (unsigned dim = 0; dim < THREAD_BLOCK_SIZE; ++dim) {
                for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                    for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                        temp[internal_i][internal_j] += A_cache[dim][threadIdx.y * INTERNAL_BLOCK_SIZE + internal_j] * B_cache[dim][threadIdx.x * INTERNAL_BLOCK_SIZE + internal_i];
                    }
                }
            }
            __syncthreads();  // wait until all threads performed their part of the calculations
        }
    }

    // calculate the indices used in the current thread
    const auto i_idx = (blockIdx_x * blockDim_x + threadIdx_x) * INTERNAL_BLOCK_SIZE_uz;  // num_rhs
    const auto j_idx = (blockIdx_y * blockDim_y + threadIdx_y) * INTERNAL_BLOCK_SIZE_uz;  // device_num_rows

    // apply the (partial) BLAS operation and update C
    for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
        for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
            // calculate the indices to access the global data and the data with respect to the current device
            const auto global_i_idx = i_idx + static_cast<std::size_t>(internal_i);
            const auto device_global_j_idx = j_idx + static_cast<std::size_t>(internal_j);
            const auto global_j_idx = device_row_offset + device_global_j_idx;

            // be sure to not perform out-of-bounds accesses
            if (global_i_idx < num_rhs && device_global_j_idx < device_num_rows) {
                C[global_j_idx * (num_rhs + PADDING_SIZE_uz) + global_i_idx] = alpha * temp[internal_i][internal_j] + beta * C[global_j_idx * (num_rhs + PADDING_SIZE_uz) + global_i_idx];  // SoA
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
 * @param[in] device_num_rows the number of rows in @p A and number of rows in @p B; thr rows in @p A are potentially distributed across multiple devices
 * @param[in] device_row_offset the first row this device is responsible for
 * @param[in] alpha the scalar alpha value
 * @param[in] A the matrix @p A
 * @param[in] B the matrix @p B
 * @param[in] beta the scalar beta value
 * @param[in,out] C the matrix @p C, also used as result matrix
 * @param[in] grid_x_offset the offset in x-dimension into the data points if more than one execution grid has to be used
 * @param[in] grid_y_offset the offset in y-dimension into the data points if more than one execution grid has to be used
 */
__global__ void device_kernel_symm_mirror(const std::size_t num_rows, const std::size_t num_rhs, const std::size_t num_mirror_rows, const std::size_t device_num_rows, const std::size_t device_row_offset, const real_type alpha, const real_type *A, const real_type *B, const real_type beta, real_type *C, const std::size_t grid_x_offset, const std::size_t grid_y_offset) {
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
    __shared__ real_type A_cache[THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];
    __shared__ real_type B_cache[THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];

    // create a thread private array used for internal caching
    real_type temp[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE]{};

    {
        // calculate the indices used in the current thread, pays attention to coalesced memory accesses
        const auto i_idx_linear = blockIdx_x * blockDim_x * INTERNAL_BLOCK_SIZE_uz + threadIdx_x;  // num_rhs
        const auto j_idx_linear = blockIdx_y * blockDim_y * INTERNAL_BLOCK_SIZE_uz + threadIdx_x;  // num_mirror_rows

        // iterate over the remaining values using blocking to be able to cache them for faster memory accesses
        for (std::size_t dim_block = 0; dim_block < device_num_rows; dim_block += THREAD_BLOCK_SIZE_uz) {
            // load data into shared memory
            for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                // calculate the indices to access the global data, pays attention to coalesced memory accesses
                const auto global_i_idx_linear = i_idx_linear + static_cast<std::size_t>(internal) * THREAD_BLOCK_SIZE_uz;
                const auto global_j_idx_linear = j_idx_linear + static_cast<std::size_t>(internal) * THREAD_BLOCK_SIZE_uz;

                // store the values in the shared memory
                A_cache[threadIdx.y][internal * THREAD_BLOCK_SIZE + threadIdx.x] = A[(dim_block + threadIdx_y) * (num_rows - device_row_offset + PADDING_SIZE_uz) - (dim_block + threadIdx_y - std::size_t{ 1 }) * (dim_block + threadIdx_y) / std::size_t{ 2 } + device_num_rows - (dim_block + threadIdx_y) + global_j_idx_linear];  // SoA, upper triangular matrix only
                B_cache[threadIdx.y][internal * THREAD_BLOCK_SIZE + threadIdx.x] = B[(device_row_offset + dim_block + threadIdx_y) * (num_rhs + PADDING_SIZE_uz) + global_i_idx_linear];                                                                                                                                               // SoA
            }
            __syncthreads();  // wait until all threads loaded their part of the data

            // perform the dot product calculation
            for (unsigned dim = 0; dim < THREAD_BLOCK_SIZE; ++dim) {
                for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                    for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                        temp[internal_i][internal_j] += A_cache[dim][threadIdx.y * INTERNAL_BLOCK_SIZE + internal_j] * B_cache[dim][threadIdx.x * INTERNAL_BLOCK_SIZE + internal_i];
                    }
                }
            }
            __syncthreads();  // wait until all threads performed their part of the calculations
        }
    }

    // calculate the indices used in the current thread
    const auto i_idx = (blockIdx_x * blockDim_x + threadIdx_x) * INTERNAL_BLOCK_SIZE_uz;  // num_rhs
    const auto j_idx = (blockIdx_y * blockDim_y + threadIdx_y) * INTERNAL_BLOCK_SIZE_uz;  // num_mirror_rows

    // apply the (remaining) BLAS operation and update C
    for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
        for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
            // calculate the indices to access the global data and the data with respect to the current device
            const auto global_i_idx = i_idx + static_cast<std::size_t>(internal_i);
            const auto partial_global_j_idx = j_idx + static_cast<std::size_t>(internal_j);
            const auto global_j_idx = device_row_offset + device_num_rows + partial_global_j_idx;

            // be sure to not perform out-of-bounds accesses
            if (global_i_idx < num_rhs && partial_global_j_idx < num_mirror_rows) {
                C[global_j_idx * (num_rhs + PADDING_SIZE_uz) + global_i_idx] = alpha * temp[internal_i][internal_j] + beta * C[global_j_idx * (num_rhs + PADDING_SIZE_uz) + global_i_idx];  // SoA
            }
        }
    }
}

/**
 * @brief Perform a simple inplace matrix addition: lhs += rhs.
 * @param[in] num_cols the number of columns in both matrices
 * @param[in,out] lhs the first matrix (updated inplace)
 * @param[in] rhs the second matrix
 * @param[in] grid_x_offset the offset in x-dimension into the data points if more than one execution grid has to be used
 * @param[in] grid_y_offset the offset in y-dimension into the data points if more than one execution grid has to be used
 */
__global__ void device_kernel_inplace_matrix_add(const std::size_t num_cols, real_type *lhs, const real_type *rhs, const std::size_t grid_x_offset, const std::size_t grid_y_offset) {
    // cast all values to 64-bit std::size_t to prevent potential 32-bit overflows
    constexpr auto INTERNAL_BLOCK_SIZE_uz = static_cast<std::size_t>(INTERNAL_BLOCK_SIZE);
    constexpr auto PADDING_SIZE_uz = static_cast<std::size_t>(PADDING_SIZE);

    const auto threadIdx_x = static_cast<std::size_t>(threadIdx.x);                // current thread in block x-dimension
    const auto threadIdx_y = static_cast<std::size_t>(threadIdx.y);                // current thread in block y-dimension
    const auto blockDim_x = static_cast<std::size_t>(blockDim.x);                  // number of threads in block x-dimension
    const auto blockDim_y = static_cast<std::size_t>(blockDim.y);                  // number of threads in block y-dimension
    const auto blockIdx_x = static_cast<std::size_t>(blockIdx.x) + grid_x_offset;  // current block in grid x-dimension + offsets if the grid size is too large
    const auto blockIdx_y = static_cast<std::size_t>(blockIdx.y) + grid_y_offset;  // current block in grid y-dimension + offsets if the grid size is too large

    // calculate the indices used in the current thread
    const auto i_idx = (blockIdx_x * blockDim_x + threadIdx_x) * INTERNAL_BLOCK_SIZE_uz;  // num_rows
    const auto j_idx = (blockIdx_y * blockDim_y + threadIdx_y) * INTERNAL_BLOCK_SIZE_uz;  // num_rhs

    for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
        for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
            // calculate the indices to access the global data
            const auto global_i_idx = i_idx + static_cast<std::size_t>(internal_i);
            const auto global_j_idx = j_idx + static_cast<std::size_t>(internal_j);

            lhs[global_i_idx * (num_cols + PADDING_SIZE_uz) + global_j_idx] += rhs[global_i_idx * (num_cols + PADDING_SIZE_uz) + global_j_idx];  // SoA
        }
    }
}

/**
 * @brief Perform a simple inplace matrix scale: lhs *= scalar.
 * @param[in] num_cols the number of columns in the matrix
 * @param[in,out] lhs the matrix (updated inplace)
 * @param[in] scale the value to scale
 * @param[in] grid_x_offset the offset in x-dimension into the data points if more than one execution grid has to be used
 * @param[in] grid_y_offset the offset in y-dimension into the data points if more than one execution grid has to be used
 */
__global__ void device_kernel_inplace_matrix_scale(const std::size_t num_cols, real_type *lhs, const real_type scale, const std::size_t grid_x_offset, const std::size_t grid_y_offset) {
    // cast all values to 64-bit std::size_t to prevent potential 32-bit overflows
    constexpr auto INTERNAL_BLOCK_SIZE_uz = static_cast<std::size_t>(INTERNAL_BLOCK_SIZE);
    constexpr auto PADDING_SIZE_uz = static_cast<std::size_t>(PADDING_SIZE);

    const auto threadIdx_x = static_cast<std::size_t>(threadIdx.x);                // current thread in block x-dimension
    const auto threadIdx_y = static_cast<std::size_t>(threadIdx.y);                // current thread in block y-dimension
    const auto blockDim_x = static_cast<std::size_t>(blockDim.x);                  // number of threads in block x-dimension
    const auto blockDim_y = static_cast<std::size_t>(blockDim.y);                  // number of threads in block y-dimension
    const auto blockIdx_x = static_cast<std::size_t>(blockIdx.x) + grid_x_offset;  // current block in grid x-dimension + offsets if the grid size is too large
    const auto blockIdx_y = static_cast<std::size_t>(blockIdx.y) + grid_y_offset;  // current block in grid y-dimension + offsets if the grid size is too large

    // calculate the indices used in the current thread
    const auto i_idx = (blockIdx_x * blockDim_x + threadIdx_x) * INTERNAL_BLOCK_SIZE_uz;  // num_rows
    const auto j_idx = (blockIdx_y * blockDim_y + threadIdx_y) * INTERNAL_BLOCK_SIZE_uz;  // num_rhs

    for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
        for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
            // calculate the indices to access the global data
            const auto global_i_idx = i_idx + static_cast<std::size_t>(internal_i);
            const auto global_j_idx = j_idx + static_cast<std::size_t>(internal_j);

            lhs[global_i_idx * (num_cols + PADDING_SIZE_uz) + global_j_idx] *= scale;  // SoA
        }
    }
}

}  // namespace plssvm::cuda::detail

#endif  // PLSSVM_BACKENDS_CUDA_KERNEL_CG_EXPLICIT_BLAS_CUH_
