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

#ifndef PLSSVM_BACKENDS_HIP_KERNEL_CG_EXPLICIT_BLAS_HIP_HPP_
#define PLSSVM_BACKENDS_HIP_KERNEL_CG_EXPLICIT_BLAS_HIP_HPP_
#pragma once

#include "plssvm/constants.hpp"  // plssvm::real_type

#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"

#include <cstddef>  // std::size_t

namespace plssvm::hip::detail {

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
    const auto threadIdx_x = static_cast<std::size_t>(threadIdx.x);                // current thread in block x-dimension
    const auto threadIdx_y = static_cast<std::size_t>(threadIdx.y);                // current thread in block y-dimension
    const auto blockDim_x = static_cast<std::size_t>(blockDim.x);                  // number of threads in block x-dimension
    const auto blockDim_y = static_cast<std::size_t>(blockDim.y);                  // number of threads in block y-dimension
    const auto blockIdx_x = static_cast<std::size_t>(blockIdx.x) + grid_x_offset;  // current block in grid x-dimension + offsets if the grid size is too large
    const auto blockIdx_y = static_cast<std::size_t>(blockIdx.y) + grid_y_offset;  // current block in grid y-dimension + offsets if the grid size is too large

    // calculate the indices used in the current thread
    const auto global_i_idx = blockIdx_x * blockDim_x + threadIdx_x;         // num_rhs
    const auto device_global_j_idx = blockIdx_y * blockDim_y + threadIdx_y;  // device_num_rows
    const auto global_j_idx = device_row_offset + device_global_j_idx;

    // be sure to not perform out-of-bounds accesses
    if (global_i_idx < num_rhs && device_global_j_idx < device_num_rows) {
        real_type temp{ 0.0 };

        // iterate over all values
        for (std::size_t dim = 0; dim < (num_rows - device_row_offset); ++dim) {
            real_type A_cache{ 0.0 };
            // determine on which side of the diagonal we are located
            if (dim < device_global_j_idx) {
                A_cache = A[dim * (num_rows - device_row_offset) + device_global_j_idx - dim * (dim + std::size_t{ 1 }) / std::size_t{ 2 }];  // SoA, upper triangular matrix only
            } else {
                A_cache = A[device_global_j_idx * (num_rows - device_row_offset) + dim - device_global_j_idx * (device_global_j_idx + std::size_t{ 1 }) / std::size_t{ 2 }];  // SoA, upper triangular matrix only
            }
            // perform the dot product calculation
            temp += A_cache * B[global_i_idx * num_rows + device_row_offset + dim];  // AoS
        }

        // apply the (partial) BLAS operation and update C
        C[global_i_idx * num_rows + global_j_idx] = alpha * temp + beta * C[global_i_idx * num_rows + global_j_idx];  // AoS
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
    const auto threadIdx_x = static_cast<std::size_t>(threadIdx.x);                // current thread in block x-dimension
    const auto threadIdx_y = static_cast<std::size_t>(threadIdx.y);                // current thread in block y-dimension
    const auto blockDim_x = static_cast<std::size_t>(blockDim.x);                  // number of threads in block x-dimension
    const auto blockDim_y = static_cast<std::size_t>(blockDim.y);                  // number of threads in block y-dimension
    const auto blockIdx_x = static_cast<std::size_t>(blockIdx.x) + grid_x_offset;  // current block in grid x-dimension + offsets if the grid size is too large
    const auto blockIdx_y = static_cast<std::size_t>(blockIdx.y) + grid_y_offset;  // current block in grid y-dimension + offsets if the grid size is too large

    // calculate the indices used in the current thread
    const auto global_i_idx = blockIdx_x * blockDim_x + threadIdx_x;          // num_rhs
    const auto partial_global_j_idx = blockIdx_y * blockDim_y + threadIdx_y;  // num_mirror_rows
    const auto global_j_idx = device_row_offset + device_num_rows + partial_global_j_idx;

    // be sure to not perform out-of-bounds accesses
    if (global_i_idx < num_rhs && partial_global_j_idx < num_mirror_rows && global_j_idx < num_rows) {
        real_type temp{ 0.0 };

        // iterate over all values
        for (std::size_t dim = 0; dim < device_num_rows; ++dim) {
            // perform the dot product calculation
            temp += A[dim * (num_rows - device_row_offset) - (dim - std::size_t{ 1 }) * dim / std::size_t{ 2 } + device_num_rows - dim + partial_global_j_idx] *  // SoA, upper triangular matrix only
                    B[global_i_idx * num_rows + device_row_offset + dim];                                                                                         // AoS
        }

        // apply the (remaining) BLAS operation and update C
        C[global_i_idx * num_rows + global_j_idx] = alpha * temp + beta * C[global_i_idx * num_rows + global_j_idx];  // AoS
    }
}

/**
 * @brief Perform a simple inplace matrix addition: lhs += rhs.
 * @param[in] num_rows the number of rows in both matrices
 * @param[in] num_cols the number of columns in both matrices
 * @param[in,out] lhs the first matrix (updated inplace)
 * @param[in] rhs the second matrix
 * @param[in] grid_x_offset the offset in x-dimension into the data points if more than one execution grid has to be used
 * @param[in] grid_y_offset the offset in y-dimension into the data points if more than one execution grid has to be used
 */
__global__ void device_kernel_inplace_matrix_add(const std::size_t num_rows, const std::size_t num_cols, real_type *lhs, const real_type *rhs, const std::size_t grid_x_offset, const std::size_t grid_y_offset) {
    // cast all values to 64-bit std::size_t to prevent potential 32-bit overflows
    const auto threadIdx_x = static_cast<std::size_t>(threadIdx.x);                // current thread in block x-dimension
    const auto threadIdx_y = static_cast<std::size_t>(threadIdx.y);                // current thread in block y-dimension
    const auto blockDim_x = static_cast<std::size_t>(blockDim.x);                  // number of threads in block x-dimension
    const auto blockDim_y = static_cast<std::size_t>(blockDim.y);                  // number of threads in block y-dimension
    const auto blockIdx_x = static_cast<std::size_t>(blockIdx.x) + grid_x_offset;  // current block in grid x-dimension + offsets if the grid size is too large
    const auto blockIdx_y = static_cast<std::size_t>(blockIdx.y) + grid_y_offset;  // current block in grid y-dimension + offsets if the grid size is too large

    // calculate the indices used in the current thread
    const auto global_i_idx = blockIdx_x * blockDim_x + threadIdx_x;  // num_rows
    const auto global_j_idx = blockIdx_y * blockDim_y + threadIdx_y;  // num_rhs

    if (global_i_idx < num_rows && global_j_idx < num_cols) {
        lhs[global_j_idx * num_rows + global_i_idx] += rhs[global_j_idx * num_rows + global_i_idx];  // AoS
    }
}

/**
 * @brief Perform a simple inplace matrix scale: lhs *= scalar.
 * @param[in] num_rows the number of rows in the matrix
 * @param[in] num_cols the number of columns in the matrix
 * @param[in,out] lhs the matrix (updated inplace)
 * @param[in] scale the value to scale
 * @param[in] grid_x_offset the offset in x-dimension into the data points if more than one execution grid has to be used
 * @param[in] grid_y_offset the offset in y-dimension into the data points if more than one execution grid has to be used
 */
__global__ void device_kernel_inplace_matrix_scale(const std::size_t num_rows, const std::size_t num_cols, real_type *lhs, const real_type scale, const std::size_t grid_x_offset, const std::size_t grid_y_offset) {
    // cast all values to 64-bit std::size_t to prevent potential 32-bit overflows
    const auto threadIdx_x = static_cast<std::size_t>(threadIdx.x);                // current thread in block x-dimension
    const auto threadIdx_y = static_cast<std::size_t>(threadIdx.y);                // current thread in block y-dimension
    const auto blockDim_x = static_cast<std::size_t>(blockDim.x);                  // number of threads in block x-dimension
    const auto blockDim_y = static_cast<std::size_t>(blockDim.y);                  // number of threads in block y-dimension
    const auto blockIdx_x = static_cast<std::size_t>(blockIdx.x) + grid_x_offset;  // current block in grid x-dimension + offsets if the grid size is too large
    const auto blockIdx_y = static_cast<std::size_t>(blockIdx.y) + grid_y_offset;  // current block in grid y-dimension + offsets if the grid size is too large

    // calculate the indices used in the current thread
    const auto global_i_idx = blockIdx_x * blockDim_x + threadIdx_x;  // num_rows
    const auto global_j_idx = blockIdx_y * blockDim_y + threadIdx_y;  // num_rhs

    if (global_i_idx < num_rows && global_j_idx < num_cols) {
        lhs[global_j_idx * num_rows + global_i_idx] *= scale;  // AoS
    }
}

}  // namespace plssvm::hip::detail

#endif  // PLSSVM_BACKENDS_HIP_KERNEL_CG_EXPLICIT_BLAS_HIP_HPP_
