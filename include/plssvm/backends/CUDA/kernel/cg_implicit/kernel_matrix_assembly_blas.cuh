/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Functions for implicitly assembling the kernel matrix using the CUDA backend.
 */

#ifndef PLSSVM_BACKENDS_CUDA_KERNEL_CG_IMPLICIT_KERNEL_MATRIX_ASSEMBLY_BLAS_CUH_
#define PLSSVM_BACKENDS_CUDA_KERNEL_CG_IMPLICIT_KERNEL_MATRIX_ASSEMBLY_BLAS_CUH_
#pragma once

#include "plssvm/backends/CUDA/kernel/detail/atomics.cuh"    // atomicAdd for double precision floating point numbers on older CUDA hardware
#include "plssvm/backends/CUDA/kernel/kernel_functions.cuh"  // plssvm::cuda::detail::{feature_reduce, apply_kernel_function}
#include "plssvm/constants.hpp"                              // plssvm::real_type
#include "plssvm/kernel_function_types.hpp"                  // plssvm::kernel_function_type

#include <cstddef>  // std::size_t

namespace plssvm::cuda::detail {

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
    const auto threadIdx_x = static_cast<std::size_t>(threadIdx.x);                // current thread in block x-dimension
    const auto threadIdx_y = static_cast<std::size_t>(threadIdx.y);                // current thread in block y-dimension
    const auto blockDim_x = static_cast<std::size_t>(blockDim.x);                  // number of threads in block x-dimension
    const auto blockDim_y = static_cast<std::size_t>(blockDim.y);                  // number of threads in block y-dimension
    const auto blockIdx_x = static_cast<std::size_t>(blockIdx.x) + grid_x_offset;  // current block in grid x-dimension + offsets if the grid size is too large
    const auto blockIdx_y = static_cast<std::size_t>(blockIdx.y) + grid_y_offset;  // current block in grid y-dimension + offsets if the grid size is too large

    // calculate the indices used in the current thread
    const auto device_global_i_idx = blockIdx_x * blockDim_x + threadIdx_x;
    const auto global_i_idx = device_row_offset + device_global_i_idx;
    const auto device_global_j_idx = blockIdx_y * blockDim_y + threadIdx_y;
    const auto global_j_idx = device_row_offset + device_global_j_idx;

    // be sure to not perform out-of-bounds accesses (only using the upper triangular matrix)
    if (device_global_i_idx < (num_rows - device_row_offset) && device_global_j_idx < device_num_rows && global_i_idx >= global_j_idx) {
        //*************************************************************************//
        //                   inplace kernel matrix construction                    //
        //*************************************************************************//
        real_type temp{ 0.0 };
        // perform the feature reduction calculation
        for (std::size_t feature = 0; feature < num_features; ++feature) {
            temp += detail::feature_reduce<kernel_function>(data[feature * (num_rows + std::size_t{ 1 }) + global_i_idx],
                                                            data[feature * (num_rows + std::size_t{ 1 }) + global_j_idx]);
        }

        // apply the final kernel function
        temp = detail::apply_kernel_function<kernel_function>(temp, kernel_function_parameter...) + QA_cost - q[global_i_idx] - q[global_j_idx];
        // apply the cost on the diagonal
        if (global_i_idx == global_j_idx) {
            temp += cost;
        }

        //*************************************************************************//
        //     calculate C += alpha * temp * B for the UPPER triangular matrix     //
        //*************************************************************************//
        for (std::size_t class_idx = 0; class_idx < num_classes; ++class_idx) {
            const real_type B_cache = alpha * B[global_i_idx * num_classes + class_idx];
            atomicAdd(&C[global_j_idx * num_classes + class_idx], temp * B_cache);
        }

        // set potential diagonal entries in temp to 0.0 such that we don't apply the main diagonal twice to C
        if (global_i_idx == global_j_idx) {
            temp = real_type{ 0.0 };
        }

        //*************************************************************************//
        //     calculate C += alpha * temp * B for the LOWER triangular matrix     //
        //*************************************************************************//
        for (std::size_t class_idx = 0; class_idx < num_classes; ++class_idx) {
            const real_type B_cache = alpha * B[global_j_idx * num_classes + class_idx];
            atomicAdd(&C[global_i_idx * num_classes + class_idx], temp * B_cache);
        }
    }
}

}  // namespace plssvm::cuda::detail

#endif  // PLSSVM_BACKENDS_CUDA_KERNEL_CG_IMPLICIT_KERNEL_MATRIX_ASSEMBLY_BLAS_CUH_
