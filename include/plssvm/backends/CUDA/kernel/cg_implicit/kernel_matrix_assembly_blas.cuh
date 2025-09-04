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
#include "plssvm/constants.hpp"                              // plssvm::real_type, plssvm::THREAD_BLOCK_SIZE
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
    constexpr auto THREAD_BLOCK_SIZE_uz = static_cast<std::size_t>(THREAD_BLOCK_SIZE);

    const auto threadIdx_x = static_cast<std::size_t>(threadIdx.x);                // current thread in block x-dimension
    const auto threadIdx_y = static_cast<std::size_t>(threadIdx.y);                // current thread in block y-dimension
    const auto blockDim_x = static_cast<std::size_t>(blockDim.x);                  // number of threads in block x-dimension
    const auto blockDim_y = static_cast<std::size_t>(blockDim.y);                  // number of threads in block y-dimension
    const auto blockIdx_x = static_cast<std::size_t>(blockIdx.x) + grid_x_offset;  // current block in grid x-dimension + offsets if the grid size is too large
    const auto blockIdx_y = static_cast<std::size_t>(blockIdx.y) + grid_y_offset;  // current block in grid y-dimension + offsets if the grid size is too large

    // calculate the indices used in the current thread
    const auto i_idx = blockIdx_x * blockDim_x + threadIdx_x;  // num_rows - device_row_offset
    const auto j_idx = blockIdx_y * blockDim_y + threadIdx_y;  // device_num_rows

    // calculate the indices used in the current thread, pays attention to coalesced memory accesses
    const auto i_idx_linear = blockIdx_x * blockDim_x + threadIdx_x;  // num_rows - device_row_offset
    const auto j_idx_linear = blockIdx_y * blockDim_y + threadIdx_x;  // device_num_rows

    // create two shared memory arrays used for caching
    __shared__ real_type cache_one[THREAD_BLOCK_SIZE][THREAD_BLOCK_SIZE];
    __shared__ real_type cache_two[THREAD_BLOCK_SIZE][THREAD_BLOCK_SIZE];

    // only calculate the upper triangular matrix -> can't use threadIdx since all threads in a warp must progress further
    if (blockIdx_x >= blockIdx_y) {
        real_type temp{ 0.0 };

        //*************************************************************************//
        //                   inplace kernel matrix construction                    //
        //*************************************************************************//
        {
            // rename the shared memory array
            auto data_i_cache = cache_one;
            auto data_j_cache = cache_two;

            // calculate the indices to access the global data, pays attention to coalesced memory accesses
            const auto global_i_idx_linear = device_row_offset + i_idx_linear;
            const auto global_j_idx_linear = device_row_offset + j_idx_linear;

            // iterate over all features using blocking to be able to cache them for faster memory accesses
            for (std::size_t feature_block = 0; feature_block < num_features; feature_block += THREAD_BLOCK_SIZE_uz) {
                // zero-out shared memory
                data_i_cache[threadIdx.y][threadIdx.x] = real_type{ 0.0 };
                data_j_cache[threadIdx.y][threadIdx.x] = real_type{ 0.0 };

                // load data into shared memory
                if (feature_block + threadIdx_y < num_features) {
                    if (global_i_idx_linear < num_rows) {
                        data_i_cache[threadIdx.y][threadIdx.x] = data[(feature_block + threadIdx_y) * (num_rows + std::size_t{ 1 }) + global_i_idx_linear];  // SoA
                    }
                    if (global_j_idx_linear < num_rows) {
                        data_j_cache[threadIdx.y][threadIdx.x] = data[(feature_block + threadIdx_y) * (num_rows + std::size_t{ 1 }) + global_j_idx_linear];  // SoA
                    }
                }
                __syncthreads();  // wait until all threads loaded their part of the data

                // perform the feature reduction calculation
                for (unsigned feature = 0; feature < THREAD_BLOCK_SIZE; ++feature) {
                    temp += detail::feature_reduce<kernel_function>(data_i_cache[feature][threadIdx.x],
                                                                    data_j_cache[feature][threadIdx.y]);
                }
                __syncthreads();  // wait until all threads performed their part of the calculations
            }
        }

        // calculate the indices to access the global data and the data with respect to the current device
        const auto global_i_idx = device_row_offset + i_idx;
        const auto global_j_idx = device_row_offset + j_idx;

        // be sure to not perform out-of-bounds accesses (only using the upper triangular matrix)
        if (i_idx < (num_rows - device_row_offset) && j_idx < device_num_rows && global_i_idx >= global_j_idx) {
            // apply the final kernel function
            temp = detail::apply_kernel_function<kernel_function>(temp, kernel_function_parameter...) + QA_cost - q[global_i_idx] - q[global_j_idx];
            // apply the cost on the diagonal
            if (global_i_idx == global_j_idx) {
                temp += cost;
            }
        } else {
            // be sure to set the value to zero otherwise
            temp = real_type{ 0.0 };
        }

        //*************************************************************************//
        //     calculate C += alpha * temp * B for the UPPER triangular matrix     //
        //*************************************************************************//
        {
            // rename the shared memory array
            auto B_cache = cache_one;
            auto C_out_cache = cache_two;

            // calculate the indices to access the global data, pays attention to coalesced memory accesses
            const auto global_i_idx_linear = device_row_offset + i_idx_linear;

            // iterate over all classes using blocking to be able to cache them for faster memory accesses
            for (std::size_t class_block = 0; class_block < num_classes; class_block += THREAD_BLOCK_SIZE_uz) {
                // zero-out shared memory
                B_cache[threadIdx.y][threadIdx.x] = real_type{ 0.0 };
                C_out_cache[threadIdx.y][threadIdx.x] = real_type{ 0.0 };
                __syncthreads();  // wait until all threads set the values to zero (necessary since the index order changes in the next load operation)

                // load data into shared memory
                if (class_block + threadIdx_y < num_classes && global_i_idx_linear < num_rows) {
                    B_cache[threadIdx.x][threadIdx.y] = alpha * B[global_i_idx_linear * num_classes + class_block + threadIdx_y];  // SoA
                }
                __syncthreads();  // wait until all threads loaded their part of the data

                // calculate intermediate results and store them in shared memory
                for (unsigned class_idx = 0; class_idx < THREAD_BLOCK_SIZE; ++class_idx) {
                    C_out_cache[threadIdx.y][(class_idx + threadIdx.x) % THREAD_BLOCK_SIZE] += temp * B_cache[threadIdx.x][(class_idx + threadIdx.x) % THREAD_BLOCK_SIZE];
                    __syncthreads();  // wait until all threads performed their part of the calculations
                }

                // atomically add the intermediate cached results to the C matrix
                if (class_block + threadIdx_x < num_classes && global_j_idx < num_rows) {
                    atomicAdd(&C[global_j_idx * num_classes + class_block + threadIdx_x], C_out_cache[threadIdx.y][threadIdx.x]);  // SoA
                }
                __syncthreads();  // wait until all threads updated C with their values
            }
        }

        // set potential diagonal entries in temp to 0.0 such that we don't apply the main diagonal twice to C
        // update the diagonal
        if (global_i_idx == global_j_idx) {
            temp = real_type{ 0.0 };
        }

        //*************************************************************************//
        //     calculate C += alpha * temp * B for the LOWER triangular matrix     //
        //*************************************************************************//
        {
            // rename the shared memory array
            auto B_cache = cache_one;
            auto C_out_cache = cache_two;

            // calculate the indices to access the global data, pays attention to coalesced memory accesses
            const auto global_j_idx_linear = device_row_offset + j_idx_linear;

            // iterate over all classes using blocking to be able to cache them for faster memory accesses
            for (std::size_t class_block = 0; class_block < num_classes; class_block += THREAD_BLOCK_SIZE_uz) {
                // zero-out shared memory
                B_cache[threadIdx.y][threadIdx.x] = real_type{ 0.0 };
                C_out_cache[threadIdx.y][threadIdx.x] = real_type{ 0.0 };

                // load data into shared memory
                if (class_block + threadIdx_y < num_classes && global_j_idx_linear < num_rows) {
                    B_cache[threadIdx.y][threadIdx.x] = alpha * B[global_j_idx_linear * num_classes + class_block + threadIdx_y];  // SoA
                }
                __syncthreads();  // wait until all threads loaded their part of the data

                // calculate intermediate results and store them in shared memory
                for (unsigned class_idx = 0; class_idx < THREAD_BLOCK_SIZE; ++class_idx) {
                    C_out_cache[(class_idx + threadIdx.y) % THREAD_BLOCK_SIZE][threadIdx.x] += temp * B_cache[(class_idx + threadIdx.y) % THREAD_BLOCK_SIZE][threadIdx.y];
                    __syncthreads();  // wait until all threads performed their part of the calculations
                }

                // atomically add the intermediate cached results to the C matrix
                if (class_block + threadIdx_y < num_classes && global_i_idx < num_rows) {
                    atomicAdd(&C[global_i_idx * num_classes + class_block + threadIdx_y], C_out_cache[threadIdx.y][threadIdx.x]);  // SoA
                }
                __syncthreads();  // wait until all threads updated C with their values
            }
        }
    }
}

}  // namespace plssvm::cuda::detail

#endif  // PLSSVM_BACKENDS_CUDA_KERNEL_CG_IMPLICIT_KERNEL_MATRIX_ASSEMBLY_BLAS_CUH_
