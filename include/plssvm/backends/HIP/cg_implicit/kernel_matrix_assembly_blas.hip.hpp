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

#include "plssvm/constants.hpp"  // plssvm::{real_type, THREAD_BLOCK_SIZE, INTERNAL_BLOCK_SIZE, FEATURE_BLOCK_SIZE, PADDING_SIZE}

#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"

namespace plssvm::hip {

/**
 * @brief Perform an implicit BLAS SYMM-like operation: `C = alpha * A * B + C` where `A` is the implicitly calculated kernel matrix using the linear kernel function \f$\vec{u}^T \cdot \vec{v}\f$ (never actually stored, reducing the amount of needed global memory), @p B and @p C are matrices, and @p alpha is a scalar.
 * @note The beta factor is already applied to C before this kernel starts!
 * @param[in] alpha the scalar alpha value
 * @param[in] q the vector used in the dimensional reduction
 * @param[in] data_d the data points to calculate the implicit kernel matrix from
 * @param[in] num_rows the number of data points
 * @param[in] num_features the number of features per data point
 * @param[in] QA_cost the scalar used in the dimensional reduction
 * @param[in] cost the cost factor the diagonal is scaled with
 * @param[in] B the matrix @p B
 * @param[in,out] C the matrix @p C
 * @param[in] num_classes the number of classes in the data set
 */
__global__ void device_kernel_assembly_linear_symm(const real_type alpha, const real_type *q, const real_type *data_d, const unsigned long long num_rows, const unsigned long long num_features, const real_type QA_cost, const real_type cost, const real_type *B, real_type *C, const unsigned long long num_classes) {
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

                    // apply B and C
                    for (unsigned long long class_idx = 0; class_idx < num_classes; ++class_idx) {
                        atomicAdd(&C[global_i * (num_classes + PADDING_SIZE) + class_idx], alpha * temp_ij * B[global_j * (num_classes + PADDING_SIZE) + class_idx]);
                        if (global_i != global_j) {
                            atomicAdd(&C[global_j * (num_classes + PADDING_SIZE) + class_idx], alpha * temp_ij * B[global_i * (num_classes + PADDING_SIZE) + class_idx]);
                        }
                    }
                }
            }
        }
    }
}

/**
 * @brief Perform an implicit BLAS SYMM-like operation: `C = alpha * A * B + C` where `A` is the implicitly calculated kernel matrix using the polynomial kernel function \f$(gamma \cdot \vec{u}^T \cdot \vec{v} + coef0)^{degree}\f$ (never actually stored, reducing the amount of needed global memory), @p B and @p C are matrices, and @p alpha is a scalar.
 * @note The beta factor is already applied to C before this kernel starts!
 * @param[in] alpha the scalar alpha value
 * @param[in] q the vector used in the dimensional reduction
 * @param[in] data_d the data points to calculate the implicit kernel matrix from
 * @param[in] num_rows the number of data points
 * @param[in] num_features the number of features per data point
 * @param[in] QA_cost the scalar used in the dimensional reduction
 * @param[in] cost the cost factor the diagonal is scaled with
 * @param[in] degree parameter used in the polynomial kernel function
 * @param[in] gamma parameter used in the polynomial kernel function
 * @param[in] coef0 parameter used in the polynomial kernel function
 * @param[in] B the matrix @p B
 * @param[in,out] C the matrix @p C
 * @param[in] num_classes the number of classes in the data set
 */
__global__ void device_kernel_assembly_polynomial_symm(const real_type alpha, const real_type *q, const real_type *data_d, const unsigned long long num_rows, const unsigned long long num_features, const real_type QA_cost, const real_type cost, const int degree, const real_type gamma, const real_type coef0, const real_type *B, real_type *C, const unsigned long long num_classes) {
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

                    // apply B and C
                    for (unsigned long long class_idx = 0; class_idx < num_classes; ++class_idx) {
                        atomicAdd(&C[global_i * (num_classes + PADDING_SIZE) + class_idx], alpha * temp_ij * B[global_j * (num_classes + PADDING_SIZE) + class_idx]);
                        if (global_i != global_j) {
                            atomicAdd(&C[global_j * (num_classes + PADDING_SIZE) + class_idx], alpha * temp_ij * B[global_i * (num_classes + PADDING_SIZE) + class_idx]);
                        }
                    }
                }
            }
        }
    }
}

/**
 * @brief Perform an implicit BLAS SYMM-like operation: `C = alpha * A * B + C` where `A` is the implicitly calculated kernel matrix using the rbf kernel function \f$e^{(-gamma \cdot |\vec{u} - \vec{v}|^2)}\f$ (never actually stored, reducing the amount of needed global memory), @p B and @p C are matrices, and @p alpha is a scalar.
 * @note The beta factor is already applied to C before this kernel starts!
 * @param[in] alpha the scalar alpha value
 * @param[in] q the vector used in the dimensional reduction
 * @param[in] data_d the data points to calculate the implicit kernel matrix from
 * @param[in] num_rows the number of data points
 * @param[in] num_features the number of features per data point
 * @param[in] QA_cost the scalar used in the dimensional reduction
 * @param[in] cost the cost factor the diagonal is scaled with
 * @param[in] gamma parameter used in the rbf kernel function
 * @param[in] B the matrix @p B
 * @param[in,out] C the matrix @p C
 * @param[in] num_classes the number of classes in the data set
 */
__global__ void device_kernel_assembly_rbf_symm(const real_type alpha, const real_type *q, const real_type *data_d, const unsigned long long num_rows, const unsigned long long num_features, const real_type QA_cost, const real_type cost, const real_type gamma, const real_type *B, real_type *C, const unsigned long long num_classes) {
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

                    // apply B and C
                    for (unsigned long long class_idx = 0; class_idx < num_classes; ++class_idx) {
                        atomicAdd(&C[global_i * (num_classes + PADDING_SIZE) + class_idx], alpha * temp_ij * B[global_j * (num_classes + PADDING_SIZE) + class_idx]);
                        if (global_i != global_j) {
                            atomicAdd(&C[global_j * (num_classes + PADDING_SIZE) + class_idx], alpha * temp_ij * B[global_i * (num_classes + PADDING_SIZE) + class_idx]);
                        }
                    }
                }
            }
        }
    }
}

}  // namespace plssvm::hip

#endif  // PLSSVM_BACKENDS_HIP_CG_IMPLICIT_KERNEL_MATRIX_ASSEMBLY_BLAS_HIP_HPP_
