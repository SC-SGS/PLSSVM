/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines the kernel functions for the C-SVM using the HIP backend.
 */

#ifndef PLSSVM_BACKENDS_HIP_CG_IMPLICIT_SVM_KERNEL_HPP_
#define PLSSVM_BACKENDS_HIP_CG_IMPLICIT_SVM_KERNEL_HPP_
#pragma once

#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"

#include "plssvm/constants.hpp"  // plssvm::THREAD_BLOCK_SIZE, plssvm::INTERNAL_BLOCK_SIZE

namespace plssvm::hip {

/**
 * @brief Calculates the C-SVM kernel using the linear kernel function.
 * @details Supports multi-GPU execution.
 * @tparam real_type the type of the data
 * @param[in] q the `q` vector
 * @param[out] ret the result vector
 * @param[in] d the right-hand side of the equation
 * @param[in] data_d the one-dimension data matrix
 * @param[in] QA_cost the bottom right matrix entry multiplied by cost
 * @param[in] cost 1 / the cost parameter in the C-SVM
 * @param[in] num_rows the number of columns in the data matrix
 * @param[in] feature_range  number of features used for the calculation on the device @p id
 * @param[in] add denotes whether the values are added or subtracted from the result vector
 * @param[in] id the id of the current device
 */
template <typename real_type>
__global__ void device_kernel_linear(const real_type *q, real_type *ret, const real_type *d, const real_type *data_d, const real_type QA_cost, const real_type cost, const kernel_index_type num_rows, const kernel_index_type feature_range, const real_type add, const kernel_index_type id) {
    kernel_index_type i = blockIdx.x * blockDim.x * INTERNAL_BLOCK_SIZE;
    kernel_index_type j = blockIdx.y * blockDim.y * INTERNAL_BLOCK_SIZE;

    __shared__ real_type data_intern_i[THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE];
    __shared__ real_type data_intern_j[THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE];
    real_type matr[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE] = { { 0.0 } };
    real_type data_j[INTERNAL_BLOCK_SIZE];

    if (i >= j) {
        i += threadIdx.x * INTERNAL_BLOCK_SIZE;
        const kernel_index_type ji = j + threadIdx.x * INTERNAL_BLOCK_SIZE;
        j += threadIdx.y * INTERNAL_BLOCK_SIZE;
        // cache data
        for (kernel_index_type vec_index = 0; vec_index < feature_range * num_rows; vec_index += num_rows) {
            __syncthreads();
            #pragma unroll INTERNAL_BLOCK_SIZE
            for (kernel_index_type block_id = 0; block_id < INTERNAL_BLOCK_SIZE; ++block_id) {
                const kernel_index_type idx = block_id % THREAD_BLOCK_SIZE;
                if (threadIdx.y == idx) {
                    data_intern_i[threadIdx.x][block_id] = data_d[block_id + vec_index + i];
                }
                const kernel_index_type idx_2 = block_id + INTERNAL_BLOCK_SIZE % THREAD_BLOCK_SIZE;
                if (threadIdx.y == idx_2) {
                    data_intern_j[threadIdx.x][block_id] = data_d[block_id + vec_index + ji];
                }
            }
            __syncthreads();

            #pragma unroll INTERNAL_BLOCK_SIZE
            for (kernel_index_type data_index = 0; data_index < INTERNAL_BLOCK_SIZE; ++data_index) {
                data_j[data_index] = data_intern_j[threadIdx.y][data_index];
            }

            #pragma unroll INTERNAL_BLOCK_SIZE
            for (kernel_index_type l = 0; l < INTERNAL_BLOCK_SIZE; ++l) {
                const real_type data_i = data_intern_i[threadIdx.x][l];
                #pragma unroll INTERNAL_BLOCK_SIZE
                for (kernel_index_type k = 0; k < INTERNAL_BLOCK_SIZE; ++k) {
                    matr[k][l] += data_i * data_j[k];
                }
            }
        }

        #pragma unroll INTERNAL_BLOCK_SIZE
        for (kernel_index_type x = 0; x < INTERNAL_BLOCK_SIZE; ++x) {
            real_type ret_jx = 0.0;
            #pragma unroll INTERNAL_BLOCK_SIZE
            for (kernel_index_type y = 0; y < INTERNAL_BLOCK_SIZE; ++y) {
                real_type temp;
                if (id == 0) {
                    temp = (matr[x][y] + QA_cost - q[i + y] - q[j + x]) * add;
                } else {
                    temp = matr[x][y] * add;
                }
                if (i + x > j + y) {
                    // upper triangular matrix
                    atomicAdd(&ret[i + y], temp * d[j + x]);
                    ret_jx += temp * d[i + y];
                } else if (i + x == j + y) {
                    // diagonal
                    if (id == 0) {
                        ret_jx += (temp + cost * add) * d[i + y];
                    } else {
                        ret_jx += temp * d[i + y];
                    }
                }
            }
            atomicAdd(&ret[j + x], ret_jx);
        }
    }
}

/**
 * @brief Calculates the C-SVM kernel using the polynomial kernel function.
 * @details Currently only single GPU execution is supported.
 * @tparam real_type the type of the data
 * @param[in] q the `q` vector
 * @param[out] ret the result vector
 * @param[in] d the right-hand side of the equation
 * @param[in] data_d the one-dimension data matrix
 * @param[in] QA_cost he bottom right matrix entry multiplied by cost
 * @param[in] cost 1 / the cost parameter in the C-SVM
 * @param[in] num_rows the number of columns in the data matrix
 * @param[in] num_cols the number of rows in the data matrix
 * @param[in] add denotes whether the values are added or subtracted from the result vector
 * @param[in] degree the degree parameter used in the polynomial kernel function
 * @param[in] gamma the gamma parameter used in the polynomial kernel function
 * @param[in] coef0 the coef0 parameter used in the polynomial kernel function
 */
template <typename real_type>
__global__ void device_kernel_polynomial(const real_type *q, real_type *ret, const real_type *d, const real_type *data_d, const real_type QA_cost, const real_type cost, const kernel_index_type num_rows, const kernel_index_type num_cols, const real_type add, const int degree, const real_type gamma, const real_type coef0) {
    kernel_index_type i = blockIdx.x * blockDim.x * INTERNAL_BLOCK_SIZE;
    kernel_index_type j = blockIdx.y * blockDim.y * INTERNAL_BLOCK_SIZE;

    __shared__ real_type data_intern_i[THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE];
    __shared__ real_type data_intern_j[THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE];
    real_type matr[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE] = { { 0.0 } };
    real_type data_j[INTERNAL_BLOCK_SIZE];

    if (i >= j) {
        i += threadIdx.x * INTERNAL_BLOCK_SIZE;
        const kernel_index_type ji = j + threadIdx.x * INTERNAL_BLOCK_SIZE;
        j += threadIdx.y * INTERNAL_BLOCK_SIZE;
        for (kernel_index_type vec_index = 0; vec_index < num_cols * num_rows; vec_index += num_rows) {
            __syncthreads();
            #pragma unroll INTERNAL_BLOCK_SIZE
            for (kernel_index_type block_id = 0; block_id < INTERNAL_BLOCK_SIZE; ++block_id) {
                const kernel_index_type idx = block_id % THREAD_BLOCK_SIZE;
                if (threadIdx.y == idx) {
                    data_intern_i[threadIdx.x][block_id] = data_d[block_id + vec_index + i];
                }
                const kernel_index_type idx_2 = block_id + INTERNAL_BLOCK_SIZE % THREAD_BLOCK_SIZE;
                if (threadIdx.y == idx_2) {
                    data_intern_j[threadIdx.x][block_id] = data_d[block_id + vec_index + ji];
                }
            }
            __syncthreads();

            #pragma unroll INTERNAL_BLOCK_SIZE
            for (kernel_index_type data_index = 0; data_index < INTERNAL_BLOCK_SIZE; ++data_index) {
                data_j[data_index] = data_intern_j[threadIdx.y][data_index];
            }

            #pragma unroll INTERNAL_BLOCK_SIZE
            for (kernel_index_type l = 0; l < INTERNAL_BLOCK_SIZE; ++l) {
                const real_type data_i = data_intern_i[threadIdx.x][l];
                #pragma unroll INTERNAL_BLOCK_SIZE
                for (kernel_index_type k = 0; k < INTERNAL_BLOCK_SIZE; ++k) {
                    matr[k][l] += data_i * data_j[k];
                }
            }
        }

        #pragma unroll INTERNAL_BLOCK_SIZE
        for (kernel_index_type x = 0; x < INTERNAL_BLOCK_SIZE; ++x) {
            real_type ret_jx = 0.0;
            #pragma unroll INTERNAL_BLOCK_SIZE
            for (kernel_index_type y = 0; y < INTERNAL_BLOCK_SIZE; ++y) {
                const real_type temp = (pow(gamma * matr[x][y] + coef0, degree) + QA_cost - q[i + y] - q[j + x]) * add;
                if (i + x > j + y) {
                    // upper triangular matrix
                    atomicAdd(&ret[i + y], temp * d[j + x]);
                    ret_jx += temp * d[i + y];
                } else if (i + x == j + y) {
                    // diagonal
                    ret_jx += (temp + cost * add) * d[i + y];
                }
            }
            atomicAdd(&ret[j + x], ret_jx);
        }
    }
}

/**
 * @brief Calculates the C-SVM kernel using the radial basis function kernel function.
 * @details Currently only single GPU execution is supported.
 * @tparam real_type the type of the data
 * @param[in] q the `q` vector
 * @param[out] ret the result vector
 * @param[in] d the right-hand side of the equation
 * @param[in] data_d the one-dimension data matrix
 * @param[in] QA_cost he bottom right matrix entry multiplied by cost
 * @param[in] cost 1 / the cost parameter in the C-SVM
 * @param[in] num_rows the number of columns in the data matrix
 * @param[in] num_cols the number of rows in the data matrix
 * @param[in] add denotes whether the values are added or subtracted from the result vector
 * @param[in] gamma the gamma parameter used in the rbf kernel function
 */
template <typename real_type>
__global__ void device_kernel_rbf(const real_type *q, real_type *ret, const real_type *d, const real_type *data_d, const real_type QA_cost, const real_type cost, const kernel_index_type num_rows, const kernel_index_type num_cols, const real_type add, const real_type gamma) {
    kernel_index_type i = blockIdx.x * blockDim.x * INTERNAL_BLOCK_SIZE;
    kernel_index_type j = blockIdx.y * blockDim.y * INTERNAL_BLOCK_SIZE;

    __shared__ real_type data_intern_i[THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE];
    __shared__ real_type data_intern_j[THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE];
    real_type matr[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE] = { { 0.0 } };
    real_type data_j[INTERNAL_BLOCK_SIZE];

    if (i >= j) {
        i += threadIdx.x * INTERNAL_BLOCK_SIZE;
        const kernel_index_type ji = j + threadIdx.x * INTERNAL_BLOCK_SIZE;
        j += threadIdx.y * INTERNAL_BLOCK_SIZE;
        for (kernel_index_type vec_index = 0; vec_index < num_cols * num_rows; vec_index += num_rows) {
            __syncthreads();
            #pragma unroll INTERNAL_BLOCK_SIZE
            for (kernel_index_type block_id = 0; block_id < INTERNAL_BLOCK_SIZE; ++block_id) {
                const kernel_index_type idx = block_id % THREAD_BLOCK_SIZE;
                if (threadIdx.y == idx) {
                    data_intern_i[threadIdx.x][block_id] = data_d[block_id + vec_index + i];
                }
                const kernel_index_type idx2 = block_id + INTERNAL_BLOCK_SIZE % THREAD_BLOCK_SIZE;
                if (threadIdx.y == idx2) {
                    data_intern_j[threadIdx.x][block_id] = data_d[block_id + vec_index + ji];
                }
            }
            __syncthreads();

            #pragma unroll INTERNAL_BLOCK_SIZE
            for (kernel_index_type data_index = 0; data_index < INTERNAL_BLOCK_SIZE; ++data_index) {
                data_j[data_index] = data_intern_j[threadIdx.y][data_index];
            }

            #pragma unroll INTERNAL_BLOCK_SIZE
            for (kernel_index_type l = 0; l < INTERNAL_BLOCK_SIZE; ++l) {
                const real_type data_i = data_intern_i[threadIdx.x][l];
                #pragma unroll INTERNAL_BLOCK_SIZE
                for (kernel_index_type k = 0; k < INTERNAL_BLOCK_SIZE; ++k) {
                    matr[k][l] += (data_i - data_j[k]) * (data_i - data_j[k]);
                }
            }
        }

        #pragma unroll INTERNAL_BLOCK_SIZE
        for (kernel_index_type x = 0; x < INTERNAL_BLOCK_SIZE; ++x) {
            real_type ret_jx = 0.0;
            #pragma unroll INTERNAL_BLOCK_SIZE
            for (kernel_index_type y = 0; y < INTERNAL_BLOCK_SIZE; ++y) {
                const real_type temp = (exp(-gamma * matr[x][y]) + QA_cost - q[i + y] - q[j + x]) * add;
                if (i + x > j + y) {
                    // upper triangular matrix
                    atomicAdd(&ret[i + y], temp * d[j + x]);
                    ret_jx += temp * d[i + y];
                } else if (i + x == j + y) {
                    // diagonal
                    ret_jx += (temp + cost * add) * d[i + y];
                }
            }
            atomicAdd(&ret[j + x], ret_jx);
        }
    }
}

}  // namespace plssvm::hip

#endif  // PLSSVM_BACKENDS_HIP_CG_IMPLICIT_SVM_KERNEL_HPP_
