/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines HIP functions for generating the `q` vector.
 */

#pragma once

#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"

#include "plssvm/constants.hpp"  // plssvm::THREAD_BLOCK_SIZE, plssvm::INTERNAL_BLOCK_SIZE

namespace plssvm::hip {

template <typename real_type>
__global__ void device_kernel_q_linear(real_type *q, const real_type *data_d, const real_type *data_last, const kernel_index_type num_rows, const kernel_index_type feature_range) {
    const kernel_index_type index = blockIdx.x * blockDim.x + threadIdx.x;
    real_type temp{ 0.0 };
    for (kernel_index_type i = 0; i < feature_range; ++i) {
        temp += data_d[i * num_rows + index] * data_last[i];
    }
    q[index] = temp;
}

template <typename real_type>
__global__ void device_kernel_q_poly(real_type *q, const real_type *data_d, const real_type *data_last, const kernel_index_type num_rows, const kernel_index_type num_cols, const int degree, const real_type gamma, const real_type coef0) {
    const kernel_index_type index = blockIdx.x * blockDim.x + threadIdx.x;
    real_type temp{ 0.0 };
    for (kernel_index_type i = 0; i < num_cols; ++i) {
        temp += data_d[i * num_rows + index] * data_last[i];
    }
    q[index] = pow(gamma * temp + coef0, degree);
}

template <typename real_type>
__global__ void device_kernel_q_radial(real_type *q, const real_type *data_d, const real_type *data_last, const kernel_index_type num_rows, const kernel_index_type num_cols, const real_type gamma) {
    const kernel_index_type index = blockIdx.x * blockDim.x + threadIdx.x;
    real_type temp{ 0.0 };
    for (kernel_index_type i = 0; i < num_cols; ++i) {
        temp += (data_d[i * num_rows + index] - data_last[i]) * (data_d[i * num_rows + index] - data_last[i]);
    }
    q[index] = exp(-gamma * temp);
}

}  // namespace plssvm::hip
