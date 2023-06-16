/**
* @author Alexander Van Craen
* @author Marcel Breyer
* @copyright 2018-today The PLSSVM project - All Rights Reserved
* @license This file is part of the PLSSVM project which is released under the MIT license.
*          See the LICENSE.md file in the project root for full license information.
*/

#include "plssvm/backends/CUDA/kernel_matrix_assembly.cuh"

#include "plssvm/backends/CUDA/detail/atomics.cuh"  // atomicAdd for double precision floating point numbers on older CUDA hardware
#include "plssvm/constants.hpp"                     // plssvm::kernel_index_type, plssvm::THREAD_BLOCK_SIZE, plssvm::INTERNAL_BLOCK_SIZE

namespace plssvm::cuda {

template <typename real_type>
__global__ void device_kernel_assembly_linear(const real_type *q, real_type *ret, const real_type *data_d, const real_type QA_cost, const real_type cost, const kernel_index_type num_rows, const kernel_index_type num_features) {
    const kernel_index_type i = blockIdx.x * blockDim.x + threadIdx.x;
    const kernel_index_type j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < num_rows && j < num_rows) {
        real_type temp{ 0.0 };
        for (kernel_index_type dim = 0; dim < num_features; ++dim) {
            temp += data_d[dim * num_rows + i] * data_d[dim * num_rows + j];
        }
        temp = temp + QA_cost - q[i] - q[j];
        if (i == j) {
            temp += cost;
        }

        ret[i * num_rows + j] = temp;
    }
}
template __global__ void device_kernel_assembly_linear(const float *, float *, const float *, const float, const float, const kernel_index_type, const kernel_index_type);
template __global__ void device_kernel_assembly_linear(const double *, double *, const double *, const double, const double, const kernel_index_type, const kernel_index_type);

template <typename real_type>
__global__ void device_kernel_assembly_polynomial(const real_type *q, real_type *ret, const real_type *data_d, const real_type QA_cost, const real_type cost, const kernel_index_type num_rows, const kernel_index_type num_features, const int degree, const real_type gamma, const real_type coef0) {
    const kernel_index_type i = blockIdx.x * blockDim.x + threadIdx.x;
    const kernel_index_type j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < num_rows && j < num_rows) {
        real_type temp{ 0.0 };
        for (kernel_index_type dim = 0; dim < num_features; ++dim) {
            temp += data_d[dim * num_rows + i] * data_d[dim * num_rows + j];
        }
        temp = pow(gamma * temp + coef0, degree) + QA_cost - q[i] - q[j];
        if (i == j) {
            temp += cost;
        }

        ret[i * num_rows + j] = temp;
    }
}
template __global__ void device_kernel_assembly_polynomial(const float *, float *, const float *, const float, const float, const kernel_index_type, const kernel_index_type, const int, const float, const float);
template __global__ void device_kernel_assembly_polynomial(const double *, double *, const double *, const double, const double, const kernel_index_type, const kernel_index_type, const int, const double, const double);


template <typename real_type>
__global__ void device_kernel_assembly_rbf(const real_type *q, real_type *ret, const real_type *data_d, const real_type QA_cost, const real_type cost, const kernel_index_type num_rows, const kernel_index_type num_features, const real_type gamma) {
    const kernel_index_type i = blockIdx.x * blockDim.x + threadIdx.x;
    const kernel_index_type j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < num_rows && j < num_rows) {
        real_type temp{ 0.0 };
        for (kernel_index_type dim = 0; dim < num_features; ++dim) {
            const real_type d = data_d[dim * num_rows + i] - data_d[dim * num_rows + j];
            temp += d * d;
        }
        temp = exp(-gamma * temp) + QA_cost - q[i] - q[j];
        if (i == j) {
            temp += cost;
        }

        ret[i * num_rows + j] = temp;
    }
}
template __global__ void device_kernel_assembly_rbf(const float *, float *, const float *, const float, const float, const kernel_index_type, const kernel_index_type, const float);
template __global__ void device_kernel_assembly_rbf(const double *, double *, const double *, const double, const double, const kernel_index_type, const kernel_index_type, const double);


}