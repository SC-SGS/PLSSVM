/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/CUDA/cg_explicit/kernel_matrix_assembly.cuh"

#include "plssvm/constants.hpp"  // plssvm::real_type, plssvm::kernel_index_type

namespace plssvm::cuda {

__global__ void device_kernel_assembly_linear(const real_type *q, real_type *ret, const real_type *data_d, const real_type QA_cost, const real_type cost, const kernel_index_type dept, const kernel_index_type num_features) {
    const unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < dept && j < dept) {
        real_type temp{ 0.0 };
        for (unsigned long long dim = 0; dim < num_features; ++dim) {
            temp += data_d[i * num_features + dim] * data_d[j * num_features + dim];
        }
        temp = temp + QA_cost - q[i] - q[j];
        if (i == j) {
            temp += cost;
        }

        ret[i * dept + j] = temp;
    }
}

__global__ void device_kernel_assembly_polynomial(const real_type *q, real_type *ret, const real_type *data_d, const real_type QA_cost, const real_type cost, const kernel_index_type num_rows, const kernel_index_type num_features, const int degree, const real_type gamma, const real_type coef0) {
    const unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < num_rows && j < num_rows) {
        real_type temp{ 0.0 };
        for (unsigned long long dim = 0; dim < num_features; ++dim) {
            temp += data_d[i * num_features + dim] * data_d[j * num_features + dim];
        }
        temp = pow(gamma * temp + coef0, degree) + QA_cost - q[i] - q[j];
        if (i == j) {
            temp += cost;
        }

        ret[i * num_rows + j] = temp;
    }
}

__global__ void device_kernel_assembly_rbf(const real_type *q, real_type *ret, const real_type *data_d, const real_type QA_cost, const real_type cost, const kernel_index_type num_rows, const kernel_index_type num_features, const real_type gamma) {
    const unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < num_rows && j < num_rows) {
        real_type temp{ 0.0 };
        for (unsigned long long dim = 0; dim < num_features; ++dim) {
            const real_type d = data_d[i * num_features + dim] - data_d[j * num_features + dim];
            temp += d * d;
        }
        temp = exp(-gamma * temp) + QA_cost - q[i] - q[j];
        if (i == j) {
            temp += cost;
        }

        ret[i * num_rows + j] = temp;
    }
}

}  // namespace plssvm::cuda