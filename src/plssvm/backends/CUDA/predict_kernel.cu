/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/CUDA/predict_kernel.cuh"

#include "plssvm/backends/CUDA/detail/atomics.cuh"  // atomicAdd for double precision floating point numbers on older CUDA hardware
#include "plssvm/constants.hpp"                     // plssvm::real_type

namespace plssvm::cuda {

__global__ void device_kernel_w_linear(real_type *w_d, const real_type *alpha_d, const real_type *sv_d, const unsigned long long num_classes, const unsigned long long num_sv, const unsigned long long num_features) {
    const unsigned long long feature_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long class_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (feature_idx < num_features && class_idx < num_classes) {
        real_type temp{ 0.0 };
        for (unsigned long long sv = 0; sv < num_sv; ++sv) {
            temp += alpha_d[class_idx * num_sv + sv] * sv_d[feature_idx * (num_sv + THREAD_BLOCK_PADDING) + sv];
        }
        w_d[class_idx * num_features + feature_idx] = temp;
    }
}

__global__ void device_kernel_predict_linear(real_type *out_d, const real_type *w_d, const real_type *rho_d, const real_type *predict_points_d, const unsigned long long num_classes, const unsigned long long num_predict_points, const unsigned long long num_features) {
    const unsigned long long predict_points_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long class_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (predict_points_idx < num_predict_points && class_idx < num_classes) {
        real_type temp{ 0.0 };
        for (unsigned long long dim = 0; dim < num_features; ++dim) {
            temp += w_d[class_idx * num_features + dim] * predict_points_d[dim * (num_predict_points + THREAD_BLOCK_PADDING) + predict_points_idx];
        }
        out_d[predict_points_idx * num_classes + class_idx] = temp - rho_d[class_idx];
    }
}

__global__ void device_kernel_predict_polynomial(real_type *out_d, const real_type *alpha_d, const real_type *rho_d, const real_type *sv_d, const real_type *predict_points_d, const unsigned long long num_classes, const unsigned long long num_sv, const unsigned long long num_predict_points, const unsigned long long num_features, const int degree, const real_type gamma, const real_type coef0) {
    const unsigned long long sv_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long predict_points_idx = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned long long class_idx = blockIdx.z * blockDim.z + threadIdx.z;

    if (sv_idx < num_sv && predict_points_idx < num_predict_points && class_idx < num_classes) {
        real_type temp{ 0.0 };
        // perform dot product
        for (unsigned long long dim = 0; dim < num_features; ++dim) {
            temp += sv_d[dim * (num_sv + THREAD_BLOCK_PADDING) + sv_idx] * predict_points_d[dim * (num_predict_points + THREAD_BLOCK_PADDING) + predict_points_idx];
        }

        // apply degree, gamma, and coef0, alpha and rho
        temp = alpha_d[class_idx * num_sv + sv_idx] * pow(gamma * temp + coef0, degree);
        if (sv_idx == 0) {
            temp -= rho_d[class_idx];
        }

        atomicAdd(&out_d[predict_points_idx * num_classes + class_idx], temp);
    }
}

__global__ void device_kernel_predict_rbf(real_type *out_d, const real_type *alpha_d, const real_type *rho_d, const real_type *sv_d, const real_type *predict_points_d, const unsigned long long num_classes, const unsigned long long num_sv, const unsigned long long num_predict_points, const unsigned long long num_features, const real_type gamma) {
    const unsigned long long sv_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long predict_points_idx = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned long long class_idx = blockIdx.z * blockDim.z + threadIdx.z;

    if (sv_idx < num_sv && predict_points_idx < num_predict_points && class_idx < num_classes) {
        real_type temp{ 0.0 };
        // perform dist calculation
        for (unsigned long long dim = 0; dim < num_features; ++dim) {
            const real_type diff = sv_d[dim * (num_sv + THREAD_BLOCK_PADDING) + sv_idx] - predict_points_d[dim * (num_predict_points + THREAD_BLOCK_PADDING) + predict_points_idx];
            temp += diff * diff;
        }

        // apply degree, gamma, and coef0, alpha and rho
        temp = alpha_d[class_idx * num_sv + sv_idx] * exp(-gamma * temp);
        if (sv_idx == 0) {
            temp -= rho_d[class_idx];
        }

        atomicAdd(&out_d[predict_points_idx * num_classes + class_idx], temp);
    }
}

}  // namespace plssvm::cuda
