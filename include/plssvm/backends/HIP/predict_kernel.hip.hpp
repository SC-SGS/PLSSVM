/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines the functions used for prediction for the C-SVM using the HIP backend.
 */

#ifndef PLSSVM_BACKENDS_HIP_PREDICT_KERNEL_HIP_HPP_
#define PLSSVM_BACKENDS_HIP_PREDICT_KERNEL_HIP_HPP_
#pragma once

#include "plssvm/constants.hpp"  // plssvm::real_type, plssvm::kernel_index_type

#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"


namespace plssvm::hip {

__global__ void device_kernel_w_linear(real_type *w_d, const real_type *alpha_d, const real_type *sv_d, const kernel_index_type num_classes, const kernel_index_type num_sv, const kernel_index_type num_features) {
    const kernel_index_type feature_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const kernel_index_type class_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (feature_idx < num_features && class_idx < num_classes) {
        real_type temp{ 0.0 };
        for (kernel_index_type sv = 0; sv < num_sv; ++sv) {
            temp += alpha_d[class_idx * num_sv + sv] * sv_d[sv * num_features + feature_idx];
        }
        w_d[class_idx * num_features + feature_idx] = temp;
    }
}

__global__ void device_kernel_predict_linear(real_type *out_d, const real_type *w_d, const real_type *rho_d, const real_type *predict_points_d, const kernel_index_type num_classes, const kernel_index_type num_predict_points, const kernel_index_type num_features) {
    const kernel_index_type predict_point_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const kernel_index_type class_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (predict_point_idx < num_predict_points && class_idx < num_classes) {
        real_type temp{ 0.0 };
        for (kernel_index_type dim = 0; dim < num_features; ++dim) {
            temp += w_d[class_idx * num_features + dim] * predict_points_d[predict_point_idx * num_features + dim];
        }
        out_d[predict_point_idx * num_classes + class_idx] = temp - rho_d[class_idx];
    }
}

__global__ void device_kernel_predict_polynomial(real_type *out_d, const real_type *alpha_d, const real_type *rho_d, const real_type *sv_d, const real_type *predict_points_d, const kernel_index_type num_classes, const kernel_index_type num_sv, const kernel_index_type num_predict_points, const kernel_index_type num_features, const int degree, const real_type gamma, const real_type coef0) {
    const kernel_index_type sv_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const kernel_index_type predict_points_idx = blockIdx.y * blockDim.y + threadIdx.y;
    const kernel_index_type class_idx = blockIdx.z * blockDim.z + threadIdx.z;

    if (sv_idx < num_sv && predict_points_idx < num_predict_points && class_idx < num_classes) {
        real_type temp{ 0.0 };
        // perform dot product
        for (kernel_index_type dim = 0; dim < num_features; ++dim) {
            temp += sv_d[sv_idx * num_features + dim] * predict_points_d[predict_points_idx * num_features + dim];
        }

        // apply degree, gamma, and coef0, alpha and rho
        temp = alpha_d[class_idx * num_sv + sv_idx] * pow(gamma * temp + coef0, degree);
        if (sv_idx == 0) {
            temp -= rho_d[class_idx];
        }

        atomicAdd(&out_d[predict_points_idx * num_classes + class_idx], temp);
    }
}

__global__ void device_kernel_predict_rbf(real_type *out_d, const real_type *alpha_d, const real_type *rho_d, const real_type *sv_d, const real_type *predict_points_d, const kernel_index_type num_classes, const kernel_index_type num_sv, const kernel_index_type num_predict_points, const kernel_index_type num_features, const real_type gamma) {
    const kernel_index_type sv_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const kernel_index_type predict_points_idx = blockIdx.y * blockDim.y + threadIdx.y;
    const kernel_index_type class_idx = blockIdx.z * blockDim.z + threadIdx.z;

    if (sv_idx < num_sv && predict_points_idx < num_predict_points && class_idx < num_classes) {
        real_type temp{ 0.0 };
        // perform dot product
        for (kernel_index_type dim = 0; dim < num_features; ++dim) {
            const real_type diff = sv_d[sv_idx * num_features + dim] - predict_points_d[predict_points_idx * num_features + dim];
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

}  // namespace plssvm::hip

#endif  // PLSSVM_BACKENDS_HIP_PREDICT_KERNEL_HIP_HPP_