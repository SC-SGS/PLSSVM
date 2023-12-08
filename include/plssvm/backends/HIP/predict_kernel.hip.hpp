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

#include "plssvm/constants.hpp"  // plssvm::{real_type, THREAD_BLOCK_SIZE, INTERNAL_BLOCK_SIZE, FEATURE_BLOCK_SIZE, PADDING_SIZE}

#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"

namespace plssvm::hip {

/**
 * @brief Calculate the `q` vector used to speedup the prediction using the linear kernel function.
 * @param[in,out] w_d the vector to speedup the linear prediction
 * @param[in] alpha_d the previously learned weights
 * @param[in] sv_d the support vectors
 * @param[in] num_classes the number of classes
 * @param[in] num_sv the number of support vectors
 */
__global__ void device_kernel_w_linear(real_type *w_d, const real_type *alpha_d, const real_type *sv_d, const unsigned long long num_classes, const unsigned long long num_sv) {
    const unsigned long long feature_idx = (blockIdx.x * blockDim.x + threadIdx.x) * INTERNAL_BLOCK_SIZE;
    const unsigned long long feature_idx_linear = blockIdx.x * blockDim.x * INTERNAL_BLOCK_SIZE + threadIdx.x;
    const unsigned long long class_idx = (blockIdx.y * blockDim.y + threadIdx.y) * INTERNAL_BLOCK_SIZE;
    const unsigned long long class_cached_idx_linear = blockIdx.y * blockDim.y * INTERNAL_BLOCK_SIZE + threadIdx.x;

    __shared__ real_type data_cache_feature[THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];
    __shared__ real_type data_cache_alpha[THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];

    real_type temp[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE] = { { 0.0 } };

    for (unsigned long long sv = 0; sv < num_sv; sv += THREAD_BLOCK_SIZE) {
        // load data into shared memory
        for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
            const unsigned long long global_feature_idx = feature_idx_linear + internal * THREAD_BLOCK_SIZE;
            const unsigned long long global_class_idx = class_cached_idx_linear + internal * THREAD_BLOCK_SIZE;

            data_cache_feature[threadIdx.y][internal * THREAD_BLOCK_SIZE + threadIdx.x] = sv_d[global_feature_idx * (num_sv + PADDING_SIZE) + sv + threadIdx.y];  // SoA
            data_cache_alpha[threadIdx.y][internal * THREAD_BLOCK_SIZE + threadIdx.x] = alpha_d[global_class_idx * (num_sv + PADDING_SIZE) + sv + threadIdx.y];   // AoS
        }
        __syncthreads();

        // calculation
        for (unsigned block_dim = 0; block_dim < THREAD_BLOCK_SIZE; ++block_dim) {
            for (unsigned internal_feature = 0; internal_feature < INTERNAL_BLOCK_SIZE; ++internal_feature) {
                for (unsigned internal_class = 0; internal_class < INTERNAL_BLOCK_SIZE; ++internal_class) {
                    temp[internal_feature][internal_class] += data_cache_alpha[block_dim][threadIdx.y * INTERNAL_BLOCK_SIZE + internal_class] * data_cache_feature[block_dim][threadIdx.x * INTERNAL_BLOCK_SIZE + internal_feature];
                }
            }
        }
        __syncthreads();
    }

    for (unsigned internal_feature = 0; internal_feature < INTERNAL_BLOCK_SIZE; ++internal_feature) {
        for (unsigned internal_class = 0; internal_class < INTERNAL_BLOCK_SIZE; ++internal_class) {
            const unsigned long long global_feature_idx = feature_idx + internal_feature;
            const unsigned long long global_class_idx = class_idx + internal_class;

            w_d[global_feature_idx * (num_classes + PADDING_SIZE) + global_class_idx] = temp[internal_feature][internal_class];
        }
    }
}

/**
 * @brief Predict the @p predict_points_d using the linear kernel speeding up the calculation using the @p w_d vector.
 * @param[out] out_d the predicted values
 * @param[in] w_d the vector to speedup the calculations
 * @param[in] rho_d the previously learned bias
 * @param[in] predict_points_d the data points to predict
 * @param[in] num_classes the number of classes
 * @param[in] num_predict_points the number of data points to predict
 * @param[in] num_features the number of features per data point
 */
__global__ void device_kernel_predict_linear(real_type *out_d, const real_type *w_d, const real_type *rho_d, const real_type *predict_points_d, const unsigned long long num_classes, const unsigned long long num_predict_points, const unsigned long long num_features) {
    const unsigned long long pp_idx = (blockIdx.x * blockDim.x + threadIdx.x) * INTERNAL_BLOCK_SIZE;
    const unsigned long long pp_idx_linear = blockIdx.x * blockDim.x * INTERNAL_BLOCK_SIZE + threadIdx.x;
    const unsigned long long class_idx = (blockIdx.y * blockDim.y + threadIdx.y) * INTERNAL_BLOCK_SIZE;
    const unsigned long long class_cached_idx_linear = blockIdx.y * blockDim.y * INTERNAL_BLOCK_SIZE + threadIdx.x;

    __shared__ real_type data_cache_pp[FEATURE_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];
    __shared__ real_type data_cache_w[FEATURE_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];

    real_type temp[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE] = { { 0.0 } };

    for (unsigned long long dim = 0; dim < num_features; dim += FEATURE_BLOCK_SIZE) {
        // load data into shared memory
        for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
            const unsigned long long global_pp_idx = pp_idx_linear + internal * THREAD_BLOCK_SIZE;
            const unsigned long long global_class_idx = class_cached_idx_linear + internal * THREAD_BLOCK_SIZE;

            data_cache_pp[threadIdx.y][internal * THREAD_BLOCK_SIZE + threadIdx.x] = predict_points_d[(dim + threadIdx.y) * (num_predict_points + PADDING_SIZE) + global_pp_idx];
            data_cache_pp[threadIdx.y + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + threadIdx.x] = predict_points_d[(dim + threadIdx.y + THREAD_BLOCK_SIZE) * (num_predict_points + PADDING_SIZE) + global_pp_idx];
            data_cache_w[threadIdx.y][internal * THREAD_BLOCK_SIZE + threadIdx.x] = w_d[(dim + threadIdx.y) * (num_classes + PADDING_SIZE) + global_class_idx];
            data_cache_w[threadIdx.y + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + threadIdx.x] = w_d[(dim + threadIdx.y + THREAD_BLOCK_SIZE) * (num_classes + PADDING_SIZE) + global_class_idx];
        }
        __syncthreads();

        // calculation
        for (unsigned block_dim = 0; block_dim < FEATURE_BLOCK_SIZE; ++block_dim) {
            for (unsigned internal_pd = 0; internal_pd < INTERNAL_BLOCK_SIZE; ++internal_pd) {
                for (unsigned internal_class = 0; internal_class < INTERNAL_BLOCK_SIZE; ++internal_class) {
                    temp[internal_pd][internal_class] += data_cache_w[block_dim][threadIdx.y * INTERNAL_BLOCK_SIZE + internal_class] * data_cache_pp[block_dim][threadIdx.x * INTERNAL_BLOCK_SIZE + internal_pd];
                }
            }
        }
        __syncthreads();
    }

    for (unsigned internal_pd = 0; internal_pd < INTERNAL_BLOCK_SIZE; ++internal_pd) {
        for (unsigned internal_class = 0; internal_class < INTERNAL_BLOCK_SIZE; ++internal_class) {
            const unsigned long long global_pp_idx = pp_idx + internal_pd;
            const unsigned long long global_class_idx = class_idx + internal_class;

            out_d[global_pp_idx * (num_classes + PADDING_SIZE) + global_class_idx] = temp[internal_pd][internal_class] - rho_d[global_class_idx];
        }
    }
}

/**
 * @brief Predict the @p predict_points_d using the polynomial.
 * @param[in] out_d the predicted values
 * @param[in] alpha_d the previously learned weights
 * @param[in] rho_d the previously learned biases
 * @param[in] sv_d the support vectors
 * @param[in] predict_points_d the data points to predict
 * @param[in] num_classes the number of classes
 * @param[in] num_sv the number of support vectors
 * @param[in] num_predict_points the number of data points to predict
 * @param[in] num_features the number of features per data point
 * @param[in] degree the parameter in the polynomial kernel function
 * @param[in] gamma the parameter in the polynomial kernel function
 * @param[in] coef0 the parameter in the polynomial kernel function
 */
__global__ void device_kernel_predict_polynomial(real_type *out_d, const real_type *alpha_d, const real_type *rho_d, const real_type *sv_d, const real_type *predict_points_d, const unsigned long long num_classes, const unsigned long long num_sv, const unsigned long long num_predict_points, const unsigned long long num_features, const int degree, const real_type gamma, const real_type coef0) {
    const unsigned long long pp_idx = (blockIdx.x * blockDim.x + threadIdx.x) * INTERNAL_BLOCK_SIZE;
    const unsigned long long pp_idx_linear = blockIdx.x * blockDim.x * INTERNAL_BLOCK_SIZE + threadIdx.x;
    const unsigned long long sv_idx = (blockIdx.y * blockDim.y + threadIdx.y) * INTERNAL_BLOCK_SIZE;
    const unsigned long long sv_cached_idx_linear = blockIdx.y * blockDim.y * INTERNAL_BLOCK_SIZE + threadIdx.x;

    __shared__ real_type data_cache_pp[FEATURE_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];
    __shared__ real_type data_cache_sv[FEATURE_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];

    real_type temp[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE] = { { 0.0 } };

    for (unsigned long long dim = 0; dim < num_features; dim += FEATURE_BLOCK_SIZE) {
        // load data into shared memory
        for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
            const unsigned long long global_pp_idx = pp_idx_linear + internal * THREAD_BLOCK_SIZE;
            const unsigned long long global_sv_idx = sv_cached_idx_linear + internal * THREAD_BLOCK_SIZE;

            data_cache_pp[threadIdx.y][internal * THREAD_BLOCK_SIZE + threadIdx.x] = predict_points_d[(dim + threadIdx.y) * (num_predict_points + PADDING_SIZE) + global_pp_idx];
            data_cache_pp[threadIdx.y + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + threadIdx.x] = predict_points_d[(dim + threadIdx.y + THREAD_BLOCK_SIZE) * (num_predict_points + PADDING_SIZE) + global_pp_idx];
            data_cache_sv[threadIdx.y][internal * THREAD_BLOCK_SIZE + threadIdx.x] = sv_d[(dim + threadIdx.y) * (num_sv + PADDING_SIZE) + global_sv_idx];
            data_cache_sv[threadIdx.y + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + threadIdx.x] = sv_d[(dim + threadIdx.y + THREAD_BLOCK_SIZE) * (num_sv + PADDING_SIZE) + global_sv_idx];
        }
        __syncthreads();

        // calculation
        for (unsigned block_dim = 0; block_dim < FEATURE_BLOCK_SIZE; ++block_dim) {
            for (unsigned internal_pd = 0; internal_pd < INTERNAL_BLOCK_SIZE; ++internal_pd) {
                for (unsigned internal_sv = 0; internal_sv < INTERNAL_BLOCK_SIZE; ++internal_sv) {
                    temp[internal_pd][internal_sv] += data_cache_sv[block_dim][threadIdx.y * INTERNAL_BLOCK_SIZE + internal_sv] * data_cache_pp[block_dim][threadIdx.x * INTERNAL_BLOCK_SIZE + internal_pd];
                }
            }
        }
        __syncthreads();
    }

    for (unsigned internal_pd = 0; internal_pd < INTERNAL_BLOCK_SIZE; ++internal_pd) {
        for (unsigned internal_sv = 0; internal_sv < INTERNAL_BLOCK_SIZE; ++internal_sv) {
            const unsigned long long global_pp_idx = pp_idx + internal_pd;
            const unsigned long long global_sv_idx = sv_idx + internal_sv;

            const real_type temp_pp_sv = temp[internal_pd][internal_sv];
            for (unsigned long long class_idx = 0; class_idx < num_classes; ++class_idx) {
                // apply degree, gamma, and coef0, alpha and rho
                real_type class_temp = alpha_d[class_idx * (num_sv + PADDING_SIZE) + global_sv_idx] * pow(gamma * temp_pp_sv + coef0, degree);
                if (global_sv_idx == 0) {
                    class_temp -= rho_d[class_idx];
                }

                atomicAdd(&out_d[global_pp_idx * (num_classes + PADDING_SIZE) + class_idx], class_temp);
            }
        }
    }
}

/**
 * @brief Predict the @p predict_points_d using the rbf.
 * @param[in] out_d the predicted values
 * @param[in] alpha_d the previously learned weights
 * @param[in] rho_d the previously learned biases
 * @param[in] sv_d the support vectors
 * @param[in] predict_points_d the data points to predict
 * @param[in] num_classes the number of classes
 * @param[in] num_sv the number of support vectors
 * @param[in] num_predict_points the number of data points to predict
 * @param[in] num_features the number of features per data point
 * @param[in] gamma the parameter in the rbf kernel function
 */
__global__ void device_kernel_predict_rbf(real_type *out_d, const real_type *alpha_d, const real_type *rho_d, const real_type *sv_d, const real_type *predict_points_d, const unsigned long long num_classes, const unsigned long long num_sv, const unsigned long long num_predict_points, const unsigned long long num_features, const real_type gamma) {
    const unsigned long long pp_idx = (blockIdx.x * blockDim.x + threadIdx.x) * INTERNAL_BLOCK_SIZE;
    const unsigned long long pp_idx_linear = blockIdx.x * blockDim.x * INTERNAL_BLOCK_SIZE + threadIdx.x;
    const unsigned long long sv_idx = (blockIdx.y * blockDim.y + threadIdx.y) * INTERNAL_BLOCK_SIZE;
    const unsigned long long sv_cached_idx_linear = blockIdx.y * blockDim.y * INTERNAL_BLOCK_SIZE + threadIdx.x;

    __shared__ real_type data_cache_pp[FEATURE_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];
    __shared__ real_type data_cache_sv[FEATURE_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];

    real_type temp[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE] = { { 0.0 } };

    for (unsigned long long dim = 0; dim < num_features; dim += FEATURE_BLOCK_SIZE) {
        // load data into shared memory
        for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
            const unsigned long long global_pp_idx = pp_idx_linear + internal * THREAD_BLOCK_SIZE;
            const unsigned long long global_sv_idx = sv_cached_idx_linear + internal * THREAD_BLOCK_SIZE;

            data_cache_pp[threadIdx.y][internal * THREAD_BLOCK_SIZE + threadIdx.x] = predict_points_d[(dim + threadIdx.y) * (num_predict_points + PADDING_SIZE) + global_pp_idx];
            data_cache_pp[threadIdx.y + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + threadIdx.x] = predict_points_d[(dim + threadIdx.y + THREAD_BLOCK_SIZE) * (num_predict_points + PADDING_SIZE) + global_pp_idx];
            data_cache_sv[threadIdx.y][internal * THREAD_BLOCK_SIZE + threadIdx.x] = sv_d[(dim + threadIdx.y) * (num_sv + PADDING_SIZE) + global_sv_idx];
            data_cache_sv[threadIdx.y + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + threadIdx.x] = sv_d[(dim + threadIdx.y + THREAD_BLOCK_SIZE) * (num_sv + PADDING_SIZE) + global_sv_idx];
        }
        __syncthreads();

        // calculation
        for (unsigned block_dim = 0; block_dim < FEATURE_BLOCK_SIZE; ++block_dim) {
            for (unsigned internal_pd = 0; internal_pd < INTERNAL_BLOCK_SIZE; ++internal_pd) {
                for (unsigned internal_sv = 0; internal_sv < INTERNAL_BLOCK_SIZE; ++internal_sv) {
                    const real_type d = data_cache_sv[block_dim][threadIdx.y * INTERNAL_BLOCK_SIZE + internal_sv] - data_cache_pp[block_dim][threadIdx.x * INTERNAL_BLOCK_SIZE + internal_pd];
                    temp[internal_pd][internal_sv] += d * d;
                }
            }
        }
        __syncthreads();
    }

    for (unsigned internal_pd = 0; internal_pd < INTERNAL_BLOCK_SIZE; ++internal_pd) {
        for (unsigned internal_sv = 0; internal_sv < INTERNAL_BLOCK_SIZE; ++internal_sv) {
            const unsigned long long global_pp_idx = pp_idx + internal_pd;
            const unsigned long long global_sv_idx = sv_idx + internal_sv;

            const real_type temp_pp_sv = temp[internal_pd][internal_sv];
            for (unsigned long long class_idx = 0; class_idx < num_classes; ++class_idx) {
                // apply gamma, alpha and rho
                real_type class_temp = alpha_d[class_idx * (num_sv + PADDING_SIZE) + global_sv_idx] * exp(-gamma * temp_pp_sv);
                if (global_sv_idx == 0) {
                    class_temp -= rho_d[class_idx];
                }

                atomicAdd(&out_d[global_pp_idx * (num_classes + PADDING_SIZE) + class_idx], class_temp);
            }
        }
    }
}

}  // namespace plssvm::hip

#endif  // PLSSVM_BACKENDS_HIP_PREDICT_KERNEL_HIP_HPP_