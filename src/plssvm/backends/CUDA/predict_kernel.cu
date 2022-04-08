/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/CUDA/predict_kernel.cuh"

#include "plssvm/backends/CUDA/detail/atomics.cuh"  // atomicAdd for double precision floating point numbers on older CUDA hardware
#include "plssvm/constants.hpp"                     // plssvm::kernel_index_type, plssvm::THREAD_BLOCK_SIZE, plssvm::INTERNAL_BLOCK_SIZE

namespace plssvm::cuda {

template <typename real_type>
__global__ void device_kernel_w_linear(real_type *w_d, const real_type *data_d, const real_type *data_last_d, const real_type *alpha_d, const kernel_index_type num_data_points, const kernel_index_type num_features) {
    const kernel_index_type index = blockIdx.x * blockDim.x + threadIdx.x;
    real_type temp{ 0.0 };
    if (index < num_features) {
        for (kernel_index_type dat = 0; dat < num_data_points - 1; ++dat) {
            temp += alpha_d[dat] * data_d[dat + (num_data_points - 1 + THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE) * index];
        }
        temp += alpha_d[num_data_points - 1] * data_last_d[index];
        w_d[index] = temp;
    }
}
template __global__ void device_kernel_w_linear(float *, const float *, const float *, const float *, const kernel_index_type, const kernel_index_type);
template __global__ void device_kernel_w_linear(double *, const double *, const double *, const double *, const kernel_index_type, const kernel_index_type);

template <typename real_type>
__global__ void device_kernel_predict_poly(real_type *out_d, const real_type *data_d, const real_type *data_last_d, const real_type *alpha_d, const kernel_index_type num_data_points, const real_type *points, const kernel_index_type num_predict_points, const kernel_index_type num_features, const int degree, const real_type gamma, const real_type coef0) {
    const kernel_index_type data_point_index = blockIdx.x * blockDim.x + threadIdx.x;
    const kernel_index_type predict_point_index = blockIdx.y * blockDim.y + threadIdx.y;

    real_type temp{ 0.0 };
    if (predict_point_index < num_predict_points) {
        for (kernel_index_type feature_index = 0; feature_index < num_features; ++feature_index) {
            if (data_point_index == num_data_points) {
                temp += data_last_d[feature_index] * points[predict_point_index + (num_predict_points + THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE) * feature_index];
            } else {
                temp += data_d[data_point_index + (num_data_points - 1 + THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE) * feature_index] * points[predict_point_index + (num_predict_points + THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE) * feature_index];
            }
        }

        temp = alpha_d[data_point_index] * pow(gamma * temp + coef0, degree);

        atomicAdd(&out_d[predict_point_index], temp);
    }
}

template __global__ void device_kernel_predict_poly(float *, const float *, const float *, const float *, const kernel_index_type, const float *, const kernel_index_type, const kernel_index_type, const int, const float, const float);
template __global__ void device_kernel_predict_poly(double *, const double *, const double *, const double *, const kernel_index_type, const double *, const kernel_index_type, const kernel_index_type, const int, const double, const double);

template <typename real_type>
__global__ void device_kernel_predict_radial(real_type *out_d, const real_type *data_d, const real_type *data_last_d, const real_type *alpha_d, const kernel_index_type num_data_points, const real_type *points, const kernel_index_type num_predict_points, const kernel_index_type num_features, const real_type gamma) {
    const kernel_index_type data_point_index = blockIdx.x * blockDim.x + threadIdx.x;
    const kernel_index_type predict_point_index = blockIdx.y * blockDim.y + threadIdx.y;

    real_type temp{ 0.0 };
    if (predict_point_index < num_predict_points) {
        for (kernel_index_type feature_index = 0; feature_index < num_features; ++feature_index) {
            if (data_point_index == num_data_points) {
                temp += (data_last_d[feature_index] - points[predict_point_index + (num_predict_points + THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE) * feature_index]) * (data_last_d[feature_index] - points[predict_point_index + (num_predict_points + THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE) * feature_index]);
            } else {
                temp += (data_d[data_point_index + (num_data_points - 1 + THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE) * feature_index] - points[predict_point_index + (num_predict_points + THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE) * feature_index]) * (data_d[data_point_index + (num_data_points - 1 + THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE) * feature_index] - points[predict_point_index + (num_predict_points + THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE) * feature_index]);
            }
        }

        temp = alpha_d[data_point_index] * exp(-gamma * temp);

        atomicAdd(&out_d[predict_point_index], temp);
    }
}

template __global__ void device_kernel_predict_radial(float *, const float *, const float *, const float *, const kernel_index_type, const float *, const kernel_index_type, const kernel_index_type, const float);
template __global__ void device_kernel_predict_radial(double *, const double *, const double *, const double *, const kernel_index_type, const double *, const kernel_index_type, const kernel_index_type, const double);

}  // namespace plssvm::cuda
