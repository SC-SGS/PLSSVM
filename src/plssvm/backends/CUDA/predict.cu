/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright
 */

#include "plssvm/backends/CUDA/csvm.hpp"
#include "plssvm/backends/CUDA/detail/device_ptr.cuh"  // plssvm::detail::cuda::device_ptr
#include "plssvm/constants.hpp"

namespace plssvm::cuda {

template <typename real_type>
__global__ void kernel_w(real_type *w_d, const real_type *data_d, const real_type *data_last_d, const real_type *alpha_d, const int num_data_points, const int num_features) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    real_type temp = 0;
    if (index < num_features) {
        for (int dat = 0; dat < num_data_points - 1; ++dat) {
            temp += alpha_d[dat] * data_d[dat + (num_data_points - 1 + THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE) * index];
        }
        temp += alpha_d[num_data_points - 1] * data_last_d[index];
        w_d[index] = temp;
    }
}
template __global__ void kernel_w(float *, const float *, const float *, const float *, const int, const int);
template __global__ void kernel_w(double *, const double *, const double *, const double *, const int, const int);

template <typename real_type>
__global__ void predict_points_poly(real_type *out_d, const real_type *data_d, const real_type *data_last_d, const real_type *alpha_d, const int num_data_points, const real_type *points, const int num_predict_points, const int num_features, const int degree, const real_type gamma, const real_type coef0) {
    int data_point_index = blockIdx.x * blockDim.x + threadIdx.x;
    int predict_point_index = blockIdx.y * blockDim.y + threadIdx.y;

    real_type temp = 0;
    for (int feature_index = 0; feature_index < num_features; ++feature_index) {
        if (data_point_index == num_data_points) {
            temp += data_last_d[feature_index] * points[predict_point_index + (num_predict_points) *feature_index];
        } else {
            temp += data_d[data_point_index + (num_data_points - 1 + THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE) * feature_index] * points[predict_point_index + (num_predict_points) *feature_index];
        }
    }

    temp = alpha_d[data_point_index] * pow(gamma * temp + coef0, static_cast<real_type>(degree));

    atomicAdd(&out_d[predict_point_index], temp);
}

template __global__ void predict_points_poly(float *, const float *, const float *, const float *, const int, const float *, const int, const int, const int, const float, const float);
template __global__ void predict_points_poly(double *, const double *, const double *, const double *, const int, const double *, const int, const int, const int, const double, const double);

template <typename real_type>
__global__ void predict_points_rbf(real_type *out_d, const real_type *data_d, const real_type *data_last_d, const real_type *alpha_d, const int num_data_points, const real_type *points, const int num_predict_points, const int num_features, const real_type gamma) {
    int data_point_index = blockIdx.x * blockDim.x + threadIdx.x;
    int predict_point_index = blockIdx.y * blockDim.y + threadIdx.y;

    real_type temp = 0;
    for (int feature_index = 0; feature_index < num_features; ++feature_index) {
        if (data_point_index == num_data_points) {
            temp += (data_last_d[feature_index] - points[predict_point_index + (num_predict_points) *feature_index]) * (data_last_d[feature_index] - points[predict_point_index + (num_predict_points) *feature_index]);
        } else {
            temp += (data_d[data_point_index + (num_data_points - 1 + THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE) * feature_index] - points[predict_point_index + (num_predict_points) *feature_index]) * (data_d[data_point_index + (num_data_points - 1 + THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE) * feature_index] - points[predict_point_index + (num_predict_points) *feature_index]);
        }
    }

    temp = alpha_d[data_point_index] * exp(-gamma * temp);

    atomicAdd(&out_d[predict_point_index], temp);
}

template __global__ void predict_points_rbf(float *, const float *, const float *, const float *, const int, const float *, const int, const int, const float);
template __global__ void predict_points_rbf(double *, const double *, const double *, const double *, const int, const double *, const int, const int, const double);

}  // namespace plssvm::cuda
