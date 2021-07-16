#include "plssvm/backends/CUDA/cuda-kernel.cuh"

namespace plssvm {

template <typename real_type>
__global__ void kernel_q_linear(real_type *q, const real_type *data_d, const real_type *datlast, const int Nrows, const int start, const int end) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    real_type temp{ 0.0 };
    for (int i = start; i < end; ++i) {
        temp += data_d[i * Nrows + index] * datlast[i];
    }
    q[index] = temp;
}
template __global__ void kernel_q_linear(float *, const float *, const float *, const int, const int, const int);
template __global__ void kernel_q_linear(double *, const double *, const double *, const int, const int, const int);

template <typename real_type>
__global__ void kernel_q_poly(real_type *q, const real_type *data_d, const real_type *datlast, const int Nrows, const int Ncols, const real_type degree, const real_type gamma, const real_type coef0) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    real_type temp{ 0.0 };
    for (int i = 0; i < Ncols; ++i) {
        temp += data_d[i * Nrows + index] * datlast[i];
    }
    q[index] = pow(gamma * temp + coef0, degree);
}
template __global__ void kernel_q_poly(float *, const float *, const float *, const int, const int, const float, const float, const float);
template __global__ void kernel_q_poly(double *, const double *, const double *, const int, const int, const double, const double, const double);

template <typename real_type>
__global__ void kernel_q_radial(real_type *q, const real_type *data_d, const real_type *datlast, const int Nrows, const int Ncols, const real_type gamma) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    real_type temp{ 0.0 };
    for (int i = 0; i < Ncols; ++i) {
        temp += (data_d[i * Nrows + index] - datlast[i]) * (data_d[i * Nrows + index] - datlast[i]);
    }
    q[index] = exp(-gamma * temp);
}
template __global__ void kernel_q_radial(float *, const float *, const float *, const int, const int, const float);
template __global__ void kernel_q_radial(double *, const double *, const double *, const int, const int, const double);

}  // namespace plssvm