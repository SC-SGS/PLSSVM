#include "plssvm/backends/CUDA/cuda-kernel.cuh"
#include "plssvm/typedef.hpp"

namespace plssvm {

template <typename real_type>
__global__ void init(real_type *vec, real_type value, int size) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) {
        vec[id] = value;
    }
}
template __global__ void init(float *, float, int);
template __global__ void init(double *, double, int);

template <typename real_type>
__global__ void add_mult(real_type *vec1, const real_type *vec2, const real_type value, const int dim) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < dim) {
        vec1[id] += vec2[id] * value;
    }
}
template __global__ void add_mult(float *, const float *, const float, const int);
template __global__ void add_mult(double *, const double *, const double, const int);

template <typename real_type>
__global__ void add_inplace(real_type *vec1, const real_type *vec2, const int dim) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < dim) {
        vec1[id] += vec2[id];
    }
}
template __global__ void add_inplace(float *, const float *, const int);
template __global__ void add_inplace(double *, const double *, const int);

template <typename real_type>
__global__ void dot(real_type *a, real_type *b, real_type *c, const int dim) {
    __shared__ real_type temp[THREADS_PER_BLOCK];
    const int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < dim) {
        temp[threadIdx.x] = a[index] * b[index];
    } else {
        temp[threadIdx.x] = 0.0;
    }
    __syncthreads();
    if (0 == threadIdx.x) {
        real_type sum = 0;
        for (int i = 0; i < THREADS_PER_BLOCK; i++) {
            sum += temp[i];
        }
        atomicAdd(c, sum);
    }
}
template __global__ void dot(float *, float *, float *, const int);
template __global__ void dot(double *, double *, double *, const int);

template <typename real_type>
__global__ void kernel_q_old(real_type *q, real_type *data_d, real_type *datlast, const int Ncols, const int Nrows) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    real_type temp = 0;
    for (int i = 0; i < Ncols; ++i) {
        temp += data_d[i * Nrows + index] * datlast[i];
    }
    q[index] = temp;
}
template __global__ void kernel_q_old(float *, float *, float *, const int, const int);
template __global__ void kernel_q_old(double *, double *, double *, const int, const int);

template <typename real_type>
__global__ void kernel_q_linear(real_type *q, const real_type *data_d, const real_type *datlast, const int Nrows, const int start, const int end) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    real_type temp{0.0};
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
    real_type temp{0.0};
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
    real_type temp{0.0};
    for (int i = 0; i < Ncols; ++i) {
        temp += (data_d[i * Nrows + index] - datlast[i]) * (data_d[i * Nrows + index] - datlast[i]);
    }
    q[index] = exp(-gamma * temp);
}
template __global__ void kernel_q_radial(float *, const float *, const float *, const int, const int, const float);
template __global__ void kernel_q_radial(double *, const double *, const double *, const int, const int, const double);

}  // namespace plssvm