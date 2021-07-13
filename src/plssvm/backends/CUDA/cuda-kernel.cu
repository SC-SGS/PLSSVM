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
__global__ void kernel_q(real_type *q, const real_type *data_d, const real_type *datlast, const int Nrows, const int start, const int end) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    real_type temp = 0;
    for (int i = start; i < end; ++i) {
        temp += data_d[i * Nrows + index] * datlast[i];
    }
    q[index] = temp;  // TODO: nachschauen += ?
}
template __global__ void kernel_q(float *, const float *, const float *, const int, const int, const int);
template __global__ void kernel_q(double *, const double *, const double *, const int, const int, const int);

}  // namespace plssvm