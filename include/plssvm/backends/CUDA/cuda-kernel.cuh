#pragma once

namespace plssvm {

template <typename real_type>
__global__ void init(real_type *vec, real_type value, int size);

template <typename real_type>
__global__ void add_mult(real_type *vec1, const real_type *vec2, const real_type value, const int dim);

template <typename real_type>
__global__ void add_inplace(real_type *vec1, const real_type *vec2, const int dim);

template <typename real_type>
__global__ void dot(real_type *a, real_type *b, real_type *c, const int dim);

template <typename real_type>
__global__ void kernel_q(real_type *q, const real_type *data_d, const real_type *datlast, const int Nrows, const int start, const int end);

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ __forceinline__ double atomicAdd(double *address, double val) {
    unsigned long long int *address_as_ull = (unsigned long long int *) address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

}  // namespace plssvm