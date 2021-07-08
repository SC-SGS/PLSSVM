#pragma once

#include <plssvm/CSVM.hpp>
#include <plssvm/typedef.hpp>

namespace plssvm {

__global__ void init(real_t *vec, real_t value, int size);

__global__ void add_mult(real_t *vec1, const real_t *vec2, const real_t value, const int dim);

__global__ void add_inplace(real_t *vec1, const real_t *vec2, const int dim);

__global__ void dot(real_t *a, real_t *b, real_t *c, const int dim);

__global__ void kernel_q(real_t *q,
                         const real_t *data_d,
                         const real_t *datlast,
                         const int Nrows,
                         const int start,
                         const int end);

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ __forceinline__ double atomicAdd(double *address, double val) {
    unsigned long long int *address_as_ull =
        (unsigned long long int *) address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

}  // namespace plssvm