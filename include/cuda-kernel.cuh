#pragma once

#include "CSVM.hpp"


__global__ void init(real_t* vec, real_t value, int size);

__global__ void add_mult(real_t* vec1, real_t* vec2, real_t value, int dim);

__global__ void kernel_q(real_t *q, real_t *data_d, real_t *datlast,const int Ncols, const int Nrows);


#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ __forceinline__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                          (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
            old = atomicCAS(address_as_ull, assumed,__double_as_longlong(val +
                               __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif
