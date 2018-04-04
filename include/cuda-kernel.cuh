#include "CSVM.hpp"
__global__ void init(double* vec, double value, int size);

__global__ void add_mult(double* vec1, double* vec2, double value, int dim);

__global__ void kernel_q(double *q, double *data_d, double *datlast,const int Ncols, const int Nrows);


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
