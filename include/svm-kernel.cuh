#ifndef SVM_KERNEL_H
#define SVM_KERNEL_H

#include "CSVM.hpp"
//#include "cuda-kernel.cuh"
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* address, double val)
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

__global__  void kernel_linear(double *q, double *ret, double *d, double *data_d,const double QA_cost, const double cost,const int Ncols,const int Nrows,const int add);

__global__ void kernel_poly(double *q, double *ret, double *d, double *data_d,const double QA_cost, const double cost,const int Ncols,const int Nrows,const int add, const double gamma, const double coef0 ,const double degree);

__global__  void kernel_radial(double *q, double *ret, double *d, double *data_d,const double QA_cost, const double cost,const int Ncols,const int Nrows,const int add, const double gamma);


#endif //SVM_KERNEL_H
