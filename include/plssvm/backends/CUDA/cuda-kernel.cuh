#pragma once

namespace plssvm {

template <typename real_type>
__global__ void kernel_q_linear(real_type *q, const real_type *data_d, const real_type *datlast, const int Nrows, const int start, const int end);
template <typename real_type>
__global__ void kernel_q_poly(real_type *q, const real_type *data_d, const real_type *datlast, const int Nrows, const int Ncols, const real_type degree, const real_type gamma, const real_type coef0);
template <typename real_type>
__global__ void kernel_q_radial(real_type *q, const real_type *data_d, const real_type *datlast, const int Nrows, const int Ncols, const real_type gamma);

// TODO: move separate header
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