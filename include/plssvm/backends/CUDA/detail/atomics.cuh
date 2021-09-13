/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright
 *
 * @brief Defines an atomic add function for double precision floating point types for older CUDA architectures.
 */

#pragma once

namespace plssvm::cuda::detail {

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
/**
 * @brief Atomically add the double precision value @p val to the value denoted by @p address.
 * @param[in,out] address the value to increment
 * @param[in] val the value to add
 */
__device__ __forceinline__ double atomicAdd(double *address, const double val) {
    unsigned long long int *address_as_ull = (unsigned long long int *) address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

}  // namespace plssvm::cuda::detail