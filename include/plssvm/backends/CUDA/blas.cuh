//
// Created by breyerml on 19.06.23.
//

#ifndef PLSSVM_BACKENDS_CUDA_BLAS_CUH_
#define PLSSVM_BACKENDS_CUDA_BLAS_CUH_

#include "plssvm/constants.hpp"  // plssvm::kernel_index_type

namespace plssvm::cuda {

template <typename real_type>
__global__ void device_kernel_gemm(const kernel_index_type m, const kernel_index_type n, const kernel_index_type k, const real_type alpha, const real_type *A, const real_type *B, const real_type beta, real_type *C);

}  // namespace plssvm::cuda

#endif  // PLSSVM_BACKENDS_CUDA_CG_BLAS_CUH_
