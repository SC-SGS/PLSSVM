//
// Created by breyerml on 19.06.23.
//

#ifndef PLSSVM_BACKENDS_CUDA_CG_BLAS_CUH_
#define PLSSVM_BACKENDS_CUDA_CG_BLAS_CUH_

#include "plssvm/constants.hpp"  // plssvm::kernel_index_type

namespace plssvm::cuda {

template <typename real_type>
__global__ void device_kernel_matmul1(const kernel_index_type m, const kernel_index_type n, const kernel_index_type k, const real_type alpha, const real_type *A, const real_type *B, real_type *RET);

template <typename real_type>
__global__ void device_kernel_matmul2(const kernel_index_type m, const kernel_index_type n, const kernel_index_type k, const real_type alpha, const real_type *A, const real_type *B, const real_type beta, const real_type *C, real_type *RET);

}  // namespace plssvm::cuda

#endif  // PLSSVM_BACKENDS_CUDA_CG_BLAS_CUH_
