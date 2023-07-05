/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Functions for explicitly performing a BLAS GEMM like matrix-matrix multiplication using the CUDA backend.
 */

#ifndef PLSSVM_BACKENDS_CUDA_CG_EXPLICIT_BLAS_CUH_
#define PLSSVM_BACKENDS_CUDA_CG_EXPLICIT_BLAS_CUH_
#pragma once

#include "plssvm/constants.hpp"  // plssvm::kernel_index_type

namespace plssvm::cuda {

template <typename real_type>
__global__ void device_kernel_gemm(const kernel_index_type m, const kernel_index_type n, const kernel_index_type k, const real_type alpha, const real_type *A, const real_type *B, const real_type beta, real_type *C);

}  // namespace plssvm::cuda

#endif  // PLSSVM_BACKENDS_CUDA_CG_EXPLICIT_BLAS_CUH_
