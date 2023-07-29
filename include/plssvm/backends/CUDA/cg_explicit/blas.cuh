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

#include "plssvm/constants.hpp"  // plssvm::real_type

namespace plssvm::cuda {

/**
 * @brief Perform an explicit BLAS GEMM operation: `C = alpha * A * B + beta * C` where @p A is a `m x k` matrix, @p B is a `k x n` matrix, @p C is a `m x n` matrix, and @p alpha and @p beta are scalars.
 * @param[in] m the number of rows in @p A and @p C
 * @param[in] n the number of columns in @p B and @p C
 * @param[in] k the number of rows in @p A and number of columns in @p B
 * @param[in] alpha the scalar alpha value
 * @param[in] A the matrix @p A
 * @param[in] B the matrix @p B
 * @param[in] beta the scalar beta value
 * @param[in,out] C the matrix @p C, also used as result matrix
 */
__global__ void device_kernel_gemm(const unsigned long long m, const unsigned long long n, const unsigned long long k, const real_type alpha, const real_type *A, const real_type *B, const real_type beta, real_type *C);

/**
 * @brief Perform an explicit BLAS SYMM operation: `C = alpha * A * B + beta * C` where @p A is a `m x k` symmetric matrix (memory optimized), @p B is a `k x n` matrix, @p C is a `m x n` matrix, and @p alpha and @p beta are scalars.
 * @param[in] m the number of rows in @p A and @p C
 * @param[in] n the number of columns in @p B and @p C
 * @param[in] k the number of rows in @p A and number of columns in @p B
 * @param[in] alpha the scalar alpha value
 * @param[in] A the matrix @p A
 * @param[in] B the matrix @p B
 * @param[in] beta the scalar beta value
 * @param[in,out] C the matrix @p C, also used as result matrix
 */
__global__ void device_kernel_symm(const unsigned long long m, const unsigned long long n, const unsigned long long k, const real_type alpha, const real_type *A, const real_type *B, const real_type beta, real_type *C);

}  // namespace plssvm::cuda

#endif  // PLSSVM_BACKENDS_CUDA_CG_EXPLICIT_BLAS_CUH_
