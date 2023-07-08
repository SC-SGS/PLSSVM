/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/CUDA/cg_explicit/blas.cuh"

#include "plssvm/constants.hpp"  // plssvm::real_type

namespace plssvm::cuda {

__global__ void device_kernel_gemm(const unsigned long long m, const unsigned long long n, const unsigned long long k, const real_type alpha, const real_type *A, const real_type *B, const real_type beta, real_type *C) {
    // compute: C = alpha * A * B + beta * C with A in m x k, B in n x k, and C in n x m, alpha, beta as scalar
    const unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < m && j < n) {
        real_type temp{ 0.0 };
        for (unsigned long long dim = 0; dim < k; ++dim) {
            temp += A[i * k + dim] * B[j * k + dim];
        }
        C[j * m + i] = alpha * temp + beta * C[j * m + i];
    }
}

}  // namespace plssvm::cuda