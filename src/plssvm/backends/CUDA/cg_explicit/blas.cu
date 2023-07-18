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
        unsigned long long offset = 0;
        // left of the diagonal -> use symmetrically mirrored values
        for (unsigned long long dim = 0; dim < i; ++dim) {
            offset += dim;
            temp += A[dim * k + i - offset] * B[dim * n + j];
        }
        // diagonal + right of the diagonal -> use contiguous values
        offset += i;
        for (unsigned long long dim = i; dim < k; ++dim) {
            temp += A[i * k + dim - offset] * B[dim * n + j];
        }
        C[i * n + j] = alpha * temp + beta * C[i * n + j];
    }
}

}  // namespace plssvm::cuda