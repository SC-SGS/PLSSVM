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
    const unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;  // # rhs
    const unsigned long long j = blockIdx.y * blockDim.y + threadIdx.y;  // # rows

    constexpr unsigned long long WARP_SIZE = 32;
    constexpr unsigned long long BLOCK_SIZE = 16;

    __shared__ real_type A_cache[BLOCK_SIZE][WARP_SIZE];
    __shared__ real_type B_cache[BLOCK_SIZE][WARP_SIZE];

    real_type temp{ 0.0 };

    for (unsigned long long dim = 0; dim < k; dim += BLOCK_SIZE) {
        // zero out shared memory
        if (threadIdx.x < BLOCK_SIZE) {
            A_cache[threadIdx.x][threadIdx.y] = real_type{ 0.0 };
        }
        if (threadIdx.y < BLOCK_SIZE) {
            B_cache[threadIdx.y][threadIdx.x] = real_type{ 0.0 };
        }

        // load data into shared memory
        if (threadIdx.y < BLOCK_SIZE) {
            B_cache[threadIdx.y][threadIdx.x] = B[(dim + threadIdx.y) * n + i];
        }
        if (threadIdx.x < BLOCK_SIZE) {
            if (dim + threadIdx.x < j) {
                A_cache[threadIdx.x][threadIdx.y] = A[(dim + threadIdx.x) * k + j - (dim + threadIdx.x) * (dim + threadIdx.x + 1) / 2];
            } else {
                A_cache[threadIdx.x][threadIdx.y] = A[j * k + dim + threadIdx.x - j * (j + 1) / 2];
            }
        }
        __syncthreads();

        // calculation
        for (unsigned long long block_dim = 0; block_dim < BLOCK_SIZE; ++block_dim) {
            temp += A_cache[block_dim][threadIdx.y] * B_cache[block_dim][threadIdx.x];
        }
        __syncthreads();
    }

    if (i < n && j < m) {
        C[j * n + i] = alpha * temp + beta * C[j * n + i];
    }
}

}  // namespace plssvm::cuda