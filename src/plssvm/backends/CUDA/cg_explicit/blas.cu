/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/CUDA/cg_explicit/blas.cuh"

#include "plssvm/constants.hpp"  // plssvm::real_type, plssvm::THREAD_BLOCK_SIZE, plssvm::FEATURE_BLOCK_SIZE

namespace plssvm::cuda {

__global__ void device_kernel_gemm(const unsigned long long m, const unsigned long long n, const unsigned long long k, const real_type alpha, const real_type *A, const real_type *B, const real_type beta, real_type *C) {
    // compute: C = alpha * A * B + beta * C with A in m x k, B in n x k, and C in n x m, alpha, beta as scalar
    const unsigned long long i = (blockIdx.x * blockDim.x + threadIdx.x) * INTERNAL_BLOCK_SIZE;  // # rhs
    const unsigned long long i_linear = blockIdx.x * blockDim.x * INTERNAL_BLOCK_SIZE + threadIdx.x;
    const unsigned long long j = (blockIdx.y * blockDim.y + threadIdx.y) * INTERNAL_BLOCK_SIZE;  // # rows
    const unsigned long long j_cached_idx_linear = blockIdx.y * blockDim.y * INTERNAL_BLOCK_SIZE + threadIdx.x;

    __shared__ real_type A_cache[FEATURE_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];
    __shared__ real_type B_cache[FEATURE_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];

    real_type temp[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE] = { 0.0 };

    for (unsigned long long dim = 0; dim < k; dim += FEATURE_BLOCK_SIZE) {
        // zero out shared memory
        for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
            A_cache[threadIdx.y][internal * THREAD_BLOCK_SIZE + threadIdx.x] = real_type{ 0.0 };
            A_cache[threadIdx.y + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + threadIdx.x] = real_type{ 0.0 };
            B_cache[threadIdx.y][internal * THREAD_BLOCK_SIZE + threadIdx.x] = real_type{ 0.0 };
            B_cache[threadIdx.y + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + threadIdx.x] = real_type{ 0.0 };
        }

        // load data into shared memory
        for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
            const unsigned long long global_i = i_linear + internal * THREAD_BLOCK_SIZE;
            const unsigned long long global_j = j_cached_idx_linear + internal * THREAD_BLOCK_SIZE;

            if (global_j < k) {
                if (dim + threadIdx.y < k) {
                    A_cache[threadIdx.y][internal * THREAD_BLOCK_SIZE + threadIdx.x] = A[(dim + threadIdx.y) * k + global_j];
                }
                if (dim + threadIdx.y + THREAD_BLOCK_SIZE < k) {
                    A_cache[threadIdx.y + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + threadIdx.x] = A[(dim + threadIdx.y + THREAD_BLOCK_SIZE) * k + global_j];
                }
            }

            if (global_i < n) {
                if (dim + threadIdx.y < k) {
                    B_cache[threadIdx.y][internal * THREAD_BLOCK_SIZE + threadIdx.x] = B[(dim + threadIdx.y) * n + global_i];
                }
                if (dim + threadIdx.y + THREAD_BLOCK_SIZE < k) {
                    B_cache[threadIdx.y + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + threadIdx.x] = B[(dim + threadIdx.y + THREAD_BLOCK_SIZE) * n + global_i];
                }
            }
        }
        __syncthreads();

        // calculation
        for (unsigned block_dim = 0; block_dim < FEATURE_BLOCK_SIZE; ++block_dim) {
            for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                    temp[internal_i][internal_j] += A_cache[block_dim][threadIdx.y * INTERNAL_BLOCK_SIZE + internal_j] * B_cache[block_dim][threadIdx.x * INTERNAL_BLOCK_SIZE + internal_i];
                }
            }
        }
        __syncthreads();
    }

    for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
        for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
            const unsigned long long global_i = i + internal_i;
            const unsigned long long global_j = j + internal_j;

            if (global_i < n && global_j < m) {
                C[global_j * n + global_i] = alpha * temp[internal_i][internal_j] + beta * C[global_j * n + global_i];
            }
        }
    }
}

__global__ void device_kernel_symm(const unsigned long long m, const unsigned long long n, const unsigned long long k, const real_type alpha, const real_type *A, const real_type *B, const real_type beta, real_type *C) {
    // compute: C = alpha * A * B + beta * C with A in m x k, B in n x k, and C in n x m, alpha, beta as scalar
    const unsigned long long i = (blockIdx.x * blockDim.x + threadIdx.x) * INTERNAL_BLOCK_SIZE;  // # rhs
    const unsigned long long i_linear = blockIdx.x * blockDim.x * INTERNAL_BLOCK_SIZE + threadIdx.x;
    const unsigned long long j = (blockIdx.y * blockDim.y + threadIdx.y) * INTERNAL_BLOCK_SIZE;  // # rows
    const unsigned long long j_cached_idx_linear = blockIdx.y * blockDim.y * INTERNAL_BLOCK_SIZE + threadIdx.x;

    __shared__ real_type A_cache[FEATURE_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];
    __shared__ real_type B_cache[FEATURE_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];

    real_type temp[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE] = { 0.0 };

    for (unsigned long long dim = 0; dim < k; dim += FEATURE_BLOCK_SIZE) {
        // zero out shared memory
        for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
            A_cache[threadIdx.y][internal * THREAD_BLOCK_SIZE + threadIdx.x] = real_type{ 0.0 };
            A_cache[threadIdx.y + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + threadIdx.x] = real_type{ 0.0 };
            B_cache[threadIdx.y][internal * THREAD_BLOCK_SIZE + threadIdx.x] = real_type{ 0.0 };
            B_cache[threadIdx.y + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + threadIdx.x] = real_type{ 0.0 };
        }

        // load data into shared memory
        for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
            const unsigned long long global_i = i_linear + internal * THREAD_BLOCK_SIZE;
            const unsigned long long global_j = j_cached_idx_linear + internal * THREAD_BLOCK_SIZE;

            if (dim + threadIdx.y < k) {
                if (dim + threadIdx.y < global_j) {
                    if (global_j < k) {
                        A_cache[threadIdx.y][internal * THREAD_BLOCK_SIZE + threadIdx.x] = A[(dim + threadIdx.y) * k + global_j - (dim + threadIdx.y) * (dim + threadIdx.y + 1) / 2];
                    }
                } else {
                    A_cache[threadIdx.y][internal * THREAD_BLOCK_SIZE + threadIdx.x] = A[global_j * k + dim + threadIdx.y - global_j * (global_j + 1) / 2];
                }
            }

            if (dim + threadIdx.y + THREAD_BLOCK_SIZE < k) {
                if (dim + threadIdx.y + THREAD_BLOCK_SIZE < global_j) {
                    if (global_j < k) {
                        A_cache[threadIdx.y + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + threadIdx.x] = A[(dim + threadIdx.y + THREAD_BLOCK_SIZE) * k + global_j - (dim + threadIdx.y + THREAD_BLOCK_SIZE) * (dim + threadIdx.y + THREAD_BLOCK_SIZE + 1) / 2];
                    }
                } else {
                    A_cache[threadIdx.y + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + threadIdx.x] = A[global_j * k + dim + threadIdx.y + THREAD_BLOCK_SIZE - global_j * (global_j + 1) / 2];
                }
            }

            if (global_i < n) {
                if (dim + threadIdx.y < k) {
                    B_cache[threadIdx.y][internal * THREAD_BLOCK_SIZE + threadIdx.x] = B[(dim + threadIdx.y) * n + global_i];
                }
                if (dim + threadIdx.y + THREAD_BLOCK_SIZE < k) {
                    B_cache[threadIdx.y + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + threadIdx.x] = B[(dim + threadIdx.y + THREAD_BLOCK_SIZE) * n + global_i];
                }
            }
        }
        __syncthreads();

        // calculation
        for (unsigned block_dim = 0; block_dim < FEATURE_BLOCK_SIZE; ++block_dim) {
            for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                    temp[internal_i][internal_j] += A_cache[block_dim][threadIdx.y * INTERNAL_BLOCK_SIZE + internal_j] * B_cache[block_dim][threadIdx.x * INTERNAL_BLOCK_SIZE + internal_i];
                }
            }
        }
        __syncthreads();
    }

    for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
        for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
            const unsigned long long global_i = i + internal_i;
            const unsigned long long global_j = j + internal_j;

            if (global_i < n && global_j < m) {
                C[global_j * n + global_i] = alpha * temp[internal_i][internal_j] + beta * C[global_j * n + global_i];
            }
        }
    }
}

}  // namespace plssvm::cuda