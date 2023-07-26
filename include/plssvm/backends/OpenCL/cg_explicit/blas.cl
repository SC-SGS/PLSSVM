/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Functions for explicitly performing a BLAS GEMM like matrix-matrix multiplication using the OpenCL backend.
 */

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

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
__kernel void device_kernel_gemm(const ulong m, const ulong n, const ulong k, const real_type alpha, __global const real_type *A, __global const real_type *B, const real_type beta, __global real_type *C) {
    // compute: C = alpha * A * B + beta * C with A in m x k, B in n x k, and C in n x m, alpha, beta as scalar
    const ulong i = get_global_id(0);  // # rhs
    const ulong j = get_global_id(1);  // # rows
    const ulong j_cached_idx = get_group_id(1) * get_local_size(1) + get_local_id(0);

    __local real_type A_cache[FEATURE_BLOCK_SIZE_OLD][THREAD_BLOCK_SIZE_OLD];
    __local real_type B_cache[FEATURE_BLOCK_SIZE_OLD][THREAD_BLOCK_SIZE_OLD];

    real_type temp = 0.0;

    for (ulong dim = 0; dim < k; dim += FEATURE_BLOCK_SIZE_OLD) {
        // zero out shared memory
        if (get_local_id(1) < FEATURE_BLOCK_SIZE_OLD) {
            A_cache[get_local_id(1)][get_local_id(0)] = 0.0;
            B_cache[get_local_id(1)][get_local_id(0)] = 0.0;
        }

        // load data into shared memory
        if (get_local_id(1) < FEATURE_BLOCK_SIZE_OLD && dim + get_local_id(1) < k) {
            if (dim + get_local_id(1) < j_cached_idx) {
                if (j_cached_idx < k) {
                    A_cache[get_local_id(1)][get_local_id(0)] = A[(dim + get_local_id(1)) * k + j_cached_idx - (dim + get_local_id(1)) * (dim + get_local_id(1) + 1) / 2];
                }
            } else {
                A_cache[get_local_id(1)][get_local_id(0)] = A[j_cached_idx * k + dim + get_local_id(1) - j_cached_idx * (j_cached_idx + 1) / 2];
            }
            if (i < n) {
                B_cache[get_local_id(1)][get_local_id(0)] = B[(dim + get_local_id(1)) * n + i];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // calculation
        for (ulong block_dim = 0; block_dim < FEATURE_BLOCK_SIZE_OLD; ++block_dim) {
            temp += A_cache[block_dim][get_local_id(1)] * B_cache[block_dim][get_local_id(0)];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (i < n && j < m) {
        C[j * n + i] = alpha * temp + beta * C[j * n + i];
    }
}