/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Functions for implicitly assembling the kernel matrix using the OpenCL backend.
 */

// #include "detail/atomics.cl"  // atomicAdd -> included via string concatenation when building the device kernels

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

/**
 * @brief Perform an implicit BLAS SYMM-like operation: `C = alpha * A * B + C` where `A` is the implicitly calculated kernel matrix using the kernel function determined at runtime (never actually stored, reducing the amount of needed global memory), @p B and @p C are matrices, and @p alpha is a scalar.
 * @details The `PLSSVM_OPENCL_KERNEL_FUNCTION_PARAMETER_LIST`, `PLSSVM_OPENCL_KERNEL_FUNCTION_PARAMETER`, `PLSSVM_OPENCL_FEATURE_REDUCE_FUNCTION`, and `PLSSVM_OPENCL_APPLY_KERNEL_FUNCTION` placeholder will be replaced by the correct values upon kernel construction.
 * @note The beta factor is already applied to C before this kernel starts!
 * @param[in] alpha the scalar alpha value
 * @param[in] q the vector used in the dimensional reduction
 * @param[in] data_d the data points to calculate the implicit kernel matrix from
 * @param[in] num_rows the number of data points
 * @param[in] num_features the number of features per data point
 * @param[in] QA_cost the scalar used in the dimensional reduction
 * @param[in] cost the cost factor the diagonal is scaled with
 * @param[in] B the matrix @p B
 * @param[in,out] C the matrix @p C
 * @param[in] num_classes the number of classes in the data set
 * @param[in] PLSSVM_OPENCL_KERNEL_FUNCTION_PARAMETER_LIST a placeholder that is used to string replace the correct kernel parameter (attention: no comma!)
 */
__kernel void device_kernel_assembly_symm(const real_type alpha, const __global real_type *q, const __global real_type *data_d, const ulong num_rows, const ulong device_num_rows, const ulong row_offset, const ulong num_features, const real_type QA_cost, const real_type cost, const __global real_type *B, __global real_type *C, const ulong num_classes PLSSVM_OPENCL_KERNEL_FUNCTION_PARAMETER_LIST) {
    const ulong i = get_global_id(0) * INTERNAL_BLOCK_SIZE;
    const ulong i_linear = get_group_id(0) * get_local_size(0) * INTERNAL_BLOCK_SIZE + get_local_id(0);
    const ulong j = get_global_id(1) * INTERNAL_BLOCK_SIZE;
    const ulong j_cached_idx_linear = get_group_id(1) * get_local_size(1) * INTERNAL_BLOCK_SIZE + get_local_id(0);

    __local real_type data_cache_i[FEATURE_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];
    __local real_type data_cache_j[FEATURE_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];

    if (get_group_id(0) >= get_group_id(1)) {
        real_type temp[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE] = { 0.0 };

        for (ulong dim = 0; dim < num_features; dim += FEATURE_BLOCK_SIZE) {
            // load data into shared memory
            for (uint internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                const ulong global_i = row_offset + i_linear + internal * THREAD_BLOCK_SIZE;
                const ulong global_j = row_offset + j_cached_idx_linear + internal * THREAD_BLOCK_SIZE;

                data_cache_i[get_local_id(1)][internal * THREAD_BLOCK_SIZE + get_local_id(0)] = data_d[(dim + get_local_id(1)) * (num_rows + 1 + PADDING_SIZE) + global_i];
                data_cache_i[get_local_id(1) + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + get_local_id(0)] = data_d[(dim + get_local_id(1) + THREAD_BLOCK_SIZE) * (num_rows + 1 + PADDING_SIZE) + global_i];
                data_cache_j[get_local_id(1)][internal * THREAD_BLOCK_SIZE + get_local_id(0)] = data_d[(dim + get_local_id(1)) * (num_rows + 1 + PADDING_SIZE) + global_j];
                data_cache_j[get_local_id(1) + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + get_local_id(0)] = data_d[(dim + get_local_id(1) + THREAD_BLOCK_SIZE) * (num_rows + 1 + PADDING_SIZE) + global_j];
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            // calculation
            for (uint block_dim = 0; block_dim < FEATURE_BLOCK_SIZE; ++block_dim) {
                for (uint internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                    for (uint internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                        temp[internal_i][internal_j] += PLSSVM_OPENCL_FEATURE_REDUCE_FUNCTION(data_cache_i[block_dim][get_local_id(0) * INTERNAL_BLOCK_SIZE + internal_i], data_cache_j[block_dim][get_local_id(1) * INTERNAL_BLOCK_SIZE + internal_j]);
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        // update temp using the kernel function taking the dimensional reduction into account and apply the cost to the diagonal
        for (uint internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
            for (uint internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                const ulong global_i = row_offset + i + internal_i;
                const ulong device_global_i = i + internal_i;
                const ulong global_j = row_offset + j + internal_j;
                const ulong device_global_j = j + internal_j;

                if ((device_global_i < (num_rows - row_offset) && device_global_j < device_num_rows && global_i >= global_j)) {
                    temp[internal_i][internal_j] = PLSSVM_OPENCL_APPLY_KERNEL_FUNCTION(temp[internal_i][internal_j] PLSSVM_OPENCL_KERNEL_FUNCTION_PARAMETER) + QA_cost - q[global_i] - q[global_j];
                    if (global_i == global_j) {
                        temp[internal_i][internal_j] += cost;
                    }
                } else {
                    temp[internal_i][internal_j] = 0.0;
                }
            }
        }

        // calculate C += alpha * temp * B for the UPPER triangular matrix
        {
            real_type (*B_cache)[FEATURE_BLOCK_SIZE] = (real_type (*)[FEATURE_BLOCK_SIZE]) data_cache_i;
            real_type (*C_out_cache)[FEATURE_BLOCK_SIZE] = (real_type (*)[FEATURE_BLOCK_SIZE]) data_cache_j;

            for (ulong dim = 0; dim < num_classes; dim += FEATURE_BLOCK_SIZE) {
                // load data into shared memory
                for (uint internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                    const ulong global_i = row_offset + i_linear + internal * THREAD_BLOCK_SIZE;

                    B_cache[internal * THREAD_BLOCK_SIZE + get_local_id(0)][get_local_id(1)] = alpha * B[global_i * (num_classes + PADDING_SIZE) + dim + get_local_id(1)];
                    B_cache[internal * THREAD_BLOCK_SIZE + get_local_id(0)][get_local_id(1) + THREAD_BLOCK_SIZE] = alpha * B[global_i * (num_classes + PADDING_SIZE) + dim + get_local_id(1) + THREAD_BLOCK_SIZE];

                    C_out_cache[internal * THREAD_BLOCK_SIZE + get_local_id(0)][get_local_id(1)] = 0.0;
                    C_out_cache[internal * THREAD_BLOCK_SIZE + get_local_id(0)][get_local_id(1) + THREAD_BLOCK_SIZE] = 0.0;
                }
                barrier(CLK_LOCAL_MEM_FENCE);

                // calculate intermediate results and store them in shared memory
                for (uint class_idx = 0; class_idx < FEATURE_BLOCK_SIZE; ++class_idx) {
                    for (uint internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                        for (uint internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                            C_out_cache[get_local_id(1) * INTERNAL_BLOCK_SIZE + internal_j][(class_idx + get_local_id(0)) % FEATURE_BLOCK_SIZE] +=
                                temp[internal_i][internal_j] * B_cache[get_local_id(0) * INTERNAL_BLOCK_SIZE + internal_i][(class_idx + get_local_id(0)) % FEATURE_BLOCK_SIZE];
                        }
                    }
                    barrier(CLK_LOCAL_MEM_FENCE);
                }

                // add intermediate cached results to C
                for (uint internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                    const ulong global_j = row_offset + j + internal;
                    atomicAdd(&C[global_j * (num_classes + PADDING_SIZE) + dim + get_local_id(0)], C_out_cache[get_local_id(1) * INTERNAL_BLOCK_SIZE + internal][get_local_id(0)]);
                    atomicAdd(&C[global_j * (num_classes + PADDING_SIZE) + dim + get_local_id(0) + THREAD_BLOCK_SIZE], C_out_cache[get_local_id(1) * INTERNAL_BLOCK_SIZE + internal][get_local_id(0) + THREAD_BLOCK_SIZE]);
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
        }

        // set potential diagonal entries in temp to 0.0 such that we don't apply the main diagonal twice to C
        for (uint internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
            for (uint internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                const ulong global_i = row_offset + i + internal_i;
                const ulong global_j = row_offset + j + internal_j;

                if (global_i == global_j) {
                    temp[internal_i][internal_j] = 0.0;
                }
            }
        }

        // calculate C += alpha * temp * B for the LOWER triangular matrix
        {
            real_type (*B_cache)[INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE] = (real_type (*)[INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE]) data_cache_i;
            real_type (*C_out_cache)[INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE] = (real_type (*)[INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE]) data_cache_j;

            for (ulong dim = 0; dim < num_classes; dim += FEATURE_BLOCK_SIZE) {
                // load data into shared memory
                for (uint internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                    const ulong global_j = row_offset + j_cached_idx_linear + internal * THREAD_BLOCK_SIZE;

                    B_cache[get_local_id(1)][internal * THREAD_BLOCK_SIZE + get_local_id(0)] = alpha * B[global_j * (num_classes + PADDING_SIZE) + dim + get_local_id(1)];
                    B_cache[get_local_id(1) + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + get_local_id(0)] = alpha * B[global_j * (num_classes + PADDING_SIZE) + dim + get_local_id(1) + THREAD_BLOCK_SIZE];

                    C_out_cache[get_local_id(1)][internal * THREAD_BLOCK_SIZE + get_local_id(0)] = 0.0;
                    C_out_cache[get_local_id(1) + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + get_local_id(0)] = 0.0;
                }
                barrier(CLK_LOCAL_MEM_FENCE);

                // calculate intermediate results and store them in shared memory
                for (uint class_idx = 0; class_idx < FEATURE_BLOCK_SIZE; ++class_idx) {
                    for (uint internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                        for (uint internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                            C_out_cache[(class_idx + get_local_id(1)) % FEATURE_BLOCK_SIZE][internal_i * THREAD_BLOCK_SIZE + get_local_id(0)] +=
                                temp[internal_i][internal_j] * B_cache[(class_idx + get_local_id(1)) % FEATURE_BLOCK_SIZE][get_local_id(1) * INTERNAL_BLOCK_SIZE + internal_j];
                        }
                    }
                    barrier(CLK_LOCAL_MEM_FENCE);
                }

                // add intermediate cached results to C
                for (uint internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                    const ulong global_i = row_offset + i + internal;
                    atomicAdd(&C[global_i * (num_classes + PADDING_SIZE) + dim + get_local_id(1)], C_out_cache[get_local_id(1)][internal * THREAD_BLOCK_SIZE + get_local_id(0)]);
                    atomicAdd(&C[global_i * (num_classes + PADDING_SIZE) + dim + get_local_id(1) + THREAD_BLOCK_SIZE], C_out_cache[get_local_id(1) + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + get_local_id(0)]);
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
        }
    }
}
