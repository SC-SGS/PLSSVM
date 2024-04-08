/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Functions for explicitly assemblying the kernel matrix using the OpenCL backend.
 */

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

/**
 * @brief Create the explicit kernel matrix using the kernel function determined at runtime.
 * @details The `PLSSVM_OPENCL_KERNEL_FUNCTION_PARAMETER_LIST`, `PLSSVM_OPENCL_KERNEL_FUNCTION_PARAMETER`, `PLSSVM_OPENCL_FEATURE_REDUCE_FUNCTION`, and `PLSSVM_OPENCL_APPLY_KERNEL_FUNCTION` placeholder will be replaced by the correct values upon kernel construction.
 * @param[out] ret the calculated kernel matrix
 * @param[in] data_d the data points to calculate the kernel matrix from
 * @param[in] num_rows the total number of data points (= total number of rows)
 * @param[in] device_num_rows the number of rows the current device is responsible for
 * @param[in] row_offset the first row in @p data_d the current device is responsible for
 * @param[in] num_features the number of features per data point
 * @param[in] q the vector used in the dimensional reduction
 * @param[in] QA_cost the scalar used in the dimensional reduction
 * @param[in] cost the cost factor the diagonal is scaled with
 * @param[in] PLSSVM_OPENCL_KERNEL_FUNCTION_PARAMETER_LIST a placeholder that is used to string replace the correct kernel parameter (attention: no comma!)
 */
__kernel void device_kernel_assembly(__global real_type *ret, const __global real_type *data_d, const ulong num_rows, const ulong device_num_rows, const ulong row_offset, const ulong num_features, const __global real_type *q, const real_type QA_cost, const real_type cost PLSSVM_OPENCL_KERNEL_FUNCTION_PARAMETER_LIST) {
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

        for (uint internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
            for (uint internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                const ulong device_global_i = i + internal_i;
                const ulong global_i = row_offset + i + internal_i;
                const ulong device_global_j = j + internal_j;
                const ulong global_j = row_offset + j + internal_j;

                if (device_global_i < (num_rows - row_offset) && device_global_j < device_num_rows && global_i >= global_j) {
                    real_type temp_ij = temp[internal_i][internal_j];
                    temp_ij = PLSSVM_OPENCL_APPLY_KERNEL_FUNCTION(temp_ij PLSSVM_OPENCL_KERNEL_FUNCTION_PARAMETER) + QA_cost - q[global_i] - q[global_j];
                    if (global_i == global_j) {
                        temp_ij += cost;
                    }
                    ret[device_global_j * (num_rows - row_offset + PADDING_SIZE) - device_global_j * (device_global_j + 1) / 2 + device_global_i] = temp_ij;
                }
            }
        }
    }
}
