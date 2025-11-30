/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Functions for explicitly assembling the kernel matrix using the OpenCL backend.
 */

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

/**
 * @brief Create the explicit kernel matrix using the kernel function determined at runtime.
 * @details The `PLSSVM_OPENCL_KERNEL_FUNCTION_PARAMETER_LIST`, `PLSSVM_OPENCL_KERNEL_FUNCTION_PARAMETER`, `PLSSVM_OPENCL_FEATURE_REDUCE_FUNCTION`, and `PLSSVM_OPENCL_APPLY_KERNEL_FUNCTION` placeholder will be replaced by the correct values upon kernel construction.
 * @param[out] kernel_matrix the calculated kernel matrix
 * @param[in] data the data points to calculate the kernel matrix from
 * @param[in] num_rows the total number of data points (= total number of rows)
 * @param[in] device_num_rows the number of rows the current device is responsible for
 * @param[in] device_row_offset the first row in @p data_d the current device is responsible for
 * @param[in] num_features the number of features per data point
 * @param[in] q the vector used in the dimensional reduction
 * @param[in] QA_cost the scalar used in the dimensional reduction
 * @param[in] cost the cost factor the diagonal is scaled with
 * @param[in] grid_x_offset the offset in x-dimension into the data points if more than one execution grid has to be used
 * @param[in] grid_y_offset the offset in y-dimension into the data points if more than one execution grid has to be used
 * @param[in] PLSSVM_OPENCL_KERNEL_FUNCTION_PARAMETER_LIST a placeholder that is used to string replace the correct kernel parameter (attention: no comma!; Args... only added for Doxygen)
 */
__kernel void device_kernel_assembly(__global real_type *kernel_matrix, const __global real_type *data, const ulong num_rows, const ulong device_num_rows, const ulong device_row_offset, const ulong num_features, const __global real_type *q, const real_type QA_cost, const real_type cost, const ulong grid_x_offset, const ulong grid_y_offset PLSSVM_OPENCL_KERNEL_FUNCTION_PARAMETER_LIST) {
    // cast values to 32-bit unsigned int values to prevent implicit conversions
    const uint local_id_0 = get_local_id(0);
    const uint local_id_1 = get_local_id(1);

    // cast all values to 64-bit unsigned long long to prevent potential 32-bit overflows
    const ulong threadIdx_x = get_local_id(0);                 // current work-item in work-group x-dimension
    const ulong threadIdx_y = get_local_id(1);                 // current work-item in work-group y-dimension
    const ulong blockDim_x = get_local_size(0);                // number of work-items in work-group x-dimension
    const ulong blockDim_y = get_local_size(1);                // number of work-items in work-group y-dimension
    const ulong blockIdx_x = get_group_id(0) + grid_x_offset;  // current work-group in global range x-dimension + offsets if the global range is too large
    const ulong blockIdx_y = get_group_id(1) + grid_y_offset;  // current work-group in global range y-dimension + offsets if the global range is too large

    // create two local memory arrays used for caching
    __local real_type data_i_cache[THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];
    __local real_type data_j_cache[THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];

    // only calculate the upper triangular matrix -> can't use get_local_id() since all work-items in a work-group must progress further
    if (blockIdx_x >= blockIdx_y) {
        // create a private memory array used for internal caching
        real_type temp[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE] = { (real_type) 0.0 };

        {
            // calculate the indices used in the current work-item, pays attention to coalesced memory accesses
            const ulong i_idx_linear = blockIdx_x * blockDim_x * INTERNAL_BLOCK_SIZE_uz + threadIdx_x;  // num_rows - device_row_offset
            const ulong j_idx_linear = blockIdx_y * blockDim_y * INTERNAL_BLOCK_SIZE_uz + threadIdx_x;  // device_num_rows

            // iterate over all features using blocking to be able to cache them for faster memory accesses
            for (ulong feature_block = 0; feature_block < num_features; feature_block += THREAD_BLOCK_SIZE_uz) {
                // load data into local memory
                for (uint internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                    // calculate the indices to access the global data, pays attention to coalesced memory accesses
                    const ulong global_i_idx_linear = device_row_offset + i_idx_linear + (ulong) internal * THREAD_BLOCK_SIZE_uz;
                    const ulong global_j_idx_linear = device_row_offset + j_idx_linear + (ulong) internal * THREAD_BLOCK_SIZE_uz;

                    // store the values in the local memory
                    data_i_cache[local_id_1][internal * THREAD_BLOCK_SIZE + local_id_0] = data[(feature_block + threadIdx_y) * (num_rows + (ulong) 1 + PADDING_SIZE_uz) + global_i_idx_linear];  // SoA
                    data_j_cache[local_id_1][internal * THREAD_BLOCK_SIZE + local_id_0] = data[(feature_block + threadIdx_y) * (num_rows + (ulong) 1 + PADDING_SIZE_uz) + global_j_idx_linear];  // SoA
                }
                barrier(CLK_LOCAL_MEM_FENCE);  // wait until all work-items loaded their part of the data

#if defined(PLSSVM_OPENCL_TARGET_CPUS)
                // perform the feature reduction calculation, the feature is the fastest moving index
                for (uint internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                    for (uint internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                        real_type sum = 0.0;
                        for (uint feature = 0; feature < THREAD_BLOCK_SIZE; ++feature) {
                            sum += PLSSVM_OPENCL_FEATURE_REDUCE_FUNCTION(data_i_cache[feature][local_id_0 * INTERNAL_BLOCK_SIZE + internal_i], data_j_cache[feature][local_id_1 * INTERNAL_BLOCK_SIZE + internal_j]);
                        }
                        temp[internal_i][internal_j] += sum;
                    }
                }
#else
                // perform the feature reduction calculation, the feature is the slowest moving index
                for (uint feature = 0; feature < THREAD_BLOCK_SIZE; ++feature) {
                    for (uint internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                        for (uint internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                            temp[internal_i][internal_j] += PLSSVM_OPENCL_FEATURE_REDUCE_FUNCTION(data_i_cache[feature][local_id_0 * INTERNAL_BLOCK_SIZE + internal_i], data_j_cache[feature][local_id_1 * INTERNAL_BLOCK_SIZE + internal_j]);
                        }
                    }
                }
#endif
                barrier(CLK_LOCAL_MEM_FENCE);  // wait until all work-items performed their part of the calculations
            }
        }

        // calculate the indices used in the current work-item
        const ulong i_idx = (blockIdx_x * blockDim_x + threadIdx_x) * INTERNAL_BLOCK_SIZE_uz;  // num_rows - device_row_offset
        const ulong j_idx = (blockIdx_y * blockDim_y + threadIdx_y) * INTERNAL_BLOCK_SIZE_uz;  // device_num_rows

        // apply the remaining part of the kernel function and store the value in the output kernel matrix
        for (uint internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
            for (uint internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                // calculate the indices to access the global data and the data with respect to the current device
                const ulong device_global_i_idx = i_idx + (ulong) internal_i;
                const ulong global_i_idx = device_row_offset + device_global_i_idx;
                const ulong device_global_j_idx = j_idx + (ulong) internal_j;
                const ulong global_j_idx = device_row_offset + device_global_j_idx;

                // be sure to not perform out-of-bounds accesses (only using the upper triangular matrix)
                if (device_global_i_idx < (num_rows - device_row_offset) && device_global_j_idx < device_num_rows && global_i_idx >= global_j_idx) {
                    real_type temp_ij = temp[internal_i][internal_j];
                    // apply the final kernel function
                    temp_ij = PLSSVM_OPENCL_APPLY_KERNEL_FUNCTION(temp_ij PLSSVM_OPENCL_KERNEL_FUNCTION_PARAMETER) + QA_cost - q[global_i_idx] - q[global_j_idx];
                    // apply the cost on the diagonal
                    if (global_i_idx == global_j_idx) {
                        temp_ij += cost;
                    }
                    // update the upper triangular kernel matrix
                    kernel_matrix[device_global_j_idx * (num_rows - device_row_offset + PADDING_SIZE_uz) - device_global_j_idx * (device_global_j_idx + (ulong) 1) / (ulong) 2 + device_global_i_idx] = temp_ij;
                }
            }
        }
    }
}
