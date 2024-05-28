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
 * @param[out] kernel_matrix_d the calculated kernel matrix
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
__kernel void device_kernel_assembly(__global real_type *kernel_matrix_d, const __global real_type *data_d, const ulong num_rows, const ulong device_num_rows, const ulong row_offset, const ulong num_features, const __global real_type *q, const real_type QA_cost, const real_type cost, const ulong grid_x_offset, const ulong grid_y_offset PLSSVM_OPENCL_KERNEL_FUNCTION_PARAMETER_LIST) {
    // cast values to 32-bit unsigned int values to prevent implicit conversions
    const uint local_id_0 = get_local_id(0);
    const uint local_id_1 = get_local_id(1);

    // cast all values to 64-bit unsigned long long to prevent potential 32-bit overflows
    const ulong threadIdx_x = get_local_id(0);                 // current thread in block x-dimension
    const ulong threadIdx_y = get_local_id(1);                 // current thread in block y-dimension
    const ulong blockDim_x = get_local_size(0);                // number of threads in block x-dimension
    const ulong blockDim_y = get_local_size(1);                // number of threads in block y-dimension
    const ulong blockIdx_x = get_group_id(0) + grid_x_offset;  // current block in grid x-dimension + offsets if the grid size would be too large
    const ulong blockIdx_y = get_group_id(1) + grid_y_offset;  // current block in grid y-dimension + offsets if the grid size would be too large

    // calculate the indices used in the current thread
    const ulong i = (blockIdx_x * blockDim_x + threadIdx_x) * INTERNAL_BLOCK_SIZE_ul;
    const ulong i_linear = blockIdx_x * blockDim_x * INTERNAL_BLOCK_SIZE_ul + threadIdx_x;
    const ulong j = (blockIdx_y * blockDim_y + threadIdx_y) * INTERNAL_BLOCK_SIZE_ul;
    const ulong j_linear = blockIdx_y * blockDim_y * INTERNAL_BLOCK_SIZE_ul + threadIdx_x;

    // create the local memory arrays used for caching data point features
    __local real_type data_cache_i[FEATURE_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];
    __local real_type data_cache_j[FEATURE_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];

    // only calculate the upper triangular matrix -> can't use get_local_id() since all work-items in a work-group must progress further
    if (blockIdx_x >= blockIdx_y) {
        // create a thread private array used for internal caching
        real_type temp[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE] = { (real_type) 0.0 };

        // iterate over all features using blocking to be able to cache them for faster memory accesses
        for (ulong dim = 0; dim < num_features; dim += FEATURE_BLOCK_SIZE_ul) {
            // load data into local memory
            for (uint internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                const ulong global_i = row_offset + i_linear + (ulong) internal * THREAD_BLOCK_SIZE_ul;
                const ulong global_j = row_offset + j_linear + (ulong) internal * THREAD_BLOCK_SIZE_ul;

                // FEATURE_BLOCK_SIZE = 2 * THREAD_BLOCK_SIZE -> store twice as many values in the local memory
                data_cache_i[local_id_1][internal * THREAD_BLOCK_SIZE + local_id_0] = data_d[(dim + threadIdx_y) * (num_rows + (ulong) 1 + PADDING_SIZE_ul) + global_i];
                data_cache_i[local_id_1 + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + local_id_0] = data_d[(dim + threadIdx_y + THREAD_BLOCK_SIZE_ul) * (num_rows + (ulong) 1 + PADDING_SIZE_ul) + global_i];
                data_cache_j[local_id_1][internal * THREAD_BLOCK_SIZE + local_id_0] = data_d[(dim + threadIdx_y) * (num_rows + (ulong) 1 + PADDING_SIZE_ul) + global_j];
                data_cache_j[local_id_1 + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + local_id_0] = data_d[(dim + threadIdx_y + THREAD_BLOCK_SIZE_ul) * (num_rows + (ulong) 1 + PADDING_SIZE_ul) + global_j];
            }
            barrier(CLK_LOCAL_MEM_FENCE);  // wait until all work-items loaded their part of the data

            // perform the feature reduction calculation
            for (uint block_dim = 0; block_dim < FEATURE_BLOCK_SIZE; ++block_dim) {
                for (uint internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                    for (uint internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                        temp[internal_i][internal_j] += PLSSVM_OPENCL_FEATURE_REDUCE_FUNCTION(data_cache_i[block_dim][local_id_0 * INTERNAL_BLOCK_SIZE + internal_i], data_cache_j[block_dim][local_id_1 * INTERNAL_BLOCK_SIZE + internal_j]);
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);  // wait until all work-items performed their part of the calculations
        }

        // apply the remaining part of the kernel function and store the value in the output kernel matrix
        for (uint internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
            for (uint internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                const ulong device_global_i = i + (ulong) internal_i;
                const ulong global_i = row_offset + i + (ulong) internal_i;
                const ulong device_global_j = j + (ulong) internal_j;
                const ulong global_j = row_offset + j + (ulong) internal_j;

                // be sure to not perform out of bounds accesses for the kernel matrix (only using the upper triangular matrix)
                if (device_global_i < (num_rows - row_offset) && device_global_j < device_num_rows && global_i >= global_j) {
                    real_type temp_ij = temp[internal_i][internal_j];
                    temp_ij = PLSSVM_OPENCL_APPLY_KERNEL_FUNCTION(temp_ij PLSSVM_OPENCL_KERNEL_FUNCTION_PARAMETER) + QA_cost - q[global_i] - q[global_j];
                    // apply the cost on the diagonal
                    if (global_i == global_j) {
                        temp_ij += cost;
                    }
                    // update the kernel matrix
                    kernel_matrix_d[device_global_j * (num_rows - row_offset + PADDING_SIZE_ul) - device_global_j * (device_global_j + (ulong) 1) / (ulong) 2 + device_global_i] = temp_ij;
                }
            }
        }
    }
}
