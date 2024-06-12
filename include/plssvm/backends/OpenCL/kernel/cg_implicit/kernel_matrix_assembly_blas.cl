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
 * @param[in] device_num_rows the number of rows the current device is responsible for
 * @param[in] row_offset the first row in @p data_d the current device is responsible for
 * @param[in] num_features the number of features per data point
 * @param[in] QA_cost the scalar used in the dimensional reduction
 * @param[in] cost the cost factor the diagonal is scaled with
 * @param[in] B the matrix @p B
 * @param[in,out] C the matrix @p C
 * @param[in] num_classes the number of classes in the data set
 * @param[in] grid_x_offset the offset in x-dimension into the data points if more than one execution grid has to be used
 * @param[in] grid_y_offset the offset in y-dimension into the data points if more than one execution grid has to be used
 * @param[in] PLSSVM_OPENCL_KERNEL_FUNCTION_PARAMETER_LIST a placeholder that is used to string replace the correct kernel parameter (attention: no comma!; Args... only added for Doxygen)
 */
__kernel void device_kernel_assembly_symm(const real_type alpha, const __global real_type *q, const __global real_type *data_d, const ulong num_rows, const ulong device_num_rows, const ulong row_offset, const ulong num_features, const real_type QA_cost, const real_type cost, const __global real_type *B, __global real_type *C, const ulong num_classes, const ulong grid_x_offset, const ulong grid_y_offset PLSSVM_OPENCL_KERNEL_FUNCTION_PARAMETER_LIST) {
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

    // only calculate the upper triangular matrix -> can't use threadIdx since all threads in a warp must progress further
    if (blockIdx_x >= blockIdx_y) {
        // create a thread private array used for internal caching
        real_type temp[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE] = { (real_type) 0.0 };

        // iterate over all features using blocking to be able to cache them for faster memory accesses
        for (ulong dim = 0; dim < num_features; dim += FEATURE_BLOCK_SIZE_ul) {
            // load data into local memory
            for (uint internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                const ulong global_i = row_offset + i_linear + (ulong) internal * THREAD_BLOCK_SIZE_ul;
                const ulong global_j = row_offset + j_linear + (ulong) internal * THREAD_BLOCK_SIZE_ul;

                // FEATURE_BLOCK_SIZE = 2 * THREAD_BLOCK_SIZE -> store twice as many values in the shared memory
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
                const ulong global_i = row_offset + i + (ulong) internal_i;
                const ulong device_global_i = i + (ulong) internal_i;
                const ulong global_j = row_offset + j + (ulong) internal_j;
                const ulong device_global_j = j + (ulong) internal_j;

                // be sure to not perform out of bounds accesses for the kernel matrix (only using the upper triangular matrix)
                if ((device_global_i < (num_rows - row_offset) && device_global_j < device_num_rows && global_i >= global_j)) {
                    temp[internal_i][internal_j] = PLSSVM_OPENCL_APPLY_KERNEL_FUNCTION(temp[internal_i][internal_j] PLSSVM_OPENCL_KERNEL_FUNCTION_PARAMETER) + QA_cost - q[global_i] - q[global_j];
                    // apply the cost on the diagonal
                    if (global_i == global_j) {
                        temp[internal_i][internal_j] += cost;
                    }
                } else {
                    temp[internal_i][internal_j] = (real_type) 0.0;
                }
            }
        }

        // calculate C += alpha * temp * B for the UPPER triangular matrix
        {
            // reinterpret cache arrays with interchanged dimensions
            real_type (*B_cache)[FEATURE_BLOCK_SIZE] = (real_type (*)[FEATURE_BLOCK_SIZE]) data_cache_i;
            real_type (*C_out_cache)[FEATURE_BLOCK_SIZE] = (real_type (*)[FEATURE_BLOCK_SIZE]) data_cache_j;

            // iterate over all classes using blocking to be able to cache them for faster memory accesses
            for (ulong dim = 0; dim < num_classes; dim += FEATURE_BLOCK_SIZE_ul) {
                // load data into local memory
                for (uint internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                    const ulong global_i = row_offset + i_linear + (ulong) internal * THREAD_BLOCK_SIZE_ul;

                    // FEATURE_BLOCK_SIZE = 2 * THREAD_BLOCK_SIZE -> store twice as many values in the shared memory
                    B_cache[internal * THREAD_BLOCK_SIZE + local_id_0][local_id_1] = alpha * B[global_i * (num_classes + PADDING_SIZE_ul) + dim + threadIdx_y];
                    B_cache[internal * THREAD_BLOCK_SIZE + local_id_0][local_id_1 + THREAD_BLOCK_SIZE] = alpha * B[global_i * (num_classes + PADDING_SIZE_ul) + dim + threadIdx_y + THREAD_BLOCK_SIZE_ul];
                    C_out_cache[internal * THREAD_BLOCK_SIZE + local_id_0][local_id_1] = (real_type) 0.0;
                    C_out_cache[internal * THREAD_BLOCK_SIZE + local_id_0][local_id_1 + THREAD_BLOCK_SIZE] = (real_type) 0.0;
                }
                barrier(CLK_LOCAL_MEM_FENCE);  // wait until all work-items loaded their part of the data

                // calculate intermediate results and store them in local memory
                for (uint class_idx = 0; class_idx < FEATURE_BLOCK_SIZE; ++class_idx) {
                    for (uint internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                        for (uint internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                            C_out_cache[local_id_1 * INTERNAL_BLOCK_SIZE + internal_j][(class_idx + local_id_0) % FEATURE_BLOCK_SIZE] +=
                                temp[internal_i][internal_j] * B_cache[local_id_0 * INTERNAL_BLOCK_SIZE + internal_i][(class_idx + local_id_0) % FEATURE_BLOCK_SIZE];
                        }
                    }
                    barrier(CLK_LOCAL_MEM_FENCE);  // wait until all work-items performed their part of the calculations
                }

                // add intermediate cached results to C
                for (uint internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                    const ulong global_j = row_offset + j + (ulong) internal;
                    atomicAdd(&C[global_j * (num_classes + PADDING_SIZE_ul) + dim + threadIdx_x], C_out_cache[local_id_1 * INTERNAL_BLOCK_SIZE + internal][local_id_0]);
                    atomicAdd(&C[global_j * (num_classes + PADDING_SIZE_ul) + dim + threadIdx_x + THREAD_BLOCK_SIZE_ul], C_out_cache[local_id_1 * INTERNAL_BLOCK_SIZE + internal][local_id_0 + THREAD_BLOCK_SIZE]);
                }
                barrier(CLK_LOCAL_MEM_FENCE);  // wai until all threads updated C with their values
            }
        }

        // set potential diagonal entries in temp to 0.0 such that we don't apply the main diagonal twice to C
        for (uint internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
            for (uint internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                const ulong global_i = row_offset + i + (ulong) internal_i;
                const ulong global_j = row_offset + j + (ulong) internal_j;

                if (global_i == global_j) {
                    temp[internal_i][internal_j] = (real_type) 0.0;
                }
            }
        }

        // calculate C += alpha * temp * B for the LOWER triangular matrix
        {
            // reinterpret cache arrays with interchanged dimensions
            real_type (*B_cache)[INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE] = (real_type (*)[INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE]) data_cache_i;
            real_type (*C_out_cache)[INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE] = (real_type (*)[INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE]) data_cache_j;

            // iterate over all classes using blocking to be able to cache them for faster memory accesses
            for (ulong dim = 0; dim < num_classes; dim += FEATURE_BLOCK_SIZE_ul) {
                // load data into local memory
                for (uint internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                    const ulong global_j = row_offset + j_linear + (ulong) internal * THREAD_BLOCK_SIZE_ul;

                    // FEATURE_BLOCK_SIZE = 2 * THREAD_BLOCK_SIZE -> store twice as many values in the shared memory
                    B_cache[local_id_1][internal * THREAD_BLOCK_SIZE + local_id_0] = alpha * B[global_j * (num_classes + PADDING_SIZE_ul) + dim + threadIdx_y];
                    B_cache[local_id_1 + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + local_id_0] = alpha * B[global_j * (num_classes + PADDING_SIZE_ul) + dim + threadIdx_y + THREAD_BLOCK_SIZE_ul];
                    C_out_cache[local_id_1][internal * THREAD_BLOCK_SIZE + local_id_0] = (real_type) 0.0;
                    C_out_cache[local_id_1 + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + local_id_0] = (real_type) 0.0;
                }
                barrier(CLK_LOCAL_MEM_FENCE);  // wait until all work-items loaded their part of the data

                // calculate intermediate results and store them in shared memory
                for (uint class_idx = 0; class_idx < FEATURE_BLOCK_SIZE; ++class_idx) {
                    for (uint internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                        for (uint internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                            C_out_cache[(class_idx + local_id_1) % FEATURE_BLOCK_SIZE][internal_i * THREAD_BLOCK_SIZE + local_id_0] +=
                                temp[internal_i][internal_j] * B_cache[(class_idx + local_id_1) % FEATURE_BLOCK_SIZE][local_id_1 * INTERNAL_BLOCK_SIZE + internal_j];
                        }
                    }
                    barrier(CLK_LOCAL_MEM_FENCE);  // wait until all work-items performed their part of the calculations
                }

                // add intermediate cached results to C
                for (uint internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                    const ulong global_i = row_offset + i + (ulong) internal;
                    atomicAdd(&C[global_i * (num_classes + PADDING_SIZE_ul) + dim + threadIdx_y], C_out_cache[local_id_1][internal * THREAD_BLOCK_SIZE + local_id_0]);
                    atomicAdd(&C[global_i * (num_classes + PADDING_SIZE_ul) + dim + threadIdx_y + THREAD_BLOCK_SIZE_ul], C_out_cache[local_id_1 + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + local_id_0]);
                }
                barrier(CLK_LOCAL_MEM_FENCE);   // wait until all threads updated C with their values
            }
        }
    }
}
