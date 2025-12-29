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
 * @param[in] data the data points to calculate the implicit kernel matrix from
 * @param[in] num_rows the total number of data points (= total number of rows)
 * @param[in] device_num_rows the number of rows the current device is responsible for
 * @param[in] device_row_offset the first row in @p data the current device is responsible for
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
__kernel void device_kernel_assembly_symm(const real_type alpha, const __global real_type *q, const __global real_type *data, const ulong num_rows, const ulong device_num_rows, const ulong device_row_offset, const ulong num_features, const real_type QA_cost, const real_type cost, const __global real_type *B, __global real_type *C, const ulong num_classes, const ulong grid_x_offset, const ulong grid_y_offset PLSSVM_OPENCL_KERNEL_FUNCTION_PARAMETER_LIST) {
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

    // calculate the indices used in the current work-item
    const ulong i_idx = (blockIdx_x * blockDim_x + threadIdx_x) * INTERNAL_BLOCK_SIZE_uz;  // num_rows - device_row_offset
    const ulong j_idx = (blockIdx_y * blockDim_y + threadIdx_y) * INTERNAL_BLOCK_SIZE_uz;  // device_num_rows

    // calculate the indices used in the current work-item, pays attention to coalesced memory accesses
    const ulong i_idx_linear = blockIdx_x * blockDim_x * INTERNAL_BLOCK_SIZE_uz + threadIdx_x;  // num_rows - device_row_offset
    const ulong j_idx_linear = blockIdx_y * blockDim_y * INTERNAL_BLOCK_SIZE_uz + threadIdx_x;  // device_num_rows

    // create two local memory arrays used for caching
    __local real_type cache_one[THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];
    __local real_type cache_two[THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];

    // only calculate the upper triangular matrix -> can't use threadIdx since all work-items in a warp must progress further
    if (blockIdx_x >= blockIdx_y) {
        // create a thread private array used for internal caching
        real_type temp[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE] = { (real_type) 0.0 };

        //*************************************************************************//
        //                   inplace kernel matrix construction                    //
        //*************************************************************************//
        {
            // reinterpret the local memory arrays to be of shape [THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE]
            __local real_type(*data_i_cache)[INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE] = (__local real_type(*)[INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE]) cache_one;
            __local real_type(*data_j_cache)[INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE] = (__local real_type(*)[INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE]) cache_two;

            // iterate over all features using blocking to be able to cache them for faster memory accesses
            for (ulong feature_block = 0; feature_block < num_features; feature_block += THREAD_BLOCK_SIZE_uz) {
                // zero-out shared memory
                for (uint internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                    data_i_cache[local_id_1][internal * THREAD_BLOCK_SIZE + local_id_0] = (real_type) 0.0;
                    data_j_cache[local_id_1][internal * THREAD_BLOCK_SIZE + local_id_0] = (real_type) 0.0;
                }

                // load data into local memory
                for (uint internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                    // calculate the indices to access the global data, pays attention to coalesced memory accesses
                    const ulong global_i_idx_linear = device_row_offset + i_idx_linear + (ulong) internal * THREAD_BLOCK_SIZE_uz;
                    const ulong global_j_idx_linear = device_row_offset + j_idx_linear + (ulong) internal * THREAD_BLOCK_SIZE_uz;

                    // store the values in the local memory
                    if (feature_block + threadIdx_y < num_features) {
                        if (global_i_idx_linear < num_rows) {
                            data_i_cache[local_id_1][internal * THREAD_BLOCK_SIZE + local_id_0] = data[(feature_block + threadIdx_y) * (num_rows + (ulong) 1) + global_i_idx_linear];  // SoA
                        }
                        if (global_j_idx_linear < num_rows) {
                            data_j_cache[local_id_1][internal * THREAD_BLOCK_SIZE + local_id_0] = data[(feature_block + threadIdx_y) * (num_rows + (ulong) 1) + global_j_idx_linear];  // SoA
                        }
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);  // wait until all work-items loaded their part of the data

                // perform the feature reduction calculation
                for (uint feature = 0; feature < THREAD_BLOCK_SIZE; ++feature) {
                    for (uint internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                        for (uint internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                            temp[internal_i][internal_j] += PLSSVM_OPENCL_FEATURE_REDUCE_FUNCTION(data_i_cache[feature][local_id_0 * INTERNAL_BLOCK_SIZE + internal_i],
                                                                                                  data_j_cache[feature][local_id_1 * INTERNAL_BLOCK_SIZE + internal_j]);
                        }
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);  // wait until all work-items performed their part of the calculations
            }
        }

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
                    // apply the final kernel function
                    temp[internal_i][internal_j] = PLSSVM_OPENCL_APPLY_KERNEL_FUNCTION(temp[internal_i][internal_j] PLSSVM_OPENCL_KERNEL_FUNCTION_PARAMETER) + QA_cost - q[global_i_idx] - q[global_j_idx];
                    // apply the cost on the diagonal
                    if (global_i_idx == global_j_idx) {
                        temp[internal_i][internal_j] += cost;
                    }
                } else {
                    // be sure to set the value to zero otherwise
                    temp[internal_i][internal_j] = (real_type) 0.0;
                }
            }
        }

        //*************************************************************************//
        //     calculate C += alpha * temp * B for the UPPER triangular matrix     //
        //*************************************************************************//
        {
            // reinterpret the local memory arrays to be of shape [INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE][THREAD_BLOCK_SIZE]
            __local real_type(*B_cache)[THREAD_BLOCK_SIZE] = (__local real_type(*)[THREAD_BLOCK_SIZE]) cache_one;
            __local real_type(*C_out_cache)[THREAD_BLOCK_SIZE] = (__local real_type(*)[THREAD_BLOCK_SIZE]) cache_two;

            // iterate over all classes using blocking to be able to cache them for faster memory accesses
            for (ulong class_block = 0; class_block < num_classes; class_block += THREAD_BLOCK_SIZE_uz) {
                // zero-out shared memory
                for (uint internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                    B_cache[internal * THREAD_BLOCK_SIZE + local_id_0][local_id_1] = (real_type) 0.0;
                    C_out_cache[internal * THREAD_BLOCK_SIZE + local_id_0][local_id_1] = (real_type) 0.0;
                }

                // load data into local memory
                for (uint internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                    // calculate the indices to access the global data, pays attention to coalesced memory accesses
                    const ulong global_i_idx_linear = device_row_offset + i_idx_linear + (ulong) internal * THREAD_BLOCK_SIZE_uz;

                    // store the values in the local memory
                    if (class_block + threadIdx_y < num_classes && global_i_idx_linear < num_rows) {
                        B_cache[internal * THREAD_BLOCK_SIZE + local_id_0][local_id_1] = alpha * B[global_i_idx_linear * num_classes + class_block + threadIdx_y];  // SoA
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);  // wait until all work-items loaded their part of the data

                // calculate intermediate results and store them in local memory
                for (uint class_idx = 0; class_idx < THREAD_BLOCK_SIZE; ++class_idx) {
                    for (uint internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                        for (uint internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                            C_out_cache[local_id_1 * INTERNAL_BLOCK_SIZE + internal_j][(class_idx + local_id_0) % THREAD_BLOCK_SIZE] +=
                                temp[internal_i][internal_j] * B_cache[local_id_0 * INTERNAL_BLOCK_SIZE + internal_i][(class_idx + local_id_0) % THREAD_BLOCK_SIZE];
                        }
                    }
                    barrier(CLK_LOCAL_MEM_FENCE);  // wait until all work-items performed their part of the calculations
                }

                // atomically add the intermediate cached results to the C matrix
                for (uint internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                    // calculate the indices to access the global data
                    const ulong global_j_idx = device_row_offset + j_idx + (ulong) internal;

                    if (class_block + threadIdx_x < num_classes && global_j_idx < num_rows) {
                        atomicAdd(&C[global_j_idx * num_classes + class_block + threadIdx_x], C_out_cache[local_id_1 * INTERNAL_BLOCK_SIZE + internal][local_id_0]);  // SoA
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);  // wai until all work-items updated C with their values
            }
        }

        // set potential diagonal entries in temp to 0.0 such that we don't apply the main diagonal twice to C
        for (uint internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
            for (uint internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                // calculate the indices to access the global data
                const ulong global_i_idx = device_row_offset + i_idx + (ulong) internal_i;
                const ulong global_j_idx = device_row_offset + j_idx + (ulong) internal_j;

                if (global_i_idx == global_j_idx) {
                    temp[internal_i][internal_j] = (real_type) 0.0;
                }
            }
        }

        //*************************************************************************//
        //     calculate C += alpha * temp * B for the LOWER triangular matrix     //
        //*************************************************************************//
        {
            // reinterpret the local memory arrays to be of shape [THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE]
            __local real_type(*B_cache)[INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE] = (__local real_type(*)[INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE]) cache_one;
            __local real_type(*C_out_cache)[INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE] = (__local real_type(*)[INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE]) cache_two;

            // iterate over all classes using blocking to be able to cache them for faster memory accesses
            for (ulong class_block = 0; class_block < num_classes; class_block += THREAD_BLOCK_SIZE_uz) {
                // zero-out shared memory
                for (uint internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                    B_cache[local_id_1][internal * THREAD_BLOCK_SIZE + local_id_0] = (real_type) 0.0;
                    C_out_cache[local_id_1][internal * THREAD_BLOCK_SIZE + local_id_0] = (real_type) 0.0;
                }

                // load data into local memory
                for (uint internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                    // calculate the indices to access the global data, pays attention to coalesced memory accesses
                    const ulong global_j_idx_linear = device_row_offset + j_idx_linear + (ulong) internal * THREAD_BLOCK_SIZE_uz;

                    // store the values in the local memory
                    if (class_block + threadIdx_y < num_classes && global_j_idx_linear < num_rows) {
                        B_cache[local_id_1][internal * THREAD_BLOCK_SIZE + local_id_0] = alpha * B[global_j_idx_linear * num_classes + class_block + threadIdx_y];  // SoA
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);  // wait until all work-items loaded their part of the data

                // calculate intermediate results and store them in local memory
                for (uint class_idx = 0; class_idx < THREAD_BLOCK_SIZE; ++class_idx) {
                    for (uint internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                        for (uint internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                            C_out_cache[(class_idx + local_id_1) % THREAD_BLOCK_SIZE][internal_i * THREAD_BLOCK_SIZE + local_id_0] +=
                                temp[internal_i][internal_j] * B_cache[(class_idx + local_id_1) % THREAD_BLOCK_SIZE][local_id_1 * INTERNAL_BLOCK_SIZE + internal_j];
                        }
                    }
                    barrier(CLK_LOCAL_MEM_FENCE);  // wait until all work-items performed their part of the calculations
                }

                // atomically add the intermediate cached results to the C matrix
                for (uint internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                    // calculate the indices to access the global data
                    const ulong global_i_idx = device_row_offset + i_idx + (ulong) internal;

                    if (class_block + threadIdx_y < num_classes && global_i_idx < num_rows) {
                        atomicAdd(&C[global_i_idx * num_classes + class_block + threadIdx_y], C_out_cache[local_id_1][internal * THREAD_BLOCK_SIZE + local_id_0]);  // SoA
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);  // wait until all work-items updated C with their values
            }
        }
    }
}
