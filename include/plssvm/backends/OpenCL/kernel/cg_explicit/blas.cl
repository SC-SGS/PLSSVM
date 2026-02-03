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
 * @brief Perform an explicit BLAS SYMM operation: `C = alpha * A * B + beta * C` where @p A is a `m x k` symmetric matrix (memory optimized), @p B is a `k x n` matrix, @p C is a `m x n` matrix, and @p alpha and @p beta are scalars.
 * @details In a multi-GPU setting, this function is only responsible for the rows this device is responsible for!
 * @param[in] num_rows the number of rows in @p A and @p C
 * @param[in] num_rhs the number of columns in @p B and @p C
 * @param[in] device_num_rows the number of rows in @p A and number of rows in @p B; thr rows in @p A are potentially distributed across multiple devices
 * @param[in] device_row_offset the first row this device is responsible for
 * @param[in] alpha the scalar alpha value
 * @param[in] A the matrix @p A
 * @param[in] B the matrix @p B
 * @param[in] beta the scalar beta value
 * @param[in,out] C the matrix @p C, also used as result matrix
 * @param[in] grid_x_offset the offset in x-dimension into the data points if more than one execution grid has to be used
 * @param[in] grid_y_offset the offset in y-dimension into the data points if more than one execution grid has to be used
 */
__kernel void device_kernel_symm(const ulong num_rows, const ulong num_rhs, const ulong device_num_rows, const ulong device_row_offset, const real_type alpha, const __global real_type *A, const __global real_type *B, const real_type beta, __global real_type *C, const ulong grid_x_offset, const ulong grid_y_offset) {
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
    __local real_type A_cache[THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];
    __local real_type B_cache[THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];

    // create a work-item private array used for internal caching
    real_type temp[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE] = { (real_type) 0.0 };

    {
        // calculate the indices used in the current work-item, pays attention to coalesced memory accesses
        const ulong i_idx_linear = blockIdx_x * blockDim_x * INTERNAL_BLOCK_SIZE_uz + threadIdx_x;  // num_rhs
        const ulong j_idx_linear = blockIdx_y * blockDim_y * INTERNAL_BLOCK_SIZE_uz + threadIdx_x;  // device_num_rows

        // iterate over all values using blocking to be able to cache them for faster memory accesses
        for (ulong dim_block = 0; dim_block < (num_rows - device_row_offset); dim_block += THREAD_BLOCK_SIZE_uz) {
            // load data into local memory
            for (uint internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                // calculate the indices to access the global data, pays attention to coalesced memory accesses
                const ulong global_i_idx_linear = i_idx_linear + (ulong) internal * THREAD_BLOCK_SIZE_uz;
                const ulong global_j_idx_linear = j_idx_linear + (ulong) internal * THREAD_BLOCK_SIZE_uz;

                // store the values in the local memory
                // determine on which side of the diagonal we are located
                if (dim_block + get_local_id(1) < global_j_idx_linear) {
                    A_cache[local_id_1][internal * THREAD_BLOCK_SIZE + local_id_0] = A[(dim_block + threadIdx_y) * (num_rows - device_row_offset + PADDING_SIZE_uz) + global_j_idx_linear - (dim_block + threadIdx_y) * (dim_block + threadIdx_y + (ulong) 1) / (ulong) 2];  // SoA, upper triangular matrix only
                } else {
                    A_cache[local_id_1][internal * THREAD_BLOCK_SIZE + local_id_0] = A[global_j_idx_linear * (num_rows - device_row_offset + PADDING_SIZE_uz) + dim_block + threadIdx_y - global_j_idx_linear * (global_j_idx_linear + (ulong) 1) / (ulong) 2];  // SoA, upper triangular matrix only
                }
                B_cache[local_id_1][internal * THREAD_BLOCK_SIZE + local_id_0] = B[(dim_block + device_row_offset + threadIdx_y) * (num_rhs + PADDING_SIZE_uz) + global_i_idx_linear];  // SoA
            }
            barrier(CLK_LOCAL_MEM_FENCE);  // wait until all work-items loaded their part of the data

#if defined(PLSSVM_OPENCL_TARGET_CPUS)
            // perform the dot product calculation, the dim is the fastest moving index
            for (uint internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                for (uint internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                    real_type sum = 0.0;
                    for (uint dim = 0; dim < THREAD_BLOCK_SIZE; ++dim) {
                        sum += A_cache[dim][local_id_1 * INTERNAL_BLOCK_SIZE + internal_j] * B_cache[dim][local_id_0 * INTERNAL_BLOCK_SIZE + internal_i];
                    }
                    temp[internal_i][internal_j] += sum;
                }
            }
#else
            // perform the dot product calculation, the dim is the slowest moving index
            for (uint dim = 0; dim < THREAD_BLOCK_SIZE; ++dim) {
                for (uint internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                    for (uint internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                        temp[internal_i][internal_j] += A_cache[dim][local_id_1 * INTERNAL_BLOCK_SIZE + internal_j] * B_cache[dim][local_id_0 * INTERNAL_BLOCK_SIZE + internal_i];
                    }
                }
            }
#endif
            barrier(CLK_LOCAL_MEM_FENCE);  // wait until all work-items performed their part of the calculations
        }
    }

    // calculate the indices used in the current work-item
    const ulong i_idx = (blockIdx_x * blockDim_x + threadIdx_x) * INTERNAL_BLOCK_SIZE_uz;  // num_rhs
    const ulong j_idx = (blockIdx_y * blockDim_y + threadIdx_y) * INTERNAL_BLOCK_SIZE_uz;  // device_num_rows

    // apply the (partial) BLAS operation and update C
    for (uint internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
        for (uint internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
            // calculate the indices to access the global data and the data with respect to the current device
            const ulong global_i_idx = i_idx + (ulong) internal_i;
            const ulong device_global_j_idx = j_idx + (ulong) internal_j;
            const ulong global_j_idx = device_row_offset + device_global_j_idx;

            // be sure to not perform out-of-bounds accesses
            if (global_i_idx < num_rhs && device_global_j_idx < device_num_rows) {
                C[global_j_idx * (num_rhs + PADDING_SIZE_uz) + global_i_idx] = alpha * temp[internal_i][internal_j] + beta * C[global_j_idx * (num_rhs + PADDING_SIZE_uz) + global_i_idx];  // SoA
            }
        }
    }
}

/**
 * @brief Perform an explicit BLAS SYMM operation: `C = alpha * A * B + beta * C` where @p A is a `m x k` symmetric matrix (memory optimized), @p B is a `k x n` matrix, @p C is a `m x n` matrix, and @p alpha and @p beta are scalars.
 * @details In a multi-GPU setting, this function is responsible for mirroring down the columns this device is responsible for!
 * @param[in] num_rows the number of rows in @p A and @p C
 * @param[in] num_rhs the number of columns in @p B and @p C
 * @param[in] num_mirror_rows the number of rows to mirror down
 * @param[in] device_num_rows the number of rows in @p A and number of rows in @p B; thr rows in @p A are potentially distributed across multiple devices
 * @param[in] device_row_offset the first row this device is responsible for
 * @param[in] alpha the scalar alpha value
 * @param[in] A the matrix @p A
 * @param[in] B the matrix @p B
 * @param[in] beta the scalar beta value
 * @param[in,out] C the matrix @p C, also used as result matrix
 * @param[in] grid_x_offset the offset in x-dimension into the data points if more than one execution grid has to be used
 * @param[in] grid_y_offset the offset in y-dimension into the data points if more than one execution grid has to be used
 */
__kernel void device_kernel_symm_mirror(const ulong num_rows, const ulong num_rhs, const ulong num_mirror_rows, const ulong device_num_rows, const ulong device_row_offset, const real_type alpha, const __global real_type *A, const __global real_type *B, const real_type beta, __global real_type *C, const ulong grid_x_offset, const ulong grid_y_offset) {
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
    __local real_type A_cache[THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];
    __local real_type B_cache[THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];

    // create a work-item private array used for internal caching
    real_type temp[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE] = { (real_type) 0.0 };

    {
        // calculate the indices used in the current work-item, pays attention to coalesced memory accesses
        const ulong i_idx_linear = blockIdx_x * blockDim_x * INTERNAL_BLOCK_SIZE_uz + threadIdx_x;  // num_rhs
        const ulong j_idx_linear = blockIdx_y * blockDim_y * INTERNAL_BLOCK_SIZE_uz + threadIdx_x;  // num_mirror_rows

        // iterate over the remaining values using blocking to be able to cache them for faster memory accesses
        for (ulong dim_block = 0; dim_block < device_num_rows; dim_block += THREAD_BLOCK_SIZE_uz) {
            // load data into local memory
            for (uint internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                // calculate the indices to access the global data, pays attention to coalesced memory accesses
                const ulong global_i_idx_linear = i_idx_linear + (ulong) internal * THREAD_BLOCK_SIZE_uz;
                const ulong global_j_idx_linear = j_idx_linear + (ulong) internal * THREAD_BLOCK_SIZE_uz;

                // store the values in the local memory
                A_cache[local_id_1][internal * THREAD_BLOCK_SIZE + local_id_0] = A[(dim_block + threadIdx_y) * (num_rows - device_row_offset + PADDING_SIZE_uz) - (dim_block + threadIdx_y - (ulong) 1) * (dim_block + threadIdx_y) / (ulong) 2 + device_num_rows - (dim_block + threadIdx_y) + global_j_idx_linear];  // SoA, upper triangular matrix only
                B_cache[local_id_1][internal * THREAD_BLOCK_SIZE + local_id_0] = B[(device_row_offset + dim_block + threadIdx_y) * (num_rhs + PADDING_SIZE_uz) + global_i_idx_linear];                                                                                                                                 // SoA
            }
            barrier(CLK_LOCAL_MEM_FENCE);  // wait until all work-items loaded their part of the data

#if defined(PLSSVM_OPENCL_TARGET_CPUS)
            // perform the dot product calculation, the dim is the fastest moving index
            for (uint internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                for (uint internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                    real_type sum = 0.0;
                    for (uint dim = 0; dim < THREAD_BLOCK_SIZE; ++dim) {
                        sum += A_cache[dim][local_id_1 * INTERNAL_BLOCK_SIZE + internal_j] * B_cache[dim][local_id_0 * INTERNAL_BLOCK_SIZE + internal_i];
                    }
                    temp[internal_i][internal_j] += sum;
                }
            }
#else
            // perform the dot product calculation, the dim is the slowest moving index
            for (uint dim = 0; dim < THREAD_BLOCK_SIZE; ++dim) {
                for (uint internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                    for (uint internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                        temp[internal_i][internal_j] += A_cache[dim][local_id_1 * INTERNAL_BLOCK_SIZE + internal_j] * B_cache[dim][local_id_0 * INTERNAL_BLOCK_SIZE + internal_i];
                    }
                }
            }
#endif
            barrier(CLK_LOCAL_MEM_FENCE);  // wait until all work-items performed their part of the calculations
        }
    }

    // calculate the indices used in the current work-item
    const ulong i_idx = (blockIdx_x * blockDim_x + threadIdx_x) * INTERNAL_BLOCK_SIZE_uz;  // num_rhs
    const ulong j_idx = (blockIdx_y * blockDim_y + threadIdx_y) * INTERNAL_BLOCK_SIZE_uz;  // num_mirror_rows

    // apply the (remaining) BLAS operation and update C
    for (uint internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
        for (uint internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
            // calculate the indices to access the global data and the data with respect to the current device
            const ulong global_i_idx = i_idx + (ulong) internal_i;
            const ulong partial_global_j_idx = j_idx + (ulong) internal_j;
            const ulong global_j_idx = device_row_offset + device_num_rows + partial_global_j_idx;

            // be sure to not perform out-of-bounds accesses
            if (global_i_idx < num_rhs && partial_global_j_idx < num_mirror_rows) {
                C[global_j_idx * (num_rhs + PADDING_SIZE_uz) + global_i_idx] = alpha * temp[internal_i][internal_j] + beta * C[global_j_idx * (num_rhs + PADDING_SIZE_uz) + global_i_idx];  // SoA
            }
        }
    }
}

/**
 * @brief Perform a simple inplace matrix addition: lhs += rhs.
 * @param[in] num_cols the number of columns in both matrices
 * @param[in,out] lhs the first matrix (updated inplace)
 * @param[in] rhs the second matrix
 * @param[in] grid_x_offset the offset in x-dimension into the data points if more than one execution grid has to be used
 * @param[in] grid_y_offset the offset in y-dimension into the data points if more than one execution grid has to be used
 */
__kernel void device_kernel_inplace_matrix_add(const ulong num_cols, real_type __global *lhs, const real_type __global *rhs, const ulong grid_x_offset, const ulong grid_y_offset) {
    // cast all values to 64-bit unsigned long long to prevent potential 32-bit overflows
    const ulong threadIdx_x = get_local_id(0);                 // current work-item in work-group x-dimension
    const ulong threadIdx_y = get_local_id(1);                 // current work-item in work-group y-dimension
    const ulong blockDim_x = get_local_size(0);                // number of work-items in work-group x-dimension
    const ulong blockDim_y = get_local_size(1);                // number of work-items in work-group y-dimension
    const ulong blockIdx_x = get_group_id(0) + grid_x_offset;  // current work-group in global range x-dimension + offsets if the global range is too large
    const ulong blockIdx_y = get_group_id(1) + grid_y_offset;  // current work-group in global range y-dimension + offsets if the global range is too large

    // calculate the indices used in the current work-item
    const ulong i_idx = (blockIdx_x * blockDim_x + threadIdx_x) * INTERNAL_BLOCK_SIZE_uz;  // num_rows
    const ulong j_idx = (blockIdx_y * blockDim_y + threadIdx_y) * INTERNAL_BLOCK_SIZE_uz;  // num_rhs

    for (uint internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
        for (uint internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
            // calculate the indices to access the global data
            const ulong global_i_idx = i_idx + (ulong) internal_i;
            const ulong global_j_idx = j_idx + (ulong) internal_j;

            lhs[global_i_idx * (num_cols + PADDING_SIZE_uz) + global_j_idx] += rhs[global_i_idx * (num_cols + PADDING_SIZE_uz) + global_j_idx];  // SoA
        }
    }
}

/**
 * @brief Perform a simple inplace matrix scale: lhs *= scalar.
 * @param[in] num_cols the number of columns in the matrix
 * @param[in,out] lhs the matrix (updated inplace)
 * @param[in] scale the value to scale
 * @param[in] grid_x_offset the offset in x-dimension into the data points if more than one execution grid has to be used
 * @param[in] grid_y_offset the offset in y-dimension into the data points if more than one execution grid has to be used
 */
__kernel void device_kernel_inplace_matrix_scale(const ulong num_cols, real_type __global *lhs, const real_type scale, const ulong grid_x_offset, const ulong grid_y_offset) {
    // cast all values to 64-bit unsigned long long to prevent potential 32-bit overflows
    const ulong threadIdx_x = get_local_id(0);                 // current work-item in work-group x-dimension
    const ulong threadIdx_y = get_local_id(1);                 // current work-item in work-group y-dimension
    const ulong blockDim_x = get_local_size(0);                // number of work-items in work-group x-dimension
    const ulong blockDim_y = get_local_size(1);                // number of work-items in work-group y-dimension
    const ulong blockIdx_x = get_group_id(0) + grid_x_offset;  // current work-group in global range x-dimension + offsets if the global range is too large
    const ulong blockIdx_y = get_group_id(1) + grid_y_offset;  // current work-group in global range y-dimension + offsets if the global range is too large

    // calculate the indices used in the current work-item
    const ulong i_idx = (blockIdx_x * blockDim_x + threadIdx_x) * INTERNAL_BLOCK_SIZE_uz;  // num_rows
    const ulong j_idx = (blockIdx_y * blockDim_y + threadIdx_y) * INTERNAL_BLOCK_SIZE_uz;  // num_rhs

    for (uint internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
        for (uint internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
            // calculate the indices to access the global data
            const ulong global_i_idx = i_idx + (ulong) internal_i;
            const ulong global_j_idx = j_idx + (ulong) internal_j;

            lhs[global_i_idx * (num_cols + PADDING_SIZE_uz) + global_j_idx] *= scale;  // SoA
        }
    }
}
