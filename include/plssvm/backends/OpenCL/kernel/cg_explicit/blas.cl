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
 * @param[in] device_specific_num_rows the number of rows in @p A and number of rows in @p B; thr rows in @p A are potentially distributed across multiple devices
 * @param[in] row_offset the first row this device is responsible for
 * @param[in] alpha the scalar alpha value
 * @param[in] A the matrix @p A
 * @param[in] B the matrix @p B
 * @param[in] beta the scalar beta value
 * @param[in,out] C the matrix @p C, also used as result matrix
 */
__kernel void device_kernel_symm(const ulong num_rows, const ulong num_rhs, const ulong device_specific_num_rows, const ulong row_offset, const real_type alpha, const __global real_type *A, const __global real_type *B, const real_type beta, __global real_type *C, const ulong grid_x_offset, const ulong grid_y_offset) {
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

    // calculate the indices used in the current work-item
    const ulong i = (blockIdx_x * blockDim_x + threadIdx_x) * INTERNAL_BLOCK_SIZE_ul;  // #rhs
    const ulong i_linear = blockIdx_x * blockDim_x * INTERNAL_BLOCK_SIZE_ul + threadIdx_x;
    const ulong j = (blockIdx_y * blockDim_y + threadIdx_y) * INTERNAL_BLOCK_SIZE_ul;  // # row
    const ulong j_linear = blockIdx_y * blockDim_y * INTERNAL_BLOCK_SIZE_ul + threadIdx_x;

    // create the local memory arrays used for caching data point features
    __local real_type A_cache[FEATURE_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];
    __local real_type B_cache[FEATURE_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];

    // create a thread private array used for internal caching
    real_type temp[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE] = { (real_type) 0.0 };

    // iterate over all features using blocking to be able to cache them for faster memory accesses
    for (ulong dim = 0; dim < (num_rows - row_offset); dim += FEATURE_BLOCK_SIZE_ul) {
        // load data into local memory
        for (uint internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
            const ulong global_i = i_linear + (ulong) internal * THREAD_BLOCK_SIZE_ul;
            const ulong global_j = j_linear + (ulong) internal * THREAD_BLOCK_SIZE_ul;

            // determine on which side of the diagonal we are located
            if (dim + get_local_id(1) < global_j) {
                A_cache[local_id_1][internal * THREAD_BLOCK_SIZE + local_id_0] = A[(dim + threadIdx_y) * (num_rows - row_offset + PADDING_SIZE_ul) + global_j - (dim + threadIdx_y) * (dim + threadIdx_y + (ulong) 1) / (ulong) 2];
            } else {
                A_cache[local_id_1][internal * THREAD_BLOCK_SIZE + local_id_0] = A[global_j * (num_rows - row_offset + PADDING_SIZE_ul) + dim + threadIdx_y - global_j * (global_j + (ulong) 1) / (ulong) 2];
            }
            // determine on which side of the diagonal we are located
            if (dim + get_local_id(1) + THREAD_BLOCK_SIZE < global_j) {
                A_cache[local_id_1 + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + local_id_0] = A[(dim + threadIdx_y + THREAD_BLOCK_SIZE_ul) * (num_rows - row_offset + PADDING_SIZE_ul) + global_j - (dim + threadIdx_y + THREAD_BLOCK_SIZE_ul) * (dim + threadIdx_y + THREAD_BLOCK_SIZE_ul + (ulong) 1) / (ulong) 2];
            } else {
                A_cache[local_id_1 + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + local_id_0] = A[global_j * (num_rows - row_offset + PADDING_SIZE_ul) + dim + threadIdx_y + THREAD_BLOCK_SIZE_ul - global_j * (global_j + (ulong) 1) / (ulong) 2];
            }

            B_cache[local_id_1][internal * THREAD_BLOCK_SIZE + local_id_0] = B[(dim + row_offset + threadIdx_y) * (num_rhs + PADDING_SIZE_ul) + global_i];
            B_cache[local_id_1 + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + local_id_0] = B[(dim + row_offset + threadIdx_y + THREAD_BLOCK_SIZE_ul) * (num_rhs + PADDING_SIZE_ul) + global_i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);  // wait until all work-items loaded their part of the data

        // perform the dot product calculation
        for (uint block_dim = 0; block_dim < FEATURE_BLOCK_SIZE; ++block_dim) {
            for (uint internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                for (uint internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                    temp[internal_i][internal_j] += A_cache[block_dim][local_id_1 * INTERNAL_BLOCK_SIZE + internal_j] * B_cache[block_dim][local_id_0 * INTERNAL_BLOCK_SIZE + internal_i];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);  // wait until all work-items performed their part of the calculations
    }

    // apply the (partial) BLAS operation and update C
    for (uint internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
        for (uint internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
            const ulong global_i = i + (ulong) internal_i;
            const ulong device_global_j = j + (ulong) internal_j;
            const ulong global_j = row_offset + j + (ulong) internal_j;

            // be sure to not perform out of bounds accesses
            if (global_i < num_rhs && device_global_j < device_specific_num_rows) {
                C[global_j * (num_rhs + PADDING_SIZE_ul) + global_i] = alpha * temp[internal_i][internal_j] + beta * C[global_j * (num_rhs + PADDING_SIZE_ul) + global_i];
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
 * @param[in] device_specific_num_rows the number of rows in @p A and number of rows in @p B; thr rows in @p A are potentially distributed across multiple devices
 * @param[in] row_offset the first row this device is responsible for
 * @param[in] alpha the scalar alpha value
 * @param[in] A the matrix @p A
 * @param[in] B the matrix @p B
 * @param[in] beta the scalar beta value
 * @param[in,out] C the matrix @p C, also used as result matrix
 */
__kernel void device_kernel_symm_mirror(const ulong num_rows, const ulong num_rhs, const ulong num_mirror_rows, const ulong device_specific_num_rows, const ulong row_offset, const real_type alpha, const __global real_type *A, const __global real_type *B, const real_type beta, __global real_type *C, const ulong grid_x_offset, const ulong grid_y_offset) {
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

    // calculate the indices used in the current work-item
    const ulong i = (blockIdx_x * blockDim_x + threadIdx_x) * INTERNAL_BLOCK_SIZE_ul;  // #rhs
    const ulong i_linear = blockIdx_x * blockDim_x * INTERNAL_BLOCK_SIZE_ul + threadIdx_x;
    const ulong j = (blockIdx_y * blockDim_y + threadIdx_y) * INTERNAL_BLOCK_SIZE_ul;  // # row
    const ulong j_linear = blockIdx_y * blockDim_y * INTERNAL_BLOCK_SIZE_ul + threadIdx_x;

    // create the local memory arrays used for caching data point features
    __local real_type A_cache[FEATURE_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];
    __local real_type B_cache[FEATURE_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];

    // create a thread private array used for internal caching
    real_type temp[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE] = { (real_type) 0.0 };

    // iterate over all features using blocking to be able to cache them for faster memory accesses
    for (ulong dim = 0; dim < device_specific_num_rows; dim += FEATURE_BLOCK_SIZE_ul) {
        // load data into local memory
        for (uint internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
            const ulong global_i = i_linear + (ulong) internal * THREAD_BLOCK_SIZE_ul;
            const ulong global_j = j_linear + (ulong) internal * THREAD_BLOCK_SIZE_ul;

            // FEATURE_BLOCK_SIZE = 2 * THREAD_BLOCK_SIZE -> store twice as many values in the shared memory
            A_cache[local_id_1][internal * THREAD_BLOCK_SIZE + local_id_0] = A[(dim + threadIdx_y) * (num_rows - row_offset + PADDING_SIZE_ul) - (dim + threadIdx_y - (ulong) 1) * (dim + threadIdx_y) / (ulong) 2 + device_specific_num_rows - (dim + get_local_id(1)) + global_j];
            A_cache[local_id_1 + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + local_id_0] = A[(dim + threadIdx_y + THREAD_BLOCK_SIZE_ul) * (num_rows - row_offset + PADDING_SIZE_ul) - (dim + threadIdx_y + THREAD_BLOCK_SIZE_ul - (ulong) 1) * (dim + get_local_id(1) + THREAD_BLOCK_SIZE_ul) / (ulong) 2 + device_specific_num_rows - (dim + get_local_id(1) + THREAD_BLOCK_SIZE_ul) + global_j];
            B_cache[local_id_1][internal * THREAD_BLOCK_SIZE + local_id_0] = B[(dim + row_offset + threadIdx_y) * (num_rhs + PADDING_SIZE_ul) + global_i];
            B_cache[local_id_1 + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + local_id_0] = B[(dim + row_offset + threadIdx_y + THREAD_BLOCK_SIZE_ul) * (num_rhs + PADDING_SIZE_ul) + global_i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);  // wait until all work-items loaded their part of the data

        // perform the feature reduction calculation
        for (uint block_dim = 0; block_dim < FEATURE_BLOCK_SIZE; ++block_dim) {
            for (uint internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                for (uint internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                    temp[internal_i][internal_j] += A_cache[block_dim][local_id_1 * INTERNAL_BLOCK_SIZE + internal_j] * B_cache[block_dim][local_id_0 * INTERNAL_BLOCK_SIZE + internal_i];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);  // wait until all work-items performed their part of the calculations
    }

    // apply the (remaining) BLAS operation and update C
    for (uint internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
        for (uint internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
            const ulong global_i = i + (ulong) internal_i;
            const ulong partial_global_j = j + (ulong) internal_j;
            const ulong global_j = row_offset + device_specific_num_rows + j + (ulong) internal_j;

            // be sure to not perform out of bounds accesses
            if (global_i < num_rhs && partial_global_j < num_mirror_rows) {
                C[global_j * (num_rhs + PADDING_SIZE_ul) + global_i] = alpha * temp[internal_i][internal_j] + beta * C[global_j * (num_rhs + PADDING_SIZE_ul) + global_i];
            }
        }
    }
}

/**
 * @brief Perform a simple inplace matrix addition: lhs += rhs.
 * @param[in] num_cols the number of columns in both matrices
 * @param[in,out] lhs the first matrix (updated inplace)
 * @param[in] rhs the second matrix
 */
__kernel void device_kernel_inplace_matrix_add(const ulong num_cols, real_type __global *lhs, const real_type __global *rhs, const ulong grid_x_offset, const ulong grid_y_offset) {
    // cast all values to 64-bit unsigned long long to prevent potential 32-bit overflows
    const ulong threadIdx_x = get_local_id(0);                 // current thread in block x-dimension
    const ulong threadIdx_y = get_local_id(1);                 // current thread in block y-dimension
    const ulong blockDim_x = get_local_size(0);                // number of threads in block x-dimension
    const ulong blockDim_y = get_local_size(1);                // number of threads in block y-dimension
    const ulong blockIdx_x = get_group_id(0) + grid_x_offset;  // current block in grid x-dimension + offsets if the grid size would be too large
    const ulong blockIdx_y = get_group_id(1) + grid_y_offset;  // current block in grid y-dimension + offsets if the grid size would be too large

    // calculate the indices used in the current thread
    const ulong i = (blockIdx_x * blockDim_x + threadIdx_x) * INTERNAL_BLOCK_SIZE_ul;  // # num_rows
    const ulong j = (blockIdx_y * blockDim_y + threadIdx_y) * INTERNAL_BLOCK_SIZE_ul;  // # num_rhs

    for (uint internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
        for (uint internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
            const ulong global_i = i + (ulong) internal_i;
            const ulong global_j = j + (ulong) internal_j;

            lhs[global_i * (num_cols + PADDING_SIZE_ul) + global_j] += rhs[global_i * (num_cols + PADDING_SIZE_ul) + global_j];
        }
    }
}

/**
 * @brief Perform a simple inplace matrix scale: lhs *= scalar.
 * @param[in] num_cols the number of columns in the matrix
 * @param[in,out] lhs the matrix (updated inplace)
 * @param[in] scale the value to scale
 */
__kernel void device_kernel_inplace_matrix_scale(const ulong num_cols, real_type __global *lhs, const real_type scale, const ulong grid_x_offset, const ulong grid_y_offset) {
    // cast all values to 64-bit unsigned long long to prevent potential 32-bit overflows
    const ulong threadIdx_x = get_local_id(0);                 // current thread in block x-dimension
    const ulong threadIdx_y = get_local_id(1);                 // current thread in block y-dimension
    const ulong blockDim_x = get_local_size(0);                // number of threads in block x-dimension
    const ulong blockDim_y = get_local_size(1);                // number of threads in block y-dimension
    const ulong blockIdx_x = get_group_id(0) + grid_x_offset;  // current block in grid x-dimension + offsets if the grid size would be too large
    const ulong blockIdx_y = get_group_id(1) + grid_y_offset;  // current block in grid y-dimension + offsets if the grid size would be too large

    // calculate the indices used in the current thread
    const ulong i = (blockIdx_x * blockDim_x + threadIdx_x) * INTERNAL_BLOCK_SIZE_ul;  // # num_rows
    const ulong j = (blockIdx_y * blockDim_y + threadIdx_y) * INTERNAL_BLOCK_SIZE_ul;  // # num_rhs

    for (uint internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
        for (uint internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
            const ulong global_i = i + (ulong) internal_i;
            const ulong global_j = j + (ulong) internal_j;

            lhs[global_i * (num_cols + PADDING_SIZE_ul) + global_j] *= scale;
        }
    }
}
