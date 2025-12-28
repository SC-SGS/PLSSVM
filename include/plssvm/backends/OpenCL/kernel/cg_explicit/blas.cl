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
    // cast all values to 64-bit unsigned long long to prevent potential 32-bit overflows
    const ulong threadIdx_x = get_local_id(0);                 // current work-item in work-group x-dimension
    const ulong threadIdx_y = get_local_id(1);                 // current work-item in work-group y-dimension
    const ulong blockDim_x = get_local_size(0);                // number of work-items in work-group x-dimension
    const ulong blockDim_y = get_local_size(1);                // number of work-items in work-group y-dimension
    const ulong blockIdx_x = get_group_id(0) + grid_x_offset;  // current work-group in global range x-dimension + offsets if the global range is too large
    const ulong blockIdx_y = get_group_id(1) + grid_y_offset;  // current work-group in global range y-dimension + offsets if the global range is too large

    // calculate the indices used in the current work-item
    const ulong global_i_idx = blockIdx_x * blockDim_x + threadIdx_x;         // num_rhs
    const ulong device_global_j_idx = blockIdx_y * blockDim_y + threadIdx_y;  // device_num_rows
    const ulong global_j_idx = device_row_offset + device_global_j_idx;

    // be sure to not perform out-of-bounds accesses
    if (global_i_idx < num_rhs && device_global_j_idx < device_num_rows) {
        real_type temp = 0.0;

        // iterate over all values
        for (ulong dim = 0; dim < (num_rows - device_row_offset); ++dim) {
            real_type A_cache = 0.0;
            // determine on which side of the diagonal we are located
            if (dim < device_global_j_idx) {
                A_cache = A[dim * (num_rows - device_row_offset) + device_global_j_idx - dim * (dim + (ulong) 1) / (ulong) 2];  // SoA, upper triangular matrix only
            } else {
                A_cache = A[device_global_j_idx * (num_rows - device_row_offset) + dim - device_global_j_idx * (device_global_j_idx + (ulong) 1) / (ulong) 2];  // SoA, upper triangular matrix only
            }
            // perform the dot product calculation
            temp += A_cache * B[(device_row_offset + dim) * num_rhs + global_i_idx];  // SoA
        }

        // apply the (partial) BLAS operation and update C
        C[global_j_idx * num_rhs + global_i_idx] = alpha * temp + beta * C[global_j_idx * num_rhs + global_i_idx];  // SoA
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
    // cast all values to 64-bit unsigned long long to prevent potential 32-bit overflows
    const ulong threadIdx_x = get_local_id(0);                 // current work-item in work-group x-dimension
    const ulong threadIdx_y = get_local_id(1);                 // current work-item in work-group y-dimension
    const ulong blockDim_x = get_local_size(0);                // number of work-items in work-group x-dimension
    const ulong blockDim_y = get_local_size(1);                // number of work-items in work-group y-dimension
    const ulong blockIdx_x = get_group_id(0) + grid_x_offset;  // current work-group in global range x-dimension + offsets if the global range is too large
    const ulong blockIdx_y = get_group_id(1) + grid_y_offset;  // current work-group in global range y-dimension + offsets if the global range is too large

    // calculate the indices used in the current work-item
    const ulong global_i_idx = blockIdx_x * blockDim_x + threadIdx_x;          // num_rhs
    const ulong partial_global_j_idx = blockIdx_y * blockDim_y + threadIdx_y;  // num_mirror_rows
    const ulong global_j_idx = device_row_offset + device_num_rows + partial_global_j_idx;

    // be sure to not perform out-of-bounds accesses
    if (global_i_idx < num_rhs && partial_global_j_idx < num_mirror_rows && global_j_idx < num_rows) {
        real_type temp = 0.0;

        // iterate over all values
        for (ulong dim = 0; dim < device_num_rows; ++dim) {
            // perform the dot product calculation
            temp += A[dim * (num_rows - device_row_offset) - (dim - (ulong) 1) * dim / (ulong) 2 + device_num_rows - dim + partial_global_j_idx] *  // SoA, upper triangular matrix only
                    B[(device_row_offset + dim) * num_rhs + global_i_idx];                                                                          // SoA
        }

        // apply the (remaining) BLAS operation and update C
        C[global_j_idx * num_rhs + global_i_idx] = alpha * temp + beta * C[global_j_idx * num_rhs + global_i_idx];  // SoA
    }
}

/**
 * @brief Perform a simple inplace matrix addition: lhs += rhs.
 * @param[in] num_rows the number of rows in both matrices
 * @param[in] num_cols the number of columns in both matrices
 * @param[in,out] lhs the first matrix (updated inplace)
 * @param[in] rhs the second matrix
 * @param[in] grid_x_offset the offset in x-dimension into the data points if more than one execution grid has to be used
 * @param[in] grid_y_offset the offset in y-dimension into the data points if more than one execution grid has to be used
 */
__kernel void device_kernel_inplace_matrix_add(const ulong num_rows, const ulong num_cols, real_type __global *lhs, const real_type __global *rhs, const ulong grid_x_offset, const ulong grid_y_offset) {
    // cast all values to 64-bit unsigned long long to prevent potential 32-bit overflows
    const ulong threadIdx_x = get_local_id(0);                 // current work-item in work-group x-dimension
    const ulong threadIdx_y = get_local_id(1);                 // current work-item in work-group y-dimension
    const ulong blockDim_x = get_local_size(0);                // number of work-items in work-group x-dimension
    const ulong blockDim_y = get_local_size(1);                // number of work-items in work-group y-dimension
    const ulong blockIdx_x = get_group_id(0) + grid_x_offset;  // current work-group in global range x-dimension + offsets if the global range is too large
    const ulong blockIdx_y = get_group_id(1) + grid_y_offset;  // current work-group in global range y-dimension + offsets if the global range is too large

    // calculate the indices used in the current work-item
    const ulong global_i_idx = blockIdx_x * blockDim_x + threadIdx_x;  // num_rows
    const ulong global_j_idx = blockIdx_y * blockDim_y + threadIdx_y;  // num_rhs

    if (global_i_idx < num_rows && global_j_idx < num_cols) {
        lhs[global_i_idx * num_cols + global_j_idx] += rhs[global_i_idx * num_cols + global_j_idx];  // SoA
    }
}

/**
 * @brief Perform a simple inplace matrix scale: lhs *= scalar.
 * @param[in] num_rows the number of rows in the matrix
 * @param[in] num_cols the number of columns in the matrix
 * @param[in,out] lhs the matrix (updated inplace)
 * @param[in] scale the value to scale
 * @param[in] grid_x_offset the offset in x-dimension into the data points if more than one execution grid has to be used
 * @param[in] grid_y_offset the offset in y-dimension into the data points if more than one execution grid has to be used
 */
__kernel void device_kernel_inplace_matrix_scale(const ulong num_rows, const ulong num_cols, real_type __global *lhs, const real_type scale, const ulong grid_x_offset, const ulong grid_y_offset) {
    // cast all values to 64-bit unsigned long long to prevent potential 32-bit overflows
    const ulong threadIdx_x = get_local_id(0);                 // current work-item in work-group x-dimension
    const ulong threadIdx_y = get_local_id(1);                 // current work-item in work-group y-dimension
    const ulong blockDim_x = get_local_size(0);                // number of work-items in work-group x-dimension
    const ulong blockDim_y = get_local_size(1);                // number of work-items in work-group y-dimension
    const ulong blockIdx_x = get_group_id(0) + grid_x_offset;  // current work-group in global range x-dimension + offsets if the global range is too large
    const ulong blockIdx_y = get_group_id(1) + grid_y_offset;  // current work-group in global range y-dimension + offsets if the global range is too large

    // calculate the indices used in the current work-item
    const ulong global_i_idx = blockIdx_x * blockDim_x + threadIdx_x;  // num_rows
    const ulong global_j_idx = blockIdx_y * blockDim_y + threadIdx_y;  // num_rhs

    if (global_i_idx < num_rows && global_j_idx < num_cols) {
        lhs[global_i_idx * num_cols + global_j_idx] *= scale;  // SoA
    }
}
