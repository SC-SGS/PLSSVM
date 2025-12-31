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
    // cast all values to 64-bit unsigned long long to prevent potential 32-bit overflows
    const ulong threadIdx_x = get_local_id(0);                 // current work-item in work-group x-dimension
    const ulong threadIdx_y = get_local_id(1);                 // current work-item in work-group y-dimension
    const ulong blockDim_x = get_local_size(0);                // number of work-items in work-group x-dimension
    const ulong blockDim_y = get_local_size(1);                // number of work-items in work-group y-dimension
    const ulong blockIdx_x = get_group_id(0) + grid_x_offset;  // current work-group in global range x-dimension + offsets if the global range is too large
    const ulong blockIdx_y = get_group_id(1) + grid_y_offset;  // current work-group in global range y-dimension + offsets if the global range is too large

    // calculate the indices used in the current work-item
    const ulong device_global_i_idx = blockIdx_x * blockDim_x + threadIdx_x;  // num_rows - device_row_offset
    const ulong global_i_idx = device_row_offset + device_global_i_idx;
    const ulong device_global_j_idx = blockIdx_y * blockDim_y + threadIdx_y;  // device_num_rows
    const ulong global_j_idx = device_row_offset + device_global_j_idx;

    // be sure to not perform out-of-bounds accesses (only using the upper triangular matrix)
    if (global_i_idx < num_rows && global_j_idx < num_rows && device_global_i_idx < (num_rows - device_row_offset) && device_global_j_idx < device_num_rows && global_i_idx >= global_j_idx) {
        //*************************************************************************//
        //                   inplace kernel matrix construction                    //
        //*************************************************************************//
        real_type temp = 0.0;

        // perform the feature reduction calculation
        for (ulong feature = 0; feature < num_features; ++feature) {
            temp += PLSSVM_OPENCL_FEATURE_REDUCE_FUNCTION(data[global_i_idx * num_features + feature],   // AoS
                                                          data[global_j_idx * num_features + feature]);  // AoS
        }

        // apply the final kernel function
        temp = PLSSVM_OPENCL_APPLY_KERNEL_FUNCTION(temp PLSSVM_OPENCL_KERNEL_FUNCTION_PARAMETER) + QA_cost - q[global_i_idx] - q[global_j_idx];
        // apply the cost on the diagonal
        if (global_i_idx == global_j_idx) {
            temp += cost;
        }

        //*************************************************************************//
        //     calculate C += alpha * temp * B for the UPPER triangular matrix     //
        //*************************************************************************//
        for (ulong class_idx = 0; class_idx < num_classes; ++class_idx) {
            const real_type B_cache = alpha * B[class_idx * num_rows + global_i_idx];  // AoS
            atomicAdd(&C[class_idx * num_rows + global_j_idx], temp * B_cache);        // AoS
        }

        // set potential diagonal entries in temp to 0.0 such that we don't apply the main diagonal twice to C
        if (global_i_idx == global_j_idx) {
            temp = 0.0;
        }

        //*************************************************************************//
        //     calculate C += alpha * temp * B for the LOWER triangular matrix     //
        //*************************************************************************//
        for (ulong class_idx = 0; class_idx < num_classes; ++class_idx) {
            const real_type B_cache = alpha * B[class_idx * num_rows + global_j_idx];  // AoS
            atomicAdd(&C[class_idx * num_rows + global_i_idx], temp * B_cache);        // AoS
        }
    }
}
