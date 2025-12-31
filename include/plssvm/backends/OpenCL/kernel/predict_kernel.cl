/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines the functions used for prediction for the C-SVM using the OpenCL backend.
 */

// #include "detail/atomics.cl"  // atomicAdd -> included via string concatenation when building the device kernels

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

/**
 * @brief Predict the @p predict_points using the kernel function determined at runtime.
 * @details The `PLSSVM_DEVICE_KERNEL_PREDICT_NAME`, `PLSSVM_OPENCL_KERNEL_FUNCTION_PARAMETER_LIST`, `PLSSVM_OPENCL_KERNEL_FUNCTION_PARAMETER`, `PLSSVM_OPENCL_FEATURE_REDUCE_FUNCTION`, and `PLSSVM_OPENCL_APPLY_KERNEL_FUNCTION` placeholder will be replaced by the correct values upon kernel construction.
 * @param[in] prediction the predicted values
 * @param[in] alpha the previously learned weights
 * @param[in] rho the previously learned biases
 * @param[in] support_vectors the support vectors
 * @param[in] predict_points the data points to predict
 * @param[in] num_classes the number of classes
 * @param[in] num_sv the number of support vectors
 * @param[in] num_predict_points the number of data points to predict
 * @param[in] num_features the number of features per data point
 * @param[in] grid_x_offset the offset in x-dimension into the data points if more than one execution grid has to be used
 * @param[in] grid_y_offset the offset in y-dimension into the data points if more than one execution grid has to be used
 * @param[in] PLSSVM_OPENCL_KERNEL_FUNCTION_PARAMETER_LIST a placeholder that is used to string replace the correct kernel parameter (attention: no comma!; Args... only added for Doxygen)
 */
__kernel void PLSSVM_DEVICE_KERNEL_PREDICT_NAME(__global real_type *prediction, const __global real_type *alpha, const __global real_type *rho, const __global real_type *support_vectors, const __global real_type *predict_points, const ulong num_classes, const ulong num_sv, const ulong num_predict_points, const ulong num_features, const ulong grid_x_offset, const ulong grid_y_offset PLSSVM_OPENCL_KERNEL_FUNCTION_PARAMETER_LIST) {
    // cast all values to 64-bit unsigned long long to prevent potential 32-bit overflows
    const ulong threadIdx_x = get_local_id(0);                 // current work-item in work-group x-dimension
    const ulong threadIdx_y = get_local_id(1);                 // current work-item in work-group y-dimension
    const ulong blockDim_x = get_local_size(0);                // number of work-items in work-group x-dimension
    const ulong blockDim_y = get_local_size(1);                // number of work-items in work-group y-dimension
    const ulong blockIdx_x = get_group_id(0) + grid_x_offset;  // current work-group in global range x-dimension + offsets if the global range is too large
    const ulong blockIdx_y = get_group_id(1) + grid_y_offset;  // current work-group in global range y-dimension + offsets if the global range is too large

    // calculate the indices used in the current work-item
    const ulong global_pp_idx = blockIdx_x * blockDim_x + threadIdx_x;  // num_predict_points
    const ulong global_sv_idx = blockIdx_y * blockDim_y + threadIdx_y;  // num_sv

    // be sure to not perform out-of-bounds accesses
    if (global_sv_idx < num_sv && global_pp_idx < num_predict_points) {
        real_type temp = 0.0;

        // perform the feature reduction calculation
        for (ulong feature = 0; feature < num_features; ++feature) {
            temp += PLSSVM_OPENCL_FEATURE_REDUCE_FUNCTION(support_vectors[feature * num_sv + global_sv_idx],              // SoA
                                                          predict_points[feature * num_predict_points + global_pp_idx]);  // SoA
        }

        // update temp using the respective kernel function
        temp = PLSSVM_OPENCL_APPLY_KERNEL_FUNCTION(temp PLSSVM_OPENCL_KERNEL_FUNCTION_PARAMETER);

        // iterate over all classes
        for (ulong class_idx = 0; class_idx < num_classes; ++class_idx) {
            real_type out_cache = alpha[class_idx * num_sv + global_sv_idx] * temp;  // AoS

            // the bias (rho) must only be applied once for all support vectors
            if (global_sv_idx == (ulong) 0) {
                out_cache -= rho[class_idx];
            }

            atomicAdd(&prediction[global_pp_idx * num_classes + class_idx], out_cache);  // AoS
        }
    }
}
