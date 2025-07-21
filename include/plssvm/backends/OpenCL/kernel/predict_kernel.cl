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
    __local real_type cache_one[THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];
    __local real_type cache_two[THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];

    // create a work-item private array used for internal caching
    real_type temp[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE] = { (real_type) 0.0 };

    {
        // reinterpret the local memory arrays to be of shape [THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE]
        __local real_type(*pp_cache)[INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE] = (__local real_type(*)[INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE]) cache_one;
        __local real_type(*sv_cache)[INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE] = (__local real_type(*)[INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE]) cache_two;

        // calculate the indices used in the current work-item, pays attention to coalesced memory accesses
        const ulong pp_idx_linear = blockIdx_x * blockDim_x * INTERNAL_BLOCK_SIZE_uz + threadIdx_x;  // num_predict_points
        const ulong sv_idx_linear = blockIdx_y * blockDim_y * INTERNAL_BLOCK_SIZE_uz + threadIdx_x;  // num_support_vectors

        // iterate over all features using blocking to be able to cache them for faster memory accesses
        for (ulong feature_block = 0; feature_block < num_features; feature_block += THREAD_BLOCK_SIZE_uz) {
            // load data into local memory
            for (uint internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                // calculate the indices to access the global data, pays attention to coalesced memory accesses
                const ulong global_pp_idx_linear = pp_idx_linear + (ulong) internal * THREAD_BLOCK_SIZE_uz;
                const ulong global_sv_idx_linear = sv_idx_linear + (ulong) internal * THREAD_BLOCK_SIZE_uz;

                // store the values in the local memory
                pp_cache[local_id_1][internal * THREAD_BLOCK_SIZE + local_id_0] = predict_points[(feature_block + threadIdx_y) * (num_predict_points + PADDING_SIZE_uz) + global_pp_idx_linear];  // SoA
                sv_cache[local_id_1][internal * THREAD_BLOCK_SIZE + local_id_0] = support_vectors[(feature_block + threadIdx_y) * (num_sv + PADDING_SIZE_uz) + global_sv_idx_linear];             // SoA
            }
            barrier(CLK_LOCAL_MEM_FENCE);  // wait until all work-items loaded their part of the data

#if defined(PLSSVM_OPENCL_TARGET_CPUS)
            // perform the feature reduction calculation, the feature is the fastest moving index
            for (uint internal_pp = 0; internal_pp < INTERNAL_BLOCK_SIZE; ++internal_pp) {
                for (uint internal_sv = 0; internal_sv < INTERNAL_BLOCK_SIZE; ++internal_sv) {
                    real_type sum = 0.0;
                    for (uint feature = 0; feature < THREAD_BLOCK_SIZE; ++feature) {
                        sum += PLSSVM_OPENCL_FEATURE_REDUCE_FUNCTION(sv_cache[feature][local_id_1 * INTERNAL_BLOCK_SIZE + internal_sv], pp_cache[feature][local_id_0 * INTERNAL_BLOCK_SIZE + internal_pp]);
                    }
                    temp[internal_pp][internal_sv] += sum;
                }
            }
#else
            // perform the feature reduction calculation, the feature is the slowest moving index
            for (uint feature = 0; feature < THREAD_BLOCK_SIZE; ++feature) {
                for (uint internal_pp = 0; internal_pp < INTERNAL_BLOCK_SIZE; ++internal_pp) {
                    for (uint internal_sv = 0; internal_sv < INTERNAL_BLOCK_SIZE; ++internal_sv) {
                        temp[internal_pp][internal_sv] += PLSSVM_OPENCL_FEATURE_REDUCE_FUNCTION(sv_cache[feature][local_id_1 * INTERNAL_BLOCK_SIZE + internal_sv], pp_cache[feature][local_id_0 * INTERNAL_BLOCK_SIZE + internal_pp]);
                    }
                }
            }
#endif
            barrier(CLK_LOCAL_MEM_FENCE);  // wait until all work-items performed their part of the calculations
        }
    }

    // update temp using the respective kernel function
    for (uint internal_pp = 0; internal_pp < INTERNAL_BLOCK_SIZE; ++internal_pp) {
        for (uint internal_sv = 0; internal_sv < INTERNAL_BLOCK_SIZE; ++internal_sv) {
            temp[internal_pp][internal_sv] = PLSSVM_OPENCL_APPLY_KERNEL_FUNCTION(temp[internal_pp][internal_sv] PLSSVM_OPENCL_KERNEL_FUNCTION_PARAMETER);
        }
    }

    {
        // reinterpret the local memory arrays to be of shape [THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE]
        __local real_type(*alpha_cache)[INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE] = (__local real_type(*)[INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE]) cache_one;
        __local real_type(*out_cache)[INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE] = (__local real_type(*)[INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE]) cache_two;

        // calculate the indices used in the current thread
        const ulong pp_idx = (blockIdx_x * blockDim_x + threadIdx_x) * INTERNAL_BLOCK_SIZE_uz;  // num_predict_points
        // calculate the indices used in the current thread, pays attention to coalesced memory accesses
        const ulong sv_idx_linear = blockIdx_y * blockDim_y * INTERNAL_BLOCK_SIZE_uz + threadIdx_x;  // num_support_vectors

        // iterate over all classes using blocking to be able to cache them for faster memory accesses
        for (ulong class_block = 0; class_block < num_classes; class_block += THREAD_BLOCK_SIZE_uz) {
            // load data into local memory
            for (uint internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                // calculate the indices to access the global data, pays attention to coalesced memory accesses
                const ulong global_sv_idx_linear = sv_idx_linear + (ulong) internal * THREAD_BLOCK_SIZE_uz;

                // store the values in the local memory
                alpha_cache[local_id_1][internal * THREAD_BLOCK_SIZE + local_id_0] = alpha[(class_block + threadIdx_y) * (num_sv + PADDING_SIZE_uz) + global_sv_idx_linear];  // AoS
                // the bias (rho) must only be applied once for all support vectors
                if (blockIdx_y == (ulong) 0) {
                    out_cache[local_id_1][internal * THREAD_BLOCK_SIZE + local_id_0] = -rho[class_block + threadIdx_y];
                } else {
                    out_cache[local_id_1][internal * THREAD_BLOCK_SIZE + local_id_0] = (real_type) 0.0;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);  // wait until all work-items loaded their part of the data

            // calculate intermediate results and store them in shared memory
            for (uint class_idx = 0; class_idx < THREAD_BLOCK_SIZE; ++class_idx) {
                for (uint internal_pp = 0; internal_pp < INTERNAL_BLOCK_SIZE; ++internal_pp) {
                    for (uint internal_sv = 0; internal_sv < INTERNAL_BLOCK_SIZE; ++internal_sv) {
                        out_cache[(class_idx + local_id_1) % THREAD_BLOCK_SIZE][internal_pp * THREAD_BLOCK_SIZE + local_id_0] +=
                            temp[internal_pp][internal_sv] * alpha_cache[(class_idx + local_id_1) % THREAD_BLOCK_SIZE][local_id_1 * INTERNAL_BLOCK_SIZE + internal_sv];
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);  // wait until all work-items performed their part of the calculations
            }

            // atomically add the intermediate cached results to the prediction
            for (uint internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                // calculate the indices to access the global data
                const ulong global_pp_idx = pp_idx + (ulong) internal;

                atomicAdd(&prediction[global_pp_idx * (num_classes + PADDING_SIZE_uz) + class_block + threadIdx_y], out_cache[local_id_1][internal * THREAD_BLOCK_SIZE + local_id_0]);
            }
            barrier(CLK_LOCAL_MEM_FENCE);  // wait until all work-items updated their part of the prediction
        }
    }
}
