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
 * @brief Calculate the `q` vector used to speedup the prediction using the linear kernel function.
 * @param[in,out] w the vector to speedup the linear prediction
 * @param[in] alpha the previously learned weights
 * @param[in] support_vectors the support vectors
 * @param[in] num_classes the number of classes
 * @param[in] num_sv the number of support vectors
 * @param[in] device_num_sv the number of support vectors the current device is responsible for
 * @param[in] device_sv_offset the first support vector (row in @p alpha) the current device is responsible for
 * @param[in] grid_x_offset the offset in x-dimension into the data points if more than one execution grid has to be used
 * @param[in] grid_y_offset the offset in y-dimension into the data points if more than one execution grid has to be used
 */
__kernel void device_kernel_w_linear(__global real_type *w, const __global real_type *alpha, const __global real_type *support_vectors, const ulong num_classes, const ulong num_sv, const ulong device_num_sv, const ulong device_sv_offset, const ulong grid_x_offset, const ulong grid_y_offset) {
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
    __local real_type feature_cache[THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];
    __local real_type alpha_cache[THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];

    // create a work-item private array used for internal caching
    real_type temp[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE] = { (real_type) 0.0 };

    {
        // calculate the indices used in the current work-item, pays attention to coalesced memory accesses
        const ulong feature_idx_linear = blockIdx_x * blockDim_x * INTERNAL_BLOCK_SIZE_uz + threadIdx_x;  // num_features
        const ulong class_idx_linear = blockIdx_y * blockDim_y * INTERNAL_BLOCK_SIZE_uz + threadIdx_x;    // num_classes

        // iterate over all support vectors using blocking to be able to cache them for faster memory accesses
        for (ulong sv_block = 0; sv_block < device_num_sv; sv_block += THREAD_BLOCK_SIZE_uz) {
            // load data into local memory
            for (uint internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                // calculate the indices to access the global data, pays attention to coalesced memory accesses
                const ulong global_feature_idx_linear = feature_idx_linear + (ulong) internal * THREAD_BLOCK_SIZE_uz;
                const ulong global_class_idx_linear = class_idx_linear + (ulong) internal * THREAD_BLOCK_SIZE_uz;

                // store the values in the local memory
                feature_cache[local_id_1][internal * THREAD_BLOCK_SIZE + local_id_0] = support_vectors[global_feature_idx_linear * (device_num_sv + PADDING_SIZE_uz) + sv_block + threadIdx_y];  // SoA
                alpha_cache[local_id_1][internal * THREAD_BLOCK_SIZE + local_id_0] = alpha[global_class_idx_linear * (num_sv + PADDING_SIZE_uz) + sv_block + device_sv_offset + threadIdx_y];    // AoS
            }
            barrier(CLK_LOCAL_MEM_FENCE);  // wait until all work-items loaded their part of the data

#if defined(PLSSVM_OPENCL_TARGET_CPUS)
            // perform the dot product calculation, the sv is the fastest moving index
            for (uint internal_feature = 0; internal_feature < INTERNAL_BLOCK_SIZE; ++internal_feature) {
                for (uint internal_class = 0; internal_class < INTERNAL_BLOCK_SIZE; ++internal_class) {
                    real_type sum = 0.0;
                    for (uint sv = 0; sv < THREAD_BLOCK_SIZE; ++sv) {
                        sum += alpha_cache[sv][local_id_1 * INTERNAL_BLOCK_SIZE + internal_class] * feature_cache[sv][local_id_0 * INTERNAL_BLOCK_SIZE + internal_feature];
                    }
                    temp[internal_feature][internal_class] += sum;
                }
            }
#else
            // perform the dot product calculation, the sv is the slowest moving index
            for (uint sv = 0; sv < THREAD_BLOCK_SIZE; ++sv) {
                for (uint internal_feature = 0; internal_feature < INTERNAL_BLOCK_SIZE; ++internal_feature) {
                    for (uint internal_class = 0; internal_class < INTERNAL_BLOCK_SIZE; ++internal_class) {
                        temp[internal_feature][internal_class] += alpha_cache[sv][local_id_1 * INTERNAL_BLOCK_SIZE + internal_class] * feature_cache[sv][local_id_0 * INTERNAL_BLOCK_SIZE + internal_feature];
                    }
                }
            }
#endif
            barrier(CLK_LOCAL_MEM_FENCE);  // wait until all work-items performed their part of the calculations
        }
    }

    // calculate the indices used in the current work-item
    const ulong feature_idx = (blockIdx_x * blockDim_x + threadIdx_x) * INTERNAL_BLOCK_SIZE_uz;  // num_features
    const ulong class_idx = (blockIdx_y * blockDim_y + threadIdx_y) * INTERNAL_BLOCK_SIZE_uz;    // num_classes

    // update the global w-vector with the locally cached values
    for (uint internal_feature = 0; internal_feature < INTERNAL_BLOCK_SIZE; ++internal_feature) {
        for (uint internal_class = 0; internal_class < INTERNAL_BLOCK_SIZE; ++internal_class) {
            // calculate the indices to access the global data
            const ulong global_feature_idx = feature_idx + (ulong) internal_feature;
            const ulong global_class_idx = class_idx + (ulong) internal_class;

            w[global_feature_idx * (num_classes + PADDING_SIZE_uz) + global_class_idx] = temp[internal_feature][internal_class];  // SoA
        }
    }
}

/**
 * @brief Predict the @p predict_points using the linear kernel speeding up the calculation using the @p w vector.
 * @param[out] prediction the predicted values
 * @param[in] w the vector to speedup the calculations
 * @param[in] rho the previously learned bias
 * @param[in] predict_points the data points to predict
 * @param[in] num_classes the number of classes
 * @param[in] num_predict_points the number of data points to predict
 * @param[in] num_features the number of features per data point
 * @param[in] grid_x_offset the offset in x-dimension into the data points if more than one execution grid has to be used
 * @param[in] grid_y_offset the offset in y-dimension into the data points if more than one execution grid has to be used
 */
__kernel void device_kernel_predict_linear(__global real_type *prediction, const __global real_type *w, const __global real_type *rho, const __global real_type *predict_points, const ulong num_classes, const ulong num_predict_points, const ulong num_features, const ulong grid_x_offset, const ulong grid_y_offset) {
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
    __local real_type pp_cache[THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];
    __local real_type w_cache[THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];

    // create a work-item private array used for internal caching
    real_type temp[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE] = { (real_type) 0.0 };

    {
        // calculate the indices used in the current work-item, pays attention to coalesced memory accesses
        const ulong pp_idx_linear = blockIdx_x * blockDim_x * INTERNAL_BLOCK_SIZE_uz + threadIdx_x;     // num_predict_points
        const ulong class_idx_linear = blockIdx_y * blockDim_y * INTERNAL_BLOCK_SIZE_uz + threadIdx_x;  // num_classes

        // iterate over all features using blocking to be able to cache them for faster memory accesses
        for (ulong feature_block = 0; feature_block < num_features; feature_block += THREAD_BLOCK_SIZE_uz) {
            // load data into local memory
            for (uint internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                // calculate the indices to access the global data, pays attention to coalesced memory accesses
                const ulong global_pp_idx_linear = pp_idx_linear + (ulong) internal * THREAD_BLOCK_SIZE_uz;
                const ulong global_class_idx_linear = class_idx_linear + (ulong) internal * THREAD_BLOCK_SIZE_uz;

                // store the values in the local memory
                pp_cache[local_id_1][internal * THREAD_BLOCK_SIZE + local_id_0] = predict_points[(feature_block + threadIdx_y) * (num_predict_points + PADDING_SIZE_uz) + global_pp_idx_linear];  // SoA
                w_cache[local_id_1][internal * THREAD_BLOCK_SIZE + local_id_0] = w[(feature_block + threadIdx_y) * (num_classes + PADDING_SIZE_uz) + global_class_idx_linear];                    // SoA
            }
            barrier(CLK_LOCAL_MEM_FENCE);  // wait until all work-items loaded their part of the data

#if defined(PLSSVM_OPENCL_TARGET_CPUS)
            // perform the feature reduction calculation, the feature is the fastest moving index
            for (uint internal_pp = 0; internal_pp < INTERNAL_BLOCK_SIZE; ++internal_pp) {
                for (uint internal_class = 0; internal_class < INTERNAL_BLOCK_SIZE; ++internal_class) {
                    real_type sum = 0.0 : for (uint feature = 0; feature < THREAD_BLOCK_SIZE; ++feature) {
                        sum += w_cache[feature][local_id_1 * INTERNAL_BLOCK_SIZE + internal_class] * pp_cache[feature][local_id_0 * INTERNAL_BLOCK_SIZE + internal_pp];
                    }
                    temp[internal_pp][internal_class] += sum;
                }
            }
#else
            // perform the feature reduction calculation, the feature is the slowest moving index
            for (uint feature = 0; feature < THREAD_BLOCK_SIZE; ++feature) {
                for (uint internal_pp = 0; internal_pp < INTERNAL_BLOCK_SIZE; ++internal_pp) {
                    for (uint internal_class = 0; internal_class < INTERNAL_BLOCK_SIZE; ++internal_class) {
                        temp[internal_pp][internal_class] += w_cache[feature][local_id_1 * INTERNAL_BLOCK_SIZE + internal_class] * pp_cache[feature][local_id_0 * INTERNAL_BLOCK_SIZE + internal_pp];
                    }
                }
            }
#endif
            barrier(CLK_LOCAL_MEM_FENCE);  // wait until all work-items performed their part of the calculations
        }
    }

    // calculate the indices used in the current work-item
    const ulong pp_idx = (blockIdx_x * blockDim_x + threadIdx_x) * INTERNAL_BLOCK_SIZE_uz;     // num_predict_points
    const ulong class_idx = (blockIdx_y * blockDim_y + threadIdx_y) * INTERNAL_BLOCK_SIZE_uz;  // num_classes

    // update the global array with the local one
    for (uint internal_pp = 0; internal_pp < INTERNAL_BLOCK_SIZE; ++internal_pp) {
        for (uint internal_class = 0; internal_class < INTERNAL_BLOCK_SIZE; ++internal_class) {
            // calculate the indices to access the global data
            const ulong global_pp_idx = pp_idx + (ulong) internal_pp;
            const ulong global_class_idx = class_idx + (ulong) internal_class;

            prediction[global_pp_idx * (num_classes + PADDING_SIZE_uz) + global_class_idx] = temp[internal_pp][internal_class] - rho[global_class_idx];  // AoS
        }
    }
}
