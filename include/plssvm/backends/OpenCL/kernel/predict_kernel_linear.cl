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
 * @param[in,out] w_d the vector to speedup the linear prediction
 * @param[in] alpha_d the previously learned weights
 * @param[in] sv_d the support vectors
 * @param[in] num_classes the number of classes
 * @param[in] num_sv the number of support vectors
 * @param[in] device_specific_num_sv the number of support vectors the current device is responsible for
 * @param[in] sv_offset the first support vector (row in @p alpha_d) the current device is responsible for
 */
__kernel void device_kernel_w_linear(__global real_type *w_d, const __global real_type *alpha_d, const __global real_type *sv_d, const ulong num_classes, const ulong num_sv, const ulong device_specific_num_sv, const ulong sv_offset) {
    // cast values to 32-bit unsigned int values to prevent implicit conversions
    const uint local_id_0 = get_local_id(0);
    const uint local_id_1 = get_local_id(1);

    // calculate the indices used in the current thread
    const ulong feature_idx = get_global_id(0) * INTERNAL_BLOCK_SIZE_ul;
    const ulong feature_idx_linear = get_group_id(0) * get_local_size(0) * INTERNAL_BLOCK_SIZE_ul + get_local_id(0);
    const ulong class_idx = get_global_id(1) * INTERNAL_BLOCK_SIZE_ul;
    const ulong class_linear = get_group_id(1) * get_local_size(1) * INTERNAL_BLOCK_SIZE_ul + get_local_id(0);

    // create the local memory arrays used for caching data point features
    __local real_type data_cache_feature[THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];
    __local real_type data_cache_alpha[THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];

    // create a thread private array used for internal caching
    real_type temp[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE] = { (real_type) 0.0 };

    // iterate over all support vectors using blocking to be able to cache them for faster memory accesses
    for (ulong sv = 0; sv < device_specific_num_sv; sv += THREAD_BLOCK_SIZE_ul) {
        // load data into local memory
        for (uint internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
            const ulong global_feature_idx = feature_idx_linear + (ulong) internal * THREAD_BLOCK_SIZE_ul;
            const ulong global_class_idx = class_linear + (ulong) internal * THREAD_BLOCK_SIZE_ul;

            data_cache_feature[local_id_1][internal * THREAD_BLOCK_SIZE + local_id_0] = sv_d[global_feature_idx * (device_specific_num_sv + PADDING_SIZE_ul) + sv + get_local_id(1)];  // SoA
            data_cache_alpha[local_id_1][internal * THREAD_BLOCK_SIZE + local_id_0] = alpha_d[global_class_idx * (num_sv + PADDING_SIZE_ul) + sv + sv_offset + get_local_id(1)];       // AoS
        }
        barrier(CLK_LOCAL_MEM_FENCE);  // wait until all work-items loaded their part of the data

        // perform the dot product calculation
        for (uint block_dim = 0; block_dim < THREAD_BLOCK_SIZE; ++block_dim) {
            for (uint internal_feature = 0; internal_feature < INTERNAL_BLOCK_SIZE; ++internal_feature) {
                for (uint internal_class = 0; internal_class < INTERNAL_BLOCK_SIZE; ++internal_class) {
                    temp[internal_feature][internal_class] += data_cache_alpha[block_dim][local_id_1 * INTERNAL_BLOCK_SIZE + internal_class] * data_cache_feature[block_dim][local_id_0 * INTERNAL_BLOCK_SIZE + internal_feature];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);  // wait until all threads performed their part of the calculations
    }

    // update global array with local one
    for (uint internal_feature = 0; internal_feature < INTERNAL_BLOCK_SIZE; ++internal_feature) {
        for (uint internal_class = 0; internal_class < INTERNAL_BLOCK_SIZE; ++internal_class) {
            const ulong global_feature_idx = feature_idx + (ulong) internal_feature;
            const ulong global_class_idx = class_idx + (ulong) internal_class;

            w_d[global_feature_idx * (num_classes + PADDING_SIZE_ul) + global_class_idx] = temp[internal_feature][internal_class];
        }
    }
}

/**
 * @brief Predict the @p predict_points_d using the linear kernel speeding up the calculation using the @p w_d vector.
 * @param[out] prediction_d the predicted values
 * @param[in] w_d the vector to speedup the calculations
 * @param[in] rho_d the previously learned bias
 * @param[in] predict_points_d the data points to predict
 * @param[in] num_classes the number of classes
 * @param[in] num_predict_points the number of data points to predict
 * @param[in] num_features the number of features per data point
 */
__kernel void device_kernel_predict_linear(__global real_type *prediction_d, const __global real_type *w_d, const __global real_type *rho_d, const __global real_type *predict_points_d, const ulong num_classes, const ulong num_predict_points, const ulong num_features) {
    // cast values to 32-bit unsigned int values to prevent implicit conversions
    const uint local_id_0 = get_local_id(0);
    const uint local_id_1 = get_local_id(1);

    // calculate the indices used in the current thread
    const ulong pp_idx = get_global_id(0) * INTERNAL_BLOCK_SIZE_ul;
    const ulong pp_idx_linear = get_group_id(0) * get_local_size(0) * INTERNAL_BLOCK_SIZE_ul + get_local_id(0);
    const ulong class_idx = get_global_id(1) * INTERNAL_BLOCK_SIZE_ul;
    const ulong class_cached_idx_linear = get_group_id(1) * get_local_size(1) * INTERNAL_BLOCK_SIZE_ul + get_local_id(0);

    // create the local memory arrays used for caching data point features
    __local real_type data_cache_pp[FEATURE_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];
    __local real_type data_cache_w[FEATURE_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];

    // create a thread private array used for internal caching
    real_type temp[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE] = { (real_type) 0.0 };

    // iterate over all features using blocking to be able to cache them for faster memory accesses
    for (ulong dim = 0; dim < num_features; dim += FEATURE_BLOCK_SIZE_ul) {
        // load data into local memory
        for (uint internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
            const ulong global_pp_idx = pp_idx_linear + internal * THREAD_BLOCK_SIZE;
            const ulong global_class_idx = class_cached_idx_linear + internal * THREAD_BLOCK_SIZE;

            // FEATURE_BLOCK_SIZE = 2 * THREAD_BLOCK_SIZE -> store twice as many values in the shared memory
            data_cache_pp[local_id_1][internal * THREAD_BLOCK_SIZE + local_id_0] = predict_points_d[(dim + get_local_id(1)) * (num_predict_points + PADDING_SIZE_ul) + global_pp_idx];
            data_cache_pp[local_id_1 + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + local_id_0] = predict_points_d[(dim + get_local_id(1) + THREAD_BLOCK_SIZE_ul) * (num_predict_points + PADDING_SIZE_ul) + global_pp_idx];
            data_cache_w[local_id_1][internal * THREAD_BLOCK_SIZE + local_id_0] = w_d[(dim + get_local_id(1)) * (num_classes + PADDING_SIZE_ul) + global_class_idx];
            data_cache_w[local_id_1 + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + local_id_0] = w_d[(dim + get_local_id(1) + THREAD_BLOCK_SIZE_ul) * (num_classes + PADDING_SIZE_ul) + global_class_idx];
        }
        barrier(CLK_LOCAL_MEM_FENCE);  // wait until all work-items loaded their part of the data

        // perform the dot product calculation
        for (uint block_dim = 0; block_dim < FEATURE_BLOCK_SIZE; ++block_dim) {
            for (uint internal_pd = 0; internal_pd < INTERNAL_BLOCK_SIZE; ++internal_pd) {
                for (uint internal_class = 0; internal_class < INTERNAL_BLOCK_SIZE; ++internal_class) {
                    temp[internal_pd][internal_class] += data_cache_w[block_dim][local_id_1 * INTERNAL_BLOCK_SIZE + internal_class] * data_cache_pp[block_dim][local_id_0 * INTERNAL_BLOCK_SIZE + internal_pd];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);  // wait until all work-items performed their part of the calculations
    }

    // update global array with local one
    for (uint internal_pd = 0; internal_pd < INTERNAL_BLOCK_SIZE; ++internal_pd) {
        for (uint internal_class = 0; internal_class < INTERNAL_BLOCK_SIZE; ++internal_class) {
            const ulong global_pp_idx = pp_idx + (ulong) internal_pd;
            const ulong global_class_idx = class_idx + (ulong) internal_class;

            prediction_d[global_pp_idx * (num_classes + PADDING_SIZE_ul) + global_class_idx] = temp[internal_pd][internal_class] - rho_d[global_class_idx];
        }
    }
}
