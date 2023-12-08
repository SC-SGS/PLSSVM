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
 */
__kernel void device_kernel_w_linear(__global real_type *w_d, __global const real_type *alpha_d, __global const real_type *sv_d, const ulong num_classes, const ulong num_sv) {
    const ulong feature_idx = get_global_id(0) * INTERNAL_BLOCK_SIZE;
    const ulong feature_idx_linear = get_group_id(0) * get_local_size(0) * INTERNAL_BLOCK_SIZE + get_local_id(0);
    const ulong class_idx = get_global_id(1) * INTERNAL_BLOCK_SIZE;
    const ulong class_cached_idx_linear = get_group_id(1) * get_local_size(1) * INTERNAL_BLOCK_SIZE + get_local_id(0);

    __local real_type data_cache_feature[THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];
    __local real_type data_cache_alpha[THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];

    real_type temp[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE] = { 0.0 };

    for (ulong sv = 0; sv < num_sv; sv += THREAD_BLOCK_SIZE) {
        // load data into shared memory
        for (uint internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
            const ulong global_feature_idx = feature_idx_linear + internal * THREAD_BLOCK_SIZE;
            const ulong global_class_idx = class_cached_idx_linear + internal * THREAD_BLOCK_SIZE;

            data_cache_feature[get_local_id(1)][internal * THREAD_BLOCK_SIZE + get_local_id(0)] = sv_d[global_feature_idx * (num_sv + PADDING_SIZE) + sv + get_local_id(1)];
            data_cache_alpha[get_local_id(1)][internal * THREAD_BLOCK_SIZE + get_local_id(0)] = alpha_d[global_class_idx * (num_sv + PADDING_SIZE) + sv + get_local_id(1)];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // calculation
        for (uint block_dim = 0; block_dim < THREAD_BLOCK_SIZE; ++block_dim) {
            for (uint internal_feature = 0; internal_feature < INTERNAL_BLOCK_SIZE; ++internal_feature) {
                for (uint internal_class = 0; internal_class < INTERNAL_BLOCK_SIZE; ++internal_class) {
                    temp[internal_feature][internal_class] += data_cache_alpha[block_dim][get_local_id(1) * INTERNAL_BLOCK_SIZE + internal_class] * data_cache_feature[block_dim][get_local_id(0) * INTERNAL_BLOCK_SIZE + internal_feature];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (uint internal_feature = 0; internal_feature < INTERNAL_BLOCK_SIZE; ++internal_feature) {
        for (uint internal_class = 0; internal_class < INTERNAL_BLOCK_SIZE; ++internal_class) {
            const ulong global_feature_idx = feature_idx + internal_feature;
            const ulong global_class_idx = class_idx + internal_class;

            w_d[global_feature_idx * (num_classes + PADDING_SIZE) + global_class_idx] = temp[internal_feature][internal_class];
        }
    }
}

/**
 * @brief Predict the @p predict_points_d using the linear kernel speeding up the calculation using the @p w_d vector.
 * @param[out] out_d the predicted values
 * @param[in] w_d the vector to speedup the calculations
 * @param[in] rho_d the previously learned bias
 * @param[in] predict_points_d the data points to predict
 * @param[in] num_classes the number of classes
 * @param[in] num_predict_points the number of data points to predict
 * @param[in] num_features the number of features per data point
 */
__kernel void device_kernel_predict_linear(__global real_type *out_d, __global const real_type *w_d, __global const real_type *rho_d, __global const real_type *predict_points_d, const ulong num_classes, const ulong num_predict_points, const ulong num_features) {
    const ulong pp_idx = get_global_id(0) * INTERNAL_BLOCK_SIZE;
    const ulong pp_idx_linear = get_group_id(0) * get_local_size(0) * INTERNAL_BLOCK_SIZE + get_local_id(0);
    const ulong class_idx = get_global_id(1) * INTERNAL_BLOCK_SIZE;
    const ulong class_cached_idx_linear = get_group_id(1) * get_local_size(1) * INTERNAL_BLOCK_SIZE + get_local_id(0);

    __local real_type data_cache_pp[FEATURE_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];
    __local real_type data_cache_w[FEATURE_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];

    real_type temp[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE] = { 0.0 };

    for (ulong dim = 0; dim < num_features; dim += FEATURE_BLOCK_SIZE) {
        // load data into shared memory
        for (uint internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
            const ulong global_pp_idx = pp_idx_linear + internal * THREAD_BLOCK_SIZE;
            const ulong global_class_idx = class_cached_idx_linear + internal * THREAD_BLOCK_SIZE;

            data_cache_pp[get_local_id(1)][internal * THREAD_BLOCK_SIZE + get_local_id(0)] = predict_points_d[(dim + get_local_id(1)) * (num_predict_points + PADDING_SIZE) + global_pp_idx];
            data_cache_pp[get_local_id(1) + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + get_local_id(0)] = predict_points_d[(dim + get_local_id(1) + THREAD_BLOCK_SIZE) * (num_predict_points + PADDING_SIZE) + global_pp_idx];
            data_cache_w[get_local_id(1)][internal * THREAD_BLOCK_SIZE + get_local_id(0)] = w_d[(dim + get_local_id(1)) * (num_classes + PADDING_SIZE) + global_class_idx];
            data_cache_w[get_local_id(1) + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + get_local_id(0)] = w_d[(dim + get_local_id(1) + THREAD_BLOCK_SIZE) * (num_classes + PADDING_SIZE) + global_class_idx];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // calculation
        for (uint block_dim = 0; block_dim < FEATURE_BLOCK_SIZE; ++block_dim) {
            for (uint internal_pd = 0; internal_pd < INTERNAL_BLOCK_SIZE; ++internal_pd) {
                for (uint internal_class = 0; internal_class < INTERNAL_BLOCK_SIZE; ++internal_class) {
                    temp[internal_pd][internal_class] += data_cache_w[block_dim][get_local_id(1) * INTERNAL_BLOCK_SIZE + internal_class] * data_cache_pp[block_dim][get_local_id(0) * INTERNAL_BLOCK_SIZE + internal_pd];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (uint internal_pd = 0; internal_pd < INTERNAL_BLOCK_SIZE; ++internal_pd) {
        for (uint internal_class = 0; internal_class < INTERNAL_BLOCK_SIZE; ++internal_class) {
            const ulong global_pp_idx = pp_idx + internal_pd;
            const ulong global_class_idx = class_idx + internal_class;

            out_d[global_pp_idx * (num_classes + PADDING_SIZE) + global_class_idx] = temp[internal_pd][internal_class] - rho_d[global_class_idx];
        }
    }
}

/**
 * @brief Predict the @p predict_points_d using the polynomial.
 * @param[in] out_d the predicted values
 * @param[in] alpha_d the previously learned weights
 * @param[in] rho_d the previously learned biases
 * @param[in] sv_d the support vectors
 * @param[in] predict_points_d the data points to predict
 * @param[in] num_classes the number of classes
 * @param[in] num_sv the number of support vectors
 * @param[in] num_predict_points the number of data points to predict
 * @param[in] num_features the number of features per data point
 * @param[in] degree the parameter in the polynomial kernel function
 * @param[in] gamma the parameter in the polynomial kernel function
 * @param[in] coef0 the parameter in the polynomial kernel function
 */
__kernel void device_kernel_predict_polynomial(__global real_type *out_d, __global const real_type *alpha_d, __global const real_type *rho_d, __global const real_type *sv_d, __global const real_type *predict_points_d, const ulong num_classes, const ulong num_sv, const ulong num_predict_points, const ulong num_features, const int degree, const real_type gamma, const real_type coef0) {
    const ulong pp_idx = get_global_id(0) * INTERNAL_BLOCK_SIZE;
    const ulong pp_idx_linear = get_group_id(0) * get_local_size(0) * INTERNAL_BLOCK_SIZE + get_local_id(0);
    const ulong sv_idx = get_global_id(1) * INTERNAL_BLOCK_SIZE;
    const ulong sv_cached_idx_linear = get_group_id(1) * get_local_size(1) * INTERNAL_BLOCK_SIZE + get_local_id(0);

    __local real_type data_cache_pp[FEATURE_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];
    __local real_type data_cache_sv[FEATURE_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];

    real_type temp[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE] = { 0.0 };

    for (ulong dim = 0; dim < num_features; dim += FEATURE_BLOCK_SIZE) {
        // load data into shared memory
        for (uint internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
            const ulong global_pp_idx = pp_idx_linear + internal * THREAD_BLOCK_SIZE;
            const ulong global_sv_idx = sv_cached_idx_linear + internal * THREAD_BLOCK_SIZE;

            data_cache_pp[get_local_id(1)][internal * THREAD_BLOCK_SIZE + get_local_id(0)] = predict_points_d[(dim + get_local_id(1)) * (num_predict_points + PADDING_SIZE) + global_pp_idx];
            data_cache_pp[get_local_id(1) + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + get_local_id(0)] = predict_points_d[(dim + get_local_id(1) + THREAD_BLOCK_SIZE) * (num_predict_points + PADDING_SIZE) + global_pp_idx];
            data_cache_sv[get_local_id(1)][internal * THREAD_BLOCK_SIZE + get_local_id(0)] = sv_d[(dim + get_local_id(1)) * (num_sv + PADDING_SIZE) + global_sv_idx];
            data_cache_sv[get_local_id(1) + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + get_local_id(0)] = sv_d[(dim + get_local_id(1) + THREAD_BLOCK_SIZE) * (num_sv + PADDING_SIZE) + global_sv_idx];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // calculation
        for (uint block_dim = 0; block_dim < FEATURE_BLOCK_SIZE; ++block_dim) {
            for (uint internal_pd = 0; internal_pd < INTERNAL_BLOCK_SIZE; ++internal_pd) {
                for (uint internal_sv = 0; internal_sv < INTERNAL_BLOCK_SIZE; ++internal_sv) {
                    temp[internal_pd][internal_sv] += data_cache_sv[block_dim][get_local_id(1) * INTERNAL_BLOCK_SIZE + internal_sv] * data_cache_pp[block_dim][get_local_id(0) * INTERNAL_BLOCK_SIZE + internal_pd];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (uint internal_pd = 0; internal_pd < INTERNAL_BLOCK_SIZE; ++internal_pd) {
        for (uint internal_sv = 0; internal_sv < INTERNAL_BLOCK_SIZE; ++internal_sv) {
            const ulong global_pp_idx = pp_idx + internal_pd;
            const ulong global_sv_idx = sv_idx + internal_sv;

            const real_type temp_pp_sv = temp[internal_pd][internal_sv];
            for (ulong class_idx = 0; class_idx < num_classes; ++class_idx) {
                // apply degree, gamma, and coef0, alpha and rho
                real_type class_temp = alpha_d[class_idx * (num_sv + PADDING_SIZE) + global_sv_idx] * pow(gamma * temp_pp_sv + coef0, degree);
                if (global_sv_idx == 0) {
                    class_temp -= rho_d[class_idx];
                }

                atomicAdd(&out_d[global_pp_idx * (num_classes + PADDING_SIZE) + class_idx], class_temp);
            }
        }
    }
}

/**
 * @brief Predict the @p predict_points_d using the rbf.
 * @param[in] out_d the predicted values
 * @param[in] alpha_d the previously learned weights
 * @param[in] rho_d the previously learned biases
 * @param[in] sv_d the support vectors
 * @param[in] predict_points_d the data points to predict
 * @param[in] num_classes the number of classes
 * @param[in] num_sv the number of support vectors
 * @param[in] num_predict_points the number of data points to predict
 * @param[in] num_features the number of features per data point
 * @param[in] gamma the parameter in the rbf kernel function
 */
__kernel void device_kernel_predict_rbf(__global real_type *out_d, __global const real_type *alpha_d, __global const real_type *rho_d, __global const real_type *sv_d, __global const real_type *predict_points_d, const ulong num_classes, const ulong num_sv, const ulong num_predict_points, const ulong num_features, const real_type gamma) {
    const ulong pp_idx = get_global_id(0) * INTERNAL_BLOCK_SIZE;
    const ulong pp_idx_linear = get_group_id(0) * get_local_size(0) * INTERNAL_BLOCK_SIZE + get_local_id(0);
    const ulong sv_idx = get_global_id(1) * INTERNAL_BLOCK_SIZE;
    const ulong sv_cached_idx_linear = get_group_id(1) * get_local_size(1) * INTERNAL_BLOCK_SIZE + get_local_id(0);

    __local real_type data_cache_pp[FEATURE_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];
    __local real_type data_cache_sv[FEATURE_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];

    real_type temp[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE] = { 0.0 };

    for (ulong dim = 0; dim < num_features; dim += FEATURE_BLOCK_SIZE) {
        // load data into shared memory
        for (uint internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
            const ulong global_pp_idx = pp_idx_linear + internal * THREAD_BLOCK_SIZE;
            const ulong global_sv_idx = sv_cached_idx_linear + internal * THREAD_BLOCK_SIZE;

            data_cache_pp[get_local_id(1)][internal * THREAD_BLOCK_SIZE + get_local_id(0)] = predict_points_d[(dim + get_local_id(1)) * (num_predict_points + PADDING_SIZE) + global_pp_idx];
            data_cache_pp[get_local_id(1) + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + get_local_id(0)] = predict_points_d[(dim + get_local_id(1) + THREAD_BLOCK_SIZE) * (num_predict_points + PADDING_SIZE) + global_pp_idx];
            data_cache_sv[get_local_id(1)][internal * THREAD_BLOCK_SIZE + get_local_id(0)] = sv_d[(dim + get_local_id(1)) * (num_sv + PADDING_SIZE) + global_sv_idx];
            data_cache_sv[get_local_id(1) + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + get_local_id(0)] = sv_d[(dim + get_local_id(1) + THREAD_BLOCK_SIZE) * (num_sv + PADDING_SIZE) + global_sv_idx];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // calculation
        for (uint block_dim = 0; block_dim < FEATURE_BLOCK_SIZE; ++block_dim) {
            for (uint internal_pd = 0; internal_pd < INTERNAL_BLOCK_SIZE; ++internal_pd) {
                for (uint internal_sv = 0; internal_sv < INTERNAL_BLOCK_SIZE; ++internal_sv) {
                    const real_type d = data_cache_sv[block_dim][get_local_id(1) * INTERNAL_BLOCK_SIZE + internal_sv] - data_cache_pp[block_dim][get_local_id(0) * INTERNAL_BLOCK_SIZE + internal_pd];
                    temp[internal_pd][internal_sv] += d * d;
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (uint internal_pd = 0; internal_pd < INTERNAL_BLOCK_SIZE; ++internal_pd) {
        for (uint internal_sv = 0; internal_sv < INTERNAL_BLOCK_SIZE; ++internal_sv) {
            const ulong global_pp_idx = pp_idx + internal_pd;
            const ulong global_sv_idx = sv_idx + internal_sv;

            const real_type temp_pp_sv = temp[internal_pd][internal_sv];
            for (ulong class_idx = 0; class_idx < num_classes; ++class_idx) {
                // apply gamma, alpha and rho
                real_type class_temp = alpha_d[class_idx * (num_sv + PADDING_SIZE) + global_sv_idx] * exp(-gamma * temp_pp_sv);
                if (global_sv_idx == 0) {
                    class_temp -= rho_d[class_idx];
                }

                atomicAdd(&out_d[global_pp_idx * (num_classes + PADDING_SIZE) + class_idx], class_temp);
            }
        }
    }
}