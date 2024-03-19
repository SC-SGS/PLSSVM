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
 * @brief Predict the @p predict_points_d using the kernel function determined at runtime.
 * @details The `PLSSVM_DEVICE_KERNEL_PREDICT_NAME`, `PLSSVM_OPENCL_KERNEL_FUNCTION_PARAMETER_LIST`, `PLSSVM_OPENCL_KERNEL_FUNCTION_PARAMETER`, `PLSSVM_OPENCL_FEATURE_REDUCE_FUNCTION`, and `PLSSVM_OPENCL_APPLY_KERNEL_FUNCTION` placeholder will be replaced by the correct values upon kernel construction.
 * @param[in] out_d the predicted values
 * @param[in] alpha_d the previously learned weights
 * @param[in] rho_d the previously learned biases
 * @param[in] sv_d the support vectors
 * @param[in] predict_points_d the data points to predict
 * @param[in] num_classes the number of classes
 * @param[in] num_sv the number of support vectors
 * @param[in] num_predict_points the number of data points to predict
 * @param[in] num_features the number of features per data point
 * @param[in] PLSSVM_OPENCL_KERNEL_FUNCTION_PARAMETER_LIST a placeholder that is used to string replace the correct kernel parameter (attention: no comma!)
 */
__kernel void PLSSVM_DEVICE_KERNEL_PREDICT_NAME(__global real_type *out_d, const __global real_type *alpha_d, const __global real_type *rho_d, const __global real_type *sv_d, const __global real_type *predict_points_d, const ulong num_classes, const ulong num_sv, const ulong num_predict_points, const ulong num_features PLSSVM_OPENCL_KERNEL_FUNCTION_PARAMETER_LIST) {
    const ulong pp_idx = get_global_id(0) * INTERNAL_BLOCK_SIZE;
    const ulong pp_idx_linear = get_group_id(0) * get_local_size(0) * INTERNAL_BLOCK_SIZE + get_local_id(0);
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
                    temp[internal_pd][internal_sv] += PLSSVM_OPENCL_FEATURE_REDUCE_FUNCTION(data_cache_sv[block_dim][get_local_id(1) * INTERNAL_BLOCK_SIZE + internal_sv], data_cache_pp[block_dim][get_local_id(0) * INTERNAL_BLOCK_SIZE + internal_pd]);
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // update temp using the respective kernel function
    for (uint internal_pd = 0; internal_pd < INTERNAL_BLOCK_SIZE; ++internal_pd) {
        for (uint internal_sv = 0; internal_sv < INTERNAL_BLOCK_SIZE; ++internal_sv) {
            temp[internal_pd][internal_sv] = PLSSVM_OPENCL_APPLY_KERNEL_FUNCTION(temp[internal_pd][internal_sv] PLSSVM_OPENCL_KERNEL_FUNCTION_PARAMETER);
        }
    }

    {
        real_type (*alpha_cache)[INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE] = (real_type (*)[INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE]) data_cache_pp;
        real_type (*out_cache)[INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE] = (real_type (*)[INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE]) data_cache_sv;

        for (ulong dim = 0; dim < num_classes; dim += FEATURE_BLOCK_SIZE) {
            // load data into shared memory
            for (uint internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                const ulong global_sv_idx = sv_cached_idx_linear + internal * THREAD_BLOCK_SIZE;

                alpha_cache[get_local_id(1)][internal * THREAD_BLOCK_SIZE + get_local_id(0)] = alpha_d[(dim + get_local_id(1)) * (num_sv + PADDING_SIZE) + global_sv_idx];
                alpha_cache[get_local_id(1) + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + get_local_id(0)] = alpha_d[(dim + get_local_id(1) + THREAD_BLOCK_SIZE) * (num_sv + PADDING_SIZE) + global_sv_idx];

                // the bias (rho) must only be applied once for all support vectors
                if (get_group_id(1) == 0) {
                    out_cache[get_local_id(1)][internal * THREAD_BLOCK_SIZE + get_local_id(0)] = -rho_d[dim + get_local_id(1)];
                    out_cache[get_local_id(1) + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + get_local_id(0)] = -rho_d[dim + get_local_id(1) + THREAD_BLOCK_SIZE];
                } else {
                    out_cache[get_local_id(1)][internal * THREAD_BLOCK_SIZE + get_local_id(0)] = 0.0;
                    out_cache[get_local_id(1) + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + get_local_id(0)] = 0.0;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            // calculate intermediate results and store them in shared memory
            for (uint class_idx = 0; class_idx < FEATURE_BLOCK_SIZE; ++class_idx) {
                for (uint internal_pd = 0; internal_pd < INTERNAL_BLOCK_SIZE; ++internal_pd) {
                    for (uint internal_sv = 0; internal_sv < INTERNAL_BLOCK_SIZE; ++internal_sv) {
                        out_cache[(class_idx + get_local_id(1)) % FEATURE_BLOCK_SIZE][internal_pd * THREAD_BLOCK_SIZE + get_local_id(0)] +=
                            temp[internal_pd][internal_sv] * alpha_cache[(class_idx + get_local_id(1)) % FEATURE_BLOCK_SIZE][get_local_id(1) * INTERNAL_BLOCK_SIZE + internal_sv];
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }

            // add intermediate cached results to out_d
            for (uint internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                const ulong global_pp_idx = pp_idx + internal;

                atomicAdd(&out_d[global_pp_idx * (num_classes + PADDING_SIZE) + dim + get_local_id(1)], out_cache[get_local_id(1)][internal * THREAD_BLOCK_SIZE + get_local_id(0)]);
                atomicAdd(&out_d[global_pp_idx * (num_classes + PADDING_SIZE) + dim + get_local_id(1) + THREAD_BLOCK_SIZE], out_cache[get_local_id(1) + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + get_local_id(0)]);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
}