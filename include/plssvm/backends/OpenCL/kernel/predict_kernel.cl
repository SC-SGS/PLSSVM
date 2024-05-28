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
 * @param[in] prediction_d the predicted values
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
__kernel void PLSSVM_DEVICE_KERNEL_PREDICT_NAME(__global real_type *prediction_d, const __global real_type *alpha_d, const __global real_type *rho_d, const __global real_type *sv_d, const __global real_type *predict_points_d, const ulong num_classes, const ulong num_sv, const ulong num_predict_points, const ulong num_features, const ulong grid_x_offset, const ulong grid_y_offset PLSSVM_OPENCL_KERNEL_FUNCTION_PARAMETER_LIST) {
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

    // calculate the indices used in the current thread
    const ulong pp_idx = (blockIdx_x * blockDim_x + threadIdx_x) * INTERNAL_BLOCK_SIZE_ul;
    const ulong pp_idx_linear = blockIdx_x * blockDim_x * INTERNAL_BLOCK_SIZE_ul + threadIdx_x;
    const ulong sv_idx_linear = blockIdx_y * blockDim_y * INTERNAL_BLOCK_SIZE_ul + threadIdx_x;

    // create the local memory arrays used for caching data point features
    __local real_type data_cache_pp[FEATURE_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];
    __local real_type data_cache_sv[FEATURE_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];

    // create a thread private array used for internal caching
    real_type temp[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE] = { (real_type) 0.0 };

    // iterate over all features using blocking to be able to cache them for faster memory accesses
    for (ulong dim = 0; dim < num_features; dim += FEATURE_BLOCK_SIZE_ul) {
        // load data into local memory
        for (uint internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
            const ulong global_pp_idx = pp_idx_linear + (ulong) internal * THREAD_BLOCK_SIZE_ul;
            const ulong global_sv_idx = sv_idx_linear + (ulong) internal * THREAD_BLOCK_SIZE_ul;

            // FEATURE_BLOCK_SIZE = 2 * THREAD_BLOCK_SIZE -> store twice as many values in the shared memory
            data_cache_pp[local_id_1][internal * THREAD_BLOCK_SIZE + local_id_0] = predict_points_d[(dim + threadIdx_y) * (num_predict_points + PADDING_SIZE_ul) + global_pp_idx];
            data_cache_pp[local_id_1 + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + local_id_0] = predict_points_d[(dim + threadIdx_y + THREAD_BLOCK_SIZE_ul) * (num_predict_points + PADDING_SIZE_ul) + global_pp_idx];
            data_cache_sv[local_id_1][internal * THREAD_BLOCK_SIZE + local_id_0] = sv_d[(dim + threadIdx_y) * (num_sv + PADDING_SIZE_ul) + global_sv_idx];
            data_cache_sv[local_id_1 + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + local_id_0] = sv_d[(dim + threadIdx_y + THREAD_BLOCK_SIZE_ul) * (num_sv + PADDING_SIZE_ul) + global_sv_idx];
        }
        barrier(CLK_LOCAL_MEM_FENCE);  // wait until all work-items loaded their part of the data

        // perform the feature reduction calculation
        for (uint block_dim = 0; block_dim < FEATURE_BLOCK_SIZE; ++block_dim) {
            for (uint internal_pd = 0; internal_pd < INTERNAL_BLOCK_SIZE; ++internal_pd) {
                for (uint internal_sv = 0; internal_sv < INTERNAL_BLOCK_SIZE; ++internal_sv) {
                    temp[internal_pd][internal_sv] += PLSSVM_OPENCL_FEATURE_REDUCE_FUNCTION(data_cache_sv[block_dim][local_id_1 * INTERNAL_BLOCK_SIZE + internal_sv], data_cache_pp[block_dim][local_id_0 * INTERNAL_BLOCK_SIZE + internal_pd]);
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);  // wait until all work-items performed their part of the calculations
    }

    // update temp using the respective kernel function
    for (uint internal_pd = 0; internal_pd < INTERNAL_BLOCK_SIZE; ++internal_pd) {
        for (uint internal_sv = 0; internal_sv < INTERNAL_BLOCK_SIZE; ++internal_sv) {
            temp[internal_pd][internal_sv] = PLSSVM_OPENCL_APPLY_KERNEL_FUNCTION(temp[internal_pd][internal_sv] PLSSVM_OPENCL_KERNEL_FUNCTION_PARAMETER);
        }
    }

    {
        // reinterpret cache arrays with interchanged dimensions
        real_type(*alpha_cache)[INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE] = (real_type(*)[INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE]) data_cache_pp;
        real_type(*out_cache)[INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE] = (real_type(*)[INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE]) data_cache_sv;

        // iterate over all features using blocking to be able to cache them for faster memory accesses
        for (ulong dim = 0; dim < num_classes; dim += FEATURE_BLOCK_SIZE_ul) {
            // load data into local memory
            for (uint internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                const ulong global_sv_idx = sv_idx_linear + (ulong) internal * THREAD_BLOCK_SIZE_ul;

                // FEATURE_BLOCK_SIZE = 2 * THREAD_BLOCK_SIZE -> store twice as many values in the shared memory
                alpha_cache[local_id_1][internal * THREAD_BLOCK_SIZE + local_id_0] = alpha_d[(dim + threadIdx_y) * (num_sv + PADDING_SIZE_ul) + global_sv_idx];
                alpha_cache[local_id_1 + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + local_id_0] = alpha_d[(dim + threadIdx_y + THREAD_BLOCK_SIZE_ul) * (num_sv + PADDING_SIZE_ul) + global_sv_idx];

                // the bias (rho) must only be applied once for all support vectors
                if (get_group_id(1) == (ulong) 0) {
                    out_cache[local_id_1][internal * THREAD_BLOCK_SIZE + local_id_0] = -rho_d[dim + threadIdx_y];
                    out_cache[local_id_1 + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + local_id_0] = -rho_d[dim + threadIdx_y + THREAD_BLOCK_SIZE_ul];
                } else {
                    out_cache[local_id_1][internal * THREAD_BLOCK_SIZE + local_id_0] = (real_type) 0.0;
                    out_cache[local_id_1 + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + local_id_0] = (real_type) 0.0;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);  // wait until all work-items loaded their part of the data

            // calculate intermediate results and store them in shared memory
            for (uint class_idx = 0; class_idx < FEATURE_BLOCK_SIZE; ++class_idx) {
                for (uint internal_pd = 0; internal_pd < INTERNAL_BLOCK_SIZE; ++internal_pd) {
                    for (uint internal_sv = 0; internal_sv < INTERNAL_BLOCK_SIZE; ++internal_sv) {
                        out_cache[(class_idx + local_id_1) % FEATURE_BLOCK_SIZE][internal_pd * THREAD_BLOCK_SIZE + local_id_0] +=
                            temp[internal_pd][internal_sv] * alpha_cache[(class_idx + local_id_1) % FEATURE_BLOCK_SIZE][local_id_1 * INTERNAL_BLOCK_SIZE + internal_sv];
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);  // wait until all work-items performed their part of the calculations
            }

            // add intermediate cached results to prediction_d
            for (uint internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                const ulong global_pp_idx = pp_idx + (ulong) internal;

                atomicAdd(&prediction_d[global_pp_idx * (num_classes + PADDING_SIZE_ul) + dim + threadIdx_y], out_cache[local_id_1][internal * THREAD_BLOCK_SIZE + local_id_0]);
                atomicAdd(&prediction_d[global_pp_idx * (num_classes + PADDING_SIZE_ul) + dim + threadIdx_y + THREAD_BLOCK_SIZE_ul], out_cache[local_id_1 + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + local_id_0]);
            }
            barrier(CLK_LOCAL_MEM_FENCE);  // wait until all work-items updated their part of the prediction
        }
    }
}
