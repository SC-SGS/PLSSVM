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

/**
 * @brief Calculate the `w` vector to speed up the prediction of the labels for data points using the linear kernel function.
 * @details Supports multi-GPU execution.
 * @tparam real_type the type of the data
 * @param[out] w_d the `w` vector to assemble
 * @param[in] data_d the one-dimension support vector matrix
 * @param[in] data_last_d the last row of the support vector matrix
 * @param[in] alpha_d the previously calculated weight for each data point
 * @param[in] num_data_points the total number of support vectors
 * @param[in] num_features the number of features per support vector
 */
__kernel void device_kernel_w_linear(__global real_type *w_d, __global real_type *data_d, __global real_type *data_last_d, __global real_type *alpha_d, const kernel_index_type num_data_points, const kernel_index_type num_features) {
    const kernel_index_type index = get_global_id(0);
    real_type temp = 0.0;
    if (index < num_features) {
        for (kernel_index_type dat = 0; dat < num_data_points - 1; ++dat) {
            temp += alpha_d[dat] * data_d[dat + (num_data_points - 1 + THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE) * index];
        }
        temp += alpha_d[num_data_points - 1] * data_last_d[index];
        w_d[index] = temp;
    }
}

/**
 * @brief Predicts the labels for data points using the polynomial kernel function.
 * @details Currently only single GPU execution is supported.
 * @tparam real_type the type of the data
 * @param[out] out_d the calculated predictions
 * @param[in] data_d the one-dimension support vector matrix
 * @param[in] data_last_d the last row of the support vector matrix
 * @param[in] alpha_d the previously calculated weight for each data point
 * @param[in] num_data_points the total number of support vectors
 * @param[in] points the data points to predict
 * @param[in] num_predict_points the total number of data points to predict
 * @param[in] num_features the number of features per support vector and point to predict
 * @param[in] degree the degree parameter used in the polynomial kernel function
 * @param[in] gamma the gamma parameter used in the polynomial kernel function
 * @param[in] coef0 the coef0 parameter used in the polynomial kernel function
 */
__kernel void device_kernel_predict_polynomial(__global real_type *out_d, __global const real_type *data_d, __global const real_type *data_last_d, __global const real_type *alpha_d, const kernel_index_type num_data_points, __global const real_type *points, const kernel_index_type num_predict_points, const kernel_index_type num_features, const int degree, const real_type gamma, const real_type coef0) {
    const kernel_index_type data_point_index = get_global_id(0);
    const kernel_index_type predict_point_index = get_global_id(1);

    real_type temp = 0.0;
    if (predict_point_index < num_predict_points) {
        for (kernel_index_type feature_index = 0; feature_index < num_features; ++feature_index) {
            if (data_point_index == num_data_points - 1) {
                temp += data_last_d[feature_index] * points[predict_point_index + (num_predict_points + THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE) * feature_index];
            } else {
                temp += data_d[data_point_index + (num_data_points - 1 + THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE) * feature_index] * points[predict_point_index + (num_predict_points + THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE) * feature_index];
            }
        }

        temp = alpha_d[data_point_index] * pow(gamma * temp + coef0, degree);
        atomicAdd(&out_d[predict_point_index], temp);
    }
}

/**
 * @brief Predicts the labels for data points using the radial basis functions kernel function.
 * @details Currently only single GPU execution is supported.
 * @tparam real_type the type of the data
 * @param[out] out_d the calculated predictions
 * @param[in] data_d the one-dimension support vector matrix
 * @param[in] data_last_d the last row of the support vector matrix
 * @param[in] alpha_d the previously calculated weight for each data point
 * @param[in] num_data_points the total number of support vectors
 * @param[in] points the data points to predict
 * @param[in] num_predict_points the total number of data points to predict
 * @param[in] num_features the number of features per support vector and point to predict
 * @param[in] gamma the gamma parameter used in the rbf kernel function
 */
__kernel void device_kernel_predict_rbf(__global real_type *out_d, __global const real_type *data_d, __global const real_type *data_last_d, __global const real_type *alpha_d, const kernel_index_type num_data_points, __global const real_type *points, const kernel_index_type num_predict_points, const kernel_index_type num_features, const real_type gamma) {
    const kernel_index_type data_point_index = get_global_id(0);
    const kernel_index_type predict_point_index = get_global_id(1);

    real_type temp = 0.0;
    if (predict_point_index < num_predict_points) {
        for (kernel_index_type feature_index = 0; feature_index < num_features; ++feature_index) {
            if (data_point_index == num_data_points - 1) {
                temp += (data_last_d[feature_index] - points[predict_point_index + (num_predict_points + THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE) * feature_index]) * (data_last_d[feature_index] - points[predict_point_index + (num_predict_points + THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE) * feature_index]);
            } else {
                temp += (data_d[data_point_index + (num_data_points - 1 + THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE) * feature_index] - points[predict_point_index + (num_predict_points + THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE) * feature_index]) * (data_d[data_point_index + (num_data_points - 1 + THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE) * feature_index] - points[predict_point_index + (num_predict_points + THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE) * feature_index]);
            }
        }

        temp = alpha_d[data_point_index] * exp(-gamma * temp);
        atomicAdd(&out_d[predict_point_index], temp);
    }
}