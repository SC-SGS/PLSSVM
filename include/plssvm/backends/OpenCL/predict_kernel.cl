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
 * @param[in] num_features the number of features per support vector
 */
__kernel void device_kernel_w_linear(__global real_type *w_d, __global const real_type *alpha_d, __global const real_type *sv_d, const ulong num_classes, const ulong num_sv, const ulong num_features) {
    const ulong feature_idx = get_global_id(0);
    const ulong class_idx = get_global_id(1);

    if (feature_idx < num_features && class_idx < num_classes) {
        real_type temp = 0.0;
        for (ulong sv = 0; sv < num_sv; ++sv) {
            temp += alpha_d[class_idx * num_sv + sv] * sv_d[sv * num_features + feature_idx];
        }
        w_d[class_idx * num_features + feature_idx] = temp;
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
    const ulong predict_point_idx = get_global_id(0);
    const ulong class_idx = get_global_id(1);

    if (predict_point_idx < num_predict_points && class_idx < num_classes) {
        real_type temp = 0.0;
        for (ulong dim = 0; dim < num_features; ++dim) {
            temp += w_d[class_idx * num_features + dim] * predict_points_d[predict_point_idx * num_features + dim];
        }
        out_d[predict_point_idx * num_classes + class_idx] = temp - rho_d[class_idx];
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
    const ulong sv_idx = get_global_id(0);
    const ulong predict_points_idx = get_global_id(1);
    const ulong class_idx = get_global_id(2);

    if (sv_idx < num_sv && predict_points_idx < num_predict_points && class_idx < num_classes) {
        real_type temp = 0.0;
        // perform dot product
        for (ulong dim = 0; dim < num_features; ++dim) {
            temp += sv_d[sv_idx * num_features + dim] * predict_points_d[predict_points_idx * num_features + dim];
        }

        // apply degree, gamma, and coef0, alpha and rho
        temp = alpha_d[class_idx * num_sv + sv_idx] * pow(gamma * temp + coef0, degree);
        if (sv_idx == 0) {
            temp -= rho_d[class_idx];
        }

        atomicAdd(&out_d[predict_points_idx * num_classes + class_idx], temp);
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
    const ulong sv_idx = get_global_id(0);
    const ulong predict_points_idx = get_global_id(1);
    const ulong class_idx = get_global_id(2);

    if (sv_idx < num_sv && predict_points_idx < num_predict_points && class_idx < num_classes) {
        real_type temp = 0.0;
        // perform dist calculation
        for (ulong dim = 0; dim < num_features; ++dim) {
            const real_type diff = sv_d[sv_idx * num_features + dim] - predict_points_d[predict_points_idx * num_features + dim];
            temp += diff * diff;
        }

        // apply degree, gamma, and coef0, alpha and rho
        temp = alpha_d[class_idx * num_sv + sv_idx] * exp(-gamma * temp);
        if (sv_idx == 0) {
            temp -= rho_d[class_idx];
        }

        atomicAdd(&out_d[predict_points_idx * num_classes + class_idx], temp);
    }
}