/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines CUDA functions for generating the `q` vector.
 */

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

/**
 * @brief Calculates the `q` vector using the linear C-SVM kernel.
 * @details Supports multi-GPU execution.
 * @tparam real_type the type of the data
 * @param[out] q the calculated `q` vector
 * @param[in] data_d the one-dimensional data matrix
 * @param[in] data_last the last row in the data matrix
 * @param[in] num_rows the number of rows in the data matrix
 * @param[in] feature_range number of features used for the calculation
 */
__kernel void device_kernel_q_linear(__global real_type *q, __global real_type *data_d, __global real_type *data_last, const kernel_index_type num_rows, const kernel_index_type feature_range) {
    const kernel_index_type index = get_global_id(0);
    real_type temp = 0.0;
    for (kernel_index_type i = 0; i < feature_range; ++i) {
        temp += data_d[i * num_rows + index] * data_last[i];
    }
    q[index] = temp;
}

/**
 * @brief Calculates the `q` vector using the polynomial C-SVM kernel.
 * @details Currently only single GPU execution is supported.
 * @tparam real_type the type of the data
 * @param[out] q the calculated `q` vector
 * @param[in] data_d the one-dimensional data matrix
 * @param[in] data_last the last row in the data matrix
 * @param[in] num_rows the number of rows in the data matrix
 * @param[in] num_cols the number of columns in the data matrix
 * @param[in] degree the degree parameter used in the polynomial kernel function
 * @param[in] gamma the gamma parameter used in the polynomial kernel function
 * @param[in] coef0 the coef0 parameter used in the polynomial kernel function
 */
__kernel void device_kernel_q_polynomial(__global real_type *q, __global real_type *data_d, __global real_type *data_last, const kernel_index_type num_rows, const kernel_index_type num_cols, const int degree, const real_type gamma, const real_type coef0) {
    const kernel_index_type index = get_global_id(0);
    real_type temp = 0.0;
    for (int i = 0; i < num_cols; ++i) {
        temp += data_d[i * num_rows + index] * data_last[i];
    }
    q[index] = pow(gamma * temp + coef0, degree);
}

/**
 * @brief Calculates the `q` vector using the radial basis functions C-SVM kernel.
 * @details Currently only single GPU execution is supported.
 * @tparam real_type the type of the data
 * @param[out] q the calculated `q` vector
 * @param[in] data_d the one-dimensional data matrix
 * @param[in] data_last the last row in the data matrix
 * @param[in] num_rows the number of rows in the data matrix
 * @param[in] num_cols the number of columns in the data matrix
 * @param[in] gamma the gamma parameter used in the rbf kernel function
 */
__kernel void device_kernel_q_rbf(__global real_type *q, __global real_type *data_d, __global real_type *data_last, const kernel_index_type num_rows, const kernel_index_type num_cols, const real_type gamma) {
    const kernel_index_type index = get_global_id(0);
    real_type temp = 0.0;
    for (kernel_index_type i = 0; i < num_cols; ++i) {
        temp += (data_d[i * num_rows + index] - data_last[i]) * (data_d[i * num_rows + index] - data_last[i]);
    }
    q[index] = exp(-gamma * temp);
}