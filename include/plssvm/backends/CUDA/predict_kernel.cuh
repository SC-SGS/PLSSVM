/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines the functions used for prediction for the C-SVM using the CUDA backend.
 */

#pragma once

#include "plssvm/constants.hpp"  // plssvm::kernel_index_type

namespace plssvm::cuda {

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
template <typename real_type>
__global__ void device_kernel_w_linear(real_type *w_d, const real_type *data_d, const real_type *data_last_d, const real_type *alpha_d, const kernel_index_type num_data_points, const kernel_index_type num_features);

/**
 * @brief Predicts the labels for data points using the polynomial kernel function.
 * @details Currently only single GPU execution is supported.
 * @tparam real_type the type of the data
 * @param[in] out_d the calculated predictions
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
template <typename real_type>
__global__ void device_kernel_predict_poly(real_type *out_d, const real_type *data_d, const real_type *data_last_d, const real_type *alpha_d, const kernel_index_type num_data_points, const real_type *points, const kernel_index_type num_predict_points, const kernel_index_type num_features, const int degree, const real_type gamma, const real_type coef0);

/**
 * @brief Predicts the labels for data points using the radial basis functions kernel function.
 * @details Currently only single GPU execution is supported.
 * @tparam real_type the type of the data
 * @param[in] out_d the calculated predictions
 * @param[in] data_d the one-dimension support vector matrix
 * @param[in] data_last_d the last row of the support vector matrix
 * @param[in] alpha_d the previously calculated weight for each data point
 * @param[in] num_data_points the total number of support vectors
 * @param[in] points the data points to predict
 * @param[in] num_predict_points the total number of data points to predict
 * @param[in] num_features the number of features per support vector and point to predict
 * @param[in] gamma the gamma parameter used in the rbf kernel function
 */
template <typename real_type>
__global__ void device_kernel_predict_radial(real_type *out_d, const real_type *data_d, const real_type *data_last_d, const real_type *alpha_d, const kernel_index_type num_data_points, const real_type *points, const kernel_index_type num_predict_points, const kernel_index_type num_features, const real_type gamma);

}  // namespace plssvm::cuda