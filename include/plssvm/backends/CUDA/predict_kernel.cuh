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

#ifndef PLSSVM_BACKENDS_CUDA_PREDICT_KERNEL_HPP_
#define PLSSVM_BACKENDS_CUDA_PREDICT_KERNEL_HPP_
#pragma once

#include "plssvm/constants.hpp"  // plssvm::real_type

namespace plssvm::cuda {

/**
 * @brief Calculate the `q` vector used to speedup the prediction using the linear kernel function.
 * @param[in,out] w_d the vector to speedup the linear prediction
 * @param[in] alpha_d the previously learned weights
 * @param[in] sv_d the support vectors
 * @param[in] num_classes the number of classes
 * @param[in] num_sv the number of support vectors
 */
__global__ void device_kernel_w_linear(real_type *w_d, const real_type *alpha_d, const real_type *sv_d, const unsigned long long num_classes, const unsigned long long num_sv);

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
__global__ void device_kernel_predict_linear(real_type *out_d, const real_type *w_d, const real_type *rho_d, const real_type *predict_points_d, const unsigned long long num_classes, const unsigned long long num_predict_points, const unsigned long long num_features);

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
__global__ void device_kernel_predict_polynomial(real_type *out_d, const real_type *alpha_d, const real_type *rho_d, const real_type *sv_d, const real_type *predict_points_d, const unsigned long long num_classes, const unsigned long long num_sv, const unsigned long long num_predict_points, const unsigned long long num_features, const int degree, const real_type gamma, const real_type coef0);

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
__global__ void device_kernel_predict_rbf(real_type *out_d, const real_type *alpha_d, const real_type *rho_d, const real_type *sv_d, const real_type *predict_points_d, const unsigned long long num_classes, const unsigned long long num_sv, const unsigned long long num_predict_points, const unsigned long long num_features, const real_type gamma);

}  // namespace plssvm::cuda

#endif  // PLSSVM_BACKENDS_CUDA_PREDICT_KERNEL_HPP_