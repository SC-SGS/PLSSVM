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

#include "plssvm/constants.hpp"  // plssvm::kernel_index_type

namespace plssvm::cuda {

template <typename real_type>
__global__ void device_kernel_w_linear(real_type *w_d, const real_type *alpha_d, const real_type *sv_d, const kernel_index_type num_classes, const kernel_index_type num_sv, const kernel_index_type num_features);

template <typename real_type>
__global__ void device_kernel_predict_linear(real_type *out_d, const real_type *w_d, const real_type *rho_d, const real_type *predict_points_d, const kernel_index_type num_classes, const kernel_index_type num_predict_points, const kernel_index_type num_features);

template <typename real_type>
__global__ void device_kernel_predict_polynomial(real_type *out_d, const real_type *alpha_d, const real_type *rho_d, const real_type *sv_d, const real_type *predict_points_d, const kernel_index_type num_classes, const kernel_index_type num_sv, const kernel_index_type num_predict_points, const kernel_index_type num_features, const int degree, const real_type gamma, const real_type coef0);

template <typename real_type>
__global__ void device_kernel_predict_rbf(real_type *out_d, const real_type *alpha_d, const real_type *rho_d, const real_type *sv_d, const real_type *predict_points_d, const kernel_index_type num_classes, const kernel_index_type num_sv, const kernel_index_type num_predict_points, const kernel_index_type num_features, const real_type gamma);


}  // namespace plssvm::cuda

#endif  // PLSSVM_BACKENDS_CUDA_PREDICT_KERNEL_HPP_