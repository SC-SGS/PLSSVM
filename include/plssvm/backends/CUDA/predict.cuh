/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief TODO: brief description
 */

#pragma once

namespace plssvm::cuda {
template <typename real_type>
__global__ void kernel_w(real_type *w_d, const real_type *data_d, const real_type *data_last_d, const real_type *alpha_d, const int num_data_points, const int num_features);

template <typename real_type>
__global__ void predict_points_poly(real_type *out_d, const real_type *data_d, const real_type *data_last_d, const real_type *alpha_d, const int num_data_points, const real_type *points, const int num_predict_points, const int num_features, const int degree, const real_type gamma, const real_type coef0);

template <typename real_type>
__global__ void predict_points_rbf(real_type *out_d, const real_type *data_d, const real_type *data_last_d, const real_type *alpha_d, const int num_data_points, const real_type *points, const int num_predict_points, const int num_features, const real_type gamma);

}  // namespace plssvm::cuda