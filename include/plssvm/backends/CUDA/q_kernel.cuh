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

#pragma once

namespace plssvm::cuda {

/**
 * @brief Calculates the `q` vector using the linear C-SVM kernel.
 * @details Supports multi-GPU execution.
 * @tparam real_type the type of the data
 * @param[out] q the calculated `q` vector
 * @param[in] data_d the one-dimensional data matrix
 * @param[in] data_last the last row in the data matrix
 * @param[in] num_rows the number of rows in the data matrix
 * @param[in] first_feature the first feature used in the calculations (depending on the current device)
 * @param[in] last_feature the last feature used in the calculations (depending on the current device)
 */
template <typename real_type>
__global__ void device_kernel_q_linear(real_type *q, const real_type *data_d, const real_type *data_last, const int num_rows, const int first_feature, const int last_feature);

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
template <typename real_type>
__global__ void device_kernel_q_poly(real_type *q, const real_type *data_d, const real_type *data_last, const int num_rows, const int num_cols, const int degree, const real_type gamma, const real_type coef0);

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
template <typename real_type>
__global__ void device_kernel_q_radial(real_type *q, const real_type *data_d, const real_type *data_last, const int num_rows, const int num_cols, const real_type gamma);

}  // namespace plssvm::cuda