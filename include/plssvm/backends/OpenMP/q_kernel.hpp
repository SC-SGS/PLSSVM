/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines OpenMP functions for generating the `q` vector.
 */

#ifndef PLSSVM_BACKENDS_OPENMP_Q_KERNEL_HPP_
#define PLSSVM_BACKENDS_OPENMP_Q_KERNEL_HPP_
#pragma once

#include <vector>  // std::vector

namespace plssvm::openmp {

/**
 * @brief Calculates the `q` vector using the linear C-SVM kernel.
 * @tparam real_type the type of the data
 * @param[out] q the calculated `q` vector
 * @param[in] data the two-dimensional data matrix
 */
template <typename real_type>
void device_kernel_q_linear(std::vector<real_type> &q, const std::vector<std::vector<real_type>> &data);

/**
 * @brief Calculates the `q` vector using the polynomial C-SVM kernel.
 * @tparam real_type the type of the data
 * @param[out] q the calculated `q` vector
 * @param[in] data the two-dimensional data matrix
 * @param[in] degree the degree parameter used in the polynomial kernel function
 * @param[in] gamma the gamma parameter used in the polynomial kernel function
 * @param[in] coef0 the coef0 parameter used in the polynomial kernel function
 */
template <typename real_type>
void device_kernel_q_poly(std::vector<real_type> &q, const std::vector<std::vector<real_type>> &data, int degree, real_type gamma, real_type coef0);

/**
 * @brief Calculates the `q` vector using the radial basis functions C-SVM kernel.
 * @tparam real_type the type of the data
 * @param[out] q the calculated `q` vector
 * @param[in] data the two-dimensional data matrix
 * @param[in] gamma the gamma parameter used in the rbf kernel function
 */
template <typename real_type>
void device_kernel_q_radial(std::vector<real_type> &q, const std::vector<std::vector<real_type>> &data, real_type gamma);

}  // namespace plssvm::openmp

#endif  // PLSSVM_BACKENDS_OPENMP_Q_KERNEL_HPP_