/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines the kernel functions for the C-SVM using the OpenMP backend.
 */

#ifndef PLSSVM_BACKENDS_OPENMP_SVM_KERNEL_HPP_
#define PLSSVM_BACKENDS_OPENMP_SVM_KERNEL_HPP_
#pragma once

#include <vector>  // std::vector

namespace plssvm::openmp {

/**
 * @brief Calculates the C-SVM kernel using the linear kernel function.
 * @tparam real_type the type of the data
 * @param[in] q the `q` vector
 * @param[out] ret the result vector
 * @param[in] d the right-hand side of the equation
 * @param[in] data the data matrix
 * @param[in] QA_cost he bottom right matrix entry multiplied by cost
 * @param[in] cost 1 / the cost parameter in the C-SVM
 * @param[in] add denotes whether the values are added or subtracted from the result vector
 */
template <typename real_type>
void device_kernel_linear(const std::vector<real_type> &q, std::vector<real_type> &ret, const std::vector<real_type> &d, const std::vector<std::vector<real_type>> &data, real_type QA_cost, real_type cost, real_type add);

/**
 * @brief Calculates the C-SVM kernel using the polynomial kernel function.
 * @tparam real_type the type of the data
 * @param[in] q the `q` vector
 * @param[out] ret the result vector
 * @param[in] d the right-hand side of the equation
 * @param[in] data the data matrix
 * @param[in] QA_cost the bottom right matrix entry multiplied by cost
 * @param[in] cost 1 / the cost parameter in the C-SVM
 * @param[in] add denotes whether the values are added or subtracted from the result vector
 * @param[in] degree the degree parameter used in the polynomial kernel function
 * @param[in] gamma the gamma parameter used in the polynomial kernel function
 * @param[in] coef0 the coef0 parameter used in the polynomial kernel function
 */
template <typename real_type>
void device_kernel_polynomial(const std::vector<real_type> &q, std::vector<real_type> &ret, const std::vector<real_type> &d, const std::vector<std::vector<real_type>> &data, real_type QA_cost, real_type cost, real_type add, int degree, real_type gamma, real_type coef0);

/**
 * @brief Calculates the C-SVM kernel using the radial basis function kernel function.
 * @tparam real_type the type of the data
 * @param[in] q the `q` vector
 * @param[out] ret the result vector
 * @param[in] d the right-hand side of the equation
 * @param[in] data the data matrix
 * @param[in] QA_cost he bottom right matrix entry multiplied by cost
 * @param[in] cost 1 / the cost parameter in the C-SVM
 * @param[in] add denotes whether the values are added or subtracted from the result vector
 * @param[in] gamma the gamma parameter used in the rbf kernel function
 */
template <typename real_type>
void device_kernel_rbf(const std::vector<real_type> &q, std::vector<real_type> &ret, const std::vector<real_type> &d, const std::vector<std::vector<real_type>> &data, real_type QA_cost, real_type cost, real_type add, real_type gamma);

}  // namespace plssvm::openmp

#endif  // PLSSVM_BACKENDS_OPENMP_SVM_KERNEL_HPP_