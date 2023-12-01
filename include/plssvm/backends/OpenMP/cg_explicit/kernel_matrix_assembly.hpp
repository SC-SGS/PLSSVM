/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Functions for explicitly assemblying the kernel matrix using the OpenMP backend.
 */

#ifndef PLSSVM_BACKENDS_OPENMP_CG_EXPLICIT_KERNEL_MATRIX_ASSEMBLY_HPP_
#define PLSSVM_BACKENDS_OPENMP_CG_EXPLICIT_KERNEL_MATRIX_ASSEMBLY_HPP_
#pragma once

#include "plssvm/constants.hpp"  // plssvm::real_type
#include "plssvm/matrix.hpp"     // plssvm::soa_matrix

#include <vector>  // std::vector

namespace plssvm::openmp {

/**
 * @brief Assemble the kernel matrix using the linear kernel function.
 * @param[in] q the `q` vector
 * @param[out] ret the resulting kernel matrix
 * @param[in] data the data matrix
 * @param[in] QA_cost he bottom right matrix entry multiplied by cost
 * @param[in] cost 1 / the cost parameter in the C-SVM
 */
void device_kernel_assembly_linear(const std::vector<real_type> &q, std::vector<real_type> &ret, const ::plssvm::soa_matrix<real_type> &data, real_type QA_cost, real_type cost);

/**
 * @brief Assemble the kernel matrix using the polynomial kernel function.
 * @param[in] q the `q` vector
 * @param[out] ret the resulting kernel matrix
 * @param[in] data the data matrix
 * @param[in] QA_cost the bottom right matrix entry multiplied by cost
 * @param[in] cost 1 / the cost parameter in the C-SVM
 * @param[in] degree the degree parameter used in the polynomial kernel function
 * @param[in] gamma the gamma parameter used in the polynomial kernel function
 * @param[in] coef0 the coef0 parameter used in the polynomial kernel function
 */
void device_kernel_assembly_polynomial(const std::vector<real_type> &q, std::vector<real_type> &ret, const ::plssvm::soa_matrix<real_type> &data, real_type QA_cost, real_type cost, int degree, real_type gamma, real_type coef0);

/**
 * @brief Assemble the kernel matrix using the radial basis function kernel function.
 * @param[in] q the `q` vector
 * @param[out] ret the resulting kernel matrix
 * @param[in] data the data matrix
 * @param[in] QA_cost he bottom right matrix entry multiplied by cost
 * @param[in] cost 1 / the cost parameter in the C-SVM
 * @param[in] gamma the gamma parameter used in the rbf kernel function
 */
void device_kernel_assembly_rbf(const std::vector<real_type> &q, std::vector<real_type> &ret, const ::plssvm::soa_matrix<real_type> &data, real_type QA_cost, real_type cost, real_type gamma);

}  // namespace plssvm::openmp

#endif  // PLSSVM_BACKENDS_OPENMP_CG_EXPLICIT_KERNEL_MATRIX_ASSEMBLY_HPP_