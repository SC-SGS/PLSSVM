/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Functions for performing a matrix-matrix multiplication using an implicit kernel matrix.
 */

#ifndef PLSSVM_BACKENDS_OPENMP_CG_IMPLICIT_KERNEL_MATRIX_ASSEMBLY_BLAS_HPP_
#define PLSSVM_BACKENDS_OPENMP_CG_IMPLICIT_KERNEL_MATRIX_ASSEMBLY_BLAS_HPP_

#include "plssvm/constants.hpp"  // plssvm::real_type
#include "plssvm/matrix.hpp"     // plssvm::aos_matrix

#include <vector>  // std::vector

namespace plssvm::openmp {

void device_kernel_assembly_linear_symm(real_type alpha, const std::vector<real_type> &q, const aos_matrix<real_type> &data, real_type QA_cost, real_type cost, const aos_matrix<real_type> &B, real_type beta, aos_matrix<real_type> &C);

void device_kernel_assembly_polynomial_symm(real_type alpha, const std::vector<real_type> &q, const aos_matrix<real_type> &data, real_type QA_cost, real_type cost, int degree, real_type gamma, real_type coef0, const aos_matrix<real_type> &B, real_type beta, aos_matrix<real_type> &C);

void device_kernel_assembly_rbf_symm(real_type alpha, const std::vector<real_type> &q, const aos_matrix<real_type> &data, real_type QA_cost, real_type cost, real_type gamma, const aos_matrix<real_type> &B, real_type beta, aos_matrix<real_type> &C);

}  // namespace plssvm::openmp

#endif  // PLSSVM_BACKENDS_OPENMP_CG_IMPLICIT_KERNEL_MATRIX_ASSEMBLY_BLAS_HPP_
