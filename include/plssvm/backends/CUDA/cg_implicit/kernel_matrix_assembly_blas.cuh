/**
* @file
* @author Alexander Van Craen
* @author Marcel Breyer
* @copyright 2018-today The PLSSVM project - All Rights Reserved
* @license This file is part of the PLSSVM project which is released under the MIT license.
*          See the LICENSE.md file in the project root for full license information.
*
* @brief Functions for implicitly assembling the kernel matrix using the CUDA backend.
*/

#ifndef PLSSVM_BACKENDS_CUDA_CG_IMPLICIT_KERNEL_MATRIX_ASSEMBLY_BLAS_CUH_
#define PLSSVM_BACKENDS_CUDA_CG_IMPLICIT_KERNEL_MATRIX_ASSEMBLY_BLAS_CUH_
#pragma once

#include "plssvm/constants.hpp"  // plssvm::real_type

namespace plssvm::cuda {

/**
 * @brief Perform an implicit BLAS SYMM-like operation: `C = alpha * A * B + C` where `A` is the implicitly calculated kernel matrix using the linear kernel function \f$\vec{u}^T \cdot \vec{v}\f$ (never actually stored, reducing the amount of needed global memory), @p B and @p C are matrices, and @p alpha is a scalar.
 * @note The beta factor is already applied to C before this kernel starts!
 * @param[in] alpha the scalar alpha value
 * @param[in] q the vector used in the dimensional reduction
 * @param[in] data_d the data points to calculate the implicit kernel matrix from
 * @param[in] num_rows the number of data points
 * @param[in] num_features the number of features per data point
 * @param[in] QA_cost the scalar used in the dimensional reduction
 * @param[in] cost the cost factor the diagonal is scaled with
 * @param[in] B the matrix @p B
 * @param[in,out] C the matrix @p C
 * @param[in] num_classes the number of classes in the data set
 */
__global__ void device_kernel_assembly_linear_symm(const real_type alpha, const real_type *q, const real_type *data_d, const unsigned long long num_rows, const unsigned long long num_features, const real_type QA_cost, const real_type cost, const real_type *B, real_type *C, const unsigned long long num_classes);

/**
 * @brief Perform an implicit BLAS SYMM-like operation: `C = alpha * A * B + C` where `A` is the implicitly calculated kernel matrix using the polynomial kernel function \f$(gamma \cdot \vec{u}^T \cdot \vec{v} + coef0)^{degree}\f$ (never actually stored, reducing the amount of needed global memory), @p B and @p C are matrices, and @p alpha is a scalar.
 * @note The beta factor is already applied to C before this kernel starts!
 * @param[in] alpha the scalar alpha value
 * @param[in] q the vector used in the dimensional reduction
 * @param[in] data_d the data points to calculate the implicit kernel matrix from
 * @param[in] num_rows the number of data points
 * @param[in] num_features the number of features per data point
 * @param[in] QA_cost the scalar used in the dimensional reduction
 * @param[in] cost the cost factor the diagonal is scaled with
 * @param[in] degree parameter used in the polynomial kernel function
 * @param[in] gamma parameter used in the polynomial kernel function
 * @param[in] coef0 parameter used in the polynomial kernel function
 * @param[in] B the matrix @p B
 * @param[in,out] C the matrix @p C
 * @param[in] num_classes the number of classes in the data set
 */
__global__ void device_kernel_assembly_polynomial_symm(const real_type alpha, const real_type *q, const real_type *data_d, const unsigned long long num_rows, const unsigned long long num_features, const real_type QA_cost, const real_type cost, const int degree, const real_type gamma, const real_type coef0, const real_type *B, real_type *C, const unsigned long long num_classes);

/**
 * @brief Perform an implicit BLAS SYMM-like operation: `C = alpha * A * B + C` where `A` is the implicitly calculated kernel matrix using the rbf kernel function \f$e^{(-gamma \cdot |\vec{u} - \vec{v}|^2)}\f$ (never actually stored, reducing the amount of needed global memory), @p B and @p C are matrices, and @p alpha is a scalar.
 * @note The beta factor is already applied to C before this kernel starts!
 * @param[in] alpha the scalar alpha value
 * @param[in] q the vector used in the dimensional reduction
 * @param[in] data_d the data points to calculate the implicit kernel matrix from
 * @param[in] num_rows the number of data points
 * @param[in] num_features the number of features per data point
 * @param[in] QA_cost the scalar used in the dimensional reduction
 * @param[in] cost the cost factor the diagonal is scaled with
 * @param[in] gamma parameter used in the rbf kernel function
 * @param[in] B the matrix @p B
 * @param[in,out] C the matrix @p C
 * @param[in] num_classes the number of classes in the data set
 */
__global__ void device_kernel_assembly_rbf_symm(const real_type alpha, const real_type *q, const real_type *data_d, const unsigned long long num_rows, const unsigned long long num_features, const real_type QA_cost, const real_type cost, const real_type gamma, const real_type *B, real_type *C, const unsigned long long num_classes);


}  // namespace plssvm::cuda

#endif  // PLSSVM_BACKENDS_CUDA_CG_IMPLICIT_KERNEL_MATRIX_ASSEMBLY_BLAS_CUH_
