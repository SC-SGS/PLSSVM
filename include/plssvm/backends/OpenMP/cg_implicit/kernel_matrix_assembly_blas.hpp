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

#include "plssvm/constants.hpp"              // plssvm::real_type
#include "plssvm/detail/assert.hpp"          // PLSSVM_ASSERT
#include "plssvm/detail/operators.hpp"       // overloaded arithmetic operations for a plssvm::matrix
#include "plssvm/kernel_function_types.hpp"  // plssvm::kernel_function_type
#include "plssvm/kernel_functions.hpp"       // plssvm::kernel_function
#include "plssvm/matrix.hpp"                 // aos_matrix

#include <cstddef>  // std::size_t
#include <vector>   // std::vector

namespace plssvm::openmp {

namespace detail {

/**
 * @brief Perform an implicit BLAS SYMM-like operation: `C = alpha * A * B + C` where `A` is the implicitly calculated kernel matrix using the @p kernel function (never actually stored, reducing the amount of needed global memory), @p B and @p C are matrices, and @p alpha is a scalar.
 * @tparam kernel the compile-time kernel function to use
 * @tparam Args the types of the potential additional arguments for the @p kernel function
 * @param alpha the scalar alpha value
 * @param[in] q the `q` vector
 * @param[in] data the data matrix
 * @param[in] QA_cost he bottom right matrix entry multiplied by cost
 * @param[in] cost 1 / the cost parameter in the C-SVM
 * @param[in] B the matrix @p B
 * @param[in] beta the beta alpha value
 * @param[in,out] C the matrix @p C
 * @param args the potential additional arguments for the @p kernel function
 */
template <kernel_function_type kernel, typename... Args>
void device_kernel_assembly_symm(const real_type alpha, const std::vector<real_type> &q, const aos_matrix<real_type> &data, const real_type QA_cost, const real_type cost, const aos_matrix<real_type> &B, const real_type beta, aos_matrix<real_type> &C, Args... args) {
    PLSSVM_ASSERT(q.size() == data.num_rows() - 1, "Sizes mismatch!: {} != {}", q.size(), data.num_rows() - 1);
    PLSSVM_ASSERT(cost != real_type{ 0.0 }, "cost must not be 0.0 since it is 1 / plssvm::cost!");
    PLSSVM_ASSERT(B.shape() == C.shape(), "The matrices B and C must have the same shape!");
    PLSSVM_ASSERT(B.num_cols() == q.size(), "The number of columns in B ({}) must be the same as the values in q ({})!", B.num_cols(), q.size());

    using namespace operators;

    const std::size_t dept = q.size();

    // alpha * A * B + beta * C
    C *= beta;

// loop over all rows in the IMPLICIT kernel matrix
#pragma omp parallel for schedule(dynamic)
    for (std::size_t km_row = 0; km_row < dept; ++km_row) {
        // loop over all columns in the IMPLICIT kernel matrix
        // half number of computations by exploiting symmetry
        // NOTE: diagonal NOT included!
        for (std::size_t km_col = km_row + 1; km_col < dept; ++km_col) {
            // calculate kernel matrix entry
            const real_type temp = kernel_function<kernel>(data, km_row, data, km_col, args...) + QA_cost - q[km_row] - q[km_col];

            // calculate the values of alpha * A * B
            for (std::size_t row = 0; row < B.num_rows(); ++row) {
#pragma omp atomic
                C(row, km_row) += alpha * temp * B(row, km_col);
// symmetry
#pragma omp atomic
                C(row, km_col) += alpha * temp * B(row, km_row);
            }
        }
    }
// handle diagonal
#pragma omp parallel for
    for (std::size_t i = 0; i < dept; ++i) {
        const real_type temp = kernel_function<kernel>(data, i, data, i, args...) + cost + QA_cost - q[i] - q[i];

        // calculate the values of alpha * A * B
        for (std::size_t row = 0; row < B.num_rows(); ++row) {
            C(row, i) += alpha * temp * B(row, i);
        }
    }
}

}  // namespace detail

/**
 * @brief Perform an implicit BLAS SYMM-like operation: `C = alpha * A * B + C` where `A` is the implicitly calculated kernel matrix using the linear kernel function \f$\vec{u}^T \cdot \vec{v}\f$ (never actually stored, reducing the amount of needed global memory), @p B and @p C are matrices, and @p alpha is a scalar.
 * @param alpha the scalar alpha value
 * @param[in] q the `q` vector
 * @param[in] data the data matrix
 * @param[in] QA_cost he bottom right matrix entry multiplied by cost
 * @param[in] cost 1 / the cost parameter in the C-SVM
 * @param[in] B the matrix @p B
 * @param[in] beta the beta alpha value
 * @param[in,out] C the matrix @p C
 */
void device_kernel_assembly_linear_symm(const real_type alpha, const std::vector<real_type> &q, const aos_matrix<real_type> &data, const real_type QA_cost, const real_type cost, const aos_matrix<real_type> &B, const real_type beta, aos_matrix<real_type> &C) {
    detail::device_kernel_assembly_symm<kernel_function_type::linear>(alpha, q, data, QA_cost, cost, B, beta, C);
}

/**
 * @brief Perform an implicit BLAS SYMM-like operation: `C = alpha * A * B + C` where `A` is the implicitly calculated kernel matrix using the polynomial kernel function \f$(gamma \cdot \vec{u}^T \cdot \vec{v} + coef0)^{degree}\f$ (never actually stored, reducing the amount of needed global memory), @p B and @p C are matrices, and @p alpha is a scalar.
 * @param alpha the scalar alpha value
 * @param[in] q the `q` vector
 * @param[in] data the data matrix
 * @param[in] QA_cost he bottom right matrix entry multiplied by cost
 * @param[in] cost 1 / the cost parameter in the C-SVM
 * @param[in] degree the degree parameter used in the polynomial kernel function
 * @param[in] gamma the gamma parameter used in the polynomial kernel function
 * @param[in] coef0 the coef0 parameter used in the polynomial kernel function
 * @param[in] B the matrix @p B
 * @param[in] beta the beta alpha value
 * @param[in,out] C the matrix @p C
 */
void device_kernel_assembly_polynomial_symm(const real_type alpha, const std::vector<real_type> &q, const aos_matrix<real_type> &data, const real_type QA_cost, const real_type cost, const int degree, const real_type gamma, const real_type coef0, const aos_matrix<real_type> &B, const real_type beta, aos_matrix<real_type> &C) {
    PLSSVM_ASSERT(gamma > real_type{ 0.0 }, "gamma must be greater than 0, but is {}!", gamma);

    detail::device_kernel_assembly_symm<kernel_function_type::polynomial>(alpha, q, data, QA_cost, cost, B, beta, C, degree, gamma, coef0);
}

/**
 * @brief Perform an implicit BLAS SYMM-like operation: `C = alpha * A * B + C` where `A` is the implicitly calculated kernel matrix using the rbf kernel function \f$e^{(-gamma \cdot |\vec{u} - \vec{v}|^2)}\f$ (never actually stored, reducing the amount of needed global memory), @p B and @p C are matrices, and @p alpha is a scalar.
 * @param alpha the scalar alpha value
 * @param[in] q the `q` vector
 * @param[in] data the data matrix
 * @param[in] QA_cost he bottom right matrix entry multiplied by cost
 * @param[in] cost 1 / the cost parameter in the C-SVM
 * @param[in] gamma the gamma parameter used in the polynomial kernel function
 * @param[in] B the matrix @p B
 * @param[in] beta the beta alpha value
 * @param[in,out] C the matrix @p C
 */
void device_kernel_assembly_rbf_symm(const real_type alpha, const std::vector<real_type> &q, const aos_matrix<real_type> &data, const real_type QA_cost, const real_type cost, const real_type gamma, const aos_matrix<real_type> &B, const real_type beta, aos_matrix<real_type> &C) {
    PLSSVM_ASSERT(gamma > real_type{ 0.0 }, "gamma must be greater than 0, but is {}!", gamma);

    detail::device_kernel_assembly_symm<kernel_function_type::rbf>(alpha, q, data, QA_cost, cost, B, beta, C, gamma);
}

}  // namespace plssvm::openmp

#endif  // PLSSVM_BACKENDS_OPENMP_CG_IMPLICIT_KERNEL_MATRIX_ASSEMBLY_BLAS_HPP_
