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

#ifndef PLSSVM_BACKENDS_STDPAR_KERNEL_CG_IMPLICIT_KERNEL_MATRIX_ASSEMBLY_BLAS_HPP_
#define PLSSVM_BACKENDS_STDPAR_KERNEL_CG_IMPLICIT_KERNEL_MATRIX_ASSEMBLY_BLAS_HPP_
#pragma once

#include "plssvm/constants.hpp"              // plssvm::real_type
#include "plssvm/detail/assert.hpp"          // PLSSVM_ASSERT
#include "plssvm/detail/operators.hpp"       // overloaded arithmetic operations for a plssvm::matrix
#include "plssvm/kernel_function_types.hpp"  // plssvm::kernel_function_type
#include "plssvm/kernel_functions.hpp"       // plssvm::kernel_function
#include "plssvm/matrix.hpp"                 // aos_matrix

#include <atomic>   // std::atomic_ref
#include <cstddef>  // std::size_t
#include <execution>
#include <ranges>
#include <vector>  // std::vector

// TODO: correct macro name!?
#if defined(PLSSVM_STDPAR_BACKEND_USE_ADAPTIVECPP)
    #include "plssvm/backens/SYCL/detail/atomics.hpp"  // atomic_op
using plssvm::sycl::detail::atomic_op;
#else
template <typename T>
using atomic_op = std::atomic_ref<T>;
#endif

namespace plssvm::stdpar::detail {

/**
 * @brief Perform an implicit BLAS SYMM-like operation: `C = alpha * A * B + C` where `A` is the implicitly calculated kernel matrix using the @p kernel function (never actually stored, reducing the amount of needed global memory), @p B and @p C are matrices, and @p alpha is a scalar.
 * @tparam kernel the compile-time kernel function to use
 * @tparam layout the compile-time layout type for the matrices
 * @tparam Args the types of the potential additional arguments for the @p kernel function
 * @param[in] alpha the scalar alpha value
 * @param[in] q the `q` vector
 * @param[in] data the data matrix
 * @param[in] QA_cost he bottom right matrix entry multiplied by cost
 * @param[in] cost 1 / the cost parameter in the C-SVM
 * @param[in] B the matrix @p B
 * @param[in] beta the beta alpha value
 * @param[in,out] C the matrix @p C
 * @param[in] args the potential additional arguments for the @p kernel function
 */
template <kernel_function_type kernel, layout_type layout, typename... Args>
inline void device_kernel_assembly_symm(const real_type alpha, const std::vector<real_type> &q, const matrix<real_type, layout> &data, const real_type QA_cost, const real_type cost, const matrix<real_type, layout> &B, const real_type beta, matrix<real_type, layout> &C, Args... args) {
    PLSSVM_ASSERT(q.size() == data.num_rows() - 1, "Sizes mismatch!: {} != {}", q.size(), data.num_rows() - 1);
    PLSSVM_ASSERT(cost != real_type{ 0.0 }, "cost must not be 0.0 since it is 1 / plssvm::cost!");
    PLSSVM_ASSERT(B.shape() == C.shape(), "The matrices B and C must have the same shape!");
    PLSSVM_ASSERT(B.num_cols() == q.size(), "The number of columns in B ({}) must be the same as the values in q ({})!", B.num_cols(), q.size());

    using namespace operators;

    const std::size_t dept = q.size();

    // alpha * A * B + beta * C
    C *= beta;

    const auto is = std::views::cartesian_product(
        std::views::iota(std::size_t{ 0 }, dept),
        std::views::iota(std::size_t{ 0 }, dept));

    std::for_each(std::execution::par_unseq, is.begin(), is.end(), [&](auto i) {
        const auto [km_row_idx, km_col_idx] = i;

        // half number of computations by exploiting symmetry
        if (km_row_idx <= km_col_idx) {
            real_type temp = kernel_function<kernel>(data, km_row_idx, data, km_col_idx, args...) + QA_cost - q[km_row_idx] - q[km_col_idx];

            // apply cost to diagonal
            if (km_row_idx == km_col_idx) {
                temp += cost;
                // calculate the values of alpha * A * B
                for (std::size_t row = 0; row < B.num_rows(); ++row) {
                    atomic_op<real_type>{ C(row, km_row_idx) } += alpha * temp * B(row, km_row_idx);
                }
            } else {
                // calculate the values of alpha * A * B
                for (std::size_t row = 0; row < B.num_rows(); ++row) {
                    atomic_op<real_type>{ C(row, km_row_idx) } += alpha * temp * B(row, km_col_idx);
                    // symmetry
                    atomic_op<real_type>{ C(row, km_col_idx) } += alpha * temp * B(row, km_row_idx);
                }
            }
        }
    });
}

}  // namespace plssvm::stdpar::detail

#endif  // PLSSVM_BACKENDS_STDPAR_KERNEL_CG_IMPLICIT_KERNEL_MATRIX_ASSEMBLY_BLAS_HPP_
