/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Functions for explicitly assembling the kernel matrix using the OpenMP backend.
 */

#ifndef PLSSVM_BACKENDS_OPENMP_CG_EXPLICIT_KERNEL_MATRIX_ASSEMBLY_HPP_
#define PLSSVM_BACKENDS_OPENMP_CG_EXPLICIT_KERNEL_MATRIX_ASSEMBLY_HPP_
#pragma once

#include "plssvm/constants.hpp"              // plssvm::real_type, plssvm::OPENMP_BLOCK_SIZE
#include "plssvm/detail/assert.hpp"          // PLSSVM_ASSERT
#include "plssvm/kernel_function_types.hpp"  // plssvm::kernel_function_type
#include "plssvm/kernel_functions.hpp"       // plssvm::kernel_function
#include "plssvm/matrix.hpp"                 // plssvm::aos_matrix

#include <cstddef>  // std::size_t
#include <vector>   // std::vector

namespace plssvm::openmp {

namespace detail {

/**
 * @brief Assemble the kernel matrix using the @p kernel function.
 * @tparam kernel the compile-time kernel function to use
 * @tparam Args the types of the potential additional arguments for the @p kernel function
 * @param[in] q the `q` vector
 * @param[out] ret the resulting kernel matrix
 * @param[in] data the data matrix
 * @param[in] QA_cost he bottom right matrix entry multiplied by cost
 * @param[in] cost 1 / the cost parameter in the C-SVM
 * @param args the potential additional arguments for the @p kernel function
 */
template <kernel_function_type kernel, typename... Args>
void device_kernel_assembly(const std::vector<real_type> &q, std::vector<real_type> &ret, const aos_matrix<real_type> &data, const real_type QA_cost, const real_type cost, Args... args) {
    PLSSVM_ASSERT(q.size() == data.num_rows() - 1, "Sizes mismatch!: {} != {}", q.size(), data.num_rows() - 1);
#if defined(PLSSVM_USE_GEMM)
    PLSSVM_ASSERT(ret.size() == q.size() * q.size(), "Sizes mismatch (GEMM)!: {} != {}", ret.size(), q.size() * q.size());
#else
    PLSSVM_ASSERT(ret.size() == q.size() * (q.size() + 1) / 2, "Sizes mismatch (SYMM)!: {} != {}", ret.size(), q.size() * (q.size() + 1) / 2);
#endif
    PLSSVM_ASSERT(cost != real_type{ 0.0 }, "cost must not be 0.0 since it is 1 / plssvm::cost!");

    const std::size_t dept = q.size();

#pragma omp parallel for collapse(2) schedule(dynamic)
    for (std::size_t row = 0; row < dept; row += OPENMP_BLOCK_SIZE) {
        for (std::size_t col = 0; col < dept; col += OPENMP_BLOCK_SIZE) {
            // perform operations on the current block
            for (std::size_t row_block = 0; row_block < OPENMP_BLOCK_SIZE; ++row_block) {
                for (std::size_t col_block = 0; col_block < OPENMP_BLOCK_SIZE; ++col_block) {
                    const std::size_t row_idx = row + row_block;
                    const std::size_t col_idx = col + col_block;

                    // use symmetry and only calculate upper triangular matrix
                    if (row_idx < dept && col_idx < dept && row_idx <= col_idx) {
                        real_type temp = kernel_function<kernel>(data, row_idx, data, col_idx, args...) + QA_cost - q[row_idx] - q[col_idx];

                        // apply cost to diagonal
                        if (row_idx == col_idx) {
                            temp += cost;
                        }

#if defined(PLSSVM_USE_GEMM)
                        ret[row_idx * dept + col_idx] = temp;
                        ret[col_idx * dept + row_idx] = temp;
#else
                        ret[row_idx * dept + col_idx - row_idx * (row_idx + 1) / 2] = temp;
#endif
                    }
                }
            }
        }
    }
}

}  // namespace detail

/**
 * @brief Assemble the kernel matrix using the linear kernel function.
 * @param[in] q the `q` vector
 * @param[out] ret the resulting kernel matrix
 * @param[in] data the data matrix
 * @param[in] QA_cost he bottom right matrix entry multiplied by cost
 * @param[in] cost 1 / the cost parameter in the C-SVM
 */
inline void device_kernel_assembly_linear(const std::vector<real_type> &q, std::vector<real_type> &ret, const aos_matrix<real_type> &data, const real_type QA_cost, const real_type cost) {
    detail::device_kernel_assembly<kernel_function_type::linear>(q, ret, data, QA_cost, cost);
}

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
inline void device_kernel_assembly_polynomial(const std::vector<real_type> &q, std::vector<real_type> &ret, const aos_matrix<real_type> &data, const real_type QA_cost, const real_type cost, const int degree, const real_type gamma, const real_type coef0) {
    PLSSVM_ASSERT(gamma > real_type{ 0.0 }, "gamma must be greater than 0, but is {}!", gamma);

    detail::device_kernel_assembly<kernel_function_type::polynomial>(q, ret, data, QA_cost, cost, degree, gamma, coef0);
}

/**
 * @brief Assemble the kernel matrix using the radial basis function kernel function.
 * @param[in] q the `q` vector
 * @param[out] ret the resulting kernel matrix
 * @param[in] data the data matrix
 * @param[in] QA_cost he bottom right matrix entry multiplied by cost
 * @param[in] cost 1 / the cost parameter in the C-SVM
 * @param[in] gamma the gamma parameter used in the rbf kernel function
 */
inline void device_kernel_assembly_rbf(const std::vector<real_type> &q, std::vector<real_type> &ret, const aos_matrix<real_type> &data, const real_type QA_cost, const real_type cost, const real_type gamma) {
    PLSSVM_ASSERT(gamma > real_type{ 0.0 }, "gamma must be greater than 0, but is {}!", gamma);

    detail::device_kernel_assembly<kernel_function_type::rbf>(q, ret, data, QA_cost, cost, gamma);
}

}  // namespace plssvm::openmp

#endif  // PLSSVM_BACKENDS_OPENMP_CG_EXPLICIT_KERNEL_MATRIX_ASSEMBLY_HPP_
