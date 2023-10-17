/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Functions used for testing the correctness of the PLSSVM implementation.
 */

#ifndef PLSSVM_TESTS_BACKENDS_COMPARE_HPP_
#define PLSSVM_TESTS_BACKENDS_COMPARE_HPP_
#pragma once

#include "plssvm/matrix.hpp"     // plssvm::matrix, plssvm::layout_type
#include "plssvm/parameter.hpp"  // plssvm::parameter

#include <cstddef>  // std::size_t
#include <vector>   // std::vector

namespace compare {

namespace detail {

/**
 * @brief Compute the value of te two vectors @p x and @p y using the linear kernel function: \f$\vec{x}^T \cdot \vec{y}\f$.
 * @details Adds the final results in a block scheme to mimic the calculations done in the respective backends.
 * @tparam real_type the type of the data
 * @param[in] x the first vector
 * @param[in] y the second vector
 * @param[in] num_devices used to mimic the floating point operation order in case of multi device execution
 * @return the result after applying the kernel function (`[[nodiscard]]`)
 */
template <typename real_type>
[[nodiscard]] real_type linear_kernel(const std::vector<real_type> &x, const std::vector<real_type> &y, std::size_t num_devices = 1);
/**
 * @brief Compute the value of te two vectors @p x and @p y using the polynomial kernel function: \f$(gamma \cdot \vec{x}^T \cdot \vec{y} + coef0)^{degree}\f$.
 * @tparam real_type the type of the data
 * @param[in] x the first vector
 * @param[in] y the second vector
 * @param[in] degree parameter in the polynomial kernel function
 * @param[in] gamma parameter in the polynomial kernel function
 * @param[in] coef0 parameter in the polynomial kernel function
 * @return the result after applying the kernel function (`[[nodiscard]]`)
 */
template <typename real_type>
[[nodiscard]] real_type polynomial_kernel(const std::vector<real_type> &x, const std::vector<real_type> &y, int degree, real_type gamma, real_type coef0);
/**
 * @brief Compute the value of te two vectors @p x and @p y using the radial basis function kernel function: \f$e^{(-gamma \cdot |\vec{x} - \vec{y}|^2)}\f$.
 * @tparam real_type the type of the data
 * @param[in] x the first vector
 * @param[in] y the second vector
 * @param[in] gamma parameter in the kernel function
 * @return the result after applying the rbf kernel function (`[[nodiscard]]`)
 */
template <typename real_type>
[[nodiscard]] real_type rbf_kernel(const std::vector<real_type> &x, const std::vector<real_type> &y, real_type gamma);

template <typename real_type, plssvm::layout_type layout>
[[nodiscard]] real_type linear_kernel(const plssvm::matrix<real_type, layout> &X, std::size_t i, const plssvm::matrix<real_type, layout> &Y, std::size_t j, std::size_t num_devices = 1);

template <typename real_type, plssvm::layout_type layout>
[[nodiscard]] real_type polynomial_kernel(const plssvm::matrix<real_type, layout> &X, std::size_t i, const plssvm::matrix<real_type, layout> &Y, std::size_t j, int degree, real_type gamma, real_type coef0);

template <typename real_type, plssvm::layout_type layout>
[[nodiscard]] real_type rbf_kernel(const plssvm::matrix<real_type, layout> &X, std::size_t i, const plssvm::matrix<real_type, layout> &Y, std::size_t j, real_type gamma);

}  // namespace detail

/**
 * @brief Computes the value of the two vectors @p x and @p y using the parameters @p params.
 * @details Single core execution for a deterministic order of floating point operations.
 * @tparam real_type the type of the data
 * @param[in] params the parameter used in the kernel function
 * @param[in] x the first vector
 * @param[in] y the second vector
 * @param[in] num_devices used to mimic the floating point operation order in case of multi device execution (`[[maybe_unused]]`)
 * @return the result after applying the kernel function (`[[nodiscard]]`)
 */
template <typename real_type>
[[nodiscard]] real_type kernel_function(const plssvm::parameter &params, const std::vector<real_type> &x, const std::vector<real_type> &y, std::size_t num_devices = 1);
template <typename real_type, plssvm::layout_type layout>
[[nodiscard]] real_type kernel_function(const plssvm::parameter &params, const plssvm::matrix<real_type, layout> &X, std::size_t i, const plssvm::matrix<real_type, layout> &Y, std::size_t j, std::size_t num_devices = 1);

/**
 * @brief Computes the `q` vector, a subvector of the least-squares matrix equation, using the kernel function determined by @p params.
 * @details Single core execution for a deterministic order of floating point operations.
 * @tparam real_type the type of the data
 * @param[in] params the parameter used in the kernel function
 * @param[in] data the data points
 * @param[in] num_devices used to mimic the floating point operation order in case of multi device execution (`[[maybe_unused]]`)
 * @return the generated `q` vector (`[[nodiscard]]`)
 */
template <typename real_type>
[[nodiscard]] std::vector<real_type> generate_q(const plssvm::parameter &params, const plssvm::aos_matrix<real_type> &data, std::size_t num_devices = 1);

/**
 * @brief Computes the kernel matrix using the kernel function determined by @p params.
 * @details Single core execution for a deterministic order of floating point operations.
 * @tparam real_type the type of the data
 * @param[in] params the parameter used in the kernel function
 * @param[in] data the data points
 * @param[in] q the `q` vector from the dimensional reduction
 * @param[in] QA_cost the `QA_cost` value from the dimensional reduction
 * @param[in] num_devices used to mimic the floating point operation order in case of multi device execution (`[[maybe_unused]]`)
 * @return the kernel matrix (`[[nodiscard]]`)
 */
template <typename real_type>
[[nodiscard]] std::vector<std::vector<real_type>> assemble_kernel_matrix(const plssvm::parameter &params, const std::vector<std::vector<real_type>> &data, const std::vector<real_type> &q, real_type QA_cost, std::size_t num_devices = 1);

/**
 * @brief Compute the `w` vector used to speedup the prediction when using the linear kernel.
 * @details Single core execution for a deterministic order of floating point operations.
 * @tparam real_type the type of the data
 * @param[in] support_vectors the previously learned support vectors
 * @param[in] weights the previously learned weights
 * @return the resulting `w` vector to speedup the prediction when using the linear kernel (`[[nodiscard]]`)
 */
template <typename real_type, plssvm::layout_type layout>
[[nodiscard]] std::vector<real_type> calculate_w(const plssvm::matrix<real_type, layout> &support_vectors, const std::vector<real_type> &weights);

/**
 * @brief Computes the device kernel, using the kernel function determined by @p params.
 * @details Single core execution for a deterministic order of floating point operations.
 * @tparam real_type the type of the data
 * @param[in] params the parameter used in the kernel function
 * @param[in] data the data points
 * @param[in] rhs the right-hand side of the equation
 * @param[in] q the previously calculated `q` vector
 * @param[in] QA_cost the bottom right matrix entry multiplied by cost
 * @param[in] add denotes whether the values are added or subtracted from the result vector
 * @return the resulting `x` vector of Ax=b (`[[nodiscard]]`)
 */
template <typename real_type>
[[nodiscard]] std::vector<real_type> device_kernel_function(const plssvm::parameter &params, const std::vector<std::vector<real_type>> &data, const std::vector<real_type> &rhs, const std::vector<real_type> &q, real_type QA_cost, real_type add);

}  // namespace compare

#endif  // PLSSVM_TESTS_BACKENDS_COMPARE_HPP_