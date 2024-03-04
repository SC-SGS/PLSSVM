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

#ifndef PLSSVM_TESTS_BACKENDS_GROUND_TRUTH_HPP_
#define PLSSVM_TESTS_BACKENDS_GROUND_TRUTH_HPP_
#pragma once

#include "plssvm/detail/data_distribution.hpp"  // plssvm::detail::data_distribution
#include "plssvm/matrix.hpp"                    // plssvm::matrix, plssvm::aos_matrix, plssvm::soa_matrix, plssvm::layout_type
#include "plssvm/parameter.hpp"                 // plssvm::parameter

#include <cstddef>  // std::size_t
#include <utility>  // std::pair
#include <vector>   // std::vector

namespace ground_truth {

//*************************************************************************************************************************************//
//                                                          kernel functions                                                           //
//*************************************************************************************************************************************//

namespace detail {

/**
 * @brief Compute the value of the two vectors @p x and @p y using the linear kernel function: \f$\vec{x}^T \cdot \vec{y}\f$.
 * @details Adds the final results in a block scheme to mimic the calculations done in the respective backends.
 * @tparam real_type the type of the data
 * @param[in] x the first vector
 * @param[in] y the second vector
 * @return the result after applying the kernel function (`[[nodiscard]]`)
 */
template <typename real_type>
[[nodiscard]] real_type linear_kernel(const std::vector<real_type> &x, const std::vector<real_type> &y);
/**
 * @brief Compute the value of the two vectors @p x and @p y using the polynomial kernel function: \f$(gamma \cdot \vec{x}^T \cdot \vec{y} + coef0)^{degree}\f$.
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
 * @brief Compute the value of the two vectors @p x and @p y using the radial basis function kernel function: \f$e^{(-gamma \cdot |\vec{x} - \vec{y}|^2)}\f$.
 * @tparam real_type the type of the data
 * @param[in] x the first vector
 * @param[in] y the second vector
 * @param[in] gamma parameter in the kernel function
 * @return the result after applying the rbf kernel function (`[[nodiscard]]`)
 */
template <typename real_type>
[[nodiscard]] real_type rbf_kernel(const std::vector<real_type> &x, const std::vector<real_type> &y, real_type gamma);

/**
 * @brief Compute the value of the two matrices @p X and @p Y at rows @p i and @p j respectively using the linear kernel function: \f$\vec{x}^T \cdot \vec{y}\f$.
 * @details Adds the final results in a block scheme to mimic the calculations done in the respective backends.
 * @tparam real_type the type of the data
 * @param[in] X the first matrix
 * @param[in] i the row used in the first matrix
 * @param[in] Y the second matrix
 * @param[in] j the row used in the second matrix
 * @return the result after applying the kernel function (`[[nodiscard]]`)
 */
template <typename real_type, plssvm::layout_type layout>
[[nodiscard]] real_type linear_kernel(const plssvm::matrix<real_type, layout> &X, std::size_t i, const plssvm::matrix<real_type, layout> &Y, std::size_t j);
/**
 * @brief Compute the value of the two matrices @p X and @p Y at rows @p i and @p j respectively using the polynomial kernel function: \f$(gamma \cdot \vec{x}^T \cdot \vec{y} + coef0)^{degree}\f$.
 * @tparam real_type the type of the data
 * @param[in] X the first matrix
 * @param[in] i the row used in the first matrix
 * @param[in] Y the second matrix
 * @param[in] j the row used in the second matrix
 * @param[in] degree parameter in the polynomial kernel function
 * @param[in] gamma parameter in the polynomial kernel function
 * @param[in] coef0 parameter in the polynomial kernel function
 * @return the result after applying the kernel function (`[[nodiscard]]`)
 */
template <typename real_type, plssvm::layout_type layout>
[[nodiscard]] real_type polynomial_kernel(const plssvm::matrix<real_type, layout> &X, std::size_t i, const plssvm::matrix<real_type, layout> &Y, std::size_t j, int degree, real_type gamma, real_type coef0);
/**
 * @brief Compute the value of the two matrices @p X and @p Y at rows @p i and @p j respectively using the radial basis function kernel function: \f$e^{(-gamma \cdot |\vec{x} - \vec{y}|^2)}\f$.
 * @tparam real_type the type of the data
 * @param[in] X the first matrix
 * @param[in] i the row used in the first matrix
 * @param[in] Y the second matrix
 * @param[in] j the row used in the second matrix
 * @param[in] gamma parameter in the kernel function
 * @return the result after applying the rbf kernel function (`[[nodiscard]]`)
 */
template <typename real_type, plssvm::layout_type layout>
[[nodiscard]] real_type rbf_kernel(const plssvm::matrix<real_type, layout> &X, std::size_t i, const plssvm::matrix<real_type, layout> &Y, std::size_t j, real_type gamma);

/**
 * @brief Predict the values for the @p predict_points (starting at @p row_offset using @p device_specific_num_rows number of predict points) using the previously learned @p weights and @p support_vectors.
 * @tparam real_type the type of the data
 * @param[in] params the parameters used in the kernel function
 * @param[in] w the `w` vector to speed up the linear kernel function prediction
 * @param[in] weights the previously learned weights
 * @param[in] rho the previous learned bias
 * @param[in] support_vectors the previously learned support vectors
 * @param[in] predict_points the new points to predict
 * @param[in] row_offset the first predict points to consider
 * @param[in] device_specific_num_rows the number of predict points to calculate the predict values for
 * @return the predict values per new predict point and class (`[[nodiscard]]`)
 */
template <typename real_type>
[[nodiscard]] plssvm::aos_matrix<real_type> predict_values(const plssvm::parameter &params, const plssvm::soa_matrix<real_type> &w, const plssvm::aos_matrix<real_type> &weights, const std::vector<real_type> &rho, const plssvm::soa_matrix<real_type> &support_vectors, const plssvm::soa_matrix<real_type> &predict_points, std::size_t row_offset, std::size_t device_specific_num_rows);


}  // namespace detail

/**
 * @brief Computes the value of the two vectors @p x and @p y using the parameters @p params.
 * @details Single core execution for a deterministic order of floating point operations.
 * @tparam real_type the type of the data
 * @param[in] params the parameter used in the kernel function
 * @param[in] x the first vector
 * @param[in] y the second vector
 * @return the result after applying the kernel function (`[[nodiscard]]`)
 */
template <typename real_type>
[[nodiscard]] real_type kernel_function(const plssvm::parameter &params, const std::vector<real_type> &x, const std::vector<real_type> &y);
/**
 * @brief Computes the value of the two matrices @p X and @p Y at rows @p i and @p j respectively using the parameters @p params.
 * @details Single core execution for a deterministic order of floating point operations.
 * @tparam real_type the type of the data
 * @param[in] params the parameter used in the kernel function
 * @param[in] X the first matrix
 * @param[in] i the row used in the first matrix
 * @param[in] Y the second matrix
 * @param[in] j the row used in the second matrix
 * @return the result after applying the kernel function (`[[nodiscard]]`)
 */
template <typename real_type, plssvm::layout_type layout>
[[nodiscard]] real_type kernel_function(const plssvm::parameter &params, const plssvm::matrix<real_type, layout> &X, std::size_t i, const plssvm::matrix<real_type, layout> &Y, std::size_t j);

//*************************************************************************************************************************************//
//                                                                 fit                                                                 //
//*************************************************************************************************************************************//

/**
 * @brief Computes the `q` vector, a subvector of the least-squares matrix equation, using the kernel function determined by @p params for the dimensional reduction.
 * @details Single core execution for a deterministic order of floating point operations.
 * @tparam real_type the type of the data
 * @param[in] params the parameter used in the kernel function
 * @param[in] data the data points
 * @return the generated `q` vector (`[[nodiscard]]`)
 */
template <typename real_type>
[[nodiscard]] std::pair<std::vector<real_type>, real_type> perform_dimensional_reduction(const plssvm::parameter &params, const plssvm::soa_matrix<real_type> &data);

/**
 * @brief Computes the @p device_id specific kernel matrix using the kernel function determined by @p params exploiting the kernel matrix's symmetry.
 * @details Single core execution for a deterministic order of floating point operations.
 * @tparam real_type the type of the data
 * @param[in] params the parameter used in the kernel function
 * @param[in] data the data points
 * @param[in] q the `q` vector from the dimensional reduction
 * @param[in] QA_cost the `QA_cost` value from the dimensional reduction
 * @param[in] dist the data distribution
 * @param[in] device_id the device to emulate
 * @return the kernel matrix (upper triangle matrix) (`[[nodiscard]]`)
 */
template <typename real_type>
[[nodiscard]] std::vector<real_type> assemble_device_specific_kernel_matrix(const plssvm::parameter &params, const plssvm::soa_matrix<real_type> &data, const std::vector<real_type> &q, real_type QA_cost, const plssvm::detail::data_distribution &dist, std::size_t device_id);

/**
 * @brief Computes the **full** kernel matrix using the kernel function determined by @p params.
 * @details Single core execution for a deterministic order of floating point operations.
 * @tparam real_type the type of the data
 * @param[in] params the parameter used in the kernel function
 * @param[in] data the data points
 * @param[in] q the `q` vector from the dimensional reduction
 * @param[in] QA_cost the `QA_cost` value from the dimensional reduction
 * @return the kernel matrix (`[[nodiscard]]`)
 */
template <typename real_type>
[[nodiscard]] plssvm::aos_matrix<real_type> assemble_full_kernel_matrix(const plssvm::parameter &params, const plssvm::soa_matrix<real_type> &data, const std::vector<real_type> &q, real_type QA_cost);

/**
 * @brief Perform a BLAS Level 3 GEMM operation: `C = alpha * A * B + beta * C`
 * @tparam real_type the type of the data
 * @param[in] alpha the scaling factor for the result of `A * B`
 * @param[in] A the A matrix
 * @param[in] B the B matrix
 * @param[in] beta the scaling factor for the C matrix
 * @param[in, out] C the C matrix (also the result matrix)
 */
template <typename real_type>
void gemm(real_type alpha, const plssvm::aos_matrix<real_type> &A, const plssvm::soa_matrix<real_type> &B, real_type beta, plssvm::soa_matrix<real_type> &C);

/**
 * @brief Perform a @p device_id specific BLAS Level 3 GEMM operation: `C = alpha * A * B + beta * C`
 * @tparam real_type the type of the data
 * @param[in] alpha the scaling factor for the result of `A * B`
 * @param[in] A the A matrix
 * @param[in] B the B matrix
 * @param[in] dist the data distribution
 * @param[in] device_id the device to emulate
 * @param[in, out] C the C matrix (also the partial result matrix)
 */
template <typename real_type>
void device_specific_gemm(real_type alpha, const plssvm::aos_matrix<real_type> &A, const plssvm::soa_matrix<real_type> &B, plssvm::soa_matrix<real_type> &C, const plssvm::detail::data_distribution &dist, std::size_t device_id);

//*************************************************************************************************************************************//
//                                                               predict                                                               //
//*************************************************************************************************************************************//

/**
 * @brief Compute the `w` vector used to speedup the prediction when using the linear kernel.
 * @details Single core execution for a deterministic order of floating point operations.
 * @tparam real_type the type of the data
 * @param[in] weights the previously learned weights
 * @param[in] support_vectors the previously learned support vectors
 * @return the resulting `w` vector to speedup the prediction when using the linear kernel (`[[nodiscard]]`)
 */
template <typename real_type>
[[nodiscard]] plssvm::soa_matrix<real_type> calculate_w(const plssvm::aos_matrix<real_type> &weights, const plssvm::soa_matrix<real_type> &support_vectors);

/**
 * @brief Compute the partial `w` vector for the device with @p device_id used to speedup the prediction when using the linear kernel.
 * @details Single core execution for a deterministic order of floating point operations.
 * @tparam real_type the type of the data
 * @param[in] weights the previously learned weights
 * @param[in] support_vectors the previously learned support vectors
 * @param[in] dist the data distribution
 * @param[in] device_id the device to emulate
 * @return the resulting partial `w` vector to speedup the prediction when using the linear kernel (`[[nodiscard]]`)
 */
template <typename real_type>
[[nodiscard]] plssvm::soa_matrix<real_type> calculate_device_specific_w(const plssvm::aos_matrix<real_type> &weights, const plssvm::soa_matrix<real_type> &support_vectors, const plssvm::detail::data_distribution &dist, std::size_t device_id);

/**
 * @brief Predict the values for the @p predict_points using the previously learned @p weights and @p support_vectors.
 * @tparam real_type the type of the data
 * @param[in] params the parameters used in the kernel function
 * @param[in] w the `w` vector to speed up the linear kernel function prediction
 * @param[in] weights the previously learned weights
 * @param[in] rho the previous learned bias
 * @param[in] support_vectors the previously learned support vectors
 * @param[in] predict_points the new points to predict
 * @return the predict values per new predict point and class (`[[nodiscard]]`)
 */
template <typename real_type>
[[nodiscard]] plssvm::aos_matrix<real_type> predict_values(const plssvm::parameter &params, const plssvm::soa_matrix<real_type> &w, const plssvm::aos_matrix<real_type> &weights, const std::vector<real_type> &rho, const plssvm::soa_matrix<real_type> &support_vectors, const plssvm::soa_matrix<real_type> &predict_points);

/**
 * @brief Predict the values for the @p device_id specific @p predict_points according to @p dist using the previously learned @p weights and @p support_vectors.
 * @tparam real_type the type of the data
 * @param[in] params the parameters used in the kernel function
 * @param[in] w the `w` vector to speed up the linear kernel function prediction
 * @param[in] weights the previously learned weights
 * @param[in] rho the previous learned bias
 * @param[in] support_vectors the previously learned support vectors
 * @param[in] predict_points the new points to predict (only the device specific points are really used)
 * @param[in] dist the data distribution
 * @param[in] device_id the device to emulate
 * @return the predict values per @p device_id specific predict point and class (`[[nodiscard]]`)
 */
template <typename real_type>
[[nodiscard]] plssvm::aos_matrix<real_type> predict_device_specific_values(const plssvm::parameter &params, const plssvm::soa_matrix<real_type> &w, const plssvm::aos_matrix<real_type> &weights, const std::vector<real_type> &rho, const plssvm::soa_matrix<real_type> &support_vectors, const plssvm::soa_matrix<real_type> &predict_points, const plssvm::detail::data_distribution &dist, std::size_t device_id);

}  // namespace ground_truth

#endif  // PLSSVM_TESTS_BACKENDS_GROUND_TRUTH_HPP_
