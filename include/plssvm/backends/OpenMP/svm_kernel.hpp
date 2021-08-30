/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright
 *
 * @brief Defines the kernel functions for the C-SVM using the OpenMP backend.
 */

#pragma once

#include <vector>  // std::vector

namespace plssvm::openmp {

/**
 * @brief Calculates the C-SVM kernel using the linear kernel function.
 * @tparam real_type the type of the data
 * @param[in] q the `q` vector
 * @param[in] ret the result vector
 * @param[in] d the right-hand side of the equation
 * @param[in] data the data matrix
 * @param[in] QA_cost he bottom right matrix entry multiplied by cost
 * @param[in] cost 1 / the cost parameter in the C-SVM
 * @param[in] add denotes whether the values are added or subtracted from the result vector
 */
template <typename real_type>
void device_kernel_linear(const std::vector<real_type> &q, std::vector<real_type> &ret, const std::vector<real_type> &d, const std::vector<std::vector<real_type>> &data, real_type QA_cost, real_type cost, int add);

/**
 * @brief Calculates the C-SVM kernel using the polynomial kernel function.
 * @tparam real_type the type of the data
 * @param[in] q the `q` vector
 * @param[in] ret the result vector
 * @param[in] d the right-hand side of the equation
 * @param[in] data the data matrix
 * @param[in] QA_cost the bottom right matrix entry multiplied by cost
 * @param[in] cost 1 / the cost parameter in the C-SVM
 * @param[in] add denotes whether the values are added or subtracted from the result vector
 * @param[in] degree the degree parameter used in the polynomial kernel function
 * @param[in] gamma the gamma parameter used in the polynomial kernel function
 * @param[in] coef0 the coef0 parameter used in the polynomial kernel function
 *
 * @attention Currently not implemented!
 */
template <typename real_type>
void device_kernel_poly(const std::vector<real_type> &q, std::vector<real_type> &ret, const std::vector<real_type> &d, const std::vector<std::vector<real_type>> &data, real_type QA_cost, real_type cost, int add, int degree, real_type gamma, real_type coef0);

/**
 * @brief Calculates the C-SVM kernel using the radial basis function kernel function.
 * @tparam real_type the type of the data
 * @param[in] q the `q` vector
 * @param[in] ret the result vector
 * @param[in] d the right-hand side of the equation
 * @param[in] data the data matrix
 * @param[in] QA_cost he bottom right matrix entry multiplied by cost
 * @param[in] cost 1 / the cost parameter in the C-SVM
 * @param[in] add denotes whether the values are added or subtracted from the result vector
 * @param[in] gamma the gamma parameter used in the rbf kernel function
 *
 * @attention Currently not implemented!
 */
template <typename real_type>
void device_kernel_radial(const std::vector<real_type> &q, std::vector<real_type> &ret, const std::vector<real_type> &d, const std::vector<std::vector<real_type>> &data, real_type QA_cost, real_type cost, int add, real_type gamma);

}  // namespace plssvm::openmp