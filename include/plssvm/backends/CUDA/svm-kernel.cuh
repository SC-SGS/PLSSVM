/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright
 *
 * @brief Defines a C-SVM using the CUDA backend.
 */

#pragma once

namespace plssvm::cuda {

/**
 * @brief Calculates the C-SVM kernel using the linear kernel function.
 * @details Supports multi-GPU execution.
 * @tparam real_type the type of the data
 * @param[in] q the `q` vector
 * @param[in] ret the result vector
 * @param[in] d the right-hand side of the equation
 * @param[in] data_d the one-dimension data matrix
 * @param[in] QA_cost he bottom right matrix entry multiplied by cost
 * @param[in] cost 1 / the cost parameter in the C-SVM
 * @param[in] Nrows the number of columns in the data matrix
 * @param[in] add denotes whether the values are added or subtracted from the result vector
 * @param[in] start the first feature used in the calculations (depending on the current device)
 * @param[in] end the last feature used in the calculations (depending on the current device)
 */
template <typename real_type>
__global__ void kernel_linear(const real_type *q, real_type *ret, const real_type *d, const real_type *data_d, const real_type QA_cost, const real_type cost, const int Nrows, const int add, const int start, const int end);

/**
 * @brief Calculates the C-SVM kernel using the polynomial kernel function.
 * @details Currently only single GPU execution is supported.
 * @tparam real_type the type of the data
 * @param[in] q the `q` vector
 * @param[in] ret the result vector
 * @param[in] d the right-hand side of the equation
 * @param[in] data_d the one-dimension data matrix
 * @param[in] QA_cost he bottom right matrix entry multiplied by cost
 * @param[in] cost 1 / the cost parameter in the C-SVM
 * @param[in] Ncols the number of rows in the data matrix
 * @param[in] Nrows the number of columns in the data matrix
 * @param[in] add denotes whether the values are added or subtracted from the result vector
 * @param[in] degree the degree parameter used in the polynomial kernel function
 * @param[in] gamma the gamma parameter used in the polynomial kernel function
 * @param[in] coef0 the coef0 parameter used in the polynomial kernel function
 */
template <typename real_type>
__global__ void kernel_poly(const real_type *q, real_type *ret, const real_type *d, const real_type *data_d, const real_type QA_cost, const real_type cost, const int Ncols, const int Nrows, const int add, const real_type gamma, const real_type coef0, const real_type degree);

/**
 * @brief Calculates the C-SVM kernel using the radial basis function kernel function.
 * @details Currently only single GPU execution is supported.
 * @tparam real_type the type of the data
 * @param[in] q the `q` vector
 * @param[in] ret the result vector
 * @param[in] d the right-hand side of the equation
 * @param[in] data_d the one-dimension data matrix
 * @param[in] QA_cost he bottom right matrix entry multiplied by cost
 * @param[in] cost 1 / the cost parameter in the C-SVM
 * @param[in] Ncols the number of rows in the data matrix
 * @param[in] Nrows the number of columns in the data matrix
 * @param[in] add denotes whether the values are added or subtracted from the result vector
 * @param[in] gamma the gamma parameter used in the rbf kernel function
 */
template <typename real_type>
__global__ void kernel_radial(const real_type *q, real_type *ret, const real_type *d, const real_type *data_d, const real_type QA_cost, const real_type cost, const int Ncols, const int Nrows, const int add, const real_type gamma);

}  // namespace plssvm::cuda