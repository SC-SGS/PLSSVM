/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright
 *
 * @brief Defines CUDA functions for generating the `q` vector.
 */

#pragma once

namespace plssvm::cuda {

/**
 * @brief Calculates the `q` vector using the linear C-SVM kernel.
 * @details Supports multi-GPU execution.
 * @tparam real_type the type of the data
 * @param[out] q the calculated `q` vector
 * @param[in] data_d the one-dimensional data matrix
 * @param[in] data_last the last row in the data matrix
 * @param[in] Nrows the number of rows in the data matrix
 * @param[in] start the first feature used in the calculations (depending on the current device)
 * @param[in] end the last feature used in the calculations (depending on the current device)
 */
template <typename real_type>
__global__ void kernel_q_linear(real_type *q, const real_type *data_d, const real_type *data_last, const int Nrows, const int start, const int end);

/**
 * @brief Calculates the `q` vector using the polynomial C-SVM kernel.
 * @details Currently only single GPU execution is supported.
 * @tparam real_type the type of the data
 * @param[out] q the calculated `q` vector
 * @param[in] data_d the one-dimensional data matrix
 * @param[in] data_last the last row in the data matrix
 * @param[in] Nrows the number of rows in the data matrix
 * @param[in] Ncols the number of columns in the data matrix
 * @param[in] degree the degree parameter used in the polynomial kernel function
 * @param[in] gamma the gamma parameter used in the polynomial kernel function
 * @param[in] coef0 the coef0 parameter used in the polynomial kernel function
 */
template <typename real_type>
__global__ void kernel_q_poly(real_type *q, const real_type *data_d, const real_type *data_last, const int Nrows, const int Ncols, const real_type degree, const real_type gamma, const real_type coef0);

/**
 * @brief Calculates the `q` vector using the radial basis functions C-SVM kernel.
 * @details Currently only single GPU execution is supported.
 * @tparam real_type the type of the data
 * @param[out] q the calculated `q` vector
 * @param[in] data_d the one-dimensional data matrix
 * @param[in] data_last the last row in the data matrix
 * @param[in] Nrows the number of rows in the data matrix
 * @param[in] Ncols the number of columns in the data matrix
 * @param[in] gamma the gamma parameter used in the rbf kernel function
 */
template <typename real_type>
__global__ void kernel_q_radial(real_type *q, const real_type *data_d, const real_type *data_last, const int Nrows, const int Ncols, const real_type gamma);

}  // namespace plssvm::cuda