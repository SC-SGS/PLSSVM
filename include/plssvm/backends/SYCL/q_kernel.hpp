/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines SYCL functions for generating the `q` vector.
 */

#ifndef PLSSVM_BACKENDS_SYCL_Q_KERNEL_HPP_
#define PLSSVM_BACKENDS_SYCL_Q_KERNEL_HPP_
#pragma once

#include "plssvm/constants.hpp"  // plssvm::kernel_index_type

#include "sycl/sycl.hpp"  // sycl::nd_item, sycl::pow, sycl::exp

namespace plssvm::sycl::detail {

/**
 * @brief Functor to calculate the `q` vector using the linear C-SVM kernel.
 * @details Supports multi-GPU execution.
 * @tparam T the type of the data
 */
template <typename T>
class device_kernel_q_linear {
  public:
    /// The type of the data.
    using real_type = T;

    /**
     * @brief Construct a new device kernel calculating the `q` vector using the linear C-SVM kernel.
     * @param[out] q the calculated `q` vector
     * @param[in] data_d the one-dimensional data matrix
     * @param[in] data_last the last row in the data matrix
     * @param[in] num_rows the number of rows in the data matrix
     * @param[in] feature_range number of features used for the calculation
     */
    device_kernel_q_linear(real_type *q, const real_type *data_d, const real_type *data_last, const kernel_index_type num_rows, const kernel_index_type feature_range) :
        q_{ q }, data_d_{ data_d }, data_last_{ data_last }, num_rows_{ num_rows }, feature_range_{ feature_range } {}

    /**
     * @brief Function call operator overload performing the actual calculation.
     * @param[in] index the [`sycl::id`](https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#id-class)
     *                  identifying an instance of the functor executing at each point in a [`sycl::range`](https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#range-class)
     */
    void operator()(::sycl::id<1> index) const {
        real_type temp{ 0.0 };
        for (kernel_index_type i = 0; i < feature_range_; ++i) {
            temp += data_d_[i * num_rows_ + index] * data_last_[i];
        }
        q_[index] = temp;
    }

  private:
    /// @cond Doxygen_suppress
    real_type *q_;
    const real_type *data_d_;
    const real_type *data_last_;
    const kernel_index_type num_rows_;
    const kernel_index_type feature_range_;
    /// @endcond
};

/**
 * @brief Functor to calculate the `q` vector using the polynomial C-SVM kernel.
 * @details Currently only single GPU execution is supported.
 * @tparam T the type of the data
 */
template <typename T>
class device_kernel_q_polynomial {
  public:
    /// The type of the data.
    using real_type = T;

    /**
     * @brief Construct a new device kernel calculating the `q` vector using the polynomial C-SVM kernel.
     * @param[out] q the calculated `q` vector
     * @param[in] data_d the one-dimensional data matrix
     * @param[in] data_last the last row in the data matrix
     * @param[in] num_rows the number of rows in the data matrix
     * @param[in] num_cols the number of columns in the data matrix
     * @param[in] degree the degree parameter used in the polynomial kernel function
     * @param[in] gamma the gamma parameter used in the polynomial kernel function
     * @param[in] coef0 the coef0 parameter used in the polynomial kernel function
     */
    device_kernel_q_polynomial(real_type *q, const real_type *data_d, const real_type *data_last, const kernel_index_type num_rows, const kernel_index_type num_cols, const int degree, const real_type gamma, const real_type coef0) :
        q_{ q }, data_d_{ data_d }, data_last_{ data_last }, num_rows_{ num_rows }, num_cols_{ num_cols }, degree_{ degree }, gamma_{ gamma }, coef0_{ coef0 } {}

    /**
     * @brief Function call operator overload performing the actual calculation.
     * @param[in] index the [`sycl::id`](https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#id-class)
     *                  identifying an instance of the functor executing at each point in a [`sycl::range`](https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#range-class)
     */
    void operator()(::sycl::id<1> index) const {
        real_type temp{ 0.0 };
        for (kernel_index_type i = 0; i < num_cols_; ++i) {
            temp += data_d_[i * num_rows_ + index] * data_last_[i];
        }
        q_[index] = ::sycl::pow(gamma_ * temp + coef0_, static_cast<real_type>(degree_));
    }

  private:
    /// @cond Doxygen_suppress
    real_type *q_;
    const real_type *data_d_;
    const real_type *data_last_;
    const kernel_index_type num_rows_;
    const kernel_index_type num_cols_;
    const int degree_;
    const real_type gamma_;
    const real_type coef0_;
    /// @endcond
};

/**
 * @brief Functor to calculate the `q` vector using the radial basis functions C-SVM kernel.
 * @details Currently only single GPU execution is supported.
 * @tparam T the type of the data
 */
template <typename T>
class device_kernel_q_rbf {
  public:
    /// The type of the data.
    using real_type = T;

    /**
     * @brief Construct a new device kernel calculating the `q` vector using the radial basis functions C-SVM kernel.
     * @param[out] q the calculated `q` vector
     * @param[in] data_d the one-dimensional data matrix
     * @param[in] data_last the last row in the data matrix
     * @param[in] num_rows the number of rows in the data matrix
     * @param[in] num_cols the number of columns in the data matrix
     * @param[in] gamma the gamma parameter used in the rbf kernel function
     */
    device_kernel_q_rbf(real_type *q, const real_type *data_d, const real_type *data_last, const kernel_index_type num_rows, const kernel_index_type num_cols, const real_type gamma) :
        q_{ q }, data_d_{ data_d }, data_last_{ data_last }, num_rows_{ num_rows }, num_cols_{ num_cols }, gamma_{ gamma } {}

    /**
     * @brief Function call operator overload performing the actual calculation.
     * @param[in] index the [`sycl::id`](https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#id-class)
     *                  identifying an instance of the functor executing at each point in a [`sycl::range`](https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#range-class)
     */
    void operator()(::sycl::id<1> index) const {
        real_type temp{ 0.0 };
        for (kernel_index_type i = 0; i < num_cols_; ++i) {
            temp += (data_d_[i * num_rows_ + index] - data_last_[i]) * (data_d_[i * num_rows_ + index] - data_last_[i]);
        }
        q_[index] = ::sycl::exp(-gamma_ * temp);
    }

  private:
    /// @cond Doxygen_suppress
    real_type *q_;
    const real_type *data_d_;
    const real_type *data_last_;
    const kernel_index_type num_rows_;
    const kernel_index_type num_cols_;
    const real_type gamma_;
    /// @endcond
};

}  // namespace plssvm::sycl

#endif  // PLSSVM_BACKENDS_SYCL_Q_KERNEL_HPP_