/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright
 *
 * @brief Defines SYCL functions for generating the `q` vector.
 */

#pragma once

#include "sycl/sycl.hpp"  // sycl::nd_item, sycl::pow, sycl::exp

namespace plssvm::sycl {

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
     * @param[in] first_feature the first feature used in the calculations (depending on the current device)
     * @param[in] last_feature the last feature used in the calculations (depending on the current device)
     */
    device_kernel_q_linear(real_type *q, const real_type *data_d, const real_type *data_last, int num_rows, int first_feature, int last_feature) :
        q_{ q }, data_d_{ data_d }, data_last_{ data_last }, num_rows_{ num_rows }, first_feature_{ first_feature }, last_feature_{ last_feature } {}

    /**
     * @brief Function call operator overload performing the actual calculation.
     * @param[in] item the [`sycl::item`](https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#subsec:item.class)
     *                 identifying an instance of the functor executing at each point in a [`sycl::range`](https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#range-class)
     */
    void operator()(::sycl::nd_item<1> item) const {
        const auto index = item.get_global_linear_id();
        real_type temp{ 0.0 };
        for (int i = first_feature_; i < last_feature_; ++i) {
            temp += data_d_[i * num_rows_ + index] * data_last_[i];
        }
        q_[index] = temp;
    }

  private:
    real_type *q_;
    const real_type *data_d_;
    const real_type *data_last_;
    const int num_rows_;
    const int first_feature_;
    const int last_feature_;
};

/**
 * @brief Functor to calculate the `q` vector using the polynomial C-SVM kernel.
 * @details Currently only single GPU execution is supported.
 * @tparam T the type of the data
 */
template <typename T>
class device_kernel_q_poly {
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
    device_kernel_q_poly(real_type *q, const real_type *data_d, const real_type *data_last, int num_rows, int num_cols, int degree, real_type gamma, real_type coef0) :
        q_{ q }, data_d_{ data_d }, data_last_{ data_last }, num_rows_{ num_rows }, num_cols_{ num_cols }, degree_{ degree }, gamma_{ gamma }, coef0_{ coef0 } {}

    /**
     * @brief Function call operator overload performing the actual calculation.
     * @param[in] item the [`sycl::item`](https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#subsec:item.class)
     *                 identifying an instance of the functor executing at each point in a [`sycl::range`](https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#range-class)
     */
    void operator()(::sycl::nd_item<1> item) const {
        const auto index = item.get_global_linear_id();
        real_type temp{ 0.0 };
        for (int i = 0; i < num_cols_; ++i) {
            temp += data_d_[i * num_rows_ + index] * data_last_[i];
        }
        q_[index] = ::sycl::pow(gamma_ * temp + coef0_, static_cast<real_type>(degree_));
    }

  private:
    real_type *q_;
    const real_type *data_d_;
    const real_type *data_last_;
    const int num_rows_;
    const int num_cols_;
    const int degree_;
    const real_type gamma_;
    const real_type coef0_;
};

/**
 * @brief Functor to calculate the `q` vector using the radial basis functions C-SVM kernel.
 * @details Currently only single GPU execution is supported.
 * @tparam T the type of the data
 */
template <typename T>
class device_kernel_q_radial {
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
     * @param[in] gamma the gamma parameter used in the polynomial kernel function
     */
    device_kernel_q_radial(real_type *q, const real_type *data_d, const real_type *data_last, int num_rows, int num_cols, real_type gamma) :
        q_{ q }, data_d_{ data_d }, data_last_{ data_last }, num_rows_{ num_rows }, num_cols_{ num_cols }, gamma_{ gamma } {}

    /**
     * @brief Function call operator overload performing the actual calculation.
     * @param[in] item the [`sycl::item`](https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#subsec:item.class)
     *                 identifying an instance of the functor executing at each point in a [`sycl::range`](https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#range-class)
     */
    void operator()(::sycl::nd_item<1> item) const {
        const auto index = item.get_global_linear_id();
        real_type temp{ 0.0 };
        for (int i = 0; i < num_cols_; ++i) {
            temp += (data_d_[i * num_rows_ + index] - data_last_[i]) * (data_d_[i * num_rows_ + index] - data_last_[i]);
        }
        q_[index] = ::sycl::exp(-gamma_ * temp);
    }

  private:
    real_type *q_;
    const real_type *data_d_;
    const real_type *data_last_;
    const int num_rows_;
    const int num_cols_;
    const real_type gamma_;
};

}  // namespace plssvm::sycl
