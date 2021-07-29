/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright
 *
 * @brief Defines SYCL functions for generating the `q` vector.
 */

#pragma once

#include "sycl/sycl.hpp"  // sycl::item

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
    device_kernel_q_linear(real_type *q, const real_type *data_d, const real_type *data_last, int num_rows, int first_feature, int last_feature);

    /**
     * @brief Function call operator overload performing the actual calculation.
     * @param[in] item the [`sycl::item`](https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#subsec:item.class)
     *                 identifying an instance of the functor executing at each point in a [`sycl::range`](https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#range-class)
     */
    SYCL_EXTERNAL void operator()(::sycl::item<1> item) const;

  private:
    real_type *q_;
    const real_type *data_d_;
    const real_type *data_last_;
    const int num_rows_;
    const int first_feature_;
    const int last_feature_;
};

extern template class device_kernel_q_linear<float>;
extern template class device_kernel_q_linear<double>;

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
    device_kernel_q_poly(real_type *q, const real_type *data_d, const real_type *data_last, int num_rows, int num_cols, real_type degree, real_type gamma, real_type coef0);

    /**
     * @brief Function call operator overload performing the actual calculation.
     * @param[in] item the [`sycl::item`](https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#subsec:item.class)
     *                 identifying an instance of the functor executing at each point in a [`sycl::range`](https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#range-class)
     */
    SYCL_EXTERNAL void operator()(::sycl::item<1> item) const;

  private:
    real_type *q_;
    const real_type *data_d_;
    const real_type *data_last_;
    const int num_rows_;
    const int num_cols_;
    const real_type degree_;
    const real_type gamma_;
    const real_type coef0_;
};

extern template class device_kernel_q_poly<float>;
extern template class device_kernel_q_poly<double>;

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
    device_kernel_q_radial(real_type *q, const real_type *data_d, const real_type *data_last, int num_rows, int num_cols, real_type gamma);

    /**
     * @brief Function call operator overload performing the actual calculation.
     * @param[in] item the [`sycl::item`](https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#subsec:item.class)
     *                 identifying an instance of the functor executing at each point in a [`sycl::range`](https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#range-class)
     */
    SYCL_EXTERNAL void operator()(::sycl::item<1> item) const;

  private:
    real_type *q_;
    const real_type *data_d_;
    const real_type *data_last_;
    const int num_rows_;
    const int num_cols_;
    const real_type gamma_;
};

extern template class device_kernel_q_radial<float>;
extern template class device_kernel_q_radial<double>;

}  // namespace plssvm::sycl