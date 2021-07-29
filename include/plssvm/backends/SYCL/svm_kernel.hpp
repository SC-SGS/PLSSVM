/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright
 *
 * @brief Defines the kernel functions for the C-SVM using the SYCL backend.
 */

#pragma once

#include "sycl/sycl.hpp"  // sycl::nd_item, sycl::handler, sycl::accessor, sycl::access::mode, sycl::access::target

namespace plssvm::sycl {

// TODO: change to ::sycl::accessor once implemented
/**
 * @brief Shortcut alias for a SYCL local accessor.
 * @tparam T the type of the accessed values
 */
template <typename T>
using local_accessor = ::sycl::accessor<T, 2, ::sycl::access::mode::read_write, ::sycl::access::target::local>;

/**
 * @brief Calculates the C-SVM kernel using the linear kernel function.
 * @details Supports multi-GPU execution.
 * @tparam T the type of the data
 */
template <typename T>
class device_kernel_linear {
  public:
    /// The type of the data.
    using real_type = T;

    /**
     * @brief Construct a new device kernel calculating the `q` vector using the linear C-SVM kernel.
     * @param[in] cgh [`sycl::handler`](https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#sec:handlerClass) used to allocate the local memory
     * @param[in] q the `q` vector
     * @param[in] ret the result vector
     * @param[in] d the right-hand side of the equation
     * @param[in] data_d the one-dimension data matrix
     * @param[in] QA_cost he bottom right matrix entry multiplied by cost
     * @param[in] cost 1 / the cost parameter in the C-SVM
     * @param[in] num_rows the number of columns in the data matrix
     * @param[in] add denotes whether the values are added or subtracted from the result vector
     * @param[in] first_feature the first feature used in the calculations (depending on the current device)
     * @param[in] last_feature the last feature used in the calculations (depending on the current device)
     */
    device_kernel_linear(::sycl::handler &cgh, const real_type *q, real_type *ret, const real_type *d, const real_type *data_d, real_type QA_cost, real_type cost, int num_rows, int add, int first_feature, int last_feature);

    /**
     * @brief Function call operator overload performing the actual calculation.
     * @param[in] nd_idx the [`sycl::nd_item`](https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#nditem-class)
     *                   identifying an instance of the functor executing at each point in a [`sycl::range`](https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#range-class)
     */
    SYCL_EXTERNAL void operator()(::sycl::nd_item<2> nd_idx) const;

  private:
    local_accessor<real_type> data_intern_i_;
    local_accessor<real_type> data_intern_j_;

    const real_type *q_;
    real_type *ret_;
    const real_type *d_;
    const real_type *data_d_;
    const real_type QA_cost_;
    const real_type cost_;
    const int num_rows_;
    const int add_;
    const int first_feature_;
    const int last_feature_;
};

extern template class device_kernel_linear<float>;
extern template class device_kernel_linear<double>;

/**
 * @brief Calculates the C-SVM kernel using the polynomial kernel function.
 * @details Currently only single GPU execution is supported.
 * @tparam T the type of the data
 */
template <typename T>
class device_kernel_poly {
  public:
    /// The type of the data.
    using real_type = T;

    /**
     * @brief Construct a new device kernel calculating the `q` vector using the polynomial C-SVM kernel.
     * @param[in] cgh [`sycl::handler`](https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#sec:handlerClass) used to allocate the local memory
     * @param[in] q the `q` vector
     * @param[in] ret the result vector
     * @param[in] d the right-hand side of the equation
     * @param[in] data_d the one-dimension data matrix
     * @param[in] QA_cost he bottom right matrix entry multiplied by cost
     * @param[in] cost 1 / the cost parameter in the C-SVM
     * @param[in] num_rows the number of columns in the data matrix
     * @param[in] num_cols the number of rows in the data matrix
     * @param[in] add denotes whether the values are added or subtracted from the result vector
     * @param[in] degree the degree parameter used in the polynomial kernel function
     * @param[in] gamma the gamma parameter used in the polynomial kernel function
     * @param[in] coef0 the coef0 parameter used in the polynomial kernel function
     */
    device_kernel_poly(::sycl::handler &cgh, const real_type *q, real_type *ret, const real_type *d, const real_type *data_d, real_type QA_cost, real_type cost, int num_rows, int num_cols, int add, real_type degree, real_type gamma, real_type coef0);

    /**
     * @brief Function call operator overload performing the actual calculation.
     * @param[in] nd_idx the [`sycl::nd_item`](https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#nditem-class)
     *                   identifying an instance of the functor executing at each point in a [`sycl::range`](https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#range-class)
     */
    SYCL_EXTERNAL void operator()(::sycl::nd_item<2> nd_idx) const;

  private:
    local_accessor<real_type> data_intern_i_;
    local_accessor<real_type> data_intern_j_;

    const real_type *q_;
    real_type *ret_;
    const real_type *d_;
    const real_type *data_d_;
    const real_type QA_cost_;
    const real_type cost_;
    const int num_rows_;
    const int num_cols_;
    const int add_;
    const real_type degree_;
    const real_type gamma_;
    const real_type coef0_;
};

extern template class device_kernel_poly<float>;
extern template class device_kernel_poly<double>;

/**
 * @brief Calculates the C-SVM kernel using the radial basis functions kernel function.
 * @details Currently only single GPU execution is supported.
 * @tparam T the type of the data
 */
template <typename T>
class device_kernel_radial {
  public:
    /// The type of the data.
    using real_type = T;

    /**
     * @brief Construct a new device kernel calculating the `q` vector using the radial basis functions C-SVM kernel.
     * @param[in] cgh [`sycl::handler`](https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#sec:handlerClass) used to allocate the local memory
     * @param[in] q the `q` vector
     * @param[in] ret the result vector
     * @param[in] d the right-hand side of the equation
     * @param[in] data_d the one-dimension data matrix
     * @param[in] QA_cost he bottom right matrix entry multiplied by cost
     * @param[in] cost 1 / the cost parameter in the C-SVM
     * @param[in] num_rows the number of columns in the data matrix
     * @param[in] num_cols the number of rows in the data matrix
     * @param[in] add denotes whether the values are added or subtracted from the result vector
     * @param[in] gamma the gamma parameter used in the polynomial kernel function
     */
    device_kernel_radial(::sycl::handler &cgh, const real_type *q, real_type *ret, const real_type *d, const real_type *data_d, real_type QA_cost, real_type cost, int num_rows, int num_cols, int add, real_type gamma);

    /**
     * @brief Function call operator overload performing the actual calculation.
     * @param[in] nd_idx the [`sycl::nd_item`](https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#nditem-class)
     *                   identifying an instance of the functor executing at each point in a [`sycl::range`](https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#range-class)
     */
    SYCL_EXTERNAL void operator()(::sycl::nd_item<2> nd_idx) const;

  private:
    local_accessor<real_type> data_intern_i_;
    local_accessor<real_type> data_intern_j_;

    const real_type *q_;
    real_type *ret_;
    const real_type *d_;
    const real_type *data_d_;
    const real_type QA_cost_;
    const real_type cost_;
    const int num_rows_;
    const int num_cols_;
    const int add_;
    const real_type gamma_;
};

extern template class device_kernel_radial<float>;
extern template class device_kernel_radial<double>;

}  // namespace plssvm::sycl