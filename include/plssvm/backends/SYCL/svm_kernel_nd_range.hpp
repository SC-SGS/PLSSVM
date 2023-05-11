/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines the kernel functions for the C-SVM in the nd_range formulation using the SYCL backend.
 */

#ifndef PLSSVM_BACKENDS_SYCL_SVM_KERNEL_ND_RANGE_HPP_
#define PLSSVM_BACKENDS_SYCL_SVM_KERNEL_ND_RANGE_HPP_
#pragma once

#include "plssvm/backends/SYCL/detail/atomics.hpp"  // plssvm::sycl::detail::atomic_op
#include "plssvm/constants.hpp"                     // plssvm::kernel_index_type, plssvm::THREAD_BLOCK_SIZE, plssvm::INTERNAL_BLOCK_SIZE

#include "sycl/sycl.hpp"                            // sycl::nd_item, sycl::local_accessor, sycl::range, sycl::group_barrier, sycl::pow, sycl::exp, sycl::atomic_ref

#include <cstddef>                                  // std::size_t (cant' use kernel_index_type because of comparisons with unsigned long values)

namespace plssvm::sycl::detail {

/**
 * @brief Calculates the C-SVM kernel using the nd_range formulation and the linear kernel function.
 * @details Supports multi-GPU execution.
 * @tparam T the type of the data
 */
template <typename T>
class nd_range_device_kernel_linear {
  public:
    /// The type of the data.
    using real_type = T;

    /**
     * @brief Construct a new device kernel calculating the C-SVM kernel using the linear C-SVM kernel.
     * @param[in] cgh [`sycl::handler`](https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#sec:handlerClass) used to allocate the local memory
     * @param[in] q the `q` vector
     * @param[out] ret the result vector
     * @param[in] d the right-hand side of the equation
     * @param[in] data_d the one-dimension data matrix
     * @param[in] QA_cost he bottom right matrix entry multiplied by cost
     * @param[in] cost 1 / the cost parameter in the C-SVM
     * @param[in] num_rows the number of columns in the data matrix
     * @param[in] feature_range number of features used for the calculation on the device @p id
     * @param[in] add denotes whether the values are added or subtracted from the result vector
     * @param[in] id the id of the device
     */
    nd_range_device_kernel_linear(::sycl::handler &cgh, const real_type *q, real_type *ret, const real_type *d, const real_type *data_d, const real_type QA_cost, const real_type cost, const kernel_index_type num_rows, const kernel_index_type feature_range, const real_type add, const kernel_index_type id) :
        data_intern_i_{ ::sycl::range<2>{ THREAD_BLOCK_SIZE, INTERNAL_BLOCK_SIZE }, cgh }, data_intern_j_{ ::sycl::range<2>{ THREAD_BLOCK_SIZE, INTERNAL_BLOCK_SIZE }, cgh }, q_{ q }, ret_{ ret }, d_{ d }, data_d_{ data_d }, QA_cost_{ QA_cost }, cost_{ cost }, num_rows_{ num_rows }, feature_range_{ feature_range }, add_{ add }, device_{ id } {}

    /**
     * @brief Function call operator overload performing the actual calculation.
     * @param[in] nd_idx the [`sycl::nd_item`](https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#nditem-class)
     *                   identifying an instance of the functor executing at each point in a [`sycl::range`](https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#range-class)
     */
    void operator()(::sycl::nd_item<2> nd_idx) const {
        kernel_index_type i = nd_idx.get_group(0) * nd_idx.get_local_range(0) * INTERNAL_BLOCK_SIZE;
        kernel_index_type j = nd_idx.get_group(1) * nd_idx.get_local_range(1) * INTERNAL_BLOCK_SIZE;

        real_type matr[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE] = { { 0.0 } };
        real_type data_j[INTERNAL_BLOCK_SIZE] = { { 0.0 } };

        if (nd_idx.get_local_range(0) < THREAD_BLOCK_SIZE && nd_idx.get_local_range(1) == 0) {
            #pragma unroll INTERNAL_BLOCK_SIZE
            for (kernel_index_type block_id = 0; block_id < INTERNAL_BLOCK_SIZE; ++block_id) {
                data_intern_i_[nd_idx.get_local_range(0)][block_id] = 0.0;
                data_intern_j_[nd_idx.get_local_range(0)][block_id] = 0.0;
            }
        }
        ::sycl::group_barrier(nd_idx.get_group());

        if (i >= j) {
            i += nd_idx.get_local_id(0) * INTERNAL_BLOCK_SIZE;
            j += nd_idx.get_local_id(1) * INTERNAL_BLOCK_SIZE;

            // cache data
            for (kernel_index_type vec_index = 0; vec_index < feature_range_ * num_rows_; vec_index += num_rows_) {
                ::sycl::group_barrier(nd_idx.get_group());
                #pragma unroll INTERNAL_BLOCK_SIZE
                for (kernel_index_type block_id = 0; block_id < INTERNAL_BLOCK_SIZE; ++block_id) {
                    const std::size_t idx = block_id % THREAD_BLOCK_SIZE;
                    if (nd_idx.get_local_id(1) == idx) {
                        data_intern_i_[nd_idx.get_local_id(0)][block_id] = data_d_[block_id + vec_index + i];
                    }
                    const std::size_t idx_2 = block_id % THREAD_BLOCK_SIZE;
                    if (nd_idx.get_local_id(0) == idx_2) {
                        data_intern_j_[nd_idx.get_local_id(1)][block_id] = data_d_[block_id + vec_index + j];
                    }
                }
                ::sycl::group_barrier(nd_idx.get_group());

                #pragma unroll INTERNAL_BLOCK_SIZE
                for (kernel_index_type data_index = 0; data_index < INTERNAL_BLOCK_SIZE; ++data_index) {
                    data_j[data_index] = data_intern_j_[nd_idx.get_local_id(1)][data_index];
                }

                #pragma unroll INTERNAL_BLOCK_SIZE
                for (kernel_index_type l = 0; l < INTERNAL_BLOCK_SIZE; ++l) {
                    const real_type data_i = data_intern_i_[nd_idx.get_local_id(0)][l];
                    #pragma unroll INTERNAL_BLOCK_SIZE
                    for (kernel_index_type k = 0; k < INTERNAL_BLOCK_SIZE; ++k) {
                        matr[k][l] += data_i * data_j[k];
                    }
                }
            }

            #pragma unroll INTERNAL_BLOCK_SIZE
            for (kernel_index_type x = 0; x < INTERNAL_BLOCK_SIZE; ++x) {
                real_type ret_jx = 0.0;
                #pragma unroll INTERNAL_BLOCK_SIZE
                for (kernel_index_type y = 0; y < INTERNAL_BLOCK_SIZE; ++y) {
                    real_type temp;
                    if (device_ == 0) {
                        temp = (matr[x][y] + QA_cost_ - q_[i + y] - q_[j + x]) * add_;
                    } else {
                        temp = matr[x][y] * add_;
                    }
                    if (i + x > j + y) {
                        // upper triangular matrix
                        detail::atomic_op<real_type>{ ret_[i + y] } += temp * d_[j + x];
                        ret_jx += temp * d_[i + y];
                    } else if (i + x == j + y) {
                        // diagonal
                        if (device_ == 0) {
                            ret_jx += (temp + cost_ * add_) * d_[i + y];
                        } else {
                            ret_jx += temp * d_[i + y];
                        }
                    }
                }
                detail::atomic_op<real_type>{ ret_[j + x] } += ret_jx;
            }
        }
    }

  private:
    /// Local memory used for internal memory access optimizations.
    ::sycl::local_accessor<real_type, 2> data_intern_i_;
    /// Local memory used for internal memory access optimizations.
    ::sycl::local_accessor<real_type, 2> data_intern_j_;

    /// @cond Doxygen_suppress
    const real_type *q_;
    real_type *ret_;
    const real_type *d_;
    const real_type *data_d_;
    const real_type QA_cost_;
    const real_type cost_;
    const kernel_index_type num_rows_;
    const kernel_index_type feature_range_;
    const real_type add_;
    const kernel_index_type device_;
    /// @endcond
};

/**
 * @brief Calculates the C-SVM kernel using the nd_range formulation and the polynomial kernel function.
 * @details Currently only single GPU execution is supported.
 * @tparam T the type of the data
 */
template <typename T>
class nd_range_device_kernel_polynomial {
  public:
    /// The type of the data.
    using real_type = T;

    /**
     * @brief Construct a new device kernel calculating the C-SVM kernel using the polynomial C-SVM kernel.
     * @param[in] cgh [`sycl::handler`](https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#sec:handlerClass) used to allocate the local memory
     * @param[in] q the `q` vector
     * @param[out] ret the result vector
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
    nd_range_device_kernel_polynomial(::sycl::handler &cgh, const real_type *q, real_type *ret, const real_type *d, const real_type *data_d, const real_type QA_cost, const real_type cost, const kernel_index_type num_rows, const kernel_index_type num_cols, const real_type add, const int degree, const real_type gamma, const real_type coef0) :
        data_intern_i_{ ::sycl::range<2>{ THREAD_BLOCK_SIZE, INTERNAL_BLOCK_SIZE }, cgh }, data_intern_j_{ ::sycl::range<2>{ THREAD_BLOCK_SIZE, INTERNAL_BLOCK_SIZE }, cgh }, q_{ q }, ret_{ ret }, d_{ d }, data_d_{ data_d }, QA_cost_{ QA_cost }, cost_{ cost }, num_rows_{ num_rows }, num_cols_{ num_cols }, add_{ add }, degree_{ degree }, gamma_{ gamma }, coef0_{ coef0 } {}

    /**
     * @brief Function call operator overload performing the actual calculation.
     * @param[in] nd_idx the [`sycl::nd_item`](https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#nditem-class)
     *                   identifying an instance of the functor executing at each point in a [`sycl::range`](https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#range-class)
     */
    void operator()(::sycl::nd_item<2> nd_idx) const {
        kernel_index_type i = nd_idx.get_group(0) * nd_idx.get_local_range(0) * INTERNAL_BLOCK_SIZE;
        kernel_index_type j = nd_idx.get_group(1) * nd_idx.get_local_range(1) * INTERNAL_BLOCK_SIZE;

        real_type matr[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE] = { { 0.0 } };
        real_type data_j[INTERNAL_BLOCK_SIZE] = { { 0.0 } };

        if (nd_idx.get_local_range(0) < THREAD_BLOCK_SIZE && nd_idx.get_local_range(1) == 0) {
            #pragma unroll INTERNAL_BLOCK_SIZE
            for (kernel_index_type block_id = 0; block_id < INTERNAL_BLOCK_SIZE; ++block_id) {
                data_intern_i_[nd_idx.get_local_range(0)][block_id] = 0.0;
                data_intern_j_[nd_idx.get_local_range(0)][block_id] = 0.0;
            }
        }
        ::sycl::group_barrier(nd_idx.get_group());

        if (i >= j) {
            i += nd_idx.get_local_id(0) * INTERNAL_BLOCK_SIZE;
            j += nd_idx.get_local_id(1) * INTERNAL_BLOCK_SIZE;

            // cache data
            for (kernel_index_type vec_index = 0; vec_index < num_cols_ * num_rows_; vec_index += num_rows_) {
                ::sycl::group_barrier(nd_idx.get_group());
                #pragma unroll INTERNAL_BLOCK_SIZE
                for (kernel_index_type block_id = 0; block_id < INTERNAL_BLOCK_SIZE; ++block_id) {
                    const std::size_t idx = block_id % THREAD_BLOCK_SIZE;
                    if (nd_idx.get_local_id(1) == idx) {
                        data_intern_i_[nd_idx.get_local_id(0)][block_id] = data_d_[block_id + vec_index + i];
                    }
                    const std::size_t idx_2 = block_id % THREAD_BLOCK_SIZE;
                    if (nd_idx.get_local_id(0) == idx_2) {
                        data_intern_j_[nd_idx.get_local_id(1)][block_id] = data_d_[block_id + vec_index + j];
                    }
                }
                ::sycl::group_barrier(nd_idx.get_group());

                #pragma unroll INTERNAL_BLOCK_SIZE
                for (kernel_index_type data_index = 0; data_index < INTERNAL_BLOCK_SIZE; ++data_index) {
                    data_j[data_index] = data_intern_j_[nd_idx.get_local_id(1)][data_index];
                }

                #pragma unroll INTERNAL_BLOCK_SIZE
                for (kernel_index_type l = 0; l < INTERNAL_BLOCK_SIZE; ++l) {
                    const real_type data_i = data_intern_i_[nd_idx.get_local_id(0)][l];
                    #pragma unroll INTERNAL_BLOCK_SIZE
                    for (kernel_index_type k = 0; k < INTERNAL_BLOCK_SIZE; ++k) {
                        matr[k][l] += data_i * data_j[k];
                    }
                }
            }

            #pragma unroll INTERNAL_BLOCK_SIZE
            for (kernel_index_type x = 0; x < INTERNAL_BLOCK_SIZE; ++x) {
                real_type ret_jx = 0.0;
                #pragma unroll INTERNAL_BLOCK_SIZE
                for (kernel_index_type y = 0; y < INTERNAL_BLOCK_SIZE; ++y) {
                    const real_type temp = (::sycl::pow(gamma_ * matr[x][y] + coef0_, static_cast<real_type>(degree_)) + QA_cost_ - q_[i + y] - q_[j + x]) * add_;
                    if (i + x > j + y) {
                        // upper triangular matrix
                        detail::atomic_op<real_type>{ ret_[i + y] } += temp * d_[j + x];
                        ret_jx += temp * d_[i + y];
                    } else if (i + x == j + y) {
                        // diagonal
                        ret_jx += (temp + cost_ * add_) * d_[i + y];
                    }
                }
                detail::atomic_op<real_type>{ ret_[j + x] } += ret_jx;
            }
        }
    }

  private:
    /// Local memory used for internal memory access optimizations.
    ::sycl::local_accessor<real_type, 2> data_intern_i_;
    /// Local memory used for internal memory access optimizations.
    ::sycl::local_accessor<real_type, 2> data_intern_j_;

    /// @cond Doxygen_suppress
    const real_type *q_;
    real_type *ret_;
    const real_type *d_;
    const real_type *data_d_;
    const real_type QA_cost_;
    const real_type cost_;
    const kernel_index_type num_rows_;
    const kernel_index_type num_cols_;
    const real_type add_;
    const int degree_;
    const real_type gamma_;
    const real_type coef0_;
    /// @endcond
};

/**
 * @brief Calculates the C-SVM kernel using the nd_range formulation and the radial basis functions kernel function.
 * @details Currently only single GPU execution is supported.
 * @tparam T the type of the data
 */
template <typename T>
class nd_range_device_kernel_rbf {
  public:
    /// The type of the data.
    using real_type = T;

    /**
     * @brief Construct a new device kernel calculating the C-SVM kernel using the radial basis functions C-SVM kernel.
     * @param[in] cgh [`sycl::handler`](https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#sec:handlerClass) used to allocate the local memory
     * @param[in] q the `q` vector
     * @param[out] ret the result vector
     * @param[in] d the right-hand side of the equation
     * @param[in] data_d the one-dimension data matrix
     * @param[in] QA_cost he bottom right matrix entry multiplied by cost
     * @param[in] cost 1 / the cost parameter in the C-SVM
     * @param[in] num_rows the number of columns in the data matrix
     * @param[in] num_cols the number of rows in the data matrix
     * @param[in] add denotes whether the values are added or subtracted from the result vector
     * @param[in] gamma the gamma parameter used in the rbf kernel function
     */
    nd_range_device_kernel_rbf(::sycl::handler &cgh, const real_type *q, real_type *ret, const real_type *d, const real_type *data_d, const real_type QA_cost, const real_type cost, const kernel_index_type num_rows, const kernel_index_type num_cols, const real_type add, const real_type gamma) :
        data_intern_i_{ ::sycl::range<2>{ THREAD_BLOCK_SIZE, INTERNAL_BLOCK_SIZE }, cgh }, data_intern_j_{ ::sycl::range<2>{ THREAD_BLOCK_SIZE, INTERNAL_BLOCK_SIZE }, cgh }, q_{ q }, ret_{ ret }, d_{ d }, data_d_{ data_d }, QA_cost_{ QA_cost }, cost_{ cost }, num_rows_{ num_rows }, num_cols_{ num_cols }, add_{ add }, gamma_{ gamma } {}

    /**
     * @brief Function call operator overload performing the actual calculation.
     * @param[in] nd_idx the [`sycl::nd_item`](https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#nditem-class)
     *                   identifying an instance of the functor executing at each point in a [`sycl::range`](https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#range-class)
     */
    void operator()(::sycl::nd_item<2> nd_idx) const {
        kernel_index_type i = nd_idx.get_group(0) * nd_idx.get_local_range(0) * INTERNAL_BLOCK_SIZE;
        kernel_index_type j = nd_idx.get_group(1) * nd_idx.get_local_range(1) * INTERNAL_BLOCK_SIZE;

        real_type matr[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE] = { { 0.0 } };
        real_type data_j[INTERNAL_BLOCK_SIZE] = { { 0.0 } };

        if (nd_idx.get_local_range(0) < THREAD_BLOCK_SIZE && nd_idx.get_local_range(1) == 0) {
            #pragma unroll INTERNAL_BLOCK_SIZE
            for (kernel_index_type block_id = 0; block_id < INTERNAL_BLOCK_SIZE; ++block_id) {
                data_intern_i_[nd_idx.get_local_range(0)][block_id] = 0.0;
                data_intern_j_[nd_idx.get_local_range(0)][block_id] = 0.0;
            }
        }
        ::sycl::group_barrier(nd_idx.get_group());

        if (i >= j) {
            i += nd_idx.get_local_id(0) * INTERNAL_BLOCK_SIZE;
            j += nd_idx.get_local_id(1) * INTERNAL_BLOCK_SIZE;

            // cache data
            for (kernel_index_type vec_index = 0; vec_index < num_cols_ * num_rows_; vec_index += num_rows_) {
                ::sycl::group_barrier(nd_idx.get_group());
                #pragma unroll INTERNAL_BLOCK_SIZE
                for (kernel_index_type block_id = 0; block_id < INTERNAL_BLOCK_SIZE; ++block_id) {
                    const std::size_t idx = block_id % THREAD_BLOCK_SIZE;
                    if (nd_idx.get_local_id(1) == idx) {
                        data_intern_i_[nd_idx.get_local_id(0)][block_id] = data_d_[block_id + vec_index + i];
                    }
                    const std::size_t idx_2 = block_id % THREAD_BLOCK_SIZE;
                    if (nd_idx.get_local_id(0) == idx_2) {
                        data_intern_j_[nd_idx.get_local_id(1)][block_id] = data_d_[block_id + vec_index + j];
                    }
                }
                ::sycl::group_barrier(nd_idx.get_group());

                #pragma unroll INTERNAL_BLOCK_SIZE
                for (kernel_index_type data_index = 0; data_index < INTERNAL_BLOCK_SIZE; ++data_index) {
                    data_j[data_index] = data_intern_j_[nd_idx.get_local_id(1)][data_index];
                }

                #pragma unroll INTERNAL_BLOCK_SIZE
                for (kernel_index_type l = 0; l < INTERNAL_BLOCK_SIZE; ++l) {
                    const real_type data_i = data_intern_i_[nd_idx.get_local_id(0)][l];
                    #pragma unroll INTERNAL_BLOCK_SIZE
                    for (kernel_index_type k = 0; k < INTERNAL_BLOCK_SIZE; ++k) {
                        matr[k][l] += (data_i - data_j[k]) * (data_i - data_j[k]);
                    }
                }
            }

            #pragma unroll INTERNAL_BLOCK_SIZE
            for (kernel_index_type x = 0; x < INTERNAL_BLOCK_SIZE; ++x) {
                real_type ret_jx = 0.0;
                #pragma unroll INTERNAL_BLOCK_SIZE
                for (kernel_index_type y = 0; y < INTERNAL_BLOCK_SIZE; ++y) {
                    const real_type temp = (::sycl::exp(-gamma_ * matr[x][y]) + QA_cost_ - q_[i + y] - q_[j + x]) * add_;
                    if (i + x > j + y) {
                        // upper triangular matrix
                        detail::atomic_op<real_type>{ ret_[i + y] } += temp * d_[j + x];
                        ret_jx += temp * d_[i + y];
                    } else if (i + x == j + y) {
                        // diagonal
                        ret_jx += (temp + cost_ * add_) * d_[i + y];
                    }
                }
                detail::atomic_op<real_type>{ ret_[j + x] } += ret_jx;
            }
        }
    }

  private:
    /// Local memory used for internal memory access optimizations.
    ::sycl::local_accessor<real_type, 2> data_intern_i_;
    /// Local memory used for internal memory access optimizations.
    ::sycl::local_accessor<real_type, 2> data_intern_j_;

    /// @cond Doxygen_suppress
    const real_type *q_;
    real_type *ret_;
    const real_type *d_;
    const real_type *data_d_;
    const real_type QA_cost_;
    const real_type cost_;
    const kernel_index_type num_rows_;
    const kernel_index_type num_cols_;
    const real_type add_;
    const real_type gamma_;
    /// @endcond
};

}  // namespace plssvm::sycl::detail

#endif  // PLSSVM_BACKENDS_SYCL_SVM_KERNEL_ND_RANGE_HPP_