/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines the kernel functions for the C-SVM using the SYCL backend.
 */

#ifndef PLSSVM_BACKENDS_SYCL_SVM_KERNEL_HIERARCHICAL_HPP_
#define PLSSVM_BACKENDS_SYCL_SVM_KERNEL_HIERARCHICAL_HPP_
#pragma once

#include "plssvm/backends/SYCL/detail/atomics.hpp"  // plssvm::sycl::detail::atomic_op
#include "plssvm/constants.hpp"                     // plssvm::kernel_index_type, plssvm::THREAD_BLOCK_SIZE, plssvm::INTERNAL_BLOCK_SIZE

#include "sycl/sycl.hpp"                            // sycl::queue, sycl::handler, sycl::h_item, sycl::range, sycl::private_memory, sycl::pow, sycl::exp

#include <cstddef>                                  // std::size_t (cant' use kernel_index_type because of comparisons with unsigned long values)

namespace plssvm::sycl::detail {

/**
 * @brief Calculates the C-SVM kernel using the hierarchical formulation and the linear kernel function.
 * @details Supports multi-GPU execution.
 * @tparam T the type of the data
 */
template <typename T>
class hierarchical_device_kernel_linear {
  public:
    /// The type of the data.
    using real_type = T;

    /**
     * @brief Construct a new device kernel calculating the C-SVM kernel using the linear C-SVM kernel.
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
    hierarchical_device_kernel_linear(const real_type *q, real_type *ret, const real_type *d, const real_type *data_d, const real_type QA_cost, const real_type cost, const kernel_index_type num_rows, const kernel_index_type feature_range, const real_type add, const kernel_index_type id) :
        q_{ q }, ret_{ ret }, d_{ d }, data_d_{ data_d }, QA_cost_{ QA_cost }, cost_{ cost }, num_rows_{ num_rows }, feature_range_{ feature_range }, add_{ add }, device_{ id } {}

    /**
     * @brief Function call operator overload performing the actual calculation.
     * @param[in] group the [`sycl::group`](https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#group-class)
     *                  identifying an instance of the currently execution work-group
     */
    void operator()(::sycl::group<2> group) const {
        // allocate shared memory
        real_type data_intern_i[THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE];
        real_type data_intern_j[THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE];

        // allocate memory for work-item local variables
        // -> accessible across different 'parallel_for_work_item' invocations
        ::sycl::private_memory<real_type[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE], 2> private_matr{ group };
        ::sycl::private_memory<real_type[INTERNAL_BLOCK_SIZE], 2> private_data_j{ group };
        ::sycl::private_memory<kernel_index_type, 2> private_i{ group };
        ::sycl::private_memory<kernel_index_type, 2> private_j{ group };
        ::sycl::private_memory<bool, 2> private_cond{ group };

        // initialize private variables
        group.parallel_for_work_item([&](::sycl::h_item<2> idx) {
            // indices and diagonal condition
            private_i(idx) = group[0] * idx.get_local_range(0) * INTERNAL_BLOCK_SIZE;
            private_j(idx) = group[1] * idx.get_local_range(1) * INTERNAL_BLOCK_SIZE;
            private_cond(idx) = private_i(idx) >= private_j(idx);
            if (private_cond(idx)) {
                private_i(idx) += idx.get_local_id(0) * INTERNAL_BLOCK_SIZE;
                private_j(idx) += idx.get_local_id(1) * INTERNAL_BLOCK_SIZE;
            }

            // matrix
            #pragma unroll INTERNAL_BLOCK_SIZE
            for (kernel_index_type i = 0; i < INTERNAL_BLOCK_SIZE; ++i) {
                #pragma unroll INTERNAL_BLOCK_SIZE
                for (kernel_index_type j = 0; j < INTERNAL_BLOCK_SIZE; ++j) {
                    private_matr(idx)[i][j] = real_type{ 0.0 };
                }
            }
        });

        // implicit group barrier

        // load data from global in shared memory
        for (kernel_index_type vec_index = 0; vec_index < feature_range_ * num_rows_; vec_index += num_rows_) {
            group.parallel_for_work_item([&](::sycl::h_item<2> idx) {
                if (private_cond(idx)) {
                    #pragma unroll INTERNAL_BLOCK_SIZE
                    for (kernel_index_type block_id = 0; block_id < INTERNAL_BLOCK_SIZE; ++block_id) {
                        const std::size_t idx_1 = block_id % THREAD_BLOCK_SIZE;
                        if (idx.get_local_id(1) == idx_1) {
                            data_intern_i[idx.get_local_id(0)][block_id] = data_d_[block_id + vec_index + private_i(idx)];
                        }
                        const std::size_t idx_2 = block_id % THREAD_BLOCK_SIZE;
                        if (idx.get_local_id(0) == idx_2) {
                            data_intern_j[idx.get_local_id(1)][block_id] = data_d_[block_id + vec_index + private_j(idx)];
                        }
                    }
                }
            });

            // implicit group barrier

            // load data from shared in private memory and perform scalar product
            group.parallel_for_work_item([&](::sycl::h_item<2> idx) {
                if (private_cond(idx)) {
                    #pragma unroll INTERNAL_BLOCK_SIZE
                    for (kernel_index_type data_index = 0; data_index < INTERNAL_BLOCK_SIZE; ++data_index) {
                        private_data_j(idx)[data_index] = data_intern_j[idx.get_local_id(1)][data_index];
                    }

                    #pragma unroll INTERNAL_BLOCK_SIZE
                    for (kernel_index_type l = 0; l < INTERNAL_BLOCK_SIZE; ++l) {
                        const real_type data_i = data_intern_i[idx.get_local_id(0)][l];
                        #pragma unroll INTERNAL_BLOCK_SIZE
                        for (kernel_index_type k = 0; k < INTERNAL_BLOCK_SIZE; ++k) {
                            private_matr(idx)[k][l] += data_i * private_data_j(idx)[k];
                        }
                    }
                }
            });

            // implicit group barrier
        }

        // kernel function
        group.parallel_for_work_item([&](::sycl::h_item<2> idx) {
            if (private_cond(idx)) {
                #pragma unroll INTERNAL_BLOCK_SIZE
                for (kernel_index_type x = 0; x < INTERNAL_BLOCK_SIZE; ++x) {
                    real_type ret_jx = 0.0;
                    #pragma unroll INTERNAL_BLOCK_SIZE
                    for (kernel_index_type y = 0; y < INTERNAL_BLOCK_SIZE; ++y) {
                        real_type temp;
                        if (device_ == 0) {
                            temp = (private_matr(idx)[x][y] + QA_cost_ - q_[private_i(idx) + y] - q_[private_j(idx) + x]) * add_;
                        } else {
                            temp = private_matr(idx)[x][y] * add_;
                        }
                        if (private_i(idx) + x > private_j(idx) + y) {
                            // upper triangular matrix
                            detail::atomic_op<real_type>{ ret_[private_i(idx) + y] } += temp * d_[private_j(idx) + x];
                            ret_jx += temp * d_[private_i(idx) + y];
                        } else if (private_i(idx) + x == private_j(idx) + y) {
                            // diagonal
                            if (device_ == 0) {
                                ret_jx += (temp + cost_ * add_) * d_[private_i(idx) + y];
                            } else {
                                ret_jx += temp * d_[private_i(idx) + y];
                            }
                        }
                    }
                    detail::atomic_op<real_type>{ ret_[private_j(idx) + x] } += ret_jx;
                }
            }
        });
    }

  private:
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
 * @brief Calculates the C-SVM kernel using the hierarchical formulation and the polynomial kernel function.
 * @details Currently only single GPU execution is supported.
 * @tparam T the type of the data
 */
template <typename T>
class hierarchical_device_kernel_polynomial {
  public:
    /// The type of the data.
    using real_type = T;

    /**
     * @brief Construct a new device kernel calculating the C-SVM kernel using the polynomial C-SVM kernel.
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
    hierarchical_device_kernel_polynomial(const real_type *q, real_type *ret, const real_type *d, const real_type *data_d, const real_type QA_cost, const real_type cost, const kernel_index_type num_rows, const kernel_index_type num_cols, const real_type add, const int degree, const real_type gamma, const real_type coef0) :
        q_{ q }, ret_{ ret }, d_{ d }, data_d_{ data_d }, QA_cost_{ QA_cost }, cost_{ cost }, num_rows_{ num_rows }, num_cols_{ num_cols }, add_{ add }, degree_{ degree }, gamma_{ gamma }, coef0_{ coef0 } {}

    /**
     * @brief Function call operator overload performing the actual calculation.
     * @param[in] group the [`sycl::group`](https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#group-class)
     *                  identifying an instance of the currently execution work-group
     */
    void operator()(::sycl::group<2> group) const {
        // allocate shared memory
        real_type data_intern_i[THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE];
        real_type data_intern_j[THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE];

        // allocate memory for work-item local variables
        // -> accessible across different 'parallel_for_work_item' invocations
        ::sycl::private_memory<real_type[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE], 2> private_matr{ group };
        ::sycl::private_memory<real_type[INTERNAL_BLOCK_SIZE], 2> private_data_j{ group };
        ::sycl::private_memory<kernel_index_type, 2> private_i{ group };
        ::sycl::private_memory<kernel_index_type, 2> private_j{ group };
        ::sycl::private_memory<bool, 2> private_cond{ group };

        // initialize private variables
        group.parallel_for_work_item([&](::sycl::h_item<2> idx) {
            // indices and diagonal condition
            private_i(idx) = group[0] * idx.get_local_range(0) * INTERNAL_BLOCK_SIZE;
            private_j(idx) = group[1] * idx.get_local_range(1) * INTERNAL_BLOCK_SIZE;
            private_cond(idx) = private_i(idx) >= private_j(idx);
            if (private_cond(idx)) {
                private_i(idx) += idx.get_local_id(0) * INTERNAL_BLOCK_SIZE;
                private_j(idx) += idx.get_local_id(1) * INTERNAL_BLOCK_SIZE;
            }

            // matrix
            #pragma unroll INTERNAL_BLOCK_SIZE
            for (kernel_index_type i = 0; i < INTERNAL_BLOCK_SIZE; ++i) {
                #pragma unroll INTERNAL_BLOCK_SIZE
                for (kernel_index_type j = 0; j < INTERNAL_BLOCK_SIZE; ++j) {
                    private_matr(idx)[i][j] = real_type{ 0.0 };
                }
            }
        });

        // implicit group barrier

        // load data from global in shared memory
        for (kernel_index_type vec_index = 0; vec_index < num_cols_ * num_rows_; vec_index += num_rows_) {
            group.parallel_for_work_item([&](::sycl::h_item<2> idx) {
                if (private_cond(idx)) {
                    #pragma unroll INTERNAL_BLOCK_SIZE
                    for (kernel_index_type block_id = 0; block_id < INTERNAL_BLOCK_SIZE; ++block_id) {
                        const std::size_t idx_1 = block_id % THREAD_BLOCK_SIZE;
                        if (idx.get_local_id(1) == idx_1) {
                            data_intern_i[idx.get_local_id(0)][block_id] = data_d_[block_id + vec_index + private_i(idx)];
                        }
                        const std::size_t idx_2 = block_id % THREAD_BLOCK_SIZE;
                        if (idx.get_local_id(0) == idx_2) {
                            data_intern_j[idx.get_local_id(1)][block_id] = data_d_[block_id + vec_index + private_j(idx)];
                        }
                    }
                }
            });

            // implicit group barrier

            // load data from shared in private memory and perform scalar product
            group.parallel_for_work_item([&](::sycl::h_item<2> idx) {
                if (private_cond(idx)) {
                    #pragma unroll INTERNAL_BLOCK_SIZE
                    for (kernel_index_type data_index = 0; data_index < INTERNAL_BLOCK_SIZE; ++data_index) {
                        private_data_j(idx)[data_index] = data_intern_j[idx.get_local_id(1)][data_index];
                    }

                    #pragma unroll INTERNAL_BLOCK_SIZE
                    for (kernel_index_type l = 0; l < INTERNAL_BLOCK_SIZE; ++l) {
                        const real_type data_i = data_intern_i[idx.get_local_id(0)][l];
                        #pragma unroll INTERNAL_BLOCK_SIZE
                        for (kernel_index_type k = 0; k < INTERNAL_BLOCK_SIZE; ++k) {
                            private_matr(idx)[k][l] += data_i * private_data_j(idx)[k];
                        }
                    }
                }
            });

            // implicit group barrier
        }

        // kernel function
        group.parallel_for_work_item([&](::sycl::h_item<2> idx) {
            if (private_cond(idx)) {
                #pragma unroll INTERNAL_BLOCK_SIZE
                for (kernel_index_type x = 0; x < INTERNAL_BLOCK_SIZE; ++x) {
                    real_type ret_jx = 0.0;
                    #pragma unroll INTERNAL_BLOCK_SIZE
                    for (kernel_index_type y = 0; y < INTERNAL_BLOCK_SIZE; ++y) {
                        const real_type temp = (::sycl::pow(gamma_ * private_matr(idx)[x][y] + coef0_, static_cast<real_type>(degree_)) + QA_cost_ - q_[private_i(idx) + y] - q_[private_j(idx) + x]) * add_;
                        if (private_i(idx) + x > private_j(idx) + y) {
                            // upper triangular matrix
                            detail::atomic_op<real_type>{ ret_[private_i(idx) + y] } += temp * d_[private_j(idx) + x];
                            ret_jx += temp * d_[private_i(idx) + y];
                        } else if (private_i(idx) + x == private_j(idx) + y) {
                            // diagonal
                            ret_jx += (temp + cost_ * add_) * d_[private_i(idx) + y];
                        }
                    }
                    detail::atomic_op<real_type>{ ret_[private_j(idx) + x] } += ret_jx;
                }
            }
        });
    }

  private:
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
 * @brief Calculates the C-SVM kernel using the hierarchical formulation and the radial basis functions kernel function.
 * @details Currently only single GPU execution is supported.
 * @tparam T the type of the data
 */
template <typename T>
class hierarchical_device_kernel_rbf {
  public:
    /// The type of the data.
    using real_type = T;

    /**
     * @brief Construct a new device kernel calculating the C-SVM kernel using the radial basis functions kernel function.
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
    hierarchical_device_kernel_rbf(const real_type *q, real_type *ret, const real_type *d, const real_type *data_d, const real_type QA_cost, const real_type cost, const kernel_index_type num_rows, const kernel_index_type num_cols, const real_type add, const real_type gamma) :
        q_{ q }, ret_{ ret }, d_{ d }, data_d_{ data_d }, QA_cost_{ QA_cost }, cost_{ cost }, num_rows_{ num_rows }, num_cols_{ num_cols }, add_{ add }, gamma_{ gamma } {}

    /**
     * @brief Function call operator overload performing the actual calculation.
     * @param[in] group the [`sycl::group`](https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#group-class)
     *                  identifying an instance of the currently execution work-group
     */
    void operator()(::sycl::group<2> group) const {
        // allocate shared memory
        real_type data_intern_i[THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE];
        real_type data_intern_j[THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE];

        // allocate memory for work-item local variables
        // -> accessible across different 'parallel_for_work_item' invocations
        ::sycl::private_memory<real_type[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE], 2> private_matr{ group };
        ::sycl::private_memory<real_type[INTERNAL_BLOCK_SIZE], 2> private_data_j{ group };
        ::sycl::private_memory<kernel_index_type, 2> private_i{ group };
        ::sycl::private_memory<kernel_index_type, 2> private_j{ group };
        ::sycl::private_memory<bool, 2> private_cond{ group };

        // initialize private variables
        group.parallel_for_work_item([&](::sycl::h_item<2> idx) {
            // indices and diagonal condition
            private_i(idx) = group[0] * idx.get_local_range(0) * INTERNAL_BLOCK_SIZE;
            private_j(idx) = group[1] * idx.get_local_range(1) * INTERNAL_BLOCK_SIZE;
            private_cond(idx) = private_i(idx) >= private_j(idx);
            if (private_cond(idx)) {
                private_i(idx) += idx.get_local_id(0) * INTERNAL_BLOCK_SIZE;
                private_j(idx) += idx.get_local_id(1) * INTERNAL_BLOCK_SIZE;
            }

            // matrix
            #pragma unroll INTERNAL_BLOCK_SIZE
            for (kernel_index_type i = 0; i < INTERNAL_BLOCK_SIZE; ++i) {
                #pragma unroll INTERNAL_BLOCK_SIZE
                for (kernel_index_type j = 0; j < INTERNAL_BLOCK_SIZE; ++j) {
                    private_matr(idx)[i][j] = real_type{ 0.0 };
                }
            }
        });

        // implicit group barrier

        // load data from global in shared memory
        for (kernel_index_type vec_index = 0; vec_index < num_cols_ * num_rows_; vec_index += num_rows_) {
            group.parallel_for_work_item([&](::sycl::h_item<2> idx) {
                if (private_cond(idx)) {
                    #pragma unroll INTERNAL_BLOCK_SIZE
                    for (kernel_index_type block_id = 0; block_id < INTERNAL_BLOCK_SIZE; ++block_id) {
                        const std::size_t idx_1 = block_id % THREAD_BLOCK_SIZE;
                        if (idx.get_local_id(1) == idx_1) {
                            data_intern_i[idx.get_local_id(0)][block_id] = data_d_[block_id + vec_index + private_i(idx)];
                        }
                        const std::size_t idx_2 = block_id % THREAD_BLOCK_SIZE;
                        if (idx.get_local_id(0) == idx_2) {
                            data_intern_j[idx.get_local_id(1)][block_id] = data_d_[block_id + vec_index + private_j(idx)];
                        }
                    }
                }
            });

            // implicit group barrier

            // load data from shared in private memory and perform scalar product
            group.parallel_for_work_item([&](::sycl::h_item<2> idx) {
                if (private_cond(idx)) {
                    #pragma unroll INTERNAL_BLOCK_SIZE
                    for (kernel_index_type data_index = 0; data_index < INTERNAL_BLOCK_SIZE; ++data_index) {
                        private_data_j(idx)[data_index] = data_intern_j[idx.get_local_id(1)][data_index];
                    }

                    #pragma unroll INTERNAL_BLOCK_SIZE
                    for (kernel_index_type l = 0; l < INTERNAL_BLOCK_SIZE; ++l) {
                        const real_type data_i = data_intern_i[idx.get_local_id(0)][l];
                        #pragma unroll INTERNAL_BLOCK_SIZE
                        for (kernel_index_type k = 0; k < INTERNAL_BLOCK_SIZE; ++k) {
                            private_matr(idx)[k][l] += (data_i - private_data_j(idx)[k]) * (data_i - private_data_j(idx)[k]);
                        }
                    }
                }
            });

            // implicit group barrier
        }

        // kernel function
        group.parallel_for_work_item([&](::sycl::h_item<2> idx) {
            if (private_cond(idx)) {
                #pragma unroll INTERNAL_BLOCK_SIZE
                for (kernel_index_type x = 0; x < INTERNAL_BLOCK_SIZE; ++x) {
                    real_type ret_jx = 0.0;
                    #pragma unroll INTERNAL_BLOCK_SIZE
                    for (kernel_index_type y = 0; y < INTERNAL_BLOCK_SIZE; ++y) {
                        const real_type temp = (::sycl::exp(-gamma_ * private_matr(idx)[x][y]) + QA_cost_ - q_[private_i(idx) + y] - q_[private_j(idx) + x]) * add_;
                        if (private_i(idx) + x > private_j(idx) + y) {
                            // upper triangular matrix
                            detail::atomic_op<real_type>{ ret_[private_i(idx) + y] } += temp * d_[private_j(idx) + x];
                            ret_jx += temp * d_[private_i(idx) + y];
                        } else if (private_i(idx) + x == private_j(idx) + y) {
                            // diagonal
                            ret_jx += (temp + cost_ * add_) * d_[private_i(idx) + y];
                        }
                    }
                    detail::atomic_op<real_type>{ ret_[private_j(idx) + x] } += ret_jx;
                }
            }
        });
    }

  private:
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

#endif  // PLSSVM_BACKENDS_SYCL_SVM_KERNEL_HIERARCHICAL_HPP_