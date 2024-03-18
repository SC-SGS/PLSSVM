/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Functions for implicitly assembling the kernel matrix using the SYCL backend.
 */

#ifndef PLSSVM_BACKENDS_SYCL_CG_IMPLICIT_KERNEL_MATRIX_ASSEMBLY_BLAS_HPP_
#define PLSSVM_BACKENDS_SYCL_CG_IMPLICIT_KERNEL_MATRIX_ASSEMBLY_BLAS_HPP_
#pragma once

#include "plssvm/backends/SYCL/detail/atomics.hpp"                // plssvm::sycl::detail::atomic_op
#include "plssvm/backends/SYCL/detail/standard_layout_tuple.hpp"  // plssvm::sycl::detail::standard_layout_tuple
#include "plssvm/backends/SYCL/kernel/kernel_functions.hpp"       // plssvm::sycl::detail::{feature_reduce, apply_kernel_function}
#include "plssvm/constants.hpp"                                   // plssvm::{real_type, THREAD_BLOCK_SIZE, INTERNAL_BLOCK_SIZE, FEATURE_BLOCK_SIZE, PADDING_SIZE}
#include "plssvm/kernel_function_types.hpp"                       // plssvm::kernel_function_type

#include "sycl/sycl.hpp"  // sycl::nd_item

namespace plssvm::sycl::detail {

/**
 * @brief Perform an implicit BLAS SYMM-like operation: `C = alpha * A * B + C` where `A` is the implicitly calculated kernel matrix using the @p kernel_function (never actually stored, reducing the amount of needed global memory), @p B and @p C are matrices, and @p alpha is a scalar.
 * @tparam kernel_function the type of the used kernel function
 * @tparam Args the types of the parameters necessary for the specific kernel function
 */
template <kernel_function_type kernel_function, typename... Args>
class device_kernel_assembly_symm {
  public:
    /**
     * @brief Initialize the SYCL kernel function object.
     * @param[in] cgh the SYCL handler used to allocate the local memory
     * @param[in] alpha the scalar alpha value
     * @param[in] q the vector used in the dimensional reduction
     * @param[in] data_d the data points to calculate the implicit kernel matrix from
     * @param[in] num_rows the total number of data points (= total number of rows)
     * @param[in] device_num_rows the number of rows the current device is responsible for
     * @param[in] row_offset the first row in @p data_d the current device is responsible for
     * @param[in] num_features the number of features per data point
     * @param[in] QA_cost the scalar used in the dimensional reduction
     * @param[in] cost the cost factor the diagonal is scaled with
     * @param[in] B the matrix @p B
     * @param[in,out] C the matrix @p C
     * @param[in] num_classes the number of classes in the data set
     * @param[in] kernel_function_parameter the parameters necessary to apply the @p kernel_function
     */
    device_kernel_assembly_symm(::sycl::handler &cgh, const real_type alpha, const real_type *q, const real_type *data_d, const unsigned long long num_rows, const unsigned long long device_num_rows, const unsigned long long row_offset, const unsigned long long num_features, const real_type QA_cost, const real_type cost, const real_type *B, real_type *C, const unsigned long long num_classes, Args... kernel_function_parameter) :
        data_cache_i_{ ::sycl::range<1>{ FEATURE_BLOCK_SIZE * INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE }, cgh },  // [FEATURE_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE]
        data_cache_j_{ ::sycl::range<1>{ FEATURE_BLOCK_SIZE * INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE }, cgh },  // [FEATURE_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE]
        alpha_{ alpha },
        q_{ q },
        data_d_{ data_d },
        num_rows_{ num_rows },
        device_num_rows_{ device_num_rows },
        row_offset_{ row_offset },
        num_features_{ num_features },
        QA_cost_{ QA_cost },
        cost_{ cost },
        B_{ B },
        C_{ C },
        num_classes_{ num_classes },
        kernel_function_parameter_{ detail::make_standard_layout_tuple(std::forward<Args>(kernel_function_parameter)...) } { }

    /**
     * @brief Function call operator overload performing the actual calculation.
     * @param[in] nd_idx indices representing the current point in the execution space
     */
    void operator()(::sycl::nd_item<2> nd_idx) const {
        const unsigned long long i = nd_idx.get_global_id(1) * INTERNAL_BLOCK_SIZE;
        const unsigned long long i_linear = nd_idx.get_group(1) * nd_idx.get_local_range(1) * INTERNAL_BLOCK_SIZE + nd_idx.get_local_id(1);
        const unsigned long long j = nd_idx.get_global_id(0) * INTERNAL_BLOCK_SIZE;
        const unsigned long long j_cached_idx_linear = nd_idx.get_group(0) * nd_idx.get_local_range(0) * INTERNAL_BLOCK_SIZE + nd_idx.get_local_id(1);

        if (nd_idx.get_group(1) >= nd_idx.get_group(0)) {
            real_type temp[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE] = { { 0.0 } };

            {
                for (unsigned long long dim = 0; dim < num_features_; dim += FEATURE_BLOCK_SIZE) {
                    // load data into shared memory
                    for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                        const unsigned long long global_i = row_offset_ + i_linear + internal * THREAD_BLOCK_SIZE;
                        const unsigned long long global_j = row_offset_ + j_cached_idx_linear + internal * THREAD_BLOCK_SIZE;

                        data_cache_i_[nd_idx.get_local_id(0) * INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE + internal * THREAD_BLOCK_SIZE + nd_idx.get_local_id(1)] = data_d_[(dim + nd_idx.get_local_id(0)) * (num_rows_ + 1 + PADDING_SIZE) + global_i];
                        data_cache_i_[(nd_idx.get_local_id(0) + THREAD_BLOCK_SIZE) * INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE + internal * THREAD_BLOCK_SIZE + nd_idx.get_local_id(1)] = data_d_[(dim + nd_idx.get_local_id(0) + THREAD_BLOCK_SIZE) * (num_rows_ + 1 + PADDING_SIZE) + global_i];
                        data_cache_j_[nd_idx.get_local_id(0) * INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE + internal * THREAD_BLOCK_SIZE + nd_idx.get_local_id(1)] = data_d_[(dim + nd_idx.get_local_id(0)) * (num_rows_ + 1 + PADDING_SIZE) + global_j];
                        data_cache_j_[(nd_idx.get_local_id(0) + THREAD_BLOCK_SIZE) * INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE + internal * THREAD_BLOCK_SIZE + nd_idx.get_local_id(1)] = data_d_[(dim + nd_idx.get_local_id(0) + THREAD_BLOCK_SIZE) * (num_rows_ + 1 + PADDING_SIZE) + global_j];
                    }
                    nd_idx.barrier();

                    // calculation
                    for (unsigned block_dim = 0; block_dim < FEATURE_BLOCK_SIZE; ++block_dim) {
                        for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                            for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                                temp[internal_i][internal_j] += detail::feature_reduce<kernel_function>(data_cache_i_[block_dim * INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE + nd_idx.get_local_id(1) * INTERNAL_BLOCK_SIZE + internal_i],
                                                                                                        data_cache_j_[block_dim * INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE + nd_idx.get_local_id(0) * INTERNAL_BLOCK_SIZE + internal_j]);
                            }
                        }
                    }
                    nd_idx.barrier();
                }
            }

            // update temp using the rbf kernel function taking the dimensional reduction into account and apply the cost to the diagonal
            for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                    const unsigned long long global_i = row_offset_ + i + internal_i;
                    const unsigned long long device_global_i = i + internal_i;
                    const unsigned long long global_j = row_offset_ + j + internal_j;
                    const unsigned long long device_global_j = j + internal_j;

                    if (device_global_i < (num_rows_ - row_offset_) && device_global_j < device_num_rows_ && global_i >= global_j) {
                        temp[internal_i][internal_j] = detail::apply_kernel_function<kernel_function>(temp[internal_i][internal_j], kernel_function_parameter_) + QA_cost_ - q_[global_i] - q_[global_j];
                        if (global_i == global_j) {
                            temp[internal_i][internal_j] += cost_;
                        }
                    } else {
                        temp[internal_i][internal_j] = 0.0;
                    }
                }
            }

            // calculate C += alpha * temp * B for the UPPER triangular matrix
            {
                auto &B_cache = data_cache_i_;      // [INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE][FEATURE_BLOCK_SIZE]
                auto &C_out_cache = data_cache_j_;  // [INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE][FEATURE_BLOCK_SIZE]

                for (unsigned long long dim = 0; dim < num_classes_; dim += FEATURE_BLOCK_SIZE) {
                    // load data into shared memory
                    for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                        const unsigned long long global_i = row_offset_ + i_linear + internal * THREAD_BLOCK_SIZE;

                        B_cache[(internal * THREAD_BLOCK_SIZE + nd_idx.get_local_id(1)) * FEATURE_BLOCK_SIZE + nd_idx.get_local_id(0)] = alpha_ * B_[global_i * (num_classes_ + PADDING_SIZE) + dim + nd_idx.get_local_id(0)];
                        B_cache[(internal * THREAD_BLOCK_SIZE + nd_idx.get_local_id(1)) * FEATURE_BLOCK_SIZE + nd_idx.get_local_id(0) + THREAD_BLOCK_SIZE] = alpha_ * B_[global_i * (num_classes_ + PADDING_SIZE) + dim + nd_idx.get_local_id(0) + THREAD_BLOCK_SIZE];

                        C_out_cache[(internal * THREAD_BLOCK_SIZE + nd_idx.get_local_id(1)) * FEATURE_BLOCK_SIZE + nd_idx.get_local_id(0)] = 0.0;
                        C_out_cache[(internal * THREAD_BLOCK_SIZE + nd_idx.get_local_id(1)) * FEATURE_BLOCK_SIZE + nd_idx.get_local_id(0) + THREAD_BLOCK_SIZE] = 0.0;
                    }
                    nd_idx.barrier();

                    // calculate intermediate results and store them in shared memory
                    for (unsigned class_idx = 0; class_idx < FEATURE_BLOCK_SIZE; ++class_idx) {
                        for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                            for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                                C_out_cache[(nd_idx.get_local_id(0) * INTERNAL_BLOCK_SIZE + internal_j) * FEATURE_BLOCK_SIZE + (class_idx + nd_idx.get_local_id(1)) % FEATURE_BLOCK_SIZE] +=
                                    temp[internal_i][internal_j] * B_cache[(nd_idx.get_local_id(1) * INTERNAL_BLOCK_SIZE + internal_i) * FEATURE_BLOCK_SIZE + (class_idx + nd_idx.get_local_id(1)) % FEATURE_BLOCK_SIZE];
                            }
                        }
                        nd_idx.barrier();
                    }

                    // add intermediate cached results to C
                    for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                        const unsigned long long global_j = row_offset_ + j + internal;
                        detail::atomic_op<real_type>{ C_[global_j * (num_classes_ + PADDING_SIZE) + dim + nd_idx.get_local_id(1)] } += C_out_cache[(nd_idx.get_local_id(0) * INTERNAL_BLOCK_SIZE + internal) * FEATURE_BLOCK_SIZE + nd_idx.get_local_id(1)];
                        detail::atomic_op<real_type>{ C_[global_j * (num_classes_ + PADDING_SIZE) + dim + nd_idx.get_local_id(1) + THREAD_BLOCK_SIZE] } += C_out_cache[(nd_idx.get_local_id(0) * INTERNAL_BLOCK_SIZE + internal) * FEATURE_BLOCK_SIZE + nd_idx.get_local_id(1) + THREAD_BLOCK_SIZE];
                    }
                    nd_idx.barrier();
                }
            }

            // set potential diagonal entries in temp to 0.0 such that we don't apply the main diagonal twice to C
            for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                    const unsigned long long global_i = row_offset_ + i + internal_i;
                    const unsigned long long global_j = row_offset_ + j + internal_j;

                    if (global_i == global_j) {
                        temp[internal_i][internal_j] = 0.0;
                    }
                }
            }

            // calculate C += alpha * temp * B for the LOWER triangular matrix
            {
                auto &B_cache = data_cache_i_;      // [FEATURE_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE]
                auto &C_out_cache = data_cache_j_;  // [FEATURE_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE]

                for (unsigned long long dim = 0; dim < num_classes_; dim += FEATURE_BLOCK_SIZE) {
                    // load data into shared memory
                    for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                        const unsigned long long global_j = row_offset_ + j_cached_idx_linear + internal * THREAD_BLOCK_SIZE;

                        B_cache[nd_idx.get_local_id(0) * INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE + internal * THREAD_BLOCK_SIZE + nd_idx.get_local_id(1)] = alpha_ * B_[global_j * (num_classes_ + PADDING_SIZE) + dim + nd_idx.get_local_id(0)];
                        B_cache[(nd_idx.get_local_id(0) + THREAD_BLOCK_SIZE) * INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE + internal * THREAD_BLOCK_SIZE + nd_idx.get_local_id(1)] = alpha_ * B_[global_j * (num_classes_ + PADDING_SIZE) + dim + nd_idx.get_local_id(0) + THREAD_BLOCK_SIZE];

                        C_out_cache[nd_idx.get_local_id(0) * INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE + internal * THREAD_BLOCK_SIZE + nd_idx.get_local_id(1)] = 0.0;
                        C_out_cache[(nd_idx.get_local_id(0) + THREAD_BLOCK_SIZE) * INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE + internal * THREAD_BLOCK_SIZE + nd_idx.get_local_id(1)] = 0.0;
                    }
                    nd_idx.barrier();

                    // calculate intermediate results and store them in shared memory
                    for (unsigned class_idx = 0; class_idx < FEATURE_BLOCK_SIZE; ++class_idx) {
                        for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                            for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                                C_out_cache[((class_idx + nd_idx.get_local_id(0)) % FEATURE_BLOCK_SIZE) * INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE + internal_i * THREAD_BLOCK_SIZE + nd_idx.get_local_id(1)] +=
                                    temp[internal_i][internal_j] * B_cache[((class_idx + nd_idx.get_local_id(0)) % FEATURE_BLOCK_SIZE) * INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE + nd_idx.get_local_id(0) * INTERNAL_BLOCK_SIZE + internal_j];
                            }
                        }
                        nd_idx.barrier();
                    }

                    // add intermediate cached results to C
                    for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                        const unsigned long long global_i = row_offset_ + i + internal;
                        detail::atomic_op<real_type>{ C_[global_i * (num_classes_ + PADDING_SIZE) + dim + nd_idx.get_local_id(0)] } += C_out_cache[nd_idx.get_local_id(0) * INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE + internal * THREAD_BLOCK_SIZE + nd_idx.get_local_id(1)];
                        detail::atomic_op<real_type>{ C_[global_i * (num_classes_ + PADDING_SIZE) + dim + nd_idx.get_local_id(0) + THREAD_BLOCK_SIZE] } += C_out_cache[(nd_idx.get_local_id(0) + THREAD_BLOCK_SIZE) * INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE + internal * THREAD_BLOCK_SIZE + nd_idx.get_local_id(1)];
                    }
                    nd_idx.barrier();
                }
            }
        }
    }

  private:
    /// Local memory used for internal memory access optimizations.
    ::sycl::local_accessor<real_type, 1> data_cache_i_;
    /// Local memory used for internal memory access optimizations.
    ::sycl::local_accessor<real_type, 1> data_cache_j_;

    /// @cond Doxygen_suppress
    const real_type alpha_;
    const real_type *q_;
    const real_type *data_d_;
    const unsigned long long num_rows_;
    const unsigned long long device_num_rows_;
    const unsigned long long row_offset_;
    const unsigned long long num_features_;
    const real_type QA_cost_;
    const real_type cost_;
    const real_type *B_;
    real_type *C_;
    const unsigned long long num_classes_;
    const detail::standard_layout_tuple<Args...> kernel_function_parameter_;
    /// @endcond
};

}  // namespace plssvm::sycl::detail

#endif  // PLSSVM_BACKENDS_SYCL_CG_IMPLICIT_KERNEL_MATRIX_ASSEMBLY_BLAS_HPP_
