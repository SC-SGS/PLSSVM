/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Functions for explicitly assembling the kernel matrix using the SYCL backend.
 */

#ifndef PLSSVM_BACKENDS_SYCL_CG_EXPLICIT_KERNEL_MATRIX_ASSEMBLY_HPP_
#define PLSSVM_BACKENDS_SYCL_CG_EXPLICIT_KERNEL_MATRIX_ASSEMBLY_HPP_
#pragma once

#include "plssvm/backends/SYCL/detail/standard_layout_tuple.hpp"  // plssvm::sycl::detail::standard_layout_tuple
#include "plssvm/backends/SYCL/kernel/kernel_functions.hpp"       // plssvm::sycl::detail::{feature_reduce, apply_kernel_function}
#include "plssvm/constants.hpp"                                   // plssvm::{real_type, THREAD_BLOCK_SIZE, INTERNAL_BLOCK_SIZE, FEATURE_BLOCK_SIZE, PADDING_SIZE}
#include "plssvm/kernel_function_types.hpp"                       // plssvm::kernel_function_type

#include "sycl/sycl.hpp"  // sycl::nd_item

namespace plssvm::sycl::detail {

/**
 * @brief Create the explicit kernel matrix using the @p kernel_function.
 * @tparam kernel_function the type of the used kernel function
 * @tparam Args the types of the parameters necessary for the specific kernel function; stored in a `standard_layout_tuple`
 */
template <kernel_function_type kernel_function, typename... Args>
class device_kernel_assembly {
  public:
    /**
     * @brief Initialize the SYCL kernel function object.
     * @param[in] cgh the SYCL handler used to allocate the local memory
     * @param[out] ret the calculated kernel matrix
     * @param[in] data_d the data points to calculate the kernel matrix from
     * @param[in] num_rows the number of data points
     * @param[in] device_num_rows the number of rows the current device is responsible for
     * @param[in] row_offset the first row in @p data_d the current device is responsible for
     * @param[in] num_features the number of features per data point
     * @param[in] q the vector used in the dimensional reduction
     * @param[in] QA_cost the scalar used in the dimensional reduction
     * @param[in] cost the cost factor the diagonal is scaled with
     * @param[in] kernel_function_parameter the parameters necessary to apply the @p kernel_function
     */
    device_kernel_assembly(::sycl::handler &cgh, real_type *ret, const real_type *data_d, const unsigned long long num_rows, const unsigned long long device_num_rows, const unsigned long long row_offset, const unsigned long long num_features, const real_type *q, const real_type QA_cost, const real_type cost, Args... kernel_function_parameter) :
        data_cache_i_{ ::sycl::range<2>{ FEATURE_BLOCK_SIZE, INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE }, cgh },
        data_cache_j_{ ::sycl::range<2>{ FEATURE_BLOCK_SIZE, INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE }, cgh },
        ret_{ ret },
        data_d_{ data_d },
        num_rows_{ num_rows },
        device_num_rows_{ device_num_rows },
        row_offset_{ row_offset },
        num_features_{ num_features },
        q_{ q },
        QA_cost_{ QA_cost },
        cost_{ cost },
        kernel_function_parameter_{ detail::make_standard_layout_tuple(std::forward<Args>(kernel_function_parameter)...) } {
    }

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

            for (unsigned long long dim = 0; dim < num_features_; dim += FEATURE_BLOCK_SIZE) {
                // load data into shared memory
                for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                    const unsigned long long global_i = row_offset_ + i_linear + internal * THREAD_BLOCK_SIZE;
                    const unsigned long long global_j = row_offset_ + j_cached_idx_linear + internal * THREAD_BLOCK_SIZE;

                    data_cache_i_[nd_idx.get_local_id(0)][internal * THREAD_BLOCK_SIZE + nd_idx.get_local_id(1)] = data_d_[(dim + nd_idx.get_local_id(0)) * (num_rows_ + 1 + PADDING_SIZE) + global_i];
                    data_cache_i_[nd_idx.get_local_id(0) + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + nd_idx.get_local_id(1)] = data_d_[(dim + nd_idx.get_local_id(0) + THREAD_BLOCK_SIZE) * (num_rows_ + 1 + PADDING_SIZE) + global_i];
                    data_cache_j_[nd_idx.get_local_id(0)][internal * THREAD_BLOCK_SIZE + nd_idx.get_local_id(1)] = data_d_[(dim + nd_idx.get_local_id(0)) * (num_rows_ + 1 + PADDING_SIZE) + global_j];
                    data_cache_j_[nd_idx.get_local_id(0) + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + nd_idx.get_local_id(1)] = data_d_[(dim + nd_idx.get_local_id(0) + THREAD_BLOCK_SIZE) * (num_rows_ + 1 + PADDING_SIZE) + global_j];
                }
                nd_idx.barrier();

                // calculation
                for (unsigned block_dim = 0; block_dim < FEATURE_BLOCK_SIZE; ++block_dim) {
                    for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                        for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                            temp[internal_i][internal_j] += detail::feature_reduce<kernel_function>(data_cache_i_[block_dim][nd_idx.get_local_id(1) * INTERNAL_BLOCK_SIZE + internal_i],
                                                                                                    data_cache_j_[block_dim][nd_idx.get_local_id(0) * INTERNAL_BLOCK_SIZE + internal_j]);
                        }
                    }
                }
                nd_idx.barrier();
            }

            for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                    const unsigned long long device_global_i = i + internal_i;
                    const unsigned long long global_i = row_offset_ + i + internal_i;
                    const unsigned long long device_global_j = j + internal_j;
                    const unsigned long long global_j = row_offset_ + j + internal_j;

                    if (device_global_i < (num_rows_ - row_offset_) && device_global_j < device_num_rows_ && global_i >= global_j) {
                        real_type temp_ij = temp[internal_i][internal_j];
                        temp_ij = detail::apply_kernel_function<kernel_function>(temp_ij, kernel_function_parameter_) + QA_cost_ - q_[global_i] - q_[global_j];
                        if (global_i == global_j) {
                            temp_ij += cost_;
                        }
                        ret_[device_global_j * (num_rows_ - row_offset_ + PADDING_SIZE) - device_global_j * (device_global_j + 1) / 2 + device_global_i] = temp_ij;
                    }
                }
            }
        }
    }

  private:
    /// Local memory used for internal memory access optimizations.
    ::sycl::local_accessor<real_type, 2> data_cache_i_;
    /// Local memory used for internal memory access optimizations.
    ::sycl::local_accessor<real_type, 2> data_cache_j_;

    /// @cond Doxygen_suppress
    real_type *ret_;
    const real_type *data_d_;
    const unsigned long long num_rows_;
    const unsigned long long device_num_rows_;
    const unsigned long long row_offset_;
    const unsigned long long num_features_;
    const real_type *q_;
    const real_type QA_cost_;
    const real_type cost_;
    const detail::standard_layout_tuple<Args...> kernel_function_parameter_;
    /// @endcond
};

}  // namespace plssvm::sycl::detail

#endif  // PLSSVM_BACKENDS_SYCL_CG_EXPLICIT_KERNEL_MATRIX_ASSEMBLY_HPP_
