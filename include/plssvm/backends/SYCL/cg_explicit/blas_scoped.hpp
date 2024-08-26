/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Functions for explicitly performing a BLAS GEMM like matrix-matrix multiplication using the SYCL backend with AdaptiveCpp's scoped parallelism kernels.
 */

#ifndef PLSSVM_BACKENDS_SYCL_CG_EXPLICIT_BLAS_SCOPED_HPP_
#define PLSSVM_BACKENDS_SYCL_CG_EXPLICIT_BLAS_SCOPED_HPP_
#pragma once

#include "plssvm/constants.hpp"  // plssvm::{real_type, THREAD_BLOCK_SIZE, INTERNAL_BLOCK_SIZE, FEATURE_BLOCK_SIZE, PADDING_SIZE}

#include "sycl/sycl.hpp"  // sycl::pown, sycl::exp
                          // sycl::memory_environment, sycl::require_local_mem, sycl::require_private_mem, sycl::distribute_items_and_wait, sycl::s_item

#include <sycl/ext/intel/fpga_extensions.hpp> // TODO: guard


namespace plssvm::sycl::detail::scoped {

/**
 * @brief Perform an explicit BLAS GEMM operation: `C = alpha * A * B + beta * C` where A is a `m x k` matrix, B is a `k x n` matrix, C is a `m x n` matrix, and alpha and beta are scalars.
 */
class device_kernel_gemm {
  public:
    /**
     * @brief Initialize the SYCL kernel function object.
     * @param[in] m the number of rows in @p A and @p C
     * @param[in] n the number of columns in @p B and @p C
     * @param[in] k the number of rows in @p A and number of columns in @p B
     * @param[in] alpha the scalar alpha value
     * @param[in] A the matrix @p A
     * @param[in] B the matrix @p B
     * @param[in] beta the scalar beta value
     * @param[in,out] C the matrix @p C, also used as result matrix
     */
    device_kernel_gemm(const unsigned long long m, const unsigned long long n, const unsigned long long k, const real_type alpha, const real_type *A, const real_type *B, const real_type beta, real_type *C) :
        m_{ m },
        n_{ n },
        k_{ k },
        alpha_{ alpha },
        A_{ A },
        B_{ B },
        beta_{ beta },
        C_{ C } { }

    /**
     * @brief Function call operator overload performing the actual calculation.
     * @tparam T the implementation defined type of the group to iterate
     * @param[in] group group representing the current point in the execution space
     */
    template <typename T>
    void operator()(T group) const {
        // compute: C = alpha * A * B + beta * C with A in m x k, B in n x k, and C in n x m, alpha, beta as scalar
        ::sycl::memory_environment(group,
                                   ::sycl::require_local_mem<real_type[FEATURE_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE]>(),
                                   ::sycl::require_local_mem<real_type[FEATURE_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE]>(),
                                   ::sycl::require_private_mem<unsigned long long>(),
                                   ::sycl::require_private_mem<unsigned long long>(),
                                   ::sycl::require_private_mem<unsigned long long>(),
                                   ::sycl::require_private_mem<unsigned long long>(),
                                   ::sycl::require_private_mem<std::array<std::array<real_type, INTERNAL_BLOCK_SIZE>, INTERNAL_BLOCK_SIZE>>({}),
                                   [&](auto &A_cache, auto &B_cache, auto &i, auto &i_cached_idx_linear, auto &j, auto &j_linear, auto &temp) {
                                       // initialize private and local variables
                                       ::sycl::distribute_items_and_wait(group, [&](::sycl::s_item<2> idx) {
                                           // indices and diagonal condition
                                           i(idx) = (group[0] * group.get_logical_local_range(0) + idx.get_local_id(group, 0)) * INTERNAL_BLOCK_SIZE;
                                           i_cached_idx_linear(idx) = group[0] * group.get_logical_local_range(0) * INTERNAL_BLOCK_SIZE + idx.get_local_id(group, 1);
                                           j(idx) = (group[1] * group.get_logical_local_range(1) + idx.get_local_id(group, 1)) * INTERNAL_BLOCK_SIZE;
                                           j_linear(idx) = group[1] * group.get_logical_local_range(1) * INTERNAL_BLOCK_SIZE + idx.get_local_id(group, 1);
                                       });

                                       for (unsigned long long dim = 0; dim < k_; dim += FEATURE_BLOCK_SIZE) {
                                           // load data into shared memory
                                           ::sycl::distribute_items_and_wait(group, [&](::sycl::s_item<2> idx) {
                                               for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                                                   const unsigned long long global_i = i_cached_idx_linear(idx) + internal * THREAD_BLOCK_SIZE;
                                                   const unsigned long long global_j = j_linear(idx) + internal * THREAD_BLOCK_SIZE;

                                                   A_cache[idx.get_local_id(group, 0)][internal * THREAD_BLOCK_SIZE + idx.get_local_id(group, 1)] = A_[(dim + idx.get_local_id(group, 0)) * (k_ + PADDING_SIZE) + global_i];
                                                   A_cache[idx.get_local_id(group, 0) + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + idx.get_local_id(group, 1)] = A_[(dim + idx.get_local_id(group, 0) + THREAD_BLOCK_SIZE) * (k_ + PADDING_SIZE) + global_i];

                                                   B_cache[idx.get_local_id(group, 0)][internal * THREAD_BLOCK_SIZE + idx.get_local_id(group, 1)] = B_[(dim + idx.get_local_id(group, 0)) * (n_ + PADDING_SIZE) + global_j];
                                                   B_cache[idx.get_local_id(group, 0) + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + idx.get_local_id(group, 1)] = B_[(dim + idx.get_local_id(group, 0) + THREAD_BLOCK_SIZE) * (n_ + PADDING_SIZE) + global_j];
                                               }
                                           });

                                           // perform calculations
                                           ::sycl::distribute_items_and_wait(group, [&](::sycl::s_item<2> idx) {
                                               for (unsigned block_dim = 0; block_dim < FEATURE_BLOCK_SIZE; ++block_dim) {
                                                   for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                                                       for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                                                           temp(idx)[internal_i][internal_j] += A_cache[block_dim][idx.get_local_id(group, 0) * INTERNAL_BLOCK_SIZE + internal_i] * B_cache[block_dim][idx.get_local_id(group, 1) * INTERNAL_BLOCK_SIZE + internal_j];
                                                       }
                                                   }
                                               }
                                           });
                                       }

                                       ::sycl::distribute_items_and_wait(group, [&](::sycl::s_item<2> idx) {
                                           for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                                               for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                                                   const unsigned long long global_i = i(idx) + internal_i;
                                                   const unsigned long long global_j = j(idx) + internal_j;

                                                   C_[global_i * (n_ + PADDING_SIZE) + global_j] = alpha_ * temp(idx)[internal_i][internal_j] + beta_ * C_[global_i * (n_ + PADDING_SIZE) + global_j];
                                               }
                                           }
                                       });
                                   });
    }

  private:
    /// @cond Doxygen_suppress
    [[maybe_unused]] const unsigned long long m_;
    const unsigned long long n_;
    const unsigned long long k_;
    const real_type alpha_;
    const real_type *A_;
    const real_type *B_;
    const real_type beta_;
    real_type *C_;
    /// @endcond
};

/**
 * @brief Perform an explicit BLAS SYMM operation: `C = alpha * A * B + beta * C` where A is a `m x k` symmetric matrix (memory optimized), B is a `k x n` matrix, C is a `m x n` matrix, and alpha and beta are scalars.
 */
class device_kernel_symm {
  public:
    /**
     * @brief Initialize the SYCL kernel function object.
     * @param[in] m the number of rows in @p A and @p C
     * @param[in] n the number of columns in @p B and @p C
     * @param[in] k the number of rows in @p A and number of columns in @p B
     * @param[in] alpha the scalar alpha value
     * @param[in] A the matrix @p A
     * @param[in] B the matrix @p B
     * @param[in] beta the scalar beta value
     * @param[in,out] C the matrix @p C, also used as result matrix
     */
    device_kernel_symm(const unsigned long long m, const unsigned long long n, const unsigned long long k, const real_type alpha, const real_type *A, const real_type *B, const real_type beta, real_type *C) :
        m_{ m },
        n_{ n },
        k_{ k },
        alpha_{ alpha },
        A_{ A },
        B_{ B },
        beta_{ beta },
        C_{ C } { }

    /**
     * @brief Function call operator overload performing the actual calculation.
     * @tparam T the implementation defined type of the group to iterate
     * @param[in] group group representing the current point in the execution space
     */
    template <typename T>
    void operator()(T group) const {
        // compute: C = alpha * A * B + beta * C with A in m x k, B in n x k, and C in n x m, alpha, beta as scalar
        ::sycl::memory_environment(group,
                                   ::sycl::require_local_mem<real_type[FEATURE_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE]>(),
                                   ::sycl::require_local_mem<real_type[FEATURE_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE]>(),
                                   ::sycl::require_private_mem<unsigned long long>(),
                                   ::sycl::require_private_mem<unsigned long long>(),
                                   ::sycl::require_private_mem<unsigned long long>(),
                                   ::sycl::require_private_mem<unsigned long long>(),
                                   ::sycl::require_private_mem<std::array<std::array<real_type, INTERNAL_BLOCK_SIZE>, INTERNAL_BLOCK_SIZE>>({}),
                                   [&](auto &A_cache, auto &B_cache, auto &i, auto &i_cached_idx_linear, auto &j, auto &j_linear, auto &temp) {
                                       // initialize private and local variables
                                       ::sycl::distribute_items_and_wait(group, [&](::sycl::s_item<2> idx) {
                                           // indices and diagonal condition
                                           i(idx) = (group[0] * group.get_logical_local_range(0) + idx.get_local_id(group, 0)) * INTERNAL_BLOCK_SIZE;
                                           i_cached_idx_linear(idx) = group[0] * group.get_logical_local_range(0) * INTERNAL_BLOCK_SIZE + idx.get_local_id(group, 1);
                                           j(idx) = (group[1] * group.get_logical_local_range(1) + idx.get_local_id(group, 1)) * INTERNAL_BLOCK_SIZE;
                                           j_linear(idx) = group[1] * group.get_logical_local_range(1) * INTERNAL_BLOCK_SIZE + idx.get_local_id(group, 1);
                                       });

                                       for (unsigned long long dim = 0; dim < k_; dim += FEATURE_BLOCK_SIZE) {
                                           // load data into shared memory
                                           ::sycl::distribute_items_and_wait(group, [&](::sycl::s_item<2> idx) {
                                               for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                                                   const unsigned long long global_i = i_cached_idx_linear(idx) + internal * THREAD_BLOCK_SIZE;
                                                   const unsigned long long global_j = j_linear(idx) + internal * THREAD_BLOCK_SIZE;

                                                   // determine on which side of the diagonal we are located
                                                   if (dim + idx.get_local_id(group, 0) < global_i) {
                                                       A_cache[idx.get_local_id(group, 0)][internal * THREAD_BLOCK_SIZE + idx.get_local_id(group, 1)] = A_[(dim + idx.get_local_id(group, 0)) * (k_ + PADDING_SIZE) + global_i - (dim + idx.get_local_id(group, 0)) * (dim + idx.get_local_id(group, 0) + 1) / 2];
                                                   } else {
                                                       A_cache[idx.get_local_id(group, 0)][internal * THREAD_BLOCK_SIZE + idx.get_local_id(group, 1)] = A_[global_i * (k_ + PADDING_SIZE) + dim + idx.get_local_id(group, 0) - global_i * (global_i + 1) / 2];
                                                   }
                                                   // determine on which side of the diagonal we are located
                                                   if (dim + idx.get_local_id(group, 0) + THREAD_BLOCK_SIZE < global_i) {
                                                       A_cache[idx.get_local_id(group, 0) + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + idx.get_local_id(group, 1)] = A_[(dim + idx.get_local_id(group, 0) + THREAD_BLOCK_SIZE) * (k_ + PADDING_SIZE) + global_i - (dim + idx.get_local_id(group, 0) + THREAD_BLOCK_SIZE) * (dim + idx.get_local_id(group, 0) + THREAD_BLOCK_SIZE + 1) / 2];
                                                   } else {
                                                       A_cache[idx.get_local_id(group, 0) + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + idx.get_local_id(group, 1)] = A_[global_i * (k_ + PADDING_SIZE) + dim + idx.get_local_id(group, 0) + THREAD_BLOCK_SIZE - global_i * (global_i + 1) / 2];
                                                   }

                                                   B_cache[idx.get_local_id(group, 0)][internal * THREAD_BLOCK_SIZE + idx.get_local_id(group, 1)] = B_[(dim + idx.get_local_id(group, 0)) * (n_ + PADDING_SIZE) + global_j];
                                                   B_cache[idx.get_local_id(group, 0) + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + idx.get_local_id(group, 1)] = B_[(dim + idx.get_local_id(group, 0) + THREAD_BLOCK_SIZE) * (n_ + PADDING_SIZE) + global_j];
                                               }
                                           });

                                           // perform calculations
                                           ::sycl::distribute_items_and_wait(group, [&](::sycl::s_item<2> idx) {
                                               for (unsigned block_dim = 0; block_dim < FEATURE_BLOCK_SIZE; ++block_dim) {
                                                   for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                                                       for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                                                           temp(idx)[internal_i][internal_j] += A_cache[block_dim][idx.get_local_id(group, 0) * INTERNAL_BLOCK_SIZE + internal_i] * B_cache[block_dim][idx.get_local_id(group, 1) * INTERNAL_BLOCK_SIZE + internal_j];
                                                       }
                                                   }
                                               }
                                           });
                                       }

                                       ::sycl::distribute_items_and_wait(group, [&](::sycl::s_item<2> idx) {
                                           for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                                               for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                                                   const unsigned long long global_i = i(idx) + internal_i;
                                                   const unsigned long long global_j = j(idx) + internal_j;

                                                   C_[global_i * (n_ + PADDING_SIZE) + global_j] = alpha_ * temp(idx)[internal_i][internal_j] + beta_ * C_[global_i * (n_ + PADDING_SIZE) + global_j];
                                               }
                                           }
                                       });
                                   });
    }

  private:
    /// @cond Doxygen_suppress
    [[maybe_unused]] const unsigned long long m_;
    const unsigned long long n_;
    const unsigned long long k_;
    const real_type alpha_;
    const real_type *A_;
    const real_type *B_;
    const real_type beta_;
    real_type *C_;
    /// @endcond
};

}  // namespace plssvm::sycl::detail::scoped

#endif  // PLSSVM_BACKENDS_SYCL_CG_EXPLICIT_BLAS_SCOPED_HPP_
