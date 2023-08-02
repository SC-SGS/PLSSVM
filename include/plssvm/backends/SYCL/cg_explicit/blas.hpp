/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Functions for explicitly performing a BLAS GEMM like matrix-matrix multiplication using the SYCL backend.
 */

#ifndef PLSSVM_BACKENDS_SYCL_CG_EXPLICIT_BLAS_HPP_
#define PLSSVM_BACKENDS_SYCL_CG_EXPLICIT_BLAS_HPP_
#pragma once

#include "plssvm/constants.hpp"  // plssvm::real_type, plssvm::THREAD_BLOCK_SIZE, plssvm::FEATURE_BLOCK_SIZE
#include "plssvm/backends/SYCL/detail/constants.hpp"  // PLSSVM_SYCL_BACKEND_COMPILER_HIPSYCL

#include "sycl/sycl.hpp"  // sycl::nd_item

namespace plssvm::sycl::detail {

/**
 * @brief Perform an explicit BLAS GEMM operation: `C = alpha * A * B + beta * C` where A is a `m x k` matrix, B is a `k x n` matrix, C is a `m x n` matrix, and alpha and beta are scalars.
 */
class device_kernel_gemm {
  public:
    /**
     * @brief Initialize the SYCL kernel function object.
     * @param[in] cgh the SYCL handler used to allocate the local memory
     * @param[in] m the number of rows in @p A and @p C
     * @param[in] n the number of columns in @p B and @p C
     * @param[in] k the number of rows in @p A and number of columns in @p B
     * @param[in] alpha the scalar alpha value
     * @param[in] A the matrix @p A
     * @param[in] B the matrix @p B
     * @param[in] beta the scalar beta value
     * @param[in,out] C the matrix @p C, also used as result matrix
     */
    device_kernel_gemm(::sycl::handler &cgh, const unsigned long long m, const unsigned long long n, const unsigned long long k, const real_type alpha, const real_type *A, const real_type *B, const real_type beta, real_type *C) :
        A_cache_{ ::sycl::range<2>{ FEATURE_BLOCK_SIZE, THREAD_BLOCK_SIZE }, cgh }, B_cache_{ ::sycl::range<2>{ FEATURE_BLOCK_SIZE, THREAD_BLOCK_SIZE }, cgh },
        m_{ m }, n_{ n }, k_{ k }, alpha_{ alpha }, A_{ A }, B_{ B }, beta_{ beta }, C_{ C } {}

    /**
     * @brief Function call operator overload performing the actual calculation.
     * @param[in] nd_idx indices representing the current point in the execution space
     */
    void operator()(::sycl::nd_item<2> nd_idx) const {
        // compute: C = alpha * A * B + beta * C with A in m x k, B in n x k, and C in n x m, alpha, beta as scalar
        const unsigned long long i = nd_idx.get_global_id(0);
        const unsigned long long i_cached_idx = nd_idx.get_group(0) * nd_idx.get_local_range(0) + nd_idx.get_local_id(1);
        const unsigned long long j = nd_idx.get_global_id(1);

        real_type temp{ 0.0 };

        for (unsigned long long dim = 0; dim < k_; dim += FEATURE_BLOCK_SIZE) {
            // zero out shared memory
            if (nd_idx.get_local_id(0) < FEATURE_BLOCK_SIZE) {
                A_cache_[nd_idx.get_local_id(0)][nd_idx.get_local_id(1)] = real_type{ 0.0 };
                B_cache_[nd_idx.get_local_id(0)][nd_idx.get_local_id(1)] = real_type{ 0.0 };
            }
#if PLSSVM_SYCL_BACKEND_COMPILER == PLSSVM_SYCL_BACKEND_COMPILER_HIPSYCL
            nd_idx.barrier();  // shouldn't be necessary
#endif

            // load data into shared memory
            if (nd_idx.get_local_id(0) < FEATURE_BLOCK_SIZE && dim + nd_idx.get_local_id(0) < k_) {
                if (i_cached_idx < k_) {
                    A_cache_[nd_idx.get_local_id(0)][nd_idx.get_local_id(1)] = A_[(dim + nd_idx.get_local_id(0)) * k_ + i_cached_idx - (dim + nd_idx.get_local_id(0)) * (dim + nd_idx.get_local_id(0) + 1) / 2];
                }
                if (j < n_) {
                    B_cache_[nd_idx.get_local_id(0)][nd_idx.get_local_id(1)] = B_[(dim + nd_idx.get_local_id(0)) * n_ + j];
                }
            }
            nd_idx.barrier();

            // calculation
            for (unsigned long long block_dim = 0; block_dim < FEATURE_BLOCK_SIZE; ++block_dim) {
                temp += A_cache_[block_dim][nd_idx.get_local_id(0)] * B_cache_[block_dim][nd_idx.get_local_id(1)];
            }
            nd_idx.barrier();
        }

        if (i < m_ && j < n_) {
            C_[i * n_ + j] = alpha_ * temp + beta_ * C_[i * n_ + j];
        }
    }

  private:
    /// Local memory used for internal memory access optimizations.
    ::sycl::local_accessor<real_type, 2> A_cache_;
    /// Local memory used for internal memory access optimizations.
    ::sycl::local_accessor<real_type, 2> B_cache_;

    /// @cond Doxygen_suppress
    const unsigned long long m_;
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
     * @param[in] cgh the SYCL handler used to allocate the local memory
     * @param[in] m the number of rows in @p A and @p C
     * @param[in] n the number of columns in @p B and @p C
     * @param[in] k the number of rows in @p A and number of columns in @p B
     * @param[in] alpha the scalar alpha value
     * @param[in] A the matrix @p A
     * @param[in] B the matrix @p B
     * @param[in] beta the scalar beta value
     * @param[in,out] C the matrix @p C, also used as result matrix
     */
    device_kernel_symm(::sycl::handler &cgh, const unsigned long long m, const unsigned long long n, const unsigned long long k, const real_type alpha, const real_type *A, const real_type *B, const real_type beta, real_type *C) :
        A_cache_{ ::sycl::range<2>{ FEATURE_BLOCK_SIZE, THREAD_BLOCK_SIZE }, cgh }, B_cache_{ ::sycl::range<2>{ FEATURE_BLOCK_SIZE, THREAD_BLOCK_SIZE }, cgh },
        m_{ m }, n_{ n }, k_{ k }, alpha_{ alpha }, A_{ A }, B_{ B }, beta_{ beta }, C_{ C } {}

    /**
     * @brief Function call operator overload performing the actual calculation.
     * @param[in] nd_idx indices representing the current point in the execution space
     */
    void operator()(::sycl::nd_item<2> nd_idx) const {
        // compute: C = alpha * A * B + beta * C with A in m x k, B in n x k, and C in n x m, alpha, beta as scalar
        const unsigned long long i = nd_idx.get_global_id(0);
        const unsigned long long i_cached_idx = nd_idx.get_group(0) * nd_idx.get_local_range(0) + nd_idx.get_local_id(1);
        const unsigned long long j = nd_idx.get_global_id(1);

        real_type temp{ 0.0 };

        for (unsigned long long dim = 0; dim < k_; dim += FEATURE_BLOCK_SIZE) {
            // zero out shared memory
            if (nd_idx.get_local_id(0) < FEATURE_BLOCK_SIZE) {
                A_cache_[nd_idx.get_local_id(0)][nd_idx.get_local_id(1)] = real_type{ 0.0 };
                B_cache_[nd_idx.get_local_id(0)][nd_idx.get_local_id(1)] = real_type{ 0.0 };
            }
#if PLSSVM_SYCL_BACKEND_COMPILER == PLSSVM_SYCL_BACKEND_COMPILER_HIPSYCL
            nd_idx.barrier();  // shouldn't be necessary
#endif

            // load data into shared memory
            if (nd_idx.get_local_id(0) < FEATURE_BLOCK_SIZE && dim + nd_idx.get_local_id(0) < k_) {
                if (dim + nd_idx.get_local_id(0) < i_cached_idx) {
                    if (i_cached_idx < k_) {
                        A_cache_[nd_idx.get_local_id(0)][nd_idx.get_local_id(1)] = A_[(dim + nd_idx.get_local_id(0)) * k_ + i_cached_idx - (dim + nd_idx.get_local_id(0)) * (dim + nd_idx.get_local_id(0) + 1) / 2];
                    }
                } else {
                    A_cache_[nd_idx.get_local_id(0)][nd_idx.get_local_id(1)] = A_[i_cached_idx * k_ + dim + nd_idx.get_local_id(0) - i_cached_idx * (i_cached_idx + 1) / 2];
                }
                if (j < n_) {
                    B_cache_[nd_idx.get_local_id(0)][nd_idx.get_local_id(1)] = B_[(dim + nd_idx.get_local_id(0)) * n_ + j];
                }
            }
            nd_idx.barrier();

            // calculation
            for (unsigned long long block_dim = 0; block_dim < FEATURE_BLOCK_SIZE; ++block_dim) {
                temp += A_cache_[block_dim][nd_idx.get_local_id(0)] * B_cache_[block_dim][nd_idx.get_local_id(1)];
            }
            nd_idx.barrier();
        }

        if (i < m_ && j < n_) {
            C_[i * n_ + j] = alpha_ * temp + beta_ * C_[i * n_ + j];
        }
    }

  private:
    /// Local memory used for internal memory access optimizations.
    ::sycl::local_accessor<real_type, 2> A_cache_;
    /// Local memory used for internal memory access optimizations.
    ::sycl::local_accessor<real_type, 2> B_cache_;

    /// @cond Doxygen_suppress
    const unsigned long long m_;
    const unsigned long long n_;
    const unsigned long long k_;
    const real_type alpha_;
    const real_type *A_;
    const real_type *B_;
    const real_type beta_;
    real_type *C_;
    /// @endcond
};

}  // namespace plssvm::sycl::detail

#endif  // PLSSVM_BACKENDS_SYCL_CG_EXPLICIT_BLAS_HPP_
