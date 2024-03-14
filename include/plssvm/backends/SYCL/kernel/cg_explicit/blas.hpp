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

#include "plssvm/constants.hpp"  // plssvm::{real_type, THREAD_BLOCK_SIZE, INTERNAL_BLOCK_SIZE, FEATURE_BLOCK_SIZE, PADDING_SIZE}

#include "sycl/sycl.hpp"  // sycl::nd_item

namespace plssvm::sycl {

/**
 * @brief Perform an explicit BLAS SYMM operation: `C = alpha * A * B + beta * C` where @p A is a `m x k` symmetric matrix (memory optimized), @p B is a `k x n` matrix, @p C is a `m x n` matrix, and @p alpha and @p beta are scalars.
 */
class device_kernel_symm {
  public:
    /**
     * @brief Initialize the SYCL kernel function object.
     * @param[in] cgh the SYCL handler used to allocate the local memory
     * @param[in] num_rows the number of rows in @p A and @p C
     * @param[in] num_rhs the number of columns in @p B and @p C
     * @param[in] device_specific_num_rows the number of rows in @p A and number of rows in @p B; thr rows in @p A are potentially distributed across multiple devices
     * @param[in] row_offset the first row this device is responsible for
     * @param[in] alpha the scalar alpha value
     * @param[in] A the matrix @p A
     * @param[in] B the matrix @p B
     * @param[in] beta the scalar beta value
     * @param[in,out] C the matrix @p C, also used as result matrix
     */
    device_kernel_symm(::sycl::handler &cgh, const unsigned long long num_rows, const unsigned long long num_rhs, const unsigned long long device_specific_num_rows, const unsigned long long row_offset, const real_type alpha, const real_type *A, const real_type *B, const real_type beta, real_type *C) :
        A_cache_{ ::sycl::range<2>{ FEATURE_BLOCK_SIZE, INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE }, cgh },
        B_cache_{ ::sycl::range<2>{ FEATURE_BLOCK_SIZE, INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE }, cgh },
        num_rows_{ num_rows },
        num_rhs_{ num_rhs },
        device_specific_num_rows_{ device_specific_num_rows },
        row_offset_{ row_offset },
        alpha_{ alpha },
        A_{ A },
        B_{ B },
        beta_{ beta },
        C_{ C } { }

    /**
     * @brief Function call operator overload performing the actual calculation.
     * @param[in] nd_idx indices representing the current point in the execution space
     */
    void operator()(::sycl::nd_item<2> nd_idx) const {
        const unsigned long long i = nd_idx.get_global_id(1) * INTERNAL_BLOCK_SIZE;
        const unsigned long long i_linear = nd_idx.get_group(1) * nd_idx.get_local_range(1) * INTERNAL_BLOCK_SIZE + nd_idx.get_local_id(1);
        const unsigned long long j = nd_idx.get_global_id(0) * INTERNAL_BLOCK_SIZE;
        const unsigned long long j_cached_idx_linear = nd_idx.get_group(0) * nd_idx.get_local_range(0) * INTERNAL_BLOCK_SIZE + nd_idx.get_local_id(1);

        real_type temp[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE] = { { 0.0 } };

        for (unsigned long long dim = 0; dim < (num_rows_ - row_offset_); dim += FEATURE_BLOCK_SIZE) {
            // load data into shared memory
            for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                const unsigned long long global_i = i_linear + internal * THREAD_BLOCK_SIZE;
                const unsigned long long global_j = j_cached_idx_linear + internal * THREAD_BLOCK_SIZE;

                // determine on which side of the diagonal we are located
                if (dim + nd_idx.get_local_id(0) < global_j) {
                    A_cache_[nd_idx.get_local_id(0)][internal * THREAD_BLOCK_SIZE + nd_idx.get_local_id(1)] = A_[(dim + nd_idx.get_local_id(0)) * (num_rows_ - row_offset_ + PADDING_SIZE) + global_j - (dim + nd_idx.get_local_id(0)) * (dim + nd_idx.get_local_id(0) + 1) / 2];
                } else {
                    A_cache_[nd_idx.get_local_id(0)][internal * THREAD_BLOCK_SIZE + nd_idx.get_local_id(1)] = A_[global_j * (num_rows_ - row_offset_ + PADDING_SIZE) + dim + nd_idx.get_local_id(0) - global_j * (global_j + 1) / 2];
                }
                // determine on which side of the diagonal we are located
                if (dim + nd_idx.get_local_id(0) + THREAD_BLOCK_SIZE < global_j) {
                    A_cache_[nd_idx.get_local_id(0) + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + nd_idx.get_local_id(1)] = A_[(dim + nd_idx.get_local_id(0) + THREAD_BLOCK_SIZE) * (num_rows_ - row_offset_ + PADDING_SIZE) + global_j - (dim + nd_idx.get_local_id(0) + THREAD_BLOCK_SIZE) * (dim + nd_idx.get_local_id(0) + THREAD_BLOCK_SIZE + 1) / 2];
                } else {
                    A_cache_[nd_idx.get_local_id(0) + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + nd_idx.get_local_id(1)] = A_[global_j * (num_rows_ - row_offset_ + PADDING_SIZE) + dim + nd_idx.get_local_id(0) + THREAD_BLOCK_SIZE - global_j * (global_j + 1) / 2];
                }

                B_cache_[nd_idx.get_local_id(0)][internal * THREAD_BLOCK_SIZE + nd_idx.get_local_id(1)] = B_[(dim + row_offset_ + nd_idx.get_local_id(0)) * (num_rhs_ + PADDING_SIZE) + global_i];
                B_cache_[nd_idx.get_local_id(0) + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + nd_idx.get_local_id(1)] = B_[(dim + row_offset_ + nd_idx.get_local_id(0) + THREAD_BLOCK_SIZE) * (num_rhs_ + PADDING_SIZE) + global_i];
            }
            nd_idx.barrier();

            // calculation
            for (unsigned block_dim = 0; block_dim < FEATURE_BLOCK_SIZE; ++block_dim) {
                for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                    for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                        temp[internal_i][internal_j] += A_cache_[block_dim][nd_idx.get_local_id(0) * INTERNAL_BLOCK_SIZE + internal_j] * B_cache_[block_dim][nd_idx.get_local_id(1) * INTERNAL_BLOCK_SIZE + internal_i];
                    }
                }
            }
            nd_idx.barrier();
        }

        for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
            for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                const unsigned long long global_i = i + internal_i;
                const unsigned long long device_global_j = j + internal_j;
                const unsigned long long global_j = row_offset_ + j + internal_j;

                if (global_i < num_rhs_ && device_global_j < device_specific_num_rows_) {
                    C_[global_j * (num_rhs_ + PADDING_SIZE) + global_i] = alpha_ * temp[internal_i][internal_j] + beta_ * C_[global_j * (num_rhs_ + PADDING_SIZE) + global_i];
                }
            }
        }
    }

  private:
    /// Local memory used for internal memory access optimizations.
    ::sycl::local_accessor<real_type, 2> A_cache_;
    /// Local memory used for internal memory access optimizations.
    ::sycl::local_accessor<real_type, 2> B_cache_;

    /// @cond Doxygen_suppress
    const unsigned long long num_rows_;
    const unsigned long long num_rhs_;
    const unsigned long long device_specific_num_rows_;
    const unsigned long long row_offset_;
    const real_type alpha_;
    const real_type *A_;
    const real_type *B_;
    const real_type beta_;
    real_type *C_;
    /// @endcond
};

/**
 * @brief Perform an explicit BLAS SYMM operation: `C = alpha * A * B + beta * C` where @p A is a `m x k` symmetric matrix (memory optimized), @p B is a `k x n` matrix, @p C is a `m x n` matrix, and @p alpha and @p beta are scalars.
 * @details In a multi-GPU setting, this function is responsible for mirroring down the columns this device is responsible for!
 */
class device_kernel_symm_mirror {
  public:
    /**
     * @brief Initialize the SYCL kernel function object.
     * @param[in] cgh the SYCL handler used to allocate the local memory
     * @param[in] num_rows the number of rows in @p A and @p C
     * @param[in] num_rhs the number of columns in @p B and @p C
     * @param[in] device_specific_num_rows the number of rows in @p A and number of rows in @p B; thr rows in @p A are potentially distributed across multiple devices
     * @param[in] row_offset the first row this device is responsible for
     * @param[in] alpha the scalar alpha value
     * @param[in] A the matrix @p A
     * @param[in] B the matrix @p B
     * @param[in] beta the scalar beta value
     * @param[in,out] C the matrix @p C, also used as result matrix
     */
    device_kernel_symm_mirror(::sycl::handler &cgh, const unsigned long long num_rows, const unsigned long long num_rhs, const unsigned long long num_mirror_rows, const unsigned long long device_specific_num_rows, const unsigned long long row_offset, const real_type alpha, const real_type *A, const real_type *B, const real_type beta, real_type *C) :
        A_cache_{ ::sycl::range<2>{ FEATURE_BLOCK_SIZE, INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE }, cgh },
        B_cache_{ ::sycl::range<2>{ FEATURE_BLOCK_SIZE, INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE }, cgh },
        num_rows_{ num_rows },
        num_rhs_{ num_rhs },
        num_mirror_rows_{ num_mirror_rows },
        device_specific_num_rows_{ device_specific_num_rows },
        row_offset_{ row_offset },
        alpha_{ alpha },
        A_{ A },
        B_{ B },
        beta_{ beta },
        C_{ C } { }

    /**
     * @brief Function call operator overload performing the actual calculation.
     * @param[in] nd_idx indices representing the current point in the execution space
     */
    void operator()(::sycl::nd_item<2> nd_idx) const {
        const unsigned long long i = nd_idx.get_global_id(1) * INTERNAL_BLOCK_SIZE;
        const unsigned long long i_linear = nd_idx.get_group(1) * nd_idx.get_local_range(1) * INTERNAL_BLOCK_SIZE + nd_idx.get_local_id(1);
        const unsigned long long j = nd_idx.get_global_id(0) * INTERNAL_BLOCK_SIZE;
        const unsigned long long j_cached_idx_linear = nd_idx.get_group(0) * nd_idx.get_local_range(0) * INTERNAL_BLOCK_SIZE + nd_idx.get_local_id(1);

        real_type temp[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE] = { { 0.0 } };

        for (unsigned long long dim = 0; dim < device_specific_num_rows_; dim += FEATURE_BLOCK_SIZE) {
            // load data into shared memory
            for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                const unsigned long long global_i = i_linear + internal * THREAD_BLOCK_SIZE;
                const unsigned long long global_j = j_cached_idx_linear + internal * THREAD_BLOCK_SIZE;

                // determine on which side of the diagonal we are located
                A_cache_[nd_idx.get_local_id(0)][internal * THREAD_BLOCK_SIZE + nd_idx.get_local_id(1)] = A_[(dim + nd_idx.get_local_id(0)) * (num_rows_ - row_offset_ + PADDING_SIZE) - (dim + nd_idx.get_local_id(0) - 1) * (dim + nd_idx.get_local_id(0)) / 2 + device_specific_num_rows_ - (dim + nd_idx.get_local_id(0)) + global_j];
                A_cache_[nd_idx.get_local_id(0) + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + nd_idx.get_local_id(1)] = A_[(dim + nd_idx.get_local_id(0) + THREAD_BLOCK_SIZE) * (num_rows_ - row_offset_ + PADDING_SIZE) - (dim + nd_idx.get_local_id(0) + THREAD_BLOCK_SIZE - 1) * (dim + nd_idx.get_local_id(0) + THREAD_BLOCK_SIZE) / 2 + device_specific_num_rows_ - (dim + nd_idx.get_local_id(0) + THREAD_BLOCK_SIZE) + global_j];

                B_cache_[nd_idx.get_local_id(0)][internal * THREAD_BLOCK_SIZE + nd_idx.get_local_id(1)] = B_[(dim + row_offset_ + nd_idx.get_local_id(0)) * (num_rhs_ + PADDING_SIZE) + global_i];
                B_cache_[nd_idx.get_local_id(0) + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + nd_idx.get_local_id(1)] = B_[(dim + row_offset_ + nd_idx.get_local_id(0) + THREAD_BLOCK_SIZE) * (num_rhs_ + PADDING_SIZE) + global_i];
            }
            nd_idx.barrier();

            // calculation
            for (unsigned block_dim = 0; block_dim < FEATURE_BLOCK_SIZE; ++block_dim) {
                for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                    for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                        temp[internal_i][internal_j] += A_cache_[block_dim][nd_idx.get_local_id(0) * INTERNAL_BLOCK_SIZE + internal_j] * B_cache_[block_dim][nd_idx.get_local_id(1) * INTERNAL_BLOCK_SIZE + internal_i];
                    }
                }
            }
            nd_idx.barrier();
        }

        for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
            for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                const unsigned long long global_i = i + internal_i;
                const unsigned long long partial_global_j = j + internal_j;
                const unsigned long long global_j = row_offset_ + device_specific_num_rows_ + j + internal_j;

                if (global_i < num_rhs_ && partial_global_j < num_mirror_rows_) {
                    C_[global_j * (num_rhs_ + PADDING_SIZE) + global_i] = alpha_ * temp[internal_i][internal_j] + beta_ * C_[global_j * (num_rhs_ + PADDING_SIZE) + global_i];
                }
            }
        }
    }

  private:
    /// Local memory used for internal memory access optimizations.
    ::sycl::local_accessor<real_type, 2> A_cache_;
    /// Local memory used for internal memory access optimizations.
    ::sycl::local_accessor<real_type, 2> B_cache_;

    /// @cond Doxygen_suppress
    const unsigned long long num_rows_;
    const unsigned long long num_rhs_;
    const unsigned long long num_mirror_rows_;
    const unsigned long long device_specific_num_rows_;
    const unsigned long long row_offset_;
    const real_type alpha_;
    const real_type *A_;
    const real_type *B_;
    const real_type beta_;
    real_type *C_;
    /// @endcond
};

/**
 * @brief Perform a simple inplace matrix addition: lhs += rhs.
 */
class device_kernel_inplace_matrix_add {
  public:
    /**
     * @brief Initialize the SYCL kernel function object.
     * @param[in] num_cols the number of columns in both matrices
     * @param[in,out] lhs the first matrix (updated inplace)
     * @param[in] rhs the second matrix
     */
    device_kernel_inplace_matrix_add(const unsigned long long num_cols, real_type *lhs, const real_type *rhs) :
        num_cols_{ num_cols },
        lhs_{ lhs },
        rhs_{ rhs } { }

    /**
     * @brief Function call operator overload performing the actual calculation.
     * @param[in] nd_idx indices representing the current point in the execution space
     */
    void operator()(::sycl::nd_item<2> nd_idx) const {
        const unsigned long long i = nd_idx.get_global_id(1) * INTERNAL_BLOCK_SIZE;  // # num_rows
        const unsigned long long j = nd_idx.get_global_id(0) * INTERNAL_BLOCK_SIZE;  // # num_rhs

        for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
            for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                const unsigned long long global_i = i + internal_i;
                const unsigned long long global_j = j + internal_j;

                lhs_[global_i * (num_cols_ + PADDING_SIZE) + global_j] += rhs_[global_i * (num_cols_ + PADDING_SIZE) + global_j];
            }
        }
    }

  private:
    /// @cond Doxygen_suppress
    const unsigned long long num_cols_;
    real_type *lhs_;
    const real_type *rhs_;
    /// @endcond
};

/**
 * @brief Perform a simple inplace matrix scale: lhs *= scalar.
 */
class device_kernel_inplace_matrix_scale {
  public:
    /**
     * @brief Initialize the SYCL kernel function object.
     * @param[in] num_cols the number of columns in the matrix
     * @param[in,out] lhs the first matrix (updated inplace)
     * @param[in] scale the value to scale
     */
    device_kernel_inplace_matrix_scale(const unsigned long long num_cols, real_type *lhs, const real_type scale) :
        num_cols_{ num_cols },
        lhs_{ lhs },
        scale_{ scale } { }

    /**
     * @brief Function call operator overload performing the actual calculation.
     * @param[in] nd_idx indices representing the current point in the execution space
     */
    void operator()(::sycl::nd_item<2> nd_idx) const {
        const unsigned long long i = nd_idx.get_global_id(1) * INTERNAL_BLOCK_SIZE;  // # num_rows
        const unsigned long long j = nd_idx.get_global_id(0) * INTERNAL_BLOCK_SIZE;  // # num_rhs

        for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
            for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                const unsigned long long global_i = i + internal_i;
                const unsigned long long global_j = j + internal_j;

                lhs_[global_i * (num_cols_ + PADDING_SIZE) + global_j] *= scale_;
            }
        }
    }

  private:
    /// @cond Doxygen_suppress
    const unsigned long long num_cols_;
    real_type *lhs_;
    const real_type scale_;
    /// @endcond
};

}  // namespace plssvm::sycl::detail

#endif  // PLSSVM_BACKENDS_SYCL_CG_EXPLICIT_BLAS_HPP_
