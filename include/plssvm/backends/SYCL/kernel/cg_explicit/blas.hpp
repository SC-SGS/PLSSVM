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

#include <cstddef>  // std::size_t

namespace plssvm::sycl::detail {

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
    device_kernel_symm(::sycl::handler &cgh, const std::size_t num_rows, const std::size_t num_rhs, const std::size_t device_specific_num_rows, const std::size_t row_offset, const real_type alpha, const real_type *A, const real_type *B, const real_type beta, real_type *C) :
        A_cache_{ ::sycl::range<2>{ static_cast<std::size_t>(FEATURE_BLOCK_SIZE), static_cast<std::size_t>(INTERNAL_BLOCK_SIZE) * static_cast<std::size_t>(THREAD_BLOCK_SIZE) }, cgh },
        B_cache_{ ::sycl::range<2>{ static_cast<std::size_t>(FEATURE_BLOCK_SIZE), static_cast<std::size_t>(INTERNAL_BLOCK_SIZE) * static_cast<std::size_t>(THREAD_BLOCK_SIZE) }, cgh },
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
        // cast values to 32-bit unsigned int values to prevent implicit conversions
        const unsigned local_id_0 = nd_idx.get_local_id(0);
        const unsigned local_id_1 = nd_idx.get_local_id(1);

        // cast all values to 64-bit std::size_t to prevent potential 32-bit overflows
        const auto INTERNAL_BLOCK_SIZE_uz = static_cast<std::size_t>(INTERNAL_BLOCK_SIZE);
        const auto THREAD_BLOCK_SIZE_uz = static_cast<std::size_t>(THREAD_BLOCK_SIZE);
        const auto FEATURE_BLOCK_SIZE_uz = static_cast<std::size_t>(FEATURE_BLOCK_SIZE);
        const auto PADDING_SIZE_uz = static_cast<std::size_t>(PADDING_SIZE);

        // calculate the indices used in the current work-item
        const auto i = nd_idx.get_global_id(1) * INTERNAL_BLOCK_SIZE_uz;
        const auto i_linear = nd_idx.get_group(1) * nd_idx.get_local_range(1) * INTERNAL_BLOCK_SIZE_uz + nd_idx.get_local_id(1);
        const auto j = nd_idx.get_global_id(0) * INTERNAL_BLOCK_SIZE_uz;
        const auto j_linear = nd_idx.get_group(0) * nd_idx.get_local_range(0) * INTERNAL_BLOCK_SIZE_uz + nd_idx.get_local_id(1);

        // create a work-item private array used for internal caching
        real_type temp[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE]{};

        // iterate over all features using blocking to be able to cache them for faster memory accesses
        for (unsigned long long dim = 0; dim < (num_rows_ - row_offset_); dim += FEATURE_BLOCK_SIZE_uz) {
            // load data into local memory
            for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                const auto global_i = i_linear + static_cast<std::size_t>(internal) * THREAD_BLOCK_SIZE_uz;
                const auto global_j = j_linear + static_cast<std::size_t>(internal) * THREAD_BLOCK_SIZE_uz;

                // determine on which side of the diagonal we are located
                if (dim + nd_idx.get_local_id(0) < global_j) {
                    A_cache_[local_id_0][internal * THREAD_BLOCK_SIZE + local_id_1] = A_[(dim + nd_idx.get_local_id(0)) * (num_rows_ - row_offset_ + PADDING_SIZE_uz) + global_j - (dim + nd_idx.get_local_id(0)) * (dim + nd_idx.get_local_id(0) + std::size_t{ 1 }) / std::size_t{ 2 }];
                } else {
                    A_cache_[local_id_0][internal * THREAD_BLOCK_SIZE + local_id_1] = A_[global_j * (num_rows_ - row_offset_ + PADDING_SIZE_uz) + dim + nd_idx.get_local_id(0) - global_j * (global_j + std::size_t{ 1 }) / std::size_t{ 2 }];
                }
                // determine on which side of the diagonal we are located
                if (dim + nd_idx.get_local_id(0) + THREAD_BLOCK_SIZE < global_j) {
                    A_cache_[local_id_0 + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + local_id_1] = A_[(dim + nd_idx.get_local_id(0) + THREAD_BLOCK_SIZE_uz) * (num_rows_ - row_offset_ + PADDING_SIZE_uz) + global_j - (dim + nd_idx.get_local_id(0) + THREAD_BLOCK_SIZE_uz) * (dim + nd_idx.get_local_id(0) + THREAD_BLOCK_SIZE_uz + std::size_t{ 1 }) / std::size_t{ 2 }];
                } else {
                    A_cache_[local_id_0 + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + local_id_1] = A_[global_j * (num_rows_ - row_offset_ + PADDING_SIZE_uz) + dim + nd_idx.get_local_id(0) + THREAD_BLOCK_SIZE_uz - global_j * (global_j + std::size_t{ 1 }) / std::size_t{ 2 }];
                }

                B_cache_[local_id_0][internal * THREAD_BLOCK_SIZE + local_id_1] = B_[(dim + row_offset_ + nd_idx.get_local_id(0)) * (num_rhs_ + PADDING_SIZE_uz) + global_i];
                B_cache_[local_id_0 + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + local_id_1] = B_[(dim + row_offset_ + nd_idx.get_local_id(0) + THREAD_BLOCK_SIZE_uz) * (num_rhs_ + PADDING_SIZE_uz) + global_i];
            }
            nd_idx.barrier();  // wait until all work-items loaded their part of the data

            // perform the dot product calculation
            for (unsigned block_dim = 0; block_dim < FEATURE_BLOCK_SIZE; ++block_dim) {
                for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                    for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                        temp[internal_i][internal_j] += A_cache_[block_dim][local_id_0 * INTERNAL_BLOCK_SIZE + internal_j] * B_cache_[block_dim][local_id_1 * INTERNAL_BLOCK_SIZE + internal_i];
                    }
                }
            }
            nd_idx.barrier();  // wait until all work-items performed their part of the calculations
        }

        // apply the (partial) BLAS operation and update C
        for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
            for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                const auto global_i = i + static_cast<std::size_t>(internal_i);
                const auto device_global_j = j + static_cast<std::size_t>(internal_j);
                const auto global_j = row_offset_ + j + static_cast<std::size_t>(internal_j);

                // be sure to not perform out of bounds accesses
                if (global_i < num_rhs_ && device_global_j < device_specific_num_rows_) {
                    C_[global_j * (num_rhs_ + PADDING_SIZE_uz) + global_i] = alpha_ * temp[internal_i][internal_j] + beta_ * C_[global_j * (num_rhs_ + PADDING_SIZE_uz) + global_i];
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
    const std::size_t num_rows_;
    const std::size_t num_rhs_;
    const std::size_t device_specific_num_rows_;
    const std::size_t row_offset_;
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
     * @param[in] num_mirror_rows the number of rows to mirror down
     * @param[in] device_specific_num_rows the number of rows in @p A and number of rows in @p B; thr rows in @p A are potentially distributed across multiple devices
     * @param[in] row_offset the first row this device is responsible for
     * @param[in] alpha the scalar alpha value
     * @param[in] A the matrix @p A
     * @param[in] B the matrix @p B
     * @param[in] beta the scalar beta value
     * @param[in,out] C the matrix @p C, also used as result matrix
     */
    device_kernel_symm_mirror(::sycl::handler &cgh, const std::size_t num_rows, const std::size_t num_rhs, const std::size_t num_mirror_rows, const std::size_t device_specific_num_rows, const std::size_t row_offset, const real_type alpha, const real_type *A, const real_type *B, const real_type beta, real_type *C) :
        A_cache_{ ::sycl::range<2>{ static_cast<std::size_t>(FEATURE_BLOCK_SIZE), static_cast<std::size_t>(INTERNAL_BLOCK_SIZE) * static_cast<std::size_t>(THREAD_BLOCK_SIZE) }, cgh },
        B_cache_{ ::sycl::range<2>{ static_cast<std::size_t>(FEATURE_BLOCK_SIZE), static_cast<std::size_t>(INTERNAL_BLOCK_SIZE) * static_cast<std::size_t>(THREAD_BLOCK_SIZE) }, cgh },
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
        // cast values to 32-bit unsigned int values to prevent implicit conversions
        const unsigned local_id_0 = nd_idx.get_local_id(0);
        const unsigned local_id_1 = nd_idx.get_local_id(1);

        // cast all values to 64-bit std::size_t to prevent potential 32-bit overflows
        const auto INTERNAL_BLOCK_SIZE_uz = static_cast<std::size_t>(INTERNAL_BLOCK_SIZE);
        const auto THREAD_BLOCK_SIZE_uz = static_cast<std::size_t>(THREAD_BLOCK_SIZE);
        const auto FEATURE_BLOCK_SIZE_uz = static_cast<std::size_t>(FEATURE_BLOCK_SIZE);
        const auto PADDING_SIZE_uz = static_cast<std::size_t>(PADDING_SIZE);

        // calculate the indices used in the current work-item
        const auto i = nd_idx.get_global_id(1) * INTERNAL_BLOCK_SIZE_uz;
        const auto i_linear = nd_idx.get_group(1) * nd_idx.get_local_range(1) * INTERNAL_BLOCK_SIZE_uz + nd_idx.get_local_id(1);
        const auto j = nd_idx.get_global_id(0) * INTERNAL_BLOCK_SIZE_uz;
        const auto j_linear = nd_idx.get_group(0) * nd_idx.get_local_range(0) * INTERNAL_BLOCK_SIZE_uz + nd_idx.get_local_id(1);

        // create a work-item private array used for internal caching
        real_type temp[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE]{};

        // iterate over the remaining features using blocking to be able to cache them for faster memory accesses
        for (std::size_t dim = 0; dim < device_specific_num_rows_; dim += FEATURE_BLOCK_SIZE_uz) {
            // load data into shared memory
            for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                const unsigned long long global_i = i_linear + static_cast<std::size_t>(internal) * THREAD_BLOCK_SIZE_uz;
                const unsigned long long global_j = j_linear + static_cast<std::size_t>(internal) * THREAD_BLOCK_SIZE_uz;

                // FEATURE_BLOCK_SIZE = 2 * THREAD_BLOCK_SIZE -> store twice as many values in the local memory
                A_cache_[local_id_0][internal * THREAD_BLOCK_SIZE + local_id_1] = A_[(dim + nd_idx.get_local_id(0)) * (num_rows_ - row_offset_ + PADDING_SIZE_uz) - (dim + nd_idx.get_local_id(0) - std::size_t{ 1 }) * (dim + nd_idx.get_local_id(0)) / std::size_t{ 2 } + device_specific_num_rows_ - (dim + nd_idx.get_local_id(0)) + global_j];
                A_cache_[local_id_0 + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + local_id_1] = A_[(dim + nd_idx.get_local_id(0) + THREAD_BLOCK_SIZE_uz) * (num_rows_ - row_offset_ + PADDING_SIZE_uz) - (dim + nd_idx.get_local_id(0) + THREAD_BLOCK_SIZE_uz - std::size_t{ 1 }) * (dim + nd_idx.get_local_id(0) + THREAD_BLOCK_SIZE_uz) / std::size_t{ 2 } + device_specific_num_rows_ - (dim + nd_idx.get_local_id(0) + THREAD_BLOCK_SIZE_uz) + global_j];

                B_cache_[local_id_0][internal * THREAD_BLOCK_SIZE + local_id_1] = B_[(dim + row_offset_ + nd_idx.get_local_id(0)) * (num_rhs_ + PADDING_SIZE_uz) + global_i];
                B_cache_[local_id_0 + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + local_id_1] = B_[(dim + row_offset_ + nd_idx.get_local_id(0) + THREAD_BLOCK_SIZE_uz) * (num_rhs_ + PADDING_SIZE_uz) + global_i];
            }
            nd_idx.barrier();  // wait until all threads loaded their part of the data

            // perform the feature reduction calculation
            for (unsigned block_dim = 0; block_dim < FEATURE_BLOCK_SIZE; ++block_dim) {
                for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                    for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                        temp[internal_i][internal_j] += A_cache_[block_dim][local_id_0 * INTERNAL_BLOCK_SIZE + internal_j] * B_cache_[block_dim][local_id_1 * INTERNAL_BLOCK_SIZE + internal_i];
                    }
                }
            }
            nd_idx.barrier();  // wait until all threads performed their part of the calculations
        }

        // apply the (remaining) BLAS operation and update C
        for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
            for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                const auto global_i = i + static_cast<std::size_t>(internal_i);
                const auto partial_global_j = j + static_cast<std::size_t>(internal_j);
                const auto global_j = row_offset_ + device_specific_num_rows_ + j + static_cast<std::size_t>(internal_j);

                // be sure to not perform out of bounds accesses
                if (global_i < num_rhs_ && partial_global_j < num_mirror_rows_) {
                    C_[global_j * (num_rhs_ + PADDING_SIZE_uz) + global_i] = alpha_ * temp[internal_i][internal_j] + beta_ * C_[global_j * (num_rhs_ + PADDING_SIZE_uz) + global_i];
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
    const std::size_t num_rows_;
    const std::size_t num_rhs_;
    const std::size_t num_mirror_rows_;
    const std::size_t device_specific_num_rows_;
    const std::size_t row_offset_;
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
    device_kernel_inplace_matrix_add(const std::size_t num_cols, real_type *lhs, const real_type *rhs) :
        num_cols_{ num_cols },
        lhs_{ lhs },
        rhs_{ rhs } { }

    /**
     * @brief Function call operator overload performing the actual calculation.
     * @param[in] nd_idx indices representing the current point in the execution space
     */
    void operator()(::sycl::nd_item<2> nd_idx) const {
        // cast all values to 64-bit std::size_t to prevent potential 32-bit overflows
        const auto INTERNAL_BLOCK_SIZE_uz = static_cast<std::size_t>(INTERNAL_BLOCK_SIZE);
        const auto PADDING_SIZE_uz = static_cast<std::size_t>(PADDING_SIZE);

        // calculate the indices used in the current work-item
        const auto i = nd_idx.get_global_id(1) * INTERNAL_BLOCK_SIZE_uz;  // # num_rows
        const auto j = nd_idx.get_global_id(0) * INTERNAL_BLOCK_SIZE_uz;  // # num_rhs

        for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
            for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                const auto global_i = i + static_cast<std::size_t>(internal_i);
                const auto global_j = j + static_cast<std::size_t>(internal_j);

                lhs_[global_i * (num_cols_ + PADDING_SIZE_uz) + global_j] += rhs_[global_i * (num_cols_ + PADDING_SIZE_uz) + global_j];
            }
        }
    }

  private:
    /// @cond Doxygen_suppress
    const std::size_t num_cols_;
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
    device_kernel_inplace_matrix_scale(const std::size_t num_cols, real_type *lhs, const real_type scale) :
        num_cols_{ num_cols },
        lhs_{ lhs },
        scale_{ scale } { }

    /**
     * @brief Function call operator overload performing the actual calculation.
     * @param[in] nd_idx indices representing the current point in the execution space
     */
    void operator()(::sycl::nd_item<2> nd_idx) const {
        // cast all values to 64-bit std::size_t to prevent potential 32-bit overflows
        const auto INTERNAL_BLOCK_SIZE_uz = static_cast<std::size_t>(INTERNAL_BLOCK_SIZE);
        const auto PADDING_SIZE_uz = static_cast<std::size_t>(PADDING_SIZE);

        // calculate the indices used in the current work-item
        const auto i = nd_idx.get_global_id(1) * INTERNAL_BLOCK_SIZE_uz;  // # num_rows
        const auto j = nd_idx.get_global_id(0) * INTERNAL_BLOCK_SIZE_uz;  // # num_rhs

        for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
            for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                const auto global_i = i + static_cast<std::size_t>(internal_i);
                const auto global_j = j + static_cast<std::size_t>(internal_j);

                lhs_[global_i * (num_cols_ + PADDING_SIZE_uz) + global_j] *= scale_;
            }
        }
    }

  private:
    /// @cond Doxygen_suppress
    const std::size_t num_cols_;
    real_type *lhs_;
    const real_type scale_;
    /// @endcond
};

}  // namespace plssvm::sycl::detail

#endif  // PLSSVM_BACKENDS_SYCL_CG_EXPLICIT_BLAS_HPP_
