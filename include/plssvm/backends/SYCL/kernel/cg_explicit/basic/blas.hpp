/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Functions for explicitly performing a BLAS GEMM like matrix-matrix multiplication using the SYCL backend and the basic data parallel kernels.
 */

#ifndef PLSSVM_BACKENDS_SYCL_CG_EXPLICIT_BASIC_BLAS_HPP_
#define PLSSVM_BACKENDS_SYCL_CG_EXPLICIT_BASIC_BLAS_HPP_
#pragma once

#include "plssvm/constants.hpp"  // plssvm::{real_type, THREAD_BLOCK_SIZE, INTERNAL_BLOCK_SIZE, FEATURE_BLOCK_SIZE, PADDING_SIZE}

#include "sycl/sycl.hpp"  // sycl::item

#include <cstddef>  // std::size_t

namespace plssvm::sycl::detail::basic {

/**
 * @brief Perform an explicit BLAS SYMM operation: `C = alpha * A * B + beta * C` where @p A is a `m x k` symmetric matrix (memory optimized), @p B is a `k x n` matrix, @p C is a `m x n` matrix, and @p alpha and @p beta are scalars.
 * @details Uses SYCL's basic data parallel kernels.
 */
class device_kernel_symm {
  public:
    /**
     * @brief Initialize the SYCL kernel function object.
     * @param[in] num_rows the number of rows in @p A and @p C
     * @param[in] num_rhs the number of columns in @p B and @p C
     * @param[in] device_specific_num_rows the number of rows in @p A and number of rows in @p B; thr rows in @p A are potentially distributed across multiple devices
     * @param[in] row_offset the first row this device is responsible for
     * @param[in] alpha the scalar alpha value
     * @param[in] A the matrix @p A
     * @param[in] B the matrix @p B
     * @param[in] beta the scalar beta value
     * @param[in,out] C the matrix @p C, also used as result matrix
     * @param[in] grid_x_offset the offset in x-dimension into the data points if more than one execution grid has to be used
     * @param[in] grid_y_offset the offset in y-dimension into the data points if more than one execution grid has to be used
     */
    device_kernel_symm(const std::size_t num_rows, const std::size_t num_rhs, const std::size_t device_specific_num_rows, const std::size_t row_offset, const real_type alpha, const real_type *A, const real_type *B, const real_type beta, real_type *C, const std::size_t grid_x_offset, const std::size_t grid_y_offset) :
        num_rows_{ num_rows },
        num_rhs_{ num_rhs },
        device_specific_num_rows_{ device_specific_num_rows },
        row_offset_{ row_offset },
        alpha_{ alpha },
        A_{ A },
        B_{ B },
        beta_{ beta },
        C_{ C },
        grid_x_offset_{ grid_x_offset },
        grid_y_offset_{ grid_y_offset } { }

    /**
     * @brief Function call operator overload performing the actual calculation.
     * @param[in] idx indices representing the current point in the execution space
     */
    void operator()(::sycl::item<2> idx) const {
        const auto INTERNAL_BLOCK_SIZE_uz = static_cast<std::size_t>(INTERNAL_BLOCK_SIZE);
        const auto PADDING_SIZE_uz = static_cast<std::size_t>(PADDING_SIZE);

        // calculate the indices used in the current work-item
        const std::size_t i = (idx.get_id(1) + grid_x_offset_ * THREAD_BLOCK_SIZE) * INTERNAL_BLOCK_SIZE_uz;
        const std::size_t j = (idx.get_id(0) + grid_y_offset_ * THREAD_BLOCK_SIZE) * INTERNAL_BLOCK_SIZE_uz;

        // create a work-item private array used for internal caching
        real_type temp[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE]{};

        // iterate over all features using blocking to be able to cache them for faster memory accesses
        for (unsigned long long dim = 0; dim < (num_rows_ - row_offset_); ++dim) {
            // perform the dot product calculation
            for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                    const auto global_i = i + static_cast<std::size_t>(internal_i);
                    const auto global_j = j + static_cast<std::size_t>(internal_j);

                    real_type A_val = 0.0;
                    // determine on which side of the diagonal we are located
                    if (dim < global_j) {
                        A_val = A_[dim * (num_rows_ - row_offset_ + PADDING_SIZE_uz) + global_j - dim * (dim + std::size_t{ 1 }) / std::size_t{ 2 }];
                    } else {
                        A_val = A_[global_j * (num_rows_ - row_offset_ + PADDING_SIZE_uz) + dim - global_j * (global_j + std::size_t{ 1 }) / std::size_t{ 2 }];
                    }

                    temp[internal_i][internal_j] += A_val * B_[(dim + row_offset_) * (num_rhs_ + PADDING_SIZE_uz) + global_i];
                }
            }
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
    const std::size_t grid_x_offset_;
    const std::size_t grid_y_offset_;
    /// @endcond
};

/**
 * @brief Perform an explicit BLAS SYMM operation: `C = alpha * A * B + beta * C` where @p A is a `m x k` symmetric matrix (memory optimized), @p B is a `k x n` matrix, @p C is a `m x n` matrix, and @p alpha and @p beta are scalars.
 * @details In a multi-GPU setting, this function is responsible for mirroring down the columns this device is responsible for!
 *          Uses SYCL's basic data parallel kernels.
 */
class device_kernel_symm_mirror {
  public:
    /**
     * @brief Initialize the SYCL kernel function object.
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
     * @param[in] grid_x_offset the offset in x-dimension into the data points if more than one execution grid has to be used
     * @param[in] grid_y_offset the offset in y-dimension into the data points if more than one execution grid has to be used
     */
    device_kernel_symm_mirror(const std::size_t num_rows, const std::size_t num_rhs, const std::size_t num_mirror_rows, const std::size_t device_specific_num_rows, const std::size_t row_offset, const real_type alpha, const real_type *A, const real_type *B, const real_type beta, real_type *C, const std::size_t grid_x_offset, const std::size_t grid_y_offset) :
        num_rows_{ num_rows },
        num_rhs_{ num_rhs },
        num_mirror_rows_{ num_mirror_rows },
        device_specific_num_rows_{ device_specific_num_rows },
        row_offset_{ row_offset },
        alpha_{ alpha },
        A_{ A },
        B_{ B },
        beta_{ beta },
        C_{ C },
        grid_x_offset_{ grid_x_offset },
        grid_y_offset_{ grid_y_offset } { }

    /**
     * @brief Function call operator overload performing the actual calculation.
     * @param[in] idx indices representing the current point in the execution space
     */
    void operator()(::sycl::item<2> idx) const {
        const auto INTERNAL_BLOCK_SIZE_uz = static_cast<std::size_t>(INTERNAL_BLOCK_SIZE);
        const auto PADDING_SIZE_uz = static_cast<std::size_t>(PADDING_SIZE);

        // calculate the indices used in the current work-item
        const std::size_t i = (idx.get_id(1) + grid_x_offset_ * THREAD_BLOCK_SIZE) * INTERNAL_BLOCK_SIZE_uz;
        const std::size_t j = (idx.get_id(0) + grid_y_offset_ * THREAD_BLOCK_SIZE) * INTERNAL_BLOCK_SIZE_uz;

        // create a work-item private array used for internal caching
        real_type temp[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE]{};

        // iterate over the remaining features using blocking to be able to cache them for faster memory accesses
        for (std::size_t dim = 0; dim < device_specific_num_rows_; ++dim) {
            // perform the feature reduction calculation
            for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                    const auto global_i = i + static_cast<std::size_t>(internal_i);
                    const auto global_j = j + static_cast<std::size_t>(internal_j);

                    temp[internal_i][internal_j] += A_[(dim) * (num_rows_ - row_offset_ + PADDING_SIZE_uz) - (dim - std::size_t{ 1 }) * dim / std::size_t{ 2 } + device_specific_num_rows_ - dim + global_j] * B_[(dim + row_offset_) * (num_rhs_ + PADDING_SIZE_uz) + global_i];
                }
            }
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
    const std::size_t grid_x_offset_;
    const std::size_t grid_y_offset_;
    /// @endcond
};

/**
 * @brief Perform a simple inplace matrix addition: lhs += rhs.
 * @details Uses SYCL's basic data parallel kernels.
 */
class device_kernel_inplace_matrix_add {
  public:
    /**
     * @brief Initialize the SYCL kernel function object.
     * @param[in] num_cols the number of columns in both matrices
     * @param[in,out] lhs the first matrix (updated inplace)
     * @param[in] rhs the second matrix
     * @param[in] grid_x_offset the offset in x-dimension into the data points if more than one execution grid has to be used
     * @param[in] grid_y_offset the offset in y-dimension into the data points if more than one execution grid has to be used
     */
    device_kernel_inplace_matrix_add(const std::size_t num_cols, real_type *lhs, const real_type *rhs, const std::size_t grid_x_offset, const std::size_t grid_y_offset) :
        num_cols_{ num_cols },
        lhs_{ lhs },
        rhs_{ rhs },
        grid_x_offset_{ grid_x_offset },
        grid_y_offset_{ grid_y_offset } { }

    /**
     * @brief Function call operator overload performing the actual calculation.
     * @param[in] idx indices representing the current point in the execution space
     */
    void operator()(::sycl::item<2> idx) const {
        // cast all values to 64-bit std::size_t to prevent potential 32-bit overflows
        const auto INTERNAL_BLOCK_SIZE_uz = static_cast<std::size_t>(INTERNAL_BLOCK_SIZE);
        const auto PADDING_SIZE_uz = static_cast<std::size_t>(PADDING_SIZE);

        // calculate the indices used in the current work-item
        const std::size_t i = (idx.get_id(1) + grid_x_offset_ * THREAD_BLOCK_SIZE) * INTERNAL_BLOCK_SIZE_uz;
        const std::size_t j = (idx.get_id(0) + grid_y_offset_ * THREAD_BLOCK_SIZE) * INTERNAL_BLOCK_SIZE_uz;

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
    const std::size_t grid_x_offset_;
    const std::size_t grid_y_offset_;
    /// @endcond
};

/**
 * @brief Perform a simple inplace matrix scale: lhs *= scalar.
 * @details Uses SYCL's basic data parallel kernels.
 */
class device_kernel_inplace_matrix_scale {
  public:
    /**
     * @brief Initialize the SYCL kernel function object.
     * @param[in] num_cols the number of columns in the matrix
     * @param[in,out] lhs the first matrix (updated inplace)
     * @param[in] scale the value to scale
     * @param[in] grid_x_offset the offset in x-dimension into the data points if more than one execution grid has to be used
     * @param[in] grid_y_offset the offset in y-dimension into the data points if more than one execution grid has to be used
     */
    device_kernel_inplace_matrix_scale(const std::size_t num_cols, real_type *lhs, const real_type scale, const std::size_t grid_x_offset, const std::size_t grid_y_offset) :
        num_cols_{ num_cols },
        lhs_{ lhs },
        scale_{ scale },
        grid_x_offset_{ grid_x_offset },
        grid_y_offset_{ grid_y_offset } { }

    /**
     * @brief Function call operator overload performing the actual calculation.
     * @param[in] idx indices representing the current point in the execution space
     */
    void operator()(::sycl::item<2> idx) const {
        // cast all values to 64-bit std::size_t to prevent potential 32-bit overflows
        const auto INTERNAL_BLOCK_SIZE_uz = static_cast<std::size_t>(INTERNAL_BLOCK_SIZE);
        const auto PADDING_SIZE_uz = static_cast<std::size_t>(PADDING_SIZE);

        // calculate the indices used in the current work-item
        const std::size_t i = (idx.get_id(1) + grid_x_offset_ * THREAD_BLOCK_SIZE) * INTERNAL_BLOCK_SIZE_uz;
        const std::size_t j = (idx.get_id(0) + grid_y_offset_ * THREAD_BLOCK_SIZE) * INTERNAL_BLOCK_SIZE_uz;

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
    const std::size_t grid_x_offset_;
    const std::size_t grid_y_offset_;
    /// @endcond
};

}  // namespace plssvm::sycl::detail::basic

#endif  // PLSSVM_BACKENDS_SYCL_CG_EXPLICIT_BASIC_BLAS_HPP_
