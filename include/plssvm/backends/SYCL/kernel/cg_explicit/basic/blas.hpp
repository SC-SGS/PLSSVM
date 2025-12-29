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

#ifndef PLSSVM_BACKENDS_SYCL_KERNEL_CG_EXPLICIT_BASIC_BLAS_HPP_
#define PLSSVM_BACKENDS_SYCL_KERNEL_CG_EXPLICIT_BASIC_BLAS_HPP_
#pragma once

#include "plssvm/backends/SYCL/data_parallel_kernels.hpp"  // plssvm::sycl::data_parallel_kernel
#include "plssvm/constants.hpp"                            // plssvm::{real_type, THREAD_BLOCK_SIZE}

#include "sycl/sycl.hpp"  // sycl::item

#include <array>    // std::array
#include <cstddef>  // std::size_t

namespace plssvm::sycl::detail::basic {

/**
 * @brief Perform an explicit BLAS SYMM operation: `C = alpha * A * B + beta * C` where @p A is a `m x k` symmetric matrix (memory optimized), @p B is a `k x n` matrix, @p C is a `m x n` matrix, and @p alpha and @p beta are scalars.
 * @details Uses SYCL's basic data parallel kernels.
 */
class device_kernel_symm {
  public:
    /// The used SYCL data parallel kernel.
    constexpr static sycl::data_parallel_kernel data_parallel_kernel_type = sycl::data_parallel_kernel::basic;

    /**
     * @brief Initialize the SYCL kernel function object.
     * @param[in] num_rows the number of rows in @p A and @p C
     * @param[in] num_rhs the number of columns in @p B and @p C
     * @param[in] device_num_rows the number of rows in @p A and number of rows in @p B; thr rows in @p A are potentially distributed across multiple devices
     * @param[in] device_row_offset the first row this device is responsible for
     * @param[in] alpha the scalar alpha value
     * @param[in] A the matrix @p A
     * @param[in] B the matrix @p B
     * @param[in] beta the scalar beta value
     * @param[in,out] C the matrix @p C, also used as result matrix
     * @param[in] grid_x_offset the offset in x-dimension into the data points if more than one execution grid has to be used
     * @param[in] grid_y_offset the offset in y-dimension into the data points if more than one execution grid has to be used
     */
    device_kernel_symm(const std::size_t num_rows, const std::size_t num_rhs, const std::size_t device_num_rows, const std::size_t device_row_offset, const real_type alpha, const real_type *A, const real_type *B, const real_type beta, real_type *C, const std::size_t grid_x_offset, const std::size_t grid_y_offset) :
        num_rows_{ num_rows },
        num_rhs_{ num_rhs },
        device_num_rows_{ device_num_rows },
        device_row_offset_{ device_row_offset },
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
        // cast all values to 64-bit std::size_t to prevent potential 32-bit overflows
        constexpr auto INTERNAL_BLOCK_SIZE_uz = static_cast<std::size_t>(INTERNAL_BLOCK_SIZE);
        constexpr auto THREAD_BLOCK_SIZE_uz = static_cast<std::size_t>(THREAD_BLOCK_SIZE);

        // calculate the indices used in the current work-item
        const auto i_idx = (idx.get_id(1) + grid_x_offset_ * THREAD_BLOCK_SIZE_uz) * INTERNAL_BLOCK_SIZE_uz;  // num_rhs
        const auto j_idx = (idx.get_id(0) + grid_y_offset_ * THREAD_BLOCK_SIZE_uz) * INTERNAL_BLOCK_SIZE_uz;  // device_num_rows

        // create a work-item private array used for internal caching
        std::array<std::array<real_type, INTERNAL_BLOCK_SIZE_uz>, INTERNAL_BLOCK_SIZE_uz> temp{};

        // perform the dot product calculation
        for (std::size_t dim = 0; dim < (num_rows_ - device_row_offset_); ++dim) {
            for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                    // calculate the indices to access the global data
                    const auto global_i_idx = i_idx + static_cast<std::size_t>(internal_i);
                    const auto global_j_idx = j_idx + static_cast<std::size_t>(internal_j);

                    real_type A_cache = 0.0;
                    if (global_j_idx < device_num_rows_) {
                        // determine on which side of the diagonal we are located
                        if (dim < global_j_idx) {
                            A_cache = A_[dim * (num_rows_ - device_row_offset_) + global_j_idx - dim * (dim + std::size_t{ 1 }) / std::size_t{ 2 }];  // SoA, upper triangular matrix only
                        } else {
                            A_cache = A_[global_j_idx * (num_rows_ - device_row_offset_) + dim - global_j_idx * (global_j_idx + std::size_t{ 1 }) / std::size_t{ 2 }];  // SoA, upper triangular matrix only
                        }
                    }
                    real_type B_cache = 0.0;
                    if (global_i_idx < num_rhs_) {
                        B_cache = B_[(dim + device_row_offset_) * num_rhs_ + global_i_idx];
                    }

                    temp[internal_i][internal_j] += A_cache * B_cache;  // SoA
                }
            }
        }

        // apply the (partial) BLAS operation and update C
        for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
            for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                // calculate the indices to access the global data and the data with respect to the current device
                const auto global_i_idx = i_idx + static_cast<std::size_t>(internal_i);
                const auto device_global_j_idx = j_idx + static_cast<std::size_t>(internal_j);
                const auto global_j_idx = device_row_offset_ + device_global_j_idx;

                // be sure to not perform out-of-bounds accesses
                if (global_i_idx < num_rhs_ && device_global_j_idx < device_num_rows_) {
                    C_[global_j_idx * num_rhs_ + global_i_idx] = alpha_ * temp[internal_i][internal_j] + beta_ * C_[global_j_idx * num_rhs_ + global_i_idx];  // SoA
                }
            }
        }
    }

  private:
    /// @cond Doxygen_suppress
    const std::size_t num_rows_;
    const std::size_t num_rhs_;
    const std::size_t device_num_rows_;
    const std::size_t device_row_offset_;
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
    /// The used SYCL data parallel kernel.
    constexpr static sycl::data_parallel_kernel data_parallel_kernel_type = sycl::data_parallel_kernel::basic;

    /**
     * @brief Initialize the SYCL kernel function object.
     * @param[in] num_rows the number of rows in @p A and @p C
     * @param[in] num_rhs the number of columns in @p B and @p C
     * @param[in] num_mirror_rows the number of rows to mirror down
     * @param[in] device_num_rows the number of rows in @p A and number of rows in @p B; thr rows in @p A are potentially distributed across multiple devices
     * @param[in] device_row_offset the first row this device is responsible for
     * @param[in] alpha the scalar alpha value
     * @param[in] A the matrix @p A
     * @param[in] B the matrix @p B
     * @param[in] beta the scalar beta value
     * @param[in,out] C the matrix @p C, also used as result matrix
     * @param[in] grid_x_offset the offset in x-dimension into the data points if more than one execution grid has to be used
     * @param[in] grid_y_offset the offset in y-dimension into the data points if more than one execution grid has to be used
     */
    device_kernel_symm_mirror(const std::size_t num_rows, const std::size_t num_rhs, const std::size_t num_mirror_rows, const std::size_t device_num_rows, const std::size_t device_row_offset, const real_type alpha, const real_type *A, const real_type *B, const real_type beta, real_type *C, const std::size_t grid_x_offset, const std::size_t grid_y_offset) :
        num_rows_{ num_rows },
        num_rhs_{ num_rhs },
        num_mirror_rows_{ num_mirror_rows },
        device_num_rows_{ device_num_rows },
        device_row_offset_{ device_row_offset },
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
        // cast all values to 64-bit std::size_t to prevent potential 32-bit overflows
        constexpr auto INTERNAL_BLOCK_SIZE_uz = static_cast<std::size_t>(INTERNAL_BLOCK_SIZE);
        constexpr auto THREAD_BLOCK_SIZE_uz = static_cast<std::size_t>(THREAD_BLOCK_SIZE);

        // calculate the indices used in the current work-item
        const auto i_idx = (idx.get_id(1) + grid_x_offset_ * THREAD_BLOCK_SIZE_uz) * INTERNAL_BLOCK_SIZE_uz;  // num_rhs
        const auto j_idx = (idx.get_id(0) + grid_y_offset_ * THREAD_BLOCK_SIZE_uz) * INTERNAL_BLOCK_SIZE_uz;  // num_mirror_rows

        // create a work-item private array used for internal caching
        std::array<std::array<real_type, INTERNAL_BLOCK_SIZE_uz>, INTERNAL_BLOCK_SIZE_uz> temp{};

        // perform the dot product calculation
        for (std::size_t dim = 0; dim < device_num_rows_; ++dim) {
            for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                    // calculate the indices to access the global data
                    const auto global_i_idx = i_idx + static_cast<std::size_t>(internal_i);
                    const auto global_j_idx = j_idx + static_cast<std::size_t>(internal_j);

                    real_type A_cache = 0.0;
                    if (global_j_idx < num_rows_) {
                        A_cache = A_[dim * (num_rows_ - device_row_offset_) - (dim - std::size_t{ 1 }) * dim / std::size_t{ 2 } + device_num_rows_ - dim + global_j_idx];  // SoA, upper triangular matrix only
                    }
                    real_type B_cache = 0.0;
                    if (global_i_idx < num_rhs_) {
                        B_cache = B_[(dim + device_row_offset_) * num_rhs_ + global_i_idx];  // SoA
                    }

                    temp[internal_i][internal_j] += A_cache * B_cache;
                }
            }
        }

        // apply the (remaining) BLAS operation and update C
        for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
            for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                // calculate the indices to access the global data and the data with respect to the current device
                const auto global_i_idx = i_idx + static_cast<std::size_t>(internal_i);
                const auto partial_global_j_idx = j_idx + static_cast<std::size_t>(internal_j);
                const auto global_j_idx = device_row_offset_ + device_num_rows_ + partial_global_j_idx;

                // be sure to not perform out-of-bounds accesses
                if (global_i_idx < num_rhs_ && partial_global_j_idx < num_mirror_rows_ && global_j_idx < num_rows_) {
                    C_[global_j_idx * num_rhs_ + global_i_idx] = alpha_ * temp[internal_i][internal_j] + beta_ * C_[global_j_idx * num_rhs_ + global_i_idx];  // SoA
                }
            }
        }
    }

  private:
    /// @cond Doxygen_suppress
    const std::size_t num_rows_;
    const std::size_t num_rhs_;
    const std::size_t num_mirror_rows_;
    const std::size_t device_num_rows_;
    const std::size_t device_row_offset_;
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
    /// The used SYCL data parallel kernel.
    constexpr static sycl::data_parallel_kernel data_parallel_kernel_type = sycl::data_parallel_kernel::basic;

    /**
     * @brief Initialize the SYCL kernel function object.
     * @param[in] num_rows the number of rows in both matrices
     * @param[in] num_cols the number of columns in both matrices
     * @param[in,out] lhs the first matrix (updated inplace)
     * @param[in] rhs the second matrix
     * @param[in] grid_x_offset the offset in x-dimension into the data points if more than one execution grid has to be used
     * @param[in] grid_y_offset the offset in y-dimension into the data points if more than one execution grid has to be used
     */
    device_kernel_inplace_matrix_add(const std::size_t num_rows, const std::size_t num_cols, real_type *lhs, const real_type *rhs, const std::size_t grid_x_offset, const std::size_t grid_y_offset) :
        num_rows_{ num_rows },
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
        constexpr auto INTERNAL_BLOCK_SIZE_uz = static_cast<std::size_t>(INTERNAL_BLOCK_SIZE);
        constexpr auto THREAD_BLOCK_SIZE_uz = static_cast<std::size_t>(THREAD_BLOCK_SIZE);

        // calculate the indices used in the current work-item
        const auto i_idx = (idx.get_id(1) + grid_x_offset_ * THREAD_BLOCK_SIZE_uz) * INTERNAL_BLOCK_SIZE_uz;  // num_rows
        const auto j_idx = (idx.get_id(0) + grid_y_offset_ * THREAD_BLOCK_SIZE_uz) * INTERNAL_BLOCK_SIZE_uz;  // num_rhs

        for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
            for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                // calculate the indices to access the global data
                const auto global_i_idx = i_idx + static_cast<std::size_t>(internal_i);
                const auto global_j_idx = j_idx + static_cast<std::size_t>(internal_j);

                if (global_i_idx < num_rows_ && global_j_idx < num_cols_) {
                    lhs_[global_i_idx * num_cols_ + global_j_idx] += rhs_[global_i_idx * num_cols_ + global_j_idx];  // SoA
                }
            }
        }
    }

  private:
    /// @cond Doxygen_suppress
    const std::size_t num_rows_;
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
    /// The used SYCL data parallel kernel.
    constexpr static sycl::data_parallel_kernel data_parallel_kernel_type = sycl::data_parallel_kernel::basic;

    /**
     * @brief Initialize the SYCL kernel function object.
     * @param[in] num_rows the number of rows in the matrix
     * @param[in] num_cols the number of columns in the matrix
     * @param[in,out] lhs the first matrix (updated inplace)
     * @param[in] scale the value to scale
     * @param[in] grid_x_offset the offset in x-dimension into the data points if more than one execution grid has to be used
     * @param[in] grid_y_offset the offset in y-dimension into the data points if more than one execution grid has to be used
     */
    device_kernel_inplace_matrix_scale(const std::size_t num_rows, const std::size_t num_cols, real_type *lhs, const real_type scale, const std::size_t grid_x_offset, const std::size_t grid_y_offset) :
        num_rows_{ num_rows },
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
        constexpr auto INTERNAL_BLOCK_SIZE_uz = static_cast<std::size_t>(INTERNAL_BLOCK_SIZE);
        constexpr auto THREAD_BLOCK_SIZE_uz = static_cast<std::size_t>(THREAD_BLOCK_SIZE);

        // calculate the indices used in the current work-item
        const auto i_idx = (idx.get_id(1) + grid_x_offset_ * THREAD_BLOCK_SIZE_uz) * INTERNAL_BLOCK_SIZE_uz;  // num_rows
        const auto j_idx = (idx.get_id(0) + grid_y_offset_ * THREAD_BLOCK_SIZE_uz) * INTERNAL_BLOCK_SIZE_uz;  // num_rhs

        for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
            for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                // calculate the indices to access the global data
                const auto global_i_idx = i_idx + static_cast<std::size_t>(internal_i);
                const auto global_j_idx = j_idx + static_cast<std::size_t>(internal_j);

                if (global_i_idx < num_rows_ && global_j_idx < num_cols_) {
                    lhs_[global_i_idx * num_cols_ + global_j_idx] *= scale_;  // SoA
                }
            }
        }
    }

  private:
    /// @cond Doxygen_suppress
    const std::size_t num_rows_;
    const std::size_t num_cols_;
    real_type *lhs_;
    const real_type scale_;
    const std::size_t grid_x_offset_;
    const std::size_t grid_y_offset_;
    /// @endcond
};

}  // namespace plssvm::sycl::detail::basic

#endif  // PLSSVM_BACKENDS_SYCL_KERNEL_CG_EXPLICIT_BASIC_BLAS_HPP_
