/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Functions for explicitly performing a BLAS GEMM like matrix-matrix multiplication using the SYCL backend and the work-group data parallel kernels.
 */

#ifndef PLSSVM_BACKENDS_SYCL_CG_EXPLICIT_WORK_GROUP_BLAS_HPP_
#define PLSSVM_BACKENDS_SYCL_CG_EXPLICIT_WORK_GROUP_BLAS_HPP_
#pragma once

#include "plssvm/backends/SYCL/data_parallel_kernels.hpp"  // plssvm::sycl::data_parallel_kernel
#include "plssvm/constants.hpp"                            // plssvm::real_type
#include "plssvm/target_platforms.hpp"                     // plssvm::target_platform

#include "sycl/sycl.hpp"  // sycl::handler, sycl::range, sycl::nd_item

#include <cstddef>  // std::size_t

namespace plssvm::sycl::detail::work_group {

/**
 * @brief Perform an explicit BLAS SYMM operation: `C = alpha * A * B + beta * C` where @p A is a `m x k` symmetric matrix (memory optimized), @p B is a `k x n` matrix, @p C is a `m x n` matrix, and @p alpha and @p beta are scalars.
 * @details Uses SYCL's work-group data parallel kernels.
 */
class device_kernel_symm {
  public:
    /// The used SYCL data parallel kernel.
    constexpr static sycl::data_parallel_kernel data_parallel_kernel_type = sycl::data_parallel_kernel::work_group;

    /**
     * @brief Initialize the SYCL kernel function object.
     * @param[in] cgh the SYCL handler used to allocate the local memory
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
    device_kernel_symm(::sycl::handler &cgh, const std::size_t num_rows, const std::size_t num_rhs, const std::size_t device_num_rows, const std::size_t device_row_offset, const real_type alpha, const real_type *A, const real_type *B, const real_type beta, real_type *C, const std::size_t grid_x_offset, const std::size_t grid_y_offset) :
        A_cache_{ ::sycl::range<2>{ static_cast<std::size_t>(THREAD_BLOCK_SIZE), static_cast<std::size_t>(THREAD_BLOCK_SIZE) }, cgh },
        B_cache_{ ::sycl::range<2>{ static_cast<std::size_t>(THREAD_BLOCK_SIZE), static_cast<std::size_t>(THREAD_BLOCK_SIZE) }, cgh },
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
     * @param[in] nd_idx indices representing the current point in the execution space
     */
    void operator()(::sycl::nd_item<2> nd_idx) const {
        // cast values to 32-bit unsigned int values to prevent implicit conversions
        const auto local_id_0 = static_cast<unsigned>(nd_idx.get_local_id(0));
        const auto local_id_1 = static_cast<unsigned>(nd_idx.get_local_id(1));

        // cast all values to 64-bit std::size_t to prevent potential 32-bit overflows
        constexpr auto THREAD_BLOCK_SIZE_uz = static_cast<std::size_t>(THREAD_BLOCK_SIZE);

        const auto threadIdx_x = static_cast<std::size_t>(nd_idx.get_local_id(0));               // current work-item in work-group x-dimension
        const auto threadIdx_y = static_cast<std::size_t>(nd_idx.get_local_id(1));               // current work-item in work-group y-dimension
        const auto blockDim_x = static_cast<std::size_t>(nd_idx.get_local_range(0));             // number of work-items in work-group x-dimension
        const auto blockDim_y = static_cast<std::size_t>(nd_idx.get_local_range(1));             // number of work-items in work-group y-dimension
        const auto blockIdx_x = static_cast<std::size_t>(nd_idx.get_group(0)) + grid_x_offset_;  // current work-group in global range x-dimension + offsets if the global range is too large
        const auto blockIdx_y = static_cast<std::size_t>(nd_idx.get_group(1)) + grid_y_offset_;  // current work-group in global range y-dimension + offsets if the global range is too large

        real_type temp{ 0.0 };

        {
            // calculate the indices used in the current work-item, pays attention to coalesced memory accesses
            const auto global_i_idx_linear = blockIdx_y * blockDim_y + threadIdx_y;  // num_rhs
            const auto global_j_idx_linear = blockIdx_x * blockDim_x + threadIdx_y;  // device_num_rows

            // iterate over all values using blocking to be able to cache them for faster memory accesses
            for (std::size_t dim_block = 0; dim_block < (num_rows_ - device_row_offset_); dim_block += THREAD_BLOCK_SIZE_uz) {
                // zero-out local memory
                A_cache_[local_id_0][local_id_1] = real_type{ 0.0 };
                B_cache_[local_id_0][local_id_1] = real_type{ 0.0 };

                // load data into local memory
                if (dim_block + threadIdx_x < num_rows_ - device_row_offset_) {
                    if (global_j_idx_linear < device_num_rows_) {
                        // determine on which side of the diagonal we are located
                        if (dim_block + threadIdx_x < global_j_idx_linear) {
                            A_cache_[local_id_0][local_id_1] = A_[(dim_block + threadIdx_x) * (num_rows_ - device_row_offset_) + global_j_idx_linear - (dim_block + threadIdx_x) * (dim_block + threadIdx_x + std::size_t{ 1 }) / std::size_t{ 2 }];  // SoA, upper triangular matrix only
                        } else {
                            A_cache_[local_id_0][local_id_1] = A_[global_j_idx_linear * (num_rows_ - device_row_offset_) + dim_block + threadIdx_x - global_j_idx_linear * (global_j_idx_linear + std::size_t{ 1 }) / std::size_t{ 2 }];  // SoA, upper triangular matrix only
                        }
                    }
                    if (global_i_idx_linear < num_rhs_) {
                        B_cache_[local_id_0][local_id_1] = B_[(dim_block + device_row_offset_ + threadIdx_x) * num_rhs_ + global_i_idx_linear];  // SoA
                    }
                }
                nd_idx.barrier();  // wait until all work-items loaded their part of the data

                // perform the dot product calculation
                for (unsigned dim = 0; dim < THREAD_BLOCK_SIZE; ++dim) {
                    temp += A_cache_[dim][local_id_0] * B_cache_[dim][local_id_1];
                }
                nd_idx.barrier();  // wait until all work-items performed their part of the calculations
            }
        }

        // calculate the indices used in the current thread
        const auto global_i_idx = blockIdx_y * blockDim_y + threadIdx_y;         // num_rhs
        const auto device_global_j_idx = blockIdx_x * blockDim_x + threadIdx_x;  // device_num_rows
        const auto global_j_idx = device_row_offset_ + device_global_j_idx;

        // be sure to not perform out-of-bounds accesses
        if (global_i_idx < num_rhs_ && device_global_j_idx < device_num_rows_ && global_j_idx < num_rows_) {
            C_[global_j_idx * num_rhs_ + global_i_idx] = alpha_ * temp + beta_ * C_[global_j_idx * num_rhs_ + global_i_idx];  // SoA
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
 *          Uses SYCL's work-group data parallel kernels.
 */
class device_kernel_symm_mirror {
  public:
    /// The used SYCL data parallel kernel.
    constexpr static sycl::data_parallel_kernel data_parallel_kernel_type = sycl::data_parallel_kernel::work_group;

    /**
     * @brief Initialize the SYCL kernel function object.
     * @param[in] cgh the SYCL handler used to allocate the local memory
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
    device_kernel_symm_mirror(::sycl::handler &cgh, const std::size_t num_rows, const std::size_t num_rhs, const std::size_t num_mirror_rows, const std::size_t device_num_rows, const std::size_t device_row_offset, const real_type alpha, const real_type *A, const real_type *B, const real_type beta, real_type *C, const std::size_t grid_x_offset, const std::size_t grid_y_offset) :
        A_cache_{ ::sycl::range<2>{ static_cast<std::size_t>(THREAD_BLOCK_SIZE), static_cast<std::size_t>(THREAD_BLOCK_SIZE) }, cgh },
        B_cache_{ ::sycl::range<2>{ static_cast<std::size_t>(THREAD_BLOCK_SIZE), static_cast<std::size_t>(THREAD_BLOCK_SIZE) }, cgh },
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
     * @param[in] nd_idx indices representing the current point in the execution space
     */
    void operator()(::sycl::nd_item<2> nd_idx) const {
        // cast values to 32-bit unsigned int values to prevent implicit conversions
        const auto local_id_0 = static_cast<unsigned>(nd_idx.get_local_id(0));
        const auto local_id_1 = static_cast<unsigned>(nd_idx.get_local_id(1));

        // cast all values to 64-bit std::size_t to prevent potential 32-bit overflows
        constexpr auto THREAD_BLOCK_SIZE_uz = static_cast<std::size_t>(THREAD_BLOCK_SIZE);

        const auto threadIdx_x = static_cast<std::size_t>(nd_idx.get_local_id(0));               // current work-item in work-group x-dimension
        const auto threadIdx_y = static_cast<std::size_t>(nd_idx.get_local_id(1));               // current work-item in work-group y-dimension
        const auto blockDim_x = static_cast<std::size_t>(nd_idx.get_local_range(0));             // number of work-items in work-group x-dimension
        const auto blockDim_y = static_cast<std::size_t>(nd_idx.get_local_range(1));             // number of work-items in work-group y-dimension
        const auto blockIdx_x = static_cast<std::size_t>(nd_idx.get_group(0)) + grid_x_offset_;  // current work-group in global range x-dimension + offsets if the global range is too large
        const auto blockIdx_y = static_cast<std::size_t>(nd_idx.get_group(1)) + grid_y_offset_;  // current work-group in global range y-dimension + offsets if the global range is too large

        real_type temp{ 0.0 };

        {
            // calculate the indices used in the current thread, pays attention to coalesced memory accesses
            const auto global_i_idx_linear = blockIdx_y * blockDim_y + threadIdx_y;  // num_rhs
            const auto global_j_idx_linear = blockIdx_x * blockDim_x + threadIdx_y;  // num_mirror_rows

            // iterate over the remaining values using blocking to be able to cache them for faster memory accesses
            for (std::size_t dim_block = 0; dim_block < device_num_rows_; dim_block += THREAD_BLOCK_SIZE_uz) {
                // zero-out local memory
                A_cache_[local_id_0][local_id_1] = real_type{ 0.0 };
                B_cache_[local_id_0][local_id_1] = real_type{ 0.0 };

                // load data into local memory
                if (dim_block + threadIdx_x < device_num_rows_) {
                    if (global_j_idx_linear < num_mirror_rows_) {
                        A_cache_[local_id_0][local_id_1] = A_[(dim_block + threadIdx_x) * (num_rows_ - device_row_offset_) - (dim_block + threadIdx_x - std::size_t{ 1 }) * (dim_block + threadIdx_x) / std::size_t{ 2 } + device_num_rows_ - (dim_block + threadIdx_x) + global_j_idx_linear];  // SoA, upper triangular matrix only
                    }
                    if (global_i_idx_linear < num_rhs_) {
                        B_cache_[local_id_0][local_id_1] = B_[(device_row_offset_ + dim_block + threadIdx_x) * num_rhs_ + global_i_idx_linear];  // SoA
                    }
                }
                nd_idx.barrier();  // wait until all work-items loaded their part of the data

                // perform the dot product calculation
                for (unsigned dim = 0; dim < THREAD_BLOCK_SIZE; ++dim) {
                    temp += A_cache_[dim][local_id_0] * B_cache_[dim][local_id_1];
                }
                nd_idx.barrier();  // wait until all work-items performed their part of the calculations
            }
        }

        // calculate the indices used in the current work-item
        const auto global_i_idx = blockIdx_y * blockDim_y + threadIdx_y;          // num_rhs
        const auto partial_global_j_idx = blockIdx_x * blockDim_x + threadIdx_x;  // num_mirror_rows
        const auto global_j_idx = device_row_offset_ + device_num_rows_ + partial_global_j_idx;

        // be sure to not perform out-of-bounds accesses
        if (global_i_idx < num_rhs_ && partial_global_j_idx < num_mirror_rows_ && global_j_idx < num_rows_) {
            C_[global_j_idx * num_rhs_ + global_i_idx] = alpha_ * temp + beta_ * C_[global_j_idx * num_rhs_ + global_i_idx];  // SoA
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
 * @details Uses SYCL's work-group data parallel kernels.
 */
class device_kernel_inplace_matrix_add {
  public:
    /// The used SYCL data parallel kernel.
    constexpr static sycl::data_parallel_kernel data_parallel_kernel_type = sycl::data_parallel_kernel::work_group;

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
     * @param[in] nd_idx indices representing the current point in the execution space
     */
    void operator()(::sycl::nd_item<2> nd_idx) const {
        const auto threadIdx_x = static_cast<std::size_t>(nd_idx.get_local_id(0));               // current work-item in work-group x-dimension
        const auto threadIdx_y = static_cast<std::size_t>(nd_idx.get_local_id(1));               // current work-item in work-group y-dimension
        const auto blockDim_x = static_cast<std::size_t>(nd_idx.get_local_range(0));             // number of work-items in work-group x-dimension
        const auto blockDim_y = static_cast<std::size_t>(nd_idx.get_local_range(1));             // number of work-items in work-group y-dimension
        const auto blockIdx_x = static_cast<std::size_t>(nd_idx.get_group(0)) + grid_x_offset_;  // current work-group in global range x-dimension + offsets if the global range is too large
        const auto blockIdx_y = static_cast<std::size_t>(nd_idx.get_group(1)) + grid_y_offset_;  // current work-group in global range y-dimension + offsets if the global range is too large

        // calculate the indices used in the current work-item
        const auto global_i_idx = blockIdx_y * blockDim_y + threadIdx_y;  // num_rows
        const auto global_j_idx = blockIdx_x * blockDim_x + threadIdx_x;  // num_rhs

        if (global_i_idx < num_rows_ && global_j_idx < num_cols_) {
            lhs_[global_i_idx * num_cols_ + global_j_idx] += rhs_[global_i_idx * num_cols_ + global_j_idx];  // SoA
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
 * @details Uses SYCL's work-group data parallel kernels.
 */
class device_kernel_inplace_matrix_scale {
  public:
    /// The used SYCL data parallel kernel.
    constexpr static sycl::data_parallel_kernel data_parallel_kernel_type = sycl::data_parallel_kernel::work_group;

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
     * @param[in] nd_idx indices representing the current point in the execution space
     */
    void operator()(::sycl::nd_item<2> nd_idx) const {
        const auto threadIdx_x = static_cast<std::size_t>(nd_idx.get_local_id(0));               // current work-item in work-group x-dimension
        const auto threadIdx_y = static_cast<std::size_t>(nd_idx.get_local_id(1));               // current work-item in work-group y-dimension
        const auto blockDim_x = static_cast<std::size_t>(nd_idx.get_local_range(0));             // number of work-items in work-group x-dimension
        const auto blockDim_y = static_cast<std::size_t>(nd_idx.get_local_range(1));             // number of work-items in work-group y-dimension
        const auto blockIdx_x = static_cast<std::size_t>(nd_idx.get_group(0)) + grid_x_offset_;  // current work-group in global range x-dimension + offsets if the global range is too large
        const auto blockIdx_y = static_cast<std::size_t>(nd_idx.get_group(1)) + grid_y_offset_;  // current work-group in global range y-dimension + offsets if the global range is too large

        // calculate the indices used in the current work-item
        const auto global_i_idx = blockIdx_y * blockDim_y + threadIdx_y;  // num_rows
        const auto global_j_idx = blockIdx_x * blockDim_x + threadIdx_x;  // num_rhs

        if (global_i_idx < num_rows_ && global_j_idx < num_cols_) {
            lhs_[global_i_idx * num_cols_ + global_j_idx] *= scale_;  // SoA
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

}  // namespace plssvm::sycl::detail::work_group

#endif  // PLSSVM_BACKENDS_SYCL_CG_EXPLICIT_WORK_GROUP_BLAS_HPP_
