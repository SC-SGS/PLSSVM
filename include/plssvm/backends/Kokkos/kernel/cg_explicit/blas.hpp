/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Functions for explicitly performing a BLAS GEMM like matrix-matrix multiplication using the Kokkos backend.
 */

#ifndef PLSSVM_BACKENDS_KOKKOS_CG_EXPLICIT_BLAS_HPP_
#define PLSSVM_BACKENDS_KOKKOS_CG_EXPLICIT_BLAS_HPP_
#pragma once

#include "plssvm/constants.hpp"  // plssvm::{real_type, THREAD_BLOCK_SIZE}

#include "Kokkos_Core.hpp"  // KOKKOS_INLINE_FUNCTION, Kokkos::View, Kokkos::TeamPolicy

#include <cstddef>  // std::size_t

namespace plssvm::kokkos::detail {

/**
 * @brief Perform an explicit BLAS SYMM operation: `C = alpha * A * B + beta * C` where @p A is a `m x k` symmetric matrix (memory optimized), @p B is a `k x n` matrix, @p C is a `m x n` matrix, and @p alpha and @p beta are scalars.
 * @tparam ExecutionSpace the Kokkos::ExecutionSpace used to execute the kernel
 */
template <typename ExecutionSpace>
class device_kernel_symm {
    /**
     * @brief The type of the used Kokkos::View.
     */
    template <typename T>
    using device_view_type = Kokkos::View<T *, ExecutionSpace>;

  public:
    /**
     * @brief Initialize the Kokkos kernel function object.
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
     * @param[in] grid_size_x the size of the execution grid in x-dimension
     */
    device_kernel_symm(const std::size_t num_rows, const std::size_t num_rhs, const std::size_t device_num_rows, const std::size_t device_row_offset, const real_type alpha, device_view_type<const real_type> A, device_view_type<const real_type> B, const real_type beta, device_view_type<real_type> C, const std::size_t grid_x_offset, const std::size_t grid_y_offset, const std::size_t grid_size_x) :
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
        grid_y_offset_{ grid_y_offset },
        grid_size_x_{ grid_size_x } { }

    /**
     * @brief Function call operator overload performing the actual calculation.
     * @param[in] team the Kokkos team representing the current point in the execution space
     */
    KOKKOS_INLINE_FUNCTION
    void operator()(const typename Kokkos::TeamPolicy<ExecutionSpace>::member_type &team) const {
        // cast all values to 64-bit std::size_t to prevent potential 32-bit overflows
        constexpr auto THREAD_BLOCK_SIZE_uz = static_cast<std::size_t>(THREAD_BLOCK_SIZE);

        const auto threadIdx_x = static_cast<std::size_t>(team.team_rank()) / THREAD_BLOCK_SIZE_uz;            // current thread in team x-dimension
        const auto threadIdx_y = static_cast<std::size_t>(team.team_rank()) % THREAD_BLOCK_SIZE_uz;            // current thread in team y-dimension
        const auto blockDim_x = THREAD_BLOCK_SIZE_uz;                                                          // number of threads in team x-dimension
        const auto blockDim_y = THREAD_BLOCK_SIZE_uz;                                                          // number of threads in team y-dimension
        const auto blockIdx_x = static_cast<std::size_t>(team.league_rank()) % grid_size_x_ + grid_x_offset_;  // current team in league x-dimension + offsets if the league size is too large
        const auto blockIdx_y = static_cast<std::size_t>(team.league_rank()) / grid_size_x_ + grid_y_offset_;  // current team in league y-dimension + offsets if the league size is too large

        // calculate the indices used in the current thread
        const auto global_i_idx = blockIdx_x * blockDim_x + threadIdx_x;         // num_rhs
        const auto device_global_j_idx = blockIdx_y * blockDim_y + threadIdx_y;  // device_num_rows
        const auto global_j_idx = device_row_offset_ + device_global_j_idx;

        // be sure to not perform out-of-bounds accesses
        if (global_i_idx < num_rhs_ && device_global_j_idx < device_num_rows_) {
            real_type temp{ 0.0 };

            // iterate over all values
            for (std::size_t dim = 0; dim < (num_rows_ - device_row_offset_); ++dim) {
                real_type A_cache{ 0.0 };
                // determine on which side of the diagonal we are located
                if (dim < device_global_j_idx) {
                    A_cache = A_[dim * (num_rows_ - device_row_offset_) + device_global_j_idx - dim * (dim + std::size_t{ 1 }) / std::size_t{ 2 }];  // SoA, upper triangular matrix only
                } else {
                    A_cache = A_[device_global_j_idx * (num_rows_ - device_row_offset_) + dim - device_global_j_idx * (device_global_j_idx + std::size_t{ 1 }) / std::size_t{ 2 }];  // SoA, upper triangular matrix only
                }
                // perform the dot product calculation
                temp += A_cache * B_[(device_row_offset_ + dim) * num_rhs_ + global_i_idx];  // SoA
            }

            // apply the (partial) BLAS operation and update C
            C_[global_j_idx * num_rhs_ + global_i_idx] = alpha_ * temp + beta_ * C_[global_j_idx * num_rhs_ + global_i_idx];  // SoA
        }
    }

  private:
    /// @cond Doxygen_suppress
    const std::size_t num_rows_;
    const std::size_t num_rhs_;
    const std::size_t device_num_rows_;
    const std::size_t device_row_offset_;
    const real_type alpha_;
    device_view_type<const real_type> A_;
    device_view_type<const real_type> B_;
    const real_type beta_;
    device_view_type<real_type> C_;
    const std::size_t grid_x_offset_;
    const std::size_t grid_y_offset_;
    const std::size_t grid_size_x_;
    /// @endcond
};

/**
 * @brief Perform an explicit BLAS SYMM operation: `C = alpha * A * B + beta * C` where @p A is a `m x k` symmetric matrix (memory optimized), @p B is a `k x n` matrix, @p C is a `m x n` matrix, and @p alpha and @p beta are scalars.
 * @details In a multi-GPU setting, this function is responsible for mirroring down the columns this device is responsible for!
 * @tparam ExecutionSpace the Kokkos::ExecutionSpace used to execute the kernel
 */
template <typename ExecutionSpace>
class device_kernel_symm_mirror {
    /**
     * @brief The type of the used Kokkos::View.
     */
    template <typename T>
    using device_view_type = Kokkos::View<T *, ExecutionSpace>;

  public:
    /**
     * @brief Initialize the Kokkos kernel function object.
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
     * @param[in] grid_size_x the size of the execution grid in x-dimension
     */
    device_kernel_symm_mirror(const std::size_t num_rows, const std::size_t num_rhs, const std::size_t num_mirror_rows, const std::size_t device_num_rows, const std::size_t device_row_offset, const real_type alpha, device_view_type<const real_type> A, device_view_type<const real_type> B, const real_type beta, device_view_type<real_type> C, const std::size_t grid_x_offset, const std::size_t grid_y_offset, const std::size_t grid_size_x) :
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
        grid_y_offset_{ grid_y_offset },
        grid_size_x_{ grid_size_x } { }

    /**
     * @brief Function call operator overload performing the actual calculation.
     * @param[in] team the Kokkos team representing the current point in the execution space
     */
    KOKKOS_INLINE_FUNCTION
    void operator()(const typename Kokkos::TeamPolicy<ExecutionSpace>::member_type &team) const {
        // cast all values to 64-bit std::size_t to prevent potential 32-bit overflows
        constexpr auto THREAD_BLOCK_SIZE_uz = static_cast<std::size_t>(THREAD_BLOCK_SIZE);

        const auto threadIdx_x = static_cast<std::size_t>(team.team_rank()) / THREAD_BLOCK_SIZE_uz;            // current thread in team x-dimension
        const auto threadIdx_y = static_cast<std::size_t>(team.team_rank()) % THREAD_BLOCK_SIZE_uz;            // current thread in team y-dimension
        const auto blockDim_x = THREAD_BLOCK_SIZE_uz;                                                          // number of threads in team x-dimension
        const auto blockDim_y = THREAD_BLOCK_SIZE_uz;                                                          // number of threads in team y-dimension
        const auto blockIdx_x = static_cast<std::size_t>(team.league_rank()) % grid_size_x_ + grid_x_offset_;  // current team in league x-dimension + offsets if the league size is too large
        const auto blockIdx_y = static_cast<std::size_t>(team.league_rank()) / grid_size_x_ + grid_y_offset_;  // current team in league y-dimension + offsets if the league size is too large

        // calculate the indices used in the current thread
        const auto global_i_idx = blockIdx_x * blockDim_x + threadIdx_x;          // num_rhs
        const auto partial_global_j_idx = blockIdx_y * blockDim_y + threadIdx_y;  // num_mirror_rows
        const auto global_j_idx = device_row_offset_ + device_num_rows_ + partial_global_j_idx;

        // be sure to not perform out-of-bounds accesses
        if (global_i_idx < num_rhs_ && partial_global_j_idx < num_mirror_rows_ && global_j_idx < num_rows_) {
            real_type temp{ 0.0 };

            // iterate over all values
            for (std::size_t dim = 0; dim < device_num_rows_; ++dim) {
                // perform the dot product calculation
                temp += A_[dim * (num_rows_ - device_row_offset_) - (dim - std::size_t{ 1 }) * dim / std::size_t{ 2 } + device_num_rows_ - dim + partial_global_j_idx] *  // SoA, upper triangular matrix only
                        B_[(device_row_offset_ + dim) * num_rhs_ + global_i_idx];                                                                                           // SoA
            }

            // apply the (remaining) BLAS operation and update C
            C_[global_j_idx * num_rhs_ + global_i_idx] = alpha_ * temp + beta_ * C_[global_j_idx * num_rhs_ + global_i_idx];  // SoA
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
    device_view_type<const real_type> A_;
    device_view_type<const real_type> B_;
    const real_type beta_;
    device_view_type<real_type> C_;
    const std::size_t grid_x_offset_;
    const std::size_t grid_y_offset_;
    const std::size_t grid_size_x_;
    /// @endcond
};

/**
 * @brief Perform a simple inplace matrix addition: lhs += rhs.
 * @tparam ExecutionSpace the Kokkos::ExecutionSpace used to execute the kernel
 */
template <typename ExecutionSpace>
class device_kernel_inplace_matrix_add {
    /**
     * @brief The type of the used Kokkos::View.
     */
    template <typename T>
    using device_view_type = Kokkos::View<T *, ExecutionSpace>;

  public:
    /**
     * @brief Initialize the Kokkos kernel function object.
     * @param[in] num_rows the number of rows in the matrix
     * @param[in] num_cols the number of columns in both matrices
     * @param[in,out] lhs the first matrix (updated inplace)
     * @param[in] rhs the second matrix
     * @param[in] grid_x_offset the offset in x-dimension into the data points if more than one execution grid has to be used
     * @param[in] grid_y_offset the offset in y-dimension into the data points if more than one execution grid has to be used
     * @param[in] grid_size_x the size of the execution grid in x-dimension
     */
    device_kernel_inplace_matrix_add(const std::size_t num_rows, const std::size_t num_cols, device_view_type<real_type> lhs, device_view_type<const real_type> rhs, const std::size_t grid_x_offset, const std::size_t grid_y_offset, const std::size_t grid_size_x) :
        num_rows_{ num_rows },
        num_cols_{ num_cols },
        lhs_{ lhs },
        rhs_{ rhs },
        grid_x_offset_{ grid_x_offset },
        grid_y_offset_{ grid_y_offset },
        grid_size_x_{ grid_size_x } { }

    /**
     * @brief Function call operator overload performing the actual calculation.
     * @param[in] team the Kokkos team representing the current point in the execution space
     */
    KOKKOS_INLINE_FUNCTION
    void operator()(const typename Kokkos::TeamPolicy<ExecutionSpace>::member_type &team) const {
        // cast all values to 64-bit std::size_t to prevent potential 32-bit overflows
        constexpr auto THREAD_BLOCK_SIZE_uz = static_cast<std::size_t>(THREAD_BLOCK_SIZE);

        const auto threadIdx_x = static_cast<std::size_t>(team.team_rank()) / THREAD_BLOCK_SIZE_uz;            // current thread in team x-dimension
        const auto threadIdx_y = static_cast<std::size_t>(team.team_rank()) % THREAD_BLOCK_SIZE_uz;            // current thread in team y-dimension
        const auto blockDim_x = THREAD_BLOCK_SIZE_uz;                                                          // number of threads in team x-dimension
        const auto blockDim_y = THREAD_BLOCK_SIZE_uz;                                                          // number of threads in team y-dimension
        const auto blockIdx_x = static_cast<std::size_t>(team.league_rank()) % grid_size_x_ + grid_x_offset_;  // current team in league x-dimension + offsets if the league size is too large
        const auto blockIdx_y = static_cast<std::size_t>(team.league_rank()) / grid_size_x_ + grid_y_offset_;  // current team in league y-dimension + offsets if the league size is too large

        // calculate the indices used in the current thread
        const auto global_i_idx = blockIdx_x * blockDim_x + threadIdx_x;  // num_rows
        const auto global_j_idx = blockIdx_y * blockDim_y + threadIdx_y;  // num_rhs

        if (global_i_idx < num_rows_ && global_j_idx < num_cols_) {
            lhs_[global_i_idx * num_cols_ + global_j_idx] += rhs_[global_i_idx * num_cols_ + global_j_idx];  // SoA
        }
    }

  private:
    /// @cond Doxygen_suppress
    const std::size_t num_rows_;
    const std::size_t num_cols_;
    device_view_type<real_type> lhs_;
    device_view_type<const real_type> rhs_;
    const std::size_t grid_x_offset_;
    const std::size_t grid_y_offset_;
    const std::size_t grid_size_x_;
    /// @endcond
};

/**
 * @brief Perform a simple inplace matrix scale: lhs *= scalar.
 * @tparam ExecutionSpace the Kokkos::ExecutionSpace used to execute the kernel
 */
template <typename ExecutionSpace>
class device_kernel_inplace_matrix_scale {
    /**
     * @brief The type of the used Kokkos::View.
     */
    template <typename T>
    using device_view_type = Kokkos::View<T *, ExecutionSpace>;

  public:
    /**
     * @brief Initialize the Kokkos kernel function object.
     * @param[in] num_rows the number of rows in the matrix
     * @param[in] num_cols the number of columns in the matrix
     * @param[in,out] lhs the first matrix (updated inplace)
     * @param[in] scale the value to scale
     * @param[in] grid_x_offset the offset in x-dimension into the data points if more than one execution grid has to be used
     * @param[in] grid_y_offset the offset in y-dimension into the data points if more than one execution grid has to be used
     * @param[in] grid_size_x the size of the execution grid in x-dimension
     */
    device_kernel_inplace_matrix_scale(const std::size_t num_rows, const std::size_t num_cols, device_view_type<real_type> lhs, const real_type scale, const std::size_t grid_x_offset, const std::size_t grid_y_offset, const std::size_t grid_size_x) :
        num_rows_{ num_rows },
        num_cols_{ num_cols },
        lhs_{ lhs },
        scale_{ scale },
        grid_x_offset_{ grid_x_offset },
        grid_y_offset_{ grid_y_offset },
        grid_size_x_{ grid_size_x } { }

    /**
     * @brief Function call operator overload performing the actual calculation.
     * @param[in] team the Kokkos team representing the current point in the execution space
     */
    KOKKOS_INLINE_FUNCTION
    void operator()(const typename Kokkos::TeamPolicy<ExecutionSpace>::member_type &team) const {
        // cast all values to 64-bit std::size_t to prevent potential 32-bit overflows
        constexpr auto THREAD_BLOCK_SIZE_uz = static_cast<std::size_t>(THREAD_BLOCK_SIZE);

        const auto threadIdx_x = static_cast<std::size_t>(team.team_rank()) / THREAD_BLOCK_SIZE_uz;            // current thread in team x-dimension
        const auto threadIdx_y = static_cast<std::size_t>(team.team_rank()) % THREAD_BLOCK_SIZE_uz;            // current thread in team y-dimension
        const auto blockDim_x = THREAD_BLOCK_SIZE_uz;                                                          // number of threads in team x-dimension
        const auto blockDim_y = THREAD_BLOCK_SIZE_uz;                                                          // number of threads in team y-dimension
        const auto blockIdx_x = static_cast<std::size_t>(team.league_rank()) % grid_size_x_ + grid_x_offset_;  // current team in league x-dimension + offsets if the league size is too large
        const auto blockIdx_y = static_cast<std::size_t>(team.league_rank()) / grid_size_x_ + grid_y_offset_;  // current team in league y-dimension + offsets if the league size is too large

        // calculate the indices used in the current thread
        const auto global_i_idx = blockIdx_x * blockDim_x + threadIdx_x;  // num_rows
        const auto global_j_idx = blockIdx_y * blockDim_y + threadIdx_y;  // num_rhs

        if (global_i_idx < num_rows_ && global_j_idx < num_cols_) {
            lhs_[global_i_idx * num_cols_ + global_j_idx] *= scale_;  // SoA
        }
    }

  private:
    /// @cond Doxygen_suppress
    const std::size_t num_rows_;
    const std::size_t num_cols_;
    device_view_type<real_type> lhs_;
    const real_type scale_;
    const std::size_t grid_x_offset_;
    const std::size_t grid_y_offset_;
    const std::size_t grid_size_x_;
    /// @endcond
};

}  // namespace plssvm::kokkos::detail

#endif  // PLSSVM_BACKENDS_KOKKOS_CG_EXPLICIT_BLAS_HPP_
