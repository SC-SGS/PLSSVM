/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 *-+ * @brief Functions for explicitly performing a BLAS GEMM like matrix-matrix multiplication using the Kokkos backend.
 */

#ifndef PLSSVM_BACKENDS_KOKKOS_KERNEL_CG_EXPLICIT_BLAS_HPP_
#define PLSSVM_BACKENDS_KOKKOS_KERNEL_CG_EXPLICIT_BLAS_HPP_
#pragma once

#include "plssvm/constants.hpp"  // plssvm::{real_type, THREAD_BLOCK_SIZE}

#include "Kokkos_Core.hpp"  // KOKKOS_INLINE_FUNCTION, Kokkos::View, Kokkos::TeamPolicy, Kokkos::TeamPolicy, Kokkos::mdspan

#include <cstddef>  // std::size_t
#include <utility>  // std::move

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
        A_{ std::move(A) },
        B_{ std::move(B) },
        beta_{ beta },
        C_{ std::move(C) },
        grid_x_offset_{ grid_x_offset },
        grid_y_offset_{ grid_y_offset },
        grid_size_x_{ grid_size_x } { }

    /**
     * @brief Function call operator overload performing the actual calculation.
     * @param[in] team the Kokkos team representing the current point in the execution space
     */
    KOKKOS_INLINE_FUNCTION
    void operator()(const typename Kokkos::TeamPolicy<ExecutionSpace>::member_type &team) const {
        // cast values to 32-bit unsigned int values to prevent implicit conversions
        const auto team_rank_x = static_cast<unsigned>(team.team_rank()) / THREAD_BLOCK_SIZE;
        const auto team_rank_y = static_cast<unsigned>(team.team_rank()) % THREAD_BLOCK_SIZE;

        // cast all values to 64-bit std::size_t to prevent potential 32-bit overflows
        constexpr auto INTERNAL_BLOCK_SIZE_uz = static_cast<std::size_t>(INTERNAL_BLOCK_SIZE);
        constexpr auto THREAD_BLOCK_SIZE_uz = static_cast<std::size_t>(THREAD_BLOCK_SIZE);

        const auto threadIdx_x = static_cast<std::size_t>(team.team_rank()) / THREAD_BLOCK_SIZE_uz;            // current thread in team x-dimension
        const auto threadIdx_y = static_cast<std::size_t>(team.team_rank()) % THREAD_BLOCK_SIZE_uz;            // current thread in team y-dimension
        const auto blockDim_x = THREAD_BLOCK_SIZE_uz;                                                          // number of threads in team x-dimension
        const auto blockDim_y = THREAD_BLOCK_SIZE_uz;                                                          // number of threads in team y-dimension
        const auto blockIdx_x = static_cast<std::size_t>(team.league_rank()) % grid_size_x_ + grid_x_offset_;  // current team in league x-dimension + offsets if the league size is too large
        const auto blockIdx_y = static_cast<std::size_t>(team.league_rank()) / grid_size_x_ + grid_y_offset_;  // current team in league y-dimension + offsets if the league size is too large

        // create two scratchpad memory arrays used for caching
        constexpr std::size_t scratchpad_size = THREAD_BLOCK_SIZE_uz * THREAD_BLOCK_SIZE_uz * INTERNAL_BLOCK_SIZE_uz;
        real_type *scratchpad_ptr = static_cast<real_type *>(team.team_shmem().get_shmem(std::size_t{ 2 } * scratchpad_size * sizeof(real_type)));
        Kokkos::mdspan<real_type, Kokkos::dextents<std::size_t, 2>> A_cache{ scratchpad_ptr, THREAD_BLOCK_SIZE_uz, INTERNAL_BLOCK_SIZE_uz * THREAD_BLOCK_SIZE_uz };
        Kokkos::mdspan<real_type, Kokkos::dextents<std::size_t, 2>> B_cache{ scratchpad_ptr + scratchpad_size, THREAD_BLOCK_SIZE_uz, INTERNAL_BLOCK_SIZE_uz * THREAD_BLOCK_SIZE_uz };

        // create a thread private array used for internal caching
        real_type temp[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE]{};

        {
            // calculate the indices used in the current thread, pays attention to coalesced memory accesses
            const auto i_idx_linear = blockIdx_x * blockDim_x * INTERNAL_BLOCK_SIZE_uz + threadIdx_x;  // num_rhs
            const auto j_idx_linear = blockIdx_y * blockDim_y * INTERNAL_BLOCK_SIZE_uz + threadIdx_x;  // device_num_rows

            // iterate over all values using blocking to be able to cache them for faster memory accesses
            for (std::size_t dim_block = 0; dim_block < (num_rows_ - device_row_offset_); dim_block += THREAD_BLOCK_SIZE_uz) {
                // zero-out shared memory
                for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                    A_cache(team_rank_y, internal * THREAD_BLOCK_SIZE + team_rank_x) = real_type{ 0.0 };
                    B_cache(team_rank_y, internal * THREAD_BLOCK_SIZE + team_rank_x) = real_type{ 0.0 };
                }

                // load data into scratchpad memory
                for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                    // calculate the indices to access the global data, pays attention to coalesced memory accesses
                    const auto global_i_idx_linear = i_idx_linear + static_cast<std::size_t>(internal) * THREAD_BLOCK_SIZE_uz;
                    const auto global_j_idx_linear = j_idx_linear + static_cast<std::size_t>(internal) * THREAD_BLOCK_SIZE_uz;

                    // store the values in the scratchpad memory
                    if (dim_block + threadIdx_y < num_rows_ - device_row_offset_) {
                        if (global_j_idx_linear < device_num_rows_) {
                            // determine on which side of the diagonal we are located
                            if (dim_block + threadIdx_y < global_j_idx_linear) {
                                A_cache(team_rank_y, internal * THREAD_BLOCK_SIZE + team_rank_x) = A_[(dim_block + threadIdx_y) * (num_rows_ - device_row_offset_) + global_j_idx_linear - (dim_block + threadIdx_y) * (dim_block + threadIdx_y + std::size_t{ 1 }) / std::size_t{ 2 }];  // SoA, upper triangular matrix only
                            } else {
                                A_cache(team_rank_y, internal * THREAD_BLOCK_SIZE + team_rank_x) = A_[global_j_idx_linear * (num_rows_ - device_row_offset_) + dim_block + threadIdx_y - global_j_idx_linear * (global_j_idx_linear + std::size_t{ 1 }) / std::size_t{ 2 }];  // SoA, upper triangular matrix only
                            }
                        }
                        if (global_i_idx_linear < num_rhs_) {
                            B_cache(team_rank_y, internal * THREAD_BLOCK_SIZE + team_rank_x) = B_[(dim_block + device_row_offset_ + threadIdx_y) * num_rhs_ + global_i_idx_linear];  // SoA
                        }
                    }
                }
                team.team_barrier();  // wait until all threads loaded their part of the data

                // perform the dot product calculation
                for (unsigned dim = 0; dim < THREAD_BLOCK_SIZE; ++dim) {
                    for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                        for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                            temp[internal_i][internal_j] += A_cache(dim, team_rank_y * INTERNAL_BLOCK_SIZE + internal_j) * B_cache(dim, team_rank_x * INTERNAL_BLOCK_SIZE + internal_i);
                        }
                    }
                }
                team.team_barrier();  // wait until all threads performed their part of the calculations
            }
        }

        // calculate the indices used in the current thread
        const auto i_idx = (blockIdx_x * blockDim_x + threadIdx_x) * INTERNAL_BLOCK_SIZE_uz;  // num_rhs
        const auto j_idx = (blockIdx_y * blockDim_y + threadIdx_y) * INTERNAL_BLOCK_SIZE_uz;  // device_num_rows

        // apply the (partial) BLAS operation and update C
        for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
            for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                // calculate the indices to access the global data and the data with respect to the current device
                const auto global_i_idx = i_idx + static_cast<std::size_t>(internal_i);
                const auto device_global_j_idx = j_idx + static_cast<std::size_t>(internal_j);
                const auto global_j_idx = device_row_offset_ + device_global_j_idx;

                // be sure to not perform out-of-bounds accesses
                if (global_i_idx < num_rhs_ && global_j_idx < num_rows_ && device_global_j_idx < device_num_rows_) {
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
        A_{ std::move(A) },
        B_{ std::move(B) },
        beta_{ beta },
        C_{ std::move(C) },
        grid_x_offset_{ grid_x_offset },
        grid_y_offset_{ grid_y_offset },
        grid_size_x_{ grid_size_x } { }

    /**
     * @brief Function call operator overload performing the actual calculation.
     * @param[in] team the Kokkos team representing the current point in the execution space
     */
    KOKKOS_INLINE_FUNCTION
    void operator()(const typename Kokkos::TeamPolicy<ExecutionSpace>::member_type &team) const {
        // cast values to 32-bit unsigned int values to prevent implicit conversions
        const auto team_rank_x = static_cast<unsigned>(team.team_rank()) / THREAD_BLOCK_SIZE;
        const auto team_rank_y = static_cast<unsigned>(team.team_rank()) % THREAD_BLOCK_SIZE;

        // cast all values to 64-bit std::size_t to prevent potential 32-bit overflows
        constexpr auto INTERNAL_BLOCK_SIZE_uz = static_cast<std::size_t>(INTERNAL_BLOCK_SIZE);
        constexpr auto THREAD_BLOCK_SIZE_uz = static_cast<std::size_t>(THREAD_BLOCK_SIZE);

        const auto threadIdx_x = static_cast<std::size_t>(team.team_rank()) / THREAD_BLOCK_SIZE_uz;            // current thread in team x-dimension
        const auto threadIdx_y = static_cast<std::size_t>(team.team_rank()) % THREAD_BLOCK_SIZE_uz;            // current thread in team y-dimension
        const auto blockDim_x = THREAD_BLOCK_SIZE_uz;                                                          // number of threads in team x-dimension
        const auto blockDim_y = THREAD_BLOCK_SIZE_uz;                                                          // number of threads in team y-dimension
        const auto blockIdx_x = static_cast<std::size_t>(team.league_rank()) % grid_size_x_ + grid_x_offset_;  // current team in league x-dimension + offsets if the league size is too large
        const auto blockIdx_y = static_cast<std::size_t>(team.league_rank()) / grid_size_x_ + grid_y_offset_;  // current team in league y-dimension + offsets if the league size is too large

        // create two shared memory arrays used for caching
        constexpr std::size_t scratchpad_size = THREAD_BLOCK_SIZE_uz * THREAD_BLOCK_SIZE_uz * INTERNAL_BLOCK_SIZE_uz;
        real_type *scratchpad_ptr = static_cast<real_type *>(team.team_shmem().get_shmem(std::size_t{ 2 } * scratchpad_size * sizeof(real_type)));
        Kokkos::mdspan<real_type, Kokkos::dextents<std::size_t, 2>> A_cache{ scratchpad_ptr, THREAD_BLOCK_SIZE_uz, INTERNAL_BLOCK_SIZE_uz * THREAD_BLOCK_SIZE_uz };
        Kokkos::mdspan<real_type, Kokkos::dextents<std::size_t, 2>> B_cache{ scratchpad_ptr + scratchpad_size, THREAD_BLOCK_SIZE_uz, INTERNAL_BLOCK_SIZE_uz * THREAD_BLOCK_SIZE_uz };

        // create a thread private array used for internal caching
        real_type temp[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE]{};

        {
            // calculate the indices used in the current thread, pays attention to coalesced memory accesses
            const auto i_idx_linear = blockIdx_x * blockDim_x * INTERNAL_BLOCK_SIZE_uz + threadIdx_x;  // num_rhs
            const auto j_idx_linear = blockIdx_y * blockDim_y * INTERNAL_BLOCK_SIZE_uz + threadIdx_x;  // num_mirror_rows

            // iterate over the remaining values using blocking to be able to cache them for faster memory accesses
            for (std::size_t dim_block = 0; dim_block < device_num_rows_; dim_block += THREAD_BLOCK_SIZE_uz) {
                // zero-out shared memory
                for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                    A_cache(team_rank_y, internal * THREAD_BLOCK_SIZE + team_rank_x) = real_type{ 0.0 };
                    B_cache(team_rank_y, internal * THREAD_BLOCK_SIZE + team_rank_x) = real_type{ 0.0 };
                }

                // load data into scratchpad memory
                for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                    // calculate the indices to access the global data, pays attention to coalesced memory accesses
                    const auto global_i_idx_linear = i_idx_linear + static_cast<std::size_t>(internal) * THREAD_BLOCK_SIZE_uz;
                    const auto global_j_idx_linear = j_idx_linear + static_cast<std::size_t>(internal) * THREAD_BLOCK_SIZE_uz;

                    // store the values in the scratchpad memory
                    if (dim_block + threadIdx_y < device_num_rows_) {
                        if (global_j_idx_linear < num_mirror_rows_) {
                            A_cache(team_rank_y, internal * THREAD_BLOCK_SIZE + team_rank_x) = A_[(dim_block + threadIdx_y) * (num_rows_ - device_row_offset_) - (dim_block + threadIdx_y - std::size_t{ 1 }) * (dim_block + threadIdx_y) / std::size_t{ 2 } + device_num_rows_ - (dim_block + threadIdx_y) + global_j_idx_linear];  // SoA, upper triangular matrix only
                        }
                        if (global_i_idx_linear < num_rhs_) {
                            B_cache(team_rank_y, internal * THREAD_BLOCK_SIZE + team_rank_x) = B_[(device_row_offset_ + dim_block + threadIdx_y) * num_rhs_ + global_i_idx_linear];  // SoA
                        }
                    }
                }
                team.team_barrier();  // wait until all threads loaded their part of the data

                // perform the dot product calculation
                for (unsigned dim = 0; dim < THREAD_BLOCK_SIZE; ++dim) {
                    for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                        for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                            temp[internal_i][internal_j] += A_cache(dim, team_rank_y * INTERNAL_BLOCK_SIZE + internal_j) * B_cache(dim, team_rank_x * INTERNAL_BLOCK_SIZE + internal_i);
                        }
                    }
                }
                team.team_barrier();  // wait until all threads performed their part of the calculations
            }
        }

        // calculate the indices used in the current thread
        const auto i_idx = (blockIdx_x * blockDim_x + threadIdx_x) * INTERNAL_BLOCK_SIZE_uz;  // num_rhs
        const auto j_idx = (blockIdx_y * blockDim_y + threadIdx_y) * INTERNAL_BLOCK_SIZE_uz;  // num_mirror_rows

        // apply the (remaining) BLAS operation and update C
        for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
            for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                // calculate the indices to access the global data and the data with respect to the current device
                const auto global_i_idx = i_idx + static_cast<std::size_t>(internal_i);
                const auto partial_global_j_idx = j_idx + static_cast<std::size_t>(internal_j);
                const auto global_j_idx = device_row_offset_ + device_num_rows_ + partial_global_j_idx;

                // be sure to not perform out-of-bounds accesses
                if (global_i_idx < num_rhs_ && global_j_idx < num_rows_ && partial_global_j_idx < num_mirror_rows_) {
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
     * @param[in] num_rows the number of rows in both matrices
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
        lhs_{ std::move(lhs) },
        rhs_{ std::move(rhs) },
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
        constexpr auto INTERNAL_BLOCK_SIZE_uz = static_cast<std::size_t>(INTERNAL_BLOCK_SIZE);
        constexpr auto THREAD_BLOCK_SIZE_uz = static_cast<std::size_t>(THREAD_BLOCK_SIZE);

        const auto threadIdx_x = static_cast<std::size_t>(team.team_rank()) / THREAD_BLOCK_SIZE_uz;            // current thread in team x-dimension
        const auto threadIdx_y = static_cast<std::size_t>(team.team_rank()) % THREAD_BLOCK_SIZE_uz;            // current thread in team y-dimension
        const auto blockDim_x = THREAD_BLOCK_SIZE_uz;                                                          // number of threads in team x-dimension
        const auto blockDim_y = THREAD_BLOCK_SIZE_uz;                                                          // number of threads in team y-dimension
        const auto blockIdx_x = static_cast<std::size_t>(team.league_rank()) % grid_size_x_ + grid_x_offset_;  // current team in league x-dimension + offsets if the league size is too large
        const auto blockIdx_y = static_cast<std::size_t>(team.league_rank()) / grid_size_x_ + grid_y_offset_;  // current team in league y-dimension + offsets if the league size is too large

        // calculate the indices used in the current thread
        const auto i_idx = (blockIdx_x * blockDim_x + threadIdx_x) * INTERNAL_BLOCK_SIZE_uz;  // num_rows
        const auto j_idx = (blockIdx_y * blockDim_y + threadIdx_y) * INTERNAL_BLOCK_SIZE_uz;  // num_rhs

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
     * @param[in,out] lhs the matrix (updated inplace)
     * @param[in] scale the value to scale
     * @param[in] grid_x_offset the offset in x-dimension into the data points if more than one execution grid has to be used
     * @param[in] grid_y_offset the offset in y-dimension into the data points if more than one execution grid has to be used
     * @param[in] grid_size_x the size of the execution grid in x-dimension
     */
    device_kernel_inplace_matrix_scale(const std::size_t num_rows, const std::size_t num_cols, device_view_type<real_type> lhs, const real_type scale, const std::size_t grid_x_offset, const std::size_t grid_y_offset, const std::size_t grid_size_x) :
        num_rows_{ num_rows },
        num_cols_{ num_cols },
        lhs_{ std::move(lhs) },
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
        constexpr auto INTERNAL_BLOCK_SIZE_uz = static_cast<std::size_t>(INTERNAL_BLOCK_SIZE);
        constexpr auto THREAD_BLOCK_SIZE_uz = static_cast<std::size_t>(THREAD_BLOCK_SIZE);

        const auto threadIdx_x = static_cast<std::size_t>(team.team_rank()) / THREAD_BLOCK_SIZE_uz;            // current thread in team x-dimension
        const auto threadIdx_y = static_cast<std::size_t>(team.team_rank()) % THREAD_BLOCK_SIZE_uz;            // current thread in team y-dimension
        const auto blockDim_x = THREAD_BLOCK_SIZE_uz;                                                          // number of threads in team x-dimension
        const auto blockDim_y = THREAD_BLOCK_SIZE_uz;                                                          // number of threads in team y-dimension
        const auto blockIdx_x = static_cast<std::size_t>(team.league_rank()) % grid_size_x_ + grid_x_offset_;  // current team in league x-dimension + offsets if the league size is too large
        const auto blockIdx_y = static_cast<std::size_t>(team.league_rank()) / grid_size_x_ + grid_y_offset_;  // current team in league y-dimension + offsets if the league size is too large

        // calculate the indices used in the current thread
        const auto i_idx = (blockIdx_x * blockDim_x + threadIdx_x) * INTERNAL_BLOCK_SIZE_uz;  // num_rows
        const auto j_idx = (blockIdx_y * blockDim_y + threadIdx_y) * INTERNAL_BLOCK_SIZE_uz;  // num_rhs

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
    device_view_type<real_type> lhs_;
    const real_type scale_;
    const std::size_t grid_x_offset_;
    const std::size_t grid_y_offset_;
    const std::size_t grid_size_x_;
    /// @endcond
};

}  // namespace plssvm::kokkos::detail

#endif  // PLSSVM_BACKENDS_KOKKOS_KERNEL_CG_EXPLICIT_BLAS_HPP_
