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

#include "plssvm/backends/Kokkos/detail/typedefs.hpp"  // plssvm::kokkos::detail::device_view_type
#include "plssvm/constants.hpp"                        // plssvm::{real_type, THREAD_BLOCK_SIZE, INTERNAL_BLOCK_SIZE, FEATURE_BLOCK_SIZE, PADDING_SIZE}

#include "Kokkos_Core.hpp"  // KOKKOS_INLINE_FUNCTION, Kokkos::TeamPolicy, Kokkos::mdspan, Kokkos::dextents

#include <cstddef>  // std::size_t

namespace plssvm::kokkos::detail {

/**
 * @brief Perform an explicit BLAS SYMM operation: `C = alpha * A * B + beta * C` where @p A is a `m x k` symmetric matrix (memory optimized), @p B is a `k x n` matrix, @p C is a `m x n` matrix, and @p alpha and @p beta are scalars.
 */
class device_kernel_symm {
  public:
    /**
     * @brief Initialize the Kokkos kernel function object.
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
    device_kernel_symm(const std::size_t num_rows, const std::size_t num_rhs, const std::size_t device_specific_num_rows, const std::size_t row_offset, const real_type alpha, device_view_type<const real_type> A, device_view_type<const real_type> B, const real_type beta, device_view_type<real_type> C, const std::size_t grid_x_offset, const std::size_t grid_y_offset, const std::size_t grid_size_x) :
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
        grid_y_offset_{ grid_y_offset },
        grid_size_x_{ grid_size_x } { }

    KOKKOS_INLINE_FUNCTION
    void operator()(const Kokkos::TeamPolicy<>::member_type &team) const {
        // cast all values to 64-bit std::size_t to prevent potential 32-bit overflows
        const auto INTERNAL_BLOCK_SIZE_sz = static_cast<std::size_t>(INTERNAL_BLOCK_SIZE);
        const auto THREAD_BLOCK_SIZE_sz = static_cast<std::size_t>(THREAD_BLOCK_SIZE);
        const auto FEATURE_BLOCK_SIZE_sz = static_cast<std::size_t>(FEATURE_BLOCK_SIZE);
        const auto PADDING_SIZE_sz = static_cast<std::size_t>(PADDING_SIZE);
        const auto threadIdx_x = static_cast<std::size_t>(team.team_rank()) / THREAD_BLOCK_SIZE_sz;            // current thread in block x-dimension
        const auto threadIdx_y = static_cast<std::size_t>(team.team_rank()) % THREAD_BLOCK_SIZE_sz;            // current thread in block y-dimension
        const auto blockDim_x = THREAD_BLOCK_SIZE_sz;                                                          // number of threads in block x-dimension
        const auto blockDim_y = THREAD_BLOCK_SIZE_sz;                                                          // number of threads in block y-dimension
        const auto blockIdx_x = static_cast<std::size_t>(team.league_rank()) % grid_size_x_ + grid_x_offset_;  // current block in grid x-dimension + offsets if the grid size would be too large
        const auto blockIdx_y = static_cast<std::size_t>(team.league_rank()) / grid_size_x_ + grid_y_offset_;  // current block in grid y-dimension + offsets if the grid size would be too large

        // calculate the indices used in the current thread
        const auto i = (blockIdx_x * blockDim_x + threadIdx_x) * INTERNAL_BLOCK_SIZE_sz;  // # rhs -> num_rhs
        const auto i_linear = blockIdx_x * blockDim_x * INTERNAL_BLOCK_SIZE_sz + threadIdx_x;
        const auto j = (blockIdx_y * blockDim_y + threadIdx_y) * INTERNAL_BLOCK_SIZE_sz;  // # rows -> num_mirror_rows
        const auto j_linear = blockIdx_y * blockDim_y * INTERNAL_BLOCK_SIZE_sz + threadIdx_x;

        // create the shared memory arrays used for caching data point features
        constexpr std::size_t shmem_size = FEATURE_BLOCK_SIZE * THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE;
        real_type *data_cache_ptr = static_cast<real_type *>(team.team_shmem().get_shmem(2 * shmem_size * sizeof(real_type)));
        Kokkos::mdspan<real_type, Kokkos::dextents<std::size_t, 2>> A_cache{ data_cache_ptr, FEATURE_BLOCK_SIZE, INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE };
        Kokkos::mdspan<real_type, Kokkos::dextents<std::size_t, 2>> B_cache{ data_cache_ptr + shmem_size, FEATURE_BLOCK_SIZE, INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE };

        // create a thread private array used for internal caching
        real_type temp[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE]{};

        // iterate over all features using blocking to be able to cache them for faster memory accesses
        for (std::size_t dim = 0; dim < (num_rows_ - row_offset_); dim += FEATURE_BLOCK_SIZE_sz) {
            // load data into shared memory
            for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                const auto global_i = i_linear + static_cast<std::size_t>(internal) * THREAD_BLOCK_SIZE_sz;
                const auto global_j = j_linear + static_cast<std::size_t>(internal) * THREAD_BLOCK_SIZE_sz;

                // determine on which side of the diagonal we are located
                if (dim + threadIdx_y < global_j) {
                    A_cache(threadIdx_y, internal * THREAD_BLOCK_SIZE + threadIdx_x) = A_[(dim + threadIdx_y) * (num_rows_ - row_offset_ + PADDING_SIZE_sz) + global_j - (dim + threadIdx_y) * (dim + threadIdx_y + std::size_t{ 1 }) / std::size_t{ 2 }];
                } else {
                    A_cache(threadIdx_y, internal * THREAD_BLOCK_SIZE + threadIdx_x) = A_[global_j * (num_rows_ - row_offset_ + PADDING_SIZE_sz) + dim + threadIdx_y - global_j * (global_j + std::size_t{ 1 }) / std::size_t{ 2 }];
                }
                // determine on which side of the diagonal we are located
                if (dim + threadIdx_y + THREAD_BLOCK_SIZE < global_j) {
                    A_cache(threadIdx_y + THREAD_BLOCK_SIZE, internal * THREAD_BLOCK_SIZE + threadIdx_x) = A_[(dim + threadIdx_y + THREAD_BLOCK_SIZE_sz) * (num_rows_ - row_offset_ + PADDING_SIZE_sz) + global_j - (dim + threadIdx_y + THREAD_BLOCK_SIZE_sz) * (dim + threadIdx_y + THREAD_BLOCK_SIZE_sz + std::size_t{ 1 }) / std::size_t{ 2 }];
                } else {
                    A_cache(threadIdx_y + THREAD_BLOCK_SIZE, internal * THREAD_BLOCK_SIZE + threadIdx_x) = A_[global_j * (num_rows_ - row_offset_ + PADDING_SIZE_sz) + dim + threadIdx_y + THREAD_BLOCK_SIZE_sz - global_j * (global_j + std::size_t{ 1 }) / std::size_t{ 2 }];
                }

                B_cache(threadIdx_y, internal * THREAD_BLOCK_SIZE + threadIdx_x) = B_[(dim + row_offset_ + threadIdx_y) * (num_rhs_ + PADDING_SIZE_sz) + global_i];
                B_cache(threadIdx_y + THREAD_BLOCK_SIZE, internal * THREAD_BLOCK_SIZE + threadIdx_x) = B_[(dim + row_offset_ + threadIdx_y + THREAD_BLOCK_SIZE_sz) * (num_rhs_ + PADDING_SIZE_sz) + global_i];
            }
            team.team_barrier();  // wait until all threads loaded their part of the data

            // perform the dot product calculation
            for (unsigned block_dim = 0; block_dim < FEATURE_BLOCK_SIZE; ++block_dim) {
                for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                    for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                        temp[internal_i][internal_j] += A_cache(block_dim, threadIdx_y * INTERNAL_BLOCK_SIZE + internal_j) * B_cache(block_dim, threadIdx_x * INTERNAL_BLOCK_SIZE + internal_i);
                    }
                }
            }
            team.team_barrier();  // wait until all threads performed their part of the calculations
        }

        // apply the (partial) BLAS operation and update C
        for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
            for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                const auto global_i = i + static_cast<std::size_t>(internal_i);
                const auto device_global_j = j + static_cast<std::size_t>(internal_j);
                const auto global_j = row_offset_ + j + static_cast<std::size_t>(internal_j);

                // be sure to not perform out of bounds accesses
                if (global_i < num_rhs_ && device_global_j < device_specific_num_rows_) {
                    C_[global_j * (num_rhs_ + PADDING_SIZE_sz) + global_i] = alpha_ * temp[internal_i][internal_j] + beta_ * C_[global_j * (num_rhs_ + PADDING_SIZE_sz) + global_i];
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
 */
class device_kernel_symm_mirror {
  public:
    /**
     * @brief Initialize the Kokkos kernel function object.
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
    device_kernel_symm_mirror(const std::size_t num_rows, const std::size_t num_rhs, const std::size_t num_mirror_rows, const std::size_t device_specific_num_rows, const std::size_t row_offset, const real_type alpha, device_view_type<const real_type> A, device_view_type<const real_type> B, const real_type beta, device_view_type<real_type> C, const std::size_t grid_x_offset, const std::size_t grid_y_offset, const std::size_t grid_size_x) :
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
        grid_y_offset_{ grid_y_offset },
        grid_size_x_{ grid_size_x } { }

    KOKKOS_INLINE_FUNCTION
    void operator()(const Kokkos::TeamPolicy<>::member_type &team) const {
        // cast all values to 64-bit std::size_t to prevent potential 32-bit overflows
        const auto INTERNAL_BLOCK_SIZE_sz = static_cast<std::size_t>(INTERNAL_BLOCK_SIZE);
        const auto THREAD_BLOCK_SIZE_sz = static_cast<std::size_t>(THREAD_BLOCK_SIZE);
        const auto FEATURE_BLOCK_SIZE_sz = static_cast<std::size_t>(FEATURE_BLOCK_SIZE);
        const auto PADDING_SIZE_sz = static_cast<std::size_t>(PADDING_SIZE);
        const auto threadIdx_x = static_cast<std::size_t>(team.team_rank()) / THREAD_BLOCK_SIZE_sz;            // current thread in block x-dimension
        const auto threadIdx_y = static_cast<std::size_t>(team.team_rank()) % THREAD_BLOCK_SIZE_sz;            // current thread in block y-dimension
        const auto blockDim_x = THREAD_BLOCK_SIZE_sz;                                                          // number of threads in block x-dimension
        const auto blockDim_y = THREAD_BLOCK_SIZE_sz;                                                          // number of threads in block y-dimension
        const auto blockIdx_x = static_cast<std::size_t>(team.league_rank()) % grid_size_x_ + grid_x_offset_;  // current block in grid x-dimension + offsets if the grid size would be too large
        const auto blockIdx_y = static_cast<std::size_t>(team.league_rank()) / grid_size_x_ + grid_y_offset_;  // current block in grid y-dimension + offsets if the grid size would be too large

        // calculate the indices used in the current thread
        const auto i = (blockIdx_x * blockDim_x + threadIdx_x) * INTERNAL_BLOCK_SIZE_sz;  // # rhs -> num_rhs
        const auto i_linear = blockIdx_x * blockDim_x * INTERNAL_BLOCK_SIZE_sz + threadIdx_x;
        const auto j = (blockIdx_y * blockDim_y + threadIdx_y) * INTERNAL_BLOCK_SIZE_sz;  // # rows -> num_mirror_rows
        const auto j_linear = blockIdx_y * blockDim_y * INTERNAL_BLOCK_SIZE_sz + threadIdx_x;

        // create the shared memory arrays used for caching data point features
        constexpr std::size_t shmem_size = FEATURE_BLOCK_SIZE * THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE;
        real_type *data_cache_ptr = static_cast<real_type *>(team.team_shmem().get_shmem(2 * shmem_size * sizeof(real_type)));
        Kokkos::mdspan<real_type, Kokkos::dextents<std::size_t, 2>> A_cache{ data_cache_ptr, FEATURE_BLOCK_SIZE, INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE };
        Kokkos::mdspan<real_type, Kokkos::dextents<std::size_t, 2>> B_cache{ data_cache_ptr + shmem_size, FEATURE_BLOCK_SIZE, INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE };

        // create a thread private array used for internal caching
        real_type temp[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE]{};

        // iterate over the remaining features using blocking to be able to cache them for faster memory accesses
        for (std::size_t dim = 0; dim < device_specific_num_rows_; dim += FEATURE_BLOCK_SIZE_sz) {
            // load data into shared memory
            for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                const auto global_i = i_linear + static_cast<std::size_t>(internal) * THREAD_BLOCK_SIZE_sz;
                const auto global_j = j_linear + static_cast<std::size_t>(internal) * THREAD_BLOCK_SIZE_sz;

                // FEATURE_BLOCK_SIZE = 2 * THREAD_BLOCK_SIZE -> store twice as many values in the shared memory
                A_cache(threadIdx_y, internal * THREAD_BLOCK_SIZE + threadIdx_x) = A_[(dim + threadIdx_y) * (num_rows_ - row_offset_ + PADDING_SIZE_sz) - (dim + threadIdx_y - std::size_t{ 1 }) * (dim + threadIdx_y) / std::size_t{ 2 } + device_specific_num_rows_ - (dim + threadIdx_y) + global_j];
                A_cache(threadIdx_y + THREAD_BLOCK_SIZE, internal * THREAD_BLOCK_SIZE + threadIdx_x) = A_[(dim + threadIdx_y + THREAD_BLOCK_SIZE_sz) * (num_rows_ - row_offset_ + PADDING_SIZE_sz) - (dim + threadIdx_y + THREAD_BLOCK_SIZE_sz - std::size_t{ 1 }) * (dim + threadIdx_y + THREAD_BLOCK_SIZE_sz) / std::size_t{ 2 } + device_specific_num_rows_ - (dim + threadIdx_y + THREAD_BLOCK_SIZE_sz) + global_j];
                B_cache(threadIdx_y, internal * THREAD_BLOCK_SIZE + threadIdx_x) = B_[(row_offset_ + dim + threadIdx_y) * (num_rhs_ + PADDING_SIZE_sz) + global_i];
                B_cache(threadIdx_y + THREAD_BLOCK_SIZE, internal * THREAD_BLOCK_SIZE + threadIdx_x) = B_[(row_offset_ + dim + threadIdx_y + THREAD_BLOCK_SIZE_sz) * (num_rhs_ + PADDING_SIZE_sz) + global_i];
            }
            team.team_barrier();  // wait until all threads loaded their part of the data

            // perform the feature reduction calculation
            for (unsigned block_dim = 0; block_dim < FEATURE_BLOCK_SIZE; ++block_dim) {
                for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                    for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                        temp[internal_i][internal_j] += A_cache(block_dim, threadIdx_y * INTERNAL_BLOCK_SIZE + internal_j) * B_cache(block_dim, threadIdx_x * INTERNAL_BLOCK_SIZE + internal_i);
                    }
                }
            }
            team.team_barrier();  // wait until all threads performed their part of the calculations
        }

        // apply the (remaining) BLAS operation and update C
        for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
            for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                const auto global_i = i + static_cast<std::size_t>(internal_i);
                const auto partial_global_j = j + static_cast<std::size_t>(internal_j);
                const auto global_j = row_offset_ + device_specific_num_rows_ + j + static_cast<std::size_t>(internal_j);

                // be sure to not perform out of bounds accesses
                if (global_i < num_rhs_ && partial_global_j < num_mirror_rows_) {
                    C_[global_j * (num_rhs_ + PADDING_SIZE_sz) + global_i] = alpha_ * temp[internal_i][internal_j] + beta_ * C_[global_j * (num_rhs_ + PADDING_SIZE_sz) + global_i];
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
 */
class device_kernel_inplace_matrix_add {
  public:
    /**
     * @brief Initialize the Kokkos kernel function object.
     * @param[in] num_cols the number of columns in both matrices
     * @param[in,out] lhs the first matrix (updated inplace)
     * @param[in] rhs the second matrix
     * @param[in] grid_x_offset the offset in x-dimension into the data points if more than one execution grid has to be used
     * @param[in] grid_y_offset the offset in y-dimension into the data points if more than one execution grid has to be used
     */
    device_kernel_inplace_matrix_add(const std::size_t num_cols, device_view_type<real_type> lhs, device_view_type<const real_type> rhs, const std::size_t grid_x_offset, const std::size_t grid_y_offset, const std::size_t grid_size_x) :
        num_cols_{ num_cols },
        lhs_{ lhs },
        rhs_{ rhs },
        grid_x_offset_{ grid_x_offset },
        grid_y_offset_{ grid_y_offset },
        grid_size_x_{ grid_size_x } { }

    KOKKOS_INLINE_FUNCTION
    void operator()(const Kokkos::TeamPolicy<>::member_type &team) const {
        // cast all values to 64-bit std::size_t to prevent potential 32-bit overflows
        const auto INTERNAL_BLOCK_SIZE_sz = static_cast<std::size_t>(INTERNAL_BLOCK_SIZE);
        const auto THREAD_BLOCK_SIZE_sz = static_cast<std::size_t>(THREAD_BLOCK_SIZE);
        const auto PADDING_SIZE_sz = static_cast<std::size_t>(PADDING_SIZE);
        const auto threadIdx_x = static_cast<std::size_t>(team.team_rank()) / THREAD_BLOCK_SIZE_sz;            // current thread in block x-dimension
        const auto threadIdx_y = static_cast<std::size_t>(team.team_rank()) % THREAD_BLOCK_SIZE_sz;            // current thread in block y-dimension
        const auto blockDim_x = THREAD_BLOCK_SIZE_sz;                                                          // number of threads in block x-dimension
        const auto blockDim_y = THREAD_BLOCK_SIZE_sz;                                                          // number of threads in block y-dimension
        const auto blockIdx_x = static_cast<std::size_t>(team.league_rank()) % grid_size_x_ + grid_x_offset_;  // current block in grid x-dimension + offsets if the grid size would be too large
        const auto blockIdx_y = static_cast<std::size_t>(team.league_rank()) / grid_size_x_ + grid_y_offset_;  // current block in grid y-dimension + offsets if the grid size would be too large

        // Calculate the indices used in the current thread
        const auto i = (blockIdx_x * blockDim_x + threadIdx_x) * INTERNAL_BLOCK_SIZE_sz;  // num_rows
        const auto j = (blockIdx_y * blockDim_y + threadIdx_y) * INTERNAL_BLOCK_SIZE_sz;  // num_rhs

        for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
            for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                const auto global_i = i + static_cast<std::size_t>(internal_i);
                const auto global_j = j + static_cast<std::size_t>(internal_j);

                lhs_[global_i * (num_cols_ + PADDING_SIZE_sz) + global_j] += rhs_[global_i * (num_cols_ + PADDING_SIZE_sz) + global_j];
            }
        }
    }

  private:
    /// @cond Doxygen_suppress
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
 */
class device_kernel_inplace_matrix_scale {
  public:
    /**
     * @brief Initialize the Kokkos kernel function object.
     * @param[in] num_cols the number of columns in the matrix
     * @param[in,out] lhs the first matrix (updated inplace)
     * @param[in] scale the value to scale
     * @param[in] grid_x_offset the offset in x-dimension into the data points if more than one execution grid has to be used
     * @param[in] grid_y_offset the offset in y-dimension into the data points if more than one execution grid has to be used
     */
    device_kernel_inplace_matrix_scale(const std::size_t num_cols, device_view_type<real_type> lhs, const real_type scale, const std::size_t grid_x_offset, const std::size_t grid_y_offset, const std::size_t grid_size_x) :
        num_cols_{ num_cols },
        lhs_{ lhs },
        scale_{ scale },
        grid_x_offset_{ grid_x_offset },
        grid_y_offset_{ grid_y_offset },
        grid_size_x_{ grid_size_x } { }

    KOKKOS_INLINE_FUNCTION
    void operator()(const Kokkos::TeamPolicy<>::member_type &team) const {
        // cast all values to 64-bit std::size_t to prevent potential 32-bit overflows
        const auto INTERNAL_BLOCK_SIZE_sz = static_cast<std::size_t>(INTERNAL_BLOCK_SIZE);
        const auto THREAD_BLOCK_SIZE_sz = static_cast<std::size_t>(THREAD_BLOCK_SIZE);
        const auto PADDING_SIZE_sz = static_cast<std::size_t>(PADDING_SIZE);
        const auto threadIdx_x = static_cast<std::size_t>(team.team_rank()) / THREAD_BLOCK_SIZE_sz;            // current thread in block x-dimension
        const auto threadIdx_y = static_cast<std::size_t>(team.team_rank()) % THREAD_BLOCK_SIZE_sz;            // current thread in block y-dimension
        const auto blockDim_x = THREAD_BLOCK_SIZE_sz;                                                          // number of threads in block x-dimension
        const auto blockDim_y = THREAD_BLOCK_SIZE_sz;                                                          // number of threads in block y-dimension
        const auto blockIdx_x = static_cast<std::size_t>(team.league_rank()) % grid_size_x_ + grid_x_offset_;  // current block in grid x-dimension + offsets if the grid size would be too large
        const auto blockIdx_y = static_cast<std::size_t>(team.league_rank()) / grid_size_x_ + grid_y_offset_;  // current block in grid y-dimension + offsets if the grid size would be too large

        // Calculate the indices used in the current thread
        const auto i = (blockIdx_x * blockDim_x + threadIdx_x) * INTERNAL_BLOCK_SIZE_sz;  // num_rows
        const auto j = (blockIdx_y * blockDim_y + threadIdx_y) * INTERNAL_BLOCK_SIZE_sz;  // num_rhs

        for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
            for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                const auto global_i = i + static_cast<std::size_t>(internal_i);
                const auto global_j = j + static_cast<std::size_t>(internal_j);

                lhs_[global_i * (num_cols_ + PADDING_SIZE_sz) + global_j] *= scale_;
            }
        }
    }

  private:
    /// @cond Doxygen_suppress
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
