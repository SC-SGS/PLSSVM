/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Functions for explicitly performing a BLAS GEMM like matrix-matrix multiplication using the SYCL backend and AdaptiveCpp's scoped parallelism.
 */

#ifndef PLSSVM_BACKENDS_SYCL_CG_EXPLICIT_SCOPED_BLAS_HPP_
#define PLSSVM_BACKENDS_SYCL_CG_EXPLICIT_SCOPED_BLAS_HPP_
#pragma once

#include "plssvm/constants.hpp"  // plssvm::{real_type, THREAD_BLOCK_SIZE, INTERNAL_BLOCK_SIZE, FEATURE_BLOCK_SIZE, PADDING_SIZE}

#include "sycl/sycl.hpp"  // sycl::memory_environment, sycl::require_local_mem, sycl::require_private_mem, sycl::distribute_items_and_wait, sycl::s_item

#include <cstddef>  // std::size_t

namespace plssvm::sycl::detail::scoped {

/**
 * @brief Perform an explicit BLAS SYMM operation: `C = alpha * A * B + beta * C` where @p A is a `m x k` symmetric matrix (memory optimized), @p B is a `k x n` matrix, @p C is a `m x n` matrix, and @p alpha and @p beta are scalars.
 * @details Uses AdaptiveCpp's scoped parallelism.
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
     * @tparam T the implementation defined type of the group to iterate
     * @param[in] group group representing the current point in the execution space
     */
    template <typename T>
    void operator()(T group) const {
        ::sycl::memory_environment(group,
                                   ::sycl::require_local_mem<real_type[FEATURE_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE]>(),
                                   ::sycl::require_local_mem<real_type[FEATURE_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE]>(),
                                   ::sycl::require_private_mem<std::size_t>(),
                                   ::sycl::require_private_mem<std::size_t>(),
                                   ::sycl::require_private_mem<std::size_t>(),
                                   ::sycl::require_private_mem<std::size_t>(),
                                   ::sycl::require_private_mem<std::array<std::array<real_type, INTERNAL_BLOCK_SIZE>, INTERNAL_BLOCK_SIZE>>({}),
                                   [&](auto &A_cache, auto &B_cache, auto &i, auto &i_linear, auto &j, auto &j_linear, auto &temp) {
                                       // initialize private and local variables
                                       ::sycl::distribute_items_and_wait(group, [&](::sycl::s_item<2> idx) {
                                           const std::size_t threadIdx_x = idx.get_local_id(group, 0);       // current thread in block x-dimension
                                           const std::size_t threadIdx_y = idx.get_local_id(group, 1);       // current thread in block y-dimension
                                           const std::size_t blockDim_x = group.get_logical_local_range(0);  // number of threads in block x-dimension
                                           const std::size_t blockDim_y = group.get_logical_local_range(1);  // number of threads in block y-dimension
                                           const std::size_t blockIdx_x = group[0] + grid_x_offset_;         // current block in grid x-dimension + offsets if the grid size would be too large
                                           const std::size_t blockIdx_y = group[1] + grid_y_offset_;         // current block in grid y-dimension + offsets if the grid size would be too large

                                           const auto INTERNAL_BLOCK_SIZE_uz = static_cast<std::size_t>(INTERNAL_BLOCK_SIZE);

                                           // indices
                                           i(idx) = (blockIdx_y * blockDim_y + threadIdx_y) * INTERNAL_BLOCK_SIZE_uz;
                                           i_linear(idx) = blockIdx_y * blockDim_y * INTERNAL_BLOCK_SIZE_uz + threadIdx_y;
                                           j(idx) = (blockIdx_x * blockDim_x + threadIdx_x) * INTERNAL_BLOCK_SIZE_uz;
                                           j_linear(idx) = blockIdx_x * blockDim_x * INTERNAL_BLOCK_SIZE_uz + threadIdx_y;
                                       });

                                       for (std::size_t dim = 0; dim < (num_rows_ - row_offset_); dim += static_cast<std::size_t>(FEATURE_BLOCK_SIZE)) {
                                           // load data into shared memory
                                           ::sycl::distribute_items_and_wait(group, [&](::sycl::s_item<2> idx) {
                                               const auto local_id_0 = static_cast<unsigned>(idx.get_local_id(group, 0));
                                               const auto local_id_1 = static_cast<unsigned>(idx.get_local_id(group, 1));

                                               const std::size_t threadIdx_x = idx.get_local_id(group, 0);

                                               const auto THREAD_BLOCK_SIZE_uz = static_cast<std::size_t>(THREAD_BLOCK_SIZE);
                                               const auto PADDING_SIZE_uz = static_cast<std::size_t>(PADDING_SIZE);

                                               for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                                                   const auto global_i = i_linear(idx) + static_cast<std::size_t>(internal) * THREAD_BLOCK_SIZE_uz;
                                                   const auto global_j = j_linear(idx) + static_cast<std::size_t>(internal) * THREAD_BLOCK_SIZE_uz;

                                                   // determine on which side of the diagonal we are located
                                                   if (dim + threadIdx_x < global_j) {
                                                       A_cache[local_id_0][internal * THREAD_BLOCK_SIZE + local_id_1] = A_[(dim + threadIdx_x) * (num_rows_ - row_offset_ + PADDING_SIZE_uz) + global_j - (dim + threadIdx_x) * (dim + threadIdx_x + std::size_t{ 1 }) / std::size_t{ 2 }];
                                                   } else {
                                                       A_cache[local_id_0][internal * THREAD_BLOCK_SIZE + local_id_1] = A_[global_j * (num_rows_ - row_offset_ + PADDING_SIZE_uz) + dim + threadIdx_x - global_j * (global_j + std::size_t{ 1 }) / std::size_t{ 2 }];
                                                   }
                                                   // determine on which side of the diagonal we are located
                                                   if (dim + threadIdx_x + THREAD_BLOCK_SIZE < global_j) {
                                                       A_cache[local_id_0 + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + local_id_1] = A_[(dim + threadIdx_x + THREAD_BLOCK_SIZE_uz) * (num_rows_ - row_offset_ + PADDING_SIZE_uz) + global_j - (dim + threadIdx_x + THREAD_BLOCK_SIZE_uz) * (dim + threadIdx_x + THREAD_BLOCK_SIZE_uz + std::size_t{ 1 }) / std::size_t{ 2 }];
                                                   } else {
                                                       A_cache[local_id_0 + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + local_id_1] = A_[global_j * (num_rows_ - row_offset_ + PADDING_SIZE_uz) + dim + threadIdx_x + THREAD_BLOCK_SIZE_uz - global_j * (global_j + std::size_t{ 1 }) / std::size_t{ 2 }];
                                                   }

                                                   B_cache[local_id_0][internal * THREAD_BLOCK_SIZE + local_id_1] = B_[(dim + row_offset_ + threadIdx_x) * (num_rhs_ + PADDING_SIZE_uz) + global_i];
                                                   B_cache[local_id_0 + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + local_id_1] = B_[(dim + row_offset_ + threadIdx_x + THREAD_BLOCK_SIZE_uz) * (num_rhs_ + PADDING_SIZE_uz) + global_i];
                                               }
                                           });

                                           // perform calculations
                                           ::sycl::distribute_items_and_wait(group, [&](::sycl::s_item<2> idx) {
                                               const auto local_id_0 = static_cast<unsigned>(idx.get_local_id(group, 0));
                                               const auto local_id_1 = static_cast<unsigned>(idx.get_local_id(group, 1));

                                               for (unsigned block_dim = 0; block_dim < FEATURE_BLOCK_SIZE; ++block_dim) {
                                                   for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                                                       for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                                                           temp(idx)[internal_i][internal_j] += A_cache[block_dim][local_id_0 * INTERNAL_BLOCK_SIZE + internal_j] * B_cache[block_dim][local_id_1 * INTERNAL_BLOCK_SIZE + internal_i];
                                                       }
                                                   }
                                               }
                                           });
                                       }

                                       ::sycl::distribute_items_and_wait(group, [&](::sycl::s_item<2> idx) {
                                           const auto PADDING_SIZE_uz = static_cast<std::size_t>(PADDING_SIZE);

                                           for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                                               for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                                                   const auto global_i = i(idx) + static_cast<std::size_t>(internal_i);
                                                   const auto device_global_j = j(idx) + static_cast<std::size_t>(internal_j);
                                                   const auto global_j = row_offset_ + j(idx) + static_cast<std::size_t>(internal_j);

                                                   // be sure to not perform out of bounds accesses
                                                   if (global_i < num_rhs_ && device_global_j < device_specific_num_rows_) {
                                                       C_[global_j * (num_rhs_ + PADDING_SIZE_uz) + global_i] = alpha_ * temp(idx)[internal_i][internal_j] + beta_ * C_[global_j * (num_rhs_ + PADDING_SIZE_uz) + global_i];
                                                   }
                                               }
                                           }
                                       });
                                   });
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
 *          Uses AdaptiveCpp's scoped parallelism.
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
     * @tparam T the implementation defined type of the group to iterate
     * @param[in] group group representing the current point in the execution space
     */
    template <typename T>
    void operator()(T group) const {
        ::sycl::memory_environment(group,
                                   ::sycl::require_local_mem<real_type[FEATURE_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE]>(),
                                   ::sycl::require_local_mem<real_type[FEATURE_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE]>(),
                                   ::sycl::require_private_mem<std::size_t>(),
                                   ::sycl::require_private_mem<std::size_t>(),
                                   ::sycl::require_private_mem<std::size_t>(),
                                   ::sycl::require_private_mem<std::size_t>(),
                                   ::sycl::require_private_mem<std::array<std::array<real_type, INTERNAL_BLOCK_SIZE>, INTERNAL_BLOCK_SIZE>>({}),
                                   [&](auto &A_cache, auto &B_cache, auto &i, auto &i_linear, auto &j, auto &j_linear, auto &temp) {
                                       // initialize private and local variables
                                       ::sycl::distribute_items_and_wait(group, [&](::sycl::s_item<2> idx) {
                                           const std::size_t threadIdx_x = idx.get_local_id(group, 0);       // current thread in block x-dimension
                                           const std::size_t threadIdx_y = idx.get_local_id(group, 1);       // current thread in block y-dimension
                                           const std::size_t blockDim_x = group.get_logical_local_range(0);  // number of threads in block x-dimension
                                           const std::size_t blockDim_y = group.get_logical_local_range(1);  // number of threads in block y-dimension
                                           const std::size_t blockIdx_x = group[0] + grid_x_offset_;         // current block in grid x-dimension + offsets if the grid size would be too large
                                           const std::size_t blockIdx_y = group[1] + grid_y_offset_;         // current block in grid y-dimension + offsets if the grid size would be too large

                                           const auto INTERNAL_BLOCK_SIZE_uz = static_cast<std::size_t>(INTERNAL_BLOCK_SIZE);

                                           // indices
                                           i(idx) = (blockIdx_y * blockDim_y + threadIdx_y) * INTERNAL_BLOCK_SIZE_uz;
                                           i_linear(idx) = blockIdx_y * blockDim_y * INTERNAL_BLOCK_SIZE_uz + threadIdx_y;
                                           j(idx) = (blockIdx_x * blockDim_x + threadIdx_x) * INTERNAL_BLOCK_SIZE_uz;
                                           j_linear(idx) = blockIdx_x * blockDim_x * INTERNAL_BLOCK_SIZE_uz + threadIdx_y;
                                       });

                                       for (std::size_t dim = 0; dim < device_specific_num_rows_; dim += static_cast<std::size_t>(FEATURE_BLOCK_SIZE)) {
                                           // load data into shared memory
                                           ::sycl::distribute_items_and_wait(group, [&](::sycl::s_item<2> idx) {
                                               const auto local_id_0 = static_cast<unsigned>(idx.get_local_id(group, 0));
                                               const auto local_id_1 = static_cast<unsigned>(idx.get_local_id(group, 1));

                                               const std::size_t threadIdx_x = idx.get_local_id(group, 0);

                                               const auto THREAD_BLOCK_SIZE_uz = static_cast<std::size_t>(THREAD_BLOCK_SIZE);
                                               const auto PADDING_SIZE_uz = static_cast<std::size_t>(PADDING_SIZE);

                                               for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                                                   const auto global_i = i_linear(idx) + static_cast<std::size_t>(internal) * THREAD_BLOCK_SIZE_uz;
                                                   const auto global_j = j_linear(idx) + static_cast<std::size_t>(internal) * THREAD_BLOCK_SIZE_uz;

                                                   // FEATURE_BLOCK_SIZE = 2 * THREAD_BLOCK_SIZE -> store twice as many values in the local memory
                                                   A_cache[local_id_0][internal * THREAD_BLOCK_SIZE + local_id_1] = A_[(dim + threadIdx_x) * (num_rows_ - row_offset_ + PADDING_SIZE_uz) - (dim + threadIdx_x - std::size_t{ 1 }) * (dim + threadIdx_x) / std::size_t{ 2 } + device_specific_num_rows_ - (dim + threadIdx_x) + global_j];
                                                   A_cache[local_id_0 + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + local_id_1] = A_[(dim + threadIdx_x + THREAD_BLOCK_SIZE_uz) * (num_rows_ - row_offset_ + PADDING_SIZE_uz) - (dim + threadIdx_x + THREAD_BLOCK_SIZE_uz - std::size_t{ 1 }) * (dim + threadIdx_x + THREAD_BLOCK_SIZE_uz) / std::size_t{ 2 } + device_specific_num_rows_ - (dim + threadIdx_x + THREAD_BLOCK_SIZE_uz) + global_j];

                                                   B_cache[local_id_0][internal * THREAD_BLOCK_SIZE + local_id_1] = B_[(dim + row_offset_ + threadIdx_x) * (num_rhs_ + PADDING_SIZE_uz) + global_i];
                                                   B_cache[local_id_0 + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + local_id_1] = B_[(dim + row_offset_ + threadIdx_x + THREAD_BLOCK_SIZE_uz) * (num_rhs_ + PADDING_SIZE_uz) + global_i];
                                               }
                                           });

                                           // perform calculations
                                           ::sycl::distribute_items_and_wait(group, [&](::sycl::s_item<2> idx) {
                                               const auto local_id_0 = static_cast<unsigned>(idx.get_local_id(group, 0));
                                               const auto local_id_1 = static_cast<unsigned>(idx.get_local_id(group, 1));

                                               for (unsigned block_dim = 0; block_dim < FEATURE_BLOCK_SIZE; ++block_dim) {
                                                   for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                                                       for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                                                           temp(idx)[internal_i][internal_j] += A_cache[block_dim][local_id_0 * INTERNAL_BLOCK_SIZE + internal_j] * B_cache[block_dim][local_id_1 * INTERNAL_BLOCK_SIZE + internal_i];
                                                       }
                                                   }
                                               }
                                           });
                                       }

                                       ::sycl::distribute_items_and_wait(group, [&](::sycl::s_item<2> idx) {
                                           const auto PADDING_SIZE_uz = static_cast<std::size_t>(PADDING_SIZE);

                                           for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                                               for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                                                   const auto global_i = i(idx) + static_cast<std::size_t>(internal_i);
                                                   const auto partial_global_j = j(idx) + static_cast<std::size_t>(internal_j);
                                                   const auto global_j = row_offset_ + device_specific_num_rows_ + j(idx) + static_cast<std::size_t>(internal_j);

                                                   // be sure to not perform out of bounds accesses
                                                   if (global_i < num_rhs_ && partial_global_j < num_mirror_rows_) {
                                                       C_[global_j * (num_rhs_ + PADDING_SIZE_uz) + global_i] = alpha_ * temp(idx)[internal_i][internal_j] + beta_ * C_[global_j * (num_rhs_ + PADDING_SIZE_uz) + global_i];
                                                   }
                                               }
                                           }
                                       });
                                   });
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
 * @details Uses AdaptiveCpp's scoped parallelism.
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
     * @tparam T the implementation defined type of the group to iterate
     * @param[in] group group representing the current point in the execution space
     */
    template <typename T>
    void operator()(T group) const {
        ::sycl::memory_environment(group,
                                   [&]() {
                                       // scale
                                       ::sycl::distribute_items_and_wait(group, [&](::sycl::s_item<2> idx) {
                                           // cast all values to 64-bit std::size_t to prevent potential 32-bit overflows
                                           const std::size_t threadIdx_x = idx.get_local_id(group, 0);
                                           const std::size_t threadIdx_y = idx.get_local_id(group, 1);
                                           const std::size_t blockDim_x = group.get_logical_local_range(0);
                                           const std::size_t blockDim_y = group.get_logical_local_range(1);
                                           const std::size_t blockIdx_x = group[0] + grid_x_offset_;
                                           const std::size_t blockIdx_y = group[1] + grid_y_offset_;
                                           const auto INTERNAL_BLOCK_SIZE_uz = static_cast<std::size_t>(INTERNAL_BLOCK_SIZE);
                                           const auto PADDING_SIZE_uz = static_cast<std::size_t>(PADDING_SIZE);

                                           // indices
                                           const std::size_t i = (blockIdx_y * blockDim_y + threadIdx_y) * INTERNAL_BLOCK_SIZE_uz;
                                           const std::size_t j = (blockIdx_x * blockDim_x + threadIdx_x) * INTERNAL_BLOCK_SIZE_uz;

                                           for (std::size_t internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE_uz; ++internal_i) {
                                               for (std::size_t internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE_uz; ++internal_j) {
                                                   const std::size_t global_i = i + internal_i;
                                                   const std::size_t global_j = j + internal_j;

                                                   lhs_[global_i * (num_cols_ + PADDING_SIZE_uz) + global_j] += rhs_[global_i * (num_cols_ + PADDING_SIZE_uz) + global_j];
                                               }
                                           }
                                       });
                                   });
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
 * @details Uses AdaptiveCpp's scoped parallelism.
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
     * @tparam T the implementation defined type of the group to iterate
     * @param[in] group group representing the current point in the execution space
     */
    template <typename T>
    void operator()(T group) const {
        ::sycl::memory_environment(group,
                                   [&]() {
                                       // scale
                                       ::sycl::distribute_items_and_wait(group, [&](::sycl::s_item<2> idx) {
                                           // cast all values to 64-bit std::size_t to prevent potential 32-bit overflows
                                           const std::size_t threadIdx_x = idx.get_local_id(group, 0);
                                           const std::size_t threadIdx_y = idx.get_local_id(group, 1);
                                           const std::size_t blockDim_x = group.get_logical_local_range(0);
                                           const std::size_t blockDim_y = group.get_logical_local_range(1);
                                           const std::size_t blockIdx_x = group[0] + grid_x_offset_;
                                           const std::size_t blockIdx_y = group[1] + grid_y_offset_;
                                           const auto INTERNAL_BLOCK_SIZE_uz = static_cast<std::size_t>(INTERNAL_BLOCK_SIZE);
                                           const auto PADDING_SIZE_uz = static_cast<std::size_t>(PADDING_SIZE);

                                           // indices
                                           const std::size_t i = (blockIdx_y * blockDim_y + threadIdx_y) * INTERNAL_BLOCK_SIZE_uz;
                                           const std::size_t j = (blockIdx_x * blockDim_x + threadIdx_x) * INTERNAL_BLOCK_SIZE_uz;

                                           for (std::size_t internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE_uz; ++internal_i) {
                                               for (std::size_t internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE_uz; ++internal_j) {
                                                   const std::size_t global_i = i + internal_i;
                                                   const std::size_t global_j = j + internal_j;

                                                   lhs_[global_i * (num_cols_ + PADDING_SIZE_uz) + global_j] *= scale_;
                                               }
                                           }
                                       });
                                   });
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

}  // namespace plssvm::sycl::detail::scoped

#endif  // PLSSVM_BACKENDS_SYCL_CG_EXPLICIT_SCOPED_BLAS_HPP_
