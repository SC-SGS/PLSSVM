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

#include "plssvm/backends/SYCL/data_parallel_kernels.hpp"    // plssvm::sycl::data_parallel_kernel
#include "plssvm/constants.hpp"                              // plssvm::{real_type, THREAD_BLOCK_SIZE, INTERNAL_BLOCK_SIZE, PADDING_SIZE}
#include "plssvm/target_platforms.hpp"                       // plssvm::target_platform

#include "sycl/sycl.hpp"  // sycl::memory_environment, sycl::require_local_mem, sycl::require_private_mem, sycl::distribute_items_and_wait, sycl::s_item

#include <cstddef>  // std::size_t

namespace plssvm::sycl::detail::scoped {

/**
 * @brief Perform an explicit BLAS SYMM operation: `C = alpha * A * B + beta * C` where @p A is a `m x k` symmetric matrix (memory optimized), @p B is a `k x n` matrix, @p C is a `m x n` matrix, and @p alpha and @p beta are scalars.
 * @details Uses AdaptiveCpp's scoped parallelism.
 * @tparam target the target platform
 */
template <target_platform target>
class device_kernel_symm {
  public:
    /// The used SYCL data parallel kernel.
    constexpr static sycl::data_parallel_kernel data_parallel_kernel_type = sycl::data_parallel_kernel::scoped;

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
     * @tparam T the implementation defined type of the group to iterate
     * @param[in] group group representing the current point in the execution space
     */
    template <typename T>
    void operator()(T group) const {
        ::sycl::memory_environment(group,
                                   // the indices used in the current work-item
                                   ::sycl::require_local_mem<real_type[THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE]>(),  // A_cache
                                   ::sycl::require_local_mem<real_type[THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE]>(),  // B_cache

                                   // create two local memory arrays used for caching
                                   ::sycl::require_private_mem<std::array<std::array<real_type, INTERNAL_BLOCK_SIZE>, INTERNAL_BLOCK_SIZE>>({}),
                                   [&](auto &A_cache, auto &B_cache, auto &temp) {
                                       // iterate over all values using blocking to be able to cache them for faster memory accesses
                                       for (std::size_t dim_block = 0; dim_block < (num_rows_ - device_row_offset_); dim_block += static_cast<std::size_t>(THREAD_BLOCK_SIZE)) {
                                           // load data into local memory
                                           ::sycl::distribute_items_and_wait(group, [&](::sycl::s_item<2> idx) {
                                               // cast values to 32-bit unsigned int values to prevent implicit conversions
                                               const auto local_id_0 = static_cast<unsigned>(idx.get_local_id(group, 0));
                                               const auto local_id_1 = static_cast<unsigned>(idx.get_local_id(group, 1));

                                               // cast all values to 64-bit std::size_t to prevent potential 32-bit overflows
                                               constexpr auto THREAD_BLOCK_SIZE_uz = static_cast<std::size_t>(THREAD_BLOCK_SIZE);
                                               constexpr auto INTERNAL_BLOCK_SIZE_uz = static_cast<std::size_t>(INTERNAL_BLOCK_SIZE);
                                               constexpr auto PADDING_SIZE_uz = static_cast<std::size_t>(PADDING_SIZE);

                                               const auto threadIdx_x = static_cast<std::size_t>(idx.get_local_id(group, 0));       // current work-item in work-group x-dimension
                                               const auto threadIdx_y = static_cast<std::size_t>(idx.get_local_id(group, 1));       // current work-item in work-group y-dimension
                                               const auto blockDim_x = static_cast<std::size_t>(group.get_logical_local_range(0));  // number of work-items in work-group x-dimension
                                               const auto blockDim_y = static_cast<std::size_t>(group.get_logical_local_range(1));  // number of work-items in work-group y-dimension
                                               const auto blockIdx_x = static_cast<std::size_t>(group[0]) + grid_x_offset_;         // current work-group in global range x-dimension + offsets if the global range is too large
                                               const auto blockIdx_y = static_cast<std::size_t>(group[1]) + grid_y_offset_;         // current work-group in global range y-dimension + offsets if the global range is too large

                                               // calculate the indices to access the global data, pays attention to coalesced memory accesses
                                               const auto i_idx_linear = blockIdx_y * blockDim_y * INTERNAL_BLOCK_SIZE_uz + threadIdx_y;  // num_rhs
                                               const auto j_idx_linear = blockIdx_x * blockDim_x * INTERNAL_BLOCK_SIZE_uz + threadIdx_y;  // device_num_rows

                                               for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                                                   // calculate the indices to access the global data, pays attention to coalesced memory accesses
                                                   const auto global_i_idx_linear = i_idx_linear + static_cast<std::size_t>(internal) * THREAD_BLOCK_SIZE_uz;
                                                   const auto global_j_idx_linear = j_idx_linear + static_cast<std::size_t>(internal) * THREAD_BLOCK_SIZE_uz;

                                                   // store the values in the local memory
                                                   // determine on which side of the diagonal we are located
                                                   if (dim_block + threadIdx_x < global_j_idx_linear) {
                                                       A_cache[local_id_0][internal * THREAD_BLOCK_SIZE + local_id_1] = A_[(dim_block + threadIdx_x) * (num_rows_ - device_row_offset_ + PADDING_SIZE_uz) + global_j_idx_linear - (dim_block + threadIdx_x) * (dim_block + threadIdx_x + std::size_t{ 1 }) / std::size_t{ 2 }];  // SoA, upper triangular matrix only
                                                   } else {
                                                       A_cache[local_id_0][internal * THREAD_BLOCK_SIZE + local_id_1] = A_[global_j_idx_linear * (num_rows_ - device_row_offset_ + PADDING_SIZE_uz) + dim_block + threadIdx_x - global_j_idx_linear * (global_j_idx_linear + std::size_t{ 1 }) / std::size_t{ 2 }];  // SoA, upper triangular matrix only
                                                   }

                                                   B_cache[local_id_0][internal * THREAD_BLOCK_SIZE + local_id_1] = B_[(dim_block + device_row_offset_ + threadIdx_x) * (num_rhs_ + PADDING_SIZE_uz) + global_i_idx_linear];  // SoA
                                               }
                                           });

                                           // perform the dot product calculation
                                           ::sycl::distribute_items_and_wait(group, [&](::sycl::s_item<2> idx) {
                                               // cast values to 32-bit unsigned int values to prevent implicit conversions
                                               const auto local_id_0 = static_cast<unsigned>(idx.get_local_id(group, 0));
                                               const auto local_id_1 = static_cast<unsigned>(idx.get_local_id(group, 1));

                                               if constexpr (target == target_platform::cpu) {
                                                   // perform the dot product calculation, the dim is the fastest moving index
                                                   for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                                                       for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                                                           real_type sum{ 0.0 };
                                                           for (unsigned dim = 0; dim < THREAD_BLOCK_SIZE; ++dim) {
                                                               sum += A_cache[dim][local_id_0 * INTERNAL_BLOCK_SIZE + internal_j] * B_cache[dim][local_id_1 * INTERNAL_BLOCK_SIZE + internal_i];
                                                           }
                                                           temp(idx)[internal_i][internal_j] += sum;
                                                       }
                                                   }
                                               } else {
                                                   // perform the dot product calculation, the dim is the slowest moving index
                                                   for (unsigned dim = 0; dim < THREAD_BLOCK_SIZE; ++dim) {
                                                       for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                                                           for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                                                               temp(idx)[internal_i][internal_j] += A_cache[dim][local_id_0 * INTERNAL_BLOCK_SIZE + internal_j] * B_cache[dim][local_id_1 * INTERNAL_BLOCK_SIZE + internal_i];
                                                           }
                                                       }
                                                   }
                                               }
                                           });
                                       }

                                       // apply the (partial) BLAS operation and update C
                                       ::sycl::distribute_items_and_wait(group, [&](::sycl::s_item<2> idx) {
                                           // cast all values to 64-bit std::size_t to prevent potential 32-bit overflows
                                           constexpr auto INTERNAL_BLOCK_SIZE_uz = static_cast<std::size_t>(INTERNAL_BLOCK_SIZE);
                                           constexpr auto PADDING_SIZE_uz = static_cast<std::size_t>(PADDING_SIZE);

                                           const auto threadIdx_x = static_cast<std::size_t>(idx.get_local_id(group, 0));       // current work-item in work-group x-dimension
                                           const auto threadIdx_y = static_cast<std::size_t>(idx.get_local_id(group, 1));       // current work-item in work-group y-dimension
                                           const auto blockDim_x = static_cast<std::size_t>(group.get_logical_local_range(0));  // number of work-items in work-group x-dimension
                                           const auto blockDim_y = static_cast<std::size_t>(group.get_logical_local_range(1));  // number of work-items in work-group y-dimension
                                           const auto blockIdx_x = static_cast<std::size_t>(group[0]) + grid_x_offset_;         // current work-group in global range x-dimension + offsets if the global range is too large
                                           const auto blockIdx_y = static_cast<std::size_t>(group[1]) + grid_y_offset_;         // current work-group in global range y-dimension + offsets if the global range is too large

                                           // calculate the indices to access the global data, pays attention to coalesced memory accesses
                                           const auto i_idx = (blockIdx_y * blockDim_y + threadIdx_y) * INTERNAL_BLOCK_SIZE_uz;  // num_rhs
                                           const auto j_idx = (blockIdx_x * blockDim_x + threadIdx_x) * INTERNAL_BLOCK_SIZE_uz;  // device_num_rows

                                           for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                                               for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                                                   // calculate the indices to access the global data and the data with respect to the current device
                                                   const auto global_i_idx = i_idx + static_cast<std::size_t>(internal_i);
                                                   const auto device_global_j_idx = j_idx + static_cast<std::size_t>(internal_j);
                                                   const auto global_j_idx = device_row_offset_ + device_global_j_idx;

                                                   // be sure to not perform out-of-bounds accesses
                                                   if (global_i_idx < num_rhs_ && device_global_j_idx < device_num_rows_) {
                                                       C_[global_j_idx * (num_rhs_ + PADDING_SIZE_uz) + global_i_idx] = alpha_ * temp(idx)[internal_i][internal_j] + beta_ * C_[global_j_idx * (num_rhs_ + PADDING_SIZE_uz) + global_i_idx];  // SoA
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
 *          Uses AdaptiveCpp's scoped parallelism.
 * @tparam target the target platform
 */
template <target_platform target>
class device_kernel_symm_mirror {
  public:
    /// The used SYCL data parallel kernel.
    constexpr static sycl::data_parallel_kernel data_parallel_kernel_type = sycl::data_parallel_kernel::scoped;

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
     * @tparam T the implementation defined type of the group to iterate
     * @param[in] group group representing the current point in the execution space
     */
    template <typename T>
    void operator()(T group) const {
        ::sycl::memory_environment(group,
                                   // the indices used in the current work-item
                                   ::sycl::require_local_mem<real_type[THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE]>(),  // A_cache
                                   ::sycl::require_local_mem<real_type[THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE]>(),  // B_cache

                                   // create a private memory array used for internal caching
                                   ::sycl::require_private_mem<std::array<std::array<real_type, INTERNAL_BLOCK_SIZE>, INTERNAL_BLOCK_SIZE>>({}),
                                   [&](auto &A_cache, auto &B_cache, auto &temp) {
                                       // iterate over the remaining values using blocking to be able to cache them for faster memory accesses
                                       for (std::size_t dim_block = 0; dim_block < device_num_rows_; dim_block += static_cast<std::size_t>(THREAD_BLOCK_SIZE)) {
                                           // load data into local memory
                                           ::sycl::distribute_items_and_wait(group, [&](::sycl::s_item<2> idx) {
                                               // cast values to 32-bit unsigned int values to prevent implicit conversions
                                               const auto local_id_0 = static_cast<unsigned>(idx.get_local_id(group, 0));
                                               const auto local_id_1 = static_cast<unsigned>(idx.get_local_id(group, 1));

                                               // cast all values to 64-bit std::size_t to prevent potential 32-bit overflows
                                               constexpr auto INTERNAL_BLOCK_SIZE_uz = static_cast<std::size_t>(INTERNAL_BLOCK_SIZE);
                                               constexpr auto THREAD_BLOCK_SIZE_uz = static_cast<std::size_t>(THREAD_BLOCK_SIZE);
                                               constexpr auto PADDING_SIZE_uz = static_cast<std::size_t>(PADDING_SIZE);

                                               const auto threadIdx_x = static_cast<std::size_t>(idx.get_local_id(group, 0));       // current work-item in work-group x-dimension
                                               const auto threadIdx_y = static_cast<std::size_t>(idx.get_local_id(group, 1));       // current work-item in work-group y-dimension
                                               const auto blockDim_x = static_cast<std::size_t>(group.get_logical_local_range(0));  // number of work-items in work-group x-dimension
                                               const auto blockDim_y = static_cast<std::size_t>(group.get_logical_local_range(1));  // number of work-items in work-group y-dimension
                                               const auto blockIdx_x = static_cast<std::size_t>(group[0]) + grid_x_offset_;         // current work-group in global range x-dimension + offsets if the global range is too large
                                               const auto blockIdx_y = static_cast<std::size_t>(group[1]) + grid_y_offset_;         // current work-group in global range y-dimension + offsets if the global range is too large

                                               // calculate the indices to access the global data, pays attention to coalesced memory accesses
                                               const auto i_idx_linear = blockIdx_y * blockDim_y * INTERNAL_BLOCK_SIZE_uz + threadIdx_y;
                                               const auto j_idx_linear = blockIdx_x * blockDim_x * INTERNAL_BLOCK_SIZE_uz + threadIdx_y;

                                               for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                                                   // calculate the indices to access the global data, pays attention to coalesced memory accesses
                                                   const auto global_i_idx_linear = i_idx_linear + static_cast<std::size_t>(internal) * THREAD_BLOCK_SIZE_uz;
                                                   const auto global_j_idx_linear = j_idx_linear + static_cast<std::size_t>(internal) * THREAD_BLOCK_SIZE_uz;

                                                   // store the values in the local memory
                                                   A_cache[local_id_0][internal * THREAD_BLOCK_SIZE + local_id_1] = A_[(dim_block + threadIdx_x) * (num_rows_ - device_row_offset_ + PADDING_SIZE_uz) - (dim_block + threadIdx_x - std::size_t{ 1 }) * (dim_block + threadIdx_x) / std::size_t{ 2 } + device_num_rows_ - (dim_block + threadIdx_x) + global_j_idx_linear];  // SoA, upper triangular matrix only
                                                   B_cache[local_id_0][internal * THREAD_BLOCK_SIZE + local_id_1] = B_[(device_row_offset_ + dim_block + threadIdx_x) * (num_rhs_ + PADDING_SIZE_uz) + global_i_idx_linear];                                                                                                                                                // SoA
                                               }
                                           });

                                           // perform the dot product calculation
                                           ::sycl::distribute_items_and_wait(group, [&](::sycl::s_item<2> idx) {
                                               // cast values to 32-bit unsigned int values to prevent implicit conversions
                                               const auto local_id_0 = static_cast<unsigned>(idx.get_local_id(group, 0));
                                               const auto local_id_1 = static_cast<unsigned>(idx.get_local_id(group, 1));

                                               if constexpr (target == target_platform::cpu) {
                                                   // perform the dot product calculation, the dim is the fastest moving index
                                                   for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                                                       for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                                                           real_type sum{ 0.0 };
                                                           for (unsigned dim = 0; dim < THREAD_BLOCK_SIZE; ++dim) {
                                                               sum += A_cache[dim][local_id_0 * INTERNAL_BLOCK_SIZE + internal_j] * B_cache[dim][local_id_1 * INTERNAL_BLOCK_SIZE + internal_i];
                                                           }
                                                           temp(idx)[internal_i][internal_j] += sum;
                                                       }
                                                   }
                                               } else {
                                                   // perform the dot product calculation, the dim is the slowest moving index
                                                   for (unsigned dim = 0; dim < THREAD_BLOCK_SIZE; ++dim) {
                                                       for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                                                           for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                                                               temp(idx)[internal_i][internal_j] += A_cache[dim][local_id_0 * INTERNAL_BLOCK_SIZE + internal_j] * B_cache[dim][local_id_1 * INTERNAL_BLOCK_SIZE + internal_i];
                                                           }
                                                       }
                                                   }
                                               }
                                           });
                                       }

                                       // apply the (remaining) BLAS operation and update C
                                       ::sycl::distribute_items_and_wait(group, [&](::sycl::s_item<2> idx) {
                                           // cast all values to 64-bit std::size_t to prevent potential 32-bit overflows
                                           constexpr auto INTERNAL_BLOCK_SIZE_uz = static_cast<std::size_t>(INTERNAL_BLOCK_SIZE);
                                           constexpr auto PADDING_SIZE_uz = static_cast<std::size_t>(PADDING_SIZE);

                                           const auto threadIdx_x = static_cast<std::size_t>(idx.get_local_id(group, 0));       // current work-item in work-group x-dimension
                                           const auto threadIdx_y = static_cast<std::size_t>(idx.get_local_id(group, 1));       // current work-item in work-group y-dimension
                                           const auto blockDim_x = static_cast<std::size_t>(group.get_logical_local_range(0));  // number of work-items in work-group x-dimension
                                           const auto blockDim_y = static_cast<std::size_t>(group.get_logical_local_range(1));  // number of work-items in work-group y-dimension
                                           const auto blockIdx_x = static_cast<std::size_t>(group[0]) + grid_x_offset_;         // current work-group in global range x-dimension + offsets if the global range is too large
                                           const auto blockIdx_y = static_cast<std::size_t>(group[1]) + grid_y_offset_;         // current work-group in global range y-dimension + offsets if the global range is too large

                                           // calculate the indices to access the global data
                                           const auto i_idx = (blockIdx_y * blockDim_y + threadIdx_y) * INTERNAL_BLOCK_SIZE_uz;
                                           const auto j_idx = (blockIdx_x * blockDim_x + threadIdx_x) * INTERNAL_BLOCK_SIZE_uz;

                                           for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                                               for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                                                   // calculate the indices to access the global data and the data with respect to the current device
                                                   const auto global_i_idx = i_idx + static_cast<std::size_t>(internal_i);
                                                   const auto partial_global_j_idx = j_idx + static_cast<std::size_t>(internal_j);
                                                   const auto global_j_idx = device_row_offset_ + device_num_rows_ + partial_global_j_idx;

                                                   // be sure to not perform out-of-bounds accesses
                                                   if (global_i_idx < num_rhs_ && partial_global_j_idx < num_mirror_rows_) {
                                                       C_[global_j_idx * (num_rhs_ + PADDING_SIZE_uz) + global_i_idx] = alpha_ * temp(idx)[internal_i][internal_j] + beta_ * C_[global_j_idx * (num_rhs_ + PADDING_SIZE_uz) + global_i_idx];  // SoA
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
 * @details Uses AdaptiveCpp's scoped parallelism.
 */
class device_kernel_inplace_matrix_add {
  public:
    /// The used SYCL data parallel kernel.
    constexpr static sycl::data_parallel_kernel data_parallel_kernel_type = sycl::data_parallel_kernel::scoped;

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
                                       ::sycl::distribute_items_and_wait(group, [&](::sycl::s_item<2> idx) {
                                           // cast all values to 64-bit std::size_t to prevent potential 32-bit overflows
                                           constexpr auto INTERNAL_BLOCK_SIZE_uz = static_cast<std::size_t>(INTERNAL_BLOCK_SIZE);
                                           constexpr auto PADDING_SIZE_uz = static_cast<std::size_t>(PADDING_SIZE);

                                           const auto threadIdx_x = static_cast<std::size_t>(idx.get_local_id(group, 0));       // current work-item in work-group x-dimension
                                           const auto threadIdx_y = static_cast<std::size_t>(idx.get_local_id(group, 1));       // current work-item in work-group y-dimension
                                           const auto blockDim_x = static_cast<std::size_t>(group.get_logical_local_range(0));  // number of work-items in work-group x-dimension
                                           const auto blockDim_y = static_cast<std::size_t>(group.get_logical_local_range(1));  // number of work-items in work-group y-dimension
                                           const auto blockIdx_x = static_cast<std::size_t>(group[0]) + grid_x_offset_;         // current work-group in global range x-dimension + offsets if the global range is too large
                                           const auto blockIdx_y = static_cast<std::size_t>(group[1]) + grid_y_offset_;         // current work-group in global range y-dimension + offsets if the global range is too large

                                           // calculate the indices used in the current work-item
                                           const auto i_idx = (blockIdx_y * blockDim_y + threadIdx_y) * INTERNAL_BLOCK_SIZE_uz;  // num_rows
                                           const auto j_idx = (blockIdx_x * blockDim_x + threadIdx_x) * INTERNAL_BLOCK_SIZE_uz;  // num_rhs

                                           for (std::size_t internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE_uz; ++internal_i) {
                                               for (std::size_t internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE_uz; ++internal_j) {
                                                   // calculate the indices to access the global data
                                                   const auto global_i_idx = i_idx + static_cast<std::size_t>(internal_i);
                                                   const auto global_j_idx = j_idx + static_cast<std::size_t>(internal_j);

                                                   lhs_[global_i_idx * (num_cols_ + PADDING_SIZE_uz) + global_j_idx] += rhs_[global_i_idx * (num_cols_ + PADDING_SIZE_uz) + global_j_idx];  // SoA
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
    /// The used SYCL data parallel kernel.
    constexpr static sycl::data_parallel_kernel data_parallel_kernel_type = sycl::data_parallel_kernel::scoped;

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
                                       ::sycl::distribute_items_and_wait(group, [&](::sycl::s_item<2> idx) {
                                           // cast all values to 64-bit std::size_t to prevent potential 32-bit overflows
                                           constexpr auto INTERNAL_BLOCK_SIZE_uz = static_cast<std::size_t>(INTERNAL_BLOCK_SIZE);
                                           constexpr auto PADDING_SIZE_uz = static_cast<std::size_t>(PADDING_SIZE);

                                           const auto threadIdx_x = static_cast<std::size_t>(idx.get_local_id(group, 0));       // current work-item in work-group x-dimension
                                           const auto threadIdx_y = static_cast<std::size_t>(idx.get_local_id(group, 1));       // current work-item in work-group y-dimension
                                           const auto blockDim_x = static_cast<std::size_t>(group.get_logical_local_range(0));  // number of work-items in work-group x-dimension
                                           const auto blockDim_y = static_cast<std::size_t>(group.get_logical_local_range(1));  // number of work-items in work-group y-dimension
                                           const auto blockIdx_x = static_cast<std::size_t>(group[0]) + grid_x_offset_;         // current work-group in global range x-dimension + offsets if the global range is too large
                                           const auto blockIdx_y = static_cast<std::size_t>(group[1]) + grid_y_offset_;         // current work-group in global range y-dimension + offsets if the global range is too large

                                           // calculate the indices used in the current work-item
                                           const auto i_idx = (blockIdx_y * blockDim_y + threadIdx_y) * INTERNAL_BLOCK_SIZE_uz;  // num_rows
                                           const auto j_idx = (blockIdx_x * blockDim_x + threadIdx_x) * INTERNAL_BLOCK_SIZE_uz;  // num_rhs

                                           for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                                               for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                                                   // calculate the indices to access the global data
                                                   const auto global_i_idx = i_idx + static_cast<std::size_t>(internal_i);
                                                   const auto global_j_idx = j_idx + static_cast<std::size_t>(internal_j);

                                                   lhs_[global_i_idx * (num_cols_ + PADDING_SIZE_uz) + global_j_idx] *= scale_;  // SoA
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
