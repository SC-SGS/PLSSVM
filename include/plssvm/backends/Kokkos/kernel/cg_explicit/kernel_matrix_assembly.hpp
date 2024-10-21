/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Functions for explicitly assembling the kernel matrix using the Kokkos backend.
 */

#ifndef PLSSVM_BACKENDS_KOKKOS_CG_EXPLICIT_KERNEL_MATRIX_ASSEMBLY_HPP_
#define PLSSVM_BACKENDS_KOKKOS_CG_EXPLICIT_KERNEL_MATRIX_ASSEMBLY_HPP_
#pragma once

#include "plssvm/backends/Kokkos/detail/device_ptr.hpp"             // TODO: view type aliases
#include "plssvm/backends/Kokkos/detail/standard_layout_tuple.hpp"  // plssvm::kokkos::detail::standard_layout_tuple
#include "plssvm/backends/Kokkos/kernel/kernel_functions.hpp"       // plssvm::kokkos::detail::{feature_reduce, apply_kernel_function}
#include "plssvm/constants.hpp"                                     // plssvm::{real_type, THREAD_BLOCK_SIZE, INTERNAL_BLOCK_SIZE, FEATURE_BLOCK_SIZE, PADDING_SIZE}
#include "plssvm/kernel_function_types.hpp"                         // plssvm::kernel_function_type

#include "Kokkos_Core.hpp"  // TODO:

#include <cstddef>  // std::size_t

namespace plssvm::kokkos::detail {

/**
 * @brief Create the explicit kernel matrix using the @p kernel_function.
 * @tparam kernel_function the type of the used kernel function
 * @tparam Args the types of the parameters necessary for the specific kernel function; stored in a `standard_layout_tuple`
 */
template <kernel_function_type kernel_function, typename... Args>
class device_kernel_assembly {
  public:
    /**
     * @brief Initialize the SYCL kernel function object.
     * @param[in] cgh the SYCL handler used to allocate the local memory
     * @param[out] kernel_matrix_d the calculated kernel matrix
     * @param[in] data_d the data points to calculate the kernel matrix from
     * @param[in] num_rows the number of data points
     * @param[in] device_num_rows the number of rows the current device is responsible for
     * @param[in] row_offset the first row in @p data_d the current device is responsible for
     * @param[in] num_features the number of features per data point
     * @param[in] q the vector used in the dimensional reduction
     * @param[in] QA_cost the scalar used in the dimensional reduction
     * @param[in] cost the cost factor the diagonal is scaled with
     * @param[in] grid_x_offset the offset in x-dimension into the data points if more than one execution grid has to be used
     * @param[in] grid_y_offset the offset in y-dimension into the data points if more than one execution grid has to be used
     * @param[in] kernel_function_parameter the parameters necessary to apply the @p kernel_function
     */
    device_kernel_assembly(device_view_type<real_type> kernel_matrix_d, device_view_type<real_type> data_d, const std::size_t num_rows, const std::size_t device_num_rows, const std::size_t row_offset, const std::size_t num_features, device_view_type<real_type> q, const real_type QA_cost, const real_type cost, const std::size_t grid_x_offset, const std::size_t grid_y_offset, const std::size_t grid_size_x, Args... kernel_function_parameter) :
        kernel_matrix_d_{ kernel_matrix_d },
        data_d_{ data_d },
        num_rows_{ num_rows },
        device_num_rows_{ device_num_rows },
        row_offset_{ row_offset },
        num_features_{ num_features },
        q_{ q },
        QA_cost_{ QA_cost },
        cost_{ cost },
        grid_x_offset_{ grid_x_offset },
        grid_y_offset_{ grid_y_offset },
        grid_size_x_{ grid_size_x },
        kernel_function_parameter_{ detail::make_standard_layout_tuple(std::forward<Args>(kernel_function_parameter)...) } {
    }

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
        real_type *data_cache_ptr = static_cast<real_type *>(team.team_shmem().get_shmem(2 * shmem_size));
        Kokkos::mdspan<real_type, Kokkos::dextents<std::size_t, 2>> data_cache_i{ data_cache_ptr, FEATURE_BLOCK_SIZE, INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE };
        Kokkos::mdspan<real_type, Kokkos::dextents<std::size_t, 2>> data_cache_j{ data_cache_ptr + shmem_size, FEATURE_BLOCK_SIZE, INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE };

        // only calculate the upper triangular matrix -> can't use threadIdx since all threads in a warp must progress further
        if (blockIdx_x >= blockIdx_y) {
            // create a thread private array used for internal caching
            real_type temp[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE]{};

            // iterate over all features using blocking to be able to cache them for faster memory accesses
            for (std::size_t dim = 0; dim < num_features_; dim += FEATURE_BLOCK_SIZE_sz) {
                // load data into shared memory
                for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                    const auto global_i = row_offset_ + i_linear + static_cast<std::size_t>(internal) * THREAD_BLOCK_SIZE_sz;
                    const auto global_j = row_offset_ + j_linear + static_cast<std::size_t>(internal) * THREAD_BLOCK_SIZE_sz;

                    // FEATURE_BLOCK_SIZE = 2 * THREAD_BLOCK_SIZE -> store twice as many values in the shared memory
                    data_cache_i(threadIdx_y, internal * THREAD_BLOCK_SIZE + threadIdx_x) = data_d_[(dim + threadIdx_y) * (num_rows_ + 1ull + PADDING_SIZE_sz) + global_i];
                    data_cache_i(threadIdx_y + THREAD_BLOCK_SIZE, internal * THREAD_BLOCK_SIZE + threadIdx_x) = data_d_[(dim + threadIdx_y + THREAD_BLOCK_SIZE_sz) * (num_rows_ + 1ull + PADDING_SIZE_sz) + global_i];
                    data_cache_j(threadIdx_y, internal * THREAD_BLOCK_SIZE + threadIdx_x) = data_d_[(dim + threadIdx_y) * (num_rows_ + 1ull + PADDING_SIZE_sz) + global_j];
                    data_cache_j(threadIdx_y + THREAD_BLOCK_SIZE, internal * THREAD_BLOCK_SIZE + threadIdx_x) = data_d_[(dim + threadIdx_y + THREAD_BLOCK_SIZE_sz) * (num_rows_ + 1ull + PADDING_SIZE_sz) + global_j];
                }
                team.team_barrier();  // wait until all threads loaded their part of the data

                // perform the feature reduction calculation
                for (unsigned block_dim = 0; block_dim < FEATURE_BLOCK_SIZE; ++block_dim) {
                    for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                        for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                            temp[internal_i][internal_j] += detail::feature_reduce<kernel_function>(data_cache_i(block_dim, threadIdx_x * INTERNAL_BLOCK_SIZE + internal_i),
                                                                                                    data_cache_j(block_dim, threadIdx_y * INTERNAL_BLOCK_SIZE + internal_j));
                        }
                    }
                }
                team.team_barrier();  // wait until all threads performed their part of the calculations
            }

            // apply the remaining part of the kernel function and store the value in the output kernel matrix
            for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                    // calculate the indices to access the kernel matrix (the part stored on the current device)
                    const auto device_global_i = i + static_cast<std::size_t>(internal_i);
                    const auto global_i = row_offset_ + i + static_cast<std::size_t>(internal_i);
                    const auto device_global_j = j + static_cast<std::size_t>(internal_j);
                    const auto global_j = row_offset_ + j + static_cast<std::size_t>(internal_j);

                    // be sure to not perform out of bounds accesses for the kernel matrix (only using the upper triangular matrix)
                    if (device_global_i < (num_rows_ - row_offset_) && device_global_j < device_num_rows_ && global_i >= global_j) {
                        real_type temp_ij = temp[internal_i][internal_j];
                        temp_ij = detail::apply_kernel_function<kernel_function>(temp_ij, kernel_function_parameter_) + QA_cost_ - q_[global_i] - q_[global_j];
                        // apply the cost on the diagonal
                        if (global_i == global_j) {
                            temp_ij += cost_;
                        }
                        // update the kernel matrix
                        kernel_matrix_d_[device_global_j * (num_rows_ - row_offset_ + PADDING_SIZE_sz) - device_global_j * (device_global_j + 1ull) / 2ull + device_global_i] = temp_ij;
                    }
                }
            }
        }
    }

  private:
    /// @cond Doxygen_suppress
    device_view_type<real_type> kernel_matrix_d_;
    device_view_type<const real_type> data_d_;
    const std::size_t num_rows_;
    const std::size_t device_num_rows_;
    const std::size_t row_offset_;
    const std::size_t num_features_;
    device_view_type<const real_type> q_;
    const real_type QA_cost_;
    const real_type cost_;
    const std::size_t grid_x_offset_;
    const std::size_t grid_y_offset_;
    const std::size_t grid_size_x_;
    const detail::standard_layout_tuple<Args...> kernel_function_parameter_;
    /// @endcond
};

}  // namespace plssvm::kokkos::detail

#endif  // PLSSVM_BACKENDS_KOKKOS_CG_EXPLICIT_KERNEL_MATRIX_ASSEMBLY_HPP_
