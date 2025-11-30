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

#ifndef PLSSVM_BACKENDS_KOKKOS_KERNEL_CG_EXPLICIT_KERNEL_MATRIX_ASSEMBLY_HPP_
#define PLSSVM_BACKENDS_KOKKOS_KERNEL_CG_EXPLICIT_KERNEL_MATRIX_ASSEMBLY_HPP_
#pragma once

#include "plssvm/backends/Kokkos/detail/standard_layout_tuple.hpp"  // plssvm::kokkos::detail::standard_layout_tuple
#include "plssvm/backends/Kokkos/kernel/kernel_functions.hpp"       // plssvm::kokkos::detail::{feature_reduce, apply_kernel_function}
#include "plssvm/constants.hpp"                                     // plssvm::{real_type, THREAD_BLOCK_SIZE, INTERNAL_BLOCK_SIZE, PADDING_SIZE}
#include "plssvm/kernel_function_types.hpp"                         // plssvm::kernel_function_type
#include "plssvm/target_platforms.hpp"                              // plssvm::target_platform

#include "Kokkos_Core.hpp"  // KOKKOS_INLINE_FUNCTION, Kokkos::View, Kokkos::TeamPolicy, Kokkos::mdspan, Kokkos::dextents

#include <cstddef>  // std::size_t

namespace plssvm::kokkos::detail {

/**
 * @brief Create the explicit kernel matrix using the @p kernel_function.
 * @tparam ExecutionSpace the Kokkos::ExecutionSpace used to execute the kernel
 * @tparam target the target platform
 * @tparam kernel_function the type of the used kernel function
 * @tparam Args the types of the parameters necessary for the specific kernel function; stored in a `standard_layout_tuple`
 */
template <typename ExecutionSpace, target_platform target, kernel_function_type kernel_function, typename... Args>
class device_kernel_assembly {
    /**
     * @brief The type of the used Kokkos::View.
     */
    template <typename T>
    using device_view_type = Kokkos::View<T *, ExecutionSpace>;

  public:
    /**
     * @brief Initialize the Kokkos kernel function object.
     * @param[out] kernel_matrix the calculated kernel matrix
     * @param[in] data the data points to calculate the kernel matrix from
     * @param[in] num_rows the number of data points
     * @param[in] device_num_rows the number of rows the current device is responsible for
     * @param[in] device_row_offset the first row in @p data_d the current device is responsible for
     * @param[in] num_features the number of features per data point
     * @param[in] q the vector used in the dimensional reduction
     * @param[in] QA_cost the scalar used in the dimensional reduction
     * @param[in] cost the cost factor the diagonal is scaled with
     * @param[in] grid_x_offset the offset in x-dimension into the data points if more than one execution grid has to be used
     * @param[in] grid_y_offset the offset in y-dimension into the data points if more than one execution grid has to be used
     * @param[in] grid_size_x the size of the execution grid in x-dimension
     * @param[in] kernel_function_parameter the parameters necessary to apply the @p kernel_function
     */
    device_kernel_assembly(device_view_type<real_type> kernel_matrix, device_view_type<real_type> data, const std::size_t num_rows, const std::size_t device_num_rows, const std::size_t device_row_offset, const std::size_t num_features, device_view_type<real_type> q, const real_type QA_cost, const real_type cost, const std::size_t grid_x_offset, const std::size_t grid_y_offset, const std::size_t grid_size_x, Args... kernel_function_parameter) :
        kernel_matrix_{ kernel_matrix },
        data_{ data },
        num_rows_{ num_rows },
        device_num_rows_{ device_num_rows },
        device_row_offset_{ device_row_offset },
        num_features_{ num_features },
        q_{ q },
        QA_cost_{ QA_cost },
        cost_{ cost },
        grid_x_offset_{ grid_x_offset },
        grid_y_offset_{ grid_y_offset },
        grid_size_x_{ grid_size_x },
        kernel_function_parameter_{ detail::make_standard_layout_tuple(std::forward<Args>(kernel_function_parameter)...) } {
    }

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
        constexpr auto PADDING_SIZE_uz = static_cast<std::size_t>(PADDING_SIZE);

        const auto threadIdx_x = static_cast<std::size_t>(team.team_rank()) / THREAD_BLOCK_SIZE_uz;            // current thread in team x-dimension
        const auto threadIdx_y = static_cast<std::size_t>(team.team_rank()) % THREAD_BLOCK_SIZE_uz;            // current thread in team y-dimension
        const auto blockDim_x = THREAD_BLOCK_SIZE_uz;                                                          // number of threads in team x-dimension
        const auto blockDim_y = THREAD_BLOCK_SIZE_uz;                                                          // number of threads in team y-dimension
        const auto blockIdx_x = static_cast<std::size_t>(team.league_rank()) % grid_size_x_ + grid_x_offset_;  // current team in league x-dimension + offsets if the league size is too large
        const auto blockIdx_y = static_cast<std::size_t>(team.league_rank()) / grid_size_x_ + grid_y_offset_;  // current team in league y-dimension + offsets if the league size is too large

        // create two scratchpad memory arrays used for caching
        constexpr std::size_t scratchpad_size = THREAD_BLOCK_SIZE_uz * THREAD_BLOCK_SIZE_uz * INTERNAL_BLOCK_SIZE_uz;
        auto *scratchpad_ptr = static_cast<real_type *>(team.team_shmem().get_shmem(std::size_t{ 2 } * scratchpad_size * sizeof(real_type)));
        Kokkos::mdspan<real_type, Kokkos::dextents<std::size_t, 2>> data_i_cache{ scratchpad_ptr, THREAD_BLOCK_SIZE_uz, INTERNAL_BLOCK_SIZE_uz * THREAD_BLOCK_SIZE_uz };
        Kokkos::mdspan<real_type, Kokkos::dextents<std::size_t, 2>> data_j_cache{ scratchpad_ptr + scratchpad_size, THREAD_BLOCK_SIZE_uz, INTERNAL_BLOCK_SIZE_uz * THREAD_BLOCK_SIZE_uz };

        // only calculate the upper triangular matrix -> can't use team.team_rank() since all threads in a team must progress further
        if (blockIdx_x >= blockIdx_y) {
            // create a thread private array used for internal caching
            real_type temp[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE]{};

            {
                // calculate the indices used in the current thread, pays attention to coalesced memory accesses
                const auto i_idx_linear = blockIdx_x * blockDim_x * INTERNAL_BLOCK_SIZE_uz + threadIdx_x;  // num_rows - device_row_offset
                const auto j_idx_linear = blockIdx_y * blockDim_y * INTERNAL_BLOCK_SIZE_uz + threadIdx_x;  // device_num_rows

                // iterate over all features using blocking to be able to cache them for faster memory accesses
                for (std::size_t feature_block = 0; feature_block < num_features_; feature_block += THREAD_BLOCK_SIZE_uz) {
                    // load data into scratchpad memory
                    for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                        // calculate the indices to access the global data, pays attention to coalesced memory accesses
                        const auto global_i_idx_linear = device_row_offset_ + i_idx_linear + static_cast<std::size_t>(internal) * THREAD_BLOCK_SIZE_uz;
                        const auto global_j_idx_linear = device_row_offset_ + j_idx_linear + static_cast<std::size_t>(internal) * THREAD_BLOCK_SIZE_uz;

                        // store the values in the scratchpad memory
                        data_i_cache(team_rank_y, internal * THREAD_BLOCK_SIZE + team_rank_x) = data_[(feature_block + threadIdx_y) * (num_rows_ + std::size_t{ 1 } + PADDING_SIZE_uz) + global_i_idx_linear];  // SoA
                        data_j_cache(team_rank_y, internal * THREAD_BLOCK_SIZE + team_rank_x) = data_[(feature_block + threadIdx_y) * (num_rows_ + std::size_t{ 1 } + PADDING_SIZE_uz) + global_j_idx_linear];  // SoA
                    }
                    team.team_barrier();  // wait until all threads loaded their part of the data

                    if constexpr (target == target_platform::cpu) {
                        // perform the feature reduction calculation, the feature is the fastest moving index
                        for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                            for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                                real_type sum{ 0.0 };
                                for (unsigned feature = 0; feature < THREAD_BLOCK_SIZE; ++feature) {
                                    sum += detail::feature_reduce<kernel_function>(data_i_cache(feature, team_rank_x * INTERNAL_BLOCK_SIZE + internal_i),
                                                                                   data_j_cache(feature, team_rank_y * INTERNAL_BLOCK_SIZE + internal_j));
                                }
                                temp[internal_i][internal_j] += sum;
                            }
                        }
                    } else {
                        // perform the feature reduction calculation, the feature is the slowest moving index
                        for (unsigned feature = 0; feature < THREAD_BLOCK_SIZE; ++feature) {
                            for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                                for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                                    temp[internal_i][internal_j] += detail::feature_reduce<kernel_function>(data_i_cache(feature, team_rank_x * INTERNAL_BLOCK_SIZE + internal_i),
                                                                                                            data_j_cache(feature, team_rank_y * INTERNAL_BLOCK_SIZE + internal_j));
                                }
                            }
                        }
                    }
                    team.team_barrier();  // wait until all threads performed their part of the calculations
                }
            }

            // calculate the indices used in the current thread
            const auto i_idx = (blockIdx_x * blockDim_x + threadIdx_x) * INTERNAL_BLOCK_SIZE_uz;  // num_rows - device_row_offset
            const auto j_idx = (blockIdx_y * blockDim_y + threadIdx_y) * INTERNAL_BLOCK_SIZE_uz;  // device_num_rows

            // apply the remaining part of the kernel function and store the value in the output kernel matrix
            for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                    // calculate the indices to access the global data and the data with respect to the current device
                    const auto device_global_i_idx = i_idx + static_cast<std::size_t>(internal_i);
                    const auto global_i_idx = device_row_offset_ + device_global_i_idx;
                    const auto device_global_j_idx = j_idx + static_cast<std::size_t>(internal_j);
                    const auto global_j_idx = device_row_offset_ + device_global_j_idx;

                    // be sure to not perform out-of-bounds accesses (only using the upper triangular matrix)
                    if (device_global_i_idx < (num_rows_ - device_row_offset_) && device_global_j_idx < device_num_rows_ && global_i_idx >= global_j_idx) {
                        real_type temp_ij = temp[internal_i][internal_j];
                        // apply the final kernel function
                        temp_ij = detail::apply_kernel_function<kernel_function>(temp_ij, kernel_function_parameter_) + QA_cost_ - q_[global_i_idx] - q_[global_j_idx];
                        // apply the cost on the diagonal
                        if (global_i_idx == global_j_idx) {
                            temp_ij += cost_;
                        }
                        // update the upper triangular kernel matrix
                        kernel_matrix_[device_global_j_idx * (num_rows_ - device_row_offset_ + PADDING_SIZE_uz) - device_global_j_idx * (device_global_j_idx + std::size_t{ 1 }) / std::size_t{ 2 } + device_global_i_idx] = temp_ij;
                    }
                }
            }
        }
    }

  private:
    /// @cond Doxygen_suppress
    device_view_type<real_type> kernel_matrix_;
    device_view_type<const real_type> data_;
    const std::size_t num_rows_;
    const std::size_t device_num_rows_;
    const std::size_t device_row_offset_;
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

#endif  // PLSSVM_BACKENDS_KOKKOS_KERNEL_CG_EXPLICIT_KERNEL_MATRIX_ASSEMBLY_HPP_
