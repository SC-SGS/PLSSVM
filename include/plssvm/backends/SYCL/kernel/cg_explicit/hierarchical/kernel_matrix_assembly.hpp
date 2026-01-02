/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Functions for explicitly assembling the kernel matrix using the SYCL backend and the hierarchical data parallel kernels.
 */

#ifndef PLSSVM_BACKENDS_SYCL_KERNEL_CG_EXPLICIT_HIERARCHICAL_KERNEL_MATRIX_ASSEMBLY_HPP_
#define PLSSVM_BACKENDS_SYCL_KERNEL_CG_EXPLICIT_HIERARCHICAL_KERNEL_MATRIX_ASSEMBLY_HPP_
#pragma once

#include "plssvm/backends/SYCL/data_parallel_kernels.hpp"    // plssvm::sycl::data_parallel_kernel
#include "plssvm/backends/SYCL/kernel/kernel_functions.hpp"  // plssvm::sycl::detail::{feature_reduce, apply_kernel_function}
#include "plssvm/constants.hpp"                              // plssvm::{real_type, THREAD_BLOCK_SIZE, INTERNAL_BLOCK_SIZE, PADDING_SIZE}
#include "plssvm/kernel_function_types.hpp"                  // plssvm::kernel_function_type
#include "plssvm/target_platforms.hpp"                       // plssvm::target_platform

#include "sycl/sycl.hpp"  // sycl::group, sycl::private_memory, sycl::h_item

#include <array>    // std::array
#include <cstddef>  // std::size_t
#include <tuple>    // std::tuple, std::make_tuple

namespace plssvm::sycl::detail::hierarchical {

/**
 * @brief Create the explicit kernel matrix using the @p kernel_function.
 * @details Uses SYCL's hierarchical data parallel kernels.
 * @tparam target the target platform
 * @tparam kernel_function the type of the used kernel function
 * @tparam Args the types of the parameters necessary for the specific kernel function; stored in a `std::tuple`
 */
template <target_platform target, kernel_function_type kernel_function, typename... Args>
class device_kernel_assembly {
  public:
    /// The used SYCL data parallel kernel.
    constexpr static sycl::data_parallel_kernel data_parallel_kernel_type = sycl::data_parallel_kernel::hierarchical;

    /**
     * @brief Initialize the SYCL kernel function object.
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
     * @param[in] kernel_function_parameter the parameters necessary to apply the @p kernel_function
     */
    device_kernel_assembly(real_type *kernel_matrix, const real_type *data, const std::size_t num_rows, const std::size_t device_num_rows, const std::size_t device_row_offset, const std::size_t num_features, const real_type *q, const real_type QA_cost, const real_type cost, const std::size_t grid_x_offset, const std::size_t grid_y_offset, Args... kernel_function_parameter) :
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
        kernel_function_parameter_{ std::make_tuple(kernel_function_parameter...) } {
    }

    /**
     * @brief Function call operator overload performing the actual calculation.
     * @param[in] group indices representing the current point in the execution space
     */
    void operator()(::sycl::group<2> group) const {
        // create two local memory arrays used for caching
        std::array<std::array<real_type, static_cast<std::size_t>(INTERNAL_BLOCK_SIZE) * static_cast<std::size_t>(THREAD_BLOCK_SIZE)>, static_cast<std::size_t>(THREAD_BLOCK_SIZE)> data_i_cache{};
        std::array<std::array<real_type, static_cast<std::size_t>(INTERNAL_BLOCK_SIZE) * static_cast<std::size_t>(THREAD_BLOCK_SIZE)>, static_cast<std::size_t>(THREAD_BLOCK_SIZE)> data_j_cache{};

        // create a private memory array used for internal caching
        ::sycl::private_memory<std::array<std::array<real_type, static_cast<std::size_t>(INTERNAL_BLOCK_SIZE)>, static_cast<std::size_t>(INTERNAL_BLOCK_SIZE)>, 2> temp{ group };

        // only calculate the upper triangular matrix -> can't use get_local_id() since all work-items in a work-group must progress further
        if (group[1] + grid_y_offset_ >= group[0] + grid_x_offset_) {
            // initialize private temp matrix to zero
            group.parallel_for_work_item([&](::sycl::h_item<2> idx) {
                for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                    for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                        temp(idx)[internal_i][internal_j] = real_type{ 0.0 };
                    }
                }
            });

            // iterate over all features using blocking to be able to cache them for faster memory accesses
            for (std::size_t feature_block = 0; feature_block < num_features_; feature_block += static_cast<std::size_t>(THREAD_BLOCK_SIZE)) {
                // load data into local memory
                group.parallel_for_work_item([&](::sycl::h_item<2> idx) {
                    // cast values to 32-bit unsigned int values to prevent implicit conversions
                    const auto local_id_0 = static_cast<unsigned>(idx.get_local_id(0));
                    const auto local_id_1 = static_cast<unsigned>(idx.get_local_id(1));

                    // cast all values to 64-bit std::size_t to prevent potential 32-bit overflows
                    constexpr auto INTERNAL_BLOCK_SIZE_uz = static_cast<std::size_t>(INTERNAL_BLOCK_SIZE);
                    constexpr auto THREAD_BLOCK_SIZE_uz = static_cast<std::size_t>(THREAD_BLOCK_SIZE);
                    constexpr auto PADDING_SIZE_uz = static_cast<std::size_t>(PADDING_SIZE);

                    const auto threadIdx_x = static_cast<std::size_t>(idx.get_local_id(0));       // current work-item in work-group x-dimension
                    const auto threadIdx_y = static_cast<std::size_t>(idx.get_local_id(1));       // current work-item in work-group y-dimension
                    const auto blockDim_x = static_cast<std::size_t>(idx.get_local_range(0));     // number of work-items in work-group x-dimension
                    const auto blockDim_y = static_cast<std::size_t>(idx.get_local_range(1));     // number of work-items in work-group y-dimension
                    const auto blockIdx_x = static_cast<std::size_t>(group[0]) + grid_x_offset_;  // current work-group in global range x-dimension + offsets if the global range is too large
                    const auto blockIdx_y = static_cast<std::size_t>(group[1]) + grid_y_offset_;  // current work-group in global range y-dimension + offsets if the global range is too large

                    // calculate the indices used in the current work-item, pays attention to coalesced memory accesses
                    const auto i_idx_linear = blockIdx_y * blockDim_y * INTERNAL_BLOCK_SIZE_uz + threadIdx_y;  // num_rows - device_row_offset
                    const auto j_idx_linear = blockIdx_x * blockDim_x * INTERNAL_BLOCK_SIZE_uz + threadIdx_y;  // device_num_rows

                    for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                        // calculate the indices to access the global data, pays attention to coalesced memory accesses
                        const auto global_i_idx_linear = device_row_offset_ + i_idx_linear + static_cast<std::size_t>(internal) * THREAD_BLOCK_SIZE_uz;
                        const auto global_j_idx_linear = device_row_offset_ + j_idx_linear + static_cast<std::size_t>(internal) * THREAD_BLOCK_SIZE_uz;

                        // store the values in the local memory
                        data_i_cache[local_id_0][internal * THREAD_BLOCK_SIZE + local_id_1] = data_[(feature_block + threadIdx_x) * (num_rows_ + std::size_t{ 1 } + PADDING_SIZE_uz) + global_i_idx_linear];  // SoA
                        data_j_cache[local_id_0][internal * THREAD_BLOCK_SIZE + local_id_1] = data_[(feature_block + threadIdx_x) * (num_rows_ + std::size_t{ 1 } + PADDING_SIZE_uz) + global_j_idx_linear];  // SoA
                    }
                });

                // implicit group barrier

                // perform the feature reduction calculation
                group.parallel_for_work_item([&](::sycl::h_item<2> idx) {
                    // cast values to 32-bit unsigned int values to prevent implicit conversions
                    const auto local_id_0 = static_cast<unsigned>(idx.get_local_id(0));
                    const auto local_id_1 = static_cast<unsigned>(idx.get_local_id(1));

                    if constexpr (target == target_platform::cpu) {
                        // perform the feature reduction calculation, the feature is the fastest moving index
                        for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                            for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                                real_type sum{ 0.0 };
                                for (unsigned feature = 0; feature < THREAD_BLOCK_SIZE; ++feature) {
                                    sum += detail::feature_reduce<kernel_function>(data_i_cache[feature][local_id_1 * INTERNAL_BLOCK_SIZE + internal_i],
                                                                                   data_j_cache[feature][local_id_0 * INTERNAL_BLOCK_SIZE + internal_j]);
                                }
                                temp(idx)[internal_i][internal_j] += sum;
                            }
                        }
                    } else {
                        // perform the feature reduction calculation, the feature is the slowest moving index
                        for (unsigned feature = 0; feature < THREAD_BLOCK_SIZE; ++feature) {
                            for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                                for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                                    temp(idx)[internal_i][internal_j] += detail::feature_reduce<kernel_function>(data_i_cache[feature][local_id_1 * INTERNAL_BLOCK_SIZE + internal_i],
                                                                                                                 data_j_cache[feature][local_id_0 * INTERNAL_BLOCK_SIZE + internal_j]);
                                }
                            }
                        }
                    }
                });

                // implicit barrier
            }

            // apply the remaining part of the kernel function and store the value in the output kernel matrix
            group.parallel_for_work_item([&](::sycl::h_item<2> idx) {
                // cast all values to 64-bit std::size_t to prevent potential 32-bit overflows
                constexpr auto INTERNAL_BLOCK_SIZE_uz = static_cast<std::size_t>(INTERNAL_BLOCK_SIZE);
                constexpr auto PADDING_SIZE_uz = static_cast<std::size_t>(PADDING_SIZE);

                const auto threadIdx_x = static_cast<std::size_t>(idx.get_local_id(0));       // current work-item in work-group x-dimension
                const auto threadIdx_y = static_cast<std::size_t>(idx.get_local_id(1));       // current work-item in work-group y-dimension
                const auto blockDim_x = static_cast<std::size_t>(idx.get_local_range(0));     // number of work-items in work-group x-dimension
                const auto blockDim_y = static_cast<std::size_t>(idx.get_local_range(1));     // number of work-items in work-group y-dimension
                const auto blockIdx_x = static_cast<std::size_t>(group[0]) + grid_x_offset_;  // current work-group in global range x-dimension + offsets if the global range is too large
                const auto blockIdx_y = static_cast<std::size_t>(group[1]) + grid_y_offset_;  // current work-group in global range y-dimension + offsets if the global range is too large

                // calculate the indices used in the current work-item
                const auto i_idx = (blockIdx_y * blockDim_y + threadIdx_y) * INTERNAL_BLOCK_SIZE_uz;  // num_rows - device_row_offset
                const auto j_idx = (blockIdx_x * blockDim_x + threadIdx_x) * INTERNAL_BLOCK_SIZE_uz;  // device_num_rows

                for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                    for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                        // calculate the indices to access the global data and the data with respect to the current device
                        const auto device_global_i_idx = i_idx + static_cast<std::size_t>(internal_i);
                        const auto global_i_idx = device_row_offset_ + device_global_i_idx;
                        const auto device_global_j_idx = j_idx + static_cast<std::size_t>(internal_j);
                        const auto global_j_idx = device_row_offset_ + device_global_j_idx;

                        // be sure to not perform out-of-bounds accesses (only using the upper triangular matrix)
                        if (device_global_i_idx < (num_rows_ - device_row_offset_) && device_global_j_idx < device_num_rows_ && global_i_idx >= global_j_idx) {
                            real_type temp_ij = temp(idx)[internal_i][internal_j];
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
            });
        }
    }

  private:
    /// @cond Doxygen_suppress
    real_type *kernel_matrix_;
    const real_type *data_;
    const std::size_t num_rows_;
    const std::size_t device_num_rows_;
    const std::size_t device_row_offset_;
    const std::size_t num_features_;
    const real_type *q_;
    const real_type QA_cost_;
    const real_type cost_;
    const std::size_t grid_x_offset_;
    const std::size_t grid_y_offset_;
    const std::tuple<Args...> kernel_function_parameter_;
    /// @endcond
};

}  // namespace plssvm::sycl::detail::hierarchical

#endif  // PLSSVM_BACKENDS_SYCL_KERNEL_CG_EXPLICIT_HIERARCHICAL_KERNEL_MATRIX_ASSEMBLY_HPP_
