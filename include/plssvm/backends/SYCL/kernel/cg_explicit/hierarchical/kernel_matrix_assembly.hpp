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

#ifndef PLSSVM_BACKENDS_SYCL_CG_EXPLICIT_HIERARCHICAL_KERNEL_MATRIX_ASSEMBLY_HPP_
#define PLSSVM_BACKENDS_SYCL_CG_EXPLICIT_HIERARCHICAL_KERNEL_MATRIX_ASSEMBLY_HPP_
#pragma once

#include "plssvm/backends/SYCL/kernel/kernel_functions.hpp"  // plssvm::sycl::detail::{feature_reduce, apply_kernel_function}
#include "plssvm/constants.hpp"                              // plssvm::{real_type, THREAD_BLOCK_SIZE, INTERNAL_BLOCK_SIZE, FEATURE_BLOCK_SIZE, PADDING_SIZE}
#include "plssvm/kernel_function_types.hpp"                  // plssvm::kernel_function_type

#include "sycl/sycl.hpp"  // sycl::group, sycl::private_memory, sycl::h_item

#include <cstddef>  // std::size_t
#include <tuple>    // std::tuple, std::make_tuple

namespace plssvm::sycl::detail::hierarchical {

/**
 * @brief Create the explicit kernel matrix using the @p kernel_function.
 * @details Uses SYCL's hierarchical data parallel kernels.
 * @tparam kernel_function the type of the used kernel function
 * @tparam Args the types of the parameters necessary for the specific kernel function; stored in a `std::tuple`
 */
template <kernel_function_type kernel_function, typename... Args>
class device_kernel_assembly {
  public:
    /**
     * @brief Initialize the SYCL kernel function object.
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
    device_kernel_assembly(real_type *kernel_matrix_d, const real_type *data_d, const std::size_t num_rows, const std::size_t device_num_rows, const std::size_t row_offset, const std::size_t num_features, const real_type *q, const real_type QA_cost, const real_type cost, const std::size_t grid_x_offset, const std::size_t grid_y_offset, Args... kernel_function_parameter) :
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
        kernel_function_parameter_{ std::make_tuple(std::forward<Args>(kernel_function_parameter)...) } {
    }

    /**
     * @brief Function call operator overload performing the actual calculation.
     * @param[in] group indices representing the current point in the execution space
     */
    void operator()(::sycl::group<2> group) const {
        // allocate shared memory
        real_type data_cache_i[FEATURE_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE]; // TODO: 2D std::array?
        real_type data_cache_j[FEATURE_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];

        // calculate the indices used in the current work-item
        ::sycl::private_memory<std::size_t, 2> i{ group };
        ::sycl::private_memory<std::size_t, 2> i_linear{ group };
        ::sycl::private_memory<std::size_t, 2> j{ group };
        ::sycl::private_memory<std::size_t, 2> j_linear{ group };

        ::sycl::private_memory<real_type[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE], 2> temp{ group };

        // initialize private and local variables
        group.parallel_for_work_item([&](::sycl::h_item<2> idx) {
            const std::size_t threadIdx_x = idx.get_local_id(0);       // current thread in block x-dimension
            const std::size_t threadIdx_y = idx.get_local_id(1);       // current thread in block y-dimension
            const std::size_t blockDim_x = idx.get_local_range(0);     // number of threads in block x-dimension
            const std::size_t blockDim_y = idx.get_local_range(1);     // number of threads in block y-dimension
            const std::size_t blockIdx_x = group[0] + grid_x_offset_;  // current block in grid x-dimension + offsets if the grid size would be too large
            const std::size_t blockIdx_y = group[1] + grid_y_offset_;  // current block in grid y-dimension + offsets if the grid size would be too large

            const auto INTERNAL_BLOCK_SIZE_uz = static_cast<std::size_t>(INTERNAL_BLOCK_SIZE);

            // indices
            i(idx) = (blockIdx_y * blockDim_y + threadIdx_y) * INTERNAL_BLOCK_SIZE_uz;
            i_linear(idx) = blockIdx_y * blockDim_y * INTERNAL_BLOCK_SIZE_uz + threadIdx_y;
            j(idx) = (blockIdx_x * blockDim_x + threadIdx_x) * INTERNAL_BLOCK_SIZE_uz;
            j_linear(idx) = blockIdx_x * blockDim_x * INTERNAL_BLOCK_SIZE_uz + threadIdx_y;

            // initialize private temp matrix to zero
            for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                    temp(idx)[internal_i][internal_j] = real_type{ 0.0 };
                }
            }
        });

        // implicit group barrier

        // exploit symmetry
        if (group[1] >= group[0]) {
            for (std::size_t dim = 0; dim < num_features_; dim += static_cast<std::size_t>(FEATURE_BLOCK_SIZE)) {
                // load data into shared memory
                group.parallel_for_work_item([&](::sycl::h_item<2> idx) {
                    const auto local_id_0 = static_cast<unsigned>(idx.get_local_id(0));
                    const auto local_id_1 = static_cast<unsigned>(idx.get_local_id(1));

                    const std::size_t threadIdx_x = idx.get_local_id(0);

                    const auto THREAD_BLOCK_SIZE_uz = static_cast<std::size_t>(THREAD_BLOCK_SIZE);
                    const auto PADDING_SIZE_uz = static_cast<std::size_t>(PADDING_SIZE);

                    for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                        const auto global_i = row_offset_ + i_linear(idx) + static_cast<std::size_t>(internal) * THREAD_BLOCK_SIZE_uz;
                        const auto global_j = row_offset_ + j_linear(idx) + static_cast<std::size_t>(internal) * THREAD_BLOCK_SIZE_uz;

                        // FEATURE_BLOCK_SIZE = 2 * THREAD_BLOCK_SIZE -> store twice as many values in the local memory
                        data_cache_i[local_id_0][internal * THREAD_BLOCK_SIZE + local_id_1] = data_d_[(dim + threadIdx_x) * (num_rows_ + std::size_t{ 1 } + PADDING_SIZE_uz) + global_i];
                        data_cache_i[local_id_0 + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + local_id_1] = data_d_[(dim + threadIdx_x + THREAD_BLOCK_SIZE_uz) * (num_rows_ + std::size_t{ 1 } + PADDING_SIZE_uz) + global_i];
                        data_cache_j[local_id_0][internal * THREAD_BLOCK_SIZE + local_id_1] = data_d_[(dim + threadIdx_x) * (num_rows_ + std::size_t{ 1 } + PADDING_SIZE_uz) + global_j];
                        data_cache_j[local_id_0 + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + local_id_1] = data_d_[(dim + threadIdx_x + THREAD_BLOCK_SIZE_uz) * (num_rows_ + std::size_t{ 1 } + PADDING_SIZE_uz) + global_j];
                    }
                });

                // implicit group barrier

                // perform the feature reduction calculation
                group.parallel_for_work_item([&](::sycl::h_item<2> idx) {
                    const auto local_id_0 = static_cast<unsigned>(idx.get_local_id(0));
                    const auto local_id_1 = static_cast<unsigned>(idx.get_local_id(1));

                    for (unsigned block_dim = 0; block_dim < FEATURE_BLOCK_SIZE; ++block_dim) {
                        for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                            for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                                temp(idx)[internal_i][internal_j] += detail::feature_reduce<kernel_function>(data_cache_i[block_dim][local_id_1 * INTERNAL_BLOCK_SIZE + internal_i],
                                                                                                             data_cache_j[block_dim][local_id_0 * INTERNAL_BLOCK_SIZE + internal_j]);
                            }
                        }
                    }
                });

                // implicit barrier
            }

            // apply the remaining part of the kernel function and store the value in the output kernel matrix
            group.parallel_for_work_item([&](::sycl::h_item<2> idx) {
                const auto PADDING_SIZE_uz = static_cast<std::size_t>(PADDING_SIZE);

                for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                    for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                        // calculate the indices to access the kernel matrix (the part stored on the current device)
                        const auto device_global_i = i(idx) + static_cast<std::size_t>(internal_i);
                        const auto global_i = row_offset_ + i(idx) + static_cast<std::size_t>(internal_i);
                        const auto device_global_j = j(idx) + static_cast<std::size_t>(internal_j);
                        const auto global_j = row_offset_ + j(idx) + static_cast<std::size_t>(internal_j);

                        // be sure to not perform out of bounds accesses for the kernel matrix (only using the upper triangular matrix)
                        if (device_global_i < (num_rows_ - row_offset_) && device_global_j < device_num_rows_ && global_i >= global_j) {
                            real_type temp_ij = temp(idx)[internal_i][internal_j];
                            temp_ij = detail::apply_kernel_function<kernel_function>(temp_ij, kernel_function_parameter_) + QA_cost_ - q_[global_i] - q_[global_j];
                            // apply the cost on the diagonal
                            if (global_i == global_j) {
                                temp_ij += cost_;
                            }
                            // update the kernel matrix
                            kernel_matrix_d_[device_global_j * (num_rows_ - row_offset_ + PADDING_SIZE_uz) - device_global_j * (device_global_j + std::size_t{ 1 }) / std::size_t{ 2 } + device_global_i] = temp_ij;
                        }
                    }
                }
            });
        }
    }

  private:
    /// @cond Doxygen_suppress
    real_type *kernel_matrix_d_;
    const real_type *data_d_;
    const std::size_t num_rows_;
    const std::size_t device_num_rows_;
    const std::size_t row_offset_;
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

#endif  // PLSSVM_BACKENDS_SYCL_CG_EXPLICIT_HIERARCHICAL_KERNEL_MATRIX_ASSEMBLY_HPP_
