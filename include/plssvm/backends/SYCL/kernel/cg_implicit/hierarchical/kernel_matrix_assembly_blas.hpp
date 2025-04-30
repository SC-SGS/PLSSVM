/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Functions for implicitly assembling the kernel matrix using the SYCL backend and the hierarchical data parallel kernels.
 */

#ifndef PLSSVM_BACKENDS_SYCL_CG_IMPLICIT_HIERARCHICAL_KERNEL_MATRIX_ASSEMBLY_BLAS_HPP_
#define PLSSVM_BACKENDS_SYCL_CG_IMPLICIT_HIERARCHICAL_KERNEL_MATRIX_ASSEMBLY_BLAS_HPP_
#pragma once

#include "plssvm/backends/SYCL/detail/atomics.hpp"           // plssvm::sycl::detail::atomic_op
#include "plssvm/backends/SYCL/kernel/kernel_functions.hpp"  // plssvm::sycl::detail::{feature_reduce, apply_kernel_function}
#include "plssvm/constants.hpp"                              // plssvm::{real_type, THREAD_BLOCK_SIZE, INTERNAL_BLOCK_SIZE, FEATURE_BLOCK_SIZE, PADDING_SIZE}
#include "plssvm/kernel_function_types.hpp"                  // plssvm::kernel_function_type

#include "sycl/sycl.hpp"  // sycl::group, sycl::private_memory, sycl::h_item

#include <cstddef>  // std::size_t
#include <tuple>    // std::tuple, std::make_tuple

namespace plssvm::sycl::detail::hierarchical {

/**
 * @brief Perform an implicit BLAS SYMM-like operation: `C = alpha * A * B + C` where `A` is the implicitly calculated kernel matrix using the @p kernel_function (never actually stored, reducing the amount of needed global memory), @p B and @p C are matrices, and @p alpha is a scalar.
 * @details Uses SYCL's hierarchical data parallel kernels.
 * @tparam kernel_function the type of the used kernel function
 * @tparam Args the types of the parameters necessary for the specific kernel function
 */
template <kernel_function_type kernel_function, typename... Args>
class device_kernel_assembly_symm {
  public:
    /**
     * @brief Initialize the SYCL kernel function object.
     * @param[in] alpha the scalar alpha value
     * @param[in] q the vector used in the dimensional reduction
     * @param[in] data_d the data points to calculate the implicit kernel matrix from
     * @param[in] num_rows the total number of data points (= total number of rows)
     * @param[in] device_num_rows the number of rows the current device is responsible for
     * @param[in] row_offset the first row in @p data_d the current device is responsible for
     * @param[in] num_features the number of features per data point
     * @param[in] QA_cost the scalar used in the dimensional reduction
     * @param[in] cost the cost factor the diagonal is scaled with
     * @param[in] B the matrix @p B
     * @param[in,out] C the matrix @p C
     * @param[in] num_classes the number of classes in the data set
     * @param[in] grid_x_offset the offset in x-dimension into the data points if more than one execution grid has to be used
     * @param[in] grid_y_offset the offset in y-dimension into the data points if more than one execution grid has to be used
     * @param[in] kernel_function_parameter the parameters necessary to apply the @p kernel_function
     */
    device_kernel_assembly_symm(const real_type alpha, const real_type *q, const real_type *data_d, const std::size_t num_rows, const std::size_t device_num_rows, const std::size_t row_offset, const std::size_t num_features, const real_type QA_cost, const real_type cost, const real_type *B, real_type *C, const std::size_t num_classes, const std::size_t grid_x_offset, const std::size_t grid_y_offset, Args... kernel_function_parameter) :
        alpha_{ alpha },
        q_{ q },
        data_d_{ data_d },
        num_rows_{ num_rows },
        device_num_rows_{ device_num_rows },
        row_offset_{ row_offset },
        num_features_{ num_features },
        QA_cost_{ QA_cost },
        cost_{ cost },
        B_{ B },
        C_{ C },
        num_classes_{ num_classes },
        grid_x_offset_{ grid_x_offset },
        grid_y_offset_{ grid_y_offset },
        kernel_function_parameter_{ std::make_tuple(std::forward<Args>(kernel_function_parameter)...) } { }

    /**
     * @brief Function call operator overload performing the actual calculation.
     * @param[in] group indices representing the current point in the execution space
     */
    void operator()(::sycl::group<2> group) const {
        // allocate shared memory
        real_type data_cache_i[FEATURE_BLOCK_SIZE * INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];
        real_type data_cache_j[FEATURE_BLOCK_SIZE * INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];

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

        // only calculate the upper triangular matrix -> can't use get_local_id() since all work-items in a work-group must progress further
        if (group[1] >= group[0]) {
            // iterate over all features using blocking to be able to cache them for faster memory accesses
            for (std::size_t dim = 0; dim < num_features_; dim += static_cast<std::size_t>(FEATURE_BLOCK_SIZE)) {
                // load data into local memory
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
                        data_cache_i[local_id_0 * INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE + internal * THREAD_BLOCK_SIZE + local_id_1] = data_d_[(dim + threadIdx_x) * (num_rows_ + std::size_t{ 1 } + PADDING_SIZE_uz) + global_i];
                        data_cache_i[(local_id_0 + THREAD_BLOCK_SIZE) * INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE + internal * THREAD_BLOCK_SIZE + local_id_1] = data_d_[(dim + threadIdx_x + THREAD_BLOCK_SIZE_uz) * (num_rows_ + std::size_t{ 1 } + PADDING_SIZE_uz) + global_i];
                        data_cache_j[local_id_0 * INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE + internal * THREAD_BLOCK_SIZE + local_id_1] = data_d_[(dim + threadIdx_x) * (num_rows_ + std::size_t{ 1 } + PADDING_SIZE_uz) + global_j];
                        data_cache_j[(local_id_0 + THREAD_BLOCK_SIZE) * INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE + internal * THREAD_BLOCK_SIZE + local_id_1] = data_d_[(dim + threadIdx_x + THREAD_BLOCK_SIZE_uz) * (num_rows_ + std::size_t{ 1 } + PADDING_SIZE_uz) + global_j];
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
                                temp(idx)[internal_i][internal_j] += detail::feature_reduce<kernel_function>(data_cache_i[block_dim * INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE + local_id_1 * INTERNAL_BLOCK_SIZE + internal_i],
                                                                                                             data_cache_j[block_dim * INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE + local_id_0 * INTERNAL_BLOCK_SIZE + internal_j]);
                            }
                        }
                    }
                });

                // implicit group barrier
            }

            // apply the remaining part of the kernel function and store the value in the output kernel matrix
            group.parallel_for_work_item([&](::sycl::h_item<2> idx) {
                for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                    for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                        const auto global_i = row_offset_ + i(idx) + static_cast<std::size_t>(internal_i);
                        const auto device_global_i = i(idx) + static_cast<std::size_t>(internal_i);
                        const auto global_j = row_offset_ + j(idx) + static_cast<std::size_t>(internal_j);
                        const auto device_global_j = j(idx) + static_cast<std::size_t>(internal_j);

                        // be sure to not perform out of bounds accesses for the kernel matrix (only using the upper triangular matrix)
                        if (device_global_i < (num_rows_ - row_offset_) && device_global_j < device_num_rows_ && global_i >= global_j) {
                            temp(idx)[internal_i][internal_j] = detail::apply_kernel_function<kernel_function>(temp(idx)[internal_i][internal_j], kernel_function_parameter_) + QA_cost_ - q_[global_i] - q_[global_j];
                            // apply the cost on the diagonal
                            if (global_i == global_j) {
                                temp(idx)[internal_i][internal_j] += cost_;
                            }
                        } else {
                            // be sure to set the value to zero otherwise
                            temp(idx)[internal_i][internal_j] = real_type{ 0.0 };
                        }
                    }
                }
            });

            // implicit group barrier

            // calculate C += alpha * temp * B for the UPPER triangular matrix
            {
                // allocate shared memory
                auto &B_cache = data_cache_i;      // [INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE][FEATURE_BLOCK_SIZE]
                auto &C_out_cache = data_cache_j;  // [INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE][FEATURE_BLOCK_SIZE]

                // iterate over all classes using blocking to be able to cache them for faster memory accesses
                for (std::size_t dim = 0; dim < num_classes_; dim += static_cast<std::size_t>(FEATURE_BLOCK_SIZE)) {
                    // load data into local memory
                    group.parallel_for_work_item([&](::sycl::h_item<2> idx) {
                        const auto local_id_0 = static_cast<unsigned>(idx.get_local_id(0));
                        const auto local_id_1 = static_cast<unsigned>(idx.get_local_id(1));

                        const std::size_t threadIdx_x = idx.get_local_id(0);

                        const auto THREAD_BLOCK_SIZE_uz = static_cast<std::size_t>(THREAD_BLOCK_SIZE);
                        const auto PADDING_SIZE_uz = static_cast<std::size_t>(PADDING_SIZE);

                        for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                            const std::size_t global_i = row_offset_ + i_linear(idx) + static_cast<std::size_t>(internal) * THREAD_BLOCK_SIZE_uz;

                            // FEATURE_BLOCK_SIZE = 2 * THREAD_BLOCK_SIZE -> store twice as many values in the local memory
                            B_cache[(internal * THREAD_BLOCK_SIZE + local_id_1) * FEATURE_BLOCK_SIZE + local_id_0] = alpha_ * B_[global_i * (num_classes_ + PADDING_SIZE_uz) + dim + threadIdx_x];
                            B_cache[(internal * THREAD_BLOCK_SIZE + local_id_1) * FEATURE_BLOCK_SIZE + local_id_0 + THREAD_BLOCK_SIZE] = alpha_ * B_[global_i * (num_classes_ + PADDING_SIZE_uz) + dim + threadIdx_x + THREAD_BLOCK_SIZE_uz];
                            C_out_cache[(internal * THREAD_BLOCK_SIZE + local_id_1) * FEATURE_BLOCK_SIZE + local_id_0] = real_type{ 0.0 };
                            C_out_cache[(internal * THREAD_BLOCK_SIZE + local_id_1) * FEATURE_BLOCK_SIZE + local_id_0 + THREAD_BLOCK_SIZE] = real_type{ 0.0 };
                        }
                    });

                    // implicit group barrier

                    // calculate intermediate results and store them in shared memory
                    for (unsigned class_idx = 0; class_idx < FEATURE_BLOCK_SIZE; ++class_idx) {
                        group.parallel_for_work_item([&](::sycl::h_item<2> idx) {
                            const auto local_id_0 = static_cast<unsigned>(idx.get_local_id(0));
                            const auto local_id_1 = static_cast<unsigned>(idx.get_local_id(1));

                            for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                                for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                                    C_out_cache[(local_id_0 * INTERNAL_BLOCK_SIZE + internal_j) * FEATURE_BLOCK_SIZE + (class_idx + local_id_1) % FEATURE_BLOCK_SIZE] +=
                                        temp(idx)[internal_i][internal_j] * B_cache[(local_id_1 * INTERNAL_BLOCK_SIZE + internal_i) * FEATURE_BLOCK_SIZE + (class_idx + local_id_1) % FEATURE_BLOCK_SIZE];
                                }
                            }
                        });

                        // implicit group barrier
                    }

                    // add intermediate cached results to C
                    group.parallel_for_work_item([&](::sycl::h_item<2> idx) {
                        const auto local_id_0 = static_cast<unsigned>(idx.get_local_id(0));
                        const auto local_id_1 = static_cast<unsigned>(idx.get_local_id(1));

                        const std::size_t threadIdx_y = idx.get_local_id(1);

                        const auto THREAD_BLOCK_SIZE_uz = static_cast<std::size_t>(THREAD_BLOCK_SIZE);
                        const auto PADDING_SIZE_uz = static_cast<std::size_t>(PADDING_SIZE);

                        for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                            const auto global_j = row_offset_ + j(idx) + static_cast<std::size_t>(internal);
                            detail::atomic_op<real_type>{ C_[global_j * (num_classes_ + PADDING_SIZE_uz) + dim + threadIdx_y] } += C_out_cache[(local_id_0 * INTERNAL_BLOCK_SIZE + internal) * FEATURE_BLOCK_SIZE + local_id_1];
                            detail::atomic_op<real_type>{ C_[global_j * (num_classes_ + PADDING_SIZE_uz) + dim + threadIdx_y + THREAD_BLOCK_SIZE_uz] } += C_out_cache[(local_id_0 * INTERNAL_BLOCK_SIZE + internal) * FEATURE_BLOCK_SIZE + local_id_1 + THREAD_BLOCK_SIZE];
                        }
                    });

                    // implicit group barrier
                }
            }

            // set potential diagonal entries in temp to 0.0 such that we don't apply the main diagonal twice to C
            group.parallel_for_work_item([&](::sycl::h_item<2> idx) {
                for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                    for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                        const auto global_i = row_offset_ + i(idx) + static_cast<std::size_t>(internal_i);
                        const auto global_j = row_offset_ + j(idx) + static_cast<std::size_t>(internal_j);

                        if (global_i == global_j) {
                            temp(idx)[internal_i][internal_j] = real_type{ 0.0 };
                        }
                    }
                }
            });

            // implicit group barrier

            // calculate C += alpha * temp * B for the LOWER triangular matrix
            {
                // allocate shared memory
                auto &B_cache = data_cache_i;      // [FEATURE_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE]
                auto &C_out_cache = data_cache_j;  // [FEATURE_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE]

                // iterate over all classes using blocking to be able to cache them for faster memory accesses
                for (std::size_t dim = 0; dim < num_classes_; dim += static_cast<std::size_t>(FEATURE_BLOCK_SIZE)) {
                    group.parallel_for_work_item([&](::sycl::h_item<2> idx) {
                        const auto local_id_0 = static_cast<unsigned>(idx.get_local_id(0));
                        const auto local_id_1 = static_cast<unsigned>(idx.get_local_id(1));

                        const std::size_t threadIdx_x = idx.get_local_id(0);

                        const auto THREAD_BLOCK_SIZE_uz = static_cast<std::size_t>(THREAD_BLOCK_SIZE);
                        const auto PADDING_SIZE_uz = static_cast<std::size_t>(PADDING_SIZE);

                        // load data into local memory
                        for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                            const auto global_j = row_offset_ + j_linear(idx) + static_cast<std::size_t>(internal) * THREAD_BLOCK_SIZE_uz;

                            // FEATURE_BLOCK_SIZE = 2 * THREAD_BLOCK_SIZE -> store twice as many values in the shared memory
                            B_cache[local_id_0 * INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE + internal * THREAD_BLOCK_SIZE + local_id_1] = alpha_ * B_[global_j * (num_classes_ + PADDING_SIZE_uz) + dim + threadIdx_x];
                            B_cache[(local_id_0 + THREAD_BLOCK_SIZE) * INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE + internal * THREAD_BLOCK_SIZE + local_id_1] = alpha_ * B_[global_j * (num_classes_ + PADDING_SIZE_uz) + dim + threadIdx_x + THREAD_BLOCK_SIZE_uz];
                            C_out_cache[local_id_0 * INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE + internal * THREAD_BLOCK_SIZE + local_id_1] = real_type{ 0.0 };
                            C_out_cache[(local_id_0 + THREAD_BLOCK_SIZE) * INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE + internal * THREAD_BLOCK_SIZE + local_id_1] = real_type{ 0.0 };
                        }
                    });

                    // implicit group barrier

                    // calculate intermediate results and store them in shared memory
                    for (unsigned class_idx = 0; class_idx < FEATURE_BLOCK_SIZE; ++class_idx) {
                        group.parallel_for_work_item([&](::sycl::h_item<2> idx) {
                            const auto local_id_0 = static_cast<unsigned>(idx.get_local_id(0));
                            const auto local_id_1 = static_cast<unsigned>(idx.get_local_id(1));

                            for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                                for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                                    C_out_cache[((class_idx + local_id_0) % FEATURE_BLOCK_SIZE) * INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE + internal_i * THREAD_BLOCK_SIZE + local_id_1] +=
                                        temp(idx)[internal_i][internal_j] * B_cache[((class_idx + local_id_0) % FEATURE_BLOCK_SIZE) * INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE + local_id_0 * INTERNAL_BLOCK_SIZE + internal_j];
                                }
                            }
                        });

                        // implicit group barrier
                    }

                    // add intermediate cached results to C
                    group.parallel_for_work_item([&](::sycl::h_item<2> idx) {
                        const auto local_id_0 = static_cast<unsigned>(idx.get_local_id(0));
                        const auto local_id_1 = static_cast<unsigned>(idx.get_local_id(1));

                        const std::size_t threadIdx_x = idx.get_local_id(0);

                        const auto THREAD_BLOCK_SIZE_uz = static_cast<std::size_t>(THREAD_BLOCK_SIZE);
                        const auto PADDING_SIZE_uz = static_cast<std::size_t>(PADDING_SIZE);

                        for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                            const auto global_i = row_offset_ + i(idx) + static_cast<std::size_t>(internal);
                            detail::atomic_op<real_type>{ C_[global_i * (num_classes_ + PADDING_SIZE_uz) + dim + threadIdx_x] } += C_out_cache[local_id_0 * INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE + internal * THREAD_BLOCK_SIZE + local_id_1];
                            detail::atomic_op<real_type>{ C_[global_i * (num_classes_ + PADDING_SIZE_uz) + dim + threadIdx_x + THREAD_BLOCK_SIZE_uz] } += C_out_cache[(local_id_0 + THREAD_BLOCK_SIZE) * INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE + internal * THREAD_BLOCK_SIZE + local_id_1];
                        }
                    });

                    // implicit group barrier
                }
            }
        }
    }

  private:
    /// @cond Doxygen_suppress
    const real_type alpha_;
    const real_type *q_;
    const real_type *data_d_;
    const std::size_t num_rows_;
    const std::size_t device_num_rows_;
    const std::size_t row_offset_;
    const std::size_t num_features_;
    const real_type QA_cost_;
    const real_type cost_;
    const real_type *B_;
    real_type *C_;
    const std::size_t num_classes_;
    const std::size_t grid_x_offset_;
    const std::size_t grid_y_offset_;
    const std::tuple<Args...> kernel_function_parameter_;
    /// @endcond
};

}  // namespace plssvm::sycl::detail::hierarchical

#endif  // PLSSVM_BACKENDS_SYCL_CG_IMPLICIT_HIERARCHICAL_KERNEL_MATRIX_ASSEMBLY_BLAS_HPP_
