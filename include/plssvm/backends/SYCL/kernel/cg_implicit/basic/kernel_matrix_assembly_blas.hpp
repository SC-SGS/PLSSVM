/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Functions for implicitly assembling the kernel matrix using the SYCL backend and the basic data parallel kernels.
 */

#ifndef PLSSVM_BACKENDS_SYCL_KERNEL_CG_IMPLICIT_BASIC_KERNEL_MATRIX_ASSEMBLY_BLAS_HPP_
#define PLSSVM_BACKENDS_SYCL_KERNEL_CG_IMPLICIT_BASIC_KERNEL_MATRIX_ASSEMBLY_BLAS_HPP_
#pragma once

#include "plssvm/backends/SYCL/data_parallel_kernels.hpp"    // plssvm::sycl::data_parallel_kernel
#include "plssvm/backends/SYCL/detail/atomics.hpp"           // plssvm::sycl::detail::atomic_op
#include "plssvm/backends/SYCL/kernel/kernel_functions.hpp"  // plssvm::sycl::detail::{feature_reduce, apply_kernel_function}
#include "plssvm/constants.hpp"                              // plssvm::{real_type, THREAD_BLOCK_SIZE, INTERNAL_BLOCK_SIZE}
#include "plssvm/kernel_function_types.hpp"                  // plssvm::kernel_function_type

#include "sycl/sycl.hpp"  // sycl::item

#include <array>    // std::array
#include <cstddef>  // std::size_t
#include <tuple>    // std::tuple, std::make_tuple

namespace plssvm::sycl::detail::basic {

/**
 * @brief Perform an implicit BLAS SYMM-like operation: `C = alpha * A * B + C` where `A` is the implicitly calculated kernel matrix using the @p kernel_function (never actually stored, reducing the amount of needed global memory), @p B and @p C are matrices, and @p alpha is a scalar.
 * @details Uses SYCL's basic data parallel kernels.
 * @tparam kernel_function the type of the used kernel function
 * @tparam Args the types of the parameters necessary for the specific kernel function
 */
template <kernel_function_type kernel_function, typename... Args>
class device_kernel_assembly_symm {
  public:
    /// The used SYCL data parallel kernel.
    constexpr static sycl::data_parallel_kernel data_parallel_kernel_type = sycl::data_parallel_kernel::basic;

    /**
     * @brief Initialize the SYCL kernel function object.
     * @param[in] alpha the scalar alpha value
     * @param[in] q the vector used in the dimensional reduction
     * @param[in] data the data points to calculate the implicit kernel matrix from
     * @param[in] num_rows the total number of data points (= total number of rows)
     * @param[in] device_num_rows the number of rows the current device is responsible for
     * @param[in] device_row_offset the first row in @p data the current device is responsible for
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
    device_kernel_assembly_symm(const real_type alpha, const real_type *q, const real_type *data, const std::size_t num_rows, const std::size_t device_num_rows, const std::size_t device_row_offset, const std::size_t num_features, const real_type QA_cost, const real_type cost, const real_type *B, real_type *C, const std::size_t num_classes, const std::size_t grid_x_offset, const std::size_t grid_y_offset, Args... kernel_function_parameter) :
        alpha_{ alpha },
        q_{ q },
        data_{ data },
        num_rows_{ num_rows },
        device_num_rows_{ device_num_rows },
        device_row_offset_{ device_row_offset },
        num_features_{ num_features },
        QA_cost_{ QA_cost },
        cost_{ cost },
        B_{ B },
        C_{ C },
        num_classes_{ num_classes },
        grid_x_offset_{ grid_x_offset },
        grid_y_offset_{ grid_y_offset },
        kernel_function_parameter_{ std::make_tuple(kernel_function_parameter...) } { }

    /**
     * @brief Function call operator overload performing the actual calculation.
     * @param[in] idx indices representing the current point in the execution space
     */
    void operator()(::sycl::item<2> idx) const {
        // cast all values to 64-bit std::size_t to prevent potential 32-bit overflows
        constexpr auto INTERNAL_BLOCK_SIZE_uz = static_cast<std::size_t>(INTERNAL_BLOCK_SIZE);
        constexpr auto THREAD_BLOCK_SIZE_uz = static_cast<std::size_t>(THREAD_BLOCK_SIZE);

        // calculate the indices used in the current work-item
        const auto i_idx = (idx.get_id(1) + grid_y_offset_ * THREAD_BLOCK_SIZE_uz) * INTERNAL_BLOCK_SIZE_uz;  // num_rows - device_row_offset
        const auto j_idx = (idx.get_id(0) + grid_x_offset_ * THREAD_BLOCK_SIZE_uz) * INTERNAL_BLOCK_SIZE_uz;  // device_num_rows

        // only calculate the upper triangular matrix
        if (i_idx >= j_idx) {
            // create a work-item private array used for internal caching
            std::array<std::array<real_type, INTERNAL_BLOCK_SIZE>, INTERNAL_BLOCK_SIZE> temp{};

            //*************************************************************************//
            //                   inplace kernel matrix construction                    //
            //*************************************************************************//

            // perform the feature reduction calculation
            for (std::size_t feature = 0; feature < num_features_; ++feature) {
                for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                    for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                        // calculate the indices to access the global data
                        const auto global_i_idx = device_row_offset_ + i_idx + static_cast<std::size_t>(internal_i);
                        const auto global_j_idx = device_row_offset_ + j_idx + static_cast<std::size_t>(internal_j);

                        real_type data_i_cache = 0.0;
                        if (global_i_idx < num_rows_) {
                            data_i_cache = data_[feature * (num_rows_ + std::size_t{ 1 }) + global_i_idx];  // SoA
                        }
                        real_type data_j_cache = 0.0;
                        if (global_j_idx < num_rows_) {
                            data_j_cache = data_[feature * (num_rows_ + std::size_t{ 1 }) + global_j_idx];  // SoA
                        }

                        temp[internal_i][internal_j] += detail::feature_reduce<kernel_function>(data_i_cache, data_j_cache);
                    }
                }
            }

            // apply the remaining part of the kernel function and store the value in the output kernel matrix
            for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                    // calculate the indices to access the global data and the data with respect to the current device
                    const auto device_global_i_idx = i_idx + static_cast<std::size_t>(internal_i);
                    const auto global_i_idx = device_row_offset_ + device_global_i_idx;
                    const auto device_global_j_idx = j_idx + static_cast<std::size_t>(internal_j);
                    const auto global_j_idx = device_row_offset_ + device_global_j_idx;

                    // be sure to not perform out-of-bounds accesses (only using the upper triangular matrix)
                    if (global_i_idx < num_rows_ && global_j_idx < num_rows_ && device_global_i_idx < (num_rows_ - device_row_offset_) && device_global_j_idx < device_num_rows_ && global_i_idx >= global_j_idx) {
                        // apply the final kernel function
                        temp[internal_i][internal_j] = detail::apply_kernel_function<kernel_function>(temp[internal_i][internal_j], kernel_function_parameter_) + QA_cost_ - q_[global_i_idx] - q_[global_j_idx];
                        // apply the cost on the diagonal
                        if (global_i_idx == global_j_idx) {
                            temp[internal_i][internal_j] += cost_;
                        }
                    } else {
                        // be sure to set the value to zero otherwise
                        temp[internal_i][internal_j] = real_type{ 0.0 };
                    }
                }
            }

            //*************************************************************************//
            //                     calculate C += alpha * temp * B                     //
            //*************************************************************************//
            for (std::size_t class_block = 0; class_block < num_classes_; class_block += THREAD_BLOCK_SIZE_uz) {
                for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                    for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                        // calculate the indices to access the global data
                        const auto global_i_idx = device_row_offset_ + i_idx + static_cast<std::size_t>(internal_i);
                        const auto global_j_idx = device_row_offset_ + j_idx + static_cast<std::size_t>(internal_j);

                        if (global_i_idx < num_rows_ && global_j_idx < num_rows_) {
                            if (global_i_idx == global_j_idx) {
                                // only apply once to the diagonal
                                for (std::size_t class_idx = 0; class_idx < THREAD_BLOCK_SIZE_uz; ++class_idx) {
                                    if (class_block + class_idx < num_classes_) {
                                        detail::atomic_op<real_type>{ C_[global_i_idx * num_classes_ + class_block + class_idx] } += alpha_ * temp[internal_i][internal_j] * B_[global_i_idx * num_classes_ + class_block + class_idx];
                                    }
                                }
                            } else {
                                // apply it for the upper and lower triangular matrix
                                for (std::size_t class_idx = 0; class_idx < THREAD_BLOCK_SIZE_uz; ++class_idx) {
                                    if (class_block + class_idx < num_classes_) {
                                        detail::atomic_op<real_type>{ C_[global_i_idx * num_classes_ + class_block + class_idx] } += alpha_ * temp[internal_i][internal_j] * B_[global_j_idx * num_classes_ + class_block + class_idx];
                                        // symmetry
                                        detail::atomic_op<real_type>{ C_[global_j_idx * num_classes_ + class_block + class_idx] } += alpha_ * temp[internal_i][internal_j] * B_[global_i_idx * num_classes_ + class_block + class_idx];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

  private:
    /// @cond Doxygen_suppress
    const real_type alpha_;
    const real_type *q_;
    const real_type *data_;
    const std::size_t num_rows_;
    const std::size_t device_num_rows_;
    const std::size_t device_row_offset_;
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

}  // namespace plssvm::sycl::detail::basic

#endif  // PLSSVM_BACKENDS_SYCL_KERNEL_CG_IMPLICIT_BASIC_KERNEL_MATRIX_ASSEMBLY_BLAS_HPP_
