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

#ifndef PLSSVM_BACKENDS_SYCL_CG_IMPLICIT_BASIC_KERNEL_MATRIX_ASSEMBLY_BLAS_HPP_
#define PLSSVM_BACKENDS_SYCL_CG_IMPLICIT_BASIC_KERNEL_MATRIX_ASSEMBLY_BLAS_HPP_
#pragma once

#include "plssvm/backends/SYCL/detail/atomics.hpp"           // plssvm::sycl::detail::atomic_op
#include "plssvm/backends/SYCL/kernel/kernel_functions.hpp"  // plssvm::sycl::detail::{feature_reduce, apply_kernel_function}
#include "plssvm/constants.hpp"                              // plssvm::{real_type, THREAD_BLOCK_SIZE, INTERNAL_BLOCK_SIZE, FEATURE_BLOCK_SIZE, PADDING_SIZE}
#include "plssvm/kernel_function_types.hpp"                  // plssvm::kernel_function_type

#include "sycl/sycl.hpp"  // sycl::item

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
     * @param[in] idx indices representing the current point in the execution space
     */
    void operator()(::sycl::item<2> idx) const {
        // cast all values to 64-bit std::size_t to prevent potential 32-bit overflows
        const auto INTERNAL_BLOCK_SIZE_uz = static_cast<std::size_t>(INTERNAL_BLOCK_SIZE);
        const auto PADDING_SIZE_uz = static_cast<std::size_t>(PADDING_SIZE);

        // calculate the indices used in the current work-item
        const std::size_t i = (idx.get_id(1) + grid_x_offset_ * THREAD_BLOCK_SIZE) * INTERNAL_BLOCK_SIZE_uz;
        const std::size_t j = (idx.get_id(0) + grid_y_offset_ * THREAD_BLOCK_SIZE) * INTERNAL_BLOCK_SIZE_uz;

        // only calculate the upper triangular matrix
        if (i >= j) {
            // create a work-item private array used for internal caching
            real_type temp[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE]{};

            // iterate over all features using blocking to be able to cache them for faster memory accesses
            for (std::size_t dim = 0; dim < num_features_; ++dim) {
                // perform the feature reduction calculation
                for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                    for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                        const auto global_i = row_offset_ + i + static_cast<std::size_t>(internal_i);
                        const auto global_j = row_offset_ + j + static_cast<std::size_t>(internal_j);

                        temp[internal_i][internal_j] += detail::feature_reduce<kernel_function>(data_d_[dim * (num_rows_ + std::size_t{ 1 } + PADDING_SIZE_uz) + global_i],
                                                                                                data_d_[dim * (num_rows_ + std::size_t{ 1 } + PADDING_SIZE_uz) + global_j]);
                    }
                }
            }

            // apply the remaining part of the kernel function and store the value in the output kernel matrix
            for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                    const auto global_i = row_offset_ + i + static_cast<std::size_t>(internal_i);
                    const auto device_global_i = i + static_cast<std::size_t>(internal_i);
                    const auto global_j = row_offset_ + j + static_cast<std::size_t>(internal_j);
                    const auto device_global_j = j + static_cast<std::size_t>(internal_j);

                    // be sure to not perform out of bounds accesses for the kernel matrix (only using the upper triangular matrix)
                    if (device_global_i < (num_rows_ - row_offset_) && device_global_j < device_num_rows_ && global_i >= global_j) {
                        real_type temp_ij = temp[internal_i][internal_j];
                        temp_ij = detail::apply_kernel_function<kernel_function>(temp_ij, kernel_function_parameter_) + QA_cost_ - q_[global_i] - q_[global_j];
                        // apply the cost on the diagonal
                        if (global_i == global_j) {
                            temp_ij += cost_;
                            // calculate the values of alpha * A * B
                            for (std::size_t class_idx = 0; class_idx < num_classes_; ++class_idx) {
                                detail::atomic_op<real_type>{ C_[global_i * (num_classes_ + PADDING_SIZE_uz) + class_idx] } += alpha_ * temp_ij * B_[global_i * (num_classes_ + PADDING_SIZE_uz) + class_idx];
                            }
                        } else {
                            // calculate the values of alpha * A * B
                            for (std::size_t class_idx = 0; class_idx < num_classes_; ++class_idx) {
                                detail::atomic_op<real_type>{ C_[global_i * (num_classes_ + PADDING_SIZE_uz) + class_idx] } += alpha_ * temp_ij * B_[global_j * (num_classes_ + PADDING_SIZE_uz) + class_idx];
                                // symmetry
                                detail::atomic_op<real_type>{ C_[global_j * (num_classes_ + PADDING_SIZE_uz) + class_idx] } += alpha_ * temp_ij * B_[global_i * (num_classes_ + PADDING_SIZE_uz) + class_idx];
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

}  // namespace plssvm::sycl::detail::basic

#endif  // PLSSVM_BACKENDS_SYCL_CG_IMPLICIT_BASIC_KERNEL_MATRIX_ASSEMBLY_BLAS_HPP_
