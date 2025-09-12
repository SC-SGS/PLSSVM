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

#include "plssvm/backends/SYCL/data_parallel_kernels.hpp"    // plssvm::sycl::data_parallel_kernel
#include "plssvm/backends/SYCL/detail/atomics.hpp"           // plssvm::sycl::detail::atomic_op
#include "plssvm/backends/SYCL/kernel/kernel_functions.hpp"  // plssvm::sycl::detail::{feature_reduce, apply_kernel_function}
#include "plssvm/constants.hpp"                              // plssvm::real_type
#include "plssvm/kernel_function_types.hpp"                  // plssvm::kernel_function_type

#include "sycl/sycl.hpp"  // sycl::group, sycl::h_item

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
    /// The used SYCL data parallel kernel.
    constexpr static sycl::data_parallel_kernel data_parallel_kernel_type = sycl::data_parallel_kernel::hierarchical;

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
     * @param[in] group indices representing the current point in the execution space
     */
    void operator()(::sycl::group<2> group) const {
        group.parallel_for_work_item([&](::sycl::h_item<2> idx) {
            const auto threadIdx_x = static_cast<std::size_t>(idx.get_local_id(0));       // current work-item in work-group x-dimension
            const auto threadIdx_y = static_cast<std::size_t>(idx.get_local_id(1));       // current work-item in work-group y-dimension
            const auto blockDim_x = static_cast<std::size_t>(idx.get_local_range(0));     // number of work-items in work-group x-dimension
            const auto blockDim_y = static_cast<std::size_t>(idx.get_local_range(1));     // number of work-items in work-group y-dimension
            const auto blockIdx_x = static_cast<std::size_t>(group[0]) + grid_x_offset_;  // current work-group in global range x-dimension + offsets if the global range is too large
            const auto blockIdx_y = static_cast<std::size_t>(group[1]) + grid_y_offset_;  // current work-group in global range y-dimension + offsets if the global range is too large

            // calculate the indices used in the current work-item
            const auto device_global_i_idx = blockIdx_y * blockDim_y + threadIdx_y;  // num_rows - device_row_offset
            const auto global_i_idx = device_row_offset_ + device_global_i_idx;
            const auto device_global_j_idx = blockIdx_x * blockDim_x + threadIdx_x;  // device_num_rows
            const auto global_j_idx = device_row_offset_ + device_global_j_idx;

            // be sure to not perform out-of-bounds accesses (only using the upper triangular matrix)
            if (device_global_i_idx < (num_rows_ - device_row_offset_) && device_global_j_idx < device_num_rows_ && global_i_idx >= global_j_idx) {
                //*************************************************************************//
                //                   inplace kernel matrix construction                    //
                //*************************************************************************//
                real_type temp{ 0.0 };

                // perform the feature reduction calculation
                for (std::size_t feature = 0; feature < num_features_; ++feature) {
                    temp += detail::feature_reduce<kernel_function>(data_[global_i_idx * num_features_ + feature],   // AoS
                                                                    data_[global_j_idx * num_features_ + feature]);  // AoS
                }

                // apply the final kernel function
                temp = detail::apply_kernel_function<kernel_function>(temp, kernel_function_parameter_) + QA_cost_ - q_[global_i_idx] - q_[global_j_idx];
                // apply the cost on the diagonal
                if (global_i_idx == global_j_idx) {
                    temp += cost_;
                }

                //*************************************************************************//
                //     calculate C += alpha * temp * B for the UPPER triangular matrix     //
                //*************************************************************************//
                for (std::size_t class_idx = 0; class_idx < num_classes_; ++class_idx) {
                    const real_type B_cache = alpha_ * B_[class_idx * num_rows_ + global_i_idx];                 // AoS
                    detail::atomic_op<real_type>{ C_[class_idx * num_rows_ + global_j_idx] } += temp * B_cache;  // AoS
                }

                // set potential diagonal entries in temp to 0.0 such that we don't apply the main diagonal twice to C
                if (global_i_idx == global_j_idx) {
                    temp = real_type{ 0.0 };
                }

                //*************************************************************************//
                //     calculate C += alpha * temp * B for the LOWER triangular matrix     //
                //*************************************************************************//
                for (std::size_t class_idx = 0; class_idx < num_classes_; ++class_idx) {
                    const real_type B_cache = alpha_ * B_[class_idx * num_rows_ + global_j_idx];                 // AoS
                    detail::atomic_op<real_type>{ C_[class_idx * num_rows_ + global_i_idx] } += temp * B_cache;  // AoS
                }
            }
        });
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

}  // namespace plssvm::sycl::detail::hierarchical

#endif  // PLSSVM_BACKENDS_SYCL_CG_IMPLICIT_HIERARCHICAL_KERNEL_MATRIX_ASSEMBLY_BLAS_HPP_
