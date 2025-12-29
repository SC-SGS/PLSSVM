/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Functions for implicitly assembling the kernel matrix using the SYCL backend and AdaptiveCpp's scoped parallelism.
 */

#ifndef PLSSVM_BACKENDS_SYCL_KERNEL_CG_IMPLICIT_SCOPED_KERNEL_MATRIX_ASSEMBLY_BLAS_HPP_
#define PLSSVM_BACKENDS_SYCL_KERNEL_CG_IMPLICIT_SCOPED_KERNEL_MATRIX_ASSEMBLY_BLAS_HPP_
#pragma once

#include "plssvm/backends/SYCL/data_parallel_kernels.hpp"    // plssvm::sycl::data_parallel_kernel
#include "plssvm/backends/SYCL/detail/atomics.hpp"           // plssvm::sycl::detail::atomic_op
#include "plssvm/backends/SYCL/kernel/kernel_functions.hpp"  // plssvm::sycl::detail::{feature_reduce, apply_kernel_function}
#include "plssvm/constants.hpp"                              // plssvm::real_type
#include "plssvm/kernel_function_types.hpp"                  // plssvm::kernel_function_type

#include "sycl/sycl.hpp"  // sycl::memory_environment, sycl::distribute_items_and_wait, sycl::s_item

#include <cstddef>  // std::size_t
#include <tuple>    // std::tuple, std::make_tuple

namespace plssvm::sycl::detail::scoped {

/**
 * @brief Perform an implicit BLAS SYMM-like operation: `C = alpha * A * B + C` where `A` is the implicitly calculated kernel matrix using the @p kernel_function (never actually stored, reducing the amount of needed global memory), @p B and @p C are matrices, and @p alpha is a scalar.
 * @details Uses AdaptiveCpp's scoped parallelism.
 * @tparam kernel_function the type of the used kernel function
 * @tparam Args the types of the parameters necessary for the specific kernel function
 */
template <kernel_function_type kernel_function, typename... Args>
class device_kernel_assembly_symm {
  public:
    /// The used SYCL data parallel kernel.
    constexpr static sycl::data_parallel_kernel data_parallel_kernel_type = sycl::data_parallel_kernel::scoped;

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
     * @tparam T the implementation defined type of the group to iterate
     * @param[in] group group representing the current point in the execution space
     */
    template <typename T>
    void operator()(T group) const {
        ::sycl::memory_environment(group,
                                   // the indices used in the current work-item
                                   ::sycl::require_private_mem<std::size_t>(),  // num_rows - device_row_offset
                                   ::sycl::require_private_mem<std::size_t>(),  // device_num_rows

                                   ::sycl::require_private_mem<std::size_t>(),  // num_rows - device_row_offset
                                   ::sycl::require_private_mem<std::size_t>(),  // device_num_rows

                                   // create two local memory arrays used for caching
                                   ::sycl::require_local_mem<std::array<std::array<real_type, static_cast<std::size_t>(THREAD_BLOCK_SIZE)>, static_cast<std::size_t>(THREAD_BLOCK_SIZE)>>(),  // cache_one
                                   ::sycl::require_local_mem<std::array<std::array<real_type, static_cast<std::size_t>(THREAD_BLOCK_SIZE)>, static_cast<std::size_t>(THREAD_BLOCK_SIZE)>>(),  // cache_two

                                   ::sycl::require_private_mem<real_type>(),
                                   [&](auto &i_idx, auto &j_idx, auto &i_idx_linear, auto &j_idx_linear, auto &cache_one, auto &cache_two, auto &temp) {
                                       // initialize private and local variables
                                       ::sycl::distribute_items_and_wait(group, [&](::sycl::s_item<2> idx) {
                                           const auto threadIdx_x = static_cast<std::size_t>(idx.get_local_id(group, 0));       // current work-item in work-group x-dimension
                                           const auto threadIdx_y = static_cast<std::size_t>(idx.get_local_id(group, 1));       // current work-item in work-group y-dimension
                                           const auto blockDim_x = static_cast<std::size_t>(group.get_logical_local_range(0));  // number of work-items in work-group x-dimension
                                           const auto blockDim_y = static_cast<std::size_t>(group.get_logical_local_range(1));  // number of work-items in work-group y-dimension
                                           const auto blockIdx_x = static_cast<std::size_t>(group[0]) + grid_x_offset_;         // current work-group in global range x-dimension + offsets if the global range is too large
                                           const auto blockIdx_y = static_cast<std::size_t>(group[1]) + grid_y_offset_;         // current work-group in global range y-dimension + offsets if the global range is too large

                                           // calculate the indices to access the global data
                                           i_idx(idx) = blockIdx_y * blockDim_y + threadIdx_y;
                                           j_idx(idx) = blockIdx_x * blockDim_x + threadIdx_x;
                                           // calculate the indices to access the global data, pays attention to coalesced memory accesses
                                           i_idx_linear(idx) = blockIdx_y * blockDim_y + threadIdx_y;
                                           j_idx_linear(idx) = blockIdx_x * blockDim_x + threadIdx_y;

                                           // initialize private temp to zero
                                           temp(idx) = real_type{ 0.0 };
                                       });

                                       // only calculate the upper triangular matrix -> can't use get_local_id() since all work-items in a work-group must progress further
                                       if (group[1] >= group[0]) {
                                           //*************************************************************************//
                                           //                   inplace kernel matrix construction                    //
                                           //*************************************************************************//
                                           {
                                               // rename the local memory array
                                               auto &data_i_cache = cache_one;
                                               auto &data_j_cache = cache_two;

                                               // iterate over all features using blocking to be able to cache them for faster memory accesses
                                               for (std::size_t feature_block = 0; feature_block < num_features_; feature_block += static_cast<std::size_t>(THREAD_BLOCK_SIZE)) {
                                                   // load data into local memory
                                                   ::sycl::distribute_items_and_wait(group, [&](::sycl::s_item<2> idx) {
                                                       // cast values to 32-bit unsigned int values to prevent implicit conversions
                                                       const auto local_id_0 = static_cast<unsigned>(idx.get_local_id(group, 0));
                                                       const auto local_id_1 = static_cast<unsigned>(idx.get_local_id(group, 1));

                                                       const auto threadIdx_x = static_cast<std::size_t>(idx.get_local_id(group, 0));  // current work-item in work-group x-dimension

                                                       // zero-out local memory
                                                       data_i_cache[local_id_0][local_id_1] = real_type{ 0.0 };
                                                       data_j_cache[local_id_0][local_id_1] = real_type{ 0.0 };

                                                       // calculate the indices to access the global data, pays attention to coalesced memory accesses
                                                       const auto global_i_idx_linear = device_row_offset_ + i_idx_linear(idx);
                                                       const auto global_j_idx_linear = device_row_offset_ + j_idx_linear(idx);

                                                       // load data into local memory
                                                       if (feature_block + threadIdx_x < num_features_) {
                                                           if (global_i_idx_linear < num_rows_) {
                                                               data_i_cache[local_id_0][local_id_1] = data_[(feature_block + threadIdx_x) * (num_rows_ + std::size_t{ 1 }) + global_i_idx_linear];  // SoA
                                                           }
                                                           if (global_j_idx_linear < num_rows_) {
                                                               data_j_cache[local_id_0][local_id_1] = data_[(feature_block + threadIdx_x) * (num_rows_ + std::size_t{ 1 }) + global_j_idx_linear];  // SoA
                                                           }
                                                       }
                                                   });

                                                   // perform the feature reduction calculation
                                                   ::sycl::distribute_items_and_wait(group, [&](::sycl::s_item<2> idx) {
                                                       // cast values to 32-bit unsigned int values to prevent implicit conversions
                                                       const auto local_id_0 = static_cast<unsigned>(idx.get_local_id(group, 0));
                                                       const auto local_id_1 = static_cast<unsigned>(idx.get_local_id(group, 1));

                                                       // perform the feature reduction calculation
                                                       for (unsigned feature = 0; feature < THREAD_BLOCK_SIZE; ++feature) {
                                                           temp(idx) += detail::feature_reduce<kernel_function>(data_i_cache[feature][local_id_1],
                                                                                                                data_j_cache[feature][local_id_0]);
                                                       }
                                                   });
                                               }
                                           }

                                           // apply the remaining part of the kernel function and store the value in the output kernel matrix
                                           ::sycl::distribute_items_and_wait(group, [&](::sycl::s_item<2> idx) {
                                               // calculate the indices to access the global data and the data with respect to the current device
                                               const auto global_i_idx = device_row_offset_ + i_idx(idx);
                                               const auto global_j_idx = device_row_offset_ + j_idx(idx);

                                               // be sure to not perform out-of-bounds accesses (only using the upper triangular matrix)
                                               if (i_idx(idx) < (num_rows_ - device_row_offset_) && j_idx(idx) < device_num_rows_ && global_i_idx >= global_j_idx) {
                                                   // apply the final kernel function
                                                   temp(idx) = detail::apply_kernel_function<kernel_function>(temp(idx), kernel_function_parameter_) + QA_cost_ - q_[global_i_idx] - q_[global_j_idx];
                                                   // apply the cost on the diagonal
                                                   if (global_i_idx == global_j_idx) {
                                                       temp(idx) += cost_;
                                                   }
                                               } else {
                                                   // be sure to set the value to zero otherwise
                                                   temp(idx) = real_type{ 0.0 };
                                               }
                                           });

                                           //*************************************************************************//
                                           //     calculate C += alpha * temp * B for the UPPER triangular matrix     //
                                           //*************************************************************************//
                                           {
                                               // rename the local memory array
                                               auto &B_cache = cache_one;
                                               auto &C_out_cache = cache_two;

                                               // iterate over all classes using blocking to be able to cache them for faster memory accesses
                                               for (std::size_t class_block = 0; class_block < num_classes_; class_block += static_cast<std::size_t>(THREAD_BLOCK_SIZE)) {
                                                   // load data into local memory
                                                   ::sycl::distribute_items_and_wait(group, [&](::sycl::s_item<2> idx) {
                                                       // cast values to 32-bit unsigned int values to prevent implicit conversions
                                                       const auto local_id_0 = static_cast<unsigned>(idx.get_local_id(group, 0));
                                                       const auto local_id_1 = static_cast<unsigned>(idx.get_local_id(group, 1));

                                                       const auto threadIdx_x = static_cast<std::size_t>(idx.get_local_id(group, 0));  // current work-item in work-group x-dimension

                                                       // calculate the indices to access the global data, pays attention to coalesced memory accesses
                                                       const auto global_i_idx_linear = device_row_offset_ + i_idx_linear(idx);

                                                       // zero-out local memory
                                                       B_cache[local_id_1][local_id_0] = real_type{ 0.0 };
                                                       C_out_cache[local_id_1][local_id_0] = real_type{ 0.0 };

                                                       // load data into shared memory
                                                       if (class_block + threadIdx_x < num_classes_ && global_i_idx_linear < num_rows_) {
                                                           B_cache[local_id_1][local_id_0] = alpha_ * B_[global_i_idx_linear * num_classes_ + class_block + threadIdx_x];  // SoA
                                                       }
                                                   });

                                                   // calculate intermediate results and store them in local memory
                                                   for (unsigned class_idx = 0; class_idx < THREAD_BLOCK_SIZE; ++class_idx) {
                                                       ::sycl::distribute_items_and_wait(group, [&](::sycl::s_item<2> idx) {
                                                           // cast values to 32-bit unsigned int values to prevent implicit conversions
                                                           const auto local_id_0 = static_cast<unsigned>(idx.get_local_id(group, 0));
                                                           const auto local_id_1 = static_cast<unsigned>(idx.get_local_id(group, 1));

                                                           C_out_cache[local_id_0][(class_idx + local_id_1) % THREAD_BLOCK_SIZE] += temp(idx) * B_cache[local_id_1][(class_idx + local_id_1) % THREAD_BLOCK_SIZE];
                                                       });
                                                   }

                                                   // atomically add the intermediate cached results to the C matrix
                                                   ::sycl::distribute_items_and_wait(group, [&](::sycl::s_item<2> idx) {
                                                       // cast values to 32-bit unsigned int values to prevent implicit conversions
                                                       const auto local_id_0 = static_cast<unsigned>(idx.get_local_id(group, 0));
                                                       const auto local_id_1 = static_cast<unsigned>(idx.get_local_id(group, 1));

                                                       const auto threadIdx_y = static_cast<std::size_t>(idx.get_local_id(group, 1));  // current work-item in work-group y-dimension

                                                       // calculate the indices to access the global data
                                                       const auto global_j_idx = device_row_offset_ + j_idx(idx);

                                                       // atomically add the intermediate cached results to the C matrix
                                                       if (class_block + threadIdx_y < num_classes_ && global_j_idx < num_rows_) {
                                                           detail::atomic_op<real_type>{ C_[global_j_idx * num_classes_ + class_block + threadIdx_y] } += C_out_cache[local_id_0][local_id_1];  // SoA
                                                       }
                                                   });
                                               }
                                           }

                                           // set potential diagonal entries in temp to 0.0 such that we don't apply the main diagonal twice to C
                                           ::sycl::distribute_items_and_wait(group, [&](::sycl::s_item<2> idx) {
                                               // calculate the indices to access the global data
                                               const auto global_i_idx = device_row_offset_ + i_idx(idx);
                                               const auto global_j_idx = device_row_offset_ + j_idx(idx);

                                               if (global_i_idx == global_j_idx) {
                                                   temp(idx) = real_type{ 0.0 };
                                               }
                                           });

                                           //*************************************************************************//
                                           //     calculate C += alpha * temp * B for the LOWER triangular matrix     //
                                           //*************************************************************************//
                                           {
                                               // rename the local memory array
                                               auto &B_cache = cache_one;
                                               auto &C_out_cache = cache_two;

                                               // iterate over all classes using blocking to be able to cache them for faster memory accesses
                                               for (std::size_t class_block = 0; class_block < num_classes_; class_block += static_cast<std::size_t>(THREAD_BLOCK_SIZE)) {
                                                   ::sycl::distribute_items_and_wait(group, [&](::sycl::s_item<2> idx) {
                                                       // cast values to 32-bit unsigned int values to prevent implicit conversions
                                                       const auto local_id_0 = static_cast<unsigned>(idx.get_local_id(group, 0));
                                                       const auto local_id_1 = static_cast<unsigned>(idx.get_local_id(group, 1));

                                                       const auto threadIdx_x = static_cast<std::size_t>(idx.get_local_id(group, 0));  // current work-item in work-group x-dimension

                                                       // calculate the indices to access the global data, pays attention to coalesced memory accesses
                                                       const auto global_j_idx_linear = device_row_offset_ + j_idx_linear(idx);

                                                       // zero-out local memory
                                                       B_cache[local_id_0][local_id_1] = real_type{ 0.0 };
                                                       C_out_cache[local_id_0][local_id_1] = real_type{ 0.0 };

                                                       // load data into local memory
                                                       if (class_block + threadIdx_x < num_classes_ && global_j_idx_linear < num_rows_) {
                                                           B_cache[local_id_0][local_id_1] = alpha_ * B_[global_j_idx_linear * num_classes_ + class_block + threadIdx_x];  // SoA
                                                       }
                                                   });

                                                   // calculate intermediate results and store them in local memory
                                                   for (unsigned class_idx = 0; class_idx < THREAD_BLOCK_SIZE; ++class_idx) {
                                                       ::sycl::distribute_items_and_wait(group, [&](::sycl::s_item<2> idx) {
                                                           // cast values to 32-bit unsigned int values to prevent implicit conversions
                                                           const auto local_id_0 = static_cast<unsigned>(idx.get_local_id(group, 0));
                                                           const auto local_id_1 = static_cast<unsigned>(idx.get_local_id(group, 1));

                                                           C_out_cache[(class_idx + local_id_0) % THREAD_BLOCK_SIZE][local_id_1] += temp(idx) * B_cache[(class_idx + local_id_0) % THREAD_BLOCK_SIZE][local_id_0];
                                                       });
                                                   }

                                                   // atomically add the intermediate cached results to the C matrix
                                                   ::sycl::distribute_items_and_wait(group, [&](::sycl::s_item<2> idx) {
                                                       // cast values to 32-bit unsigned int values to prevent implicit conversions
                                                       const auto local_id_0 = static_cast<unsigned>(idx.get_local_id(group, 0));
                                                       const auto local_id_1 = static_cast<unsigned>(idx.get_local_id(group, 1));

                                                       const auto threadIdx_x = static_cast<std::size_t>(idx.get_local_id(group, 0));  // current work-item in work-group x-dimension

                                                       // calculate the indices to access the global data
                                                       const auto global_i_idx = device_row_offset_ + i_idx(idx);

                                                       // atomically add the intermediate cached results to the C matrix
                                                       if (class_block + threadIdx_x < num_classes_ && global_i_idx < num_rows_) {
                                                           detail::atomic_op<real_type>{ C_[global_i_idx * num_classes_ + class_block + threadIdx_x] } += C_out_cache[local_id_0][local_id_1];  // SoA
                                                       }
                                                   });
                                               }
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

}  // namespace plssvm::sycl::detail::scoped

#endif  // PLSSVM_BACKENDS_SYCL_KERNEL_CG_IMPLICIT_SCOPED_KERNEL_MATRIX_ASSEMBLY_BLAS_HPP_
