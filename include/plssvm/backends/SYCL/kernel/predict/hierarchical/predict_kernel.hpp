/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines the functions used for prediction for the C-SVM using the SYCL backend and the work-group data parallel kernels.
 */

#ifndef PLSSVM_BACKENDS_SYCL_KERNEL_PREDICT_HIERARCHICAL_PREDICT_KERNEL_HPP_
#define PLSSVM_BACKENDS_SYCL_KERNEL_PREDICT_HIERARCHICAL_PREDICT_KERNEL_HPP_
#pragma once

#include "plssvm/backends/SYCL/data_parallel_kernels.hpp"    // plssvm::sycl::data_parallel_kernel
#include "plssvm/backends/SYCL/detail/atomics.hpp"           // plssvm::sycl::detail::atomic_op
#include "plssvm/backends/SYCL/kernel/kernel_functions.hpp"  // plssvm::sycl::detail::{feature_reduce, apply_kernel_function}
#include "plssvm/constants.hpp"                              // plssvm::{real_type, THREAD_BLOCK_SIZE, INTERNAL_BLOCK_SIZE, PADDING_SIZE}
#include "plssvm/kernel_function_types.hpp"                  // plssvm::kernel_function_type
#include "plssvm/target_platforms.hpp"                       // plssvm::target_platform

#include "sycl/sycl.hpp"  // sycl::group, sycl::private_memory, sycl::h_item

#include <cstddef>  // std::size_t
#include <tuple>    // std::tuple, std::make_tuple

namespace plssvm::sycl::detail::hierarchical {

/**
 * @brief Calculate the `q` vector used to speedup the prediction using the linear kernel function.
 * @details Uses SYCL's hierarchical data parallel kernels.
 * @tparam target the target platform
 */
template <target_platform target>
class device_kernel_w_linear {
  public:
    /// The used SYCL data parallel kernel.
    constexpr static sycl::data_parallel_kernel data_parallel_kernel_type = sycl::data_parallel_kernel::hierarchical;

    /**
     * @brief Initialize the SYCL kernel function object.
     * @param[in,out] w the vector to speedup the linear prediction
     * @param[in] alpha the previously learned weights
     * @param[in] support_vectors the support vectors
     * @param[in] num_classes the number of classes
     * @param[in] num_sv the number of support vectors
     * @param[in] device_num_sv the number of support vectors the current device is responsible for
     * @param[in] device_sv_offset the first support vector (row in @p alpha) the current device is responsible for
     * @param[in] grid_x_offset the offset in x-dimension into the data points if more than one execution grid has to be used
     * @param[in] grid_y_offset the offset in y-dimension into the data points if more than one execution grid has to be used
     */
    device_kernel_w_linear(real_type *w, const real_type *alpha, const real_type *support_vectors, const std::size_t num_classes, const std::size_t num_sv, const std::size_t device_num_sv, const std::size_t device_sv_offset, const std::size_t grid_x_offset, const std::size_t grid_y_offset) :
        w_{ w },
        alpha_{ alpha },
        support_vectors_{ support_vectors },
        num_classes_{ num_classes },
        num_sv_{ num_sv },
        device_num_sv_{ device_num_sv },
        device_sv_offset_{ device_sv_offset },
        grid_x_offset_{ grid_x_offset },
        grid_y_offset_{ grid_y_offset } { }

    /**
     * @brief Function call operator overload performing the actual calculation.
     * @param[in] group indices representing the current point in the execution space
     */
    void operator()(::sycl::group<2> group) const {
        // create two local memory arrays used for caching
        real_type feature_cache[THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];
        real_type alpha_cache[THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];

        // create a private memory array used for internal caching
        ::sycl::private_memory<real_type[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE], 2> temp{ group };

        // initialize private temp matrix to zero
        group.parallel_for_work_item([&](::sycl::h_item<2> idx) {
            for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                    temp(idx)[internal_i][internal_j] = real_type{ 0.0 };
                }
            }
        });

        // implicit group barrier

        // iterate over all support vectors using blocking to be able to cache them for faster memory accesses
        for (std::size_t sv_block = 0; sv_block < device_num_sv_; sv_block += static_cast<std::size_t>(THREAD_BLOCK_SIZE)) {
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
                const auto feature_idx_linear = blockIdx_y * blockDim_y * INTERNAL_BLOCK_SIZE_uz + threadIdx_y;  // num_features
                const auto class_idx_linear = blockIdx_x * blockDim_x * INTERNAL_BLOCK_SIZE_uz + threadIdx_y;    // num_classes

                for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                    // calculate the indices to access the global data, pays attention to coalesced memory accesses
                    const auto global_feature_idx_linear = feature_idx_linear + static_cast<std::size_t>(internal) * THREAD_BLOCK_SIZE_uz;
                    const auto global_class_idx_linear = class_idx_linear + static_cast<std::size_t>(internal) * THREAD_BLOCK_SIZE_uz;

                    feature_cache[local_id_0][internal * THREAD_BLOCK_SIZE + local_id_1] = support_vectors_[global_feature_idx_linear * (device_num_sv_ + PADDING_SIZE_uz) + sv_block + threadIdx_x];  // SoA
                    alpha_cache[local_id_0][internal * THREAD_BLOCK_SIZE + local_id_1] = alpha_[global_class_idx_linear * (num_sv_ + PADDING_SIZE_uz) + sv_block + device_sv_offset_ + threadIdx_x];   // AoS
                }
            });

            // implicit group barrier

            // perform the dot product calculation
            group.parallel_for_work_item([&](::sycl::h_item<2> idx) {
                // cast values to 32-bit unsigned int values to prevent implicit conversions
                const auto local_id_0 = static_cast<unsigned>(idx.get_local_id(0));
                const auto local_id_1 = static_cast<unsigned>(idx.get_local_id(1));

                if constexpr (target == target_platform::cpu) {
                    // perform the dot product calculation, the sv is the fastest moving index
                    for (unsigned internal_feature = 0; internal_feature < INTERNAL_BLOCK_SIZE; ++internal_feature) {
                        for (unsigned internal_class = 0; internal_class < INTERNAL_BLOCK_SIZE; ++internal_class) {
                            real_type sum{ 0.0 };
                            for (unsigned sv = 0; sv < THREAD_BLOCK_SIZE; ++sv) {
                                sum += alpha_cache[sv][local_id_0 * INTERNAL_BLOCK_SIZE + internal_class] * feature_cache[sv][local_id_1 * INTERNAL_BLOCK_SIZE + internal_feature];
                            }
                            temp(idx)[internal_feature][internal_class] += sum;
                        }
                    }
                } else {
                    // perform the dot product calculation, the sv is the slowest moving index
                    for (unsigned sv = 0; sv < THREAD_BLOCK_SIZE; ++sv) {
                        for (unsigned internal_feature = 0; internal_feature < INTERNAL_BLOCK_SIZE; ++internal_feature) {
                            for (unsigned internal_class = 0; internal_class < INTERNAL_BLOCK_SIZE; ++internal_class) {
                                temp(idx)[internal_feature][internal_class] += alpha_cache[sv][local_id_0 * INTERNAL_BLOCK_SIZE + internal_class] * feature_cache[sv][local_id_1 * INTERNAL_BLOCK_SIZE + internal_feature];
                            }
                        }
                    }
                }
            });

            // implicit group barrier
        }

        // update the global w-vector with the locally cached values
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
            const auto feature_idx = (blockIdx_y * blockDim_y + threadIdx_y) * INTERNAL_BLOCK_SIZE_uz;  // num_features
            const auto class_idx = (blockIdx_x * blockDim_x + threadIdx_x) * INTERNAL_BLOCK_SIZE_uz;    // num_classes

            for (unsigned internal_feature = 0; internal_feature < INTERNAL_BLOCK_SIZE; ++internal_feature) {
                for (unsigned internal_class = 0; internal_class < INTERNAL_BLOCK_SIZE; ++internal_class) {
                    // calculate the indices to access the global data
                    const auto global_feature_idx = feature_idx + static_cast<std::size_t>(internal_feature);
                    const auto global_class_idx = class_idx + static_cast<std::size_t>(internal_class);

                    w_[global_feature_idx * (num_classes_ + PADDING_SIZE_uz) + global_class_idx] = temp(idx)[internal_feature][internal_class];  // SoA
                }
            }
        });
    }

  private:
    /// @cond Doxygen_suppress
    real_type *w_;
    const real_type *alpha_;
    const real_type *support_vectors_;
    const std::size_t num_classes_;
    const std::size_t num_sv_;
    const std::size_t device_num_sv_;
    const std::size_t device_sv_offset_;
    const std::size_t grid_x_offset_;
    const std::size_t grid_y_offset_;
    /// @endcond
};

/**
 * @brief Predict the @p predict_points using the linear kernel speeding up the calculation using the @p w vector.
 * @details Uses SYCL's hierarchical data parallel kernels.
 * @tparam target the target platform
 */
template <target_platform target>
class device_kernel_predict_linear {
  public:
    /// The used SYCL data parallel kernel.
    constexpr static sycl::data_parallel_kernel data_parallel_kernel_type = sycl::data_parallel_kernel::hierarchical;

    /**
     * @brief Initialize the SYCL kernel function object.
     * @param[out] prediction the predicted values
     * @param[in] w the vector to speedup the calculations
     * @param[in] rho the previously learned bias
     * @param[in] predict_points the data points to predict
     * @param[in] num_classes the number of classes
     * @param[in] num_predict_points the number of data points to predict
     * @param[in] num_features the number of features per data point
     * @param[in] grid_x_offset the offset in x-dimension into the data points if more than one execution grid has to be used
     * @param[in] grid_y_offset the offset in y-dimension into the data points if more than one execution grid has to be used
     */
    device_kernel_predict_linear(real_type *prediction, const real_type *w, const real_type *rho, const real_type *predict_points, const std::size_t num_classes, const std::size_t num_predict_points, const std::size_t num_features, const std::size_t grid_x_offset, const std::size_t grid_y_offset) :
        prediction_{ prediction },
        w_{ w },
        rho_{ rho },
        predict_points_{ predict_points },
        num_classes_{ num_classes },
        num_predict_points_{ num_predict_points },
        num_features_{ num_features },
        grid_x_offset_{ grid_x_offset },
        grid_y_offset_{ grid_y_offset } { }

    /**
     * @brief Function call operator overload performing the actual calculation.
     * @param[in] group indices representing the current point in the execution space
     */
    void operator()(::sycl::group<2> group) const {
        // create two local memory arrays used for caching
        real_type pp_cache[THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];
        real_type w_cache[THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];

        // create a private memory array used for internal caching
        ::sycl::private_memory<real_type[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE], 2> temp{ group };

        // initialize private variable
        group.parallel_for_work_item([&](::sycl::h_item<2> idx) {
            // initialize private temp matrix to zero
            for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                    temp(idx)[internal_i][internal_j] = real_type{ 0.0 };
                }
            }
        });

        // implicit group barrier

        // iterate over all features using blocking to be able to cache them for faster memory accesses
        for (std::size_t feature_block = 0; feature_block < num_features_; feature_block += static_cast<std::size_t>(THREAD_BLOCK_SIZE)) {
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

                // calculate the indices used in the current thread, pays attention to coalesced memory accesses
                const auto pp_idx_linear = blockIdx_y * blockDim_y * INTERNAL_BLOCK_SIZE_uz + threadIdx_y;     // num_predict_points
                const auto class_idx_linear = blockIdx_x * blockDim_x * INTERNAL_BLOCK_SIZE_uz + threadIdx_y;  // num_classes

                // load data into local memory
                for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                    // calculate the indices to access the global data, pays attention to coalesced memory accesses
                    const auto global_pp_idx_linear = pp_idx_linear + static_cast<std::size_t>(internal) * THREAD_BLOCK_SIZE_uz;
                    const auto global_class_idx_linear = class_idx_linear + static_cast<std::size_t>(internal) * THREAD_BLOCK_SIZE_uz;

                    // store the values in the local memory
                    pp_cache[local_id_0][internal * THREAD_BLOCK_SIZE + local_id_1] = predict_points_[(feature_block + threadIdx_x) * (num_predict_points_ + PADDING_SIZE_uz) + global_pp_idx_linear];  // SoA
                    w_cache[local_id_0][internal * THREAD_BLOCK_SIZE + local_id_1] = w_[(feature_block + threadIdx_x) * (num_classes_ + PADDING_SIZE_uz) + global_class_idx_linear];                    // SoA
                }
            });

            // implicit group barrier

            // perform the dot product calculation
            group.parallel_for_work_item([&](::sycl::h_item<2> idx) {
                // cast values to 32-bit unsigned int values to prevent implicit conversions
                const auto local_id_0 = static_cast<unsigned>(idx.get_local_id(0));
                const auto local_id_1 = static_cast<unsigned>(idx.get_local_id(1));

                if constexpr (target == target_platform::cpu) {
                    // perform the dot product calculation, the feature is the fastest moving index
                    for (unsigned internal_pp = 0; internal_pp < INTERNAL_BLOCK_SIZE; ++internal_pp) {
                        for (unsigned internal_class = 0; internal_class < INTERNAL_BLOCK_SIZE; ++internal_class) {
                            real_type sum{ 0.0 };
                            for (unsigned feature = 0; feature < THREAD_BLOCK_SIZE; ++feature) {
                                sum += w_cache[feature][local_id_0 * INTERNAL_BLOCK_SIZE + internal_class] * pp_cache[feature][local_id_1 * INTERNAL_BLOCK_SIZE + internal_pp];
                            }
                            temp(idx)[internal_pp][internal_class] += sum;
                        }
                    }
                } else {
                    // perform the dot product calculation, the feature is the slowest moving index
                    for (unsigned feature = 0; feature < THREAD_BLOCK_SIZE; ++feature) {
                        for (unsigned internal_pp = 0; internal_pp < INTERNAL_BLOCK_SIZE; ++internal_pp) {
                            for (unsigned internal_class = 0; internal_class < INTERNAL_BLOCK_SIZE; ++internal_class) {
                                temp(idx)[internal_pp][internal_class] += w_cache[feature][local_id_0 * INTERNAL_BLOCK_SIZE + internal_class] * pp_cache[feature][local_id_1 * INTERNAL_BLOCK_SIZE + internal_pp];
                            }
                        }
                    }
                }
            });

            // implicit group barrier
        }

        // update the global array with the local one
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
            const auto pp_idx = (blockIdx_y * blockDim_y + threadIdx_y) * INTERNAL_BLOCK_SIZE_uz;     // num_predict_points
            const auto class_idx = (blockIdx_x * blockDim_x + threadIdx_x) * INTERNAL_BLOCK_SIZE_uz;  // num_classes

            for (unsigned internal_pp = 0; internal_pp < INTERNAL_BLOCK_SIZE; ++internal_pp) {
                for (unsigned internal_class = 0; internal_class < INTERNAL_BLOCK_SIZE; ++internal_class) {
                    // calculate the indices to access the global data
                    const auto global_pp_idx = pp_idx + static_cast<std::size_t>(internal_pp);
                    const auto global_class_idx = class_idx + static_cast<std::size_t>(internal_class);

                    prediction_[global_pp_idx * (num_classes_ + PADDING_SIZE_uz) + global_class_idx] = temp(idx)[internal_pp][internal_class] - rho_[global_class_idx];  // AoS
                }
            }
        });
    }

  private:
    /// @cond Doxygen_suppress
    real_type *prediction_;
    const real_type *w_;
    const real_type *rho_;
    const real_type *predict_points_;
    const std::size_t num_classes_;
    const std::size_t num_predict_points_;
    const std::size_t num_features_;
    const std::size_t grid_x_offset_;
    const std::size_t grid_y_offset_;
    /// @endcond
};

/**
 * @brief Predict the @p predict_points using the @p kernel_function.
 * @details Uses SYCL's hierarchical data parallel kernels.
 * @tparam target the target platform
 * @tparam kernel_function the type of the used kernel function
 * @tparam Args the types of the parameters necessary for the specific kernel function; stored in a `std::tuple`
 */
template <target_platform target, kernel_function_type kernel_function, typename... Args>
class device_kernel_predict {
  public:
    /// The used SYCL data parallel kernel.
    constexpr static sycl::data_parallel_kernel data_parallel_kernel_type = sycl::data_parallel_kernel::hierarchical;

    /**
     * @brief Initialize the SYCL kernel function object.
     * @param[in] prediction the predicted values
     * @param[in] alpha the previously learned weights
     * @param[in] rho the previously learned biases
     * @param[in] support_vectors the support vectors
     * @param[in] predict_points the data points to predict
     * @param[in] num_classes the number of classes
     * @param[in] num_sv the number of support vectors
     * @param[in] num_predict_points the number of data points to predict
     * @param[in] num_features the number of features per data point
     * @param[in] grid_x_offset the offset in x-dimension into the data points if more than one execution grid has to be used
     * @param[in] grid_y_offset the offset in y-dimension into the data points if more than one execution grid has to be used
     * @param[in] kernel_function_parameter the parameters necessary to apply the @p kernel_function
     */
    device_kernel_predict(real_type *prediction, const real_type *alpha, const real_type *rho, const real_type *support_vectors, const real_type *predict_points, const std::size_t num_classes, const std::size_t num_sv, const std::size_t num_predict_points, const std::size_t num_features, const std::size_t grid_x_offset, const std::size_t grid_y_offset, Args... kernel_function_parameter) :
        prediction_{ prediction },
        alpha_{ alpha },
        rho_{ rho },
        support_vectors_{ support_vectors },
        predict_points_{ predict_points },
        num_classes_{ num_classes },
        num_sv_{ num_sv },
        num_predict_points_{ num_predict_points },
        num_features_{ num_features },
        grid_x_offset_{ grid_x_offset },
        grid_y_offset_{ grid_y_offset },
        kernel_function_parameter_{ std::make_tuple(kernel_function_parameter...) } { }

    /**
     * @brief Function call operator overload performing the actual calculation.
     * @param[in] group indices representing the current point in the execution space
     */
    void operator()(::sycl::group<2> group) const {
        // create two local memory arrays used for caching
        real_type cache_one[THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];
        real_type cache_two[THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];

        // create a private memory array used for internal caching
        ::sycl::private_memory<real_type[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE], 2> temp{ group };

        // initialize private variable
        group.parallel_for_work_item([&](::sycl::h_item<2> idx) {
            // initialize private temp matrix to zero
            for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                    temp(idx)[internal_i][internal_j] = real_type{ 0.0 };
                }
            }
        });

        // implicit group barrier

        {
            // rename cached arrays -> not possible due to an AdaptiveCpp runtime exception
            // auto &pp_cache = cache_one;
            // auto &sv_cache = cache_two;

            // iterate over all features using blocking to be able to cache them for faster memory accesses
            for (std::size_t feature_block = 0; feature_block < num_features_; feature_block += static_cast<std::size_t>(THREAD_BLOCK_SIZE)) {
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

                    // calculate the indices used in the current thread, pays attention to coalesced memory accesses
                    const auto pp_idx_linear = blockIdx_y * blockDim_y * INTERNAL_BLOCK_SIZE_uz + threadIdx_y;  // num_predict_points
                    const auto sv_idx_linear = blockIdx_x * blockDim_x * INTERNAL_BLOCK_SIZE_uz + threadIdx_y;  // num_support_vectors

                    // load data into local memory
                    for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                        // calculate the indices to access the global data, pays attention to coalesced memory accesses
                        const auto global_pp_idx_linear = pp_idx_linear + static_cast<std::size_t>(internal) * THREAD_BLOCK_SIZE_uz;
                        const auto global_sv_idx_linear = sv_idx_linear + static_cast<std::size_t>(internal) * THREAD_BLOCK_SIZE_uz;

                        // store the values in the local memory
                        cache_one[local_id_0][internal * THREAD_BLOCK_SIZE + local_id_1] = predict_points_[(feature_block + threadIdx_x) * (num_predict_points_ + PADDING_SIZE_uz) + global_pp_idx_linear];
                        cache_two[local_id_0][internal * THREAD_BLOCK_SIZE + local_id_1] = support_vectors_[(feature_block + threadIdx_x) * (num_sv_ + PADDING_SIZE_uz) + global_sv_idx_linear];
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
                        for (unsigned internal_pp = 0; internal_pp < INTERNAL_BLOCK_SIZE; ++internal_pp) {
                            for (unsigned internal_sv = 0; internal_sv < INTERNAL_BLOCK_SIZE; ++internal_sv) {
                                real_type sum{ 0.0 };
                                for (unsigned feature = 0; feature < THREAD_BLOCK_SIZE; ++feature) {
                                    sum += detail::feature_reduce<kernel_function>(cache_two[feature][local_id_0 * INTERNAL_BLOCK_SIZE + internal_sv],
                                                                                   cache_one[feature][local_id_1 * INTERNAL_BLOCK_SIZE + internal_pp]);
                                }
                                temp(idx)[internal_pp][internal_sv] += sum;
                            }
                        }
                    } else {
                        // perform the feature reduction calculation, the feature is the slowest moving index
                        for (unsigned feature = 0; feature < THREAD_BLOCK_SIZE; ++feature) {
                            for (unsigned internal_pp = 0; internal_pp < INTERNAL_BLOCK_SIZE; ++internal_pp) {
                                for (unsigned internal_sv = 0; internal_sv < INTERNAL_BLOCK_SIZE; ++internal_sv) {
                                    temp(idx)[internal_pp][internal_sv] += detail::feature_reduce<kernel_function>(cache_two[feature][local_id_0 * INTERNAL_BLOCK_SIZE + internal_sv],
                                                                                                                   cache_one[feature][local_id_1 * INTERNAL_BLOCK_SIZE + internal_pp]);
                                }
                            }
                        }
                    }
                });

                // implicit group barrier
            }
        }

        // update temp using the respective kernel function
        group.parallel_for_work_item([&](::sycl::h_item<2> idx) {
            for (unsigned internal_pp = 0; internal_pp < INTERNAL_BLOCK_SIZE; ++internal_pp) {
                for (unsigned internal_sv = 0; internal_sv < INTERNAL_BLOCK_SIZE; ++internal_sv) {
                    temp(idx)[internal_pp][internal_sv] = detail::apply_kernel_function<kernel_function>(temp(idx)[internal_pp][internal_sv], kernel_function_parameter_);
                }
            }
        });

        // implicit group barrier

        {
            // rename cached arrays -> not possible due to an AdaptiveCpp runtime exception
            // auto &alpha_cache = cache_one;
            // auto &out_cache = cache_two;

            // iterate over all classes using blocking to be able to cache them for faster memory accesses
            for (std::size_t class_block = 0; class_block < num_classes_; class_block += static_cast<std::size_t>(THREAD_BLOCK_SIZE)) {
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
                    const auto blockIdx_x = static_cast<std::size_t>(group[0]) + grid_x_offset_;  // current work-group in global range x-dimension + offsets if the global range is too large

                    // calculate the indices used in the current thread, pays attention to coalesced memory accesses
                    const auto sv_idx_linear = blockIdx_x * blockDim_x * INTERNAL_BLOCK_SIZE_uz + threadIdx_y;  // num_support_vectors

                    for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                        // calculate the indices to access the global data, pays attention to coalesced memory accesses
                        const auto global_sv_idx_linear = sv_idx_linear + static_cast<std::size_t>(internal) * THREAD_BLOCK_SIZE_uz;

                        // store the values in the local memory
                        cache_one[local_id_0][internal * THREAD_BLOCK_SIZE + local_id_1] = alpha_[(class_block + threadIdx_x) * (num_sv_ + PADDING_SIZE_uz) + global_sv_idx_linear];  // AoS
                        // the bias (rho) must only be applied once for all support vectors
                        if (blockIdx_x == std::size_t{ 0 }) {
                            cache_two[local_id_0][internal * THREAD_BLOCK_SIZE + local_id_1] = -rho_[class_block + threadIdx_x];
                        } else {
                            cache_two[local_id_0][internal * THREAD_BLOCK_SIZE + local_id_1] = real_type{ 0.0 };
                        }
                    }
                });

                // implicit group barrier

                // calculate intermediate results and store them in local memory
                for (unsigned class_idx = 0; class_idx < THREAD_BLOCK_SIZE; ++class_idx) {
                    group.parallel_for_work_item([&](::sycl::h_item<2> idx) {
                        // cast values to 32-bit unsigned int values to prevent implicit conversions
                        const auto local_id_0 = static_cast<unsigned>(idx.get_local_id(0));
                        const auto local_id_1 = static_cast<unsigned>(idx.get_local_id(1));

                        for (unsigned internal_pp = 0; internal_pp < INTERNAL_BLOCK_SIZE; ++internal_pp) {
                            for (unsigned internal_sv = 0; internal_sv < INTERNAL_BLOCK_SIZE; ++internal_sv) {
                                cache_two[(class_idx + local_id_0) % THREAD_BLOCK_SIZE][internal_pp * THREAD_BLOCK_SIZE + local_id_1] +=
                                    temp(idx)[internal_pp][internal_sv] * cache_one[(class_idx + local_id_0) % THREAD_BLOCK_SIZE][local_id_0 * INTERNAL_BLOCK_SIZE + internal_sv];
                            }
                        }
                    });

                    // implicit group barrier
                }

                // atomically add the intermediate cached results to the prediction
                group.parallel_for_work_item([&](::sycl::h_item<2> idx) {
                    // cast values to 32-bit unsigned int values to prevent implicit conversions
                    const auto local_id_0 = static_cast<unsigned>(idx.get_local_id(0));
                    const auto local_id_1 = static_cast<unsigned>(idx.get_local_id(1));

                    // cast all values to 64-bit std::size_t to prevent potential 32-bit overflows
                    constexpr auto INTERNAL_BLOCK_SIZE_uz = static_cast<std::size_t>(INTERNAL_BLOCK_SIZE);
                    constexpr auto PADDING_SIZE_uz = static_cast<std::size_t>(PADDING_SIZE);

                    const auto threadIdx_x = static_cast<std::size_t>(idx.get_local_id(0));       // current work-item in work-group x-dimension
                    const auto threadIdx_y = static_cast<std::size_t>(idx.get_local_id(1));       // current work-item in work-group y-dimension
                    const auto blockDim_y = static_cast<std::size_t>(idx.get_local_range(1));     // number of work-items in work-group y-dimension
                    const auto blockIdx_y = static_cast<std::size_t>(group[1]) + grid_y_offset_;  // current work-group in global range y-dimension + offsets if the global range is too large

                    // calculate the indices used in the current thread
                    const auto pp_idx = (blockIdx_y * blockDim_y + threadIdx_y) * INTERNAL_BLOCK_SIZE_uz;  // num_predict_points

                    for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                        // calculate the indices to access the global data
                        const auto global_pp_idx = pp_idx + static_cast<std::size_t>(internal);

                        detail::atomic_op<real_type>{ prediction_[global_pp_idx * (num_classes_ + PADDING_SIZE_uz) + class_block + threadIdx_x] } += cache_two[local_id_0][internal * THREAD_BLOCK_SIZE + local_id_1];
                    }
                });

                // implicit group barrier
            }
        }
    }

  private:
    /// @cond Doxygen_suppress
    real_type *prediction_;
    const real_type *alpha_;
    const real_type *rho_;
    const real_type *support_vectors_;
    const real_type *predict_points_;
    const std::size_t num_classes_;
    const std::size_t num_sv_;
    const std::size_t num_predict_points_;
    const std::size_t num_features_;
    const std::size_t grid_x_offset_;
    const std::size_t grid_y_offset_;
    const std::tuple<Args...> kernel_function_parameter_;
    /// @endcond
};

}  // namespace plssvm::sycl::detail::hierarchical

#endif  // PLSSVM_BACKENDS_SYCL_KERNEL_PREDICT_HIERARCHICAL_PREDICT_KERNEL_HPP_
