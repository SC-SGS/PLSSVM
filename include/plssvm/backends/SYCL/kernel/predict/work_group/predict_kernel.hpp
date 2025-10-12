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

#ifndef PLSSVM_BACKENDS_SYCL_KERNEL_PREDICT_WORK_GROUP_PREDICT_KERNEL_HPP_
#define PLSSVM_BACKENDS_SYCL_KERNEL_PREDICT_WORK_GROUP_PREDICT_KERNEL_HPP_
#pragma once

#include "plssvm/backends/SYCL/data_parallel_kernels.hpp"    // plssvm::sycl::data_parallel_kernel
#include "plssvm/backends/SYCL/detail/atomics.hpp"           // plssvm::sycl::detail::atomic_op
#include "plssvm/backends/SYCL/kernel/kernel_functions.hpp"  // plssvm::sycl::detail::{feature_reduce, apply_kernel_function}
#include "plssvm/constants.hpp"                              // plssvm::real_type
#include "plssvm/kernel_function_types.hpp"                  // plssvm::kernel_function_type

#include "sycl/sycl.hpp"  // sycl::handler, sycl::range, sycl::nd_item

#include <cstddef>  // std::size_t
#include <tuple>    // std::tuple, std::make_tuple

namespace plssvm::sycl::detail::work_group {

/**
 * @brief Calculate the `w` vector used to speedup the prediction using the linear kernel function.
 * @details Uses SYCL's work-group data parallel kernels.
 */
class device_kernel_w_linear {
  public:
    /// The used SYCL data parallel kernel.
    constexpr static sycl::data_parallel_kernel data_parallel_kernel_type = sycl::data_parallel_kernel::work_group;

    /**
     * @brief Initialize the SYCL kernel function object.
     * @param[in] cgh the SYCL handler used to allocate the local memory
     * @param[in,out] w the vector to speedup the linear prediction
     * @param[in] alpha the previously learned weights
     * @param[in] support_vectors the support vectors
     * @param[in] num_features the number of features
     * @param[in] num_classes the number of classes
     * @param[in] num_sv the number of support vectors
     * @param[in] device_num_sv the number of support vectors the current device is responsible for
     * @param[in] device_sv_offset the first support vector (row in @p alpha) the current device is responsible for
     * @param[in] grid_x_offset the offset in x-dimension into the data points if more than one execution grid has to be used
     * @param[in] grid_y_offset the offset in y-dimension into the data points if more than one execution grid has to be used
     */
    device_kernel_w_linear(::sycl::handler &cgh, real_type *w, const real_type *alpha, const real_type *support_vectors, const std::size_t num_features, const std::size_t num_classes, const std::size_t num_sv, const std::size_t device_num_sv, const std::size_t device_sv_offset, const std::size_t grid_x_offset, const std::size_t grid_y_offset) :
        feature_cache_{ ::sycl::range<2>{ static_cast<std::size_t>(THREAD_BLOCK_SIZE), static_cast<std::size_t>(INTERNAL_BLOCK_SIZE) * static_cast<std::size_t>(THREAD_BLOCK_SIZE) }, cgh },
        alpha_cache_{ ::sycl::range<2>{ static_cast<std::size_t>(THREAD_BLOCK_SIZE), static_cast<std::size_t>(INTERNAL_BLOCK_SIZE) * static_cast<std::size_t>(THREAD_BLOCK_SIZE) }, cgh },
        w_{ w },
        alpha_{ alpha },
        support_vectors_{ support_vectors },
        num_features_{ num_features },
        num_classes_{ num_classes },
        num_sv_{ num_sv },
        device_num_sv_{ device_num_sv },
        device_sv_offset_{ device_sv_offset },
        grid_x_offset_{ grid_x_offset },
        grid_y_offset_{ grid_y_offset } { }

    /**
     * @brief Function call operator overload performing the actual calculation.
     * @param[in] nd_idx indices representing the current point in the execution space
     */
    void operator()(::sycl::nd_item<2> nd_idx) const {
        // cast values to 32-bit unsigned int values to prevent implicit conversions
        const auto local_id_0 = static_cast<unsigned>(nd_idx.get_local_id(0));
        const auto local_id_1 = static_cast<unsigned>(nd_idx.get_local_id(1));

        // cast all values to 64-bit std::size_t to prevent potential 32-bit overflows
        constexpr auto INTERNAL_BLOCK_SIZE_uz = static_cast<std::size_t>(INTERNAL_BLOCK_SIZE);
        constexpr auto THREAD_BLOCK_SIZE_uz = static_cast<std::size_t>(THREAD_BLOCK_SIZE);

        const auto threadIdx_x = static_cast<std::size_t>(nd_idx.get_local_id(0));               // current work-item in work-group x-dimension
        const auto threadIdx_y = static_cast<std::size_t>(nd_idx.get_local_id(1));               // current work-item in work-group y-dimension
        const auto blockDim_x = static_cast<std::size_t>(nd_idx.get_local_range(0));             // number of work-items in work-group x-dimension
        const auto blockDim_y = static_cast<std::size_t>(nd_idx.get_local_range(1));             // number of work-items in work-group y-dimension
        const auto blockIdx_x = static_cast<std::size_t>(nd_idx.get_group(0)) + grid_x_offset_;  // current work-group in global range x-dimension + offsets if the global range is too large
        const auto blockIdx_y = static_cast<std::size_t>(nd_idx.get_group(1)) + grid_y_offset_;  // current work-group in global range y-dimension + offsets if the global range is too large

        // create a work-item private array used for internal caching
        real_type temp[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE]{};

        {
            // calculate the indices used in the current work-item, pays attention to coalesced memory accesses
            const auto feature_idx_linear = blockIdx_y * blockDim_y * INTERNAL_BLOCK_SIZE_uz + threadIdx_y;  // num_features
            const auto class_idx_linear = blockIdx_x * blockDim_x * INTERNAL_BLOCK_SIZE_uz + threadIdx_y;    // num_classes

            // iterate over all support vectors using blocking to be able to cache them for faster memory accesses
            for (std::size_t sv_block = 0; sv_block < device_num_sv_; sv_block += THREAD_BLOCK_SIZE_uz) {
                // zero-out local memory
                for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                    feature_cache_[local_id_0][internal * THREAD_BLOCK_SIZE + local_id_1] = real_type{ 0.0 };
                    alpha_cache_[local_id_0][internal * THREAD_BLOCK_SIZE + local_id_1] = real_type{ 0.0 };
                }

                // load data into local memory
                for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                    // calculate the indices to access the global data, pays attention to coalesced memory accesses
                    const auto global_feature_idx_linear = feature_idx_linear + static_cast<std::size_t>(internal) * THREAD_BLOCK_SIZE_uz;
                    const auto global_class_idx_linear = class_idx_linear + static_cast<std::size_t>(internal) * THREAD_BLOCK_SIZE_uz;

                    // store the values in the local memory
                    if (sv_block + threadIdx_x < device_num_sv_) {
                        if (global_feature_idx_linear < num_features_) {
                            feature_cache_[local_id_0][internal * THREAD_BLOCK_SIZE + local_id_1] = support_vectors_[global_feature_idx_linear * device_num_sv_ + sv_block + threadIdx_x];  // SoA
                        }
                        if (global_class_idx_linear < num_classes_) {
                            alpha_cache_[local_id_0][internal * THREAD_BLOCK_SIZE + local_id_1] = alpha_[global_class_idx_linear * num_sv_ + sv_block + device_sv_offset_ + threadIdx_x];  // AoS
                        }
                    }
                }
                nd_idx.barrier();  // wait until all work-items loaded their part of the data

                // perform the dot product calculation
                for (unsigned sv = 0; sv < THREAD_BLOCK_SIZE; ++sv) {
                    for (unsigned internal_feature = 0; internal_feature < INTERNAL_BLOCK_SIZE; ++internal_feature) {
                        for (unsigned internal_class = 0; internal_class < INTERNAL_BLOCK_SIZE; ++internal_class) {
                            temp[internal_feature][internal_class] += alpha_cache_[sv][local_id_0 * INTERNAL_BLOCK_SIZE + internal_class] * feature_cache_[sv][local_id_1 * INTERNAL_BLOCK_SIZE + internal_feature];
                        }
                    }
                }
                nd_idx.barrier();  // wait until all work-items performed their part of the calculations
            }
        }

        // calculate the indices used in the current work-item
        const auto feature_idx = (blockIdx_y * blockDim_y + threadIdx_y) * INTERNAL_BLOCK_SIZE_uz;  // num_features
        const auto class_idx = (blockIdx_x * blockDim_x + threadIdx_x) * INTERNAL_BLOCK_SIZE_uz;    // num_classes

        // update the global w-vector with the locally cached values
        for (unsigned internal_feature = 0; internal_feature < INTERNAL_BLOCK_SIZE; ++internal_feature) {
            for (unsigned internal_class = 0; internal_class < INTERNAL_BLOCK_SIZE; ++internal_class) {
                // calculate the indices to access the global data
                const auto global_feature_idx = feature_idx + static_cast<std::size_t>(internal_feature);
                const auto global_class_idx = class_idx + static_cast<std::size_t>(internal_class);

                if (global_feature_idx < num_features_ && global_class_idx < num_classes_) {
                    w_[global_feature_idx * num_classes_ + global_class_idx] = temp[internal_feature][internal_class];  // SoA
                }
            }
        }
    }

  private:
    /// Local memory used for internal memory access optimizations.
    ::sycl::local_accessor<real_type, 2> feature_cache_;
    /// Local memory used for internal memory access optimizations.
    ::sycl::local_accessor<real_type, 2> alpha_cache_;

    /// @cond Doxygen_suppress
    real_type *w_;
    const real_type *alpha_;
    const real_type *support_vectors_;
    const std::size_t num_features_;
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
 * @details Uses SYCL's work-group data parallel kernels.
 */
class device_kernel_predict_linear {
  public:
    /// The used SYCL data parallel kernel.
    constexpr static sycl::data_parallel_kernel data_parallel_kernel_type = sycl::data_parallel_kernel::work_group;

    /**
     * @brief Initialize the SYCL kernel function object.
     * @param[in] cgh the SYCL handler used to allocate the local memory
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
    device_kernel_predict_linear(::sycl::handler &cgh, real_type *prediction, const real_type *w, const real_type *rho, const real_type *predict_points, const std::size_t num_classes, const std::size_t num_predict_points, const std::size_t num_features, const std::size_t grid_x_offset, const std::size_t grid_y_offset) :
        pp_cache_{ ::sycl::range<2>{ static_cast<std::size_t>(THREAD_BLOCK_SIZE), static_cast<std::size_t>(INTERNAL_BLOCK_SIZE) * static_cast<std::size_t>(THREAD_BLOCK_SIZE) }, cgh },
        w_cache_{ ::sycl::range<2>{ static_cast<std::size_t>(THREAD_BLOCK_SIZE), static_cast<std::size_t>(INTERNAL_BLOCK_SIZE) * static_cast<std::size_t>(THREAD_BLOCK_SIZE) }, cgh },
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
     * @param[in] nd_idx indices representing the current point in the execution space
     */
    void operator()(::sycl::nd_item<2> nd_idx) const {
        // cast values to 32-bit unsigned int values to prevent implicit conversions
        const auto local_id_0 = static_cast<unsigned>(nd_idx.get_local_id(0));
        const auto local_id_1 = static_cast<unsigned>(nd_idx.get_local_id(1));

        // cast all values to 64-bit std::size_t to prevent potential 32-bit overflows
        constexpr auto INTERNAL_BLOCK_SIZE_uz = static_cast<std::size_t>(INTERNAL_BLOCK_SIZE);
        constexpr auto THREAD_BLOCK_SIZE_uz = static_cast<std::size_t>(THREAD_BLOCK_SIZE);

        const auto threadIdx_x = static_cast<std::size_t>(nd_idx.get_local_id(0));               // current work-item in work-group x-dimension
        const auto threadIdx_y = static_cast<std::size_t>(nd_idx.get_local_id(1));               // current work-item in work-group y-dimension
        const auto blockDim_x = static_cast<std::size_t>(nd_idx.get_local_range(0));             // number of work-items in work-group x-dimension
        const auto blockDim_y = static_cast<std::size_t>(nd_idx.get_local_range(1));             // number of work-items in work-group y-dimension
        const auto blockIdx_x = static_cast<std::size_t>(nd_idx.get_group(0)) + grid_x_offset_;  // current work-group in global range x-dimension + offsets if the global range is too large
        const auto blockIdx_y = static_cast<std::size_t>(nd_idx.get_group(1)) + grid_y_offset_;  // current work-group in global range y-dimension + offsets if the global range is too large

        // create a work-item private array used for internal caching
        real_type temp[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE]{};

        {
            // calculate the indices used in the current thread, pays attention to coalesced memory accesses
            const auto pp_idx_linear = blockIdx_y * blockDim_y * INTERNAL_BLOCK_SIZE_uz + threadIdx_y;     // num_predict_points
            const auto class_idx_linear = blockIdx_x * blockDim_x * INTERNAL_BLOCK_SIZE_uz + threadIdx_y;  // num_classes

            // iterate over all features using blocking to be able to cache them for faster memory accesses
            for (std::size_t feature_block = 0; feature_block < num_features_; feature_block += THREAD_BLOCK_SIZE_uz) {
                // zero-out local memory
                for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                    pp_cache_[local_id_0][internal * THREAD_BLOCK_SIZE + local_id_1] = real_type{ 0.0 };
                    w_cache_[local_id_0][internal * THREAD_BLOCK_SIZE + local_id_1] = real_type{ 0.0 };
                }

                // load data into local memory
                for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                    // calculate the indices to access the global data, pays attention to coalesced memory accesses
                    const auto global_pp_idx_linear = pp_idx_linear + static_cast<std::size_t>(internal) * THREAD_BLOCK_SIZE_uz;
                    const auto global_class_idx_linear = class_idx_linear + static_cast<std::size_t>(internal) * THREAD_BLOCK_SIZE_uz;

                    // store the values in the local memory
                    if (feature_block + threadIdx_x < num_features_) {
                        if (global_pp_idx_linear < num_predict_points_) {
                            pp_cache_[local_id_0][internal * THREAD_BLOCK_SIZE + local_id_1] = predict_points_[(feature_block + threadIdx_x) * num_predict_points_ + global_pp_idx_linear];  // SoA
                        }
                        if (global_class_idx_linear < num_classes_) {
                            w_cache_[local_id_0][internal * THREAD_BLOCK_SIZE + local_id_1] = w_[(feature_block + threadIdx_x) * num_classes_ + global_class_idx_linear];  // SoA
                        }
                    }
                }
                nd_idx.barrier();  // wait until all work-items loaded their part of the data

                // perform the dot product calculation
                for (unsigned feature = 0; feature < THREAD_BLOCK_SIZE; ++feature) {
                    for (unsigned internal_pp = 0; internal_pp < INTERNAL_BLOCK_SIZE; ++internal_pp) {
                        for (unsigned internal_class = 0; internal_class < INTERNAL_BLOCK_SIZE; ++internal_class) {
                            temp[internal_pp][internal_class] += w_cache_[feature][local_id_0 * INTERNAL_BLOCK_SIZE + internal_class] * pp_cache_[feature][local_id_1 * INTERNAL_BLOCK_SIZE + internal_pp];
                        }
                    }
                }
                nd_idx.barrier();  // wait until all work-items performed their part of the calculations
            }
        }

        // calculate the indices used in the current work-item
        const auto pp_idx = (blockIdx_y * blockDim_y + threadIdx_y) * INTERNAL_BLOCK_SIZE_uz;     // num_predict_points
        const auto class_idx = (blockIdx_x * blockDim_x + threadIdx_x) * INTERNAL_BLOCK_SIZE_uz;  // num_classes

        // update the global array with the local one
        for (unsigned internal_pp = 0; internal_pp < INTERNAL_BLOCK_SIZE; ++internal_pp) {
            for (unsigned internal_class = 0; internal_class < INTERNAL_BLOCK_SIZE; ++internal_class) {
                // calculate the indices to access the global data
                const auto global_pp_idx = pp_idx + static_cast<std::size_t>(internal_pp);
                const auto global_class_idx = class_idx + static_cast<std::size_t>(internal_class);

                // be sure to not perform out-of-bounds accesses
                if (global_pp_idx < num_predict_points_ && global_class_idx < num_classes_) {
                    prediction_[global_pp_idx * num_classes_ + global_class_idx] = temp[internal_pp][internal_class] - rho_[global_class_idx];  // AoS
                }
            }
        }
    }

  private:
    /// Local memory used for internal memory access optimizations.
    ::sycl::local_accessor<real_type, 2> pp_cache_;
    /// Local memory used for internal memory access optimizations.
    ::sycl::local_accessor<real_type, 2> w_cache_;

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
 * @details Uses SYCL's work-group data parallel kernels.
 * @tparam kernel_function the type of the used kernel function
 * @tparam Args the types of the parameters necessary for the specific kernel function; stored in a `std::tuple`
 */
template <kernel_function_type kernel_function, typename... Args>
class device_kernel_predict {
  public:
    /// The used SYCL data parallel kernel.
    constexpr static sycl::data_parallel_kernel data_parallel_kernel_type = sycl::data_parallel_kernel::work_group;

    /**
     * @brief Initialize the SYCL kernel function object.
     * @param[in] cgh the SYCL handler used to allocate the local memory
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
    device_kernel_predict(::sycl::handler &cgh, real_type *prediction, const real_type *alpha, const real_type *rho, const real_type *support_vectors, const real_type *predict_points, const std::size_t num_classes, const std::size_t num_sv, const std::size_t num_predict_points, const std::size_t num_features, const std::size_t grid_x_offset, const std::size_t grid_y_offset, Args... kernel_function_parameter) :
        cache_one_{ ::sycl::range<2>{ static_cast<std::size_t>(THREAD_BLOCK_SIZE), static_cast<std::size_t>(INTERNAL_BLOCK_SIZE) * static_cast<std::size_t>(THREAD_BLOCK_SIZE) }, cgh },
        cache_two_{ ::sycl::range<2>{ static_cast<std::size_t>(THREAD_BLOCK_SIZE), static_cast<std::size_t>(INTERNAL_BLOCK_SIZE) * static_cast<std::size_t>(THREAD_BLOCK_SIZE) }, cgh },
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
     * @param[in] nd_idx indices representing the current point in the execution space
     */
    void operator()(::sycl::nd_item<2> nd_idx) const {
        // cast values to 32-bit unsigned int values to prevent implicit conversions
        const auto local_id_0 = static_cast<unsigned>(nd_idx.get_local_id(0));
        const auto local_id_1 = static_cast<unsigned>(nd_idx.get_local_id(1));

        // cast all values to 64-bit std::size_t to prevent potential 32-bit overflows
        constexpr auto INTERNAL_BLOCK_SIZE_uz = static_cast<std::size_t>(INTERNAL_BLOCK_SIZE);
        constexpr auto THREAD_BLOCK_SIZE_uz = static_cast<std::size_t>(THREAD_BLOCK_SIZE);

        const auto threadIdx_x = static_cast<std::size_t>(nd_idx.get_local_id(0));               // current work-item in work-group x-dimension
        const auto threadIdx_y = static_cast<std::size_t>(nd_idx.get_local_id(1));               // current work-item in work-group y-dimension
        const auto blockDim_x = static_cast<std::size_t>(nd_idx.get_local_range(0));             // number of work-items in work-group x-dimension
        const auto blockDim_y = static_cast<std::size_t>(nd_idx.get_local_range(1));             // number of work-items in work-group y-dimension
        const auto blockIdx_x = static_cast<std::size_t>(nd_idx.get_group(0)) + grid_x_offset_;  // current work-group in global range x-dimension + offsets if the global range is too large
        const auto blockIdx_y = static_cast<std::size_t>(nd_idx.get_group(1)) + grid_y_offset_;  // current work-group in global range y-dimension + offsets if the global range is too large

        // create a work-item private array used for internal caching
        real_type temp[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE]{};

        {
            // rename cached arrays
            auto &pp_cache = cache_one_;
            auto &sv_cache = cache_two_;

            // calculate the indices used in the current thread, pays attention to coalesced memory accesses
            const auto pp_idx_linear = blockIdx_y * blockDim_y * INTERNAL_BLOCK_SIZE_uz + threadIdx_y;  // num_predict_points
            const auto sv_idx_linear = blockIdx_x * blockDim_x * INTERNAL_BLOCK_SIZE_uz + threadIdx_y;  // num_support_vectors

            // iterate over all features using blocking to be able to cache them for faster memory accesses
            for (std::size_t feature_block = 0; feature_block < num_features_; feature_block += THREAD_BLOCK_SIZE_uz) {
                // zero-out local memory
                for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                    pp_cache[local_id_0][internal * THREAD_BLOCK_SIZE + local_id_1] = real_type{ 0.0 };
                    sv_cache[local_id_0][internal * THREAD_BLOCK_SIZE + local_id_1] = real_type{ 0.0 };
                }

                // load data into local memory
                for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                    // calculate the indices to access the global data, pays attention to coalesced memory accesses
                    const auto global_pp_idx_linear = pp_idx_linear + static_cast<std::size_t>(internal) * THREAD_BLOCK_SIZE_uz;
                    const auto global_sv_idx_linear = sv_idx_linear + static_cast<std::size_t>(internal) * THREAD_BLOCK_SIZE_uz;

                    // store the values in the local memory
                    if (feature_block + threadIdx_x < num_features_) {
                        if (global_pp_idx_linear < num_predict_points_) {
                            pp_cache[local_id_0][internal * THREAD_BLOCK_SIZE + local_id_1] = predict_points_[(feature_block + threadIdx_x) * num_predict_points_ + global_pp_idx_linear];  // SoA
                        }
                        if (global_sv_idx_linear < num_sv_) {
                            sv_cache[local_id_0][internal * THREAD_BLOCK_SIZE + local_id_1] = support_vectors_[(feature_block + threadIdx_x) * num_sv_ + global_sv_idx_linear];  // SoA
                        }
                    }
                }
                nd_idx.barrier();  // wait until all work-items loaded their part of the data

                // perform the feature reduction calculation
                for (unsigned feature = 0; feature < THREAD_BLOCK_SIZE; ++feature) {
                    for (unsigned internal_pp = 0; internal_pp < INTERNAL_BLOCK_SIZE; ++internal_pp) {
                        for (unsigned internal_sv = 0; internal_sv < INTERNAL_BLOCK_SIZE; ++internal_sv) {
                            temp[internal_pp][internal_sv] += detail::feature_reduce<kernel_function>(sv_cache[feature][local_id_0 * INTERNAL_BLOCK_SIZE + internal_sv],
                                                                                                      pp_cache[feature][local_id_1 * INTERNAL_BLOCK_SIZE + internal_pp]);
                        }
                    }
                }
                nd_idx.barrier();  // wait until all work-items performed their part of the calculations
            }
        }

        // update temp using the respective kernel function
        for (unsigned internal_pp = 0; internal_pp < INTERNAL_BLOCK_SIZE; ++internal_pp) {
            for (unsigned internal_sv = 0; internal_sv < INTERNAL_BLOCK_SIZE; ++internal_sv) {
                temp[internal_pp][internal_sv] = detail::apply_kernel_function<kernel_function>(temp[internal_pp][internal_sv], kernel_function_parameter_);
            }
        }

        {
            // rename cached arrays
            auto &alpha_cache = cache_one_;
            auto &out_cache = cache_two_;

            // calculate the indices used in the current work-item
            const auto pp_idx = (blockIdx_y * blockDim_y + threadIdx_y) * INTERNAL_BLOCK_SIZE_uz;  // num_predict_points
            // calculate the indices used in the current work-item, pays attention to coalesced memory accesses
            const auto sv_idx_linear = blockIdx_x * blockDim_x * INTERNAL_BLOCK_SIZE_uz + threadIdx_y;  // num_support_vectors

            // iterate over all classes using blocking to be able to cache them for faster memory accesses
            for (std::size_t class_block = 0; class_block < num_classes_; class_block += THREAD_BLOCK_SIZE_uz) {
                // zero-out local memory
                for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                    alpha_cache[local_id_0][internal * THREAD_BLOCK_SIZE + local_id_1] = real_type{ 0.0 };
                    out_cache[local_id_0][internal * THREAD_BLOCK_SIZE + local_id_1] = real_type{ 0.0 };
                }

                // load data into local memory
                for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                    // calculate the indices to access the global data, pays attention to coalesced memory accesses
                    const auto global_sv_idx_linear = sv_idx_linear + static_cast<std::size_t>(internal) * THREAD_BLOCK_SIZE_uz;

                    // store the values in the local memory
                    if (class_block + threadIdx_x < num_classes_) {
                        if (global_sv_idx_linear < num_sv_) {
                            alpha_cache[local_id_0][internal * THREAD_BLOCK_SIZE + local_id_1] = alpha_[(class_block + threadIdx_x) * num_sv_ + global_sv_idx_linear];  // AoS
                        }
                        // the bias (rho) must only be applied once for all support vectors
                        if (blockIdx_x == std::size_t{ 0 }) {
                            out_cache[local_id_0][internal * THREAD_BLOCK_SIZE + local_id_1] = -rho_[class_block + threadIdx_x];
                        } else {
                            out_cache[local_id_0][internal * THREAD_BLOCK_SIZE + local_id_1] = real_type{ 0.0 };
                        }
                    }
                }
                nd_idx.barrier();  // wait until all work-items loaded their part of the data

                // calculate intermediate results and store them in local memory
                for (unsigned class_idx = 0; class_idx < THREAD_BLOCK_SIZE; ++class_idx) {
                    for (unsigned internal_pp = 0; internal_pp < INTERNAL_BLOCK_SIZE; ++internal_pp) {
                        for (unsigned internal_sv = 0; internal_sv < INTERNAL_BLOCK_SIZE; ++internal_sv) {
                            out_cache[(class_idx + local_id_0) % THREAD_BLOCK_SIZE][internal_pp * THREAD_BLOCK_SIZE + local_id_1] +=
                                temp[internal_pp][internal_sv] * alpha_cache[(class_idx + local_id_0) % THREAD_BLOCK_SIZE][local_id_0 * INTERNAL_BLOCK_SIZE + internal_sv];
                        }
                    }
                    nd_idx.barrier();  // wait until all work-items performed their part of the calculations
                }

                // atomically add the intermediate cached results to the prediction
                for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                    // calculate the indices to access the global data
                    const auto global_pp_idx = pp_idx + static_cast<std::size_t>(internal);

                    if (class_block + threadIdx_x < num_classes_ && global_pp_idx < num_predict_points_) {
                        detail::atomic_op<real_type>{ prediction_[global_pp_idx * num_classes_ + class_block + threadIdx_x] } += out_cache[local_id_0][internal * THREAD_BLOCK_SIZE + local_id_1];
                    }
                }
                nd_idx.barrier();  // wait until all work-items updated their part of the prediction
            }
        }
    }

  private:
    /// Local memory used for internal memory access optimizations.
    ::sycl::local_accessor<real_type, 2> cache_one_;
    /// Local memory used for internal memory access optimizations.
    ::sycl::local_accessor<real_type, 2> cache_two_;

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

}  // namespace plssvm::sycl::detail::work_group

#endif  // PLSSVM_BACKENDS_SYCL_KERNEL_PREDICT_WORK_GROUP_PREDICT_KERNEL_HPP_
