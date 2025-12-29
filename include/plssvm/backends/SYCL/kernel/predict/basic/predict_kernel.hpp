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

#ifndef PLSSVM_BACKENDS_SYCL_KERNEL_PREDICT_BASIC_PREDICT_KERNEL_HPP_
#define PLSSVM_BACKENDS_SYCL_KERNEL_PREDICT_BASIC_PREDICT_KERNEL_HPP_
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
 * @brief Calculate the `w` vector used to speedup the prediction using the linear kernel function.
 * @details Uses SYCL's basic data parallel kernels.
 */
class device_kernel_w_linear {
  public:
    /// The used SYCL data parallel kernel.
    constexpr static sycl::data_parallel_kernel data_parallel_kernel_type = sycl::data_parallel_kernel::basic;

    /**
     * @brief Initialize the SYCL kernel function object.
     * @param[out] w the vector to speedup the linear prediction
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
    device_kernel_w_linear(real_type *w, const real_type *alpha, const real_type *support_vectors, const std::size_t num_features, const std::size_t num_classes, const std::size_t num_sv, const std::size_t device_num_sv, const std::size_t device_sv_offset, const std::size_t grid_x_offset, const std::size_t grid_y_offset) :
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
     * @param[in] idx indices representing the current point in the execution space
     */
    void operator()(::sycl::item<2> idx) const {
        // cast all values to 64-bit std::size_t to prevent potential 32-bit overflows
        constexpr auto INTERNAL_BLOCK_SIZE_uz = static_cast<std::size_t>(INTERNAL_BLOCK_SIZE);
        constexpr auto THREAD_BLOCK_SIZE_uz = static_cast<std::size_t>(THREAD_BLOCK_SIZE);

        // calculate the indices used in the current work-item
        const auto feature_idx = (idx.get_id(1) + grid_x_offset_ * THREAD_BLOCK_SIZE_uz) * INTERNAL_BLOCK_SIZE_uz;  // num_features
        const auto class_idx = (idx.get_id(0) + grid_y_offset_ * THREAD_BLOCK_SIZE_uz) * INTERNAL_BLOCK_SIZE_uz;    // num_classes

        // create a work-item private array used for internal caching
        std::array<std::array<real_type, INTERNAL_BLOCK_SIZE>, INTERNAL_BLOCK_SIZE> temp{};

        // perform the dot product calculation
        for (std::size_t sv = 0; sv < device_num_sv_; ++sv) {
            for (unsigned internal_feature = 0; internal_feature < INTERNAL_BLOCK_SIZE; ++internal_feature) {
                for (unsigned internal_class = 0; internal_class < INTERNAL_BLOCK_SIZE; ++internal_class) {
                    // calculate the indices to access the global data
                    const auto global_class_idx = class_idx + static_cast<std::size_t>(internal_class);
                    const auto global_feature_idx = feature_idx + static_cast<std::size_t>(internal_feature);

                    real_type alpha_cache = 0.0;
                    if (global_class_idx < num_classes_) {
                        alpha_cache = alpha_[global_class_idx * num_sv_ + sv + device_sv_offset_];  // AoS
                    }
                    real_type sv_cache = 0.0;
                    if (global_feature_idx < num_features_) {
                        sv_cache = support_vectors_[global_feature_idx * device_num_sv_ + sv];  // SoA
                    }

                    temp[internal_feature][internal_class] += alpha_cache * sv_cache;
                }
            }
        }

        // update the global w-vector with the locally cached values
        for (unsigned internal_feature = 0; internal_feature < INTERNAL_BLOCK_SIZE; ++internal_feature) {
            for (unsigned internal_class = 0; internal_class < INTERNAL_BLOCK_SIZE; ++internal_class) {
                // calculate the indices to access the global data
                const auto global_feature_idx = feature_idx + static_cast<std::size_t>(internal_feature);
                const auto global_class_idx = class_idx + static_cast<std::size_t>(internal_class);

                // be sure to not perform out-of-bounds accesses
                if (global_feature_idx < num_features_ && global_class_idx < num_classes_) {
                    w_[global_feature_idx * num_classes_ + global_class_idx] = temp[internal_feature][internal_class];  // SoA
                }
            }
        }
    }

  private:
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
 * @details Uses SYCL's basic data parallel kernels.
 */
class device_kernel_predict_linear {
  public:
    /// The used SYCL data parallel kernel.
    constexpr static sycl::data_parallel_kernel data_parallel_kernel_type = sycl::data_parallel_kernel::basic;

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
     * @param[in] idx indices representing the current point in the execution space
     */
    void operator()(::sycl::item<2> idx) const {
        // cast all values to 64-bit std::size_t to prevent potential 32-bit overflows
        constexpr auto INTERNAL_BLOCK_SIZE_uz = static_cast<std::size_t>(INTERNAL_BLOCK_SIZE);
        constexpr auto THREAD_BLOCK_SIZE_uz = static_cast<std::size_t>(THREAD_BLOCK_SIZE);

        // calculate the indices used in the current work-item
        const auto pp_idx = (idx.get_id(1) + grid_x_offset_ * THREAD_BLOCK_SIZE_uz) * INTERNAL_BLOCK_SIZE_uz;     // num_predict_points
        const auto class_idx = (idx.get_id(0) + grid_y_offset_ * THREAD_BLOCK_SIZE_uz) * INTERNAL_BLOCK_SIZE_uz;  // num_classes

        // create a work-item private array used for internal caching
        std::array<std::array<real_type, INTERNAL_BLOCK_SIZE>, INTERNAL_BLOCK_SIZE> temp{};

        // perform the dot product calculation
        for (std::size_t feature = 0; feature < num_features_; ++feature) {
            for (unsigned internal_pp = 0; internal_pp < INTERNAL_BLOCK_SIZE; ++internal_pp) {
                for (unsigned internal_class = 0; internal_class < INTERNAL_BLOCK_SIZE; ++internal_class) {
                    // calculate the indices to access the global data
                    const auto global_pp_idx = pp_idx + static_cast<std::size_t>(internal_pp);
                    const auto global_class_idx = class_idx + static_cast<std::size_t>(internal_class);

                    real_type w_cache = 0.0;
                    if (global_class_idx < num_classes_) {
                        w_cache = w_[feature * num_classes_ + global_class_idx];
                    }
                    real_type pp_cache = 0.0;
                    if (global_pp_idx < num_predict_points_) {
                        pp_cache = predict_points_[feature * num_predict_points_ + global_pp_idx];  // SoA
                    }

                    temp[internal_pp][internal_class] += w_cache * pp_cache;
                }
            }
        }

        // update the global array with the local one
        for (unsigned internal_pp = 0; internal_pp < INTERNAL_BLOCK_SIZE; ++internal_pp) {
            for (unsigned internal_class = 0; internal_class < INTERNAL_BLOCK_SIZE; ++internal_class) {
                // calculate the indices to access the global data
                const auto global_pp_idx = pp_idx + static_cast<std::size_t>(internal_pp);
                const auto global_class_idx = class_idx + static_cast<std::size_t>(internal_class);

                if (global_pp_idx < num_predict_points_ && global_class_idx < num_classes_) {
                    prediction_[global_pp_idx * num_classes_ + global_class_idx] = temp[internal_pp][internal_class] - rho_[global_class_idx];  // AoS
                }
            }
        }
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
 * @details Uses SYCL's basic data parallel kernels.
 * @tparam kernel_function the type of the used kernel function
 * @tparam Args the types of the parameters necessary for the specific kernel function; stored in a `std::tuple`
 */
template <kernel_function_type kernel_function, typename... Args>
class device_kernel_predict {
  public:
    /// The used SYCL data parallel kernel.
    constexpr static sycl::data_parallel_kernel data_parallel_kernel_type = sycl::data_parallel_kernel::basic;

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
     * @param[in] idx indices representing the current point in the execution space
     */
    void operator()(::sycl::item<2> idx) const {
        // cast all values to 64-bit std::size_t to prevent potential 32-bit overflows
        constexpr auto INTERNAL_BLOCK_SIZE_uz = static_cast<std::size_t>(INTERNAL_BLOCK_SIZE);
        constexpr auto THREAD_BLOCK_SIZE_uz = static_cast<std::size_t>(THREAD_BLOCK_SIZE);

        // calculate the indices used in the current work-item
        const auto pp_idx = (idx.get_id(1) + grid_x_offset_ * THREAD_BLOCK_SIZE_uz) * INTERNAL_BLOCK_SIZE_uz;  // num_predict_points
        const auto sv_idx = (idx.get_id(0) + grid_y_offset_ * THREAD_BLOCK_SIZE_uz) * INTERNAL_BLOCK_SIZE_uz;  // num_support_vectors

        // create a work-item private array used for internal caching
        std::array<std::array<real_type, INTERNAL_BLOCK_SIZE>, INTERNAL_BLOCK_SIZE> temp{};

        // perform the feature reduction calculation
        for (std::size_t feature = 0; feature < num_features_; ++feature) {
            for (unsigned internal_pp = 0; internal_pp < INTERNAL_BLOCK_SIZE; ++internal_pp) {
                for (unsigned internal_sv = 0; internal_sv < INTERNAL_BLOCK_SIZE; ++internal_sv) {
                    // calculate the indices to access the global data
                    const auto global_pp_idx = pp_idx + static_cast<std::size_t>(internal_pp);
                    const auto global_sv_idx = sv_idx + static_cast<std::size_t>(internal_sv);

                    real_type sv_cache = 0.0;
                    if (global_sv_idx < num_sv_) {
                        sv_cache = support_vectors_[feature * num_sv_ + global_sv_idx];  // SoA
                    }
                    real_type pp_cache = 0.0;
                    if (global_pp_idx < num_predict_points_) {
                        pp_cache = predict_points_[feature * num_predict_points_ + global_pp_idx];  // SoA
                    }
                    temp[internal_pp][internal_sv] += detail::feature_reduce<kernel_function>(sv_cache, pp_cache);
                }
            }
        }

        // update temp using the respective kernel function
        for (unsigned internal_pp = 0; internal_pp < INTERNAL_BLOCK_SIZE; ++internal_pp) {
            for (unsigned internal_sv = 0; internal_sv < INTERNAL_BLOCK_SIZE; ++internal_sv) {
                temp[internal_pp][internal_sv] = detail::apply_kernel_function<kernel_function>(temp[internal_pp][internal_sv], kernel_function_parameter_);
            }
        }

        // iterate over all classes using blocking
        for (std::size_t class_block = 0; class_block < num_classes_; class_block += THREAD_BLOCK_SIZE_uz) {
            if (sv_idx == 0) {
                for (std::size_t class_idx = 0; class_idx < THREAD_BLOCK_SIZE_uz; ++class_idx) {
                    for (unsigned internal_pp = 0; internal_pp < INTERNAL_BLOCK_SIZE; ++internal_pp) {
                        // calculate the index to access the global data
                        const auto global_pp_idx = pp_idx + static_cast<std::size_t>(internal_pp);

                        if (global_pp_idx < num_predict_points_ && class_block + class_idx < num_classes_) {
                            detail::atomic_op<real_type>{ prediction_[global_pp_idx * num_classes_ + class_block + class_idx] } += -rho_[class_block + class_idx];
                        }
                    }
                }
            }

            // atomically add the results to the prediction
            for (unsigned internal_pp = 0; internal_pp < INTERNAL_BLOCK_SIZE; ++internal_pp) {
                for (unsigned internal_sv = 0; internal_sv < INTERNAL_BLOCK_SIZE; ++internal_sv) {
                    // calculate the indices to access the global data
                    const auto global_pp_idx = pp_idx + static_cast<std::size_t>(internal_pp);
                    const auto global_sv_idx = sv_idx + static_cast<std::size_t>(internal_sv);

                    if (global_pp_idx < num_predict_points_) {
                        for (std::size_t class_idx = 0; class_idx < THREAD_BLOCK_SIZE_uz; ++class_idx) {
                            if (class_block + class_idx < num_classes_) {
                                detail::atomic_op<real_type>{ prediction_[global_pp_idx * num_classes_ + class_block + class_idx] } +=
                                    temp[internal_pp][internal_sv] * alpha_[(class_block + class_idx) * num_sv_ + global_sv_idx];
                            }
                        }
                    }
                }
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

}  // namespace plssvm::sycl::detail::basic

#endif  // PLSSVM_BACKENDS_SYCL_KERNEL_PREDICT_BASIC_PREDICT_KERNEL_HPP_
