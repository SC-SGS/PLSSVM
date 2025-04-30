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

#include "plssvm/backends/SYCL/detail/atomics.hpp"           // plssvm::sycl::detail::atomic_op
#include "plssvm/backends/SYCL/kernel/kernel_functions.hpp"  // plssvm::sycl::detail::{feature_reduce, apply_kernel_function}
#include "plssvm/constants.hpp"                              // plssvm::{real_type, THREAD_BLOCK_SIZE, INTERNAL_BLOCK_SIZE, FEATURE_BLOCK_SIZE, PADDING_SIZE}
#include "plssvm/kernel_function_types.hpp"                  // plssvm::kernel_function_type

#include "sycl/sycl.hpp"  // sycl::item

#include <cstddef>  // std::size_t
#include <tuple>    // std::tuple, std::make_tuple

namespace plssvm::sycl::detail::basic {

/**
 * @brief Calculate the `q` vector used to speedup the prediction using the linear kernel function.
 * @details Uses SYCL's basic data parallel kernels.
 */
class device_kernel_w_linear {
  public:
    /**
     * @brief Initialize the SYCL kernel function object.
     * @param[in,out] w_d the vector to speedup the linear prediction
     * @param[in] alpha_d the previously learned weights
     * @param[in] sv_d the support vectors
     * @param[in] num_classes the number of classes
     * @param[in] num_sv the number of support vectors
     * @param[in] device_specific_num_sv the number of support vectors the current device is responsible for
     * @param[in] sv_offset the first support vector (row in @p alpha_d) the current device is responsible for
     * @param[in] grid_x_offset the offset in x-dimension into the data points if more than one execution grid has to be used
     * @param[in] grid_y_offset the offset in y-dimension into the data points if more than one execution grid has to be used
     */
    device_kernel_w_linear(real_type *w_d, const real_type *alpha_d, const real_type *sv_d, const std::size_t num_classes, const std::size_t num_sv, const std::size_t device_specific_num_sv, const std::size_t sv_offset, const std::size_t grid_x_offset, const std::size_t grid_y_offset) :
        w_d_{ w_d },
        alpha_d_{ alpha_d },
        sv_d_{ sv_d },
        num_classes_{ num_classes },
        num_sv_{ num_sv },
        device_specific_num_sv_{ device_specific_num_sv },
        sv_offset_{ sv_offset },
        grid_x_offset_{ grid_x_offset },
        grid_y_offset_{ grid_y_offset } { }

    /**
     * @brief Function call operator overload performing the actual calculation.
     * @param[in] idx indices representing the current point in the execution space
     */
    void operator()(::sycl::item<2> idx) const {
        // cast all values to 64-bit std::size_t to prevent potential 32-bit overflows
        const auto INTERNAL_BLOCK_SIZE_uz = static_cast<std::size_t>(INTERNAL_BLOCK_SIZE);
        const auto PADDING_SIZE_uz = static_cast<std::size_t>(PADDING_SIZE);

        // calculate the indices used in the current work-item
        const std::size_t feature_idx = (idx.get_id(1) + grid_x_offset_ * THREAD_BLOCK_SIZE) * INTERNAL_BLOCK_SIZE_uz;
        const std::size_t class_idx = (idx.get_id(0) + grid_y_offset_ * THREAD_BLOCK_SIZE) * INTERNAL_BLOCK_SIZE_uz;

        // create a work-item private array used for internal caching
        real_type temp[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE]{};

        // iterate over all support vectors using blocking to be able to cache them for faster memory accesses
        for (std::size_t sv = 0; sv < device_specific_num_sv_; ++sv) {
            // perform the dot product calculation
            for (unsigned internal_feature = 0; internal_feature < INTERNAL_BLOCK_SIZE; ++internal_feature) {
                for (unsigned internal_class = 0; internal_class < INTERNAL_BLOCK_SIZE; ++internal_class) {
                    const auto global_class_idx = class_idx + static_cast<std::size_t>(internal_class);
                    const auto global_feature_idx = feature_idx + static_cast<std::size_t>(internal_feature);

                    temp[internal_feature][internal_class] += alpha_d_[global_class_idx * (num_sv_ + PADDING_SIZE_uz) + sv + sv_offset_] * sv_d_[global_feature_idx * (device_specific_num_sv_ + PADDING_SIZE_uz) + sv];
                }
            }
        }

        // update global array with local one
        for (unsigned internal_feature = 0; internal_feature < INTERNAL_BLOCK_SIZE; ++internal_feature) {
            for (unsigned internal_class = 0; internal_class < INTERNAL_BLOCK_SIZE; ++internal_class) {
                const auto global_class_idx = class_idx + static_cast<std::size_t>(internal_class);
                const auto global_feature_idx = feature_idx + static_cast<std::size_t>(internal_feature);

                w_d_[global_feature_idx * (num_classes_ + PADDING_SIZE_uz) + global_class_idx] = temp[internal_feature][internal_class];
            }
        }
    }

  private:
    /// @cond Doxygen_suppress
    real_type *w_d_;
    const real_type *alpha_d_;
    const real_type *sv_d_;
    const std::size_t num_classes_;
    const std::size_t num_sv_;
    const std::size_t device_specific_num_sv_;
    const std::size_t sv_offset_;
    const std::size_t grid_x_offset_;
    const std::size_t grid_y_offset_;
    /// @endcond
};

/**
 * @brief Predict the @p predict_points_d using the linear kernel speeding up the calculation using the @p w_d vector.
 * @details Uses SYCL's basic data parallel kernels.
 */
class device_kernel_predict_linear {
  public:
    /**
     * @brief Initialize the SYCL kernel function object.
     * @param[out] prediction_d the predicted values
     * @param[in] w_d the vector to speedup the calculations
     * @param[in] rho_d the previously learned bias
     * @param[in] predict_points_d the data points to predict
     * @param[in] num_classes the number of classes
     * @param[in] num_predict_points the number of data points to predict
     * @param[in] num_features the number of features per data point
     * @param[in] grid_x_offset the offset in x-dimension into the data points if more than one execution grid has to be used
     * @param[in] grid_y_offset the offset in y-dimension into the data points if more than one execution grid has to be used
     */
    device_kernel_predict_linear(real_type *prediction_d, const real_type *w_d, const real_type *rho_d, const real_type *predict_points_d, const std::size_t num_classes, const std::size_t num_predict_points, const std::size_t num_features, const std::size_t grid_x_offset, const std::size_t grid_y_offset) :
        prediction_d_{ prediction_d },
        w_d_{ w_d },
        rho_d_{ rho_d },
        predict_points_d_{ predict_points_d },
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
        const auto INTERNAL_BLOCK_SIZE_uz = static_cast<std::size_t>(INTERNAL_BLOCK_SIZE);
        const auto PADDING_SIZE_uz = static_cast<std::size_t>(PADDING_SIZE);

        // calculate the indices used in the current work-item
        const std::size_t pp_idx = (idx.get_id(1) + grid_x_offset_ * THREAD_BLOCK_SIZE) * INTERNAL_BLOCK_SIZE_uz;
        const std::size_t class_idx = (idx.get_id(0) + grid_y_offset_ * THREAD_BLOCK_SIZE) * INTERNAL_BLOCK_SIZE_uz;

        // create a work-item private array used for internal caching
        real_type temp[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE]{};

        // iterate over all support vectors using blocking to be able to cache them for faster memory accesses
        for (std::size_t dim = 0; dim < num_features_; ++dim) {
            // perform the dot product calculation
            for (unsigned internal_pd = 0; internal_pd < INTERNAL_BLOCK_SIZE; ++internal_pd) {
                for (unsigned internal_class = 0; internal_class < INTERNAL_BLOCK_SIZE; ++internal_class) {
                    const auto global_pp_idx = pp_idx + static_cast<std::size_t>(internal_pd);
                    const auto global_class_idx = class_idx + static_cast<std::size_t>(internal_class);

                    temp[internal_pd][internal_class] += w_d_[dim * (num_classes_ + PADDING_SIZE_uz) + global_class_idx] * predict_points_d_[dim * (num_predict_points_ + PADDING_SIZE_uz) + global_pp_idx];
                }
            }
        }

        // update global array with local one
        for (unsigned internal_pd = 0; internal_pd < INTERNAL_BLOCK_SIZE; ++internal_pd) {
            for (unsigned internal_class = 0; internal_class < INTERNAL_BLOCK_SIZE; ++internal_class) {
                const auto global_class_idx = class_idx + static_cast<std::size_t>(internal_class);
                const auto global_pp_idx = pp_idx + static_cast<std::size_t>(internal_pd);

                prediction_d_[global_pp_idx * (num_classes_ + PADDING_SIZE_uz) + global_class_idx] = temp[internal_pd][internal_class] - rho_d_[global_class_idx];
            }
        }
    }

  private:
    /// @cond Doxygen_suppress
    real_type *prediction_d_;
    const real_type *w_d_;
    const real_type *rho_d_;
    const real_type *predict_points_d_;
    const std::size_t num_classes_;
    const std::size_t num_predict_points_;
    const std::size_t num_features_;
    const std::size_t grid_x_offset_;
    const std::size_t grid_y_offset_;
    /// @endcond
};

/**
 * @brief Predict the @p predict_points_d using the @p kernel_function.
 * @details Uses SYCL's basic data parallel kernels.
 * @tparam kernel_function the type of the used kernel function
 * @tparam Args the types of the parameters necessary for the specific kernel function; stored in a `std::tuple`
 */
template <kernel_function_type kernel_function, typename... Args>
class device_kernel_predict {
  public:
    /**
     * @brief Initialize the SYCL kernel function object.
     * @param[in] prediction_d the predicted values
     * @param[in] alpha_d the previously learned weights
     * @param[in] rho_d the previously learned biases
     * @param[in] sv_d the support vectors
     * @param[in] predict_points_d the data points to predict
     * @param[in] num_classes the number of classes
     * @param[in] num_sv the number of support vectors
     * @param[in] num_predict_points the number of data points to predict
     * @param[in] num_features the number of features per data point
     * @param[in] grid_x_offset the offset in x-dimension into the data points if more than one execution grid has to be used
     * @param[in] grid_y_offset the offset in y-dimension into the data points if more than one execution grid has to be used
     * @param[in] kernel_function_parameter the parameters necessary to apply the @p kernel_function
     */
    device_kernel_predict(real_type *prediction_d, const real_type *alpha_d, const real_type *rho_d, const real_type *sv_d, const real_type *predict_points_d, const std::size_t num_classes, const std::size_t num_sv, const std::size_t num_predict_points, const std::size_t num_features, const std::size_t grid_x_offset, const std::size_t grid_y_offset, Args... kernel_function_parameter) :
        prediction_d_{ prediction_d },
        alpha_d_{ alpha_d },
        rho_d_{ rho_d },
        sv_d_{ sv_d },
        predict_points_d_{ predict_points_d },
        num_classes_{ num_classes },
        num_sv_{ num_sv },
        num_predict_points_{ num_predict_points },
        num_features_{ num_features },
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
        const std::size_t pp_idx = (idx.get_id(1) + grid_x_offset_ * THREAD_BLOCK_SIZE) * INTERNAL_BLOCK_SIZE_uz;
        const std::size_t sv_idx = (idx.get_id(0) + grid_y_offset_ * THREAD_BLOCK_SIZE) * INTERNAL_BLOCK_SIZE_uz;

        // create a work-item private array used for internal caching
        real_type temp[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE]{};

        // iterate over all features using blocking to be able to cache them for faster memory accesses
        for (std::size_t dim = 0; dim < num_features_; ++dim) {
            // perform the feature reduction calculation
            for (unsigned internal_pd = 0; internal_pd < INTERNAL_BLOCK_SIZE; ++internal_pd) {
                for (unsigned internal_sv = 0; internal_sv < INTERNAL_BLOCK_SIZE; ++internal_sv) {
                    const auto global_pp_idx = pp_idx + static_cast<std::size_t>(internal_pd);
                    const auto global_sv_idx = sv_idx + static_cast<std::size_t>(internal_sv);

                    temp[internal_pd][internal_sv] += detail::feature_reduce<kernel_function>(sv_d_[dim * (num_sv_ + PADDING_SIZE_uz) + global_sv_idx],
                                                                                              predict_points_d_[dim * (num_predict_points_ + PADDING_SIZE_uz) + global_pp_idx]);
                }
            }
        }

        // update temp using the respective kernel function
        for (unsigned internal_pd = 0; internal_pd < INTERNAL_BLOCK_SIZE; ++internal_pd) {
            for (unsigned internal_sv = 0; internal_sv < INTERNAL_BLOCK_SIZE; ++internal_sv) {
                temp[internal_pd][internal_sv] = detail::apply_kernel_function<kernel_function>(temp[internal_pd][internal_sv], kernel_function_parameter_);
            }
        }

        // iterate over all features using blocking to be able to cache them for faster memory accesses
        for (std::size_t dim = 0; dim < num_classes_; ++dim) {
            if (sv_idx == 0) {
                for (unsigned internal_pd = 0; internal_pd < INTERNAL_BLOCK_SIZE; ++internal_pd) {
                    const auto global_pp_idx = pp_idx + static_cast<std::size_t>(internal_pd);
                    detail::atomic_op<real_type>{ prediction_d_[global_pp_idx * (num_classes_ + PADDING_SIZE_uz) + dim] } += -rho_d_[dim];
                }
            }

            // calculate intermediate results and store them in local memory
            for (unsigned internal_pd = 0; internal_pd < INTERNAL_BLOCK_SIZE; ++internal_pd) {
                for (unsigned internal_sv = 0; internal_sv < INTERNAL_BLOCK_SIZE; ++internal_sv) {
                    const auto global_pp_idx = pp_idx + static_cast<std::size_t>(internal_pd);
                    const auto global_sv_idx = sv_idx + static_cast<std::size_t>(internal_sv);

                    detail::atomic_op<real_type>{ prediction_d_[global_pp_idx * (num_classes_ + PADDING_SIZE_uz) + dim] } +=
                        temp[internal_pd][internal_sv] * alpha_d_[dim * (num_sv_ + PADDING_SIZE_uz) + global_sv_idx];
                }
            }
        }
    }

  private:
    /// @cond Doxygen_suppress
    real_type *prediction_d_;
    const real_type *alpha_d_;
    const real_type *rho_d_;
    const real_type *sv_d_;
    const real_type *predict_points_d_;
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
