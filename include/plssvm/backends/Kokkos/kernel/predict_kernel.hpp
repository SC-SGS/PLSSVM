/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines the functions used for prediction for the C-SVM using the Kokkos backend.
 */

#ifndef PLSSVM_BACKENDS_KOKKOS_KERNEL_PREDICT_KERNEL_HPP_
#define PLSSVM_BACKENDS_KOKKOS_KERNEL_PREDICT_KERNEL_HPP_
#pragma once

#include "plssvm/backends/Kokkos/detail/standard_layout_tuple.hpp"  // plssvm::kokkos::detail::standard_layout_tuple
#include "plssvm/backends/Kokkos/kernel/kernel_functions.hpp"       // plssvm::kokkos::detail::{feature_reduce, apply_kernel_function}
#include "plssvm/constants.hpp"                                     // plssvm::real_type
#include "plssvm/kernel_function_types.hpp"                         // plssvm::kernel_function_type
#include "plssvm/target_platforms.hpp"                              // plssvm::target_platform

#include "Kokkos_Core.hpp"  // KOKKOS_INLINE_FUNCTION, Kokkos::View, Kokkos::TeamPolicy, Kokkos::atomic_add

#include <cstddef>  // std::size_t
#include <utility>  // std::move

namespace plssvm::kokkos::detail {

/**
 * @brief Calculate the `w` vector used to speedup the prediction using the linear kernel function.
 * @tparam ExecutionSpace the Kokkos::ExecutionSpace used to execute the kernel
 */
template <typename ExecutionSpace>
class device_kernel_w_linear {
    /**
     * @brief The type of the used Kokkos::View.
     */
    template <typename T>
    using device_view_type = Kokkos::View<T *, ExecutionSpace>;

  public:
    /**
     * @brief Initialize the Kokkos kernel function object.
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
     * @param[in] grid_size_x the size of the execution grid in x-dimension
     */
    device_kernel_w_linear(device_view_type<real_type> w, device_view_type<const real_type> alpha, device_view_type<const real_type> support_vectors, const std::size_t num_features, const std::size_t num_classes, const std::size_t num_sv, const std::size_t device_num_sv, const std::size_t device_sv_offset, const std::size_t grid_x_offset, const std::size_t grid_y_offset, const std::size_t grid_size_x) :
        w_{ std::move(w) },
        alpha_{ std::move(alpha) },
        support_vectors_{ std::move(support_vectors) },
        num_features_{ num_features },
        num_classes_{ num_classes },
        num_sv_{ num_sv },
        device_num_sv_{ device_num_sv },
        device_sv_offset_{ device_sv_offset },
        grid_x_offset_{ grid_x_offset },
        grid_y_offset_{ grid_y_offset },
        grid_size_x_{ grid_size_x } { }

    /**
     * @brief Function call operator overload performing the actual calculation.
     * @param[in] team the Kokkos team representing the current point in the execution space
     */
    KOKKOS_INLINE_FUNCTION
    void operator()(const typename Kokkos::TeamPolicy<ExecutionSpace>::member_type &team) const {
        // cast all values to 64-bit std::size_t to prevent potential 32-bit overflows
        constexpr auto THREAD_BLOCK_SIZE_uz = static_cast<std::size_t>(THREAD_BLOCK_SIZE);

        const auto threadIdx_x = static_cast<std::size_t>(team.team_rank()) / THREAD_BLOCK_SIZE_uz;            // current thread in team x-dimension
        const auto threadIdx_y = static_cast<std::size_t>(team.team_rank()) % THREAD_BLOCK_SIZE_uz;            // current thread in team y-dimension
        const auto blockDim_x = THREAD_BLOCK_SIZE_uz;                                                          // number of threads in team x-dimension
        const auto blockDim_y = THREAD_BLOCK_SIZE_uz;                                                          // number of threads in team y-dimension
        const auto blockIdx_x = static_cast<std::size_t>(team.league_rank()) % grid_size_x_ + grid_x_offset_;  // current team in league x-dimension + offsets if the league size is too large
        const auto blockIdx_y = static_cast<std::size_t>(team.league_rank()) / grid_size_x_ + grid_y_offset_;  // current team in league y-dimension + offsets if the league size is too large

        // calculate the indices used in the current thread
        const auto global_feature_idx = blockIdx_x * blockDim_x + threadIdx_x;  // num_features
        const auto global_class_idx = blockIdx_y * blockDim_y + threadIdx_y;    // num_classes

        // be sure to not perform out-of-bounds accesses
        if (global_feature_idx < num_features_ && global_class_idx < num_classes_) {
            real_type temp{ 0.0 };

            // perform the dot product calculation
            for (std::size_t sv = 0; sv < device_num_sv_; ++sv) {
                temp += alpha_[global_class_idx * num_sv_ + sv + device_sv_offset_] *  // AoS
                        support_vectors_[sv * num_features_ + global_feature_idx];     // AoS
            }

            w_[global_class_idx * num_features_ + global_feature_idx] = temp;  // AoS
        }
    }

  private:
    /// @cond Doxygen_suppress
    device_view_type<real_type> w_;
    device_view_type<const real_type> alpha_;
    device_view_type<const real_type> support_vectors_;
    const std::size_t num_features_;
    const std::size_t num_classes_;
    const std::size_t num_sv_;
    const std::size_t device_num_sv_;
    const std::size_t device_sv_offset_;
    const std::size_t grid_x_offset_;
    const std::size_t grid_y_offset_;
    const std::size_t grid_size_x_;
    /// @endcond
};

/**
 * @brief Predict the @p predict_points using the linear kernel speeding up the calculation using the @p w vector.
 * @tparam ExecutionSpace the Kokkos::ExecutionSpace used to execute the kernel
 */
template <typename ExecutionSpace>
class device_kernel_predict_linear {
    /**
     * @brief The type of the used Kokkos::View.
     */
    template <typename T>
    using device_view_type = Kokkos::View<T *, ExecutionSpace>;

  public:
    /**
     * @brief Initialize the Kokkos kernel function object.
     * @param[out] prediction the predicted values
     * @param[in] w the vector to speedup the calculations
     * @param[in] rho the previously learned bias
     * @param[in] predict_points the data points to predict
     * @param[in] num_classes the number of classes
     * @param[in] num_predict_points the number of data points to predict
     * @param[in] num_features the number of features per data point
     * @param[in] grid_x_offset the offset in x-dimension into the data points if more than one execution grid has to be used
     * @param[in] grid_y_offset the offset in y-dimension into the data points if more than one execution grid has to be used
     * @param[in] grid_size_x the size of the execution grid in x-dimension
     */
    device_kernel_predict_linear(device_view_type<real_type> prediction, device_view_type<const real_type> w, device_view_type<const real_type> rho, device_view_type<const real_type> predict_points, const std::size_t num_classes, const std::size_t num_predict_points, const std::size_t num_features, const std::size_t grid_x_offset, const std::size_t grid_y_offset, const std::size_t grid_size_x) :
        prediction_{ std::move(prediction) },
        w_{ std::move(w) },
        rho_{ std::move(rho) },
        predict_points_{ std::move(predict_points) },
        num_classes_{ num_classes },
        num_predict_points_{ num_predict_points },
        num_features_{ num_features },
        grid_x_offset_{ grid_x_offset },
        grid_y_offset_{ grid_y_offset },
        grid_size_x_{ grid_size_x } { }

    /**
     * @brief Function call operator overload performing the actual calculation.
     * @param[in] team the Kokkos team representing the current point in the execution space
     */
    KOKKOS_INLINE_FUNCTION
    void operator()(const typename Kokkos::TeamPolicy<ExecutionSpace>::member_type &team) const {
        // cast all values to 64-bit std::size_t to prevent potential 32-bit overflows
        constexpr auto THREAD_BLOCK_SIZE_uz = static_cast<std::size_t>(THREAD_BLOCK_SIZE);

        const auto threadIdx_x = static_cast<std::size_t>(team.team_rank()) / THREAD_BLOCK_SIZE_uz;            // current thread in team x-dimension
        const auto threadIdx_y = static_cast<std::size_t>(team.team_rank()) % THREAD_BLOCK_SIZE_uz;            // current thread in team y-dimension
        const auto blockDim_x = THREAD_BLOCK_SIZE_uz;                                                          // number of threads in team x-dimension
        const auto blockDim_y = THREAD_BLOCK_SIZE_uz;                                                          // number of threads in team y-dimension
        const auto blockIdx_x = static_cast<std::size_t>(team.league_rank()) % grid_size_x_ + grid_x_offset_;  // current team in league x-dimension + offsets if the league size is too large
        const auto blockIdx_y = static_cast<std::size_t>(team.league_rank()) / grid_size_x_ + grid_y_offset_;  // current team in league y-dimension + offsets if the league size is too large

        // calculate the indices used in the current thread
        const auto global_pp_idx = blockIdx_x * blockDim_x + threadIdx_x;     // num_predict_points
        const auto global_class_idx = blockIdx_y * blockDim_y + threadIdx_y;  // num_classes

        // be sure to not perform out-of-bounds accesses
        if (global_pp_idx < num_predict_points_ && global_class_idx < num_classes_) {
            real_type temp{ 0.0 };

            // perform the dot product calculation
            for (std::size_t feature = 0; feature < num_features_; ++feature) {
                temp += w_[global_class_idx * num_features_ + feature] *           // AoS
                        predict_points_[global_pp_idx * num_features_ + feature];  // AoS
            }

            prediction_[global_pp_idx * num_classes_ + global_class_idx] = temp - rho_[global_class_idx];
        }
    }

  private:
    /// @cond Doxygen_suppress
    device_view_type<real_type> prediction_;
    device_view_type<const real_type> w_;
    device_view_type<const real_type> rho_;
    device_view_type<const real_type> predict_points_;
    const std::size_t num_classes_;
    const std::size_t num_predict_points_;
    const std::size_t num_features_;
    const std::size_t grid_x_offset_;
    const std::size_t grid_y_offset_;
    const std::size_t grid_size_x_;
    /// @endcond
};

/**
 * @brief Predict the @p predict_points using the @p kernel_function.
 * @tparam ExecutionSpace the Kokkos::ExecutionSpace used to execute the kernel
 * @tparam kernel_function the type of the used kernel function
 * @tparam Args the types of the parameters necessary for the specific kernel function
 */
template <typename ExecutionSpace, kernel_function_type kernel_function, typename... Args>
class device_kernel_predict {
    /**
     * @brief The type of the used Kokkos::View.
     */
    template <typename T>
    using device_view_type = Kokkos::View<T *, ExecutionSpace>;

  public:
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
     * @param[in] grid_size_x the size of the execution grid in x-dimension
     * @param[in] kernel_function_parameter the parameters necessary to apply the @p kernel_function
     */
    device_kernel_predict(device_view_type<real_type> prediction, device_view_type<const real_type> alpha, device_view_type<const real_type> rho, device_view_type<const real_type> support_vectors, device_view_type<const real_type> predict_points, const std::size_t num_classes, const std::size_t num_sv, const std::size_t num_predict_points, const std::size_t num_features, const std::size_t grid_x_offset, const std::size_t grid_y_offset, const std::size_t grid_size_x, Args... kernel_function_parameter) :
        prediction_{ std::move(prediction) },
        alpha_{ std::move(alpha) },
        rho_{ std::move(rho) },
        support_vectors_{ std::move(support_vectors) },
        predict_points_{ std::move(predict_points) },
        num_classes_{ num_classes },
        num_sv_{ num_sv },
        num_predict_points_{ num_predict_points },
        num_features_{ num_features },
        grid_x_offset_{ grid_x_offset },
        grid_y_offset_{ grid_y_offset },
        grid_size_x_{ grid_size_x },
        kernel_function_parameter_{ detail::make_standard_layout_tuple(std::forward<Args>(kernel_function_parameter)...) } { }

    /**
     * @brief Function call operator overload performing the actual calculation.
     * @param[in] team the Kokkos team representing the current point in the execution space
     */
    KOKKOS_INLINE_FUNCTION
    void operator()(const typename Kokkos::TeamPolicy<ExecutionSpace>::member_type &team) const {
        // cast all values to 64-bit std::size_t to prevent potential 32-bit overflows
        constexpr auto THREAD_BLOCK_SIZE_uz = static_cast<std::size_t>(THREAD_BLOCK_SIZE);

        const auto threadIdx_x = static_cast<std::size_t>(team.team_rank()) / THREAD_BLOCK_SIZE_uz;            // current thread in team x-dimension
        const auto threadIdx_y = static_cast<std::size_t>(team.team_rank()) % THREAD_BLOCK_SIZE_uz;            // current thread in team y-dimension
        const auto blockDim_x = THREAD_BLOCK_SIZE_uz;                                                          // number of threads in team x-dimension
        const auto blockDim_y = THREAD_BLOCK_SIZE_uz;                                                          // number of threads in team y-dimension
        const auto blockIdx_x = static_cast<std::size_t>(team.league_rank()) % grid_size_x_ + grid_x_offset_;  // current team in league x-dimension + offsets if the league size is too large
        const auto blockIdx_y = static_cast<std::size_t>(team.league_rank()) / grid_size_x_ + grid_y_offset_;  // current team in league y-dimension + offsets if the league size is too large

        // calculate the indices used in the current thread
        const auto global_pp_idx = blockIdx_x * blockDim_x + threadIdx_x;  // num_predict_points
        const auto global_sv_idx = blockIdx_y * blockDim_y + threadIdx_y;  // num_support_vectors

        // be sure to not perform out-of-bounds accesses
        if (global_sv_idx < num_sv_ && global_pp_idx < num_predict_points_) {
            real_type temp{ 0.0 };

            // perform the feature reduction calculation
            for (std::size_t feature = 0; feature < num_features_; ++feature) {
                temp += detail::feature_reduce<kernel_function>(support_vectors_[global_sv_idx * num_features_ + feature],  // AoS
                                                                predict_points_[global_pp_idx * num_features_ + feature]);  // AoS
            }

            // update temp using the respective kernel function
            temp = detail::apply_kernel_function<kernel_function>(temp, kernel_function_parameter_);

            // iterate over all classes
            for (std::size_t class_idx = 0; class_idx < num_classes_; ++class_idx) {
                real_type out_cache = alpha_[class_idx * num_sv_ + global_sv_idx] * temp;  // AoS

                // the bias (rho) must only be applied once for all support vectors
                if (global_sv_idx == std::size_t{ 0 }) {
                    out_cache -= rho_[class_idx];
                }

                Kokkos::atomic_add(&prediction_[global_pp_idx * num_classes_ + class_idx], out_cache);  // AoS
            }
        }
    }

  private:
    /// @cond Doxygen_suppress
    device_view_type<real_type> prediction_;
    device_view_type<const real_type> alpha_;
    device_view_type<const real_type> rho_;
    device_view_type<const real_type> support_vectors_;
    device_view_type<const real_type> predict_points_;
    const std::size_t num_classes_;
    const std::size_t num_sv_;
    const std::size_t num_predict_points_;
    const std::size_t num_features_;
    const std::size_t grid_x_offset_;
    const std::size_t grid_y_offset_;
    const std::size_t grid_size_x_;
    const detail::standard_layout_tuple<Args...> kernel_function_parameter_;
    /// @endcond
};

}  // namespace plssvm::kokkos::detail

#endif  // PLSSVM_BACKENDS_KOKKOS_KERNEL_PREDICT_KERNEL_HPP_
