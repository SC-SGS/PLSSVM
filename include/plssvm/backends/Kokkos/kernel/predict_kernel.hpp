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

#ifndef PLSSVM_BACKENDS_KOKKOS_PREDICT_KERNEL_HPP_
#define PLSSVM_BACKENDS_KOKKOS_PREDICT_KERNEL_HPP_
#pragma once

#include "plssvm/backends/Kokkos/kernel/kernel_functions.hpp"  // plssvm::kokkos::detail::{feature_reduce, apply_kernel_function}
#include "plssvm/constants.hpp"                                // plssvm::{real_type, THREAD_BLOCK_SIZE, INTERNAL_BLOCK_SIZE, PADDING_SIZE}
#include "plssvm/kernel_function_types.hpp"                    // plssvm::kernel_function_type
#include "plssvm/target_platforms.hpp"                         // plssvm::target_platform

#include "Kokkos_Core.hpp"  // KOKKOS_INLINE_FUNCTION, Kokkos::View, Kokkos::TeamPolicy, Kokkos::mdspan, Kokkos::dextents, Kokkos::atomic_add

#include <cstddef>  // std::size_t

namespace plssvm::kokkos::detail {

/**
 * @brief Calculate the `w` vector used to speedup the prediction using the linear kernel function.
 * @tparam ExecutionSpace the Kokkos::ExecutionSpace used to execute the kernel
 * @tparam target the target platform
 */
template <typename ExecutionSpace, target_platform target>
class device_kernel_w_linear {
    /**
     * @brief The type of the used Kokkos::View.
     */
    template <typename T>
    using device_view_type = Kokkos::View<T *, ExecutionSpace>;

  public:
    /**
     * @brief Initialize the Kokkos kernel function object.
     * @param[in,out] w the vector to speedup the linear prediction
     * @param[in] alpha the previously learned weights
     * @param[in] support_vectors the support vectors
     * @param[in] num_classes the number of classes
     * @param[in] num_sv the number of support vectors
     * @param[in] device_num_sv the number of support vectors the current device is responsible for
     * @param[in] device_sv_offset the first support vector (row in @p alpha) the current device is responsible for
     * @param[in] grid_x_offset the offset in x-dimension into the data points if more than one execution grid has to be used
     * @param[in] grid_y_offset the offset in y-dimension into the data points if more than one execution grid has to be used
     * @param[in] grid_size_x the size of the execution grid in x-dimension
     */
    device_kernel_w_linear(device_view_type<real_type> w, device_view_type<const real_type> alpha, device_view_type<const real_type> support_vectors, const std::size_t num_classes, const std::size_t num_sv, const std::size_t device_num_sv, const std::size_t device_sv_offset, const std::size_t grid_x_offset, const std::size_t grid_y_offset, const std::size_t grid_size_x) :
        w_{ w },
        alpha_{ alpha },
        support_vectors_{ support_vectors },
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
        // cast values to 32-bit unsigned int values to prevent implicit conversions
        const auto team_rank_x = static_cast<unsigned>(team.team_rank()) / THREAD_BLOCK_SIZE;
        const auto team_rank_y = static_cast<unsigned>(team.team_rank()) % THREAD_BLOCK_SIZE;

        // cast all values to 64-bit std::size_t to prevent potential 32-bit overflows
        constexpr auto INTERNAL_BLOCK_SIZE_uz = static_cast<std::size_t>(INTERNAL_BLOCK_SIZE);
        constexpr auto THREAD_BLOCK_SIZE_uz = static_cast<std::size_t>(THREAD_BLOCK_SIZE);
        constexpr auto PADDING_SIZE_uz = static_cast<std::size_t>(PADDING_SIZE);

        const auto threadIdx_x = static_cast<std::size_t>(team.team_rank()) / THREAD_BLOCK_SIZE_uz;            // current thread in team x-dimension
        const auto threadIdx_y = static_cast<std::size_t>(team.team_rank()) % THREAD_BLOCK_SIZE_uz;            // current thread in team y-dimension
        const auto blockDim_x = THREAD_BLOCK_SIZE_uz;                                                          // number of threads in team x-dimension
        const auto blockDim_y = THREAD_BLOCK_SIZE_uz;                                                          // number of threads in team y-dimension
        const auto blockIdx_x = static_cast<std::size_t>(team.league_rank()) % grid_size_x_ + grid_x_offset_;  // current team in league x-dimension + offsets if the league size is too large
        const auto blockIdx_y = static_cast<std::size_t>(team.league_rank()) / grid_size_x_ + grid_y_offset_;  // current team in league y-dimension + offsets if the league size is too large

        // create two scratchpad memory arrays used for caching
        constexpr std::size_t scratchpad_size = THREAD_BLOCK_SIZE_uz * THREAD_BLOCK_SIZE_uz * INTERNAL_BLOCK_SIZE_uz;
        real_type *scratchpad_ptr = static_cast<real_type *>(team.team_shmem().get_shmem(std::size_t{ 2 } * scratchpad_size * sizeof(real_type)));
        Kokkos::mdspan<real_type, Kokkos::dextents<std::size_t, 2>> feature_cache{ scratchpad_ptr, THREAD_BLOCK_SIZE_uz, INTERNAL_BLOCK_SIZE_uz * THREAD_BLOCK_SIZE_uz };
        Kokkos::mdspan<real_type, Kokkos::dextents<std::size_t, 2>> alpha_cache{ scratchpad_ptr + scratchpad_size, THREAD_BLOCK_SIZE_uz, INTERNAL_BLOCK_SIZE_uz * THREAD_BLOCK_SIZE_uz };

        // create a thread private array used for internal caching
        real_type temp[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE]{};

        {
            // calculate the indices used in the current thread, pays attention to coalesced memory accesses
            const auto feature_idx_linear = blockIdx_x * blockDim_x * INTERNAL_BLOCK_SIZE_uz + threadIdx_x;  // num_features
            const auto class_idx_linear = blockIdx_y * blockDim_y * INTERNAL_BLOCK_SIZE_uz + threadIdx_x;    // num_classes

            // iterate over all support vectors using blocking to be able to cache them for faster memory accesses
            for (std::size_t sv_block = 0; sv_block < device_num_sv_; sv_block += THREAD_BLOCK_SIZE_uz) {
                // load data into scratchpad memory
                for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                    // calculate the indices to access the global data, pays attention to coalesced memory accesses
                    const auto global_feature_idx_linear = feature_idx_linear + static_cast<std::size_t>(internal) * THREAD_BLOCK_SIZE_uz;
                    const auto global_class_idx_linear = class_idx_linear + static_cast<std::size_t>(internal) * THREAD_BLOCK_SIZE_uz;

                    // store the values in the scratchpad memory
                    feature_cache(team_rank_y, internal * THREAD_BLOCK_SIZE + team_rank_x) = support_vectors_[global_feature_idx_linear * (device_num_sv_ + PADDING_SIZE_uz) + sv_block + threadIdx_y];  // SoA
                    alpha_cache(team_rank_y, internal * THREAD_BLOCK_SIZE + team_rank_x) = alpha_[global_class_idx_linear * (num_sv_ + PADDING_SIZE_uz) + sv_block + device_sv_offset_ + threadIdx_y];   // AoS
                }
                team.team_barrier();  // wait until all threads loaded their part of the data

                if constexpr (target == target_platform::cpu) {
                    // perform the dot product calculation, the sv is the fastest moving index
                    for (unsigned internal_feature = 0; internal_feature < INTERNAL_BLOCK_SIZE; ++internal_feature) {
                        for (unsigned internal_class = 0; internal_class < INTERNAL_BLOCK_SIZE; ++internal_class) {
                            real_type sum{ 0.0 };
                            for (unsigned sv = 0; sv < THREAD_BLOCK_SIZE; ++sv) {
                                sum += alpha_cache(sv, team_rank_y * INTERNAL_BLOCK_SIZE + internal_class) * feature_cache(sv, team_rank_x * INTERNAL_BLOCK_SIZE + internal_feature);
                            }
                            temp[internal_feature][internal_class] += sum;
                        }
                    }
                } else {
                    // perform the dot product calculation, the sv is the slowest moving index
                    for (unsigned sv = 0; sv < THREAD_BLOCK_SIZE; ++sv) {
                        for (unsigned internal_feature = 0; internal_feature < INTERNAL_BLOCK_SIZE; ++internal_feature) {
                            for (unsigned internal_class = 0; internal_class < INTERNAL_BLOCK_SIZE; ++internal_class) {
                                temp[internal_feature][internal_class] += alpha_cache(sv, team_rank_y * INTERNAL_BLOCK_SIZE + internal_class) * feature_cache(sv, team_rank_x * INTERNAL_BLOCK_SIZE + internal_feature);
                            }
                        }
                    }
                }
                team.team_barrier();  // wait until all threads performed their part of the calculations
            }
        }

        // calculate the indices used in the current thread
        const auto feature_idx = (blockIdx_x * blockDim_x + threadIdx_x) * INTERNAL_BLOCK_SIZE_uz;  // num_features
        const auto class_idx = (blockIdx_y * blockDim_y + threadIdx_y) * INTERNAL_BLOCK_SIZE_uz;    // num_classes

        // update the global w-vector with the locally cached values
        for (unsigned internal_feature = 0; internal_feature < INTERNAL_BLOCK_SIZE; ++internal_feature) {
            for (unsigned internal_class = 0; internal_class < INTERNAL_BLOCK_SIZE; ++internal_class) {
                // calculate the indices to access the global data
                const auto global_feature_idx = feature_idx + static_cast<std::size_t>(internal_feature);
                const auto global_class_idx = class_idx + static_cast<std::size_t>(internal_class);

                w_[global_feature_idx * (num_classes_ + PADDING_SIZE_uz) + global_class_idx] = temp[internal_feature][internal_class];  // SoA
            }
        }
    }

  private:
    /// @cond Doxygen_suppress
    device_view_type<real_type> w_;
    device_view_type<const real_type> alpha_;
    device_view_type<const real_type> support_vectors_;
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
 * @tparam target the target platform
 */
template <typename ExecutionSpace, target_platform target>
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
        prediction_{ prediction },
        w_{ w },
        rho_{ rho },
        predict_points_{ predict_points },
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
        // cast values to 32-bit unsigned int values to prevent implicit conversions
        const auto team_rank_x = static_cast<unsigned>(team.team_rank()) / THREAD_BLOCK_SIZE;
        const auto team_rank_y = static_cast<unsigned>(team.team_rank()) % THREAD_BLOCK_SIZE;

        // cast all values to 64-bit std::size_t to prevent potential 32-bit overflows
        constexpr auto INTERNAL_BLOCK_SIZE_uz = static_cast<std::size_t>(INTERNAL_BLOCK_SIZE);
        constexpr auto THREAD_BLOCK_SIZE_uz = static_cast<std::size_t>(THREAD_BLOCK_SIZE);
        constexpr auto PADDING_SIZE_uz = static_cast<std::size_t>(PADDING_SIZE);

        const auto threadIdx_x = static_cast<std::size_t>(team.team_rank()) / THREAD_BLOCK_SIZE_uz;            // current thread in team x-dimension
        const auto threadIdx_y = static_cast<std::size_t>(team.team_rank()) % THREAD_BLOCK_SIZE_uz;            // current thread in team y-dimension
        const auto blockDim_x = THREAD_BLOCK_SIZE_uz;                                                          // number of threads in team x-dimension
        const auto blockDim_y = THREAD_BLOCK_SIZE_uz;                                                          // number of threads in team y-dimension
        const auto blockIdx_x = static_cast<std::size_t>(team.league_rank()) % grid_size_x_ + grid_x_offset_;  // current team in league x-dimension + offsets if the league size is too large
        const auto blockIdx_y = static_cast<std::size_t>(team.league_rank()) / grid_size_x_ + grid_y_offset_;  // current team in league y-dimension + offsets if the league size is too large

        // create two scratchpad memory arrays used for caching
        constexpr std::size_t scratchpad_size = THREAD_BLOCK_SIZE_uz * THREAD_BLOCK_SIZE_uz * INTERNAL_BLOCK_SIZE_uz;
        real_type *scratchpad_ptr = static_cast<real_type *>(team.team_shmem().get_shmem(std::size_t{ 2 } * scratchpad_size * sizeof(real_type)));
        Kokkos::mdspan<real_type, Kokkos::dextents<std::size_t, 2>> pp_cache{ scratchpad_ptr, THREAD_BLOCK_SIZE_uz, INTERNAL_BLOCK_SIZE_uz * THREAD_BLOCK_SIZE_uz };
        Kokkos::mdspan<real_type, Kokkos::dextents<std::size_t, 2>> w_cache{ scratchpad_ptr + scratchpad_size, THREAD_BLOCK_SIZE_uz, INTERNAL_BLOCK_SIZE_uz * THREAD_BLOCK_SIZE_uz };

        // create a thread private array used for internal caching
        real_type temp[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE]{};

        {
            // calculate the indices used in the current thread, pays attention to coalesced memory accesses
            const auto pp_idx_linear = blockIdx_x * blockDim_x * INTERNAL_BLOCK_SIZE_uz + threadIdx_x;     // num_predict_points
            const auto class_idx_linear = blockIdx_y * blockDim_y * INTERNAL_BLOCK_SIZE_uz + threadIdx_x;  // num_classes

            // iterate over all features using blocking to be able to cache them for faster memory accesses
            for (std::size_t feature_block = 0; feature_block < num_features_; feature_block += THREAD_BLOCK_SIZE_uz) {
                // load data into scratchpad memory
                for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                    // calculate the indices to access the global data, pays attention to coalesced memory accesses
                    const auto global_pp_idx_linear = pp_idx_linear + static_cast<std::size_t>(internal) * THREAD_BLOCK_SIZE_uz;
                    const auto global_class_idx_linear = class_idx_linear + static_cast<std::size_t>(internal) * THREAD_BLOCK_SIZE_uz;

                    // store the values in the scratchpad memory
                    pp_cache(team_rank_y, internal * THREAD_BLOCK_SIZE + team_rank_x) = predict_points_[(feature_block + threadIdx_y) * (num_predict_points_ + PADDING_SIZE_uz) + global_pp_idx_linear];  // SoA
                    w_cache(team_rank_y, internal * THREAD_BLOCK_SIZE + team_rank_x) = w_[(feature_block + threadIdx_y) * (num_classes_ + PADDING_SIZE_uz) + global_class_idx_linear];                    // SoA
                }
                team.team_barrier();  // wait until all threads loaded their part of the data

                if constexpr (target == target_platform::cpu) {
                    // perform the dot product calculation, the feature is the fastest moving index
                    for (unsigned internal_pp = 0; internal_pp < INTERNAL_BLOCK_SIZE; ++internal_pp) {
                        for (unsigned internal_class = 0; internal_class < INTERNAL_BLOCK_SIZE; ++internal_class) {
                            real_type sum{ 0.0 };
                            for (unsigned feature = 0; feature < THREAD_BLOCK_SIZE; ++feature) {
                                sum += w_cache(feature, team_rank_y * INTERNAL_BLOCK_SIZE + internal_class) * pp_cache(feature, team_rank_x * INTERNAL_BLOCK_SIZE + internal_pp);
                            }
                            temp[internal_pp][internal_class] += sum;
                        }
                    }
                } else {
                    // perform the dot product calculation, the feature is the slowest moving index
                    for (unsigned feature = 0; feature < THREAD_BLOCK_SIZE; ++feature) {
                        for (unsigned internal_pp = 0; internal_pp < INTERNAL_BLOCK_SIZE; ++internal_pp) {
                            for (unsigned internal_class = 0; internal_class < INTERNAL_BLOCK_SIZE; ++internal_class) {
                                temp[internal_pp][internal_class] += w_cache(feature, team_rank_y * INTERNAL_BLOCK_SIZE + internal_class) * pp_cache(feature, team_rank_x * INTERNAL_BLOCK_SIZE + internal_pp);
                            }
                        }
                    }
                }
                team.team_barrier();  // wait until all threads performed their part of the calculations
            }
        }

        // calculate the indices used in the current thread
        const auto pp_idx = (blockIdx_x * blockDim_x + threadIdx_x) * INTERNAL_BLOCK_SIZE_uz;     // num_predict_points
        const auto class_idx = (blockIdx_y * blockDim_y + threadIdx_y) * INTERNAL_BLOCK_SIZE_uz;  // num_classes

        // update the global array with the local one
        for (unsigned internal_pp = 0; internal_pp < INTERNAL_BLOCK_SIZE; ++internal_pp) {
            for (unsigned internal_class = 0; internal_class < INTERNAL_BLOCK_SIZE; ++internal_class) {
                // calculate the indices to access the global data
                const auto global_pp_idx = pp_idx + static_cast<std::size_t>(internal_pp);
                const auto global_class_idx = class_idx + static_cast<std::size_t>(internal_class);

                prediction_[global_pp_idx * (num_classes_ + PADDING_SIZE_uz) + global_class_idx] = temp[internal_pp][internal_class] - rho_[global_class_idx];  // AoS
            }
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
 * @tparam target the target platform
 * @tparam kernel_function the type of the used kernel function
 * @tparam Args the types of the parameters necessary for the specific kernel function
 */
template <typename ExecutionSpace, target_platform target, kernel_function_type kernel_function, typename... Args>
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
        grid_size_x_{ grid_size_x },
        kernel_function_parameter_{ detail::make_standard_layout_tuple(std::forward<Args>(kernel_function_parameter)...) } { }

    /**
     * @brief Function call operator overload performing the actual calculation.
     * @param[in] team the Kokkos team representing the current point in the execution space
     */
    KOKKOS_INLINE_FUNCTION
    void operator()(const typename Kokkos::TeamPolicy<ExecutionSpace>::member_type &team) const {
        // cast values to 32-bit unsigned int values to prevent implicit conversions
        const auto team_rank_x = static_cast<unsigned>(team.team_rank()) / THREAD_BLOCK_SIZE;
        const auto team_rank_y = static_cast<unsigned>(team.team_rank()) % THREAD_BLOCK_SIZE;

        // cast all values to 64-bit std::size_t to prevent potential 32-bit overflows
        constexpr auto INTERNAL_BLOCK_SIZE_uz = static_cast<std::size_t>(INTERNAL_BLOCK_SIZE);
        constexpr auto THREAD_BLOCK_SIZE_uz = static_cast<std::size_t>(THREAD_BLOCK_SIZE);
        constexpr auto PADDING_SIZE_uz = static_cast<std::size_t>(PADDING_SIZE);

        const auto threadIdx_x = static_cast<std::size_t>(team.team_rank()) / THREAD_BLOCK_SIZE_uz;            // current thread in team x-dimension
        const auto threadIdx_y = static_cast<std::size_t>(team.team_rank()) % THREAD_BLOCK_SIZE_uz;            // current thread in team y-dimension
        const auto blockDim_x = THREAD_BLOCK_SIZE_uz;                                                          // number of threads in team x-dimension
        const auto blockDim_y = THREAD_BLOCK_SIZE_uz;                                                          // number of threads in team y-dimension
        const auto blockIdx_x = static_cast<std::size_t>(team.league_rank()) % grid_size_x_ + grid_x_offset_;  // current team in league x-dimension + offsets if the league size is too large
        const auto blockIdx_y = static_cast<std::size_t>(team.league_rank()) / grid_size_x_ + grid_y_offset_;  // current team in league y-dimension + offsets if the league size is too large

        // get the scratchpad memory pointer for later usage
        constexpr std::size_t scratchpad_size = THREAD_BLOCK_SIZE_uz * THREAD_BLOCK_SIZE_uz * INTERNAL_BLOCK_SIZE_uz;
        real_type *scratchpad_ptr = static_cast<real_type *>(team.team_shmem().get_shmem(std::size_t{ 2 } * scratchpad_size * sizeof(real_type)));

        // create a thread private array used for internal caching
        real_type temp[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE]{};

        {
            // reinterpret the scratchpad memory to be of shape [THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE]
            Kokkos::mdspan<real_type, Kokkos::dextents<std::size_t, 2>> pp_cache{ scratchpad_ptr, THREAD_BLOCK_SIZE_uz, INTERNAL_BLOCK_SIZE_uz * THREAD_BLOCK_SIZE_uz };
            Kokkos::mdspan<real_type, Kokkos::dextents<std::size_t, 2>> sv_cache{ scratchpad_ptr + scratchpad_size, THREAD_BLOCK_SIZE_uz, INTERNAL_BLOCK_SIZE_uz * THREAD_BLOCK_SIZE_uz };

            // calculate the indices used in the current thread, pays attention to coalesced memory accesses
            const auto pp_idx_linear = blockIdx_x * blockDim_x * INTERNAL_BLOCK_SIZE_uz + threadIdx_x;  // num_predict_points
            const auto sv_idx_linear = blockIdx_y * blockDim_y * INTERNAL_BLOCK_SIZE_uz + threadIdx_x;  // num_support_vectors

            // iterate over all features using blocking to be able to cache them for faster memory accesses
            for (std::size_t feature_block = 0; feature_block < num_features_; feature_block += THREAD_BLOCK_SIZE_uz) {
                // load data into scratchpad memory
                for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                    // calculate the indices to access the global data, pays attention to coalesced memory accesses
                    const auto global_pp_idx_linear = pp_idx_linear + static_cast<std::size_t>(internal) * THREAD_BLOCK_SIZE_uz;
                    const auto global_sv_idx_linear = sv_idx_linear + static_cast<std::size_t>(internal) * THREAD_BLOCK_SIZE_uz;

                    // store the values in the scratchpad memory
                    pp_cache(team_rank_y, internal * THREAD_BLOCK_SIZE + team_rank_x) = predict_points_[(feature_block + threadIdx_y) * (num_predict_points_ + PADDING_SIZE_uz) + global_pp_idx_linear];  // SoA
                    sv_cache(team_rank_y, internal * THREAD_BLOCK_SIZE + team_rank_x) = support_vectors_[(feature_block + threadIdx_y) * (num_sv_ + PADDING_SIZE_uz) + global_sv_idx_linear];             // SoA
                }
                team.team_barrier();  // wait until all threads loaded their part of the data

                if constexpr (target == target_platform::cpu) {
                    // perform the feature reduction calculation, the feature is the fastest moving index
                    for (unsigned internal_pp = 0; internal_pp < INTERNAL_BLOCK_SIZE; ++internal_pp) {
                        for (unsigned internal_sv = 0; internal_sv < INTERNAL_BLOCK_SIZE; ++internal_sv) {
                            real_type sum{ 0.0 };
                            for (unsigned feature = 0; feature < THREAD_BLOCK_SIZE; ++feature) {
                                sum += detail::feature_reduce<kernel_function>(sv_cache(feature, team_rank_y * INTERNAL_BLOCK_SIZE + internal_sv),
                                                                               pp_cache(feature, team_rank_x * INTERNAL_BLOCK_SIZE + internal_pp));
                            }
                            temp[internal_pp][internal_sv] += sum;
                        }
                    }
                } else {
                    // perform the feature reduction calculation, the feature is the slowest moving index
                    for (unsigned feature = 0; feature < THREAD_BLOCK_SIZE; ++feature) {
                        for (unsigned internal_pp = 0; internal_pp < INTERNAL_BLOCK_SIZE; ++internal_pp) {
                            for (unsigned internal_sv = 0; internal_sv < INTERNAL_BLOCK_SIZE; ++internal_sv) {
                                temp[internal_pp][internal_sv] += detail::feature_reduce<kernel_function>(sv_cache(feature, team_rank_y * INTERNAL_BLOCK_SIZE + internal_sv),
                                                                                                          pp_cache(feature, team_rank_x * INTERNAL_BLOCK_SIZE + internal_pp));
                            }
                        }
                    }
                }
                team.team_barrier();  // wait until all threads performed their part of the calculations
            }
        }

        // update temp using the respective kernel function
        for (unsigned internal_pp = 0; internal_pp < INTERNAL_BLOCK_SIZE; ++internal_pp) {
            for (unsigned internal_sv = 0; internal_sv < INTERNAL_BLOCK_SIZE; ++internal_sv) {
                temp[internal_pp][internal_sv] = detail::apply_kernel_function<kernel_function>(temp[internal_pp][internal_sv], kernel_function_parameter_);
            }
        }

        {
            // reinterpret the scratchpad memory to be of shape [INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE][THREAD_BLOCK_SIZE]
            Kokkos::mdspan<real_type, Kokkos::dextents<std::size_t, 2>> alpha_cache{ scratchpad_ptr, THREAD_BLOCK_SIZE_uz, INTERNAL_BLOCK_SIZE_uz * THREAD_BLOCK_SIZE_uz };
            Kokkos::mdspan<real_type, Kokkos::dextents<std::size_t, 2>> out_cache{ scratchpad_ptr + scratchpad_size, THREAD_BLOCK_SIZE_uz, INTERNAL_BLOCK_SIZE_uz * THREAD_BLOCK_SIZE_uz };

            // calculate the indices used in the current thread
            const auto pp_idx = (blockIdx_x * blockDim_x + threadIdx_x) * INTERNAL_BLOCK_SIZE_uz;  // num_predict_points
            // calculate the indices used in the current thread, pays attention to coalesced memory accesses
            const auto sv_idx_linear = blockIdx_y * blockDim_y * INTERNAL_BLOCK_SIZE_uz + threadIdx_x;  // num_support_vectors

            // iterate over all classes using blocking to be able to cache them for faster memory accesses
            for (std::size_t class_block = 0; class_block < num_classes_; class_block += THREAD_BLOCK_SIZE_uz) {
                // load data into scratchpad memory
                for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                    // calculate the indices to access the global data, pays attention to coalesced memory accesses
                    const auto global_sv_idx_linear = sv_idx_linear + static_cast<std::size_t>(internal) * THREAD_BLOCK_SIZE_uz;

                    // store the values in the scratchpad memory
                    alpha_cache(team_rank_y, internal * THREAD_BLOCK_SIZE + team_rank_x) = alpha_[(class_block + threadIdx_y) * (num_sv_ + PADDING_SIZE_uz) + global_sv_idx_linear];  // AoS
                    // the bias (rho) must only be applied once for all support vectors
                    if (blockIdx_y == std::size_t{ 0 }) {
                        out_cache(team_rank_y, internal * THREAD_BLOCK_SIZE + team_rank_x) = -rho_[class_block + threadIdx_y];
                    } else {
                        out_cache(team_rank_y, internal * THREAD_BLOCK_SIZE + team_rank_x) = real_type{ 0.0 };
                    }
                }
                team.team_barrier();  // wait until all threads loaded their part of the data

                // calculate intermediate results and store them in scratchpad memory
                for (unsigned class_idx = 0; class_idx < THREAD_BLOCK_SIZE; ++class_idx) {
                    for (unsigned internal_pp = 0; internal_pp < INTERNAL_BLOCK_SIZE; ++internal_pp) {
                        for (unsigned internal_sv = 0; internal_sv < INTERNAL_BLOCK_SIZE; ++internal_sv) {
                            out_cache((class_idx + team_rank_y) % THREAD_BLOCK_SIZE, internal_pp * THREAD_BLOCK_SIZE + team_rank_x) +=
                                temp[internal_pp][internal_sv] * alpha_cache((class_idx + team_rank_y) % THREAD_BLOCK_SIZE, team_rank_y * INTERNAL_BLOCK_SIZE + internal_sv);
                        }
                    }
                    team.team_barrier();  // wait until all threads performed their part of the calculations
                }

                // atomically add the intermediate cached results to the prediction
                for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                    // calculate the indices to access the global data
                    const auto global_pp_idx = pp_idx + static_cast<std::size_t>(internal);

                    Kokkos::atomic_add(&prediction_[global_pp_idx * (num_classes_ + PADDING_SIZE_uz) + class_block + threadIdx_y], out_cache(team_rank_y, internal * THREAD_BLOCK_SIZE + team_rank_x));
                }
                team.team_barrier();  // wait until all threads updated their part of the prediction
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

#endif  // PLSSVM_BACKENDS_KOKKOS_PREDICT_KERNEL_HPP_
