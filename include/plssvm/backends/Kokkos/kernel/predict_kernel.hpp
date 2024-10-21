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

#include "plssvm/backends/Kokkos/detail/device_ptr.hpp"        // TODO: view type aliases
#include "plssvm/backends/Kokkos/kernel/kernel_functions.hpp"  // plssvm::kokkos::detail::{feature_reduce, apply_kernel_function}
#include "plssvm/constants.hpp"                                // plssvm::{real_type, THREAD_BLOCK_SIZE, INTERNAL_BLOCK_SIZE, FEATURE_BLOCK_SIZE, PADDING_SIZE}
#include "plssvm/kernel_function_types.hpp"                    // plssvm::kernel_function_type

#include "Kokkos_Core.hpp"  // TODO: Kokkos::atomic_add

namespace plssvm::kokkos::detail {

/**
 * @brief Calculate the `q` vector used to speedup the prediction using the linear kernel function.
 */
class device_kernel_w_linear {
  public:
    /**
     * @brief Initialize the Kokkos kernel function object.
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
    device_kernel_w_linear(device_view_type<real_type> w_d, device_view_type<const real_type> alpha_d, device_view_type<const real_type> sv_d, const std::size_t num_classes, const std::size_t num_sv, const std::size_t device_specific_num_sv, const std::size_t sv_offset, const std::size_t grid_x_offset, const std::size_t grid_y_offset, const std::size_t grid_size_x) :
        w_d_{ w_d },
        alpha_d_{ alpha_d },
        sv_d_{ sv_d },
        num_classes_{ num_classes },
        num_sv_{ num_sv },
        device_specific_num_sv_{ device_specific_num_sv },
        sv_offset_{ sv_offset },
        grid_x_offset_{ grid_x_offset },
        grid_y_offset_{ grid_y_offset },
        grid_size_x_{ grid_size_x } { }

    KOKKOS_INLINE_FUNCTION
    void operator()(const Kokkos::TeamPolicy<>::member_type &team) const {
        // cast all values to 64-bit std::size_t to prevent potential 32-bit overflows
        const auto INTERNAL_BLOCK_SIZE_sz = static_cast<std::size_t>(INTERNAL_BLOCK_SIZE);
        const auto THREAD_BLOCK_SIZE_sz = static_cast<std::size_t>(THREAD_BLOCK_SIZE);
        const auto PADDING_SIZE_sz = static_cast<std::size_t>(PADDING_SIZE);
        const auto threadIdx_x = static_cast<std::size_t>(team.team_rank()) / THREAD_BLOCK_SIZE_sz;            // current thread in block x-dimension
        const auto threadIdx_y = static_cast<std::size_t>(team.team_rank()) % THREAD_BLOCK_SIZE_sz;            // current thread in block y-dimension
        const auto blockDim_x = THREAD_BLOCK_SIZE_sz;                                                          // number of threads in block x-dimension
        const auto blockDim_y = THREAD_BLOCK_SIZE_sz;                                                          // number of threads in block y-dimension
        const auto blockIdx_x = static_cast<std::size_t>(team.league_rank()) % grid_size_x_ + grid_x_offset_;  // current block in grid x-dimension + offsets if the grid size would be too large
        const auto blockIdx_y = static_cast<std::size_t>(team.league_rank()) / grid_size_x_ + grid_y_offset_;  // current block in grid y-dimension + offsets if the grid size would be too large

        // calculate the indices used in the current thread
        const auto feature_idx = (blockIdx_x * blockDim_x + threadIdx_x) * INTERNAL_BLOCK_SIZE_sz;
        const auto feature_idx_linear = blockIdx_x * blockDim_x * INTERNAL_BLOCK_SIZE_sz + threadIdx_x;
        const auto class_idx = (blockIdx_y * blockDim_y + threadIdx_y) * INTERNAL_BLOCK_SIZE_sz;
        const auto class_idx_linear = blockIdx_y * blockDim_y * INTERNAL_BLOCK_SIZE_sz + threadIdx_x;

        // create the shared memory arrays used for caching data point features
        constexpr std::size_t shmem_size = THREAD_BLOCK_SIZE * THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE;
        real_type *data_cache_ptr = static_cast<real_type *>(team.team_shmem().get_shmem(2 * shmem_size));
        Kokkos::mdspan<real_type, Kokkos::dextents<std::size_t, 2>> data_cache_feature{ data_cache_ptr, THREAD_BLOCK_SIZE, INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE };
        Kokkos::mdspan<real_type, Kokkos::dextents<std::size_t, 2>> data_cache_alpha{ data_cache_ptr + shmem_size, THREAD_BLOCK_SIZE, INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE };

        // create a thread private array used for internal caching
        real_type temp[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE]{};

        // iterate over all support vectors using blocking to be able to cache them for faster memory accesses
        for (std::size_t sv = 0; sv < device_specific_num_sv_; sv += THREAD_BLOCK_SIZE_sz) {
            // load data into shared memory
            for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                const auto global_feature_idx = feature_idx_linear + static_cast<std::size_t>(internal) * THREAD_BLOCK_SIZE_sz;
                const auto global_class_idx = class_idx_linear + static_cast<std::size_t>(internal) * THREAD_BLOCK_SIZE_sz;

                data_cache_feature(threadIdx_y, internal * THREAD_BLOCK_SIZE + threadIdx_x) = sv_d_[global_feature_idx * (device_specific_num_sv_ + PADDING_SIZE_sz) + sv + threadIdx_y];  // SoA
                data_cache_alpha(threadIdx_y, internal * THREAD_BLOCK_SIZE + threadIdx_x) = alpha_d_[global_class_idx * (num_sv_ + PADDING_SIZE_sz) + sv + sv_offset_ + threadIdx_y];      // AoS
            }
            team.team_barrier();  // wait until all threads loaded their part of the data

            // perform the dot product calculation
            for (unsigned block_dim = 0; block_dim < THREAD_BLOCK_SIZE; ++block_dim) {
                for (unsigned internal_feature = 0; internal_feature < INTERNAL_BLOCK_SIZE; ++internal_feature) {
                    for (unsigned internal_class = 0; internal_class < INTERNAL_BLOCK_SIZE; ++internal_class) {
                        temp[internal_feature][internal_class] += data_cache_alpha(block_dim, threadIdx_y * INTERNAL_BLOCK_SIZE + internal_class) * data_cache_feature(block_dim, threadIdx_x * INTERNAL_BLOCK_SIZE + internal_feature);
                    }
                }
            }
            team.team_barrier();  // wait until all threads performed their part of the calculations
        }

        // update global array with local one
        for (unsigned internal_feature = 0; internal_feature < INTERNAL_BLOCK_SIZE; ++internal_feature) {
            for (unsigned internal_class = 0; internal_class < INTERNAL_BLOCK_SIZE; ++internal_class) {
                const auto global_feature_idx = feature_idx + static_cast<std::size_t>(internal_feature);
                const auto global_class_idx = class_idx + static_cast<std::size_t>(internal_class);

                w_d_[global_feature_idx * (num_classes_ + PADDING_SIZE_sz) + global_class_idx] = temp[internal_feature][internal_class];
            }
        }
    }

  private:
    /// @cond Doxygen_suppress
    device_view_type<real_type> w_d_;
    device_view_type<const real_type> alpha_d_;
    device_view_type<const real_type> sv_d_;
    const std::size_t num_classes_;
    const std::size_t num_sv_;
    const std::size_t device_specific_num_sv_;
    const std::size_t sv_offset_;
    const std::size_t grid_x_offset_;
    const std::size_t grid_y_offset_;
    const std::size_t grid_size_x_;
    /// @endcond
};

/**
 * @brief Predict the @p predict_points_d using the linear kernel speeding up the calculation using the @p w_d vector.
 */
class device_kernel_predict_linear {
  public:
    /**
     * @brief Initialize the Kokkos kernel function object.
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
    device_kernel_predict_linear(device_view_type<real_type> prediction_d, device_view_type<const real_type> w_d, device_view_type<const real_type> rho_d, device_view_type<const real_type> predict_points_d, const std::size_t num_classes, const std::size_t num_predict_points, const std::size_t num_features, const std::size_t grid_x_offset, const std::size_t grid_y_offset, const std::size_t grid_size_x) :
        prediction_d_{ prediction_d },
        w_d_{ w_d },
        rho_d_{ rho_d },
        predict_points_d_{ predict_points_d },
        num_classes_{ num_classes },
        num_predict_points_{ num_predict_points },
        num_features_{ num_features },
        grid_x_offset_{ grid_x_offset },
        grid_y_offset_{ grid_y_offset },
        grid_size_x_{ grid_size_x } { }

    KOKKOS_INLINE_FUNCTION
    void operator()(const Kokkos::TeamPolicy<>::member_type &team) const {
        // cast all values to 64-bit std::size_t to prevent potential 32-bit overflows
        const auto INTERNAL_BLOCK_SIZE_sz = static_cast<std::size_t>(INTERNAL_BLOCK_SIZE);
        const auto THREAD_BLOCK_SIZE_sz = static_cast<std::size_t>(THREAD_BLOCK_SIZE);
        const auto FEATURE_BLOCK_SIZE_sz = static_cast<std::size_t>(FEATURE_BLOCK_SIZE);
        const auto PADDING_SIZE_sz = static_cast<std::size_t>(PADDING_SIZE);
        const auto threadIdx_x = static_cast<std::size_t>(team.team_rank()) / THREAD_BLOCK_SIZE_sz;            // current thread in block x-dimension
        const auto threadIdx_y = static_cast<std::size_t>(team.team_rank()) % THREAD_BLOCK_SIZE_sz;            // current thread in block y-dimension
        const auto blockDim_x = THREAD_BLOCK_SIZE_sz;                                                          // number of threads in block x-dimension
        const auto blockDim_y = THREAD_BLOCK_SIZE_sz;                                                          // number of threads in block y-dimension
        const auto blockIdx_x = static_cast<std::size_t>(team.league_rank()) % grid_size_x_ + grid_x_offset_;  // current block in grid x-dimension + offsets if the grid size would be too large
        const auto blockIdx_y = static_cast<std::size_t>(team.league_rank()) / grid_size_x_ + grid_y_offset_;  // current block in grid y-dimension + offsets if the grid size would be too large

        // calculate the indices used in the current thread
        const auto pp_idx = (blockIdx_x * blockDim_x + threadIdx_x) * INTERNAL_BLOCK_SIZE_sz;
        const auto pp_idx_linear = blockIdx_x * blockDim_x * INTERNAL_BLOCK_SIZE_sz + threadIdx_x;
        const auto class_idx = (blockIdx_y * blockDim_y + threadIdx_y) * INTERNAL_BLOCK_SIZE_sz;
        const auto class_idx_linear = blockIdx_y * blockDim_y * INTERNAL_BLOCK_SIZE_sz + threadIdx_x;

        // create the shared memory arrays used for caching data point features
        constexpr std::size_t shmem_size = FEATURE_BLOCK_SIZE * THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE;
        real_type *data_cache_ptr = static_cast<real_type *>(team.team_shmem().get_shmem(2 * shmem_size));
        Kokkos::mdspan<real_type, Kokkos::dextents<std::size_t, 2>> data_cache_pp{ data_cache_ptr, FEATURE_BLOCK_SIZE, INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE };
        Kokkos::mdspan<real_type, Kokkos::dextents<std::size_t, 2>> data_cache_w{ data_cache_ptr + shmem_size, FEATURE_BLOCK_SIZE, INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE };

        // create a thread private array used for internal caching
        real_type temp[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE]{};

        // iterate over all features using blocking to be able to cache them for faster memory accesses
        for (std::size_t dim = 0; dim < num_features_; dim += FEATURE_BLOCK_SIZE_sz) {
            // load data into shared memory
            for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                const auto global_pp_idx = pp_idx_linear + static_cast<std::size_t>(internal) * THREAD_BLOCK_SIZE_sz;
                const auto global_class_idx = class_idx_linear + static_cast<std::size_t>(internal) * THREAD_BLOCK_SIZE_sz;

                // FEATURE_BLOCK_SIZE = 2 * THREAD_BLOCK_SIZE -> store twice as many values in the shared memory
                data_cache_pp(threadIdx_y, internal * THREAD_BLOCK_SIZE + threadIdx_x) = predict_points_d_[(dim + threadIdx_y) * (num_predict_points_ + PADDING_SIZE_sz) + global_pp_idx];
                data_cache_pp(threadIdx_y + THREAD_BLOCK_SIZE, internal * THREAD_BLOCK_SIZE + threadIdx_x) = predict_points_d_[(dim + threadIdx_y + THREAD_BLOCK_SIZE_sz) * (num_predict_points_ + PADDING_SIZE_sz) + global_pp_idx];
                data_cache_w(threadIdx_y, internal * THREAD_BLOCK_SIZE + threadIdx_x) = w_d_[(dim + threadIdx_y) * (num_classes_ + PADDING_SIZE_sz) + global_class_idx];
                data_cache_w(threadIdx_y + THREAD_BLOCK_SIZE, internal * THREAD_BLOCK_SIZE + threadIdx_x) = w_d_[(dim + threadIdx_y + THREAD_BLOCK_SIZE_sz) * (num_classes_ + PADDING_SIZE_sz) + global_class_idx];
            }
            team.team_barrier();  // wait until all threads loaded their part of the data

            // perform the dot product calculation
            for (unsigned block_dim = 0; block_dim < FEATURE_BLOCK_SIZE; ++block_dim) {
                for (unsigned internal_pd = 0; internal_pd < INTERNAL_BLOCK_SIZE; ++internal_pd) {
                    for (unsigned internal_class = 0; internal_class < INTERNAL_BLOCK_SIZE; ++internal_class) {
                        temp[internal_pd][internal_class] += data_cache_w(block_dim, threadIdx_y * INTERNAL_BLOCK_SIZE + internal_class) * data_cache_pp(block_dim, threadIdx_x * INTERNAL_BLOCK_SIZE + internal_pd);
                    }
                }
            }
            team.team_barrier();  // wait until all threads performed their part of the calculations
        }

        // update global array with local one
        for (unsigned internal_pd = 0; internal_pd < INTERNAL_BLOCK_SIZE; ++internal_pd) {
            for (unsigned internal_class = 0; internal_class < INTERNAL_BLOCK_SIZE; ++internal_class) {
                const auto global_pp_idx = pp_idx + static_cast<std::size_t>(internal_pd);
                const auto global_class_idx = class_idx + static_cast<std::size_t>(internal_class);

                prediction_d_[global_pp_idx * (num_classes_ + PADDING_SIZE_sz) + global_class_idx] = temp[internal_pd][internal_class] - rho_d_[global_class_idx];
            }
        }
    }

  private:
    /// @cond Doxygen_suppress
    device_view_type<real_type> prediction_d_;
    device_view_type<const real_type> w_d_;
    device_view_type<const real_type> rho_d_;
    device_view_type<const real_type> predict_points_d_;
    const std::size_t num_classes_;
    const std::size_t num_predict_points_;
    const std::size_t num_features_;
    const std::size_t grid_x_offset_;
    const std::size_t grid_y_offset_;
    const std::size_t grid_size_x_;
    /// @endcond
};

/**
 * @brief Predict the @p predict_points_d using the @p kernel_function.
 * @tparam kernel_function the type of the used kernel function
 * @tparam Args the types of the parameters necessary for the specific kernel function
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
    device_kernel_predict(device_view_type<real_type> prediction_d, device_view_type<const real_type> alpha_d, device_view_type<const real_type> rho_d, device_view_type<const real_type> sv_d, device_view_type<const real_type> predict_points_d, const std::size_t num_classes, const std::size_t num_sv, const std::size_t num_predict_points, const std::size_t num_features, const std::size_t grid_x_offset, const std::size_t grid_y_offset, const std::size_t grid_size_x, Args... kernel_function_parameter) :
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
        grid_size_x_{ grid_size_x },
        kernel_function_parameter_{ detail::make_standard_layout_tuple(std::forward<Args>(kernel_function_parameter)...) } { }

    KOKKOS_INLINE_FUNCTION
    void operator()(const Kokkos::TeamPolicy<>::member_type &team) const {
        // cast all values to 64-bit std::size_t to prevent potential 32-bit overflows
        const auto INTERNAL_BLOCK_SIZE_sz = static_cast<std::size_t>(INTERNAL_BLOCK_SIZE);
        const auto THREAD_BLOCK_SIZE_sz = static_cast<std::size_t>(THREAD_BLOCK_SIZE);
        const auto FEATURE_BLOCK_SIZE_sz = static_cast<std::size_t>(FEATURE_BLOCK_SIZE);
        const auto PADDING_SIZE_sz = static_cast<std::size_t>(PADDING_SIZE);
        const auto threadIdx_x = static_cast<std::size_t>(team.team_rank()) / THREAD_BLOCK_SIZE_sz;            // current thread in block x-dimension
        const auto threadIdx_y = static_cast<std::size_t>(team.team_rank()) % THREAD_BLOCK_SIZE_sz;            // current thread in block y-dimension
        const auto blockDim_x = THREAD_BLOCK_SIZE_sz;                                                          // number of threads in block x-dimension
        const auto blockDim_y = THREAD_BLOCK_SIZE_sz;                                                          // number of threads in block y-dimension
        const auto blockIdx_x = static_cast<std::size_t>(team.league_rank()) % grid_size_x_ + grid_x_offset_;  // current block in grid x-dimension + offsets if the grid size would be too large
        const auto blockIdx_y = static_cast<std::size_t>(team.league_rank()) / grid_size_x_ + grid_y_offset_;  // current block in grid y-dimension + offsets if the grid size would be too large

        // calculate the indices used in the current thread
        const auto pp_idx = (blockIdx_x * blockDim_x + threadIdx_x) * INTERNAL_BLOCK_SIZE_sz;
        const auto pp_idx_linear = blockIdx_x * blockDim_x * INTERNAL_BLOCK_SIZE_sz + threadIdx_x;
        const auto sv_idx_linear = blockIdx_y * blockDim_y * INTERNAL_BLOCK_SIZE_sz + threadIdx_x;

        constexpr std::size_t shmem_size = FEATURE_BLOCK_SIZE * THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE;
        real_type *data_cache_ptr = static_cast<real_type *>(team.team_shmem().get_shmem(2 * shmem_size));

        // create a thread private array used for internal caching
        real_type temp[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE]{};

        {
            // create the shared memory arrays used for caching data point features
            Kokkos::mdspan<real_type, Kokkos::dextents<std::size_t, 2>> data_cache_pp{ data_cache_ptr, FEATURE_BLOCK_SIZE, INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE };
            Kokkos::mdspan<real_type, Kokkos::dextents<std::size_t, 2>> data_cache_sv{ data_cache_ptr + shmem_size, FEATURE_BLOCK_SIZE, INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE };

            // iterate over all features using blocking to be able to cache them for faster memory accesses
            for (std::size_t dim = 0; dim < num_features_; dim += FEATURE_BLOCK_SIZE_sz) {
                // load data into shared memory
                for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                    const auto global_pp_idx = pp_idx_linear + static_cast<std::size_t>(internal) * THREAD_BLOCK_SIZE;
                    const auto global_sv_idx = sv_idx_linear + static_cast<std::size_t>(internal) * THREAD_BLOCK_SIZE;

                    // FEATURE_BLOCK_SIZE = 2 * THREAD_BLOCK_SIZE -> store twice as many values in the shared memory
                    data_cache_pp(threadIdx_y, internal * THREAD_BLOCK_SIZE + threadIdx_x) = predict_points_d_[(dim + threadIdx_y) * (num_predict_points_ + PADDING_SIZE_sz) + global_pp_idx];
                    data_cache_pp(threadIdx_y + THREAD_BLOCK_SIZE, internal * THREAD_BLOCK_SIZE + threadIdx_x) = predict_points_d_[(dim + threadIdx_y + THREAD_BLOCK_SIZE_sz) * (num_predict_points_ + PADDING_SIZE_sz) + global_pp_idx];
                    data_cache_sv(threadIdx_y, internal * THREAD_BLOCK_SIZE + threadIdx_x) = sv_d_[(dim + threadIdx_y) * (num_sv_ + PADDING_SIZE_sz) + global_sv_idx];
                    data_cache_sv(threadIdx_y + THREAD_BLOCK_SIZE, internal * THREAD_BLOCK_SIZE + threadIdx_x) = sv_d_[(dim + threadIdx_y + THREAD_BLOCK_SIZE_sz) * (num_sv_ + PADDING_SIZE_sz) + global_sv_idx];
                }
                team.team_barrier();  // wait until all threads loaded their part of the data

                // perform the feature reduction calculation
                for (unsigned block_dim = 0; block_dim < FEATURE_BLOCK_SIZE; ++block_dim) {
                    for (unsigned internal_pd = 0; internal_pd < INTERNAL_BLOCK_SIZE; ++internal_pd) {
                        for (unsigned internal_sv = 0; internal_sv < INTERNAL_BLOCK_SIZE; ++internal_sv) {
                            temp[internal_pd][internal_sv] += detail::feature_reduce<kernel_function>(data_cache_sv(block_dim, threadIdx_y * INTERNAL_BLOCK_SIZE + internal_sv),
                                                                                                      data_cache_pp(block_dim, threadIdx_x * INTERNAL_BLOCK_SIZE + internal_pd));
                        }
                    }
                }
                team.team_barrier();  // wait until all threads performed their part of the calculations
            }
        }

        // update temp using the respective kernel function
        for (unsigned internal_pd = 0; internal_pd < INTERNAL_BLOCK_SIZE; ++internal_pd) {
            for (unsigned internal_sv = 0; internal_sv < INTERNAL_BLOCK_SIZE; ++internal_sv) {
                temp[internal_pd][internal_sv] = detail::apply_kernel_function<kernel_function>(temp[internal_pd][internal_sv], kernel_function_parameter_);
            }
        }

        {
            // create the shared memory arrays used for caching data point features
            Kokkos::mdspan<real_type, Kokkos::dextents<std::size_t, 2>> alpha_cache{ data_cache_ptr, FEATURE_BLOCK_SIZE, INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE };
            Kokkos::mdspan<real_type, Kokkos::dextents<std::size_t, 2>> out_cache{ data_cache_ptr + shmem_size, FEATURE_BLOCK_SIZE, INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE };

            // iterate over all features using blocking to be able to cache them for faster memory accesses
            for (std::size_t dim = 0; dim < num_classes_; dim += FEATURE_BLOCK_SIZE_sz) {
                // load data into shared memory
                for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                    const std::size_t global_sv_idx = sv_idx_linear + internal * THREAD_BLOCK_SIZE;

                    // FEATURE_BLOCK_SIZE = 2 * THREAD_BLOCK_SIZE -> store twice as many values in the shared memory
                    alpha_cache(threadIdx_y, internal * THREAD_BLOCK_SIZE + threadIdx_x) = alpha_d_[(dim + threadIdx_y) * (num_sv_ + PADDING_SIZE_sz) + global_sv_idx];
                    alpha_cache(threadIdx_y + THREAD_BLOCK_SIZE, internal * THREAD_BLOCK_SIZE + threadIdx_x) = alpha_d_[(dim + threadIdx_y + THREAD_BLOCK_SIZE_sz) * (num_sv_ + PADDING_SIZE_sz) + global_sv_idx];

                    // the bias (rho) must only be applied once for all support vectors
                    if (blockIdx_y == 0ull) {
                        out_cache(threadIdx_y, internal * THREAD_BLOCK_SIZE + threadIdx_x) = -rho_d_[dim + threadIdx_y];
                        out_cache(threadIdx_y + THREAD_BLOCK_SIZE, internal * THREAD_BLOCK_SIZE + threadIdx_x) = -rho_d_[dim + threadIdx_y + THREAD_BLOCK_SIZE_sz];
                    } else {
                        out_cache(threadIdx_y, internal * THREAD_BLOCK_SIZE + threadIdx_x) = real_type{ 0.0 };
                        out_cache(threadIdx_y + THREAD_BLOCK_SIZE, internal * THREAD_BLOCK_SIZE + threadIdx_x) = real_type{ 0.0 };
                    }
                }
                team.team_barrier();  // wait until all threads loaded their part of the data

                // calculate intermediate results and store them in shared memory
                for (unsigned class_idx = 0; class_idx < FEATURE_BLOCK_SIZE; ++class_idx) {
                    for (unsigned internal_pd = 0; internal_pd < INTERNAL_BLOCK_SIZE; ++internal_pd) {
                        for (unsigned internal_sv = 0; internal_sv < INTERNAL_BLOCK_SIZE; ++internal_sv) {
                            out_cache((class_idx + threadIdx_y) % FEATURE_BLOCK_SIZE, internal_pd * THREAD_BLOCK_SIZE + threadIdx_x) +=
                                temp[internal_pd][internal_sv] * alpha_cache((class_idx + threadIdx_y) % FEATURE_BLOCK_SIZE, threadIdx_y * INTERNAL_BLOCK_SIZE + internal_sv);
                        }
                    }
                    team.team_barrier();  // wait until all threads performed their part of the calculations
                }

                // add intermediate cached results to prediction_d
                for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                    const auto global_pp_idx = pp_idx + static_cast<std::size_t>(internal);

                    Kokkos::atomic_add(&prediction_d_[global_pp_idx * (num_classes_ + PADDING_SIZE_sz) + dim + threadIdx_y], out_cache(threadIdx_y, internal * THREAD_BLOCK_SIZE + threadIdx_x));
                    Kokkos::atomic_add(&prediction_d_[global_pp_idx * (num_classes_ + PADDING_SIZE_sz) + dim + threadIdx_y + THREAD_BLOCK_SIZE_sz], out_cache(threadIdx_y + THREAD_BLOCK_SIZE, internal * THREAD_BLOCK_SIZE + threadIdx_x));
                }
                team.team_barrier();  // wait until all threads updated their part of the prediction
            }
        }
    }

  private:
    /// @cond Doxygen_suppress
    device_view_type<real_type> prediction_d_;
    device_view_type<const real_type> alpha_d_;
    device_view_type<const real_type> rho_d_;
    device_view_type<const real_type> sv_d_;
    device_view_type<const real_type> predict_points_d_;
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
