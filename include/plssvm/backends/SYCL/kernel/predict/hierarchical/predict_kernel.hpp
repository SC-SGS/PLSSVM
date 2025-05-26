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

#include "plssvm/backends/SYCL/detail/atomics.hpp"           // plssvm::sycl::detail::atomic_op
#include "plssvm/backends/SYCL/kernel/kernel_functions.hpp"  // plssvm::sycl::detail::{feature_reduce, apply_kernel_function}
#include "plssvm/constants.hpp"                              // plssvm::{real_type, THREAD_BLOCK_SIZE, INTERNAL_BLOCK_SIZE, PADDING_SIZE}
#include "plssvm/kernel_function_types.hpp"                  // plssvm::kernel_function_type

#include "sycl/sycl.hpp"  // sycl::group, sycl::private_memory, sycl::h_item

#include <cstddef>  // std::size_t
#include <tuple>    // std::tuple, std::make_tuple

namespace plssvm::sycl::detail::hierarchical {

/**
 * @brief Calculate the `q` vector used to speedup the prediction using the linear kernel function.
 * @details Uses SYCL's hierarchical data parallel kernels.
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
     * @param[in] group indices representing the current point in the execution space
     */
    void operator()(::sycl::group<2> group) const {
        // allocate shared memory
        real_type data_cache_feature[THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];
        real_type data_cache_alpha[THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];

        // calculate the indices used in the current work-item
        ::sycl::private_memory<std::size_t, 2> feature_idx{ group };
        ::sycl::private_memory<std::size_t, 2> feature_idx_linear{ group };
        ::sycl::private_memory<std::size_t, 2> class_idx{ group };
        ::sycl::private_memory<std::size_t, 2> class_idx_linear{ group };

        ::sycl::private_memory<real_type[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE], 2> temp{ group };

        // initialize private and local variables
        group.parallel_for_work_item([&](::sycl::h_item<2> idx) {
            const std::size_t threadIdx_x = idx.get_local_id(0);       // current thread in block x-dimension
            const std::size_t threadIdx_y = idx.get_local_id(1);       // current thread in block y-dimension
            const std::size_t blockDim_x = idx.get_local_range(0);     // number of threads in block x-dimension
            const std::size_t blockDim_y = idx.get_local_range(1);     // number of threads in block y-dimension
            const std::size_t blockIdx_x = group[0] + grid_x_offset_;  // current block in grid x-dimension + offsets if the grid size would be too large
            const std::size_t blockIdx_y = group[1] + grid_y_offset_;  // current block in grid y-dimension + offsets if the grid size would be too large

            const auto INTERNAL_BLOCK_SIZE_uz = static_cast<std::size_t>(INTERNAL_BLOCK_SIZE);

            // indices
            feature_idx(idx) = (blockIdx_y * blockDim_y + threadIdx_y) * INTERNAL_BLOCK_SIZE_uz;
            feature_idx_linear(idx) = blockIdx_y * blockDim_y * INTERNAL_BLOCK_SIZE_uz + threadIdx_y;
            class_idx(idx) = (blockIdx_x * blockDim_x + threadIdx_x) * INTERNAL_BLOCK_SIZE_uz;
            class_idx_linear(idx) = blockIdx_x * blockDim_x * INTERNAL_BLOCK_SIZE_uz + threadIdx_y;

            // initialize private temp matrix to zero
            for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                    temp(idx)[internal_i][internal_j] = real_type{ 0.0 };
                }
            }
        });

        // implicit group barrier

        // iterate over all support vectors using blocking to be able to cache them for faster memory accesses
        for (std::size_t sv = 0; sv < device_specific_num_sv_; sv += THREAD_BLOCK_SIZE) {
            // load data into local memory
            group.parallel_for_work_item([&](::sycl::h_item<2> idx) {
                const auto local_id_0 = static_cast<unsigned>(idx.get_local_id(0));
                const auto local_id_1 = static_cast<unsigned>(idx.get_local_id(1));

                const std::size_t threadIdx_x = idx.get_local_id(0);

                const auto THREAD_BLOCK_SIZE_uz = static_cast<std::size_t>(THREAD_BLOCK_SIZE);
                const auto PADDING_SIZE_uz = static_cast<std::size_t>(PADDING_SIZE);

                for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                    const auto global_class_idx = class_idx_linear(idx) + static_cast<std::size_t>(internal) * THREAD_BLOCK_SIZE_uz;
                    const auto global_feature_idx = feature_idx_linear(idx) + static_cast<std::size_t>(internal) * THREAD_BLOCK_SIZE_uz;

                    data_cache_feature[local_id_0][internal * THREAD_BLOCK_SIZE + local_id_1] = sv_d_[global_feature_idx * (device_specific_num_sv_ + PADDING_SIZE_uz) + sv + threadIdx_x];  // SoA
                    data_cache_alpha[local_id_0][internal * THREAD_BLOCK_SIZE + local_id_1] = alpha_d_[global_class_idx * (num_sv_ + PADDING_SIZE_uz) + sv + sv_offset_ + threadIdx_x];      // AoS
                }
            });

            // implicit group barrier

            // perform the dot product calculation
            group.parallel_for_work_item([&](::sycl::h_item<2> idx) {
                const auto local_id_0 = static_cast<unsigned>(idx.get_local_id(0));
                const auto local_id_1 = static_cast<unsigned>(idx.get_local_id(1));

                for (unsigned block_dim = 0; block_dim < THREAD_BLOCK_SIZE; ++block_dim) {
                    for (unsigned internal_feature = 0; internal_feature < INTERNAL_BLOCK_SIZE; ++internal_feature) {
                        for (unsigned internal_class = 0; internal_class < INTERNAL_BLOCK_SIZE; ++internal_class) {
                            temp(idx)[internal_feature][internal_class] += data_cache_alpha[block_dim][local_id_0 * INTERNAL_BLOCK_SIZE + internal_class] * data_cache_feature[block_dim][local_id_1 * INTERNAL_BLOCK_SIZE + internal_feature];
                        }
                    }
                }
            });

            // implicit group barrier
        }

        // update global array with local one
        group.parallel_for_work_item([&](::sycl::h_item<2> idx) {
            const auto PADDING_SIZE_uz = static_cast<std::size_t>(PADDING_SIZE);

            for (unsigned internal_feature = 0; internal_feature < INTERNAL_BLOCK_SIZE; ++internal_feature) {
                for (unsigned internal_class = 0; internal_class < INTERNAL_BLOCK_SIZE; ++internal_class) {
                    const auto global_class_idx = class_idx(idx) + static_cast<std::size_t>(internal_class);
                    const auto global_feature_idx = feature_idx(idx) + static_cast<std::size_t>(internal_feature);

                    w_d_[global_feature_idx * (num_classes_ + PADDING_SIZE_uz) + global_class_idx] = temp(idx)[internal_feature][internal_class];
                }
            }
        });
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
 * @details Uses SYCL's hierarchical data parallel kernels.
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
     * @param[in] group indices representing the current point in the execution space
     */
    void operator()(::sycl::group<2> group) const {
        // allocate shared memory
        real_type data_cache_pp[THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];
        real_type data_cache_w[THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];

        // calculate the indices used in the current work-item
        ::sycl::private_memory<std::size_t, 2> pp_idx{ group };
        ::sycl::private_memory<std::size_t, 2> pp_idx_linear{ group };
        ::sycl::private_memory<std::size_t, 2> class_idx{ group };
        ::sycl::private_memory<std::size_t, 2> class_idx_linear{ group };

        ::sycl::private_memory<real_type[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE], 2> temp{ group };

        // initialize private and local variables
        group.parallel_for_work_item([&](::sycl::h_item<2> idx) {
            const std::size_t threadIdx_x = idx.get_local_id(0);       // current thread in block x-dimension
            const std::size_t threadIdx_y = idx.get_local_id(1);       // current thread in block y-dimension
            const std::size_t blockDim_x = idx.get_local_range(0);     // number of threads in block x-dimension
            const std::size_t blockDim_y = idx.get_local_range(1);     // number of threads in block y-dimension
            const std::size_t blockIdx_x = group[0] + grid_x_offset_;  // current block in grid x-dimension + offsets if the grid size would be too large
            const std::size_t blockIdx_y = group[1] + grid_y_offset_;  // current block in grid y-dimension + offsets if the grid size would be too large

            const auto INTERNAL_BLOCK_SIZE_uz = static_cast<std::size_t>(INTERNAL_BLOCK_SIZE);

            // indices
            pp_idx(idx) = (blockIdx_y * blockDim_y + threadIdx_y) * INTERNAL_BLOCK_SIZE_uz;
            pp_idx_linear(idx) = blockIdx_y * blockDim_y * INTERNAL_BLOCK_SIZE_uz + threadIdx_y;
            class_idx(idx) = (blockIdx_x * blockDim_x + threadIdx_x) * INTERNAL_BLOCK_SIZE_uz;
            class_idx_linear(idx) = blockIdx_x * blockDim_x * INTERNAL_BLOCK_SIZE_uz + threadIdx_y;

            // initialize private temp matrix to zero
            for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                    temp(idx)[internal_i][internal_j] = real_type{ 0.0 };
                }
            }
        });

        // implicit group barrier

        // iterate over all support vectors using blocking to be able to cache them for faster memory accesses
        for (std::size_t dim = 0; dim < num_features_; dim += static_cast<std::size_t>(THREAD_BLOCK_SIZE)) {
            group.parallel_for_work_item([&](::sycl::h_item<2> idx) {
                const auto local_id_0 = static_cast<unsigned>(idx.get_local_id(0));
                const auto local_id_1 = static_cast<unsigned>(idx.get_local_id(1));

                const std::size_t threadIdx_x = idx.get_local_id(0);

                const auto THREAD_BLOCK_SIZE_uz = static_cast<std::size_t>(THREAD_BLOCK_SIZE);
                const auto PADDING_SIZE_uz = static_cast<std::size_t>(PADDING_SIZE);

                // load data into shared memory
                for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                    const auto global_pp_idx = pp_idx_linear(idx) + static_cast<std::size_t>(internal) * THREAD_BLOCK_SIZE_uz;
                    const auto global_class_idx = class_idx_linear(idx) + static_cast<std::size_t>(internal) * THREAD_BLOCK_SIZE_uz;

                    // store the values in the local memory
                    data_cache_pp[local_id_0][internal * THREAD_BLOCK_SIZE + local_id_1] = predict_points_d_[(dim + threadIdx_x) * (num_predict_points_ + PADDING_SIZE_uz) + global_pp_idx];
                    data_cache_w[local_id_0][internal * THREAD_BLOCK_SIZE + local_id_1] = w_d_[(dim + threadIdx_x) * (num_classes_ + PADDING_SIZE_uz) + global_class_idx];
                }
            });

            // implicit group barrier

            // perform the dot product calculation
            group.parallel_for_work_item([&](::sycl::h_item<2> idx) {
                const auto local_id_0 = static_cast<unsigned>(idx.get_local_id(0));
                const auto local_id_1 = static_cast<unsigned>(idx.get_local_id(1));

                for (unsigned block_dim = 0; block_dim < THREAD_BLOCK_SIZE; ++block_dim) {
                    for (unsigned internal_pd = 0; internal_pd < INTERNAL_BLOCK_SIZE; ++internal_pd) {
                        for (unsigned internal_class = 0; internal_class < INTERNAL_BLOCK_SIZE; ++internal_class) {
                            temp(idx)[internal_pd][internal_class] += data_cache_w[block_dim][local_id_0 * INTERNAL_BLOCK_SIZE + internal_class] * data_cache_pp[block_dim][local_id_1 * INTERNAL_BLOCK_SIZE + internal_pd];
                        }
                    }
                }
            });

            // implicit group barrier
        }

        // update global array with local one
        group.parallel_for_work_item([&](::sycl::h_item<2> idx) {
            const auto PADDING_SIZE_uz = static_cast<std::size_t>(PADDING_SIZE);

            for (unsigned internal_pd = 0; internal_pd < INTERNAL_BLOCK_SIZE; ++internal_pd) {
                for (unsigned internal_class = 0; internal_class < INTERNAL_BLOCK_SIZE; ++internal_class) {
                    const auto global_class_idx = class_idx(idx) + static_cast<std::size_t>(internal_class);
                    const auto global_pp_idx = pp_idx(idx) + static_cast<std::size_t>(internal_pd);

                    prediction_d_[global_pp_idx * (num_classes_ + PADDING_SIZE_uz) + global_class_idx] = temp(idx)[internal_pd][internal_class] - rho_d_[global_class_idx];
                }
            }
        });
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
 * @details Uses SYCL's hierarchical data parallel kernels.
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
     * @param[in] group indices representing the current point in the execution space
     */
    void operator()(::sycl::group<2> group) const {
        // allocate shared memory
        real_type data_cache_pp[THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];
        real_type data_cache_sv[THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];

        // calculate the indices used in the current work-item
        ::sycl::private_memory<std::size_t, 2> pp_idx{ group };
        ::sycl::private_memory<std::size_t, 2> pp_idx_linear{ group };
        ::sycl::private_memory<std::size_t, 2> sv_idx_linear{ group };

        ::sycl::private_memory<real_type[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE], 2> temp{ group };

        // initialize private and local variables
        group.parallel_for_work_item([&](::sycl::h_item<2> idx) {
            const std::size_t threadIdx_y = idx.get_local_id(1);       // current thread in block y-dimension
            const std::size_t blockDim_x = idx.get_local_range(0);     // number of threads in block x-dimension
            const std::size_t blockDim_y = idx.get_local_range(1);     // number of threads in block y-dimension
            const std::size_t blockIdx_x = group[0] + grid_x_offset_;  // current block in grid x-dimension + offsets if the grid size would be too large
            const std::size_t blockIdx_y = group[1] + grid_y_offset_;  // current block in grid y-dimension + offsets if the grid size would be too large

            const auto INTERNAL_BLOCK_SIZE_uz = static_cast<std::size_t>(INTERNAL_BLOCK_SIZE);

            // indices
            pp_idx(idx) = (blockIdx_y * blockDim_y + threadIdx_y) * INTERNAL_BLOCK_SIZE_uz;
            pp_idx_linear(idx) = blockIdx_y * blockDim_y * INTERNAL_BLOCK_SIZE_uz + threadIdx_y;
            sv_idx_linear(idx) = blockIdx_x * blockDim_x * INTERNAL_BLOCK_SIZE_uz + threadIdx_y;

            // initialize private temp matrix to zero
            for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                    temp(idx)[internal_i][internal_j] = real_type{ 0.0 };
                }
            }
        });

        // implicit group barrier

        {
            // iterate over all features using blocking to be able to cache them for faster memory accesses
            for (std::size_t dim = 0; dim < num_features_; dim += static_cast<std::size_t>(THREAD_BLOCK_SIZE)) {
                group.parallel_for_work_item([&](::sycl::h_item<2> idx) {
                    const auto local_id_0 = static_cast<unsigned>(idx.get_local_id(0));
                    const auto local_id_1 = static_cast<unsigned>(idx.get_local_id(1));

                    const std::size_t threadIdx_x = idx.get_local_id(0);

                    const auto THREAD_BLOCK_SIZE_uz = static_cast<std::size_t>(THREAD_BLOCK_SIZE);
                    const auto PADDING_SIZE_uz = static_cast<std::size_t>(PADDING_SIZE);

                    // load data into local memory
                    for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                        const auto global_pp_idx = pp_idx_linear(idx) + static_cast<std::size_t>(internal) * THREAD_BLOCK_SIZE_uz;
                        const auto global_sv_idx = sv_idx_linear(idx) + static_cast<std::size_t>(internal) * THREAD_BLOCK_SIZE_uz;

                        // store the values in the shared memory
                        data_cache_pp[local_id_0][internal * THREAD_BLOCK_SIZE + local_id_1] = predict_points_d_[(dim + threadIdx_x) * (num_predict_points_ + PADDING_SIZE_uz) + global_pp_idx];
                        data_cache_sv[local_id_0][internal * THREAD_BLOCK_SIZE + local_id_1] = sv_d_[(dim + threadIdx_x) * (num_sv_ + PADDING_SIZE_uz) + global_sv_idx];
                    }
                });

                // implicit group barrier

                // perform the feature reduction calculation
                group.parallel_for_work_item([&](::sycl::h_item<2> idx) {
                    const auto local_id_0 = static_cast<unsigned>(idx.get_local_id(0));
                    const auto local_id_1 = static_cast<unsigned>(idx.get_local_id(1));

                    for (unsigned block_dim = 0; block_dim < THREAD_BLOCK_SIZE; ++block_dim) {
                        for (unsigned internal_pd = 0; internal_pd < INTERNAL_BLOCK_SIZE; ++internal_pd) {
                            for (unsigned internal_sv = 0; internal_sv < INTERNAL_BLOCK_SIZE; ++internal_sv) {
                                temp(idx)[internal_pd][internal_sv] += detail::feature_reduce<kernel_function>(data_cache_sv[block_dim][local_id_0 * INTERNAL_BLOCK_SIZE + internal_sv],
                                                                                                               data_cache_pp[block_dim][local_id_1 * INTERNAL_BLOCK_SIZE + internal_pd]);
                            }
                        }
                    }
                });

                // implicit group barrier
            }
        }

        // update temp using the respective kernel function
        group.parallel_for_work_item([&](::sycl::h_item<2> idx) {
            for (unsigned internal_pd = 0; internal_pd < INTERNAL_BLOCK_SIZE; ++internal_pd) {
                for (unsigned internal_sv = 0; internal_sv < INTERNAL_BLOCK_SIZE; ++internal_sv) {
                    temp(idx)[internal_pd][internal_sv] = detail::apply_kernel_function<kernel_function>(temp(idx)[internal_pd][internal_sv], kernel_function_parameter_);
                }
            }
        });

        // implicit group barrier

        {
            // rename cached arrays -> can't rename the arrays due to AdaptiveCpp runtime exception
            // auto &alpha_cache = data_cache_pp;
            // auto &out_cache = data_cache_sv;

            // iterate over all features using blocking to be able to cache them for faster memory accesses
            for (std::size_t dim = 0; dim < num_classes_; dim += static_cast<std::size_t>(THREAD_BLOCK_SIZE)) {
                // load data into local memory
                group.parallel_for_work_item([&](::sycl::h_item<2> idx) {
                    const auto local_id_0 = static_cast<unsigned>(idx.get_local_id(0));
                    const auto local_id_1 = static_cast<unsigned>(idx.get_local_id(1));

                    const std::size_t blockIdx_x = group[0] + grid_x_offset_;  // current block in grid x-dimension + offsets if the grid size would be too large
                    const std::size_t threadIdx_x = idx.get_local_id(0);

                    const auto THREAD_BLOCK_SIZE_uz = static_cast<std::size_t>(THREAD_BLOCK_SIZE);
                    const auto PADDING_SIZE_uz = static_cast<std::size_t>(PADDING_SIZE);

                    for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                        const std::size_t global_sv_idx = sv_idx_linear(idx) + static_cast<std::size_t>(internal) * THREAD_BLOCK_SIZE_uz;

                        data_cache_pp[local_id_0][internal * THREAD_BLOCK_SIZE + local_id_1] = alpha_d_[(dim + threadIdx_x) * (num_sv_ + PADDING_SIZE_uz) + global_sv_idx];

                        // the bias (rho) must only be applied once for all support vectors
                        if (blockIdx_x == std::size_t{ 0 }) {
                            data_cache_sv[local_id_0][internal * THREAD_BLOCK_SIZE + local_id_1] = -rho_d_[dim + threadIdx_x];
                        } else {
                            data_cache_sv[local_id_0][internal * THREAD_BLOCK_SIZE + local_id_1] = real_type{ 0.0 };
                        }
                    }
                });

                // implicit group barrier

                // calculate intermediate results and store them in local memory
                for (unsigned class_idx = 0; class_idx < THREAD_BLOCK_SIZE; ++class_idx) {
                    group.parallel_for_work_item([&](::sycl::h_item<2> idx) {
                        const auto local_id_0 = static_cast<unsigned>(idx.get_local_id(0));
                        const auto local_id_1 = static_cast<unsigned>(idx.get_local_id(1));

                        for (unsigned internal_pd = 0; internal_pd < INTERNAL_BLOCK_SIZE; ++internal_pd) {
                            for (unsigned internal_sv = 0; internal_sv < INTERNAL_BLOCK_SIZE; ++internal_sv) {
                                data_cache_sv[(class_idx + local_id_0) % THREAD_BLOCK_SIZE][internal_pd * THREAD_BLOCK_SIZE + local_id_1] +=
                                    temp(idx)[internal_pd][internal_sv] * data_cache_pp[(class_idx + local_id_0) % THREAD_BLOCK_SIZE][local_id_0 * INTERNAL_BLOCK_SIZE + internal_sv];
                            }
                        }
                    });

                    // implicit group barrier
                }

                // add intermediate cached results to prediction_d
                group.parallel_for_work_item([&](::sycl::h_item<2> idx) {
                    const auto local_id_0 = static_cast<unsigned>(idx.get_local_id(0));
                    const auto local_id_1 = static_cast<unsigned>(idx.get_local_id(1));

                    const std::size_t threadIdx_x = idx.get_local_id(0);

                    const auto PADDING_SIZE_uz = static_cast<std::size_t>(PADDING_SIZE);

                    for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                        const auto global_pp_idx = pp_idx(idx) + static_cast<std::size_t>(internal);

                        detail::atomic_op<real_type>{ prediction_d_[global_pp_idx * (num_classes_ + PADDING_SIZE_uz) + dim + threadIdx_x] } += data_cache_sv[local_id_0][internal * THREAD_BLOCK_SIZE + local_id_1];
                    }
                });

                // implicit group barrier
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

}  // namespace plssvm::sycl::detail::hierarchical

#endif  // PLSSVM_BACKENDS_SYCL_KERNEL_PREDICT_HIERARCHICAL_PREDICT_KERNEL_HPP_
