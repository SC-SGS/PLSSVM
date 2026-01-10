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
#include "plssvm/constants.hpp"                              // plssvm::real_type, plssvm::THREAD_BLOCK_SIZE
#include "plssvm/kernel_function_types.hpp"                  // plssvm::kernel_function_type

#include "sycl/sycl.hpp"  // sycl::group, sycl::h_item

#include <cstddef>  // std::size_t
#include <tuple>    // std::tuple, std::make_tuple

namespace plssvm::sycl::detail::hierarchical {

/**
 * @brief Calculate the `w` vector used to speedup the prediction using the linear kernel function.
 * @details Uses SYCL's hierarchical data parallel kernels.
 */
class device_kernel_w_linear {
  public:
    /// The used SYCL data parallel kernel.
    constexpr static sycl::data_parallel_kernel data_parallel_kernel_type = sycl::data_parallel_kernel::hierarchical;

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
     * @param[in] group indices representing the current point in the execution space
     */
    void operator()(::sycl::group<2> group) const {
        // create two local memory arrays used for caching
        real_type feature_cache[THREAD_BLOCK_SIZE][THREAD_BLOCK_SIZE]{};
        real_type alpha_cache[THREAD_BLOCK_SIZE][THREAD_BLOCK_SIZE]{};

        // create a private memory array used for internal caching
        ::sycl::private_memory<real_type, 2> temp{ group };

        // initialize private temp to zero
        group.parallel_for_work_item([&](::sycl::h_item<2> idx) {
            temp(idx) = real_type{ 0.0 };
        });

        // implicit group barrier

        // iterate over all support vectors using blocking to be able to cache them for faster memory accesses
        for (std::size_t sv_block = 0; sv_block < device_num_sv_; sv_block += static_cast<std::size_t>(THREAD_BLOCK_SIZE)) {
            // load data into local memory
            group.parallel_for_work_item([&](::sycl::h_item<2> idx) {
                // cast values to 32-bit unsigned int values to prevent implicit conversions
                const auto local_id_0 = static_cast<unsigned>(idx.get_local_id(0));
                const auto local_id_1 = static_cast<unsigned>(idx.get_local_id(1));

                const auto threadIdx_x = static_cast<std::size_t>(idx.get_local_id(0));       // current work-item in work-group x-dimension
                const auto threadIdx_y = static_cast<std::size_t>(idx.get_local_id(1));       // current work-item in work-group y-dimension
                const auto blockDim_x = static_cast<std::size_t>(idx.get_local_range(0));     // number of work-items in work-group x-dimension
                const auto blockDim_y = static_cast<std::size_t>(idx.get_local_range(1));     // number of work-items in work-group y-dimension
                const auto blockIdx_x = static_cast<std::size_t>(group[0]) + grid_x_offset_;  // current work-group in global range x-dimension + offsets if the global range is too large
                const auto blockIdx_y = static_cast<std::size_t>(group[1]) + grid_y_offset_;  // current work-group in global range y-dimension + offsets if the global range is too large

                // calculate the indices used in the current work-item, pays attention to coalesced memory accesses
                const auto global_feature_idx_linear = blockIdx_y * blockDim_y + threadIdx_y;  // num_features
                const auto global_class_idx_linear = blockIdx_x * blockDim_x + threadIdx_y;    // num_classes

                // zero-out local memory
                feature_cache[local_id_0][local_id_1] = real_type{ 0.0 };
                alpha_cache[local_id_0][local_id_1] = real_type{ 0.0 };

                // load data into local memory
                if (sv_block + threadIdx_x < device_num_sv_) {
                    if (global_feature_idx_linear < num_features_) {
                        feature_cache[local_id_0][local_id_1] = support_vectors_[global_feature_idx_linear * device_num_sv_ + sv_block + threadIdx_x];  // SoA
                    }
                    if (global_class_idx_linear < num_classes_) {
                        alpha_cache[local_id_0][local_id_1] = alpha_[global_class_idx_linear * num_sv_ + sv_block + device_sv_offset_ + threadIdx_x];  // AoS
                    }
                }
            });

            // implicit group barrier

            // perform the dot product calculation
            group.parallel_for_work_item([&](::sycl::h_item<2> idx) {
                // cast values to 32-bit unsigned int values to prevent implicit conversions
                const auto local_id_0 = static_cast<unsigned>(idx.get_local_id(0));
                const auto local_id_1 = static_cast<unsigned>(idx.get_local_id(1));

                // perform the dot product calculation
                for (unsigned sv = 0; sv < THREAD_BLOCK_SIZE; ++sv) {
                    temp(idx) += alpha_cache[sv][local_id_0] * feature_cache[sv][local_id_1];
                }
            });

            // implicit group barrier
        }

        // update the global w-vector with the locally cached values
        group.parallel_for_work_item([&](::sycl::h_item<2> idx) {
            const auto threadIdx_x = static_cast<std::size_t>(idx.get_local_id(0));       // current work-item in work-group x-dimension
            const auto threadIdx_y = static_cast<std::size_t>(idx.get_local_id(1));       // current work-item in work-group y-dimension
            const auto blockDim_x = static_cast<std::size_t>(idx.get_local_range(0));     // number of work-items in work-group x-dimension
            const auto blockDim_y = static_cast<std::size_t>(idx.get_local_range(1));     // number of work-items in work-group y-dimension
            const auto blockIdx_x = static_cast<std::size_t>(group[0]) + grid_x_offset_;  // current work-group in global range x-dimension + offsets if the global range is too large
            const auto blockIdx_y = static_cast<std::size_t>(group[1]) + grid_y_offset_;  // current work-group in global range y-dimension + offsets if the global range is too large

            // calculate the indices used in the current work-item
            const auto global_feature_idx = blockIdx_y * blockDim_y + threadIdx_y;  // num_features
            const auto global_class_idx = blockIdx_x * blockDim_x + threadIdx_x;    // num_classes

            // be sure to not perform out-of-bounds accesses
            if (global_feature_idx < num_features_ && global_class_idx < num_classes_) {
                w_[global_feature_idx * num_classes_ + global_class_idx] = temp(idx);  // SoA
            }
        });
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
 * @details Uses SYCL's hierarchical data parallel kernels.
 */
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
        real_type pp_cache[THREAD_BLOCK_SIZE][THREAD_BLOCK_SIZE]{};
        real_type w_cache[THREAD_BLOCK_SIZE][THREAD_BLOCK_SIZE]{};

        // create a private memory array used for internal caching
        ::sycl::private_memory<real_type, 2> temp{ group };

        // initialize private variable
        group.parallel_for_work_item([&](::sycl::h_item<2> idx) {
            // initialize private temp to zero
            temp(idx) = real_type{ 0.0 };
        });

        // implicit group barrier

        // iterate over all features using blocking to be able to cache them for faster memory accesses
        for (std::size_t feature_block = 0; feature_block < num_features_; feature_block += static_cast<std::size_t>(THREAD_BLOCK_SIZE)) {
            group.parallel_for_work_item([&](::sycl::h_item<2> idx) {
                // cast values to 32-bit unsigned int values to prevent implicit conversions
                const auto local_id_0 = static_cast<unsigned>(idx.get_local_id(0));
                const auto local_id_1 = static_cast<unsigned>(idx.get_local_id(1));

                const auto threadIdx_x = static_cast<std::size_t>(idx.get_local_id(0));       // current work-item in work-group x-dimension
                const auto threadIdx_y = static_cast<std::size_t>(idx.get_local_id(1));       // current work-item in work-group y-dimension
                const auto blockDim_x = static_cast<std::size_t>(idx.get_local_range(0));     // number of work-items in work-group x-dimension
                const auto blockDim_y = static_cast<std::size_t>(idx.get_local_range(1));     // number of work-items in work-group y-dimension
                const auto blockIdx_x = static_cast<std::size_t>(group[0]) + grid_x_offset_;  // current work-group in global range x-dimension + offsets if the global range is too large
                const auto blockIdx_y = static_cast<std::size_t>(group[1]) + grid_y_offset_;  // current work-group in global range y-dimension + offsets if the global range is too large

                // calculate the indices used in the current thread, pays attention to coalesced memory accesses
                const auto global_pp_idx_linear = blockIdx_y * blockDim_y + threadIdx_y;     // num_predict_points
                const auto global_class_idx_linear = blockIdx_x * blockDim_x + threadIdx_y;  // num_classes

                // zero-out local memory
                pp_cache[local_id_0][local_id_1] = real_type{ 0.0 };
                w_cache[local_id_0][local_id_1] = real_type{ 0.0 };

                // load data into local memory
                if (feature_block + threadIdx_x < num_features_) {
                    if (global_pp_idx_linear < num_predict_points_) {
                        pp_cache[local_id_0][local_id_1] = predict_points_[(feature_block + threadIdx_x) * num_predict_points_ + global_pp_idx_linear];  // SoA
                    }
                    if (global_class_idx_linear < num_classes_) {
                        w_cache[local_id_0][local_id_1] = w_[(feature_block + threadIdx_x) * num_classes_ + global_class_idx_linear];  // SoA
                    }
                }
            });

            // implicit group barrier

            // perform the dot product calculation
            group.parallel_for_work_item([&](::sycl::h_item<2> idx) {
                // cast values to 32-bit unsigned int values to prevent implicit conversions
                const auto local_id_0 = static_cast<unsigned>(idx.get_local_id(0));
                const auto local_id_1 = static_cast<unsigned>(idx.get_local_id(1));

                // perform the dot product calculation
                for (unsigned feature = 0; feature < THREAD_BLOCK_SIZE; ++feature) {
                    temp(idx) += w_cache[feature][local_id_0] * pp_cache[feature][local_id_1];
                }
            });

            // implicit group barrier
        }

        // update the global array with the local one
        group.parallel_for_work_item([&](::sycl::h_item<2> idx) {
            const auto threadIdx_x = static_cast<std::size_t>(idx.get_local_id(0));       // current work-item in work-group x-dimension
            const auto threadIdx_y = static_cast<std::size_t>(idx.get_local_id(1));       // current work-item in work-group y-dimension
            const auto blockDim_x = static_cast<std::size_t>(idx.get_local_range(0));     // number of work-items in work-group x-dimension
            const auto blockDim_y = static_cast<std::size_t>(idx.get_local_range(1));     // number of work-items in work-group y-dimension
            const auto blockIdx_x = static_cast<std::size_t>(group[0]) + grid_x_offset_;  // current work-group in global range x-dimension + offsets if the global range is too large
            const auto blockIdx_y = static_cast<std::size_t>(group[1]) + grid_y_offset_;  // current work-group in global range y-dimension + offsets if the global range is too large

            // calculate the indices used in the current work-item
            const auto global_pp_idx = blockIdx_y * blockDim_y + threadIdx_y;     // num_predict_points
            const auto global_class_idx = blockIdx_x * blockDim_x + threadIdx_x;  // num_classes

            // be sure to not perform out-of-bounds accesses
            if (global_pp_idx < num_predict_points_ && global_class_idx < num_classes_) {
                prediction_[global_pp_idx * num_classes_ + global_class_idx] = temp(idx) - rho_[global_class_idx];  // AoS
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
 * @tparam kernel_function the type of the used kernel function
 * @tparam Args the types of the parameters necessary for the specific kernel function; stored in a `std::tuple`
 */
template <kernel_function_type kernel_function, typename... Args>
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
        real_type cache_one[THREAD_BLOCK_SIZE][THREAD_BLOCK_SIZE]{};
        real_type cache_two[THREAD_BLOCK_SIZE][THREAD_BLOCK_SIZE]{};

        // create a private memory array used for internal caching
        ::sycl::private_memory<real_type, 2> temp{ group };

        // initialize private variable
        group.parallel_for_work_item([&](::sycl::h_item<2> idx) {
            // initialize private temp to zero
            temp(idx) = real_type{ 0.0 };
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

                    const auto threadIdx_x = static_cast<std::size_t>(idx.get_local_id(0));       // current work-item in work-group x-dimension
                    const auto threadIdx_y = static_cast<std::size_t>(idx.get_local_id(1));       // current work-item in work-group y-dimension
                    const auto blockDim_x = static_cast<std::size_t>(idx.get_local_range(0));     // number of work-items in work-group x-dimension
                    const auto blockDim_y = static_cast<std::size_t>(idx.get_local_range(1));     // number of work-items in work-group y-dimension
                    const auto blockIdx_x = static_cast<std::size_t>(group[0]) + grid_x_offset_;  // current work-group in global range x-dimension + offsets if the global range is too large
                    const auto blockIdx_y = static_cast<std::size_t>(group[1]) + grid_y_offset_;  // current work-group in global range y-dimension + offsets if the global range is too large

                    // calculate the indices used in the current thread, pays attention to coalesced memory accesses
                    const auto global_pp_idx_linear = blockIdx_y * blockDim_y + threadIdx_y;  // num_predict_points
                    const auto global_sv_idx_linear = blockIdx_x * blockDim_x + threadIdx_y;  // num_sv

                    // zero-out local memory
                    cache_one[local_id_0][local_id_1] = real_type{ 0.0 };
                    cache_two[local_id_0][local_id_1] = real_type{ 0.0 };

                    // load data into local memory
                    if (feature_block + threadIdx_x < num_features_) {
                        if (global_pp_idx_linear < num_predict_points_) {
                            cache_one[local_id_0][local_id_1] = predict_points_[(feature_block + threadIdx_x) * num_predict_points_ + global_pp_idx_linear];
                        }
                        if (global_sv_idx_linear < num_sv_) {
                            cache_two[local_id_0][local_id_1] = support_vectors_[(feature_block + threadIdx_x) * num_sv_ + global_sv_idx_linear];
                        }
                    }
                });

                // implicit group barrier

                // perform the feature reduction calculation
                group.parallel_for_work_item([&](::sycl::h_item<2> idx) {
                    // cast values to 32-bit unsigned int values to prevent implicit conversions
                    const auto local_id_0 = static_cast<unsigned>(idx.get_local_id(0));
                    const auto local_id_1 = static_cast<unsigned>(idx.get_local_id(1));

                    // perform the feature reduction calculation
                    for (unsigned feature = 0; feature < THREAD_BLOCK_SIZE; ++feature) {
                        temp(idx) += detail::feature_reduce<kernel_function>(cache_two[feature][local_id_0],
                                                                             cache_one[feature][local_id_1]);
                    }
                });

                // implicit group barrier
            }
        }

        // update temp using the respective kernel function
        group.parallel_for_work_item([&](::sycl::h_item<2> idx) {
            temp(idx) = detail::apply_kernel_function<kernel_function>(temp(idx), kernel_function_parameter_);
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

                    const auto threadIdx_x = static_cast<std::size_t>(idx.get_local_id(0));       // current work-item in work-group x-dimension
                    const auto threadIdx_y = static_cast<std::size_t>(idx.get_local_id(1));       // current work-item in work-group y-dimension
                    const auto blockDim_x = static_cast<std::size_t>(idx.get_local_range(0));     // number of work-items in work-group x-dimension
                    const auto blockIdx_x = static_cast<std::size_t>(group[0]) + grid_x_offset_;  // current work-group in global range x-dimension + offsets if the global range is too large

                    // calculate the indices used in the current thread, pays attention to coalesced memory accesses
                    const auto global_sv_idx_linear = blockIdx_x * blockDim_x + threadIdx_y;  // num_support_vectors

                    // zero-out local memory
                    cache_one[local_id_0][local_id_1] = real_type{ 0.0 };
                    cache_two[local_id_0][local_id_1] = real_type{ 0.0 };

                    // load data into local memory
                    if (class_block + threadIdx_x < num_classes_) {
                        if (global_sv_idx_linear < num_sv_) {
                            cache_one[local_id_0][local_id_1] = alpha_[(class_block + threadIdx_x) * num_sv_ + global_sv_idx_linear];  // AoS
                        }
                        // the bias (rho) must only be applied once for all support vectors
                        if (blockIdx_x == std::size_t{ 0 }) {
                            cache_two[local_id_0][local_id_1] = -rho_[class_block + threadIdx_x];
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

                        cache_two[(class_idx + local_id_0) % THREAD_BLOCK_SIZE][local_id_1] += temp(idx) * cache_one[(class_idx + local_id_0) % THREAD_BLOCK_SIZE][local_id_0];
                    });

                    // implicit group barrier
                }

                // atomically add the intermediate cached results to the prediction
                group.parallel_for_work_item([&](::sycl::h_item<2> idx) {
                    // cast values to 32-bit unsigned int values to prevent implicit conversions
                    const auto local_id_0 = static_cast<unsigned>(idx.get_local_id(0));
                    const auto local_id_1 = static_cast<unsigned>(idx.get_local_id(1));

                    const auto threadIdx_x = static_cast<std::size_t>(idx.get_local_id(0));       // current work-item in work-group x-dimension
                    const auto threadIdx_y = static_cast<std::size_t>(idx.get_local_id(1));       // current work-item in work-group y-dimension
                    const auto blockDim_y = static_cast<std::size_t>(idx.get_local_range(1));     // number of work-items in work-group y-dimension
                    const auto blockIdx_y = static_cast<std::size_t>(group[1]) + grid_y_offset_;  // current work-group in global range y-dimension + offsets if the global range is too large

                    // calculate the indices used in the current thread
                    const auto global_pp_idx = blockIdx_y * blockDim_y + threadIdx_y;  // num_predict_points

                    if (class_block + threadIdx_x < num_classes_ && global_pp_idx < num_predict_points_) {
                        detail::atomic_op<real_type>{ prediction_[global_pp_idx * num_classes_ + class_block + threadIdx_x] } += cache_two[local_id_0][local_id_1];
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
