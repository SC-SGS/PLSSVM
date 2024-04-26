/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines the functions used for prediction for the C-SVM using the SYCL backend.
 */

#ifndef PLSSVM_BACKENDS_SYCL_PREDICT_KERNEL_HPP_
#define PLSSVM_BACKENDS_SYCL_PREDICT_KERNEL_HPP_
#pragma once

#include "plssvm/backends/SYCL/detail/atomics.hpp"           // plssvm::sycl::detail::atomic_op
#include "plssvm/backends/SYCL/kernel/kernel_functions.hpp"  // plssvm::sycl::detail::{feature_reduce, apply_kernel_function}
#include "plssvm/constants.hpp"                              // plssvm::{real_type, THREAD_BLOCK_SIZE, INTERNAL_BLOCK_SIZE, FEATURE_BLOCK_SIZE, PADDING_SIZE}
#include "plssvm/kernel_function_types.hpp"                  // plssvm::kernel_function_type

#include "sycl/sycl.hpp"  // sycl::item

#include <cstddef>  // std::size_t
#include <tuple>    // std::tuple, std::make_tuple

namespace plssvm::sycl::detail {

/**
 * @brief Calculate the `q` vector used to speedup the prediction using the linear kernel function.
 */
class device_kernel_w_linear {
  public:
    /**
     * @brief Initialize the SYCL kernel function object.
     * @param[in] cgh the SYCL handler used to allocate the local memory
     * @param[in,out] w_d the vector to speedup the linear prediction
     * @param[in] alpha_d the previously learned weights
     * @param[in] sv_d the support vectors
     * @param[in] num_classes the number of classes
     * @param[in] num_sv the number of support vectors
     * @param[in] device_specific_num_sv the number of support vectors the current device is responsible for
     * @param[in] sv_offset the first support vector (row in @p alpha_d) the current device is responsible for
     */
    device_kernel_w_linear(::sycl::handler &cgh, real_type *w_d, const real_type *alpha_d, const real_type *sv_d, const std::size_t num_classes, const std::size_t num_sv, const std::size_t device_specific_num_sv, const std::size_t sv_offset) :
        data_cache_feature_{ ::sycl::range<2>{ static_cast<std::size_t>(FEATURE_BLOCK_SIZE), static_cast<std::size_t>(INTERNAL_BLOCK_SIZE) * static_cast<std::size_t>(THREAD_BLOCK_SIZE) }, cgh },
        data_cache_alpha_{ ::sycl::range<2>{ static_cast<std::size_t>(FEATURE_BLOCK_SIZE), static_cast<std::size_t>(INTERNAL_BLOCK_SIZE) * static_cast<std::size_t>(THREAD_BLOCK_SIZE) }, cgh },
        w_d_{ w_d },
        alpha_d_{ alpha_d },
        sv_d_{ sv_d },
        num_classes_{ num_classes },
        num_sv_{ num_sv },
        device_specific_num_sv_{ device_specific_num_sv },
        sv_offset_{ sv_offset } { }

    /**
     * @brief Function call operator overload performing the actual calculation.
     * @param[in] nd_idx indices representing the current point in the execution space
     */
    void operator()(::sycl::nd_item<2> nd_idx) const {
        // cast values to 32-bit unsigned int values to prevent implicit conversions
        const auto local_id_0 = static_cast<unsigned>(nd_idx.get_local_id(0));
        const auto local_id_1 = static_cast<unsigned>(nd_idx.get_local_id(1));

        // cast all values to 64-bit std::size_t to prevent potential 32-bit overflows
        const auto INTERNAL_BLOCK_SIZE_uz = static_cast<std::size_t>(INTERNAL_BLOCK_SIZE);
        const auto THREAD_BLOCK_SIZE_uz = static_cast<std::size_t>(THREAD_BLOCK_SIZE);
        const auto PADDING_SIZE_uz = static_cast<std::size_t>(PADDING_SIZE);

        // calculate the indices used in the current work-item
        const auto feature_idx = nd_idx.get_global_id(1) * INTERNAL_BLOCK_SIZE_uz;
        const auto feature_idx_linear = nd_idx.get_group(1) * nd_idx.get_local_range(1) * INTERNAL_BLOCK_SIZE_uz + nd_idx.get_local_id(1);
        const auto class_idx = nd_idx.get_global_id(0) * INTERNAL_BLOCK_SIZE_uz;
        const auto class_idx_linear = nd_idx.get_group(0) * nd_idx.get_local_range(0) * INTERNAL_BLOCK_SIZE_uz + nd_idx.get_local_id(1);

        // create a work-item private array used for internal caching
        real_type temp[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE]{};

        // iterate over all support vectors using blocking to be able to cache them for faster memory accesses
        for (std::size_t sv = 0; sv < device_specific_num_sv_; sv += THREAD_BLOCK_SIZE) {
            // load data into local memory
            for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                const auto global_class_idx = class_idx_linear + static_cast<std::size_t>(internal) * THREAD_BLOCK_SIZE_uz;
                const auto global_feature_idx = feature_idx_linear + static_cast<std::size_t>(internal) * THREAD_BLOCK_SIZE_uz;

                data_cache_feature_[local_id_0][internal * THREAD_BLOCK_SIZE + local_id_1] = sv_d_[global_feature_idx * (device_specific_num_sv_ + PADDING_SIZE_uz) + sv + sv_offset_ + nd_idx.get_local_id(0)];  // SoA
                data_cache_alpha_[local_id_0][internal * THREAD_BLOCK_SIZE + local_id_1] = alpha_d_[global_class_idx * (num_sv_ + PADDING_SIZE_uz) + sv + nd_idx.get_local_id(0)];                                // AoS
            }
            nd_idx.barrier();  // wait until all work-items loaded their part of the data

            // perform the dot product calculation
            for (unsigned block_dim = 0; block_dim < THREAD_BLOCK_SIZE; ++block_dim) {
                for (unsigned internal_feature = 0; internal_feature < INTERNAL_BLOCK_SIZE; ++internal_feature) {
                    for (unsigned internal_class = 0; internal_class < INTERNAL_BLOCK_SIZE; ++internal_class) {
                        temp[internal_feature][internal_class] += data_cache_alpha_[block_dim][local_id_0 * INTERNAL_BLOCK_SIZE + internal_class] * data_cache_feature_[block_dim][local_id_1 * INTERNAL_BLOCK_SIZE + internal_feature];
                    }
                }
            }
            nd_idx.barrier();  // wait until all work-items performed their part of the calculations
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
    /// Local memory used for internal memory access optimizations.
    ::sycl::local_accessor<real_type, 2> data_cache_feature_;
    /// Local memory used for internal memory access optimizations.
    ::sycl::local_accessor<real_type, 2> data_cache_alpha_;

    /// @cond Doxygen_suppress
    real_type *w_d_;
    const real_type *alpha_d_;
    const real_type *sv_d_;
    const std::size_t num_classes_;
    const std::size_t num_sv_;
    const std::size_t device_specific_num_sv_;
    const std::size_t sv_offset_;
    /// @endcond
};

/**
 * @brief Predict the @p predict_points_d using the linear kernel speeding up the calculation using the @p w_d vector.
 */
class device_kernel_predict_linear {
  public:
    /**
     * @brief Initialize the SYCL kernel function object.
     * @param[in] cgh the SYCL handler used to allocate the local memory
     * @param[out] prediction_d the predicted values
     * @param[in] w_d the vector to speedup the calculations
     * @param[in] rho_d the previously learned bias
     * @param[in] predict_points_d the data points to predict
     * @param[in] num_classes the number of classes
     * @param[in] num_predict_points the number of data points to predict
     * @param[in] num_features the number of features per data point
     */
    device_kernel_predict_linear(::sycl::handler &cgh, real_type *prediction_d, const real_type *w_d, const real_type *rho_d, const real_type *predict_points_d, const std::size_t num_classes, const std::size_t num_predict_points, const std::size_t num_features) :
        data_cache_pp_{ ::sycl::range<2>{ static_cast<std::size_t>(FEATURE_BLOCK_SIZE), static_cast<std::size_t>(INTERNAL_BLOCK_SIZE) * static_cast<std::size_t>(THREAD_BLOCK_SIZE) }, cgh },
        data_cache_w_{ ::sycl::range<2>{ static_cast<std::size_t>(FEATURE_BLOCK_SIZE), static_cast<std::size_t>(INTERNAL_BLOCK_SIZE) * static_cast<std::size_t>(THREAD_BLOCK_SIZE) }, cgh },
        prediction_d_{ prediction_d },
        w_d_{ w_d },
        rho_d_{ rho_d },
        predict_points_d_{ predict_points_d },
        num_classes_{ num_classes },
        num_predict_points_{ num_predict_points },
        num_features_{ num_features } { }

    /**
     * @brief Function call operator overload performing the actual calculation.
     * @param[in] nd_idx indices representing the current point in the execution space
     */
    void operator()(::sycl::nd_item<2> nd_idx) const {
        // cast values to 32-bit unsigned int values to prevent implicit conversions
        const auto local_id_0 = static_cast<unsigned>(nd_idx.get_local_id(0));
        const auto local_id_1 = static_cast<unsigned>(nd_idx.get_local_id(1));

        // cast all values to 64-bit std::size_t to prevent potential 32-bit overflows
        const auto INTERNAL_BLOCK_SIZE_uz = static_cast<std::size_t>(INTERNAL_BLOCK_SIZE);
        const auto THREAD_BLOCK_SIZE_uz = static_cast<std::size_t>(THREAD_BLOCK_SIZE);
        const auto FEATURE_BLOCK_SIZE_uz = static_cast<std::size_t>(FEATURE_BLOCK_SIZE);
        const auto PADDING_SIZE_uz = static_cast<std::size_t>(PADDING_SIZE);

        // calculate the indices used in the current work-item
        const auto pp_idx = nd_idx.get_global_id(1) * INTERNAL_BLOCK_SIZE_uz;
        const auto pp_idx_linear = nd_idx.get_group(1) * nd_idx.get_local_range(1) * INTERNAL_BLOCK_SIZE_uz + nd_idx.get_local_id(1);
        const auto class_idx = nd_idx.get_global_id(0) * INTERNAL_BLOCK_SIZE_uz;
        const auto class_idx_linear = nd_idx.get_group(0) * nd_idx.get_local_range(0) * INTERNAL_BLOCK_SIZE_uz + nd_idx.get_local_id(1);

        // create a work-item private array used for internal caching
        real_type temp[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE]{};

        // iterate over all support vectors using blocking to be able to cache them for faster memory accesses
        for (std::size_t dim = 0; dim < num_features_; dim += FEATURE_BLOCK_SIZE_uz) {
            // load data into shared memory
            for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                const auto global_pp_idx = pp_idx_linear + static_cast<std::size_t>(internal) * THREAD_BLOCK_SIZE_uz;
                const auto global_class_idx = class_idx_linear + static_cast<std::size_t>(internal) * THREAD_BLOCK_SIZE_uz;

                // FEATURE_BLOCK_SIZE = 2 * THREAD_BLOCK_SIZE -> store twice as many values in the local memory
                data_cache_pp_[local_id_0][internal * THREAD_BLOCK_SIZE + local_id_1] = predict_points_d_[(dim + nd_idx.get_local_id(0)) * (num_predict_points_ + PADDING_SIZE_uz) + global_pp_idx];
                data_cache_pp_[local_id_0 + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + local_id_1] = predict_points_d_[(dim + nd_idx.get_local_id(0) + THREAD_BLOCK_SIZE_uz) * (num_predict_points_ + PADDING_SIZE_uz) + global_pp_idx];
                data_cache_w_[local_id_0][internal * THREAD_BLOCK_SIZE + local_id_1] = w_d_[(dim + nd_idx.get_local_id(0)) * (num_classes_ + PADDING_SIZE_uz) + global_class_idx];
                data_cache_w_[local_id_0 + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + local_id_1] = w_d_[(dim + nd_idx.get_local_id(0) + THREAD_BLOCK_SIZE_uz) * (num_classes_ + PADDING_SIZE_uz) + global_class_idx];
            }
            nd_idx.barrier();  // wait until all work-items loaded their part of the data

            // perform the dot product calculation
            for (unsigned block_dim = 0; block_dim < FEATURE_BLOCK_SIZE; ++block_dim) {
                for (unsigned internal_pd = 0; internal_pd < INTERNAL_BLOCK_SIZE; ++internal_pd) {
                    for (unsigned internal_class = 0; internal_class < INTERNAL_BLOCK_SIZE; ++internal_class) {
                        temp[internal_pd][internal_class] += data_cache_w_[block_dim][local_id_0 * INTERNAL_BLOCK_SIZE + internal_class] * data_cache_pp_[block_dim][local_id_1 * INTERNAL_BLOCK_SIZE + internal_pd];
                    }
                }
            }
            nd_idx.barrier();  // wait until all work-items performed their part of the calculations
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
    /// Local memory used for internal memory access optimizations.
    ::sycl::local_accessor<real_type, 2> data_cache_pp_;
    /// Local memory used for internal memory access optimizations.
    ::sycl::local_accessor<real_type, 2> data_cache_w_;

    /// @cond Doxygen_suppress
    real_type *prediction_d_;
    const real_type *w_d_;
    const real_type *rho_d_;
    const real_type *predict_points_d_;
    const std::size_t num_classes_;
    const std::size_t num_predict_points_;
    const std::size_t num_features_;
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
     * @param[in] cgh the SYCL handler used to allocate the local memory
     * @param[in] prediction_d the predicted values
     * @param[in] alpha_d the previously learned weights
     * @param[in] rho_d the previously learned biases
     * @param[in] sv_d the support vectors
     * @param[in] predict_points_d the data points to predict
     * @param[in] num_classes the number of classes
     * @param[in] num_sv the number of support vectors
     * @param[in] num_predict_points the number of data points to predict
     * @param[in] num_features the number of features per data point
     * @param[in] kernel_function_parameter the parameters necessary to apply the @p kernel_function
     */
    device_kernel_predict(::sycl::handler &cgh, real_type *prediction_d, const real_type *alpha_d, const real_type *rho_d, const real_type *sv_d, const real_type *predict_points_d, const std::size_t num_classes, const std::size_t num_sv, const std::size_t num_predict_points, const std::size_t num_features, Args... kernel_function_parameter) :
        data_cache_pp_{ ::sycl::range<2>{ static_cast<std::size_t>(FEATURE_BLOCK_SIZE), static_cast<std::size_t>(INTERNAL_BLOCK_SIZE) * static_cast<std::size_t>(THREAD_BLOCK_SIZE) }, cgh },
        data_cache_sv_{ ::sycl::range<2>{ static_cast<std::size_t>(FEATURE_BLOCK_SIZE), static_cast<std::size_t>(INTERNAL_BLOCK_SIZE) * static_cast<std::size_t>(THREAD_BLOCK_SIZE) }, cgh },
        prediction_d_{ prediction_d },
        alpha_d_{ alpha_d },
        rho_d_{ rho_d },
        sv_d_{ sv_d },
        predict_points_d_{ predict_points_d },
        num_classes_{ num_classes },
        num_sv_{ num_sv },
        num_predict_points_{ num_predict_points },
        num_features_{ num_features },
        kernel_function_parameter_{ std::make_tuple(std::forward<Args>(kernel_function_parameter)...) } { }

    /**
     * @brief Function call operator overload performing the actual calculation.
     * @param[in] nd_idx indices representing the current point in the execution space
     */
    void operator()(::sycl::nd_item<2> nd_idx) const {
        // cast values to 32-bit unsigned int values to prevent implicit conversions
        const auto local_id_0 = static_cast<unsigned>(nd_idx.get_local_id(0));
        const auto local_id_1 = static_cast<unsigned>(nd_idx.get_local_id(1));

        // cast all values to 64-bit std::size_t to prevent potential 32-bit overflows
        const auto INTERNAL_BLOCK_SIZE_uz = static_cast<std::size_t>(INTERNAL_BLOCK_SIZE);
        const auto THREAD_BLOCK_SIZE_uz = static_cast<std::size_t>(THREAD_BLOCK_SIZE);
        const auto FEATURE_BLOCK_SIZE_uz = static_cast<std::size_t>(FEATURE_BLOCK_SIZE);
        const auto PADDING_SIZE_uz = static_cast<std::size_t>(PADDING_SIZE);

        // calculate the indices used in the current work-item
        const auto pp_idx = nd_idx.get_global_id(1) * INTERNAL_BLOCK_SIZE_uz;
        const auto pp_idx_linear = nd_idx.get_group(1) * nd_idx.get_local_range(1) * INTERNAL_BLOCK_SIZE_uz + nd_idx.get_local_id(1);
        const auto sv_idx_linear = nd_idx.get_group(0) * nd_idx.get_local_range(0) * INTERNAL_BLOCK_SIZE_uz + nd_idx.get_local_id(1);

        // create a work-item private array used for internal caching
        real_type temp[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE]{};

        {
            // iterate over all features using blocking to be able to cache them for faster memory accesses
            for (std::size_t dim = 0; dim < num_features_; dim += FEATURE_BLOCK_SIZE_uz) {
                // load data into local memory
                for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                    const auto global_pp_idx = pp_idx_linear + static_cast<std::size_t>(internal) * THREAD_BLOCK_SIZE_uz;
                    const auto global_sv_idx = sv_idx_linear + static_cast<std::size_t>(internal) * THREAD_BLOCK_SIZE_uz;

                    // FEATURE_BLOCK_SIZE = 2 * THREAD_BLOCK_SIZE -> store twice as many values in the shared memory
                    data_cache_pp_[local_id_0][internal * THREAD_BLOCK_SIZE + local_id_1] = predict_points_d_[(dim + nd_idx.get_local_id(0)) * (num_predict_points_ + PADDING_SIZE_uz) + global_pp_idx];
                    data_cache_pp_[local_id_0 + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + local_id_1] = predict_points_d_[(dim + nd_idx.get_local_id(0) + THREAD_BLOCK_SIZE) * (num_predict_points_ + PADDING_SIZE_uz) + global_pp_idx];
                    data_cache_sv_[local_id_0][internal * THREAD_BLOCK_SIZE + local_id_1] = sv_d_[(dim + nd_idx.get_local_id(0)) * (num_sv_ + PADDING_SIZE_uz) + global_sv_idx];
                    data_cache_sv_[local_id_0 + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + local_id_1] = sv_d_[(dim + nd_idx.get_local_id(0) + THREAD_BLOCK_SIZE_uz) * (num_sv_ + PADDING_SIZE_uz) + global_sv_idx];
                }
                nd_idx.barrier();  // wait until all work-items loaded their part of the data

                // perform the feature reduction calculation
                for (unsigned block_dim = 0; block_dim < FEATURE_BLOCK_SIZE; ++block_dim) {
                    for (unsigned internal_pd = 0; internal_pd < INTERNAL_BLOCK_SIZE; ++internal_pd) {
                        for (unsigned internal_sv = 0; internal_sv < INTERNAL_BLOCK_SIZE; ++internal_sv) {
                            temp[internal_pd][internal_sv] += detail::feature_reduce<kernel_function>(data_cache_sv_[block_dim][local_id_0 * INTERNAL_BLOCK_SIZE + internal_sv],
                                                                                                      data_cache_pp_[block_dim][local_id_1 * INTERNAL_BLOCK_SIZE + internal_pd]);
                        }
                    }
                }
                nd_idx.barrier();  // wait until all work-items performed their part of the calculations
            }
        }

        // update temp using the respective kernel function
        for (unsigned internal_pd = 0; internal_pd < INTERNAL_BLOCK_SIZE; ++internal_pd) {
            for (unsigned internal_sv = 0; internal_sv < INTERNAL_BLOCK_SIZE; ++internal_sv) {
                temp[internal_pd][internal_sv] = detail::apply_kernel_function<kernel_function>(temp[internal_pd][internal_sv], kernel_function_parameter_);
            }
        }

        {
            // rename cached arrays
            auto &alpha_cache = data_cache_pp_;
            auto &out_cache = data_cache_sv_;

            // iterate over all features using blocking to be able to cache them for faster memory accesses
            for (std::size_t dim = 0; dim < num_classes_; dim += FEATURE_BLOCK_SIZE_uz) {
                // load data into local memory
                for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                    const std::size_t global_sv_idx = sv_idx_linear + static_cast<std::size_t>(internal) * THREAD_BLOCK_SIZE_uz;

                    alpha_cache[local_id_0][internal * THREAD_BLOCK_SIZE + local_id_1] = alpha_d_[(dim + nd_idx.get_local_id(0)) * (num_sv_ + PADDING_SIZE_uz) + global_sv_idx];
                    alpha_cache[local_id_0 + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + local_id_1] = alpha_d_[(dim + nd_idx.get_local_id(0) + THREAD_BLOCK_SIZE_uz) * (num_sv_ + PADDING_SIZE_uz) + global_sv_idx];

                    // the bias (rho) must only be applied once for all support vectors
                    if (nd_idx.get_group(0) == std::size_t{ 0 }) {
                        out_cache[local_id_0][internal * THREAD_BLOCK_SIZE + local_id_1] = -rho_d_[dim + nd_idx.get_local_id(0)];
                        out_cache[local_id_0 + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + local_id_1] = -rho_d_[dim + nd_idx.get_local_id(0) + THREAD_BLOCK_SIZE_uz];
                    } else {
                        out_cache[local_id_0][internal * THREAD_BLOCK_SIZE + local_id_1] = real_type{ 0.0 };
                        out_cache[local_id_0 + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + local_id_1] = real_type{ 0.0 };
                    }
                }
                nd_idx.barrier();  // wait until all work-items loaded their part of the data

                // calculate intermediate results and store them in local memory
                for (unsigned class_idx = 0; class_idx < FEATURE_BLOCK_SIZE; ++class_idx) {
                    for (unsigned internal_pd = 0; internal_pd < INTERNAL_BLOCK_SIZE; ++internal_pd) {
                        for (unsigned internal_sv = 0; internal_sv < INTERNAL_BLOCK_SIZE; ++internal_sv) {
                            out_cache[(class_idx + local_id_0) % FEATURE_BLOCK_SIZE][internal_pd * THREAD_BLOCK_SIZE + local_id_1] +=
                                temp[internal_pd][internal_sv] * alpha_cache[(class_idx + local_id_0) % FEATURE_BLOCK_SIZE][local_id_0 * INTERNAL_BLOCK_SIZE + internal_sv];
                        }
                    }
                    nd_idx.barrier();  // wait until all work-items performed their part of the calculations
                }

                // add intermediate cached results to prediction_d
                for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                    const auto global_pp_idx = pp_idx + static_cast<std::size_t>(internal);

                    detail::atomic_op<real_type>{ prediction_d_[global_pp_idx * (num_classes_ + PADDING_SIZE_uz) + dim + nd_idx.get_local_id(0)] } += out_cache[local_id_0][internal * THREAD_BLOCK_SIZE + local_id_1];
                    detail::atomic_op<real_type>{ prediction_d_[global_pp_idx * (num_classes_ + PADDING_SIZE_uz) + dim + nd_idx.get_local_id(0) + THREAD_BLOCK_SIZE_uz] } += out_cache[local_id_0 + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + local_id_1];
                }
                nd_idx.barrier();  // wait until all work-items updated their part of the prediction
            }
        }
    }

  private:
    /// Local memory used for internal memory access optimizations.
    ::sycl::local_accessor<real_type, 2> data_cache_pp_;
    /// Local memory used for internal memory access optimizations.
    ::sycl::local_accessor<real_type, 2> data_cache_sv_;

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
    const std::tuple<Args...> kernel_function_parameter_;
    /// @endcond
};

}  // namespace plssvm::sycl::detail

#endif  // PLSSVM_BACKENDS_SYCL_PREDICT_KERNEL_HPP_
