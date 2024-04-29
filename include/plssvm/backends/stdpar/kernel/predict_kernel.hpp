/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines the functions used for prediction for the C-SVM using the stdpar backend.
 */

#ifndef PLSSVM_BACKENDS_STDPAR_KERNEL_PREDICT_KERNEL_HPP_
#define PLSSVM_BACKENDS_STDPAR_KERNEL_PREDICT_KERNEL_HPP_
#pragma once

#include "plssvm/backends/stdpar/detail/utility.hpp"           // plssvm::stdpar::detail::atomic_ref
#include "plssvm/backends/stdpar/kernel/kernel_functions.hpp"  // plssvm::stdpar::detail::{feature_reduce, apply_kernel_function}
#include "plssvm/constants.hpp"                                // plssvm::{real_type, INTERNAL_BLOCK_SIZE, FEATURE_BLOCK_SIZE, PADDING_SIZE}
#include "plssvm/detail/assert.hpp"                            // PLSSVM_ASSERT
#include "plssvm/kernel_function_types.hpp"                    // plssvm::kernel_function_type
#include "plssvm/matrix.hpp"                                   // plssvm::aos_matrix, plssvm::soa_matrix
#include "plssvm/shape.hpp"                                    // plssvm::shape

#include <algorithm>  // std::for_each
#include <array>      // std::array
#include <cmath>      // std::fma
#include <cstddef>    // std::size_t
#include <execution>  // std::execution::par_unseq
#include <utility>    // std::pair, std::make_pair
#include <vector>     // std::vector

namespace plssvm::stdpar::detail {

/**
 * @brief Calculate the `w` vector used to speedup the prediction using the linear kernel function.
 * @param[out] w the vector to speedup the linear prediction
 * @param[in] alpha the previously learned weights
 * @param[in] support_vectors the support vectors
 */
inline void device_kernel_w_linear(soa_matrix<real_type> &w, const aos_matrix<real_type> &alpha, const soa_matrix<real_type> &support_vectors) {
    PLSSVM_ASSERT(alpha.num_cols() == support_vectors.num_rows(), "Size mismatch: {} vs {}!", alpha.num_cols(), support_vectors.num_rows());
    PLSSVM_ASSERT(w.shape() == (plssvm::shape{ alpha.num_rows(), support_vectors.num_cols() }), "Shape mismatch: {} vs {}!", w.shape(), (plssvm::shape{ alpha.num_rows(), support_vectors.num_cols() }));

    // calculate constants
    const std::size_t num_features = support_vectors.num_cols();
    const auto blocked_num_features = static_cast<std::size_t>(std::ceil(static_cast<real_type>(num_features) / INTERNAL_BLOCK_SIZE));
    const std::size_t num_classes = alpha.num_rows();
    const auto blocked_num_classes = static_cast<std::size_t>(std::ceil(static_cast<real_type>(num_classes) / INTERNAL_BLOCK_SIZE));
    const std::size_t num_support_vectors = support_vectors.num_rows();

    // cast all values to 64-bit unsigned long long to prevent potential 32-bit overflows
    const auto INTERNAL_BLOCK_SIZE_uz = static_cast<std::size_t>(INTERNAL_BLOCK_SIZE);
    const auto PADDING_SIZE_uz = static_cast<std::size_t>(PADDING_SIZE);

    // calculate indices over which we parallelize
    std::vector<std::pair<std::size_t, std::size_t>> range(blocked_num_features * blocked_num_classes);
    for (std::size_t i = 0; i < range.size(); ++i) {
        range[i] = std::make_pair(i / blocked_num_classes, i % blocked_num_classes);
    }

    std::for_each(std::execution::par_unseq, range.begin(), range.end(), [=, w_ptr = w.data(), alpha_ptr = alpha.data(), sv_ptr = support_vectors.data()](const std::pair<std::size_t, std::size_t> idx) {
        // calculate the indices used in the current thread
        const auto [feature, c] = idx;
        const std::size_t feature_idx = feature * INTERNAL_BLOCK_SIZE_uz;
        const std::size_t class_idx = c * INTERNAL_BLOCK_SIZE_uz;

        // create a thread private array used for internal caching
        std::array<std::array<real_type, INTERNAL_BLOCK_SIZE>, INTERNAL_BLOCK_SIZE> temp{};

        // iterate over all features
        for (std::size_t sv = 0; sv < num_support_vectors; ++sv) {
            // perform the feature reduction calculation
            for (unsigned internal_feature = 0; internal_feature < INTERNAL_BLOCK_SIZE; ++internal_feature) {
                for (unsigned internal_class = 0; internal_class < INTERNAL_BLOCK_SIZE; ++internal_class) {
                    const std::size_t global_feature_idx = feature_idx + static_cast<std::size_t>(internal_feature);
                    const std::size_t global_class_idx = class_idx + static_cast<std::size_t>(internal_class);

                    temp[internal_feature][internal_class] += alpha_ptr[global_class_idx * (num_support_vectors + PADDING_SIZE_uz) + sv] * sv_ptr[global_feature_idx * (num_support_vectors + PADDING_SIZE_uz) + sv];
                }
            }
        }

        // update global array with local one
        for (unsigned internal_feature = 0; internal_feature < INTERNAL_BLOCK_SIZE; ++internal_feature) {
            for (unsigned internal_class = 0; internal_class < INTERNAL_BLOCK_SIZE; ++internal_class) {
                const std::size_t global_feature_idx = feature_idx + static_cast<std::size_t>(internal_feature);
                const std::size_t global_class_idx = class_idx + static_cast<std::size_t>(internal_class);

                w_ptr[global_feature_idx * (num_classes + PADDING_SIZE_uz) + global_class_idx] = temp[internal_feature][internal_class];
            }
        }
    });
}

/**
 * @brief Predict the @p predict_points_d using the linear kernel speeding up the calculation using the @p w_d vector.
 * @param[out] prediction the predicted values
 * @param[in] w the vector to speedup the calculations
 * @param[in] rho the previously learned bias
 * @param[in] predict_points the data points to predict
 */
inline void device_kernel_predict_linear(aos_matrix<real_type> &prediction, const soa_matrix<real_type> &w, const std::vector<real_type> &rho, const soa_matrix<real_type> &predict_points) {
    PLSSVM_ASSERT(w.num_rows() == rho.size(), "Size mismatch: {} vs {}!", w.num_rows(), rho.size());
    PLSSVM_ASSERT(w.num_cols() == predict_points.num_cols(), "Size mismatch: {} vs {}!", w.num_cols(), predict_points.num_cols());
    PLSSVM_ASSERT(prediction.shape() == (plssvm::shape{ predict_points.num_rows(), w.num_rows() }), "Shape mismatch: {} vs {}!", prediction.shape(), (plssvm::shape{ predict_points.num_rows(), w.num_rows() }));

    // calculate constants
    const std::size_t num_predict_points = predict_points.num_rows();
    const auto blocked_num_predict_points = static_cast<std::size_t>(std::ceil(static_cast<real_type>(num_predict_points) / INTERNAL_BLOCK_SIZE));
    const std::size_t num_classes = prediction.num_cols();
    const auto blocked_num_classes = static_cast<std::size_t>(std::ceil(static_cast<real_type>(num_classes) / INTERNAL_BLOCK_SIZE));
    const std::size_t num_features = predict_points.num_cols();

    // cast all values to 64-bit unsigned long long to prevent potential 32-bit overflows
    const auto INTERNAL_BLOCK_SIZE_uz = static_cast<std::size_t>(INTERNAL_BLOCK_SIZE);
    const auto PADDING_SIZE_uz = static_cast<std::size_t>(PADDING_SIZE);

    // calculate indices over which we parallelize
    std::vector<std::pair<std::size_t, std::size_t>> range(blocked_num_predict_points * blocked_num_classes);
    for (std::size_t i = 0; i < range.size(); ++i) {
        range[i] = std::make_pair(i / blocked_num_classes, i % blocked_num_classes);
    }

    std::for_each(std::execution::par_unseq, range.begin(), range.end(), [=, prediction_ptr = prediction.data(), w_ptr = w.data(), rho_ptr = rho.data(), pp_ptr = predict_points.data()](const std::pair<std::size_t, std::size_t> idx) {
        // calculate the indices used in the current thread
        const auto [pp, c] = idx;
        const std::size_t pp_idx = pp * INTERNAL_BLOCK_SIZE_uz;
        const std::size_t class_idx = c * INTERNAL_BLOCK_SIZE_uz;

        // create a thread private array used for internal caching
        std::array<std::array<real_type, INTERNAL_BLOCK_SIZE>, INTERNAL_BLOCK_SIZE> temp{};

        // iterate over all features
        for (std::size_t dim = 0; dim < num_features; ++dim) {
            // perform the feature reduction calculation
            for (unsigned internal_pp = 0; internal_pp < INTERNAL_BLOCK_SIZE; ++internal_pp) {
                for (unsigned internal_class = 0; internal_class < INTERNAL_BLOCK_SIZE; ++internal_class) {
                    const std::size_t global_pp_idx = pp_idx + static_cast<std::size_t>(internal_pp);
                    const std::size_t global_class_idx = class_idx + static_cast<std::size_t>(internal_class);

                    temp[internal_pp][internal_class] += w_ptr[dim * (num_classes + PADDING_SIZE_uz) + global_class_idx] * pp_ptr[dim * (num_predict_points + PADDING_SIZE_uz) + global_pp_idx];
                }
            }
        }

        // perform the dot product calculation
        for (unsigned internal_pp = 0; internal_pp < INTERNAL_BLOCK_SIZE; ++internal_pp) {
            for (unsigned internal_class = 0; internal_class < INTERNAL_BLOCK_SIZE; ++internal_class) {
                const std::size_t global_pp_idx = pp_idx + static_cast<std::size_t>(internal_pp);
                const std::size_t global_class_idx = class_idx + static_cast<std::size_t>(internal_class);

                if (global_pp_idx < num_predict_points && global_class_idx < num_classes) {
                    prediction_ptr[global_pp_idx * (num_classes + PADDING_SIZE_uz) + global_class_idx] = temp[internal_pp][internal_class] - rho_ptr[global_class_idx];
                }
            }
        }
    });
}

/**
 * @brief Predict the @p predict_points_d using the @p kernel_function.
 * @tparam kernel the type of the used kernel function
 * @tparam Args the types of the parameters necessary for the specific kernel function
 * @param[out] prediction the predicted values
 * @param[in] alpha the previously learned weights
 * @param[in] rho the previously learned bias
 * @param[in] support_vectors the support vectors
 * @param[in] predict_points the data points to predict
 * @param[in] kernel_function_parameter the parameters necessary to apply the @p kernel_function
 */
template <kernel_function_type kernel, typename... Args>
inline void device_kernel_predict(aos_matrix<real_type> &prediction, const aos_matrix<real_type> &alpha, const std::vector<real_type> &rho, const soa_matrix<real_type> &support_vectors, const soa_matrix<real_type> &predict_points, Args... kernel_function_parameter) {
    PLSSVM_ASSERT(alpha.num_rows() == rho.size(), "Size mismatch: {} vs {}!", alpha.num_rows(), rho.size());
    PLSSVM_ASSERT(alpha.num_cols() == support_vectors.num_rows(), "Size mismatch: {} vs {}!", alpha.num_cols(), support_vectors.num_rows());
    PLSSVM_ASSERT(support_vectors.num_cols() == predict_points.num_cols(), "Size mismatch: {} vs {}!", support_vectors.num_cols(), predict_points.num_cols());
    PLSSVM_ASSERT(prediction.shape() == (plssvm::shape{ predict_points.num_rows(), alpha.num_rows() }), "Shape mismatch: {} vs {}!", prediction.shape(), (plssvm::shape{ predict_points.num_rows(), alpha.num_rows() }));

    // calculate constants
    const std::size_t num_classes = alpha.num_rows();
    const std::size_t num_support_vectors = support_vectors.num_rows();
    const auto blocked_num_support_vectors = static_cast<std::size_t>(std::ceil(static_cast<real_type>(num_support_vectors) / INTERNAL_BLOCK_SIZE));
    const std::size_t num_predict_points = predict_points.num_rows();
    const auto blocked_num_predict_points = static_cast<std::size_t>(std::ceil(static_cast<real_type>(num_predict_points) / INTERNAL_BLOCK_SIZE));
    const std::size_t num_features = predict_points.num_cols();

    // cast all values to 64-bit unsigned long long to prevent potential 32-bit overflows
    const auto INTERNAL_BLOCK_SIZE_uz = static_cast<std::size_t>(INTERNAL_BLOCK_SIZE);
    const auto PADDING_SIZE_uz = static_cast<std::size_t>(PADDING_SIZE);

    // calculate indices over which we parallelize
    std::vector<std::pair<std::size_t, std::size_t>> range(blocked_num_predict_points * blocked_num_support_vectors);
    for (std::size_t i = 0; i < range.size(); ++i) {
        range[i] = std::make_pair(i / blocked_num_support_vectors, i % blocked_num_support_vectors);
    }

    std::for_each(std::execution::par_unseq, range.begin(), range.end(), [=, prediction_ptr = prediction.data(), alpha_ptr = alpha.data(), rho_ptr = rho.data(), sv_ptr = support_vectors.data(), pp_ptr = predict_points.data()](const std::pair<std::size_t, std::size_t> idx) {
        // calculate the indices used in the current thread
        const auto [pp, sv] = idx;
        const std::size_t pp_idx = pp * INTERNAL_BLOCK_SIZE_uz;
        const std::size_t sv_idx = sv * INTERNAL_BLOCK_SIZE_uz;

        // create a thread private array used for internal caching
        std::array<std::array<real_type, INTERNAL_BLOCK_SIZE>, INTERNAL_BLOCK_SIZE> temp{};

        // iterate over all features
        for (std::size_t dim = 0; dim < num_features; ++dim) {
            // perform the feature reduction calculation
            for (unsigned internal_pp = 0; internal_pp < INTERNAL_BLOCK_SIZE; ++internal_pp) {
                for (unsigned internal_sv = 0; internal_sv < INTERNAL_BLOCK_SIZE; ++internal_sv) {
                    const std::size_t global_pp_idx = pp_idx + static_cast<std::size_t>(internal_pp);
                    const std::size_t global_sv_idx = sv_idx + static_cast<std::size_t>(internal_sv);

                    temp[internal_pp][internal_sv] += detail::feature_reduce<kernel>(sv_ptr[dim * (num_support_vectors + PADDING_SIZE_uz) + global_sv_idx],
                                                                                     pp_ptr[dim * (num_predict_points + PADDING_SIZE_uz) + global_pp_idx]);
                }
            }
        }

        // update temp using the respective kernel function
        for (unsigned internal_pp = 0; internal_pp < INTERNAL_BLOCK_SIZE; ++internal_pp) {
            for (unsigned internal_sv = 0; internal_sv < INTERNAL_BLOCK_SIZE; ++internal_sv) {
                temp[internal_pp][internal_sv] = detail::apply_kernel_function<kernel>(temp[internal_pp][internal_sv], kernel_function_parameter...);
            }
        }

        // add results to prediction
        for (std::size_t a = 0; a < num_classes; ++a) {
            for (unsigned internal_pp = 0; internal_pp < INTERNAL_BLOCK_SIZE; ++internal_pp) {
                for (unsigned internal_sv = 0; internal_sv < INTERNAL_BLOCK_SIZE; ++internal_sv) {
                    const std::size_t global_pp_idx = pp_idx + static_cast<std::size_t>(internal_pp);
                    const std::size_t global_sv_idx = sv_idx + static_cast<std::size_t>(internal_sv);

                    // be sure to not perform out of bounds accesses
                    if (global_pp_idx < num_predict_points && global_sv_idx < num_support_vectors) {
                        if (global_sv_idx == 0) {
                            atomic_ref<real_type>{ prediction_ptr[global_pp_idx * (num_classes + PADDING_SIZE_uz) + a] } += -rho_ptr[a];
                        }
                        atomic_ref<real_type>{ prediction_ptr[global_pp_idx * (num_classes + PADDING_SIZE_uz) + a] } +=
                            temp[internal_pp][internal_sv] * alpha_ptr[a * (num_support_vectors + PADDING_SIZE_uz) + global_sv_idx];
                    }
                }
            }
        }
    });
}

}  // namespace plssvm::stdpar::detail

#endif  // PLSSVM_BACKENDS_STDPAR_KERNEL_PREDICT_KERNEL_HPP_
