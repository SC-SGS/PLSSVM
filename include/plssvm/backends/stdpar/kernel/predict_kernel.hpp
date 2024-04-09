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

#include "plssvm/backends/stdpar/kernel/kernel_functions.hpp"  // plssvm::stdpar::detail::{feature_reduce, apply_kernel_function}
#include "plssvm/constants.hpp"                                // plssvm::{real_type, INTERNAL_BLOCK_SIZE, FEATURE_BLOCK_SIZE, PADDING_SIZE}
#include "plssvm/detail/assert.hpp"                            // PLSSVM_ASSERT
#include "plssvm/kernel_function_types.hpp"                    // plssvm::kernel_function_type
#include "plssvm/matrix.hpp"                                   // plssvm::aos_matrix, plssvm::soa_matrix
#include "plssvm/shape.hpp"                                    // plssvm::shape

#if defined(PLSSVM_STDPAR_BACKEND_HAS_ACPP)
    #include "plssvm/backends/SYCL/detail/atomics.hpp"  // plssvm::sycl::detail::atomic_op
template <typename T>
using atomic_ref = plssvm::sycl::detail::atomic_op<T>;
#else
// TODO: other stdpar implementations
#endif

#include <cmath>      // std::fma
#include <cstddef>    // std::size_t
#include <execution>  // std::execution::par_unseq
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

    const std::size_t num_features = support_vectors.num_cols();
    const std::size_t blocked_num_features = (num_features + PADDING_SIZE) / INTERNAL_BLOCK_SIZE;
    const std::size_t num_classes = alpha.num_rows();
    const std::size_t blocked_num_classes = (num_classes + PADDING_SIZE) / INTERNAL_BLOCK_SIZE;
    const std::size_t num_support_vectors = support_vectors.num_rows();

    std::vector<std::pair<std::size_t, std::size_t>> range(blocked_num_features * blocked_num_classes);
    for (std::size_t i = 0; i < range.size(); ++i) {
        range[i] = std::make_pair(i / blocked_num_classes, i % blocked_num_classes);
    }

    std::for_each(std::execution::par_unseq, range.cbegin(), range.cend(), [=, w_ptr = w.data(), alpha_ptr = alpha.data(), sv_ptr = support_vectors.data()](const std::pair<std::size_t, std::size_t> i) {
        const auto [feature_idx, class_idx] = i;

        real_type temp[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE] = { { 0.0 } };

        for (unsigned long long sv = 0; sv < num_support_vectors; ++sv) {
            // calculation
            for (unsigned internal_feature = 0; internal_feature < INTERNAL_BLOCK_SIZE; ++internal_feature) {
                for (unsigned internal_class = 0; internal_class < INTERNAL_BLOCK_SIZE; ++internal_class) {
                    temp[internal_feature][internal_class] += alpha_ptr[(class_idx * INTERNAL_BLOCK_SIZE + internal_class) * (num_support_vectors + PADDING_SIZE) + sv] * sv_ptr[(feature_idx * INTERNAL_BLOCK_SIZE + internal_feature) * (num_support_vectors + PADDING_SIZE) + sv];
                }
            }
        }

        for (unsigned internal_feature = 0; internal_feature < INTERNAL_BLOCK_SIZE; ++internal_feature) {
            for (unsigned internal_class = 0; internal_class < INTERNAL_BLOCK_SIZE; ++internal_class) {
                const unsigned long long global_feature_idx = feature_idx * INTERNAL_BLOCK_SIZE + internal_feature;
                const unsigned long long global_class_idx = class_idx * INTERNAL_BLOCK_SIZE + internal_class;

                w_ptr[global_feature_idx * (num_classes + PADDING_SIZE) + global_class_idx] = temp[internal_feature][internal_class];
            }
        }
    });
}

/**
 * @brief Predict the @p predict_points_d using the linear kernel speeding up the calculation using the @p w_d vector.
 * @param[out] out the predicted values
 * @param[in] w the vector to speedup the calculations
 * @param[in] rho the previously learned bias
 * @param[in] predict_points the data points to predict
 */
inline void device_kernel_predict_linear(aos_matrix<real_type> &out, const soa_matrix<real_type> &w, const std::vector<real_type> &rho, const soa_matrix<real_type> &predict_points) {
    PLSSVM_ASSERT(w.num_rows() == rho.size(), "Size mismatch: {} vs {}!", w.num_rows(), rho.size());
    PLSSVM_ASSERT(w.num_cols() == predict_points.num_cols(), "Size mismatch: {} vs {}!", w.num_cols(), predict_points.num_cols());
    PLSSVM_ASSERT(out.shape() == (plssvm::shape{ predict_points.num_rows(), w.num_rows() }), "Shape mismatch: {} vs {}!", out.shape(), (plssvm::shape{ predict_points.num_rows(), w.num_rows() }));

    const std::size_t num_predict_points = predict_points.num_rows();
    const std::size_t blocked_num_predict_points = (num_predict_points + PADDING_SIZE) / INTERNAL_BLOCK_SIZE;
    const std::size_t num_classes = out.num_cols();
    const std::size_t blocked_num_classes = (num_classes + PADDING_SIZE) / INTERNAL_BLOCK_SIZE;
    const std::size_t num_features = predict_points.num_cols();

    std::vector<std::pair<std::size_t, std::size_t>> range(blocked_num_predict_points * blocked_num_classes);
    for (std::size_t i = 0; i < range.size(); ++i) {
        range[i] = std::make_pair(i / blocked_num_classes, i % blocked_num_classes);
    }

    std::for_each(std::execution::par_unseq, range.cbegin(), range.cend(), [=, out_ptr = out.data(), w_ptr = w.data(), rho_ptr = rho.data(), pp_ptr = predict_points.data()](const std::pair<std::size_t, std::size_t> i) {
        const auto [pp_idx, class_idx] = i;

        real_type temp[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE] = { { 0.0 } };

        for (unsigned long long dim = 0; dim < num_features; ++dim) {
            // calculation
            for (unsigned internal_pp = 0; internal_pp < INTERNAL_BLOCK_SIZE; ++internal_pp) {
                for (unsigned internal_class = 0; internal_class < INTERNAL_BLOCK_SIZE; ++internal_class) {
                    temp[internal_pp][internal_class] += w_ptr[dim * (num_classes + PADDING_SIZE) + class_idx * INTERNAL_BLOCK_SIZE + internal_class] * pp_ptr[dim * (num_predict_points + PADDING_SIZE) + pp_idx * INTERNAL_BLOCK_SIZE + internal_pp];
                }
            }
        }

        for (unsigned internal_pp = 0; internal_pp < INTERNAL_BLOCK_SIZE; ++internal_pp) {
            for (unsigned internal_class = 0; internal_class < INTERNAL_BLOCK_SIZE; ++internal_class) {
                const unsigned long long global_pp_idx = pp_idx * INTERNAL_BLOCK_SIZE + internal_pp;
                const unsigned long long global_class_idx = class_idx * INTERNAL_BLOCK_SIZE + internal_class;

                out_ptr[global_pp_idx * (num_classes + PADDING_SIZE) + global_class_idx] = temp[internal_pp][internal_class] - rho_ptr[global_class_idx];
            }
        }
    });
}

/**
 * @brief Predict the @p predict_points_d using the @p kernel_function.
 * @tparam kernel_function the type of the used kernel function
 * @tparam Args the types of the parameters necessary for the specific kernel function
 * @param[out] out the predicted values
 * @param[in] alpha the previously learned weights
 * @param[in] rho the previously learned bias
 * @param[in] support_vectors the support vectors
 * @param[in] predict_points the data points to predict
 * @param[in] kernel_function_parameter the parameters necessary to apply the @p kernel_function
 */
template <kernel_function_type kernel_function, typename... Args>
inline void device_kernel_predict(aos_matrix<real_type> &out, const aos_matrix<real_type> &alpha, const std::vector<real_type> &rho, const soa_matrix<real_type> &support_vectors, const soa_matrix<real_type> &predict_points, Args... kernel_function_parameter) {
    PLSSVM_ASSERT(alpha.num_rows() == rho.size(), "Size mismatch: {} vs {}!", alpha.num_rows(), rho.size());
    PLSSVM_ASSERT(alpha.num_cols() == support_vectors.num_rows(), "Size mismatch: {} vs {}!", alpha.num_cols(), support_vectors.num_rows());
    PLSSVM_ASSERT(support_vectors.num_cols() == predict_points.num_cols(), "Size mismatch: {} vs {}!", support_vectors.num_cols(), predict_points.num_cols());
    PLSSVM_ASSERT(out.shape() == (plssvm::shape{ predict_points.num_rows(), alpha.num_rows() }), "Shape mismatch: {} vs {}!", out.shape(), (plssvm::shape{ predict_points.num_rows(), alpha.num_rows() }));

    const std::size_t num_predict_points = predict_points.num_rows();
    const std::size_t blocked_num_predict_points = (num_predict_points + PADDING_SIZE) / INTERNAL_BLOCK_SIZE;
    const std::size_t num_support_vectors = support_vectors.num_rows();
    const std::size_t blocked_num_support_vectors = (num_support_vectors + PADDING_SIZE) / INTERNAL_BLOCK_SIZE;
    const std::size_t num_classes = alpha.num_rows();
    const std::size_t num_features = predict_points.num_cols();

    std::vector<std::pair<std::size_t, std::size_t>> range(blocked_num_predict_points * blocked_num_support_vectors);
    for (std::size_t i = 0; i < range.size(); ++i) {
        range[i] = std::make_pair(i / blocked_num_support_vectors, i % blocked_num_support_vectors);
    }

    std::for_each(std::execution::par_unseq, range.cbegin(), range.cend(), [=, out_ptr = out.data(), alpha_ptr = alpha.data(), rho_ptr = rho.data(), sv_ptr = support_vectors.data(), pp_ptr = predict_points.data()](const std::pair<std::size_t, std::size_t> i) {
        const auto [pp_idx, sv_idx] = i;

        real_type temp[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE] = { { 0.0 } };

        for (unsigned long long dim = 0; dim < num_features; ++dim) {
            // calculation
            for (unsigned internal_pp = 0; internal_pp < INTERNAL_BLOCK_SIZE; ++internal_pp) {
                for (unsigned internal_sv = 0; internal_sv < INTERNAL_BLOCK_SIZE; ++internal_sv) {
                    temp[internal_pp][internal_sv] += detail::feature_reduce<kernel_function>(sv_ptr[dim * (num_support_vectors + PADDING_SIZE) + sv_idx * INTERNAL_BLOCK_SIZE + internal_sv],
                                                                                              pp_ptr[dim * (num_predict_points + PADDING_SIZE) + pp_idx * INTERNAL_BLOCK_SIZE + internal_pp]);
                }
            }
        }

        // update temp using the respective kernel function
        for (unsigned internal_pp = 0; internal_pp < INTERNAL_BLOCK_SIZE; ++internal_pp) {
            for (unsigned internal_sv = 0; internal_sv < INTERNAL_BLOCK_SIZE; ++internal_sv) {
                temp[internal_pp][internal_sv] = detail::apply_kernel_function<kernel_function>(temp[internal_pp][internal_sv], kernel_function_parameter...);
            }
        }

        for (unsigned long long dim = 0; dim < num_classes; ++dim) {
            for (unsigned internal_pp = 0; internal_pp < INTERNAL_BLOCK_SIZE; ++internal_pp) {
                for (unsigned internal_sv = 0; internal_sv < INTERNAL_BLOCK_SIZE; ++internal_sv) {
                    if (sv_idx * INTERNAL_BLOCK_SIZE + internal_sv == 0) {
                        atomic_ref<real_type>{ out_ptr[(pp_idx * INTERNAL_BLOCK_SIZE + internal_pp) * (num_classes + PADDING_SIZE) + dim] } += -rho_ptr[dim];
                    }
                    atomic_ref<real_type>{ out_ptr[(pp_idx * INTERNAL_BLOCK_SIZE + internal_pp) * (num_classes + PADDING_SIZE) + dim] } +=
                        temp[internal_pp][internal_sv] * alpha_ptr[dim * (num_support_vectors + PADDING_SIZE) + sv_idx * INTERNAL_BLOCK_SIZE + internal_sv];
                }
            }
        }
    });
}

}  // namespace plssvm::stdpar::detail

#endif  // PLSSVM_BACKENDS_STDPAR_KERNEL_PREDICT_KERNEL_HPP_
