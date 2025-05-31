/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines the functions used for prediction for the C-SVM using the OpenMP backend.
 */

#ifndef PLSSVM_BACKENDS_OPENMP_KERNEL_PREDICT_KERNEL_HPP_
#define PLSSVM_BACKENDS_OPENMP_KERNEL_PREDICT_KERNEL_HPP_
#pragma once

#include "plssvm/constants.hpp"              // plssvm::real_type
#include "plssvm/detail/assert.hpp"          // PLSSVM_ASSERT
#include "plssvm/kernel_function_types.hpp"  // plssvm::kernel_function_type
#include "plssvm/matrix.hpp"                 // plssvm::aos_matrix, plssvm::matrix
#include "plssvm/shape.hpp"                  // plssvm::shape

#include <array>    // std::array
#include <cmath>    // std::fma
#include <cstddef>  // std::size_t
#include <vector>   // std::vector

namespace plssvm::openmp::detail {

/**
 * @brief Calculate the `w` vector used to speedup the prediction using the linear kernel function.
 * @param[out] w the vector to speedup the linear prediction
 * @param[in] alpha the previously learned weights
 * @param[in] support_vectors the support vectors
 * @param[in] device_num_sv the number of support vectors the current device is responsible for
 * @param[in] device_sv_offset the first row in @p support_vectors the current device is responsible for
 */
inline void device_kernel_w_linear(soa_matrix<real_type> &w, const aos_matrix<real_type> &alpha, const soa_matrix<real_type> &support_vectors, const std::size_t device_num_sv, const std::size_t device_sv_offset) {
    PLSSVM_ASSERT(alpha.num_cols() == support_vectors.num_rows(), "Size mismatch: {} vs {}!", alpha.num_cols(), support_vectors.num_rows());
    PLSSVM_ASSERT(w.shape() == (plssvm::shape{ alpha.num_rows(), support_vectors.num_cols() }), "Shape mismatch: {} vs {}!", w.shape(), (plssvm::shape{ alpha.num_rows(), support_vectors.num_cols() }));
    PLSSVM_ASSERT(support_vectors.num_rows() >= device_num_sv, "The number of place specific sv ({}) cannot be greater the the total number of sv ({})!", device_num_sv, support_vectors.num_rows());
    PLSSVM_ASSERT(support_vectors.num_rows() >= device_sv_offset, "The sv offset ({}) cannot be greater the the total number of sv ({})!", device_sv_offset, support_vectors.num_rows());

    // calculate constants
    const std::size_t num_classes = alpha.num_rows();
    const std::size_t num_features = support_vectors.num_cols();
    const auto blocked_num_features = static_cast<std::size_t>(std::ceil(static_cast<real_type>(num_features) / INTERNAL_BLOCK_SIZE));
    const auto blocked_num_classes = static_cast<std::size_t>(std::ceil(static_cast<real_type>(num_classes) / INTERNAL_BLOCK_SIZE));

    // cast all values to 64-bit unsigned long long to prevent potential 32-bit overflows
    const auto INTERNAL_BLOCK_SIZE_uz = static_cast<std::size_t>(INTERNAL_BLOCK_SIZE);
    const auto THREAD_BLOCK_SIZE_uz = static_cast<std::size_t>(THREAD_BLOCK_SIZE);

#pragma omp parallel for collapse(2) default(none) shared(w, support_vectors, alpha) firstprivate(blocked_num_classes, blocked_num_features, num_classes, num_features, device_num_sv, device_sv_offset)
    for (std::size_t feature_block = 0; feature_block < blocked_num_features; feature_block += THREAD_BLOCK_SIZE_uz) {
        for (std::size_t class_block = 0; class_block < blocked_num_classes; class_block += THREAD_BLOCK_SIZE_uz) {
            // perform operations on the current block
            for (std::size_t feature_thread = 0; feature_thread < THREAD_BLOCK_SIZE_uz; ++feature_thread) {
                for (std::size_t class_thread = 0; class_thread < THREAD_BLOCK_SIZE_uz; ++class_thread) {
                    // calculate the indices used in the current thread
                    const std::size_t feature_idx = (feature_block + feature_thread) * INTERNAL_BLOCK_SIZE_uz;
                    const std::size_t class_idx = (class_block + class_thread) * INTERNAL_BLOCK_SIZE_uz;

                    // create a thread private array used for internal caching
                    std::array<std::array<real_type, INTERNAL_BLOCK_SIZE>, INTERNAL_BLOCK_SIZE> temp{};

                    // iterate over all support vectors
                    for (std::size_t sv_block = 0; sv_block < device_num_sv; sv_block += THREAD_BLOCK_SIZE_uz) {
                        // perform the dot product calculation
                        for (unsigned internal_feature = 0; internal_feature < INTERNAL_BLOCK_SIZE; ++internal_feature) {
                            for (unsigned internal_class = 0; internal_class < INTERNAL_BLOCK_SIZE; ++internal_class) {
                                // calculate the indices to access the global data
                                const auto global_feature_idx = feature_idx + static_cast<std::size_t>(internal_feature);
                                const auto global_class_idx = class_idx + static_cast<std::size_t>(internal_class);

                                real_type sum{ 0.0 };
                                for (std::size_t sv = 0; sv < THREAD_BLOCK_SIZE_uz; ++sv) {
                                    sum += alpha(global_class_idx, device_sv_offset + sv_block + sv) * support_vectors(device_sv_offset + sv_block + sv, global_feature_idx);
                                }
                                temp[internal_class][internal_feature] += sum;
                            }
                        }
                    }

                    // store the result back to the w vector
                    for (unsigned internal_feature = 0; internal_feature < INTERNAL_BLOCK_SIZE; ++internal_feature) {
                        for (unsigned internal_class = 0; internal_class < INTERNAL_BLOCK_SIZE; ++internal_class) {
                            // calculate the indices to access the global data
                            const auto global_feature_idx = feature_idx + static_cast<std::size_t>(internal_feature);
                            const auto global_class_idx = class_idx + static_cast<std::size_t>(internal_class);

                            w(global_class_idx, global_feature_idx) = temp[internal_class][internal_feature];
                        }
                    }
                }
            }
        }
    }
}

/**
 * @brief Predict the @p predict_points_d using the linear kernel speeding up the calculation using the @p w_d vector.
 * @param[out] prediction the predicted values
 * @param[in] w the vector to speedup the calculations
 * @param[in] rho the previously learned bias
 * @param[in] predict_points the data points to predict
 * @param[in] device_num_predict_points the number of predict points the current device is responsible for
 * @param[in] device_row_offset the first row in @p predict_points the current device is responsible for
 */
inline void device_kernel_predict_linear(aos_matrix<real_type> &prediction, const soa_matrix<real_type> &w, const std::vector<real_type> &rho, const soa_matrix<real_type> &predict_points, const std::size_t device_num_predict_points, const std::size_t device_row_offset) {
    PLSSVM_ASSERT(w.num_rows() == rho.size(), "Size mismatch: {} vs {}!", w.num_rows(), rho.size());
    PLSSVM_ASSERT(w.num_cols() == predict_points.num_cols(), "Size mismatch: {} vs {}!", w.num_cols(), predict_points.num_cols());
    PLSSVM_ASSERT(prediction.shape() == (plssvm::shape{ predict_points.num_rows(), w.num_rows() }), "Shape mismatch: {} vs {}!", prediction.shape(), (plssvm::shape{ predict_points.num_rows(), w.num_rows() }));
    PLSSVM_ASSERT(predict_points.num_rows() >= device_num_predict_points, "The number of place specific predict points ({}) cannot be greater the the total number of predict points ({})!", device_num_predict_points, predict_points.num_rows());
    PLSSVM_ASSERT(predict_points.num_rows() >= device_row_offset, "The row offset ({}) cannot be greater the the total number of predict points ({})!", device_row_offset, predict_points.num_rows());

    // calculate constants
    const std::size_t num_classes = prediction.num_cols();
    const std::size_t num_features = predict_points.num_cols();
    const auto blocked_device_num_predict_points = static_cast<std::size_t>(std::ceil(static_cast<real_type>(device_num_predict_points) / INTERNAL_BLOCK_SIZE));
    const auto blocked_num_classes = static_cast<std::size_t>(std::ceil(static_cast<real_type>(num_classes) / INTERNAL_BLOCK_SIZE));

    // cast all values to 64-bit unsigned long long to prevent potential 32-bit overflows
    const auto INTERNAL_BLOCK_SIZE_uz = static_cast<std::size_t>(INTERNAL_BLOCK_SIZE);
    const auto THREAD_BLOCK_SIZE_uz = static_cast<std::size_t>(THREAD_BLOCK_SIZE);

#pragma omp parallel for collapse(2) default(none) shared(prediction, w, rho, predict_points) firstprivate(blocked_device_num_predict_points, blocked_num_classes, device_num_predict_points, num_classes, num_features, device_row_offset)
    for (std::size_t pp_block = 0; pp_block < blocked_device_num_predict_points; pp_block += THREAD_BLOCK_SIZE_uz) {
        for (std::size_t class_block = 0; class_block < blocked_num_classes; class_block += THREAD_BLOCK_SIZE_uz) {
            // perform operations on the current block
            for (std::size_t pp_thread = 0; pp_thread < THREAD_BLOCK_SIZE_uz; ++pp_thread) {
                for (std::size_t class_thread = 0; class_thread < THREAD_BLOCK_SIZE_uz; ++class_thread) {
                    // calculate the indices used in the current thread
                    const std::size_t pp_idx = (pp_block + pp_thread) * INTERNAL_BLOCK_SIZE_uz;
                    const std::size_t class_idx = (class_block + class_thread) * INTERNAL_BLOCK_SIZE_uz;

                    // create a thread private array used for internal caching
                    std::array<std::array<real_type, INTERNAL_BLOCK_SIZE>, INTERNAL_BLOCK_SIZE> temp{};

                    // iterate over all features
                    for (std::size_t feature_block = 0; feature_block < num_features; feature_block += THREAD_BLOCK_SIZE_uz) {
                        // perform the dot product calculation
                        for (unsigned internal_pp = 0; internal_pp < INTERNAL_BLOCK_SIZE; ++internal_pp) {
                            for (unsigned internal_class = 0; internal_class < INTERNAL_BLOCK_SIZE; ++internal_class) {
                                // calculate the indices to access the global data
                                const auto global_pp_idx = device_row_offset + pp_idx + static_cast<std::size_t>(internal_pp);
                                const auto global_class_idx = class_idx + static_cast<std::size_t>(internal_class);

                                real_type sum{ 0.0 };
                                for (std::size_t feature = 0; feature < THREAD_BLOCK_SIZE_uz; ++feature) {
                                    sum += w(global_class_idx, feature_block + feature) * predict_points(global_pp_idx, feature_block + feature);
                                }
                                temp[internal_class][internal_pp] += sum;
                            }
                        }
                    }

                    // store the result back to the w vector
                    for (unsigned internal_pp = 0; internal_pp < INTERNAL_BLOCK_SIZE; ++internal_pp) {
                        for (unsigned internal_class = 0; internal_class < INTERNAL_BLOCK_SIZE; ++internal_class) {
                            // calculate the indices to access the global data
                            const auto global_pp_idx = device_row_offset + pp_idx + static_cast<std::size_t>(internal_pp);
                            const auto global_class_idx = class_idx + static_cast<std::size_t>(internal_class);

                            prediction(global_pp_idx, global_class_idx) = temp[internal_class][internal_pp] - rho[global_class_idx];
                        }
                    }
                }
            }
        }
    }
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
 * @param[in] device_num_predict_points the number of predict points the current device is responsible for
 * @param[in] device_row_offset the first row in @p predict_points the current device is responsible for
 * @param[in] kernel_function_parameter the parameters necessary to apply the @p kernel_function
 */
template <kernel_function_type kernel_function, typename... Args>
inline void device_kernel_predict(aos_matrix<real_type> &prediction, const aos_matrix<real_type> &alpha, const std::vector<real_type> &rho, const soa_matrix<real_type> &support_vectors, const soa_matrix<real_type> &predict_points, const std::size_t device_num_predict_points, const std::size_t device_row_offset, Args... kernel_function_parameter) {
    PLSSVM_ASSERT(alpha.num_rows() == rho.size(), "Size mismatch: {} vs {}!", alpha.num_rows(), rho.size());
    PLSSVM_ASSERT(alpha.num_cols() == support_vectors.num_rows(), "Size mismatch: {} vs {}!", alpha.num_cols(), support_vectors.num_rows());
    PLSSVM_ASSERT(support_vectors.num_cols() == predict_points.num_cols(), "Size mismatch: {} vs {}!", support_vectors.num_cols(), predict_points.num_cols());
    PLSSVM_ASSERT(prediction.shape() == (plssvm::shape{ predict_points.num_rows(), alpha.num_rows() }), "Shape mismatch: {} vs {}!", prediction.shape(), (plssvm::shape{ predict_points.num_rows(), alpha.num_rows() }));
    PLSSVM_ASSERT(predict_points.num_rows() >= device_num_predict_points, "The number of place specific predict points ({}) cannot be greater the the total number of predict points ({})!", device_num_predict_points, predict_points.num_rows());
    PLSSVM_ASSERT(predict_points.num_rows() >= device_row_offset, "The row offset ({}) cannot be greater the the total number of predict points ({})!", device_row_offset, predict_points.num_rows());

    // calculate constants
    const std::size_t num_classes = alpha.num_rows();
    const std::size_t num_support_vectors = support_vectors.num_rows();
    const std::size_t num_features = predict_points.num_cols();
    const auto blocked_num_support_vectors = static_cast<std::size_t>(std::ceil(static_cast<real_type>(num_support_vectors) / INTERNAL_BLOCK_SIZE));
    const auto blocked_device_num_predict_points = static_cast<std::size_t>(std::ceil(static_cast<real_type>(device_num_predict_points) / INTERNAL_BLOCK_SIZE));

    // cast all values to 64-bit unsigned long long to prevent potential 32-bit overflows
    const auto INTERNAL_BLOCK_SIZE_uz = static_cast<std::size_t>(INTERNAL_BLOCK_SIZE);
    const auto THREAD_BLOCK_SIZE_uz = static_cast<std::size_t>(THREAD_BLOCK_SIZE);

#pragma omp parallel for collapse(2)
    for (std::size_t pp_idx = 0; pp_idx < device_num_predict_points; ++pp_idx) {
        for (std::size_t class_idx = 0; class_idx < num_classes; ++class_idx) {
            prediction(device_row_offset + pp_idx, class_idx) -= rho[class_idx];
        }
    }

#pragma omp parallel for collapse(2)
    for (std::size_t pp_block = 0; pp_block < blocked_device_num_predict_points; pp_block += THREAD_BLOCK_SIZE_uz) {
        for (std::size_t sv_block = 0; sv_block < blocked_num_support_vectors; sv_block += THREAD_BLOCK_SIZE_uz) {
            // perform operations on the current block
            for (std::size_t pp_thread = 0; pp_thread < THREAD_BLOCK_SIZE_uz; ++pp_thread) {
                for (std::size_t sv_thread = 0; sv_thread < THREAD_BLOCK_SIZE_uz; ++sv_thread) {
                    // calculate the indices used in the current thread
                    const std::size_t pp_idx = (pp_block + pp_thread) * INTERNAL_BLOCK_SIZE_uz;
                    const std::size_t sv_idx = (sv_block + sv_thread) * INTERNAL_BLOCK_SIZE_uz;

                    // create a thread private array used for internal caching
                    std::array<std::array<real_type, INTERNAL_BLOCK_SIZE>, INTERNAL_BLOCK_SIZE> temp{};

                    // iterate over all features
                    for (std::size_t feature_block = 0; feature_block < num_features; feature_block += THREAD_BLOCK_SIZE_uz) {
                        // perform the feature reduction calculation
                        for (unsigned internal_pp = 0; internal_pp < INTERNAL_BLOCK_SIZE; ++internal_pp) {
                            for (unsigned internal_sv = 0; internal_sv < INTERNAL_BLOCK_SIZE; ++internal_sv) {
                                // calculate the indices to access the global data
                                const auto global_pp_idx = device_row_offset + pp_idx + static_cast<std::size_t>(internal_pp);
                                const auto global_sv_idx = sv_idx + static_cast<std::size_t>(internal_sv);

                                real_type sum{ 0.0 };
                                for (std::size_t feature = 0; feature < THREAD_BLOCK_SIZE_uz; ++feature) {
                                    sum += detail::feature_reduce<kernel_function>(support_vectors(global_sv_idx, feature_block + feature), predict_points(global_pp_idx, feature_block + feature));
                                }
                                temp[internal_sv][internal_pp] += sum;
                            }
                        }
                    }

                    // update temp using the respective kernel function
                    for (unsigned internal_pp = 0; internal_pp < INTERNAL_BLOCK_SIZE; ++internal_pp) {
                        for (unsigned internal_sv = 0; internal_sv < INTERNAL_BLOCK_SIZE; ++internal_sv) {
                            temp[internal_sv][internal_pp] = detail::apply_kernel_function<kernel_function>(temp[internal_sv][internal_pp], kernel_function_parameter...);
                        }
                    }

                    // add results to prediction
                    for (std::size_t class_block = 0; class_block < num_classes; class_block += THREAD_BLOCK_SIZE_uz) {
                        for (unsigned internal_pp = 0; internal_pp < INTERNAL_BLOCK_SIZE; ++internal_pp) {
                            for (unsigned internal_sv = 0; internal_sv < INTERNAL_BLOCK_SIZE; ++internal_sv) {
                                // calculate the indices to access the global data and the data with respect to the current device
                                const auto global_pp_idx = device_row_offset + pp_idx + static_cast<std::size_t>(internal_pp);
                                const auto global_sv_idx = sv_idx + static_cast<std::size_t>(internal_sv);

                                for (std::size_t class_idx = 0; class_idx < THREAD_BLOCK_SIZE_uz; ++class_idx) {
#pragma omp atomic
                                    prediction(global_pp_idx, class_block + class_idx) += alpha(class_block + class_idx, global_sv_idx) * temp[internal_sv][internal_pp];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

}  // namespace plssvm::openmp::detail

#endif  // PLSSVM_BACKENDS_OPENMP_KERNEL_PREDICT_KERNEL_HPP_
