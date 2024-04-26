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
 */
inline void device_kernel_w_linear(soa_matrix<real_type> &w, const aos_matrix<real_type> &alpha, const soa_matrix<real_type> &support_vectors) {
    PLSSVM_ASSERT(alpha.num_cols() == support_vectors.num_rows(), "Size mismatch: {} vs {}!", alpha.num_cols(), support_vectors.num_rows());
    PLSSVM_ASSERT(w.shape() == (plssvm::shape{ alpha.num_rows(), support_vectors.num_cols() }), "Shape mismatch: {} vs {}!", w.shape(), (plssvm::shape{ alpha.num_rows(), support_vectors.num_cols() }));

    // calculate constants
    const std::size_t num_classes = alpha.num_rows();
    const std::size_t num_support_vectors = support_vectors.num_rows();
    const std::size_t num_features = support_vectors.num_cols();

#pragma omp parallel for collapse(2) default(none) shared(w, support_vectors, alpha) firstprivate(num_classes, num_features, num_support_vectors)
    for (std::size_t a = 0; a < num_classes; ++a) {
        for (std::size_t dim = 0; dim < num_features; ++dim) {
            real_type temp{ 0.0 };
#pragma omp simd reduction(+ : temp)
            for (std::size_t idx = 0; idx < num_support_vectors; ++idx) {
                temp = std::fma(alpha(a, idx), support_vectors(idx, dim), temp);
            }
            w(a, dim) = temp;
        }
    }
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
    const std::size_t num_classes = prediction.num_cols();
    const std::size_t num_predict_points = predict_points.num_rows();
    const std::size_t num_features = predict_points.num_cols();

#pragma omp parallel for collapse(2) default(none) shared(prediction, w, rho, predict_points) firstprivate(num_classes, num_features, num_predict_points)
    for (std::size_t point_index = 0; point_index < num_predict_points; ++point_index) {
        for (std::size_t a = 0; a < num_classes; ++a) {
            real_type temp{ 0.0 };
#pragma omp simd reduction(+ : temp)
            for (std::size_t dim = 0; dim < num_features; ++dim) {
                temp = std::fma(w(a, dim), predict_points(point_index, dim), temp);
            }
            prediction(point_index, a) = temp - rho[a];
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
    const auto THREAD_BLOCK_SIZE_uz = static_cast<std::size_t>(THREAD_BLOCK_SIZE);

#pragma omp parallel for collapse(2)
    for (std::size_t point_index = 0; point_index < num_predict_points; ++point_index) {
        for (std::size_t a = 0; a < num_classes; ++a) {
            prediction(point_index, a) -= rho[a];
        }
    }

#pragma omp parallel for collapse(2)
    for (std::size_t pp = 0; pp < blocked_num_predict_points; pp += THREAD_BLOCK_SIZE_uz) {
        for (std::size_t sv = 0; sv < blocked_num_support_vectors; sv += THREAD_BLOCK_SIZE_uz) {
            // perform operations on the current block
            for (std::size_t pp_block = 0; pp_block < THREAD_BLOCK_SIZE_uz; ++pp_block) {
                for (std::size_t sv_block = 0; sv_block < THREAD_BLOCK_SIZE_uz; ++sv_block) {
                    // calculate the indices used in the current thread
                    const std::size_t pp_idx = (pp + pp_block) * INTERNAL_BLOCK_SIZE_uz;
                    const std::size_t sv_idx = (sv + sv_block) * INTERNAL_BLOCK_SIZE_uz;

                    // create a thread private array used for internal caching
                    std::array<std::array<real_type, INTERNAL_BLOCK_SIZE>, INTERNAL_BLOCK_SIZE> temp{};

                    // iterate over all features
                    for (std::size_t dim = 0; dim < num_features; ++dim) {
                        // perform the feature reduction calculation
                        for (unsigned internal_pd = 0; internal_pd < INTERNAL_BLOCK_SIZE; ++internal_pd) {
                            for (unsigned internal_sv = 0; internal_sv < INTERNAL_BLOCK_SIZE; ++internal_sv) {
                                const std::size_t global_pp_idx = pp_idx + static_cast<std::size_t>(internal_pd);
                                const std::size_t global_sv = sv_idx + static_cast<std::size_t>(internal_sv);

                                temp[internal_pd][internal_sv] += detail::feature_reduce<kernel>(support_vectors(global_sv, dim), predict_points(global_pp_idx, dim));
                            }
                        }
                    }

                    // update temp using the respective kernel function
                    for (unsigned internal_pd = 0; internal_pd < INTERNAL_BLOCK_SIZE; ++internal_pd) {
                        for (unsigned internal_sv = 0; internal_sv < INTERNAL_BLOCK_SIZE; ++internal_sv) {
                            temp[internal_pd][internal_sv] = detail::apply_kernel_function<kernel>(temp[internal_pd][internal_sv], kernel_function_parameter...);
                        }
                    }

                    // add results to prediction
                    for (std::size_t a = 0; a < num_classes; ++a) {
                        for (unsigned internal_pd = 0; internal_pd < INTERNAL_BLOCK_SIZE; ++internal_pd) {
                            for (unsigned internal_sv = 0; internal_sv < INTERNAL_BLOCK_SIZE; ++internal_sv) {
                                const std::size_t global_pp_idx = pp_idx + static_cast<std::size_t>(internal_pd);
                                const std::size_t global_sv = sv_idx + static_cast<std::size_t>(internal_sv);

                                // be sure to not perform out of bounds accesses
                                if (global_pp_idx < num_predict_points && global_sv < num_support_vectors) {
#pragma omp atomic
                                    prediction(global_pp_idx, a) += alpha(a, global_sv) * temp[internal_pd][internal_sv];
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
