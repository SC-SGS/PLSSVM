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

#include "plssvm/constants.hpp"              // plssvm::real_type
#include "plssvm/detail/assert.hpp"          // PLSSVM_ASSERT
#include "plssvm/kernel_function_types.hpp"  // plssvm::kernel_function_type
#include "plssvm/kernel_functions.hpp"       // plssvm::kernel_function
#include "plssvm/matrix.hpp"                 // plssvm::aos_matrix, plssvm::matrix
#include "plssvm/shape.hpp"                  // plssvm::shape

#include <cmath>    // std::fma
#include <cstddef>  // std::size_t
#include <ranges>
#include <execution>
#include <vector>  // std::vector

namespace plssvm::stdpar::detail {

/**
 * @brief Calculate the `w` vector used to speedup the prediction using the linear kernel function.
 * @tparam layout the compile-time layout type for the support vectors
 * @param[out] w the vector to speedup the linear prediction
 * @param[in] alpha the previously learned weights
 * @param[in] support_vectors the support vectors
 */
template <layout_type layout>
inline void device_kernel_w_linear(soa_matrix<real_type> &w, const aos_matrix<real_type> &alpha, const matrix<real_type, layout> &support_vectors) {
    PLSSVM_ASSERT(alpha.num_cols() == support_vectors.num_rows(), "Size mismatch: {} vs {}!", alpha.num_cols(), support_vectors.num_rows());
    PLSSVM_ASSERT(w.shape() == (plssvm::shape{ alpha.num_rows(), support_vectors.num_cols() }), "Shape mismatch: {} vs {}!", w.shape(), (plssvm::shape{ alpha.num_rows(), support_vectors.num_cols() }));

    const std::size_t num_classes = alpha.num_rows();
    const std::size_t num_support_vectors = support_vectors.num_rows();
    const std::size_t num_features = support_vectors.num_cols();

    const auto is = std::views::cartesian_product(
        std::views::iota(std::size_t{ 0 }, num_classes),
        std::views::iota(std::size_t{ 0 }, num_features));

    std::for_each(std::execution::par_unseq, is.begin(), is.end(), [&](auto i) {
        const auto [a, dim] = i;

        real_type temp{ 0.0 };
        for (std::size_t idx = 0; idx < num_support_vectors; ++idx) {
            temp = std::fma(alpha(a, idx), support_vectors(idx, dim), temp);
        }
        w(a, dim) = temp;
    });
}

/**
 * @brief Predict the @p predict_points_d using the linear kernel speeding up the calculation using the @p w_d vector.
 * @tparam layout the compile-time layout type for the predict points
 * @param[out] out the predicted values
 * @param[in] w the vector to speedup the calculations
 * @param[in] rho the previously learned bias
 * @param[in] predict_points the data points to predict
 */
template <layout_type layout>
inline void device_kernel_predict_linear(aos_matrix<real_type> &out, const soa_matrix<real_type> &w, const std::vector<real_type> &rho, const matrix<real_type, layout> &predict_points) {
    PLSSVM_ASSERT(w.num_rows() == rho.size(), "Size mismatch: {} vs {}!", w.num_rows(), rho.size());
    PLSSVM_ASSERT(w.num_cols() == predict_points.num_cols(), "Size mismatch: {} vs {}!", w.num_cols(), predict_points.num_cols());
    PLSSVM_ASSERT(out.shape() == (plssvm::shape{ predict_points.num_rows(), w.num_rows() }), "Shape mismatch: {} vs {}!", out.shape(), (plssvm::shape{ predict_points.num_rows(), w.num_rows() }));

    const std::size_t num_classes = out.num_cols();
    const std::size_t num_predict_points = predict_points.num_rows();
    const std::size_t num_features = predict_points.num_cols();

    const auto is = std::views::cartesian_product(
        std::views::iota(std::size_t{ 0 }, num_predict_points),
        std::views::iota(std::size_t{ 0 }, num_classes));

    std::for_each(std::execution::par_unseq, is.begin(), is.end(), [&](auto i) {
        const auto [point_index, a] = i;

        real_type temp{ 0.0 };
        for (std::size_t dim = 0; dim < num_features; ++dim) {
            temp = std::fma(w(a, dim), predict_points(point_index, dim), temp);
        }
        out(point_index, a) = temp - rho[a];
    });
}

/**
 * @brief Predict the @p predict_points_d using the @p kernel_function.
 * @tparam kernel_function the type of the used kernel function
 * @tparam layout the compile-time layout type for the matrices
 * @tparam Args the types of the parameters necessary for the specific kernel function
 * @param[out] out the predicted values
 * @param[in] alpha the previously learned weights
 * @param[in] rho the previously learned bias
 * @param[in] support_vectors the support vectors
 * @param[in] predict_points the data points to predict
 * @param[in] kernel_function_parameter the parameters necessary to apply the @p kernel_function
 */
template <kernel_function_type kernel_function, layout_type layout, typename... Args>
inline void device_kernel_predict(aos_matrix<real_type> &out, const aos_matrix<real_type> &alpha, const std::vector<real_type> &rho, const matrix<real_type, layout> &support_vectors, const matrix<real_type, layout> &predict_points, Args... kernel_function_parameter) {
    PLSSVM_ASSERT(alpha.num_rows() == rho.size(), "Size mismatch: {} vs {}!", alpha.num_rows(), rho.size());
    PLSSVM_ASSERT(alpha.num_cols() == support_vectors.num_rows(), "Size mismatch: {} vs {}!", alpha.num_cols(), support_vectors.num_rows());
    PLSSVM_ASSERT(support_vectors.num_cols() == predict_points.num_cols(), "Size mismatch: {} vs {}!", support_vectors.num_cols(), predict_points.num_cols());
    PLSSVM_ASSERT(out.shape() == (plssvm::shape{ predict_points.num_rows(), alpha.num_rows() }), "Shape mismatch: {} vs {}!", out.shape(), (plssvm::shape{ predict_points.num_rows(), alpha.num_rows() }));

    const std::size_t num_classes = alpha.num_rows();
    const std::size_t num_support_vectors = support_vectors.num_rows();
    const std::size_t num_predict_points = predict_points.num_rows();

    const auto is = std::views::iota(std::size_t{ 0 }, num_predict_points);

    std::for_each(std::execution::par_unseq, is.begin(), is.end(), [&](const auto point_index) {
        for (std::size_t a = 0; a < num_classes; ++a) {
            out(point_index, a) -= rho[a];
        }
        for (std::size_t sv_index = 0; sv_index < num_support_vectors; ++sv_index) {
            const real_type kernel_func = ::plssvm::kernel_function<kernel_function>(support_vectors, sv_index, predict_points, point_index, kernel_function_parameter...);
            for (std::size_t a = 0; a < num_classes; ++a) {
                out(point_index, a) = std::fma(alpha(a, sv_index), kernel_func, out(point_index, a));
            }
        }
    });
}

}  // namespace plssvm::stdpar::detail

#endif  // PLSSVM_BACKENDS_STDPAR_KERNEL_PREDICT_KERNEL_HPP_
