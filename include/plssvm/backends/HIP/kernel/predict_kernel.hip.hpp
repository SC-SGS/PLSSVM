/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines the functions used for prediction for the C-SVM using the HIP backend.
 */

#ifndef PLSSVM_BACKENDS_HIP_KERNEL_PREDICT_KERNEL_HIP_HPP_
#define PLSSVM_BACKENDS_HIP_KERNEL_PREDICT_KERNEL_HIP_HPP_
#pragma once

#include "plssvm/backends/HIP/kernel/detail/reinterpret_array.hip.hpp"  // plssvm::hip::detail::reinterpret_array
#include "plssvm/backends/HIP/kernel/kernel_functions.hip.hpp"          // plssvm::hip::detail::{feature_reduce, apply_kernel_function}
#include "plssvm/constants.hpp"                                         // plssvm::real_type
#include "plssvm/kernel_function_types.hpp"                             // plssvm::kernel_function_type

#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"

#include <cstddef>  // std::size_t

namespace plssvm::hip::detail {

/**
 * @brief Calculate the `w` vector used to speedup the prediction using the linear kernel function.
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
__global__ void device_kernel_w_linear(real_type *w, const real_type *alpha, const real_type *support_vectors, const std::size_t num_features, const std::size_t num_classes, const std::size_t num_sv, const std::size_t device_num_sv, const std::size_t device_sv_offset, const std::size_t grid_x_offset, const std::size_t grid_y_offset) {
    // cast all values to 64-bit std::size_t to prevent potential 32-bit overflows
    const auto threadIdx_x = static_cast<std::size_t>(threadIdx.x);                // current thread in block x-dimension
    const auto threadIdx_y = static_cast<std::size_t>(threadIdx.y);                // current thread in block y-dimension
    const auto blockDim_x = static_cast<std::size_t>(blockDim.x);                  // number of threads in block x-dimension
    const auto blockDim_y = static_cast<std::size_t>(blockDim.y);                  // number of threads in block y-dimension
    const auto blockIdx_x = static_cast<std::size_t>(blockIdx.x) + grid_x_offset;  // current block in grid x-dimension + offsets if the grid size is too large
    const auto blockIdx_y = static_cast<std::size_t>(blockIdx.y) + grid_y_offset;  // current block in grid y-dimension + offsets if the grid size is too large

    // calculate the indices used in the current thread
    const auto global_feature_idx = blockIdx_x * blockDim_x + threadIdx_x;  // num_features
    const auto global_class_idx = blockIdx_y * blockDim_y + threadIdx_y;    // num_classes

    // be sure to not perform out-of-bounds accesses
    if (global_feature_idx < num_features && global_class_idx < num_classes) {
        real_type temp{ 0.0 };

        // perform the dot product calculation
        for (std::size_t sv = 0; sv < device_num_sv; ++sv) {
            temp += alpha[global_class_idx * num_sv + sv + device_sv_offset] *  // AoS
                    support_vectors[global_feature_idx * device_num_sv + sv];   // SoA
        }

        w[global_feature_idx * num_classes + global_class_idx] = temp;  // SoA
    }
}

/**
 * @brief Predict the @p predict_points using the linear kernel speeding up the calculation using the @p w vector.
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
__global__ void device_kernel_predict_linear(real_type *prediction, const real_type *w, const real_type *rho, const real_type *predict_points, const std::size_t num_classes, const std::size_t num_predict_points, const std::size_t num_features, const std::size_t grid_x_offset, const std::size_t grid_y_offset) {
    // cast all values to 64-bit std::size_t to prevent potential 32-bit overflows
    const auto threadIdx_x = static_cast<std::size_t>(threadIdx.x);                // current thread in block x-dimension
    const auto threadIdx_y = static_cast<std::size_t>(threadIdx.y);                // current thread in block y-dimension
    const auto blockDim_x = static_cast<std::size_t>(blockDim.x);                  // number of threads in block x-dimension
    const auto blockDim_y = static_cast<std::size_t>(blockDim.y);                  // number of threads in block y-dimension
    const auto blockIdx_x = static_cast<std::size_t>(blockIdx.x) + grid_x_offset;  // current block in grid x-dimension + offsets if the grid size is too large
    const auto blockIdx_y = static_cast<std::size_t>(blockIdx.y) + grid_y_offset;  // current block in grid y-dimension + offsets if the grid size is too large

    // calculate the indices used in the current thread
    const auto global_pp_idx = blockIdx_x * blockDim_x + threadIdx_x;     // num_predict_points
    const auto global_class_idx = blockIdx_y * blockDim_y + threadIdx_y;  // num_classes

    // be sure to not perform out-of-bounds accesses
    if (global_pp_idx < num_predict_points && global_class_idx < num_classes) {
        real_type temp{ 0.0 };

        // perform the dot product calculation
        for (std::size_t feature = 0; feature < num_features; ++feature) {
            temp += w[feature * num_classes + global_class_idx] *                  // SoA
                    predict_points[feature * num_predict_points + global_pp_idx];  // SoA
        }

        prediction[global_pp_idx * num_classes + global_class_idx] = temp - rho[global_class_idx];
    }
}

/**
 * @brief Predict the @p predict_points using the @p kernel_function.
 * @tparam kernel_function the type of the used kernel function
 * @tparam Args the types of the parameters necessary for the specific kernel function
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
template <kernel_function_type kernel_function, typename... Args>
__global__ void device_kernel_predict(real_type *prediction, const real_type *alpha, const real_type *rho, const real_type *support_vectors, const real_type *predict_points, const std::size_t num_classes, const std::size_t num_sv, const std::size_t num_predict_points, const std::size_t num_features, const std::size_t grid_x_offset, const std::size_t grid_y_offset, Args... kernel_function_parameter) {
    // cast all values to 64-bit std::size_t to prevent potential 32-bit overflows
    const auto threadIdx_x = static_cast<std::size_t>(threadIdx.x);                // current thread in block x-dimension
    const auto threadIdx_y = static_cast<std::size_t>(threadIdx.y);                // current thread in block y-dimension
    const auto blockDim_x = static_cast<std::size_t>(blockDim.x);                  // number of threads in block x-dimension
    const auto blockDim_y = static_cast<std::size_t>(blockDim.y);                  // number of threads in block y-dimension
    const auto blockIdx_x = static_cast<std::size_t>(blockIdx.x) + grid_x_offset;  // current block in grid x-dimension + offsets if the grid size is too large
    const auto blockIdx_y = static_cast<std::size_t>(blockIdx.y) + grid_y_offset;  // current block in grid y-dimension + offsets if the grid size is too large

    // calculate the indices used in the current thread
    const auto global_pp_idx = blockIdx_x * blockDim_x + threadIdx_x;  // num_predict_points
    const auto global_sv_idx = blockIdx_y * blockDim_y + threadIdx_y;  // num_support_vectors

    // be sure to not perform out-of-bounds accesses
    if (global_sv_idx < num_sv && global_pp_idx < num_predict_points) {
        real_type temp{ 0.0 };

        // perform the feature reduction calculation
        for (std::size_t feature = 0; feature < num_features; ++feature) {
            temp += detail::feature_reduce<kernel_function>(support_vectors[feature * num_sv + global_sv_idx],              // SoA
                                                            predict_points[feature * num_predict_points + global_pp_idx]);  // SoA
        }

        // update temp using the respective kernel function
        temp = detail::apply_kernel_function<kernel_function>(temp, kernel_function_parameter...);

        // iterate over all classes
        for (std::size_t class_idx = 0; class_idx < num_classes; ++class_idx) {
            real_type out_cache = alpha[class_idx * num_sv + global_sv_idx] * temp;  // AoS

            // the bias (rho) must only be applied once for all support vectors
            if (global_sv_idx == std::size_t{ 0 }) {
                out_cache -= rho[class_idx];
            }

            atomicAdd(&prediction[global_pp_idx * num_classes + class_idx], out_cache);  // AoS
        }
    }
}

}  // namespace plssvm::hip::detail

#endif  // PLSSVM_BACKENDS_HIP_KERNEL_PREDICT_KERNEL_HIP_HPP_
