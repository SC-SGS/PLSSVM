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

#include "plssvm/backends/HIP/kernel/kernel_functions.hip.hpp"  // plssvm::hip::detail::{feature_reduce, apply_kernel_function}
#include "plssvm/constants.hpp"                                 // plssvm::real_type, plssvm::THREAD_BLOCK_SIZE
#include "plssvm/kernel_function_types.hpp"                     // plssvm::kernel_function_type

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
    constexpr auto THREAD_BLOCK_SIZE_uz = static_cast<std::size_t>(THREAD_BLOCK_SIZE);

    const auto threadIdx_x = static_cast<std::size_t>(threadIdx.x);                // current thread in block x-dimension
    const auto threadIdx_y = static_cast<std::size_t>(threadIdx.y);                // current thread in block y-dimension
    const auto blockDim_x = static_cast<std::size_t>(blockDim.x);                  // number of threads in block x-dimension
    const auto blockDim_y = static_cast<std::size_t>(blockDim.y);                  // number of threads in block y-dimension
    const auto blockIdx_x = static_cast<std::size_t>(blockIdx.x) + grid_x_offset;  // current block in grid x-dimension + offsets if the grid size is too large
    const auto blockIdx_y = static_cast<std::size_t>(blockIdx.y) + grid_y_offset;  // current block in grid y-dimension + offsets if the grid size is too large

    // create two shared memory arrays used for caching
    __shared__ real_type feature_cache[THREAD_BLOCK_SIZE][THREAD_BLOCK_SIZE];
    __shared__ real_type alpha_cache[THREAD_BLOCK_SIZE][THREAD_BLOCK_SIZE];

    real_type temp{ 0.0 };

    {
        // calculate the indices used in the current thread, pays attention to coalesced memory accesses
        const auto global_feature_idx_linear = blockIdx_x * blockDim_x + threadIdx_x;  // num_features
        const auto global_class_idx_linear = blockIdx_y * blockDim_y + threadIdx_x;    // num_classes

        // iterate over all support vectors using blocking to be able to cache them for faster memory accesses
        for (std::size_t sv_block = 0; sv_block < device_num_sv; sv_block += THREAD_BLOCK_SIZE_uz) {
            // zero-out shared memory
            feature_cache[threadIdx.y][threadIdx.x] = real_type{ 0.0 };
            alpha_cache[threadIdx.y][threadIdx.x] = real_type{ 0.0 };

            // load data into shared memory
            if (sv_block + threadIdx_y < device_num_sv) {
                if (global_feature_idx_linear < num_features) {
                    feature_cache[threadIdx.y][threadIdx.x] = support_vectors[global_feature_idx_linear * device_num_sv + sv_block + threadIdx_y];  // SoA
                }
                if (global_class_idx_linear < num_classes) {
                    alpha_cache[threadIdx.y][threadIdx.x] = alpha[global_class_idx_linear * num_sv + sv_block + device_sv_offset + threadIdx_y];  // AoS
                }
            }
            __syncthreads();  // wait until all threads loaded their part of the data

            // perform the dot product calculation
            for (unsigned sv = 0; sv < THREAD_BLOCK_SIZE; ++sv) {
                temp += alpha_cache[sv][threadIdx.y] * feature_cache[sv][threadIdx.x];
            }
            __syncthreads();  // wait until all threads performed their part of the calculations
        }
    }

    // calculate the indices used in the current thread
    const auto global_feature_idx = blockIdx_x * blockDim_x + threadIdx_x;  // num_features
    const auto global_class_idx = blockIdx_y * blockDim_y + threadIdx_y;    // num_classes

    // be sure to not perform out-of-bounds accesses
    if (global_feature_idx < num_features && global_class_idx < num_classes) {
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
    constexpr auto THREAD_BLOCK_SIZE_uz = static_cast<std::size_t>(THREAD_BLOCK_SIZE);

    const auto threadIdx_x = static_cast<std::size_t>(threadIdx.x);                // current thread in block x-dimension
    const auto threadIdx_y = static_cast<std::size_t>(threadIdx.y);                // current thread in block y-dimension
    const auto blockDim_x = static_cast<std::size_t>(blockDim.x);                  // number of threads in block x-dimension
    const auto blockDim_y = static_cast<std::size_t>(blockDim.y);                  // number of threads in block y-dimension
    const auto blockIdx_x = static_cast<std::size_t>(blockIdx.x) + grid_x_offset;  // current block in grid x-dimension + offsets if the grid size is too large
    const auto blockIdx_y = static_cast<std::size_t>(blockIdx.y) + grid_y_offset;  // current block in grid y-dimension + offsets if the grid size is too large

    // create two shared memory arrays used for caching
    __shared__ real_type pp_cache[THREAD_BLOCK_SIZE][THREAD_BLOCK_SIZE];
    __shared__ real_type w_cache[THREAD_BLOCK_SIZE][THREAD_BLOCK_SIZE];

    real_type temp{ 0.0 };

    {
        // calculate the indices used in the current thread, pays attention to coalesced memory accesses
        const auto global_pp_idx_linear = blockIdx_x * blockDim_x + threadIdx_x;     // num_predict_points
        const auto global_class_idx_linear = blockIdx_y * blockDim_y + threadIdx_x;  // num_classes

        // iterate over all features using blocking to be able to cache them for faster memory accesses
        for (std::size_t feature_block = 0; feature_block < num_features; feature_block += THREAD_BLOCK_SIZE_uz) {
            // zero-out shared memory
            pp_cache[threadIdx.y][threadIdx.x] = real_type{ 0.0 };
            w_cache[threadIdx.y][threadIdx.x] = real_type{ 0.0 };

            // load data into shared memory
            if (feature_block + threadIdx_y < num_features) {
                if (global_pp_idx_linear < num_predict_points) {
                    pp_cache[threadIdx.y][threadIdx.x] = predict_points[(feature_block + threadIdx_y) * num_predict_points + global_pp_idx_linear];  // SoA
                }
                if (global_class_idx_linear < num_classes) {
                    w_cache[threadIdx.y][threadIdx.x] = w[(feature_block + threadIdx_y) * num_classes + global_class_idx_linear];  // SoA
                }
            }
            __syncthreads();  // wait until all threads loaded their part of the data

            // perform the dot product calculation
            for (unsigned feature = 0; feature < THREAD_BLOCK_SIZE; ++feature) {
                temp += w_cache[feature][threadIdx.y] * pp_cache[feature][threadIdx.x];
            }
            __syncthreads();  // wait until all threads performed their part of the calculations
        }
    }

    // calculate the indices used in the current thread
    const auto global_pp_idx = blockIdx_x * blockDim_x + threadIdx_x;     // num_predict_points
    const auto global_class_idx = blockIdx_y * blockDim_y + threadIdx_y;  // num_classes

    // be sure to not perform out-of-bounds accesses
    if (global_pp_idx < num_predict_points && global_class_idx < num_classes) {
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
    constexpr auto THREAD_BLOCK_SIZE_uz = static_cast<std::size_t>(THREAD_BLOCK_SIZE);

    const auto threadIdx_x = static_cast<std::size_t>(threadIdx.x);                // current thread in block x-dimension
    const auto threadIdx_y = static_cast<std::size_t>(threadIdx.y);                // current thread in block y-dimension
    const auto blockDim_x = static_cast<std::size_t>(blockDim.x);                  // number of threads in block x-dimension
    const auto blockDim_y = static_cast<std::size_t>(blockDim.y);                  // number of threads in block y-dimension
    const auto blockIdx_x = static_cast<std::size_t>(blockIdx.x) + grid_x_offset;  // current block in grid x-dimension + offsets if the grid size is too large
    const auto blockIdx_y = static_cast<std::size_t>(blockIdx.y) + grid_y_offset;  // current block in grid y-dimension + offsets if the grid size is too large

    // create two shared memory arrays used for caching
    __shared__ real_type cache_one[THREAD_BLOCK_SIZE][THREAD_BLOCK_SIZE];
    __shared__ real_type cache_two[THREAD_BLOCK_SIZE][THREAD_BLOCK_SIZE];

    real_type temp{ 0.0 };

    {
        // rename the shared memory arrays
        auto *pp_cache = cache_one;
        auto *sv_cache = cache_two;

        // calculate the indices used in the current thread, pays attention to coalesced memory accesses
        const auto global_pp_idx_linear = blockIdx_x * blockDim_x + threadIdx_x;  // num_predict_points
        const auto global_sv_idx_linear = blockIdx_y * blockDim_y + threadIdx_x;  // num_support_vectors

        // iterate over all features using blocking to be able to cache them for faster memory accesses
        for (std::size_t feature_block = 0; feature_block < num_features; feature_block += THREAD_BLOCK_SIZE_uz) {
            // zero-out shared memory
            pp_cache[threadIdx.y][threadIdx.x] = real_type{ 0.0 };
            sv_cache[threadIdx.y][threadIdx.x] = real_type{ 0.0 };

            // load data into shared memory
            if (feature_block + threadIdx_y < num_features) {
                if (global_pp_idx_linear < num_predict_points) {
                    pp_cache[threadIdx.y][threadIdx.x] = predict_points[(feature_block + threadIdx_y) * num_predict_points + global_pp_idx_linear];  // SoA
                }
                if (global_sv_idx_linear < num_sv) {
                    sv_cache[threadIdx.y][threadIdx.x] = support_vectors[(feature_block + threadIdx_y) * num_sv + global_sv_idx_linear];  // SoA
                }
            }
            __syncthreads();  // wait until all threads loaded their part of the data

            // perform the feature reduction calculation
            for (unsigned feature = 0; feature < THREAD_BLOCK_SIZE; ++feature) {
                temp += detail::feature_reduce<kernel_function>(sv_cache[feature][threadIdx.y],
                                                                pp_cache[feature][threadIdx.x]);
            }
            __syncthreads();  // wait until all threads performed their part of the calculations
        }
    }

    // update temp using the respective kernel function
    temp = detail::apply_kernel_function<kernel_function>(temp, kernel_function_parameter...);

    {
        // rename the shared memory arrays
        auto *alpha_cache = cache_one;
        auto *out_cache = cache_two;

        // calculate the indices used in the current thread
        const auto global_pp_idx = blockIdx_x * blockDim_x + threadIdx_x;  // num_predict_points
        // calculate the indices used in the current thread, pays attention to coalesced memory accesses
        const auto global_sv_idx_linear = blockIdx_y * blockDim_y + threadIdx_x;  // num_support_vectors

        // iterate over all classes using blocking to be able to cache them for faster memory accesses
        for (std::size_t class_block = 0; class_block < num_classes; class_block += THREAD_BLOCK_SIZE_uz) {
            // zero-out shared memory
            alpha_cache[threadIdx.y][threadIdx.x] = real_type{ 0.0 };
            out_cache[threadIdx.y][threadIdx.x] = real_type{ 0.0 };

            // load data into shared memory
            if (class_block + threadIdx_y < num_classes) {
                if (global_sv_idx_linear < num_sv) {
                    alpha_cache[threadIdx.y][threadIdx.x] = alpha[(class_block + threadIdx_y) * num_sv + global_sv_idx_linear];  // AoS
                }
                // the bias (rho) must only be applied once for all support vectors
                if (blockIdx_y == std::size_t{ 0 }) {
                    out_cache[threadIdx.y][threadIdx.x] = -rho[class_block + threadIdx_y];
                }
            }
            __syncthreads();  // wait until all threads loaded their part of the data

            // calculate intermediate results and store them in shared memory
            for (unsigned class_idx = 0; class_idx < THREAD_BLOCK_SIZE; ++class_idx) {
                out_cache[(class_idx + threadIdx.y) % THREAD_BLOCK_SIZE][threadIdx.x] += temp * alpha_cache[(class_idx + threadIdx.y) % THREAD_BLOCK_SIZE][threadIdx.y];
                __syncthreads();  // wait until all threads performed their part of the calculations
            }

            // atomically add the intermediate cached results to the prediction
            if (class_block + threadIdx_y < num_classes && global_pp_idx < num_predict_points) {
                atomicAdd(&prediction[global_pp_idx * num_classes + class_block + threadIdx_y], out_cache[threadIdx.y][threadIdx.x]);  // AoS
            }
            __syncthreads();  // wait until all threads updated their part of the prediction
        }
    }
}

}  // namespace plssvm::hip::detail

#endif  // PLSSVM_BACKENDS_HIP_KERNEL_PREDICT_KERNEL_HIP_HPP_
