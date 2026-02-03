/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines the functions used for prediction for the C-SVM using the CUDA backend.
 */

#ifndef PLSSVM_BACKENDS_CUDA_KERNEL_PREDICT_KERNEL_CUH_
#define PLSSVM_BACKENDS_CUDA_KERNEL_PREDICT_KERNEL_CUH_
#pragma once

#include "plssvm/backends/CUDA/kernel/detail/atomics.cuh"            // atomicAdd for double precision floating point numbers on older CUDA hardware
#include "plssvm/backends/CUDA/kernel/detail/reinterpret_array.cuh"  // plssvm::cuda::detail::reinterpret_array
#include "plssvm/backends/CUDA/kernel/kernel_functions.cuh"          // plssvm::cuda::detail::{feature_reduce, apply_kernel_function}
#include "plssvm/constants.hpp"                                      // plssvm::{real_type, THREAD_BLOCK_SIZE, INTERNAL_BLOCK_SIZE, PADDING_SIZE}
#include "plssvm/kernel_function_types.hpp"                          // plssvm::kernel_function_type

#include <cstddef>  // std::size_t

namespace plssvm::cuda::detail {

/**
 * @brief Calculate the `w` vector used to speedup the prediction using the linear kernel function.
 * @param[out] w the vector to speedup the linear prediction
 * @param[in] alpha the previously learned weights
 * @param[in] support_vectors the support vectors
 * @param[in] num_classes the number of classes
 * @param[in] num_sv the number of support vectors
 * @param[in] device_num_sv the number of support vectors the current device is responsible for
 * @param[in] device_sv_offset the first support vector (row in @p alpha) the current device is responsible for
 * @param[in] grid_x_offset the offset in x-dimension into the data points if more than one execution grid has to be used
 * @param[in] grid_y_offset the offset in y-dimension into the data points if more than one execution grid has to be used
 */
__global__ void device_kernel_w_linear(real_type *w, const real_type *alpha, const real_type *support_vectors, const std::size_t num_classes, const std::size_t num_sv, const std::size_t device_num_sv, const std::size_t device_sv_offset, const std::size_t grid_x_offset, const std::size_t grid_y_offset) {
    // cast all values to 64-bit std::size_t to prevent potential 32-bit overflows
    constexpr auto INTERNAL_BLOCK_SIZE_uz = static_cast<std::size_t>(INTERNAL_BLOCK_SIZE);
    constexpr auto THREAD_BLOCK_SIZE_uz = static_cast<std::size_t>(THREAD_BLOCK_SIZE);
    constexpr auto PADDING_SIZE_uz = static_cast<std::size_t>(PADDING_SIZE);

    const auto threadIdx_x = static_cast<std::size_t>(threadIdx.x);                // current thread in block x-dimension
    const auto threadIdx_y = static_cast<std::size_t>(threadIdx.y);                // current thread in block y-dimension
    const auto blockDim_x = static_cast<std::size_t>(blockDim.x);                  // number of threads in block x-dimension
    const auto blockDim_y = static_cast<std::size_t>(blockDim.y);                  // number of threads in block y-dimension
    const auto blockIdx_x = static_cast<std::size_t>(blockIdx.x) + grid_x_offset;  // current block in grid x-dimension + offsets if the grid size is too large
    const auto blockIdx_y = static_cast<std::size_t>(blockIdx.y) + grid_y_offset;  // current block in grid y-dimension + offsets if the grid size is too large

    // create two shared memory arrays used for caching
    __shared__ real_type feature_cache[THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];
    __shared__ real_type alpha_cache[THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];

    // create a thread private array used for internal caching
    real_type temp[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE]{};

    {
        // calculate the indices used in the current thread, pays attention to coalesced memory accesses
        const auto feature_idx_linear = blockIdx_x * blockDim_x * INTERNAL_BLOCK_SIZE_uz + threadIdx_x;  // num_features
        const auto class_idx_linear = blockIdx_y * blockDim_y * INTERNAL_BLOCK_SIZE_uz + threadIdx_x;    // num_classes

        // iterate over all support vectors using blocking to be able to cache them for faster memory accesses
        for (std::size_t sv_block = 0; sv_block < device_num_sv; sv_block += THREAD_BLOCK_SIZE_uz) {
            // load data into shared memory
            for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                // calculate the indices to access the global data, pays attention to coalesced memory accesses
                const auto global_feature_idx_linear = feature_idx_linear + static_cast<std::size_t>(internal) * THREAD_BLOCK_SIZE_uz;
                const auto global_class_idx_linear = class_idx_linear + static_cast<std::size_t>(internal) * THREAD_BLOCK_SIZE_uz;

                // store the values in the shared memory
                feature_cache[threadIdx.y][internal * THREAD_BLOCK_SIZE + threadIdx.x] = support_vectors[global_feature_idx_linear * (device_num_sv + PADDING_SIZE_uz) + sv_block + threadIdx_y];  // SoA
                alpha_cache[threadIdx.y][internal * THREAD_BLOCK_SIZE + threadIdx.x] = alpha[global_class_idx_linear * (num_sv + PADDING_SIZE_uz) + sv_block + device_sv_offset + threadIdx_y];    // AoS
            }
            __syncthreads();  // wait until all threads loaded their part of the data

            // perform the dot product calculation
            for (unsigned sv = 0; sv < THREAD_BLOCK_SIZE; ++sv) {
                for (unsigned internal_feature = 0; internal_feature < INTERNAL_BLOCK_SIZE; ++internal_feature) {
                    for (unsigned internal_class = 0; internal_class < INTERNAL_BLOCK_SIZE; ++internal_class) {
                        temp[internal_feature][internal_class] += alpha_cache[sv][threadIdx.y * INTERNAL_BLOCK_SIZE + internal_class] * feature_cache[sv][threadIdx.x * INTERNAL_BLOCK_SIZE + internal_feature];
                    }
                }
            }
            __syncthreads();  // wait until all threads performed their part of the calculations
        }
    }

    // calculate the indices used in the current thread
    const auto feature_idx = (blockIdx_x * blockDim_x + threadIdx_x) * INTERNAL_BLOCK_SIZE_uz;  // num_features
    const auto class_idx = (blockIdx_y * blockDim_y + threadIdx_y) * INTERNAL_BLOCK_SIZE_uz;    // num_classes

    // update the global w-vector with the locally cached values
    for (unsigned internal_feature = 0; internal_feature < INTERNAL_BLOCK_SIZE; ++internal_feature) {
        for (unsigned internal_class = 0; internal_class < INTERNAL_BLOCK_SIZE; ++internal_class) {
            // calculate the indices to access the global data
            const auto global_feature_idx = feature_idx + static_cast<std::size_t>(internal_feature);
            const auto global_class_idx = class_idx + static_cast<std::size_t>(internal_class);

            w[global_feature_idx * (num_classes + PADDING_SIZE_uz) + global_class_idx] = temp[internal_feature][internal_class];  // SoA
        }
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
    constexpr auto INTERNAL_BLOCK_SIZE_uz = static_cast<std::size_t>(INTERNAL_BLOCK_SIZE);
    constexpr auto THREAD_BLOCK_SIZE_uz = static_cast<std::size_t>(THREAD_BLOCK_SIZE);
    constexpr auto PADDING_SIZE_uz = static_cast<std::size_t>(PADDING_SIZE);

    const auto threadIdx_x = static_cast<std::size_t>(threadIdx.x);                // current thread in block x-dimension
    const auto threadIdx_y = static_cast<std::size_t>(threadIdx.y);                // current thread in block y-dimension
    const auto blockDim_x = static_cast<std::size_t>(blockDim.x);                  // number of threads in block x-dimension
    const auto blockDim_y = static_cast<std::size_t>(blockDim.y);                  // number of threads in block y-dimension
    const auto blockIdx_x = static_cast<std::size_t>(blockIdx.x) + grid_x_offset;  // current block in grid x-dimension + offsets if the grid size is too large
    const auto blockIdx_y = static_cast<std::size_t>(blockIdx.y) + grid_y_offset;  // current block in grid y-dimension + offsets if the grid size is too large

    // create two shared memory arrays used for caching
    __shared__ real_type pp_cache[THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];
    __shared__ real_type w_cache[THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];

    // create a thread private array used for internal caching
    real_type temp[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE]{};

    {
        // calculate the indices used in the current thread, pays attention to coalesced memory accesses
        const auto pp_idx_linear = blockIdx_x * blockDim_x * INTERNAL_BLOCK_SIZE_uz + threadIdx_x;     // num_predict_points
        const auto class_idx_linear = blockIdx_y * blockDim_y * INTERNAL_BLOCK_SIZE_uz + threadIdx_x;  // num_classes

        // iterate over all features using blocking to be able to cache them for faster memory accesses
        for (std::size_t feature_block = 0; feature_block < num_features; feature_block += THREAD_BLOCK_SIZE_uz) {
            // load data into shared memory
            for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                // calculate the indices to access the global data, pays attention to coalesced memory accesses
                const auto global_pp_idx_linear = pp_idx_linear + static_cast<std::size_t>(internal) * THREAD_BLOCK_SIZE_uz;
                const auto global_class_idx_linear = class_idx_linear + static_cast<std::size_t>(internal) * THREAD_BLOCK_SIZE_uz;

                // store the values in the shared memory
                pp_cache[threadIdx.y][internal * THREAD_BLOCK_SIZE + threadIdx.x] = predict_points[(feature_block + threadIdx_y) * (num_predict_points + PADDING_SIZE_uz) + global_pp_idx_linear];  // SoA
                w_cache[threadIdx.y][internal * THREAD_BLOCK_SIZE + threadIdx.x] = w[(feature_block + threadIdx_y) * (num_classes + PADDING_SIZE_uz) + global_class_idx_linear];                    // SoA
            }
            __syncthreads();  // wait until all threads loaded their part of the data

            // perform the dot product calculation
            for (unsigned feature = 0; feature < THREAD_BLOCK_SIZE; ++feature) {
                for (unsigned internal_pp = 0; internal_pp < INTERNAL_BLOCK_SIZE; ++internal_pp) {
                    for (unsigned internal_class = 0; internal_class < INTERNAL_BLOCK_SIZE; ++internal_class) {
                        temp[internal_pp][internal_class] += w_cache[feature][threadIdx.y * INTERNAL_BLOCK_SIZE + internal_class] * pp_cache[feature][threadIdx.x * INTERNAL_BLOCK_SIZE + internal_pp];
                    }
                }
            }
            __syncthreads();  // wait until all threads performed their part of the calculations
        }
    }

    // calculate the indices used in the current thread
    const auto pp_idx = (blockIdx_x * blockDim_x + threadIdx_x) * INTERNAL_BLOCK_SIZE_uz;     // num_predict_points
    const auto class_idx = (blockIdx_y * blockDim_y + threadIdx_y) * INTERNAL_BLOCK_SIZE_uz;  // num_classes

    // update the global array with the local one
    for (unsigned internal_pp = 0; internal_pp < INTERNAL_BLOCK_SIZE; ++internal_pp) {
        for (unsigned internal_class = 0; internal_class < INTERNAL_BLOCK_SIZE; ++internal_class) {
            // calculate the indices to access the global data
            const auto global_pp_idx = pp_idx + static_cast<std::size_t>(internal_pp);
            const auto global_class_idx = class_idx + static_cast<std::size_t>(internal_class);

            prediction[global_pp_idx * (num_classes + PADDING_SIZE_uz) + global_class_idx] = temp[internal_pp][internal_class] - rho[global_class_idx];  // AoS
        }
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
    constexpr auto INTERNAL_BLOCK_SIZE_uz = static_cast<std::size_t>(INTERNAL_BLOCK_SIZE);
    constexpr auto THREAD_BLOCK_SIZE_uz = static_cast<std::size_t>(THREAD_BLOCK_SIZE);
    constexpr auto PADDING_SIZE_uz = static_cast<std::size_t>(PADDING_SIZE);

    const auto threadIdx_x = static_cast<std::size_t>(threadIdx.x);                // current thread in block x-dimension
    const auto threadIdx_y = static_cast<std::size_t>(threadIdx.y);                // current thread in block y-dimension
    const auto blockDim_x = static_cast<std::size_t>(blockDim.x);                  // number of threads in block x-dimension
    const auto blockDim_y = static_cast<std::size_t>(blockDim.y);                  // number of threads in block y-dimension
    const auto blockIdx_x = static_cast<std::size_t>(blockIdx.x) + grid_x_offset;  // current block in grid x-dimension + offsets if the grid size is too large
    const auto blockIdx_y = static_cast<std::size_t>(blockIdx.y) + grid_y_offset;  // current block in grid y-dimension + offsets if the grid size is too large

    // create two shared memory arrays used for caching
    __shared__ real_type cache_one[THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];
    __shared__ real_type cache_two[THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];

    // create a thread private array used for internal caching
    real_type temp[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE]{};

    {
        // reinterpret the shared memory arrays to be of shape [THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE]
        auto *pp_cache = reinterpret_array<INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE>(cache_one);
        auto *sv_cache = reinterpret_array<INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE>(cache_two);

        // calculate the indices used in the current thread, pays attention to coalesced memory accesses
        const auto pp_idx_linear = blockIdx_x * blockDim_x * INTERNAL_BLOCK_SIZE_uz + threadIdx_x;  // num_predict_points
        const auto sv_idx_linear = blockIdx_y * blockDim_y * INTERNAL_BLOCK_SIZE_uz + threadIdx_x;  // num_support_vectors

        // iterate over all features using blocking to be able to cache them for faster memory accesses
        for (std::size_t feature_block = 0; feature_block < num_features; feature_block += THREAD_BLOCK_SIZE_uz) {
            // load data into shared memory
            for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                // calculate the indices to access the global data, pays attention to coalesced memory accesses
                const auto global_pp_idx_linear = pp_idx_linear + static_cast<std::size_t>(internal) * THREAD_BLOCK_SIZE_uz;
                const auto global_sv_idx_linear = sv_idx_linear + static_cast<std::size_t>(internal) * THREAD_BLOCK_SIZE_uz;

                // store the values in the shared memory
                pp_cache[threadIdx.y][internal * THREAD_BLOCK_SIZE + threadIdx.x] = predict_points[(feature_block + threadIdx_y) * (num_predict_points + PADDING_SIZE_uz) + global_pp_idx_linear];  // SoA
                sv_cache[threadIdx.y][internal * THREAD_BLOCK_SIZE + threadIdx.x] = support_vectors[(feature_block + threadIdx_y) * (num_sv + PADDING_SIZE_uz) + global_sv_idx_linear];             // SoA
            }
            __syncthreads();  // wait until all threads loaded their part of the data

            // perform the feature reduction calculation
            for (unsigned feature = 0; feature < THREAD_BLOCK_SIZE; ++feature) {
                for (unsigned internal_pp = 0; internal_pp < INTERNAL_BLOCK_SIZE; ++internal_pp) {
                    for (unsigned internal_sv = 0; internal_sv < INTERNAL_BLOCK_SIZE; ++internal_sv) {
                        temp[internal_pp][internal_sv] += detail::feature_reduce<kernel_function>(sv_cache[feature][threadIdx.y * INTERNAL_BLOCK_SIZE + internal_sv],
                                                                                                  pp_cache[feature][threadIdx.x * INTERNAL_BLOCK_SIZE + internal_pp]);
                    }
                }
            }
            __syncthreads();  // wait until all threads performed their part of the calculations
        }
    }

    // update temp using the respective kernel function
    for (unsigned internal_pp = 0; internal_pp < INTERNAL_BLOCK_SIZE; ++internal_pp) {  // NOLINT(modernize-loop-convert): false positive range-based for loop modernize
        for (unsigned internal_sv = 0; internal_sv < INTERNAL_BLOCK_SIZE; ++internal_sv) {
            temp[internal_pp][internal_sv] = detail::apply_kernel_function<kernel_function>(temp[internal_pp][internal_sv], kernel_function_parameter...);
        }
    }

    {
        // reinterpret the shared memory arrays to be of shape [THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE]
        auto *alpha_cache = reinterpret_array<INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE>(cache_one);
        auto *out_cache = reinterpret_array<INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE>(cache_two);

        // calculate the indices used in the current thread
        const auto pp_idx = (blockIdx_x * blockDim_x + threadIdx_x) * INTERNAL_BLOCK_SIZE_uz;  // num_predict_points
        // calculate the indices used in the current thread, pays attention to coalesced memory accesses
        const auto sv_idx_linear = blockIdx_y * blockDim_y * INTERNAL_BLOCK_SIZE_uz + threadIdx_x;  // num_support_vectors

        // iterate over all classes using blocking to be able to cache them for faster memory accesses
        for (std::size_t class_block = 0; class_block < num_classes; class_block += THREAD_BLOCK_SIZE_uz) {
            // load data into shared memory
            for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                // calculate the indices to access the global data, pays attention to coalesced memory accesses
                const auto global_sv_idx_linear = sv_idx_linear + static_cast<std::size_t>(internal) * THREAD_BLOCK_SIZE_uz;

                // store the values in the shared memory
                alpha_cache[threadIdx.y][internal * THREAD_BLOCK_SIZE + threadIdx.x] = alpha[(class_block + threadIdx_y) * (num_sv + PADDING_SIZE_uz) + global_sv_idx_linear];  // AoS
                // the bias (rho) must only be applied once for all support vectors
                if (blockIdx_y == std::size_t{ 0 }) {
                    out_cache[threadIdx.y][internal * THREAD_BLOCK_SIZE + threadIdx.x] = -rho[class_block + threadIdx_y];
                } else {
                    out_cache[threadIdx.y][internal * THREAD_BLOCK_SIZE + threadIdx.x] = real_type{ 0.0 };
                }
            }
            __syncthreads();  // wait until all threads loaded their part of the data

            // calculate intermediate results and store them in shared memory
            for (unsigned class_idx = 0; class_idx < THREAD_BLOCK_SIZE; ++class_idx) {
                for (unsigned internal_pp = 0; internal_pp < INTERNAL_BLOCK_SIZE; ++internal_pp) {
                    for (unsigned internal_sv = 0; internal_sv < INTERNAL_BLOCK_SIZE; ++internal_sv) {
                        out_cache[(class_idx + threadIdx.y) % THREAD_BLOCK_SIZE][internal_pp * THREAD_BLOCK_SIZE + threadIdx.x] +=
                            temp[internal_pp][internal_sv] * alpha_cache[(class_idx + threadIdx.y) % THREAD_BLOCK_SIZE][threadIdx.y * INTERNAL_BLOCK_SIZE + internal_sv];
                    }
                }
                __syncthreads();  // wait until all threads performed their part of the calculations
            }

            // atomically add the intermediate cached results to the prediction
            for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                // calculate the indices to access the global data
                const auto global_pp_idx = pp_idx + static_cast<std::size_t>(internal);

                atomicAdd(&prediction[global_pp_idx * (num_classes + PADDING_SIZE_uz) + class_block + threadIdx_y], out_cache[threadIdx.y][internal * THREAD_BLOCK_SIZE + threadIdx.x]);
            }
            __syncthreads();  // wait until all threads updated their part of the prediction
        }
    }
}

}  // namespace plssvm::cuda::detail

#endif  // PLSSVM_BACKENDS_CUDA_KERNEL_PREDICT_KERNEL_CUH_
