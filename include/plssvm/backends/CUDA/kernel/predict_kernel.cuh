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

#ifndef PLSSVM_BACKENDS_CUDA_KERNEL_PREDICT_KERNEL_HPP_
#define PLSSVM_BACKENDS_CUDA_KERNEL_PREDICT_KERNEL_HPP_
#pragma once

#include "plssvm/backends/CUDA/kernel/detail/atomics.cuh"    // atomicAdd for double precision floating point numbers on older CUDA hardware
#include "plssvm/backends/CUDA/kernel/kernel_functions.cuh"  // plssvm::cuda::detail::{feature_reduce, apply_kernel_function}
#include "plssvm/constants.hpp"                              // plssvm::{real_type, THREAD_BLOCK_SIZE, INTERNAL_BLOCK_SIZE, FEATURE_BLOCK_SIZE, PADDING_SIZE}
#include "plssvm/kernel_function_types.hpp"                  // plssvm::kernel_function_type

namespace plssvm::cuda::detail {

/**
 * @brief Calculate the `w` vector used to speedup the prediction using the linear kernel function.
 * @param[out] w_d the vector to speedup the linear prediction
 * @param[in] alpha_d the previously learned weights
 * @param[in] sv_d the support vectors
 * @param[in] num_classes the number of classes
 * @param[in] num_sv the number of support vectors
 * @param[in] device_specific_num_sv the number of support vectors the current device is responsible for
 * @param[in] sv_offset the first support vector (row in @p alpha_d) the current device is responsible for
 */
__global__ void device_kernel_w_linear(real_type *w_d, const real_type *alpha_d, const real_type *sv_d, const unsigned long long num_classes, const unsigned long long num_sv, const unsigned long long device_specific_num_sv, const unsigned long long sv_offset, const unsigned long long grid_x_offset, const unsigned long long grid_y_offset) {
    // cast all values to 64-bit unsigned long long to prevent potential 32-bit overflows
    const auto threadIdx_x = static_cast<unsigned long long>(threadIdx.x);                // current thread in block x-dimension
    const auto threadIdx_y = static_cast<unsigned long long>(threadIdx.y);                // current thread in block y-dimension
    const auto blockDim_x = static_cast<unsigned long long>(blockDim.x);                  // number of threads in block x-dimension
    const auto blockDim_y = static_cast<unsigned long long>(blockDim.y);                  // number of threads in block y-dimension
    const auto blockIdx_x = static_cast<unsigned long long>(blockIdx.x) + grid_x_offset;  // current block in grid x-dimension + offsets if the grid size would be too large
    const auto blockIdx_y = static_cast<unsigned long long>(blockIdx.y) + grid_y_offset;  // current block in grid y-dimension + offsets if the grid size would be too large
    const auto INTERNAL_BLOCK_SIZE_ull = static_cast<unsigned long long>(INTERNAL_BLOCK_SIZE);
    const auto THREAD_BLOCK_SIZE_ull = static_cast<unsigned long long>(THREAD_BLOCK_SIZE);
    const auto PADDING_SIZE_ull = static_cast<unsigned long long>(PADDING_SIZE);

    // calculate the indices used in the current thread
    const auto feature_idx = (blockIdx_x * blockDim_x + threadIdx_x) * INTERNAL_BLOCK_SIZE_ull;
    const auto feature_idx_linear = blockIdx_x * blockDim_x * INTERNAL_BLOCK_SIZE_ull + threadIdx_x;
    const auto class_idx = (blockIdx_y * blockDim_y + threadIdx_y) * INTERNAL_BLOCK_SIZE_ull;
    const auto class_idx_linear = blockIdx_y * blockDim_y * INTERNAL_BLOCK_SIZE_ull + threadIdx_x;

    // create the shared memory arrays used for caching data point features
    __shared__ real_type data_cache_feature[THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];
    __shared__ real_type data_cache_alpha[THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];

    // create a thread private array used for internal caching
    real_type temp[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE]{};

    // iterate over all support vectors using blocking to be able to cache them for faster memory accesses
    for (unsigned long long sv = 0; sv < device_specific_num_sv; sv += THREAD_BLOCK_SIZE_ull) {
        // load data into shared memory
        for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
            const auto global_feature_idx = feature_idx_linear + static_cast<unsigned long long>(internal) * THREAD_BLOCK_SIZE_ull;
            const auto global_class_idx = class_idx_linear + static_cast<unsigned long long>(internal) * THREAD_BLOCK_SIZE_ull;

            data_cache_feature[threadIdx.y][internal * THREAD_BLOCK_SIZE + threadIdx.x] = sv_d[global_feature_idx * (device_specific_num_sv + PADDING_SIZE_ull) + sv + threadIdx_y];  // SoA
            data_cache_alpha[threadIdx.y][internal * THREAD_BLOCK_SIZE + threadIdx.x] = alpha_d[global_class_idx * (num_sv + PADDING_SIZE_ull) + sv + sv_offset + threadIdx_y];       // AoS
        }
        __syncthreads();  // wait until all threads loaded their part of the data

        // perform the dot product calculation
        for (unsigned block_dim = 0; block_dim < THREAD_BLOCK_SIZE; ++block_dim) {
            for (unsigned internal_feature = 0; internal_feature < INTERNAL_BLOCK_SIZE; ++internal_feature) {
                for (unsigned internal_class = 0; internal_class < INTERNAL_BLOCK_SIZE; ++internal_class) {
                    temp[internal_feature][internal_class] += data_cache_alpha[block_dim][threadIdx.y * INTERNAL_BLOCK_SIZE + internal_class] * data_cache_feature[block_dim][threadIdx.x * INTERNAL_BLOCK_SIZE + internal_feature];
                }
            }
        }
        __syncthreads();  // wait until all threads performed their part of the calculations
    }

    // update global array with local one
    for (unsigned internal_feature = 0; internal_feature < INTERNAL_BLOCK_SIZE; ++internal_feature) {
        for (unsigned internal_class = 0; internal_class < INTERNAL_BLOCK_SIZE; ++internal_class) {
            const auto global_feature_idx = feature_idx + static_cast<unsigned long long>(internal_feature);
            const auto global_class_idx = class_idx + static_cast<unsigned long long>(internal_class);

            w_d[global_feature_idx * (num_classes + PADDING_SIZE_ull) + global_class_idx] = temp[internal_feature][internal_class];
        }
    }
}

/**
 * @brief Predict the @p predict_points_d using the linear kernel speeding up the calculation using the @p w_d vector.
 * @param[out] prediction_d the predicted values
 * @param[in] w_d the vector to speedup the calculations
 * @param[in] rho_d the previously learned bias
 * @param[in] predict_points_d the data points to predict
 * @param[in] num_classes the number of classes
 * @param[in] num_predict_points the number of data points to predict
 * @param[in] num_features the number of features per data point
 */
__global__ void device_kernel_predict_linear(real_type *prediction_d, const real_type *w_d, const real_type *rho_d, const real_type *predict_points_d, const unsigned long long num_classes, const unsigned long long num_predict_points, const unsigned long long num_features, const unsigned long long grid_x_offset, const unsigned long long grid_y_offset) {
    // cast all values to 64-bit unsigned long long to prevent potential 32-bit overflows
    const auto threadIdx_x = static_cast<unsigned long long>(threadIdx.x);                // current thread in block x-dimension
    const auto threadIdx_y = static_cast<unsigned long long>(threadIdx.y);                // current thread in block y-dimension
    const auto blockDim_x = static_cast<unsigned long long>(blockDim.x);                  // number of threads in block x-dimension
    const auto blockDim_y = static_cast<unsigned long long>(blockDim.y);                  // number of threads in block y-dimension
    const auto blockIdx_x = static_cast<unsigned long long>(blockIdx.x) + grid_x_offset;  // current block in grid x-dimension + offsets if the grid size would be too large
    const auto blockIdx_y = static_cast<unsigned long long>(blockIdx.y) + grid_y_offset;  // current block in grid y-dimension + offsets if the grid size would be too large
    const auto INTERNAL_BLOCK_SIZE_ull = static_cast<unsigned long long>(INTERNAL_BLOCK_SIZE);
    const auto THREAD_BLOCK_SIZE_ull = static_cast<unsigned long long>(THREAD_BLOCK_SIZE);
    const auto FEATURE_BLOCK_SIZE_ull = static_cast<unsigned long long>(FEATURE_BLOCK_SIZE);
    const auto PADDING_SIZE_ull = static_cast<unsigned long long>(PADDING_SIZE);

    // calculate the indices used in the current thread
    const auto pp_idx = (blockIdx_x * blockDim_x + threadIdx_x) * INTERNAL_BLOCK_SIZE_ull;
    const auto pp_idx_linear = blockIdx_x * blockDim_x * INTERNAL_BLOCK_SIZE_ull + threadIdx_x;
    const auto class_idx = (blockIdx_y * blockDim_y + threadIdx_y) * INTERNAL_BLOCK_SIZE_ull;
    const auto class_idx_linear = blockIdx_y * blockDim_y * INTERNAL_BLOCK_SIZE_ull + threadIdx_x;

    // create the shared memory arrays used for caching data point features
    __shared__ real_type data_cache_pp[FEATURE_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];
    __shared__ real_type data_cache_w[FEATURE_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];

    // create a thread private array used for internal caching
    real_type temp[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE]{};

    // iterate over all features using blocking to be able to cache them for faster memory accesses
    for (unsigned long long dim = 0; dim < num_features; dim += FEATURE_BLOCK_SIZE_ull) {
        // load data into shared memory
        for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
            const auto global_pp_idx = pp_idx_linear + static_cast<unsigned long long>(internal) * THREAD_BLOCK_SIZE_ull;
            const auto global_class_idx = class_idx_linear + static_cast<unsigned long long>(internal) * THREAD_BLOCK_SIZE_ull;

            // FEATURE_BLOCK_SIZE = 2 * THREAD_BLOCK_SIZE -> store twice as many values in the shared memory
            data_cache_pp[threadIdx.y][internal * THREAD_BLOCK_SIZE + threadIdx.x] = predict_points_d[(dim + threadIdx_y) * (num_predict_points + PADDING_SIZE_ull) + global_pp_idx];
            data_cache_pp[threadIdx.y + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + threadIdx.x] = predict_points_d[(dim + threadIdx_y + THREAD_BLOCK_SIZE_ull) * (num_predict_points + PADDING_SIZE_ull) + global_pp_idx];
            data_cache_w[threadIdx.y][internal * THREAD_BLOCK_SIZE + threadIdx.x] = w_d[(dim + threadIdx_y) * (num_classes + PADDING_SIZE_ull) + global_class_idx];
            data_cache_w[threadIdx.y + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + threadIdx.x] = w_d[(dim + threadIdx_y + THREAD_BLOCK_SIZE_ull) * (num_classes + PADDING_SIZE_ull) + global_class_idx];
        }
        __syncthreads();  // wait until all threads loaded their part of the data

        // perform the dot product calculation
        for (unsigned block_dim = 0; block_dim < FEATURE_BLOCK_SIZE; ++block_dim) {
            for (unsigned internal_pd = 0; internal_pd < INTERNAL_BLOCK_SIZE; ++internal_pd) {
                for (unsigned internal_class = 0; internal_class < INTERNAL_BLOCK_SIZE; ++internal_class) {
                    temp[internal_pd][internal_class] += data_cache_w[block_dim][threadIdx.y * INTERNAL_BLOCK_SIZE + internal_class] * data_cache_pp[block_dim][threadIdx.x * INTERNAL_BLOCK_SIZE + internal_pd];
                }
            }
        }
        __syncthreads();  // wait until all threads performed their part of the calculations
    }

    // update global array with local one
    for (unsigned internal_pd = 0; internal_pd < INTERNAL_BLOCK_SIZE; ++internal_pd) {
        for (unsigned internal_class = 0; internal_class < INTERNAL_BLOCK_SIZE; ++internal_class) {
            const auto global_pp_idx = pp_idx + static_cast<unsigned long long>(internal_pd);
            const auto global_class_idx = class_idx + static_cast<unsigned long long>(internal_class);

            prediction_d[global_pp_idx * (num_classes + PADDING_SIZE_ull) + global_class_idx] = temp[internal_pd][internal_class] - rho_d[global_class_idx];
        }
    }
}

/**
 * @brief Predict the @p predict_points_d using the @p kernel_function.
 * @tparam kernel_function the type of the used kernel function
 * @tparam Args the types of the parameters necessary for the specific kernel function
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
template <kernel_function_type kernel_function, typename... Args>
__global__ void device_kernel_predict(real_type *prediction_d, const real_type *alpha_d, const real_type *rho_d, const real_type *sv_d, const real_type *predict_points_d, const unsigned long long num_classes, const unsigned long long num_sv, const unsigned long long num_predict_points, const unsigned long long num_features, const unsigned long long grid_x_offset, const unsigned long long grid_y_offset, Args... kernel_function_parameter) {
    // cast all values to 64-bit unsigned long long to prevent potential 32-bit overflows
    const auto threadIdx_x = static_cast<unsigned long long>(threadIdx.x);                // current thread in block x-dimension
    const auto threadIdx_y = static_cast<unsigned long long>(threadIdx.y);                // current thread in block y-dimension
    const auto blockDim_x = static_cast<unsigned long long>(blockDim.x);                  // number of threads in block x-dimension
    const auto blockDim_y = static_cast<unsigned long long>(blockDim.y);                  // number of threads in block y-dimension
    const auto blockIdx_x = static_cast<unsigned long long>(blockIdx.x) + grid_x_offset;  // current block in grid x-dimension + offsets if the grid size would be too large
    const auto blockIdx_y = static_cast<unsigned long long>(blockIdx.y) + grid_y_offset;  // current block in grid y-dimension + offsets if the grid size would be too large
    const auto INTERNAL_BLOCK_SIZE_ull = static_cast<unsigned long long>(INTERNAL_BLOCK_SIZE);
    const auto THREAD_BLOCK_SIZE_ull = static_cast<unsigned long long>(THREAD_BLOCK_SIZE);
    const auto FEATURE_BLOCK_SIZE_ull = static_cast<unsigned long long>(FEATURE_BLOCK_SIZE);
    const auto PADDING_SIZE_ull = static_cast<unsigned long long>(PADDING_SIZE);

    // calculate the indices used in the current thread
    const auto pp_idx = (blockIdx_x * blockDim_x + threadIdx_x) * INTERNAL_BLOCK_SIZE_ull;
    const auto pp_idx_linear = blockIdx_x * blockDim_x * INTERNAL_BLOCK_SIZE_ull + threadIdx_x;
    const auto sv_idx_linear = blockIdx_y * blockDim_y * INTERNAL_BLOCK_SIZE_ull + threadIdx_x;

    // create a thread private array used for internal caching
    real_type temp[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE]{};

    {
        // create the shared memory arrays used for caching data point features
        __shared__ real_type data_cache_pp[FEATURE_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];
        __shared__ real_type data_cache_sv[FEATURE_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];

        // iterate over all features using blocking to be able to cache them for faster memory accesses
        for (unsigned long long dim = 0; dim < num_features; dim += FEATURE_BLOCK_SIZE_ull) {
            // load data into shared memory
            for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                const auto global_pp_idx = pp_idx_linear + static_cast<unsigned long long>(internal) * THREAD_BLOCK_SIZE;
                const auto global_sv_idx = sv_idx_linear + static_cast<unsigned long long>(internal) * THREAD_BLOCK_SIZE;

                // FEATURE_BLOCK_SIZE = 2 * THREAD_BLOCK_SIZE -> store twice as many values in the shared memory
                data_cache_pp[threadIdx.y][internal * THREAD_BLOCK_SIZE + threadIdx.x] = predict_points_d[(dim + threadIdx_y) * (num_predict_points + PADDING_SIZE_ull) + global_pp_idx];
                data_cache_pp[threadIdx.y + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + threadIdx.x] = predict_points_d[(dim + threadIdx_y + THREAD_BLOCK_SIZE_ull) * (num_predict_points + PADDING_SIZE_ull) + global_pp_idx];
                data_cache_sv[threadIdx.y][internal * THREAD_BLOCK_SIZE + threadIdx.x] = sv_d[(dim + threadIdx_y) * (num_sv + PADDING_SIZE_ull) + global_sv_idx];
                data_cache_sv[threadIdx.y + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + threadIdx.x] = sv_d[(dim + threadIdx_y + THREAD_BLOCK_SIZE_ull) * (num_sv + PADDING_SIZE_ull) + global_sv_idx];
            }
            __syncthreads();  // wait until all threads loaded their part of the data

            // perform the feature reduction calculation
            for (unsigned block_dim = 0; block_dim < FEATURE_BLOCK_SIZE; ++block_dim) {
                for (unsigned internal_pd = 0; internal_pd < INTERNAL_BLOCK_SIZE; ++internal_pd) {
                    for (unsigned internal_sv = 0; internal_sv < INTERNAL_BLOCK_SIZE; ++internal_sv) {
                        temp[internal_pd][internal_sv] += detail::feature_reduce<kernel_function>(data_cache_sv[block_dim][threadIdx.y * INTERNAL_BLOCK_SIZE + internal_sv],
                                                                                                  data_cache_pp[block_dim][threadIdx.x * INTERNAL_BLOCK_SIZE + internal_pd]);
                    }
                }
            }
            __syncthreads();  // wait until all threads performed their part of the calculations
        }
    }

    // update temp using the respective kernel function
    for (unsigned internal_pd = 0; internal_pd < INTERNAL_BLOCK_SIZE; ++internal_pd) {
        for (unsigned internal_sv = 0; internal_sv < INTERNAL_BLOCK_SIZE; ++internal_sv) {
            temp[internal_pd][internal_sv] = detail::apply_kernel_function<kernel_function>(temp[internal_pd][internal_sv], kernel_function_parameter...);
        }
    }

    {
        // same shared memory size but with different dimensions
        __shared__ real_type alpha_cache[FEATURE_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];
        __shared__ real_type out_cache[FEATURE_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];

        // iterate over all features using blocking to be able to cache them for faster memory accesses
        for (unsigned long long dim = 0; dim < num_classes; dim += FEATURE_BLOCK_SIZE_ull) {
            // load data into shared memory
            for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                const unsigned long long global_sv_idx = sv_idx_linear + internal * THREAD_BLOCK_SIZE;

                // FEATURE_BLOCK_SIZE = 2 * THREAD_BLOCK_SIZE -> store twice as many values in the shared memory
                alpha_cache[threadIdx.y][internal * THREAD_BLOCK_SIZE + threadIdx.x] = alpha_d[(dim + threadIdx_y) * (num_sv + PADDING_SIZE_ull) + global_sv_idx];
                alpha_cache[threadIdx.y + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + threadIdx.x] = alpha_d[(dim + threadIdx_y + THREAD_BLOCK_SIZE_ull) * (num_sv + PADDING_SIZE_ull) + global_sv_idx];

                // the bias (rho) must only be applied once for all support vectors
                if (blockIdx.y == 0u) {
                    out_cache[threadIdx.y][internal * THREAD_BLOCK_SIZE + threadIdx.x] = -rho_d[dim + threadIdx_y];
                    out_cache[threadIdx.y + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + threadIdx.x] = -rho_d[dim + threadIdx_y + THREAD_BLOCK_SIZE_ull];
                } else {
                    out_cache[threadIdx.y][internal * THREAD_BLOCK_SIZE + threadIdx.x] = real_type{ 0.0 };
                    out_cache[threadIdx.y + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + threadIdx.x] = real_type{ 0.0 };
                }
            }
            __syncthreads();  // wait until all threads loaded their part of the data

            // calculate intermediate results and store them in shared memory
            for (unsigned class_idx = 0; class_idx < FEATURE_BLOCK_SIZE; ++class_idx) {
                for (unsigned internal_pd = 0; internal_pd < INTERNAL_BLOCK_SIZE; ++internal_pd) {
                    for (unsigned internal_sv = 0; internal_sv < INTERNAL_BLOCK_SIZE; ++internal_sv) {
                        out_cache[(class_idx + threadIdx.y) % FEATURE_BLOCK_SIZE][internal_pd * THREAD_BLOCK_SIZE + threadIdx.x] +=
                            temp[internal_pd][internal_sv] * alpha_cache[(class_idx + threadIdx.y) % FEATURE_BLOCK_SIZE][threadIdx.y * INTERNAL_BLOCK_SIZE + internal_sv];
                    }
                }
                __syncthreads();  // wait until all threads performed their part of the calculations
            }

            // add intermediate cached results to prediction_d
            for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                const auto global_pp_idx = pp_idx + static_cast<unsigned long long>(internal);

                atomicAdd(&prediction_d[global_pp_idx * (num_classes + PADDING_SIZE_ull) + dim + threadIdx_y], out_cache[threadIdx.y][internal * THREAD_BLOCK_SIZE + threadIdx.x]);
                atomicAdd(&prediction_d[global_pp_idx * (num_classes + PADDING_SIZE_ull) + dim + threadIdx_y + THREAD_BLOCK_SIZE_ull], out_cache[threadIdx.y + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + threadIdx.x]);
            }
            __syncthreads();  // wait until all threads updated their part of the prediction
        }
    }
}

}  // namespace plssvm::cuda::detail

#endif  // PLSSVM_BACKENDS_CUDA_KERNEL_PREDICT_KERNEL_HPP_
