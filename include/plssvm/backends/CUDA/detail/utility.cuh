/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Utility functions for the CUDA backend.
 */

#ifndef PLSSVM_BACKENDS_CUDA_DETAIL_UTILITY_HPP_
#define PLSSVM_BACKENDS_CUDA_DETAIL_UTILITY_HPP_
#pragma once

#include "fmt/ostream.h"  // fmt::formatter, fmt::ostream_formatter

/**
 * @def PLSSVM_CUDA_ERROR_CHECK
 * @brief Macro used for error checking CUDA runtime functions.
 */
#define PLSSVM_CUDA_ERROR_CHECK(err) plssvm::cuda::detail::gpu_assert((err))

namespace plssvm::cuda::detail {

/**
 * @brief Check the CUDA error @p code. If @p code signals an error, throw a plssvm::cuda::backend_exception.
 * @details The exception contains the following message: "CUDA assert 'CUDA_ERROR_NAME' (CUDA_ERROR_CODE): CUDA_ERROR_STRING".
 * @param[in] code the CUDA error code to check
 * @throws plssvm::cuda::backend_exception if the error code signals a failure
 */
void gpu_assert(cudaError_t code);

/**
 * @brief Returns the number of available CUDA devices.
 * @return the number of devices (`[[nodiscard]]`)
 */
[[nodiscard]] int get_device_count();

/**
 * @brief Set the @p device to the active CUDA device.
 * @param[in] device the now active device
 * @throws plssvm::cuda::backend_exception if the given device ID is smaller than 0 or greater or equal than the available number of devices
 */
void set_device(int device);

/**
 * @brief Returns the last error from a CUDA runtime call.
 */
void peek_at_last_error();

/**
 * @brief Wait for the compute @p device to finish.
 * @details Calls plssvm::cuda::detail::peek_at_last_error() before synchronizing.
 * @param[in] device the CUDA device to synchronize
 * @throws plssvm::cuda::backend_exception if the given device ID is smaller than 0 or greater or equal than the available number of devices
 */
void device_synchronize(int device);

}  // namespace plssvm::cuda::detail

template <>
struct fmt::formatter<cudaError_t> : fmt::ostream_formatter {};

#endif  // PLSSVM_BACKENDS_CUDA_DETAIL_UTILITY_HPP_