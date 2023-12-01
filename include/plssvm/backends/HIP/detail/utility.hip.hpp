/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Utility functions for the HIP backend.
 */

#ifndef PLSSVM_BACKENDS_HIP_DETAIL_UTILITY_HPP_
#define PLSSVM_BACKENDS_HIP_DETAIL_UTILITY_HPP_
#pragma once

#include "fmt/ostream.h"  // fmt::formatter, fmt::ostream_formatter

#include "hip/hip_runtime_api.h"  // hipError_t

/**
 * @def PLSSVM_HIP_ERROR_CHECK
 * @brief Macro used for error checking HIP runtime functions.
 */
#define PLSSVM_HIP_ERROR_CHECK(err) plssvm::hip::detail::gpu_assert((err))

namespace plssvm::hip::detail {

/**
 * @brief Check the HIP error @p code. If @p code signals an error, throw a plssvm::hip::backend_exception.
 * @details The exception contains the following message: "HIP assert 'HIP_ERROR_NAME' (HIP_ERROR_CODE): HIP_ERROR_STRING".
 * @param[in] code the HIP error code to check
 * @throws plssvm::hip::backend_exception if the error code signals a failure
 */
void gpu_assert(hipError_t code);

/**
 * @brief Returns the number of available HIP devices.
 * @return the number of devices (`[[nodiscard]]`)
 */
[[nodiscard]] int get_device_count();

/**
 * @brief Set the @p device to the active HIP device.
 * @param[in] device the now active device
 * @throws plssvm::hip::backend_exception if the given device ID is smaller than 0 or greater or equal than the available number of devices
 */
void set_device(int device);

/**
 * @brief Returns the last error from a HIP runtime call.
 */
void peek_at_last_error();

/**
 * @brief Wait for the compute @p device to finish.
 * @details Calls plssvm::hip::detail::peek_at_last_error() before synchronizing.
 * @param[in] device the HIP device to synchronize
 * @throws plssvm::hip::backend_exception if the given device ID is smaller than 0 or greater or equal than the available number of devices
 */
void device_synchronize(int device);

}  // namespace plssvm::hip::detail

template <>
struct fmt::formatter<hipError_t> : fmt::ostream_formatter {};

#endif  // PLSSVM_BACKENDS_HIP_DETAIL_UTILITY_HPP_
