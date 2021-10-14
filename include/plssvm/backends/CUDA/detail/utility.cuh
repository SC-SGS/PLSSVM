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

#pragma once

#define PLSSVM_CUDA_ERROR_CHECK(err) plssvm::cuda::detail::gpu_assert((err))

namespace plssvm::cuda::detail {

/**
 * @brief Check the CUDA error @p code. If @p code signals an error, throw a `plssvm::cuda::backend_exception`.
 * @details The exception contains the error name and error string for more debug information.
 * @param[in] code the CUDA error code to check
 * @throws `plssvm::cuda::backend_exception` if the error code signals a failure
 */
void gpu_assert(cudaError_t code);

/**
 * @brief Returns the number of available devices.
 * @return the number of devices (`[[nodiscard]]`)
 */
[[nodiscard]] int get_device_count();

/**
 * @brief Set the device @p device to the active CUDA device.
 * @param[in] device the now active device
 */
void set_device(int device);

/**
 * @brief Returns the last error from a runtime call.
 */
void peek_at_last_error();

/**
 * @brief Wait for the compute device @p device to finish.
 * @details Calls `peek_at_last_error()` before synchronizing.
 * @param[in] device the CUDA device to synchronize
 * @throws plssvm::cuda::backend_exception if the given device ID is smaller than `0` or greater or equal than the available number of devices
 */
void device_synchronize(int device);

}  // namespace plssvm::cuda::detail