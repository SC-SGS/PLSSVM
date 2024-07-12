/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines a pImpl class for a Level Zero device handle.
 */

#ifndef PLSSVM_DETAIL_TRACKING_GPU_INTEL_LEVEL_ZERO_DEVICE_HANDLE_HPP_
#define PLSSVM_DETAIL_TRACKING_GPU_INTEL_LEVEL_ZERO_DEVICE_HANDLE_HPP_
#pragma once

#include "plssvm/exceptions/exceptions.hpp"  // plssvm::hardware_sampling_exception

#include <cstddef>  // std::size_t
#include <memory>   // std::shared_ptr

namespace plssvm::detail::tracking {

/**
 * @brief PImpl class to hide the ze_driver_handle_t and ze_device_handle_t handles from the public interface (and, therefore, the "level_zero/ze_api.h" header).
 */
class level_zero_device_handle {
  public:
    /**
     * @brief Default construct an level_zero_device_handle. Such a handle may not be used in an Level Zero function!
     */
    level_zero_device_handle() = default;
    /**
     * @brief Construct an level_zero_device_handle for the device with ID @p device_id.
     * @param[in] device_id the device to get the handle for
     */
    explicit level_zero_device_handle(std::size_t device_id);

    /**
     * @brief The pImpl helper struct.
     */
    struct level_zero_device_handle_impl;

    /**
     * @brief Get the level_zero_device_handle implementation used to access the actual ze_driver_handle_t and ze_device_handle_t.
     * @throws hardware_sampling_exception if `*this` has been default constructed
     * @return the device handle (`[[nodiscard]]`)
     */
    [[nodiscard]] level_zero_device_handle_impl &get_impl() {
        if (impl == nullptr) {
            throw hardware_sampling_exception{ "Pointer to implementation is a nullptr! Maybe *this is default constructed?" };
        }
        return *impl;
    }

    /**
     * @brief Get the level_zero_device_handle implementation used to access the actual ze_driver_handle_t and ze_device_handle_t.
     * @throws hardware_sampling_exception if `*this` has been default constructed
     * @return the device handle (`[[nodiscard]]`)
     */
    [[nodiscard]] const level_zero_device_handle_impl &get_impl() const {
        if (impl == nullptr) {
            throw hardware_sampling_exception{ "Pointer to implementation is a nullptr! Maybe *this is default constructed?" };
        }
        return *impl;
    }

  private:
    /// The pointer to the actual implementation struct.
    std::shared_ptr<level_zero_device_handle_impl> impl{};
};

}  // namespace plssvm::detail::tracking

#endif  // PLSSVM_DETAIL_TRACKING_GPU_INTEL_LEVEL_ZERO_DEVICE_HANDLE_HPP_
