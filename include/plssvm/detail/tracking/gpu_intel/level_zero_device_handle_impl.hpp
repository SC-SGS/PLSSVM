/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Implements a pImpl class for a Level Zero device handle.
 */

#ifndef PLSSVM_DETAIL_TRACKING_GPU_INTEL_LEVEL_ZERO_DEVICE_HANDLE_IMPL_HPP_
#define PLSSVM_DETAIL_TRACKING_GPU_INTEL_LEVEL_ZERO_DEVICE_HANDLE_IMPL_HPP_
#pragma once

#include "plssvm/detail/tracking/gpu_intel/level_zero_device_handle.hpp"  // plssvm::detail::tracking::level_zero_device_handle
#include "plssvm/detail/tracking/gpu_intel/utility.hpp"                   // PLSSVM_LEVEL_ZERO_ERROR_CHECK

#include "level_zero/ze_api.h"  // Level Zero runtime functions

#include <cstddef>  // std::size_t
#include <memory>   // std::shared_ptr, std::make_shared
#include <vector>   // std::vector

namespace plssvm::detail::tracking {

/**
 * @brief The PImpl implementation struct encapsulating a ze_driver_handle_t and ze_device_handle_t.
 */
struct level_zero_device_handle::level_zero_device_handle_impl {
  public:
    /**
     * @brief Get the ze_driver_handle_t and ze_device_handle_t for the device with ID @p device_id.
     * @param[in] device_id the device to get the handle for
     */
    explicit level_zero_device_handle_impl(const std::size_t device_id) {
        // discover the number of drivers
        std::uint32_t driver_count{ 0 };
        PLSSVM_LEVEL_ZERO_ERROR_CHECK(zeDriverGet(&driver_count, nullptr));

        // check if only the single GPU driver has been found
        if (driver_count > 1) {
            throw hardware_sampling_exception{ fmt::format("Found too many GPU drivers ({})!", driver_count) };
        }

        // get the GPU driver
        PLSSVM_LEVEL_ZERO_ERROR_CHECK(zeDriverGet(&driver_count, &driver));

        // get all GPUs for the current driver
        std::uint32_t device_count{ 0 };
        PLSSVM_LEVEL_ZERO_ERROR_CHECK(zeDeviceGet(driver, &device_count, nullptr));

        // check if enough GPUs have been found
        if (driver_count <= device_id) {
            throw hardware_sampling_exception{ fmt::format("Found only {} GPUs, but GPU with the ID was requested!", device_count, device_id) };
        }

        // get the GPUs
        std::vector<ze_device_handle_t> all_devices(device_count);
        PLSSVM_LEVEL_ZERO_ERROR_CHECK(zeDeviceGet(driver, &device_count, all_devices.data()));

        // save the requested device
        device = all_devices[device_id];
    }

    /// The wrapped Level Zero driver handle.
    ze_driver_handle_t driver{};
    /// The wrapped Level Zero device handle.
    ze_device_handle_t device{};
};

inline level_zero_device_handle::level_zero_device_handle(const std::size_t device_id) :
    impl{ std::make_shared<level_zero_device_handle_impl>(device_id) } { }

}  // namespace plssvm::detail::tracking

#endif  // PLSSVM_DETAIL_TRACKING_GPU_INTEL_LEVEL_ZERO_DEVICE_HANDLE_IMPL_HPP_
