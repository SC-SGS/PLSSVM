/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Implements a pImpl class for an NVML device handle.
 */

#ifndef PLSSVM_DETAIL_TRACKING_GPU_NVIDIA_NVML_DEVICE_HANDLE_IMPL_HPP_
#define PLSSVM_DETAIL_TRACKING_GPU_NVIDIA_NVML_DEVICE_HANDLE_IMPL_HPP_
#pragma once

#include "plssvm/detail/tracking/gpu_nvidia/nvml_device_handle.hpp"  // plssvm::detail::tracking::nvml_device_handle
#include "plssvm/detail/tracking/gpu_nvidia/utility.hpp"             // PLSSVM_NVML_ERROR_CHECK

#include "nvml.h"  // nvmlDevice_t

#include <cstddef>  // std::size_t
#include <memory>   // std::shared_ptr, std::make_shared

namespace plssvm::detail::tracking {

/**
 * @brief The PImpl implementation struct encapsulating a nvmlDevice_t.
 */
struct nvml_device_handle::nvml_device_handle_impl {
  public:
    /**
     * @brief Get the nvmlDevice_t for the device with ID @p device_id.
     * @param[in] device_id the device to get the handle for
     */
    explicit nvml_device_handle_impl(const std::size_t device_id) {
        PLSSVM_NVML_ERROR_CHECK(nvmlDeviceGetHandleByIndex(static_cast<int>(device_id), &device));
    }

    /// The wrapped NVML device handle.
    nvmlDevice_t device{};
};

inline nvml_device_handle::nvml_device_handle(const std::size_t device_id) :
    impl{ std::make_shared<nvml_device_handle_impl>(device_id) } { }

}  // namespace plssvm::detail::tracking

#endif  // PLSSVM_DETAIL_TRACKING_GPU_NVIDIA_NVML_DEVICE_HANDLE_IMPL_HPP_
