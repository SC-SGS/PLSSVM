/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines a pImpl class for an NVML device handle.
 */

#ifndef PLSSVM_DETAIL_TRACKING_GPU_NVIDIA_NVML_DEVICE_HANDLE_HPP_
#define PLSSVM_DETAIL_TRACKING_GPU_NVIDIA_NVML_DEVICE_HANDLE_HPP_
#pragma once

#include "plssvm/exceptions/exceptions.hpp"  // plssvm::hardware_sampling_exception

#include <cstddef>  // std::size_t
#include <memory>   // std::shared_ptr

namespace plssvm::detail::tracking {

class nvml_device_handle {
  public:
    nvml_device_handle() = default;
    explicit nvml_device_handle(std::size_t device_id);

    struct nvml_device_handle_impl;

    [[nodiscard]] nvml_device_handle_impl &get_impl() {
        if (impl == nullptr) {
            throw hardware_sampling_exception{ "Pointer to implementation is a nullptr! Maybe *this is default constructed?" };
        }
        return *impl;
    }

    [[nodiscard]] const nvml_device_handle_impl &get_impl() const {
        if (impl == nullptr) {
            throw hardware_sampling_exception{ "Pointer to implementation is a nullptr! Maybe *this is default constructed?" };
        }
        return *impl;
    }

  private:
    std::shared_ptr<nvml_device_handle_impl> impl{};
};

}  // namespace plssvm::detail::tracking

#endif  // PLSSVM_DETAIL_TRACKING_GPU_NVIDIA_NVML_DEVICE_HANDLE_HPP_
