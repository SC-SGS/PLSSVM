/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright
 *
 * @brief Function object to select the used device based on the requested target platform.
 */

#pragma once

#include "plssvm/detail/string_utility.hpp"  // plssvm::detail::contains
#include "plssvm/target_platform.hpp"        // plssvm::target_platform

#include "sycl/sycl.hpp"  // sycl::device_selector, sycl::device, sycl::default_selector, sycl::cpu_selector, sycl::info::device

#include <algorithm>  // std::transform
#include <cctype>     // std::tolower
#include <string>     // std::string

namespace plssvm::sycl::detail {

// TODO: multi GPU support, check if PLATFORM has been enabled

/**
 * @brief Custom [`sycl::device_selector`](https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#sec:device-selection) selecting a device depending on the requested target platform.
 */
class device_selector final : public ::sycl::device_selector {
  public:
    /**
     * @brief Create a new custom device selector.
     * @param[in] target the requested target platform
     */
    explicit device_selector(const target_platform target) :
        target_{ target } {}

    /**
     * @brief Select a device depending on the requested target platform.
     * @param[in] device the current [`sycl::device`](https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#sec:device-class)
     * @return the device score, a negative score results in the @p device never be picked
     */
    int operator()(const ::sycl::device &device) const override {
        if (target_ == target_platform::automatic) {
            // use default device
            return ::sycl::default_selector{}.operator()(device);
        } else if (target_ == target_platform::cpu) {
            // select CPU device
            return ::sycl::cpu_selector{}.operator()(device);
        } else {
            // must be a GPU device
            if (!device.is_gpu()) {
                return -1;
            }
            std::string vendor_string = device.get_info<::sycl::info::device::vendor>();
            std::transform(vendor_string.begin(), vendor_string.end(), vendor_string.begin(), [](const char c) { return std::tolower(c); });

            switch (target_) {
                case target_platform::gpu_nvidia:
                    return ::plssvm::detail::contains(vendor_string, "nvidia") ? 100 : -1;
                case target_platform::gpu_amd:
                    return ::plssvm::detail::contains(vendor_string, "amd") ? 100 : -1;
                case target_platform::gpu_intel:
                    return ::plssvm::detail::contains(vendor_string, "intel") ? 100 : -1;
                default:
                    return -10;
            }
        }
    }

  private:
    const target_platform target_;
};

}  // namespace plssvm::sycl::detail