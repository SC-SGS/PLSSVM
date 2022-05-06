/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/@PLSSVM_SYCL_BACKEND_INCLUDE_NAME@/detail/utility.hpp"

#include "plssvm/backends/@PLSSVM_SYCL_BACKEND_INCLUDE_NAME@/detail/constants.hpp"  // PLSSVM_SYCL_BACKEND_COMPILER_DPCPP, PLSSVM_SYCL_BACKEND_COMPILER_HIPSYCL, forward declaration and namespace alias
#include "plssvm/detail/string_utility.hpp"                                         // sycl::detail::to_lower_case, sycl::detail::contains
#include "plssvm/target_platforms.hpp"                                              // plssvm::target_platform

#include "sycl/sycl.hpp"  // sycl::queue, sycl::platform, sycl::device, sycl::property::queue, sycl::info, sycl::gpu_selector

#include <memory>   // std::unique_ptr, std::make_unique
#include <string>   // std::string
#include <utility>  // std::pair, std::make_pair
#include <vector>   // std::vector

namespace plssvm::@PLSSVM_SYCL_BACKEND_NAMESPACE_NAME@::detail {

[[nodiscard]] std::vector<std::unique_ptr<detail::sycl::queue>> get_device_list_impl(const target_platform target) {
    std::vector<std::unique_ptr<detail::sycl::queue>> target_devices;
    for (const detail::sycl::platform &platform : detail::sycl::platform::get_platforms()) {
        for (const detail::sycl::device &device : platform.get_devices()) {
            if (target == target_platform::cpu && device.is_cpu()) {
#if defined(PLSSVM_HAS_CPU_TARGET)
                // select CPU device
                target_devices.emplace_back(std::make_unique<detail::sycl::queue>(device, detail::sycl::property::queue::in_order()));
#endif
            } else {
                // must be a GPU device
                if (device.is_gpu()) {
                    // get vendor name of current GPU device
                    std::string vendor_string = device.get_info<detail::sycl::info::device::vendor>();
                    // convert vendor name to lower case
                    ::plssvm::detail::to_lower_case(vendor_string);
                    // get platform name of current GPU device
                    std::string platform_string = platform.get_info<detail::sycl::info::platform::name>();
                    // convert platform name to lower case
                    ::plssvm::detail::to_lower_case(platform_string);
                    switch (target) {
#if defined(PLSSVM_HAS_NVIDIA_TARGET)
                        case target_platform::gpu_nvidia:
                            if (::plssvm::detail::contains(vendor_string, "nvidia")) {
                                target_devices.emplace_back(std::make_unique<detail::sycl::queue>(device, detail::sycl::property::queue::in_order()));
                            }
                            break;
#endif
#if defined(PLSSVM_HAS_AMD_TARGET)
                        case target_platform::gpu_amd:
                            if (::plssvm::detail::contains(vendor_string, "amd") || ::plssvm::detail::contains(vendor_string, "advanced micro devices")) {
                                target_devices.emplace_back(std::make_unique<detail::sycl::queue>(device, detail::sycl::property::queue::in_order()));
                            }
                            break;
#endif
#if defined(PLSSVM_HAS_INTEL_TARGET)
                        case target_platform::gpu_intel:
                            if (::plssvm::detail::contains(vendor_string, "intel")) {
    #if PLSSVM_SYCL_BACKEND_COMPILER == PLSSVM_SYCL_BACKEND_COMPILER_DPCPP
                                if (::plssvm::detail::contains(platform_string, PLSSVM_SYCL_BACKEND_DPCPP_BACKEND_TYPE)) {
                                    target_devices.emplace_back(std::make_unique<detail::sycl::queue>(device, detail::sycl::property::queue::in_order()));
                                }
    #elif PLSSVM_SYCL_BACKEND_COMPILER == PLSSVM_SYCL_BACKEND_COMPILER_HIPSYCL
                                target_devices.emplace_back(std::make_unique<detail::sycl::queue>(device, detail::sycl::property::queue::in_order()));
    #endif
                            }
                            break;
#endif
                        default:
                            break;
                    }
                }
            }
        }
    }
    return target_devices;
}

std::pair<std::vector<std::unique_ptr<detail::sycl::queue>>, ::plssvm::target_platform> get_device_list(const target_platform target) {
    if (target != target_platform::automatic) {
        return std::make_pair(get_device_list_impl(target), target);
    } else {
        target_platform used_target = target_platform::gpu_nvidia;
        std::vector<std::unique_ptr<detail::sycl::queue>> target_devices = get_device_list_impl(used_target);
        if (target_devices.empty()) {
            used_target = target_platform::gpu_amd;
            target_devices = get_device_list_impl(used_target);
            if (target_devices.empty()) {
                used_target = target_platform::gpu_intel;
                target_devices = get_device_list_impl(used_target);
                if (target_devices.empty()) {
                    used_target = target_platform::cpu;
                    target_devices = get_device_list_impl(used_target);
                }
            }
        }
        return std::make_pair(std::move(target_devices), used_target);
    }
}

void device_synchronize(detail::sycl::queue &queue) {
    queue.wait_and_throw();
}

}  // namespace plssvm::@PLSSVM_SYCL_BACKEND_NAMESPACE_NAME@::detail