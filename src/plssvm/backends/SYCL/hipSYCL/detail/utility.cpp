/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/SYCL/hipSYCL/detail/utility.hpp"

#include "plssvm/backends/SYCL/hipSYCL/detail/queue_impl.hpp"  // plssvm::hipsycl::detail::queue (PImpl implementation)
#include "plssvm/detail/string_utility.hpp"                    // sycl::detail::to_lower_case, sycl::detail::contains
#include "plssvm/target_platforms.hpp"                         // plssvm::target_platform

#include "sycl/sycl.hpp"  // sycl::queue, sycl::platform, sycl::device, sycl::property::queue, sycl::info, sycl::gpu_selector

#include <memory>   // std::unique_ptr, std::make_unique
#include <string>   // std::string
#include <utility>  // std::pair, std::make_pair
#include <vector>   // std::vector

namespace plssvm::hipsycl::detail {

[[nodiscard]] std::vector<queue> get_device_list_impl(const target_platform target) {
    // TODO: rewrite like OpenCL?
    std::vector<queue> target_devices;
    for (const ::sycl::platform &platform : ::sycl::platform::get_platforms()) {
        for (const ::sycl::device &device : platform.get_devices()) {
            if (target == target_platform::cpu && device.is_cpu()) {
#if defined(PLSSVM_HAS_CPU_TARGET)
                // select CPU device
                detail::queue q;
                q.impl = std::make_shared<queue::queue_impl>(device, ::sycl::property::queue::in_order());
                target_devices.emplace_back(std::move(q));
#endif
            } else {
                // must be a GPU device
                if (device.is_gpu()) {
                    detail::queue q{};
                    // get vendor name of current GPU device
                    std::string vendor_string = device.get_info<::sycl::info::device::vendor>();
                    // convert vendor name to lower case
                    ::plssvm::detail::to_lower_case(vendor_string);
                    // get platform name of current GPU device
                    std::string platform_string = platform.get_info<::sycl::info::platform::name>();
                    // convert platform name to lower case
                    ::plssvm::detail::to_lower_case(platform_string);
                    switch (target) {
#if defined(PLSSVM_HAS_NVIDIA_TARGET)
                        case target_platform::gpu_nvidia:
                            if (::plssvm::detail::contains(vendor_string, "nvidia")) {
                                q.impl = std::make_shared<queue::queue_impl>(device, ::sycl::property::queue::in_order());
                                target_devices.emplace_back(std::move(q));
                            }
                            break;
#endif
#if defined(PLSSVM_HAS_AMD_TARGET)
                        case target_platform::gpu_amd:
                            if (::plssvm::detail::contains(vendor_string, "amd")) {
                                q.impl = std::make_shared<queue::queue_impl>(device, ::sycl::property::queue::in_order());
                                target_devices.emplace_back(std::move(q));
                            }
                            break;
#endif
#if defined(PLSSVM_HAS_INTEL_TARGET)
                        case target_platform::gpu_intel:
                            if (::plssvm::detail::contains(vendor_string, "intel")) {
                                q.impl = std::make_shared<queue::queue_impl>(device, ::sycl::property::queue::in_order());
                                target_devices.emplace_back(std::move(q));
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

std::pair<std::vector<queue>, ::plssvm::target_platform> get_device_list(const target_platform target) {
    if (target != target_platform::automatic) {
        return std::make_pair(get_device_list_impl(target), target);
    } else {
        target_platform used_target = target_platform::gpu_nvidia;
        std::vector<queue> target_devices = get_device_list_impl(used_target);
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

void device_synchronize(queue &q) {
    q.impl->sycl_queue.wait_and_throw();
}

queue get_default_queue() {
    queue q;
    q.impl = std::make_unique<queue::queue_impl>();
    return q;
}

}  // namespace plssvm::hipsycl::detail