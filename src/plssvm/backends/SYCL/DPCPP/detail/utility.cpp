/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/SYCL/DPCPP/detail/utility.hpp"

#include "plssvm/backends/SYCL/DPCPP/detail/queue_impl.hpp"  // plssvm::dpcpp::detail::queue (PImpl implementation)
#include "plssvm/detail/string_utility.hpp"                  // plssvm::detail::{as_lower_case, contains}
#include "plssvm/detail/utility.hpp"                         // plssvm::detail::contains
#include "plssvm/target_platforms.hpp"                       // plssvm::target_platform, plssvm::determine_default_target_platform

#include "sycl/sycl.hpp"  // ::sycl::platform, ::sycl::device, ::sycl::property::queue, ::sycl::info

#include <map>      // std::multimap
#include <memory>   // std::make_shared
#include <sstream>  // std::ostringstream
#include <string>   // std::string
#include <utility>  // std::pair, std::make_pair, std::move
#include <vector>   // std::vector

namespace plssvm::dpcpp::detail {

[[nodiscard]] std::pair<std::vector<queue>, ::plssvm::target_platform> get_device_list(target_platform target) {
    // iterate over all platforms and save all available devices
    std::multimap<target_platform, ::sycl::device> platform_devices;
    // get the available target_platforms
    const std::vector<target_platform> available_target_platforms = list_available_target_platforms();

    // enumerate all available platforms and retrieve the associated devices
    for (const ::sycl::platform &platform : ::sycl::platform::get_platforms()) {
        // get devices associated with current platform
        for (const ::sycl::device &device : platform.get_devices()) {
            if (device.is_cpu()) {
                // the current device is a CPU
                // -> check if the CPU target has been enabled
                if (::plssvm::detail::contains(available_target_platforms, target_platform::cpu)) {
                    platform_devices.insert({ target_platform::cpu, device });
                }
            } else if (device.is_gpu()) {
                // the current device is a GPU
                // get vendor string and convert it to all lower case
                const std::string vendor_string = ::plssvm::detail::as_lower_case(device.get_info<::sycl::info::device::vendor>());
                // get platform name of current GPU device and convert it to all lower case
                const std::string platform_string = ::plssvm::detail::as_lower_case(platform.get_info<::sycl::info::platform::name>());

                // check vendor string and insert to correct target platform
                if (::plssvm::detail::contains(vendor_string, "nvidia") && ::plssvm::detail::contains(available_target_platforms, target_platform::gpu_nvidia)) {
                    platform_devices.insert({ target_platform::gpu_nvidia, device });
                } else if ((::plssvm::detail::contains(vendor_string, "amd") || ::plssvm::detail::contains(vendor_string, "advanced micro devices"))
                           && ::plssvm::detail::contains(available_target_platforms, target_platform::gpu_amd)) {
                    // select between DPC++'s OpenCL and HIP backend
                    std::ostringstream oss;
                    oss << device.get_backend();
                    if (::plssvm::detail::contains(oss.str(), PLSSVM_SYCL_BACKEND_DPCPP_GPU_AMD_BACKEND_TYPE)) {
                        platform_devices.insert({ target_platform::gpu_amd, device });
                    }
                } else if (::plssvm::detail::contains(vendor_string, "intel") || ::plssvm::detail::contains(available_target_platforms, target_platform::gpu_intel)) {
                    // select between DPC++'s OpenCL and Level-Zero backend
                    if (::plssvm::detail::contains(platform_string, PLSSVM_SYCL_BACKEND_DPCPP_BACKEND_TYPE)) {
                        platform_devices.insert({ target_platform::gpu_intel, device });
                    }
                }
            }
        }
    }

    // determine target if provided target_platform is automatic
    if (target == target_platform::automatic) {
        // get the target_platforms available on this system from the platform_devices map
        std::vector<target_platform> system_devices;
        for (const auto &[key, value] : platform_devices) {
            system_devices.push_back(key);
        }
        // determine the target_platform
        target = determine_default_target_platform(system_devices);
    }

    // create one queue for each device in the requested/determined target platform
    std::vector<queue> queues;
    const auto range = platform_devices.equal_range(target);
    for (auto it = range.first; it != range.second; ++it) {
        detail::queue q{};
        q.impl = std::make_shared<queue::queue_impl>(it->second, ::sycl::property::queue::in_order());
        queues.emplace_back(std::move(q));
    }

    return std::make_pair(std::move(queues), target);
}

void device_synchronize(const queue &q) {
    q.impl->sycl_queue.wait_and_throw();
}

queue get_default_queue() {
    queue q;
    q.impl = std::make_shared<queue::queue_impl>();
    return q;
}

}  // namespace plssvm::dpcpp::detail