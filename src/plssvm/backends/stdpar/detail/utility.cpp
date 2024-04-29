/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/stdpar/detail/utility.hpp"

#include "plssvm/detail/string_utility.hpp"  // plssvm::detail::as_lower_case
#include "plssvm/detail/utility.hpp"         // ::plssvm::detail::contains
#include "plssvm/target_platforms.hpp"       // plssvm::target_platforms

#if defined(PLSSVM_STDPAR_BACKEND_HAS_ACPP)
    #include "plssvm/backends/SYCL/AdaptiveCpp/detail/utility.hpp"  // plssvm::adaptivecpp::detail::get_adaptivecpp_version_short

    #include "sycl/sycl.hpp"  // sycl::device, sycl::info::device::vendor
#endif

#include <string>  // std::string

namespace plssvm::stdpar::detail {

#if defined(PLSSVM_STDPAR_BACKEND_HAS_ACPP)
[[nodiscard]] bool default_device_equals_target(const ::sycl::device &device, const plssvm::target_platform target) {
    const std::string vendor_string = ::plssvm::detail::as_lower_case(device.get_info<::sycl::info::device::vendor>());

    switch (target) {
        case target_platform::automatic:
            return false;  // may never occur
        case target_platform::cpu:
            return device.is_cpu();
        case target_platform::gpu_nvidia:
            return device.is_gpu() && ::plssvm::detail::contains(vendor_string, "nvidia");
        case target_platform::gpu_amd:
            return device.is_gpu() && (::plssvm::detail::contains(vendor_string, "amd") || ::plssvm::detail::contains(vendor_string, "advanced micro devices"));
        case target_platform::gpu_intel:
            return device.is_gpu() && (::plssvm::detail::contains(vendor_string, "intel") || ::plssvm::detail::contains(vendor_string, "pci:32902"));
    }
}
#endif

std::string get_stdpar_version() {
#if defined(PLSSVM_STDPAR_BACKEND_HAS_ACPP)
    return plssvm::adaptivecpp::detail::get_adaptivecpp_version_short();
#else
    return "unknown";
#endif
}

}  // namespace plssvm::stdpar::detail
