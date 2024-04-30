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

#if defined(PLSSVM_STDPAR_BACKEND_HAS_ACPP) || defined(PLSSVM_STDPAR_BACKEND_HAS_INTEL_LLVM)
    #include "sycl/sycl.hpp"  // sycl::device, sycl::info::device::vendor
#endif

#if defined(PLSSVM_STDPAR_BACKEND_HAS_ACPP)
    #include "plssvm/backends/SYCL/AdaptiveCpp/detail/utility.hpp"  // plssvm::adaptivecpp::detail::get_adaptivecpp_version_short
#elif defined(PLSSVM_STDPAR_BACKEND_HAS_DPCPP)
    #include "plssvm/backends/SYCL/AdaptiveCpp/detail/utility.hpp"  // plssvm::dpcpp::detail::{get_dpcpp_version, get_dpcpp_timestamp_version}
#endif

#include "fmt/core.h"  // fmt::format

#include <string>  // std::string

namespace plssvm::stdpar::detail {

#if defined(PLSSVM_STDPAR_BACKEND_HAS_ACPP) || defined(PLSSVM_STDPAR_BACKEND_HAS_INTEL_LLVM)
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
#elif defined(PLSSVM_STDPAR_BACKEND_HAS_NVHPC)
#if defined(PLSSVM_STDPAR_BACKEND_NVHPC_GPU)
    int runtime_version{};
    cudaRuntimeGetVersion(&runtime_version);
    // parse it to a more useful string
    int major_version = runtime_version / 1000;
    int minor_version = runtime_version % 1000 / 10;
    return fmt::format("{}.{}.{}; {}.{}", __NVCOMPILER_MAJOR__, __NVCOMPILER_MINOR__, __NVCOMPILER_PATCHLEVEL__, major_version, minor_version);
#else
    return fmt::format("{}.{}.{}", __NVCOMPILER_MAJOR__, __NVCOMPILER_MINOR__, __NVCOMPILER_PATCHLEVEL__);
#endif
#elif defined(PLSSVM_STDPAR_BACKEND_HAS_INTEL_LLVM)
    return fmt::format("{}; {}", plssvm::adaptivecpp::detail::get_dpcpp_version(), plssvm::adaptivecpp::detail::get_dpcpp_timestamp_version());
#elif defined(PLSSVM_STDPAR_BACKEND_HAS_GNU_TBB)
    return fmt::format("{}.{}.{}", __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
#else
    return "unknown";
#endif
}

}  // namespace plssvm::stdpar::detail
