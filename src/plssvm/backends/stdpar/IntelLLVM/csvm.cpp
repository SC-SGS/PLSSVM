/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/stdpar/csvm.hpp"

#include "plssvm/backend_types.hpp"                         // plssvm::backend_type
#include "plssvm/backends/stdpar/detail/utility.hpp"        // plssvm::stdpar::detail::{get_stdpar_version, default_device_equals_target}
#include "plssvm/backends/stdpar/exceptions.hpp"            // plssvm::stdpar::backend_exception
#include "plssvm/backends/stdpar/implementation_types.hpp"  // plssvm::stdpar::implementation_type
#include "plssvm/detail/logging.hpp"                        // plssvm::detail::log
#include "plssvm/detail/tracking/performance_tracker.hpp"   // plssvm::detail::tracking::tracking_entry, PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY
#include "plssvm/target_platforms.hpp"                      // plssvm::target_platform
#include "plssvm/verbosity_levels.hpp"                      // plssvm::verbosity_level

#include "sycl/sycl.hpp"  // sycl::device

#include "fmt/format.h"  // fmt::format

namespace plssvm::stdpar {

void csvm::init(const target_platform target) {
    // check whether the requested target platform has been enabled
    switch (target) {
        case target_platform::automatic:
            break;
        case target_platform::cpu:
#if !defined(PLSSVM_HAS_CPU_TARGET)
            throw backend_exception{ fmt::format("Requested target platform '{}' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!", target) };
#endif
            break;
        case target_platform::gpu_nvidia:
#if !defined(PLSSVM_HAS_NVIDIA_TARGET)
            throw backend_exception{ fmt::format("Requested target platform '{}' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!", target) };
#endif
            break;
        case target_platform::gpu_amd:
#if !defined(PLSSVM_HAS_AMD_TARGET)
            throw backend_exception{ fmt::format("Requested target platform '{}' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!", target) };
#endif
            break;
        case target_platform::gpu_intel:
#if !defined(PLSSVM_HAS_INTEL_TARGET)
            throw backend_exception{ fmt::format("Requested target platform '{}' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!", target) };
#endif
            break;
    }

    if (target == target_platform::automatic) {
        target_ = determine_default_target_platform();
    } else {
        target_ = target;
    }

    plssvm::detail::log(verbosity_level::full,
                        "\nUsing stdpar ({}; {}) as backend.\n\n",
                        plssvm::detail::tracking::tracking_entry{ "dependencies", "stdpar_implementation", this->get_implementation_type() },
                        plssvm::detail::tracking::tracking_entry{ "dependencies", "stdpar_version", detail::get_stdpar_version() });
    PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking::tracking_entry{ "backend", "backend", plssvm::backend_type::stdpar }));

    // AdaptiveCpp's stdpar per default uses the sycl default device
    const ::sycl::device default_device{};
    if (!detail::default_device_equals_target(default_device, target_)) {
        throw backend_exception{ fmt::format("The default device {} doesn't match the requested target platform {}! Please set the environment variable ONEAPI_DEVICE_SELECTOR or change the target platform.",
                                             default_device.get_info<::sycl::info::device::name>(),
                                             target_) };
    }

    // print found stdpar devices
    plssvm::detail::log(verbosity_level::full,
                        "Found {} stdpar device(s) for the target platform {}:\n",
                        plssvm::detail::tracking::tracking_entry{ "backend", "num_devices", this->num_available_devices() },
                        plssvm::detail::tracking::tracking_entry{ "backend", "target_platform", target_ });

    const std::string device_name = default_device.get_info<::sycl::info::device::name>();
    plssvm::detail::log(verbosity_level::full, "  [0, {}]\n", device_name);
    PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking::tracking_entry{ "backend", "device", device_name }));

    plssvm::detail::log(verbosity_level::full | verbosity_level::timing,
                        "\n");
}

implementation_type csvm::get_implementation_type() const noexcept {
    return implementation_type::intel_llvm;
}

}  // namespace plssvm::stdpar
