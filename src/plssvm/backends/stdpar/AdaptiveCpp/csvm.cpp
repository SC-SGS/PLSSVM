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
#include "plssvm/detail/logging/mpi_log_untracked.hpp"      // plssvm::detail::log_untracked
#include "plssvm/detail/string_utility.hpp"                 // plssvm::detail::trim
#include "plssvm/detail/tracking/performance_tracker.hpp"   // plssvm::detail::tracking::tracking_entry, PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY
#include "plssvm/mpi/communicator.hpp"                      // plssvm::mpi::communicator
#include "plssvm/mpi/detail/information.hpp"                // plssvm::mpi::detail::gather_and_print_csvm_information
#include "plssvm/target_platforms.hpp"                      // plssvm::target_platform
#include "plssvm/verbosity_levels.hpp"                      // plssvm::verbosity_level

#include "sycl/sycl.hpp"  // sycl::device

#include "fmt/format.h"  // fmt::format

#include <string>  // std::string
#include <vector>  // std::vector

namespace plssvm::stdpar {

csvm::csvm(const target_platform target) {
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

    // update the target platform
    if (target == target_platform::automatic) {
        target_ = determine_default_target_platform();
    } else {
        target_ = target;
    }

    // AdaptiveCpp's stdpar per default uses the sycl default device
    const ::sycl::device default_device{};
    const std::vector<std::string> device_names{ std::string{ ::plssvm::detail::trim(default_device.get_info<::sycl::info::device::name>()) } };

    // check that the default device supports the requested target platform
    if (!detail::default_device_equals_target(default_device, target_)) {
        throw backend_exception{ fmt::format("The default device {} doesn't match the requested target platform {}! Please set the environment variable ACPP_VISIBILITY_MASK or change the target platform.",
                                             device_names.front(),
                                             target_) };
    }

    if (comm_.size() > 1) {
        mpi::detail::gather_and_print_csvm_information(comm_, plssvm::backend_type::stdpar, target_, device_names, fmt::format("{}", this->get_implementation_type()));
    } else {
        // use more detailed single rank command line output
        plssvm::detail::log_untracked(verbosity_level::full,
                                      comm_,
                                      "\nUsing stdpar ({}; {}; {}) as backend.\n"
                                      "Found {} stdpar device(s) for the target platform {}:\n"
                                      "  [0, {}]\n",
                                      this->get_implementation_type(),
                                      detail::get_stdpar_version(),
                                      PLSSVM_ACPP_TARGETS,
                                      this->num_available_devices(),
                                      target_,
                                      device_names.front());
    }

    plssvm::detail::log_untracked(verbosity_level::full | verbosity_level::timing,
                                  comm_,
                                  "\n");

    PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking::tracking_entry{ "dependencies", "stdpar_version", detail::get_stdpar_version() }));
    PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking::tracking_entry{ "backend", "stdpar_implementation", this->get_implementation_type() }));
    PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking::tracking_entry{ "backend", "backend", plssvm::backend_type::stdpar }));
    PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking::tracking_entry{ "backend", "target_platform", target_ }));
    PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking::tracking_entry{ "backend", "num_devices", this->num_available_devices() }));
    PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking::tracking_entry{ "backend", "device", device_names }));
    PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking::tracking_entry{ "backend", "acpp_targets", PLSSVM_ACPP_TARGETS }));
}

implementation_type csvm::get_implementation_type() const noexcept {
    return implementation_type::adaptivecpp;
}

}  // namespace plssvm::stdpar
