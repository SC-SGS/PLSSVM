/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/stdpar/csvm.hpp"

#include "plssvm/backend_types.hpp"                         // plssvm::backend_type
#include "plssvm/backends/stdpar/detail/utility.hpp"        // plssvm::stdpar::detail::get_stdpar_version
#include "plssvm/backends/stdpar/exceptions.hpp"            // plssvm::stdpar::backend_exception
#include "plssvm/backends/stdpar/implementation_types.hpp"  // plssvm::stdpar::implementation_type
#include "plssvm/detail/logging/log.hpp"                    // plssvm::detail::log
#include "plssvm/detail/logging/log_untracked.hpp"          // plssvm::detail::log_untracked
#include "plssvm/detail/string_utility.hpp"                 // plssvm::detail::trim
#include "plssvm/detail/tracking/performance_tracker.hpp"   // plssvm::detail::tracking::tracking_entry, PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY
#include "plssvm/target_platforms.hpp"                      // plssvm::target_platform
#include "plssvm/verbosity_levels.hpp"                      // plssvm::verbosity_level

#include "fmt/format.h"  // fmt::format

#include <vector>  // std::vector

namespace plssvm::stdpar {

csvm::csvm(const target_platform target) {
    // check whether the requested target platform has been enabled
    if (target != target_platform::automatic && target != target_platform::cpu && target != target_platform::gpu_nvidia) {
        throw backend_exception{ fmt::format("Invalid target platform '{}' for the {} stdpar backend!", target, this->get_implementation_type()) };
    }

    switch (target) {
        case target_platform::automatic:
#if !defined(PLSSVM_HAS_CPU_TARGET) && !defined(PLSSVM_HAS_NVIDIA_TARGET)
            throw backend_exception{ fmt::format("Requested target platform '{}' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!", determine_default_target_platform()) };
#endif
            break;
        case target_platform::cpu:
#if !defined(PLSSVM_HAS_CPU_TARGET)
            throw backend_exception{ fmt::format("Requested target platform '{}' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!", target) };
#else
            break;
#endif
        case target_platform::gpu_nvidia:
#if !defined(PLSSVM_HAS_NVIDIA_TARGET)
            throw backend_exception{ fmt::format("Requested target platform '{}' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!", target) };
#else
            break;
#endif
        default:
            // nothing to do
            break;
    }

    // update the target platform
    if (target == target_platform::automatic) {
        target_ = determine_default_target_platform();
    } else {
        target_ = target;
    }

    std::vector<std::string> device_names{};
#if defined(PLSSVM_STDPAR_BACKEND_NVHPC_GPU)
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    device_names.emplace_back(::plssvm::detail::trim(prop.name));
#endif

    if (comm_.size() > 1) {
        mpi::detail::gather_and_print_csvm_information(comm_, plssvm::backend_type::stdpar, target_, device_names, fmt::format("{}", this->get_implementation_type()));
    } else {
        // use more detailed single rank command line output
        plssvm::detail::log_untracked(verbosity_level::full,
                                      comm_,
                                      "\nUsing stdpar ({}; {}) as backend.\n"
#if defined(PLSSVM_STDPAR_BACKEND_NVHPC_GPU)
                                      "Found {} stdpar device(s) for the target platform {}:\n",
#else
                                      "Found {} stdpar device(s) for the target platform {}.\n",
#endif
                                      this->get_implementation_type(),
                                      detail::get_stdpar_version(),
                                      this->num_available_devices(),
                                      target_);
#if defined(PLSSVM_STDPAR_BACKEND_NVHPC_GPU)
        plssvm::detail::log_untracked(verbosity_level::full,
                                      comm_,
                                      "  [0, {}, {}.{}]\n",
                                      device_names.back(),
                                      prop.major,
                                      prop.minor);
#endif
    }

    plssvm::detail::log_untracked(verbosity_level::full | verbosity_level::timing,
                                  comm_,
                                  "\n");

    PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking::tracking_entry{ "dependencies", "stdpar_version", detail::get_stdpar_version() }));
    PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking::tracking_entry{ "backend", "stdpar_implementation", this->get_implementation_type() }));
    PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking::tracking_entry{ "backend", "backend", plssvm::backend_type::stdpar }));
    PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking::tracking_entry{ "backend", "target_platform", target_ }));
    PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking::tracking_entry{ "backend", "num_devices", this->num_available_devices() }));
    if (!device_names.empty()) {
        // NVHPC does not provide us with a device name if compiled for CPUs
        PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking::tracking_entry{ "backend", "device", device_names }));
    }
}

implementation_type csvm::get_implementation_type() const noexcept {
    return implementation_type::nvhpc;
}

}  // namespace plssvm::stdpar
