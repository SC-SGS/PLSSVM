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
#include "plssvm/detail/performance_tracker.hpp"            // plssvm::detail::tracking_entry, PLSSVM_DETAIL_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY
#include "plssvm/target_platforms.hpp"                      // plssvm::target_platform
#include "plssvm/verbosity_levels.hpp"                      // plssvm::verbosity_level

#include "fmt/core.h"  // fmt::format

namespace plssvm::stdpar {

void csvm::init(const target_platform target) {
    // check whether the requested target platform has been enabled
    if (target != target_platform::automatic && target != target_platform::cpu) {
        throw backend_exception{ fmt::format("Invalid target platform '{}' for the {} stdpar backend!", target, this->get_implementation_type()) };
    }
// the CPU target must be available
#if !defined(PLSSVM_HAS_CPU_TARGET)
    throw backend_exception{ "Requested target platform 'cpu' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!" };
#endif

    if (target == target_platform::automatic) {
        // GNU TBB only runs on the CPU
        target_ = target_platform::cpu;
    } else {
        target_ = target;
    }

    plssvm::detail::log(verbosity_level::full,
                        "\nUsing stdpar ({}; {}) as backend.\n\n",
                        plssvm::detail::tracking_entry{ "dependencies", "stdpar_implementation", this->get_implementation_type() },
                        plssvm::detail::tracking_entry{ "dependencies", "stdpar_version", detail::get_stdpar_version() });
    PLSSVM_DETAIL_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking_entry{ "backend", "backend", plssvm::backend_type::stdpar }));

    // print found stdpar devices
    plssvm::detail::log(verbosity_level::full,
                        "Found {} stdpar device(s) for the target platform {}:\n",
                        plssvm::detail::tracking_entry{ "backend", "num_devices", this->num_available_devices() },
                        plssvm::detail::tracking_entry{ "backend", "target_platform", target_ });

    plssvm::detail::log(verbosity_level::full | verbosity_level::timing,
                        "\n");
}

implementation_type csvm::get_implementation_type() const noexcept {
    return implementation_type::gnu_tbb;
}

}  // namespace plssvm::stdpar
