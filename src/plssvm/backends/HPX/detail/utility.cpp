/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @author Alexander Strack
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */
#include <hpx/runtime_distributed.hpp>
#include <hpx/hpx_start.hpp>                                    // hpx::{start, stop, finalize}
#include <hpx/execution.hpp>                                    // hpx::post
#include "plssvm/backends/HPX/detail/utility.hpp"

#include "plssvm/detail/string_utility.hpp"  // plssvm::detail::as_lower_case
#include "plssvm/detail/utility.hpp"         // ::plssvm::detail::contains
#include "plssvm/target_platforms.hpp"       // plssvm::target_platforms

#include "fmt/format.h"  // fmt::format

#include <string>  // std::string

namespace plssvm::hpx::detail {

// TODO: implement function
std::string get_hpx_version() {
    return "unknown";
}

int get_num_threads() {
    // get the number of used HPX threads
    return static_cast<int>(::hpx::get_num_worker_threads());
}

void start_hpx_runtime() {
    // Initialize HPX runtime, but do not run hpx_main and do not pass commandline arguments
    // Set HPX commandline arguments with the HPX_COMMANDLINE_OPTIONS="" environment variable
    ::hpx::start(nullptr, 0, nullptr);
}

void stop_hpx_runtime() {
   // Finalize all existing HPX tasks
   ::hpx::post([]{::hpx::finalize();});
   // Stop HPX runtime
   ::hpx::stop();
}
}  // namespace plssvm::hpx::detail
