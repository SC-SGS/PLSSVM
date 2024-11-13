/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @author Alexander Strack
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */
#include <hpx/hpx_start.hpp>                 // hpx::{start, stop, finalize}
#include <hpx/execution.hpp>                 // hpx::post
#include <hpx/runtime_distributed.hpp>       // ::hpx::get_num_worker_threads
#include <hpx/version.hpp>                   // ::hpx::full_version_as_string
#include "plssvm/backends/HPX/detail/utility.hpp"

#include <string>  // std::string

namespace plssvm::hpx::detail {

std::string get_hpx_version() {
    return ::hpx::full_version_as_string();
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
