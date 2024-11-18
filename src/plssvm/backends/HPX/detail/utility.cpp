/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @author Alexander Strack
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/HPX/detail/utility.hpp"

#include <hpx/runtime_distributed.hpp>  // ::hpx::get_num_worker_threads
#include <hpx/version.hpp>              // ::hpx::full_version_as_string
#include <string>                       // std::string

namespace plssvm::hpx::detail {

std::string get_hpx_version() {
    return ::hpx::full_version_as_string();
}

int get_num_threads() {
    // get the number of used HPX threads
    return static_cast<int>(::hpx::get_num_worker_threads());
}
}  // namespace plssvm::hpx::detail
