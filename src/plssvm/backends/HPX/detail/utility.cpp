/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @author Alexander Strack
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */
#include <hpx/runtime_distributed.hpp>
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

// TODO: implement function
int get_num_threads() {
    // get the number of used HPX threads
    return static_cast<int>(::hpx::get_num_worker_threads());
}
}  // namespace plssvm::hpx::detail
