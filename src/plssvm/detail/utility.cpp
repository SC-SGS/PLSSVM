/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/detail/utility.hpp"

#include "fmt/chrono.h"  // fmt::localtime
#include "fmt/core.h"    // fmt::format

#include <ctime>         // std::time
#include <string>        // std::string

namespace plssvm::detail {

std::string current_date_time() {
    return fmt::format("{:%Y-%m-%d %H:%M:%S}", fmt::localtime(std::time(nullptr)));
}

}  // namespace plssvm::detail