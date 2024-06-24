/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/detail/tracking/events.hpp"

#include "plssvm/detail/tracking/utility.hpp"  // plssvm::detail::tracking::{durations_from_reference_time, time_points_to_epoch}

#include "fmt/chrono.h"  // format std::chrono types
#include "fmt/core.h"    // fmt::format
#include "fmt/format.h"  // fmt::join

#include <chrono>   // std::chrono::steady_clock::time_point
#include <cstddef>  // std::size_t
#include <ostream>  // std::ostream
#include <string>   // std::string
#include <vector>   // std::vector

namespace plssvm::detail::tracking {

std::string events::generate_yaml_string(const std::chrono::steady_clock::time_point start_time_point) const {
    if (this->empty()) {
        // no events -> return empty string
        return std::string{};
    } else {
        // assemble string
        return fmt::format("    time_points: [{}]\n"
                           "    names: [{}]",
                           fmt::join(durations_from_reference_time(time_points_, start_time_point), ", "),
                           fmt::join(names_, ", "));
    }
}

std::ostream &operator<<(std::ostream &out, const events::event &e) {
    return out << fmt::format("time_point: {}\n"
                              "name: {}",
                              e.time_point.time_since_epoch(),
                              e.name);
}

std::ostream &operator<<(std::ostream &out, const events &e) {
    return out << fmt::format("time_points: [{}]\n"
                              "names: [{}]",
                              fmt::join(time_points_to_epoch(e.get_time_points()), ", "),
                              fmt::join(e.get_names(), ", "));
}

}  // namespace plssvm::detail::tracking
