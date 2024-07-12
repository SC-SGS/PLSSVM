/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/detail/tracking/events.hpp"

#include "plssvm/detail/assert.hpp"            // PLSSVM_ASSERT
#include "plssvm/detail/tracking/utility.hpp"  // plssvm::detail::tracking::{durations_from_reference_time, time_points_to_epoch}
#include "plssvm/exceptions/exceptions.hpp"    // plssvm::exception

#include "fmt/chrono.h"  // format std::chrono types
#include "fmt/format.h"  // fmt::format
#include "fmt/ranges.h"  // fmt::join

#include <chrono>   // std::chrono::steady_clock::time_point
#include <cstddef>  // std::size_t
#include <ostream>  // std::ostream
#include <string>   // std::string
#include <utility>  // std::move
#include <vector>   // std::vector

namespace plssvm::detail::tracking {

void events::add_event(event e) {
    this->time_points_.push_back(std::move(e.time_point));
    this->names_.push_back(fmt::format("{}", std::move(e.name)));

    PLSSVM_ASSERT(this->num_events() == this->time_points_.size(), "Error: number of event members mismatch!");
    PLSSVM_ASSERT(this->num_events() == this->names_.size(), "Error: number of event members mismatch!");
}

void events::add_event(decltype(event::time_point) time_point, decltype(event::name) name) {
    this->time_points_.push_back(std::move(time_point));
    this->names_.push_back(fmt::format("{}", std::move(name)));

    PLSSVM_ASSERT(this->num_events() == this->time_points_.size(), "Error: number of event members mismatch!");
    PLSSVM_ASSERT(this->num_events() == this->names_.size(), "Error: number of event members mismatch!");
}

auto events::operator[](const std::size_t idx) const noexcept -> event {
    PLSSVM_ASSERT(idx < this->num_events(), "Index {} is out-of-bounce for the number of events {}!", idx, this->num_events());

    return event{ time_points_[idx], names_[idx] };
}

auto events::at(const std::size_t idx) const -> event {
    if (idx >= this->num_events()) {
        throw exception{ fmt::format("Index {} is out-of-bounce for the number of events {}!", idx, this->num_events()) };
    }
    return (*this)[idx];
}

std::string events::generate_yaml_string(const std::chrono::steady_clock::time_point start_time_point) const {
    if (this->empty()) {
        // no events -> return empty string
        return std::string{};
    } else {
        std::vector<std::string> quoted_names(this->num_events());
#pragma omp parallel for
        for (std::size_t i = 0; i < this->num_events(); ++i) {
            quoted_names[i] = fmt::format("\"{}\"", names_[i]);
        }
        // assemble string
        return fmt::format("    time_points: [{}]\n"
                           "    names: [{}]",
                           fmt::join(durations_from_reference_time(time_points_, start_time_point), ", "),
                           fmt::join(quoted_names, ", "));
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
