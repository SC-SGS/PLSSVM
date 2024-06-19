/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines an event type.
 */

#ifndef PLSSVM_DETAIL_TRACKING_EVENT_HPP_
#define PLSSVM_DETAIL_TRACKING_EVENT_HPP_
#pragma once

#include "plssvm/detail/assert.hpp"  // PLSSVM_ASSERT

#include "fmt/core.h"     // fmt::formatter
#include "fmt/ostream.h"  // fmt::ostream_formatter

#include <chrono>   // std::chrono::system_clock::time_point
#include <cstddef>  // std::size_t
#include <iosfwd>   // std::ostream forward declaration
#include <string>   // std::string
#include <utility>  // std::move
#include <vector>   // std::vector

namespace plssvm::detail::tracking {

class events {
  public:
    struct event {
        std::chrono::system_clock::time_point time_point;
        std::string name;
    };

    void add_event(event e) {
        this->time_points_.push_back(std::move(e.time_point));
        this->names_.push_back(fmt::format("\"{}\"", std::move(e.name)));

        PLSSVM_ASSERT(this->num_events() == this->time_points_.size(), "Error: number of event members mismatch!");
        PLSSVM_ASSERT(this->num_events() == this->names_.size(), "Error: number of event members mismatch!");
    }

    void add_event(decltype(event::time_point) time_point, decltype(event::name) name) {
        this->time_points_.push_back(std::move(time_point));
        this->names_.push_back(fmt::format("\"{}\"", std::move(name)));

        PLSSVM_ASSERT(this->num_events() == this->time_points_.size(), "Error: number of event members mismatch!");
        PLSSVM_ASSERT(this->num_events() == this->names_.size(), "Error: number of event members mismatch!");
    }

    event operator[](const std::size_t idx) const noexcept {
        return event{ time_points_[idx], names_[idx] };
    }

    [[nodiscard]] std::size_t num_events() const noexcept { return time_points_.size(); }

    [[nodiscard]] bool empty() const noexcept { return time_points_.empty(); }

    [[nodiscard]] const auto &get_time_points() const noexcept { return time_points_; }

    [[nodiscard]] const auto &get_names() const noexcept { return names_; }

    [[nodiscard]] std::string generate_yaml_string(std::chrono::system_clock::time_point start_time_point) const;

  private:
    std::vector<decltype(event::time_point)> time_points_;
    std::vector<decltype(event::name)> names_;
};

std::ostream &operator<<(std::ostream &out, const events::event &e);
std::ostream &operator<<(std::ostream &out, const events &e);

}  // namespace plssvm::detail::tracking

template <>
struct fmt::formatter<plssvm::detail::tracking::events::event> : fmt::ostream_formatter { };

template <>
struct fmt::formatter<plssvm::detail::tracking::events> : fmt::ostream_formatter { };

#endif  // PLSSVM_DETAIL_TRACKING_EVENT_HPP_
