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

#include "plssvm/detail/assert.hpp"          // PLSSVM_ASSERT
#include "plssvm/exceptions/exceptions.hpp"  // plssvm::exception

#include "fmt/core.h"     // fmt::formatter
#include "fmt/ostream.h"  // fmt::ostream_formatter

#include <chrono>   // std::chrono::steady_clock::time_point
#include <cstddef>  // std::size_t
#include <iosfwd>   // std::ostream forward declaration
#include <string>   // std::string
#include <utility>  // std::move
#include <vector>   // std::vector

namespace plssvm::detail::tracking {

/**
 * @brief A class encapsulating all events.
 */
class events {
  public:
    /**
     * @brief A struct encapsulating a single event.
     */
    struct event {
        /// The time point this event occurred at.
        std::chrono::steady_clock::time_point time_point;
        /// The name of this event.
        std::string name;
    };

    /**
     * @brief Add a new event.
     * @param e the event
     */
    void add_event(event e) {
        this->time_points_.push_back(std::move(e.time_point));
        this->names_.push_back(fmt::format("\"{}\"", std::move(e.name)));

        PLSSVM_ASSERT(this->num_events() == this->time_points_.size(), "Error: number of event members mismatch!");
        PLSSVM_ASSERT(this->num_events() == this->names_.size(), "Error: number of event members mismatch!");
    }

    /**
     * @brief Add a new event.
     * @param[in] time_point the time point when the event occurred
     * @param[in] name the name of the event
     */
    void add_event(decltype(event::time_point) time_point, decltype(event::name) name) {
        this->time_points_.push_back(std::move(time_point));
        this->names_.push_back(fmt::format("\"{}\"", std::move(name)));

        PLSSVM_ASSERT(this->num_events() == this->time_points_.size(), "Error: number of event members mismatch!");
        PLSSVM_ASSERT(this->num_events() == this->names_.size(), "Error: number of event members mismatch!");
    }

    /**
     * @brief Return the event at index @p idx.
     * @param[in] idx the index of the event to retrieve
     * @return the event (`[[nodiscard]]`)
     */
    [[nodiscard]] event operator[](const std::size_t idx) const noexcept {
        return event{ time_points_[idx], names_[idx] };
    }

    /**
     * @brief Return the event at index @p idx.
     * @param[in] idx the index of the event to retrieve
     * @throws plssvm::exception if the requested index is out-of-bounce
     * @return the event (`[[nodiscard]]`)
     */
    [[nodiscard]] event at(const std::size_t idx) const {
        if (idx >= this->num_events()) {
            throw exception{ fmt::format("Index {} is out-of-bounce for the number of events {}!", idx, this->num_events()) };
        }
        return (*this)[idx];
    }

    /**
     * @brief Return the number of recorded events.
     * @return the number of events (`[[nodiscard]]`)
     */
    [[nodiscard]] std::size_t num_events() const noexcept { return time_points_.size(); }

    /**
     * @brief Check whether any event has been recorded yet.
     * @return `true` if no event has been recorded, `false` otherwise (`[[nodiscard]]`)
     */
    [[nodiscard]] bool empty() const noexcept { return time_points_.empty(); }

    /**
     * @brief Get all time points.
     * @return the time points (`[[nodiscard]]`)
     */
    [[nodiscard]] const auto &get_time_points() const noexcept { return time_points_; }

    /**
     * @brief Get all event names.
     * @return the event names (`[[nodiscard]]`)
     */
    [[nodiscard]] const auto &get_names() const noexcept { return names_; }

    /**
     * @brief Assemble the YAML string containing all events.
     * @param[in] start_time_point the reference time point the events occurred relative to
     * @return the YAML string (`[[nodiscard]]`)
     */
    [[nodiscard]] std::string generate_yaml_string(std::chrono::steady_clock::time_point start_time_point) const;

  private:
    /// All time points at which an event occurred.
    std::vector<decltype(event::time_point)> time_points_;
    /// The names of the respective events.
    std::vector<decltype(event::name)> names_;
};

/**
 * @brief Output the event @p e to the given output-stream @p out.
 * @param[in,out] out the output-stream to write the event to
 * @param[in] e the event
 * @return the output-stream
 */
std::ostream &operator<<(std::ostream &out, const events::event &e);

/**
 * @brief Output all events @p e to the given output-stream @p out.
 * @param[in,out] out the output-stream to write all events
 * @param[in] e all events
 * @return the output-stream
 */
std::ostream &operator<<(std::ostream &out, const events &e);

}  // namespace plssvm::detail::tracking

template <>
struct fmt::formatter<plssvm::detail::tracking::events::event> : fmt::ostream_formatter { };

template <>
struct fmt::formatter<plssvm::detail::tracking::events> : fmt::ostream_formatter { };

#endif  // PLSSVM_DETAIL_TRACKING_EVENT_HPP_
