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

#include "fmt/core.h"  // fmt::format

#include <chrono>   // std::chrono::steady_clock::duration
#include <cstddef>  // std::size_t
#include <string>   // std::string
#include <utility>  // std::move
#include <vector>   // std::vector

namespace plssvm::detail::tracking {

class events {
  public:
    struct event {
        std::chrono::milliseconds time_since_start;
        std::string name;
        // TODO: std::ostream overload for proxy
    };

    void add_event(event e) {
        this->times_since_start_.push_back(std::move(e.time_since_start));
        this->names_.push_back(fmt::format("\"{}\"", std::move(e.name)));
        // TODO: invariant!
    }

    void add_event(decltype(event::time_since_start) time_since_start, decltype(event::name) name) {
        this->times_since_start_.push_back(std::move(time_since_start));
        this->names_.push_back(fmt::format("\"{}\"", std::move(name)));
        // TODO: invariant!
    }

    event operator[](const std::size_t idx) const noexcept {
        return event{ times_since_start_[idx], names_[idx] };
    }

    [[nodiscard]] std::size_t num_events() const noexcept { return times_since_start_.size(); }

    [[nodiscard]] bool empty() const noexcept { return times_since_start_.empty(); }

    [[nodiscard]] const auto &get_times() const noexcept { return times_since_start_; }

    [[nodiscard]] const auto &get_names() const noexcept { return names_; }

  private:
    std::vector<decltype(event::time_since_start)> times_since_start_;
    std::vector<decltype(event::name)> names_;
};

}  // namespace plssvm::detail::tracking

#endif  // PLSSVM_DETAIL_TRACKING_EVENT_HPP_
