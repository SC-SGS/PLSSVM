/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/detail/tracking/hardware_sampler.hpp"

#include "plssvm/exceptions/exceptions.hpp"  // plssvm::hardware_sampling_exception

#include "fmt/chrono.h"  // format std::chrono types
#include "fmt/core.h"    // fmt::format
#include "fmt/format.h"  // fmt::join

#include <chrono>   // std::chrono::{steady_clock, system_clock, duration_cast, milliseconds}
#include <string>   // std::string
#include <thread>   // std::thread
#include <utility>  // std::move

namespace plssvm::detail::tracking {

hardware_sampler::hardware_sampler(const std::chrono::milliseconds sampling_interval) :
    sampling_interval_{ sampling_interval } { }

hardware_sampler::~hardware_sampler() = default;

void hardware_sampler::start_sampling() {
    // can't start an already running sampler
    if (sampling_started_) {
        throw hardware_sampling_exception{ "Can start every hardware sampler only once!" };
    }

    // start sampling loop
    sampling_started_ = true;
    this->resume_sampling();
    steady_clock_start_time_ = std::chrono::steady_clock::now();
    system_clock_start_time_ = std::chrono::system_clock::now();
    sampling_thread_ = std::thread{ [this]() { this->sampling_loop(); } };
}

void hardware_sampler::stop_sampling() {
    // can't stop an already stopped sampler
    if (sampling_stopped_) {
        throw hardware_sampling_exception{ "Can stop every hardware sampler only once!" };
    }

    // stop sampling
    this->pause_sampling();
    sampling_stopped_ = true;  // -> notifies the sampling std::thread
    sampling_thread_.join();
}

void hardware_sampler::pause_sampling() {
    sampling_running_ = false;  // notifies the sampling std::thread
}

void hardware_sampler::resume_sampling() {
    sampling_running_ = true;  // notifies the sampling std::thread
}

bool hardware_sampler::is_sampling() const noexcept {
    return sampling_running_;
}

void hardware_sampler::add_event(std::string name) {
    events_.add_event(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - steady_clock_start_time_), std::move(name));
}

std::string hardware_sampler::assemble_yaml_event_string() const {
    // check whether it's safe to generate the YAML entry
    if (this->is_sampling()) {
        throw hardware_sampling_exception{ "Can't create the final YAML entry if the hardware sampler is still running!" };
    }

    // generate the YAML entry
    if (events_.empty()) {
        // no events -> return empty string
        return "";
    } else if (events_.num_events() == 1) {
        // only a single event has been provided -> no join necessary and do not use []
        return fmt::format("\n"
                           "    events:\n"
                           "      time_points: {}\n"
                           "      names: {}",
                           events_.get_times().front(),
                           events_.get_names().front());
    } else {
        // assemble string
        return fmt::format("\n"
                           "    events:\n"
                           "      time_points: [{}]\n"
                           "      names: [{}]",
                           fmt::join(events_.get_times(), ", "),
                           fmt::join(events_.get_names(), ", "));
    }
}

}  // namespace plssvm::detail::tracking
