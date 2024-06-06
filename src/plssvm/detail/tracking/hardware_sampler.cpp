/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/detail/tracking/hardware_sampler.hpp"

#include "fmt/chrono.h"
#include "fmt/format.h"

#include <cstddef>    // std::size_t
#include <cstdint>    // std::uint64_t
#include <exception>  // std::exception, std::terminate
#include <iostream>   // std::cerr, std::endl
#include <mutex>      // std::call_once

namespace plssvm::detail::tracking {

hardware_sampler::hardware_sampler(unsigned long long sampling_interval) :
    sampling_interval_{ sampling_interval } { }

hardware_sampler::~hardware_sampler() = default;

// TODO: use correct exception

void hardware_sampler::start_sampling() {
    // can't start an already running sampler
    if (sampling_started_) {
        throw std::runtime_error{ "Can't start an already running sampler!" };
    }

    // start sampling loop
    sampling_started_ = true;
    this->resume_sampling();
    start_time_ = std::chrono::steady_clock::now();
    sampling_thread_ = std::thread{ [this]() { this->sampling_loop(); } };
    sampling_thread_.detach();
}

void hardware_sampler::stop_sampling() {
    // can't stop an already stopped sampler
    if (sampling_stopped_) {
        throw std::runtime_error{ "Can't stop an already stopped sampler!" };
    }

    // stop sampling
    this->pause_sampling();
    sampling_stopped_ = true;  // -> notifies the sampling std::thread
    if (sampling_thread_.joinable()) {
        sampling_thread_.join();
    }
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
    events_.add_event(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_time_), std::move(name));
}

void hardware_sampler::sampling_loop() {
    // loop until stop_sampling() is called
    while (!sampling_stopped_) {
        this->add_sample();

        // wait for sampling_interval_ milliseconds to retrieve the next sample
        std::this_thread::sleep_for(std::chrono::milliseconds{ sampling_interval_ });
    }
}

std::string hardware_sampler::assemble_yaml_event_string() const {
    // TODO: check if currently running?

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
