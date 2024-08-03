/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/detail/tracking/hardware_sampler.hpp"

#include "plssvm/exceptions/exceptions.hpp"  // plssvm::hardware_sampling_exception, plssvm::exception

#include <chrono>     // std::chrono::{steady_clock, duration_cast, milliseconds}
#include <exception>  // std::exception
#include <iostream>   // std::cerr, std::endl
#include <string>     // std::string
#include <thread>     // std::thread

namespace plssvm::detail::tracking {

hardware_sampler::hardware_sampler(const std::chrono::milliseconds sampling_interval) :
    sampling_interval_{ sampling_interval } { }

hardware_sampler::~hardware_sampler() = default;

void hardware_sampler::start_sampling() {
    // can't start an already running sampler
    if (this->has_sampling_started()) {
        throw hardware_sampling_exception{ "Can start every hardware sampler only once!" };
    }

    // start sampling loop
    sampling_started_ = true;
    this->resume_sampling();
    sampling_thread_ = std::thread{
        [this]() {
            try {
                this->sampling_loop();
            } catch (const plssvm::exception &e) {
                // print useful error message
                std::cerr << e.what_with_loc() << std::endl;
                throw;
            } catch (const std::exception &e) {
                // print useful error message
                std::cerr << e.what() << std::endl;
                throw;
            }
        }
    };
}

void hardware_sampler::stop_sampling() {
    // can't stop a hardware sampler that has never been started
    if (!this->has_sampling_started()) {
        throw hardware_sampling_exception{ "Can't stop a hardware sampler that has never been started!" };
    }
    // can't stop an already stopped sampler
    if (this->has_sampling_stopped()) {
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
    if (this->has_sampling_stopped()) {
        throw hardware_sampling_exception{ "Can't resume a hardware sampler that has already been stopped!" };
    }
    sampling_running_ = true;  // notifies the sampling std::thread
}

bool hardware_sampler::has_sampling_started() const noexcept {
    return sampling_started_;
}

bool hardware_sampler::is_sampling() const noexcept {
    return sampling_running_;
}

bool hardware_sampler::has_sampling_stopped() const noexcept {
    return sampling_stopped_;
}

}  // namespace plssvm::detail::tracking
