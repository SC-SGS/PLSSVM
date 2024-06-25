/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/detail/tracking/gpu_intel/hardware_sampler.hpp"

#include "plssvm/detail/tracking/gpu_intel/level_zero_device_handle_impl.hpp"  // plssvm::detail::tracking::level_zero_device_handle_impl
#include "plssvm/detail/tracking/gpu_intel/utility.hpp"                        // PLSSVM_LEVEL_ZERO_ERROR_CHECK
#include "plssvm/detail/tracking/hardware_sampler.hpp"                         // plssvm::detail::tracking::hardware_sampler
#include "plssvm/detail/tracking/utility.hpp"                                  // plssvm::detail::tracking::durations_from_reference_time
#include "plssvm/exceptions/exceptions.hpp"                                    // plssvm::exception, plssvm::hardware_sampling_exception

#include "fmt/chrono.h"          // format std::chrono types
#include "fmt/core.h"            // fmt::format
#include "fmt/format.h"          // fmt::join
#include "level_zero/ze_api.h"   // Level Zero runtime functions
#include "level_zero/zes_api.h"  // Level Zero runtime functions

#include <algorithm>    // std::min_element
#include <chrono>       // std::chrono::{steady_clock, duration_cast, milliseconds}
#include <cstddef>      // std::size_t
#include <exception>    // std::exception, std::terminate
#include <iostream>     // std::cerr, std::endl
#include <string>       // std::string
#include <string_view>  // std:string_view
#include <thread>       // std::this_thread
#include <vector>       // std::vector

namespace plssvm::detail::tracking {

gpu_intel_hardware_sampler::gpu_intel_hardware_sampler(const std::size_t device_id, const std::chrono::milliseconds sampling_interval) :
    hardware_sampler{ sampling_interval } {
    // make sure that zeInit is only called once for all instances
    if (instances_++ == 0) {
        PLSSVM_LEVEL_ZERO_ERROR_CHECK(zeInit(ZE_INIT_FLAG_GPU_ONLY));
        // notify that initialization has been finished
        init_finished_ = true;
    } else {
        // wait until init has been finished!
        while (!init_finished_) { }
    }

    // initialize samples -> can't be done beforehand since the device handle can only be initialized after a call to nvmlInit
    device_ = level_zero_device_handle{ device_id };
}

gpu_intel_hardware_sampler::~gpu_intel_hardware_sampler() {
    try {
        // if this hardware sampler is still sampling, stop it
        if (this->is_sampling()) {
            this->stop_sampling();
        }
        // the level zero runtime has no dedicated shut down or cleanup function
    } catch (const plssvm::exception &e) {
        std::cerr << e.what_with_loc() << std::endl;
        std::terminate();
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        std::terminate();
    }
}

std::string gpu_intel_hardware_sampler::device_identification() const {
    return fmt::format("gpu_intel_device_");  // TODO:
}

std::string gpu_intel_hardware_sampler::generate_yaml_string(const std::chrono::steady_clock::time_point start_time_point) const {
    // check whether it's safe to generate the YAML entry
    if (this->is_sampling()) {
        throw hardware_sampling_exception{ "Can't create the final YAML entry if the hardware sampler is still running!" };
    }

    return fmt::format("\n"
                       "    sampling_interval: {}\n"
                       "    time_points: [{}]\n"
                       "{}\n"
                       "{}\n"
                       "{}\n"
                       "{}\n"
                       "{}",
                       this->sampling_interval(),
                       fmt::join(durations_from_reference_time(this->time_points(), start_time_point), ", "),
                       general_samples_.generate_yaml_string(),
                       clock_samples_.generate_yaml_string(),
                       power_samples_.generate_yaml_string(),
                       memory_samples_.generate_yaml_string(),
                       temperature_samples_.generate_yaml_string());
}

void gpu_intel_hardware_sampler::sampling_loop() {
    // get the level zero handle from the device
    ze_device_handle_t device = device_.get_impl().device;

    //
    // add samples where we only have to retrieve the value once
    //

    // TODO: fixed samples

    //
    // loop until stop_sampling() is called
    //

    while (!this->has_sampling_stopped()) {
        // only sample values if the sampler currently isn't paused
        if (this->is_sampling()) {
            // add current time point
            this->add_time_point(std::chrono::steady_clock::now());

            // TODO: sampled samples
        }

        // wait for the sampling interval to pass to retrieve the next sample
        std::this_thread::sleep_for(this->sampling_interval());
    }
}

std::ostream &operator<<(std::ostream &out, const gpu_intel_hardware_sampler &sampler) {
    return out << fmt::format("sampling interval: {}\n"
                              "time points: [{}]\n\n"
                              "general samples:\n{}\n\n"
                              "clock samples:\n{}\n\n"
                              "power samples:\n{}\n\n"
                              "memory samples:\n{}\n\n"
                              "temperature samples:\n{}",
                              sampler.sampling_interval(),
                              fmt::join(time_points_to_epoch(sampler.time_points()), ", "),
                              sampler.general_samples(),
                              sampler.clock_samples(),
                              sampler.power_samples(),
                              sampler.memory_samples(),
                              sampler.temperature_samples());
}

}  // namespace plssvm::detail::tracking
