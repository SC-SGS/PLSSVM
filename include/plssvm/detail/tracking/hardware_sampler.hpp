/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines the base class for all hardware samplers.
 */

#ifndef PLSSVM_DETAIL_TRACKING_HARDWARE_SAMPLER_HPP_
#define PLSSVM_DETAIL_TRACKING_HARDWARE_SAMPLER_HPP_
#pragma once

#include "plssvm/detail/tracking/events.hpp"  // plssvm::detail::tracking::events

#include <atomic>  // std::atomic
#include <chrono>  // std::chrono::{steady_clock::time_point, milliseconds}
#include <string>  // std::string
#include <thread>  // std::thread

namespace plssvm::detail::tracking {

class hardware_sampler {
  public:
    explicit hardware_sampler(std::chrono::milliseconds sampling_interval);

    hardware_sampler(const hardware_sampler &) = delete;
    hardware_sampler(hardware_sampler &&) noexcept = delete;
    hardware_sampler &operator=(const hardware_sampler &) = delete;
    hardware_sampler &operator=(hardware_sampler &&) noexcept = delete;

    virtual ~hardware_sampler() = 0;

    void start_sampling();
    void stop_sampling();
    void pause_sampling();
    void resume_sampling();

    [[nodiscard]] bool is_sampling() const noexcept;

    [[nodiscard]] std::chrono::milliseconds sampling_interval() const noexcept { return sampling_interval_; }

    void add_event(std::string name);

    [[nodiscard]] const events &get_events() const noexcept { return events_; }

    [[nodiscard]] std::string assemble_yaml_event_string() const;

    [[nodiscard]] virtual std::string assemble_yaml_sample_string() const = 0;

    [[nodiscard]] virtual std::string device_identification() const noexcept = 0;

  protected:
    virtual void sampling_loop() = 0;

    [[nodiscard]] std::chrono::steady_clock::time_point sampling_start_time() const noexcept { return start_time_; }

    std::atomic<bool> sampling_started_{ false };
    std::atomic<bool> sampling_stopped_{ false };
    std::atomic<bool> sampling_running_{ false };

  private:
    std::thread sampling_thread_{};

    const std::chrono::milliseconds sampling_interval_;
    std::chrono::steady_clock::time_point start_time_;
    events events_{};
};

}  // namespace plssvm::detail::tracking

#endif  // PLSSVM_DETAIL_TRACKING_HARDWARE_SAMPLER_HPP_
