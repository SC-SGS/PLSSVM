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

#include <atomic>  // std::atomic
#include <chrono>  // std::chrono::{steady_clock::time_point, milliseconds}
#include <string>  // std::string
#include <thread>  // std::thread
#include <vector>  // std::vector

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

    [[nodiscard]] bool has_sampling_started() const noexcept;
    [[nodiscard]] bool is_sampling() const noexcept;
    [[nodiscard]] bool has_sampling_stopped() const noexcept;

    [[nodiscard]] std::vector<std::chrono::steady_clock::time_point> time_points() const noexcept { return time_points_; }

    [[nodiscard]] std::chrono::milliseconds sampling_interval() const noexcept { return sampling_interval_; }

    [[nodiscard]] virtual std::string generate_yaml_string(std::chrono::steady_clock::time_point start_time_point) const = 0;
    [[nodiscard]] virtual std::string device_identification() const = 0;

  protected:
    virtual void sampling_loop() = 0;

    void add_time_point(const std::chrono::steady_clock::time_point time_point) { time_points_.push_back(time_point); }

  private:
    std::atomic<bool> sampling_stopped_{ false };
    std::atomic<bool> sampling_running_{ false };
    std::atomic<bool> sampling_started_{ false };

    std::thread sampling_thread_{};

    std::vector<std::chrono::steady_clock::time_point> time_points_{};

    const std::chrono::milliseconds sampling_interval_{};
};

}  // namespace plssvm::detail::tracking

#endif  // PLSSVM_DETAIL_TRACKING_HARDWARE_SAMPLER_HPP_
