/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines a hardware sampler for CPUs using the turbostat utility (requires root).
 */

#ifndef PLSSVM_DETAIL_TRACKING_CPU_HARDWARE_SAMPLER_HPP_
#define PLSSVM_DETAIL_TRACKING_CPU_HARDWARE_SAMPLER_HPP_
#pragma once

#include "plssvm/detail/tracking/cpu/cpu_samples.hpp"   // plssvm::detail::tracking::{cpu_general_samples, clock_samples, power_samples, memory_samples, temperature_samples, gfx_samples, idle_state_samples}
#include "plssvm/detail/tracking/hardware_sampler.hpp"  // plssvm::detail::tracking::hardware_sampler

#include <chrono>  // std::chrono::{system_clock::time_point, milliseconds}, std::chrono_literals namespace
#include <string>  // std::string
#include <vector>  // std::vector

namespace plssvm::detail::tracking {

using namespace std::chrono_literals;

class cpu_hardware_sampler : public hardware_sampler {
  public:
    explicit cpu_hardware_sampler(std::chrono::milliseconds sampling_interval = PLSSVM_HARDWARE_SAMPLING_INTERVAL);

    cpu_hardware_sampler(const cpu_hardware_sampler &) = delete;
    cpu_hardware_sampler(cpu_hardware_sampler &&) noexcept = delete;
    cpu_hardware_sampler &operator=(const cpu_hardware_sampler &) = delete;
    cpu_hardware_sampler &operator=(cpu_hardware_sampler &&) noexcept = delete;

    ~cpu_hardware_sampler() override;

    [[nodiscard]] const cpu_general_samples &general_samples() const noexcept { return general_samples_; }

    [[nodiscard]] const cpu_clock_samples &clock_samples() const noexcept { return clock_samples_; }

    [[nodiscard]] const cpu_power_samples &power_samples() const noexcept { return power_samples_; }

    [[nodiscard]] const cpu_memory_samples &memory_samples() const noexcept { return memory_samples_; }

    [[nodiscard]] const cpu_temperature_samples &temperature_samples() const noexcept { return temperature_samples_; }

    [[nodiscard]] const cpu_gfx_samples &gfx_samples() const noexcept { return gfx_samples_; }

    [[nodiscard]] const cpu_idle_states_samples &idle_state_samples() const noexcept { return idle_state_samples_; }

    [[nodiscard]] std::string device_identification() const override;

    [[nodiscard]] std::string generate_yaml_string(std::chrono::system_clock::time_point start_time_point) const override;

  private:
    void sampling_loop() final;

    cpu_general_samples general_samples_;
    cpu_clock_samples clock_samples_;
    cpu_power_samples power_samples_;
    cpu_memory_samples memory_samples_;
    cpu_temperature_samples temperature_samples_;
    cpu_gfx_samples gfx_samples_;
    cpu_idle_states_samples idle_state_samples_;
};

}  // namespace plssvm::detail::tracking

#endif  // PLSSVM_DETAIL_TRACKING_CPU_HARDWARE_SAMPLER_HPP_
