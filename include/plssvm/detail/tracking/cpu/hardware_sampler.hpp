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

    [[nodiscard]] std::string device_identification() const override;

    [[nodiscard]] std::string generate_yaml_string(std::chrono::system_clock::time_point start_time_point) const override;

  private:
    void sampling_loop() final;

    std::vector<std::string> lscpu_data_lines_{};
    std::vector<std::string> turbostat_data_lines_{};
};

}  // namespace plssvm::detail::tracking

#endif  // PLSSVM_DETAIL_TRACKING_CPU_HARDWARE_SAMPLER_HPP_