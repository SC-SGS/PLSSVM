/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines a hardware sampler for Intel GPUs using Intel's Level Zero.
 */

#ifndef PLSSVM_DETAIL_TRACKING_GPU_INTEL_HARDWARE_SAMPLER_HPP_
#define PLSSVM_DETAIL_TRACKING_GPU_INTEL_HARDWARE_SAMPLER_HPP_
#pragma once

#include "plssvm/detail/tracking/hardware_sampler.hpp"  // plssvm::detail::tracking::hardware_sampler

#include <atomic>   // std::atomic
#include <chrono>   // std::chrono::{steady_clock, milliseconds}, std::chrono_literals namespace
#include <cstddef>  // std::size_t
#include <string>   // std::string

namespace plssvm::detail::tracking {

using namespace std::chrono_literals;

class gpu_intel_hardware_sampler : public hardware_sampler {
  public:
    explicit gpu_intel_hardware_sampler(std::size_t device_id, std::chrono::milliseconds sampling_interval = PLSSVM_HARDWARE_SAMPLING_INTERVAL);

    gpu_intel_hardware_sampler(const gpu_intel_hardware_sampler &) = delete;
    gpu_intel_hardware_sampler(gpu_intel_hardware_sampler &&) noexcept = delete;
    gpu_intel_hardware_sampler &operator=(const gpu_intel_hardware_sampler &) = delete;
    gpu_intel_hardware_sampler &operator=(gpu_intel_hardware_sampler &&) noexcept = delete;

    ~gpu_intel_hardware_sampler() override;

    [[nodiscard]] std::string device_identification() const override;

    [[nodiscard]] std::string generate_yaml_string(std::chrono::steady_clock::time_point start_time_point) const override;

  private:
    void sampling_loop() final;

    // TODO: correct device handle
    std::size_t device_id_;

    inline static std::atomic<int> instances_{ 0 };
    inline static std::atomic<bool> init_finished_{ false };
};

}  // namespace plssvm::detail::tracking

#endif  // PLSSVM_DETAIL_TRACKING_GPU_INTEL_HARDWARE_SAMPLER_HPP_
