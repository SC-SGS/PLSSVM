/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines a hardware sampler for NVIDIA GPUs using NVIDIA's Management Library (NVML).
 */

#ifndef PLSSVM_DETAIL_TRACKING_GPU_NVIDIA_HARDWARE_SAMPLER_HPP_
#define PLSSVM_DETAIL_TRACKING_GPU_NVIDIA_HARDWARE_SAMPLER_HPP_
#pragma once

#include "plssvm/detail/tracking/gpu_nvidia/nvml_device_handle.hpp"  // plssvm::detail::tracking::nvml_device_handle
#include "plssvm/detail/tracking/gpu_nvidia/nvml_samples.hpp"        // plssvm::detail::tracking::{nvml_general_samples, nvml_clock_samples, nvml_power_samples, nvml_memory_samples, nvml_temperature_samples}
#include "plssvm/detail/tracking/hardware_sampler.hpp"               // plssvm::detail::tracking::hardware_sampler

#include <atomic>   // std::atomic
#include <chrono>   // std::chrono::milliseconds, std::chrono_literals namespace
#include <cstddef>  // std::size_t
#include <string>   // std::string

namespace plssvm::detail::tracking {

using namespace std::chrono_literals;

class gpu_nvidia_hardware_sampler : public hardware_sampler {
  public:
    explicit gpu_nvidia_hardware_sampler(std::size_t device_id, std::chrono::milliseconds sampling_interval = PLSSVM_HARDWARE_SAMPLING_INTERVAL);

    gpu_nvidia_hardware_sampler(const gpu_nvidia_hardware_sampler &) = delete;
    gpu_nvidia_hardware_sampler(gpu_nvidia_hardware_sampler &&) noexcept = delete;
    gpu_nvidia_hardware_sampler &operator=(const gpu_nvidia_hardware_sampler &) = delete;
    gpu_nvidia_hardware_sampler &operator=(gpu_nvidia_hardware_sampler &&) noexcept = delete;

    ~gpu_nvidia_hardware_sampler() override;

    [[nodiscard]] const nvml_general_samples &general_samples() const noexcept { return general_samples_; }

    [[nodiscard]] const nvml_clock_samples &clock_samples() const noexcept { return clock_samples_; }

    [[nodiscard]] const nvml_power_samples &power_samples() const noexcept { return power_samples_; }

    [[nodiscard]] const nvml_memory_samples &memory_samples() const noexcept { return memory_samples_; }

    [[nodiscard]] const nvml_temperature_samples &temperature_samples() const noexcept { return temperature_samples_; }

    [[nodiscard]] std::string device_identification() const override;

    [[nodiscard]] std::string generate_yaml_string(std::chrono::system_clock::time_point start_time_point) const override;

  private:
    void sampling_loop() final;

    nvml_device_handle device_;

    nvml_general_samples general_samples_{};
    nvml_clock_samples clock_samples_{};
    nvml_power_samples power_samples_{};
    nvml_memory_samples memory_samples_{};
    nvml_temperature_samples temperature_samples_{};

    inline static std::atomic<int> instances_{ 0 };
    inline static std::atomic<bool> init_finished_{ false };
};

}  // namespace plssvm::detail::tracking

#endif  // PLSSVM_DETAIL_TRACKING_GPU_NVIDIA_HARDWARE_SAMPLER_HPP_
