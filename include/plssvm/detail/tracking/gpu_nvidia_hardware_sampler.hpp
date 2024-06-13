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

#include "plssvm/detail/tracking/hardware_sampler.hpp"  // plssvm::detail::tracking::hardware_sampler
#include "plssvm/detail/tracking/nvml_samples.hpp"

#include <atomic>   // std::atomic
#include <chrono>   // std::chrono::milliseconds, std::chrono_literals namespace
#include <cstddef>  // std::size_t
#include <string>   // std::string
#include <vector>   // std::vector

namespace plssvm::detail::tracking {

using namespace std::chrono_literals;

class gpu_nvidia_hardware_sampler : public hardware_sampler {
  public:
    explicit gpu_nvidia_hardware_sampler(std::size_t device_id, std::chrono::milliseconds sampling_interval = 100ms);

    gpu_nvidia_hardware_sampler(const gpu_nvidia_hardware_sampler &) = delete;
    gpu_nvidia_hardware_sampler(gpu_nvidia_hardware_sampler &&) noexcept = delete;
    gpu_nvidia_hardware_sampler &operator=(const gpu_nvidia_hardware_sampler &) = delete;
    gpu_nvidia_hardware_sampler &operator=(gpu_nvidia_hardware_sampler &&) noexcept = delete;

    ~gpu_nvidia_hardware_sampler() override;

    [[nodiscard]] std::string device_identification() const noexcept override;

    [[nodiscard]] std::string assemble_yaml_sample_string() const override;

  private:
    void sampling_loop() final;

    std::size_t device_id_;

    std::vector<std::chrono::milliseconds> time_since_start_{};
    nvml_general_samples general_samples_;
    nvml_clock_samples clock_samples_;
    nvml_power_samples power_samples_;
    nvml_memory_samples memory_samples_;
    nvml_temperature_samples temperature_samples_;

    inline static std::atomic<int> instances_{ 0 };
    inline static std::atomic<bool> init_finished_{ false };
};

}  // namespace plssvm::detail::tracking

#endif  // PLSSVM_DETAIL_TRACKING_GPU_NVIDIA_HARDWARE_SAMPLER_HPP_