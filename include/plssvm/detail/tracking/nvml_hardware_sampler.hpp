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

#ifndef PLSSVM_DETAIL_TRACKING_NVML_HARDWARE_SAMPLER_HPP_
#define PLSSVM_DETAIL_TRACKING_NVML_HARDWARE_SAMPLER_HPP_

#include "plssvm/detail/tracking/hardware_sampler.hpp"  // plssvm::detail::tracking::hardware_sampler
#include "plssvm/detail/tracking/time_point.hpp"

#include "nvml.h"
#include <atomic>   // std::atomic
#include <cstdint>  // std::uint64_t, std::int64_t, std::uint32_t
#include <mutex>    // std::mutex

namespace plssvm::detail::tracking {

struct sample_type {
    time_point_type time;

    // clock related information in MHz
    unsigned int clock_graph = 0;
    unsigned int clock_sm = 0;
    unsigned int clock_mem = 0;
    unsigned long long clock_throttle_reason = 0;
    unsigned int clock_graph_max = 0;
    unsigned int clock_sm_max = 0;
    unsigned int clock_mem_max = 0;

    // temperature related information in °C
    unsigned int fan_speed = 0;
    unsigned int temperature_gpu = 0;
    unsigned int temperature_threshold_gpu_max = 0;
    unsigned int temperature_threshold_mem_max = 0;

    // memory related information in Byte // TODO: use memory struct -> circular dependencies? :/
    unsigned long long memory_free = 0;
    unsigned long long memory_used = 0;
    unsigned long long memory_total = 0;

    // power related information in mW
    int power_state = 0;
    unsigned int power_usage = 0;
    unsigned int power_limit = 0;
    unsigned int power_default_limit = 0;
    unsigned int power_total_energy_consumption = 0;

    // general information
    int performance_state = 0;
    unsigned int utilization_gpu = 0;
    unsigned int utilization_mem = 0;
};

class nvml_hardware_sampler : public hardware_sampler<sample_type> {
    using base_type = hardware_sampler<sample_type>;

  public:
    // TODO: handle device id?!?!?
    explicit nvml_hardware_sampler(int device_id, unsigned long long sampling_interval = 100);
    ~nvml_hardware_sampler() override;

  private:
    sample_type get_sample_measurement() final;
    std::uint64_t get_total_energy_consumption() final;

    nvmlDevice_t device_;

    inline static std::atomic<int> instances_{ 0 };
    inline static std::once_flag nvml_init_once_{};
    inline static std::once_flag nvml_shutdown_once_{};
};

}  // namespace plssvm::detail::tracking

#endif  // PLSSVM_DETAIL_TRACKING_NVML_HARDWARE_SAMPLER_HPP_
